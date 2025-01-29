from dotenv import load_dotenv
import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename
from langchain.schema import Document


load_dotenv()

app = Flask(__name__)

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base"
PERSIST_DIR = "chroma_db"
MODEL_NAME = "mixtral-8x7b-32768"
LOG_DB = "interaction_logs"
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
os.environ['GROQ_API_KEY'] = groq_api_key
# Initialize services
os.environ['MONGO_URI'] = os.getenv("MONGO_URI")
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[LOG_DB]

# LangChain components
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(temperature=0.3, groq_api_key=groq_api_key, model_name=MODEL_NAME)

# Add support for JSON and Markdown files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'json', 'md'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class KnowledgeManager:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = embeddings
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize empty ChromaDB vector store with custom prompt"""
        
        self.vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )
        
        # Custom prompt template
        custom_prompt_template = """You are an AI assistant. Follow these steps:
        1. First, answer the question using your own knowledge.
        2. If unsure or needing verification, check the provided context.
        3. If still uncertain after checking context, say "I don't know."

        Context: {context}

        Question: {question}
        Helpful Answer:"""

        qa_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )

        # Create retrieval chain with custom prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
    
    def add_document(self, file_path: str):
        """Process and add a single document to the knowledge base"""
        from langchain_community.document_loaders import UnstructuredFileLoader, JSONLoader, TextLoader
        from langchain.schema import Document  
        
        try:
            file_extension = file_path.rsplit('.', 1)[-1].lower()

            if file_extension == "json":
            
                jq_schema = '.' 
                loader = JSONLoader(file_path, jq_schema=jq_schema, text_content=False)
                documents = loader.load()
                
           
                documents = [
                    Document(page_content=str(doc) if isinstance(doc, dict) else doc.page_content)
                    for doc in documents
                ]

            elif file_extension in ["md", "txt"]:
                loader = TextLoader(file_path)
                documents = loader.load()
            else:
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            if splits:
                self.vector_store.add_documents(splits)
                return len(splits)
            return 0
        except Exception as e:
            logging.error(f"Document processing error: {e}")
            raise

    def query(self, question: str) -> dict:
        """Process user query through LangChain pipeline"""
        return self.qa_chain.invoke({"query": question})

knowledge_manager = KnowledgeManager()

#
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Empty question"}), 400
    
    try:
        result = knowledge_manager.query(question)
        response = result.get("result", "No response found")  
        sources = [doc.metadata.get("source", "Unknown source") for doc in result.get("source_documents", [])]  
        
        log_interaction(question, response, sources)
        return jsonify({
            "question": question,
            "response": response,
            "sources": list(set(sources))
        })
    except Exception as e:
        logging.error(f"Query error: {e}")
        return jsonify({"error": "Processing failed"}), 500

def log_interaction(query, response, sources):
    """Log interaction with context metadata"""
    db.logs.insert_one({
        "timestamp": datetime.now(),
        "query": query,
        "response": response,
        "sources": sources,
        "context_count": len(sources)
    })

@app.route('/admin/upload', methods=['POST'])
def upload_document():
    """Handle knowledge base updates with various file formats"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process and add document
            added_chunks = knowledge_manager.add_document(file_path)
            
            # Clean up temp file
            os.remove(file_path)
            
            return jsonify({
                "status": "success",
                "message": "Document processed successfully",
                "file": filename,
                "chunks_added": added_chunks
            })
            
        except Exception as e:
            # Clean up file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/logs', methods=['GET'])
def get_logs():
    logs = list(db.logs.find().sort("timestamp", -1).limit(100))
    for log in logs:
        log['_id'] = str(log['_id'])  
    return jsonify(logs)

@app.route('/admin')
def admin():
    """Render the admin page for document upload"""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    return render_template('upload.html')

if __name__ == '__main__':
    # Create necessary directories
    for directory in [KNOWLEDGE_BASE_DIR, app.config['UPLOAD_FOLDER']]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    app.run(debug=True)