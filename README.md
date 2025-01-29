# Knowledge Assitant Bot

## Description
This project is a Flask application that utilizes LangChain for document processing and querying. It supports various file formats and logs interactions with a MongoDB database. The application allows users to upload documents, which are then processed and stored in a vector database for efficient querying.

## Solution Overview
The application is designed to provide an AI-powered assistant that can answer questions based on the content of uploaded documents. It leverages the LangChain framework to handle document loading, text splitting, and querying. The key components of the solution include:

- **Document Upload**: Users can upload various file formats (e.g., PDF, DOCX, JSON, Markdown) to the application.
- **Knowledge Management**: The application processes the uploaded documents and stores them in a vector database (Chroma) for efficient retrieval.
- **Querying**: Users can ask questions, and the application retrieves relevant information from the knowledge base to provide answers.
- **Logging**: All interactions are logged in a MongoDB database for tracking and analysis.

  

https://github.com/user-attachments/assets/164f7b26-7eba-4d01-81c3-48aa10fbf0cd


## Requirements
- Python 3.x
- MongoDB

## Setup Instructions

### 1. Create a Virtual Environment
To create a virtual environment, run the following command in your terminal:
```bash
python -m venv venv
```
### 2. Activate the Virtual Environment
- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Set Up the `.env` File
Create a `.env` file in the root directory of the project and add the following environment variables:

```plaintext
GROQ_API_KEY=your_groq_api_key_here
MONGO_URI=your_mongo_uri_here
HF_TOKEN=your_huggingface_token_here
```

- **GROQ_API_KEY**: Your API key for the Groq service.
- **MONGO_URI**: The connection string for your MongoDB database.
- **HF_TOKEN**: Your Hugging Face token for accessing models.

### 5. Run the Application
Finally, run the application with the following command:

```bash
python app.py
```

The application will start, and you can access it in your web browser at `http://127.0.0.1:5000`.


