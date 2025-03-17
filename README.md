# PDF RAG Application

A Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents using local LLMs.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is running with your preferred model
   ```bash
   # Pull your preferred model
   ollama pull llama3  # or any other model
   
   # Run the model
   ollama run llama3
   ```
4. Configure your environment variables by creating a `.env` file (you can copy from `.env.example`):
   ```
   # LLM model for Ollama
   LLM_MODEL=llama3:8b  # or any other model installed in Ollama
   
   # Other configuration options...
   WEB_PORT=5001
   ```

## Usage

### Command Line Interface

1. Place your PDF documents in the `data/pdfs` directory
2. Ingest documents into the vector store:
   ```bash
   python ingest.py
   ```
3. Query the documents:
   ```bash
   python query.py "Your question about the documents?"
   ```
4. Or run the interactive application:
   ```bash
   python app.py
   ```

### Web Interface

The application now includes a web interface for uploading PDFs and analyzing them with predefined questions.

1. Start the web server:
   ```bash
   python web_app.py
   ```
2. Open your browser and navigate to `http://localhost:5001` (or the port specified in your `.env` file)
3. Upload a PDF file
4. Customize the questions (or use the default ones)
5. Click "Analyze Document" to process the PDF
6. View the answers and download them as a markdown file

## Configuration Options

You can customize the application behavior by modifying the following variables in your `.env` file:

- `LLM_MODEL`: The Ollama model to use (e.g., `llama3:8b`, `mistral:7b`, `qwq:32b`)
- `EMBEDDING_MODEL`: The embedding model for vector search
- `PDF_DIR`: Directory for storing PDF files for CLI operation
- `UPLOAD_DIR`: Directory for uploaded PDFs via web interface
- `VECTOR_STORE_DIR`: Directory for the Chroma vector store
- `WEB_HOST`: Host to bind the web server
- `WEB_PORT`: Port for the web server
- `WEB_DEBUG`: Enable/disable debug mode for Flask

## Project Structure

- `app.py`: Main application for interactive querying
- `ingest.py`: Script to ingest PDF documents into the vector store
- `query.py`: Script to query the vector store and get responses
- `web_app.py`: Web application for uploading and analyzing PDFs
- `rag/`: Module containing the RAG pipeline components
  - `document_loader.py`: PDF loading and processing
  - `vector_store.py`: Chroma vector store setup
  - `llm.py`: Ollama LLM integration
  - `pipeline.py`: LangGraph pipeline definition
- `web/`: Web application files
  - `templates/`: HTML templates
  - `static/`: CSS and other static files 