#!/usr/bin/env python
"""
Web application for RAG queries with PDF upload.
"""
import os
import uuid
import json
import time
import re
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, Response, stream_with_context
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from rag.pipeline import RAGPipeline
from rag.document_loader import PDFProcessor

# Load environment variables
load_dotenv()

# Get configuration from environment variables
PDF_DIR = os.getenv('PDF_DIR', 'data/pdfs')
UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'data/uploads')
VECTOR_STORE_DIR = os.getenv('VECTOR_STORE_DIR', 'data/chroma')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'jina/jina-embeddings-v2-base-de')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwq:32b')
WEB_HOST = os.getenv('WEB_HOST', '0.0.0.0')
WEB_PORT = int(os.getenv('WEB_PORT', '5001'))
WEB_DEBUG = os.getenv('WEB_DEBUG', 'true').lower() == 'true'

# Create Flask app
app = Flask(__name__, 
            template_folder='web/templates', 
            static_folder='web/static')

# Configure upload folder
UPLOAD_FOLDER = UPLOAD_DIR
ALLOWED_EXTENSIONS = {'pdf'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure questions
DEFAULT_QUESTIONS = [
    "Bitte schreibe eine zusammenfassung der Akte in ca. 250 Wörtern",
    "Bitte erstelle ein formatiertes Inhaltsverzeichnis mit zusammenfassenden Titeln und Seitenangaben",
    "Bitte gib mir, formatiert, eine Ausgabe über alle Beteiligte mit Kontaktangaben, außer der Kanzlei Riedl",
    "Waren bei dem Fall Drogen, Alkohol, Medikamente oder Fahrerflucht im Spiel ?",
    "Bitte beantworte kurz und knapp wer der Schuldige in dem Fall war und wie hoch der Schaden ist"
]

# Initialize the pipeline
pipeline = RAGPipeline(
    pdf_directory=PDF_DIR,
    vector_store_dir=VECTOR_STORE_DIR,
    embedding_model=EMBEDDING_MODEL,
    llm_model=LLM_MODEL
)

# Helper functions
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_answers(pdf_path, questions=None):
    """Generate answers for the given PDF and questions."""
    if questions is None:
        questions = DEFAULT_QUESTIONS
    
    # Process the PDF
    pdf_processor = PDFProcessor()
    document = pdf_processor.load_single_document(pdf_path)
    
    if not document:
        return {"error": "Failed to process PDF"}
    
    # Create a retriever that returns the full document
    class FullDocRetriever:
        def __init__(self, doc):
            self.doc = doc
            
        def invoke(self, _query):
            return [self.doc]
    
    retriever = FullDocRetriever(document)
    
    # Generate answers for each question
    answers = {}
    for i, question in enumerate(questions):
        try:
            # Create the RAG chain with the specific question for prompt selection
            rag_chain = pipeline.ollama_llm.create_rag_chain(retriever, question=question)
            # Generate the answer
            answer = rag_chain.invoke(question)
            answers[f"question_{i+1}"] = {
                "question": question,
                "answer": answer
            }
        except Exception as e:
            answers[f"question_{i+1}"] = {
                "question": question,
                "answer": f"Error generating answer: {str(e)}"
            }
    
    return answers

def format_as_markdown(filename, answers):
    """Format answers as markdown."""
    md_content = f"# Analysis of {filename}\n\n"
    
    for key, qa in answers.items():
        if not key.startswith("question_"):
            continue
        
        md_content += f"## {qa['question']}\n\n"
        md_content += f"{clean_output(qa['answer'])}\n\n"
    
    return md_content

def format_as_text(filename, answers):
    """Format answers as plain text."""
    text_content = f"Analysis of {filename}\n{'=' * (15 + len(filename))}\n\n"
    
    for key, qa in answers.items():
        if not key.startswith("question_"):
            continue
        
        text_content += f"{qa['question']}\n{'-' * len(qa['question'])}\n\n"
        text_content += f"{clean_output(qa['answer'])}\n\n"
    
    # Clean up any HTML tags that might appear in the LLM output
    # Use a regular expression to remove all HTML tags
    text_content = re.sub(r'<[^>]*>', '', text_content)
    
    return text_content

def clean_output(text):
    """Clean performance metrics and formatting artifacts from LLM output."""
    # Remove word count metrics
    text = re.sub(r'(?i)(Diese Zusammenfassung|Der Text) (umfasst|enthält|besteht aus) \d+(-\d+)? Wörter(n)?\.?', '', text)
    text = re.sub(r'(?i)Wortanzahl: \d+(-\d+)? Wörter(n)?\.?', '', text)
    text = re.sub(r'(?i)\(\d+(-\d+)? Wörter\)', '', text)
    text = re.sub(r'(?i)Etwa \d+(-\d+)? Wörter\.?', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove any trailing whitespace or extra new lines
    text = text.strip()
    
    return text

# Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', questions=DEFAULT_QUESTIONS, model_name=LLM_MODEL)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    # Check if a file was uploaded
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['pdf']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Get questions (use default if not provided)
    questions = json.loads(request.form.get('questions', '[]'))
    if not questions:
        questions = DEFAULT_QUESTIONS
    
    # Return a unique job ID to track this processing task
    job_id = str(uuid.uuid4())
    
    # Store file info for processing
    app.config[f'job_{job_id}'] = {
        'filepath': filepath,
        'questions': questions,
        'filename': filename,
        'unique_filename': unique_filename,
        'answers': {},
        'status': 'processing'
    }
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'total_questions': len(questions)
    })

@app.route('/process/<job_id>', methods=['GET'])
def process_job(job_id):
    """Process a job incrementally and stream results."""
    if f'job_{job_id}' not in app.config:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = app.config[f'job_{job_id}']
    
    if job_data['status'] == 'completed':
        return jsonify({
            'success': True,
            'completed': True,
            'answers': job_data['answers'],
            'md_filename': job_data.get('md_filename', ''),
            'txt_filename': job_data.get('txt_filename', '')
        })
    
    def generate_results():
        # Process the PDF
        pdf_processor = PDFProcessor()
        document = pdf_processor.load_single_document(job_data['filepath'])
        
        if not document:
            yield json.dumps({
                "error": "Failed to process PDF"
            }) + "\n"
            return
        
        # Create a retriever that returns the full document
        class FullDocRetriever:
            def __init__(self, doc):
                self.doc = doc
                
            def invoke(self, _query):
                return [self.doc]
        
        retriever = FullDocRetriever(document)
        all_answers = {}
        
        # Generate answers for each question incrementally
        for i, question in enumerate(job_data['questions']):
            try:
                # Create the RAG chain with the specific question for prompt selection
                rag_chain = pipeline.ollama_llm.create_rag_chain(retriever, question=question)
                # Generate the answer
                answer = rag_chain.invoke(question)
                result = {
                    "question_index": i,
                    "question": question,
                    "answer": answer
                }
                
                # Save the answer in the job data
                all_answers[f"question_{i+1}"] = {
                    "question": question,
                    "answer": answer
                }
                
                yield json.dumps(result) + "\n"
                
            except Exception as e:
                error_result = {
                    "question_index": i,
                    "question": question,
                    "error": str(e)
                }
                all_answers[f"question_{i+1}"] = {
                    "question": question,
                    "answer": f"Error generating answer: {str(e)}"
                }
                yield json.dumps(error_result) + "\n"
        
        # After all questions are processed, create the markdown file
        md_content = format_as_markdown(job_data['filename'], all_answers)
        md_filename = f"{job_data['unique_filename'].rsplit('.', 1)[0]}.md"
        md_filepath = os.path.join(app.config['UPLOAD_FOLDER'], md_filename)
        
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        # Create text file version
        txt_content = format_as_text(job_data['filename'], all_answers)
        txt_filename = f"{job_data['unique_filename'].rsplit('.', 1)[0]}.txt"
        txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        
        # Update job status as completed
        job_data['answers'] = all_answers
        job_data['md_filename'] = md_filename
        job_data['txt_filename'] = txt_filename
        job_data['status'] = 'completed'
        
        # Send final completion message
        completion_result = {
            "completed": True,
            "md_filename": md_filename,
            "txt_filename": txt_filename
        }
        yield json.dumps(completion_result) + "\n"
    
    return Response(stream_with_context(generate_results()), 
                    mimetype='text/event-stream')

@app.route('/download/<filename>')
def download(filename):
    """Download a processed markdown file."""
    mimetype = 'text/markdown' if filename.endswith('.md') else 'text/plain'
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), 
                    mimetype=mimetype, 
                    download_name=filename,
                    as_attachment=True)

@app.route('/download-text/<filename>')
def download_text(filename):
    """Download a processed text file."""
    base_filename = filename.rsplit('.', 1)[0]
    txt_filename = f"{base_filename}.txt"
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], txt_filename), 
                    mimetype='text/plain', 
                    download_name=txt_filename,
                    as_attachment=True)

if __name__ == '__main__':
    print(f"Starting web server with LLM model: {LLM_MODEL}")
    app.run(debug=WEB_DEBUG, host=WEB_HOST, port=WEB_PORT) 