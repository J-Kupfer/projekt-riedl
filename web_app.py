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
from rag.llm import OllamaLLM

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
    # Use a simple string-based approach instead of regex to avoid potential issues
    def remove_html_tags(text):
        # Simple tag removal without regex
        result = ""
        in_tag = False
        for char in text:
            if char == '<':
                in_tag = True
            elif char == '>':
                in_tag = False
                continue
            elif not in_tag:
                result += char
        return result
    
    return remove_html_tags(text_content)

def clean_output(text):
    """Clean performance metrics and formatting artifacts from LLM output."""
    # Remove word count metrics
    text = re.sub(r'(?i)(Diese Zusammenfassung|Der Text) (umfasst|enthält|besteht aus) \d+(-\d+)? Wörter(n)?\.?', '', text)
    text = re.sub(r'(?i)Wortanzahl: \d+(-\d+)? Wörter(n)?\.?', '', text)
    text = re.sub(r'(?i)\(\d+(-\d+)? Wörter\)', '', text)
    text = re.sub(r'(?i)Etwa \d+(-\d+)? Wörter\.?', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove thinking part from reasoning models (as a backup)
    # Look for patterns that indicate thinking/reasoning before the answer
    thinking_patterns = [
        # Match sections that start with "Let me think" or "I'll think" and similar reasoning starters
        r'(Let me think|I\'ll think|Lass mich nachdenken|Ich denke nach|Thinking through this|Thinking about this|First, I\'ll analyze|Zuerst analysiere ich|Looking at this document|Let\'s start by analyzing)[^.]*?\..*?(?=\n\n|\Z)',
        # Match sections that explain reasoning/thought process
        r'(Um diese Frage zu beantworten|To answer this question|I need to look for|Ich muss nach|First I will|Zuerst werde ich)[^.]*?\..*?(?=\n\n|\Z)',
        # Match sections that discuss analysis approach
        r'(I will analyze|Ich werde analysieren|I\'ll check if|Ich prüfe ob|Let me go through|Ich gehe durch|I need to determine|Ich muss feststellen)[^.]*?\..*?(?=\n\n|\Z)'
    ]
    
    for pattern in thinking_patterns:
        # Fix: Create the regex pattern correctly without combining flags in the middle
        # First compile the pattern without the ^ anchor
        compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        # Then use it to find matches at the beginning of the text or after newlines
        matches = list(compiled_pattern.finditer(text))
        # Process matches in reverse order to avoid index issues when replacing
        for match in reversed(matches):
            # Only replace if at the start of text or after a newline
            pos = match.start()
            if pos == 0 or text[pos-1] == '\n':
                text = text[:match.start()] + text[match.end():]
    
    # Remove any lines that explicitly mention "thinking" or "reasoning"
    text = re.sub(r'(?i)^.*?\b(thinking|reasoning|denken|überlegung)\b.*$\n?', '', text, flags=re.MULTILINE)
    
    # Look for common structural indicators of thinking sections followed by the actual answer
    # This handles cases where the model uses phrases like "Now, the answer is..." or "My answer is..."
    answer_transition_patterns = [
        r'(?i).*?(Now,\s+(?:the|my)\s+answer\s+is:?|Die\s+Antwort\s+ist:?|Meine\s+Antwort\s+ist:?|In\s+summary:?|Zusammenfassend:?|To\s+summarize:?|Therefore:?|Daher:?|Deshalb:?)(.+)',
        r'(?i).*?(After\s+analyzing\s+the\s+document:?|Nach\s+Analyse\s+des\s+Dokuments:?)(.+)'
    ]
    
    for pattern in answer_transition_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Keep only the part after the transition phrase
            text = match.group(2).strip()
    
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
        try:
            # Process the PDF once
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
            
            # Pre-process document once to format it (instead of doing it for each question)
            docs = retriever.invoke("initial_query")
            formatted_docs = []
            for doc in docs:
                page_number = doc.metadata.get("page", 1)
                formatted_text = f"""
=========
PAGE NUMBER {page_number}
=========
{doc.page_content}
=========
PAGE END
========="""
                formatted_docs.append(formatted_text)
            
            context = "\n\n".join(formatted_docs)
            
            # Create a single streaming LLM instance to reuse
            streaming_llm = OllamaLLM(
                model=pipeline.ollama_llm.model_name,
                temperature=pipeline.ollama_llm.temperature,
                num_ctx=pipeline.ollama_llm.num_ctx,
                streaming=True
            )
            
            # Generate answers for each question incrementally
            for i, question in enumerate(job_data['questions']):
                try:
                    # Signal the start of this question processing
                    start_result = {
                        "question_index": i,
                        "question": question,
                        "status": "started"
                    }
                    yield json.dumps(start_result) + "\n"
                    
                    # Select the appropriate prompt template
                    prompt_template = pipeline.ollama_llm.rag_prompt_template
                    if question in pipeline.ollama_llm.question_prompts:
                        prompt_template = pipeline.ollama_llm.question_prompts[question]
                    
                    # Prepare inputs
                    inputs = {
                        "context": context,
                        "query": question
                    }
                    
                    # Format prompt with template
                    prompt = prompt_template.format_prompt(**inputs).to_string()
                    
                    # Stream the response
                    answer = ""
                    for chunk in streaming_llm.stream(prompt):
                        answer += chunk
                        # Send each chunk to the client
                        chunk_result = {
                            "question_index": i,
                            "question": question,
                            "chunk": chunk,
                            "status": "streaming"
                        }
                        yield json.dumps(chunk_result) + "\n"
                    
                    # Signal completion of this question and notify preparation for next question
                    answer_result = {
                        "question_index": i,
                        "question": question,
                        "answer": answer,
                        "status": "completed"
                    }
                    
                    # Save the answer in the job data
                    all_answers[f"question_{i+1}"] = {
                        "question": question,
                        "answer": answer
                    }
                    
                    yield json.dumps(answer_result) + "\n"
                    
                    # If not the last question, notify the user that we're preparing the next one
                    if i < len(job_data['questions']) - 1:
                        prep_result = {
                            "status": "preparing_next",
                            "message": "Preparing next question..."
                        }
                        yield json.dumps(prep_result) + "\n"
                    
                except Exception as e:
                    print(f"Error processing question {i}: {str(e)}")
                    error_result = {
                        "question_index": i,
                        "question": question,
                        "error": str(e),
                        "status": "error"
                    }
                    all_answers[f"question_{i+1}"] = {
                        "question": question,
                        "answer": f"Error generating answer: {str(e)}"
                    }
                    yield json.dumps(error_result) + "\n"
            
            # After all questions are processed, create the markdown and text files
            try:
                # Create markdown file
                md_content = format_as_markdown(job_data['filename'], all_answers)
                md_filename = f"{job_data['unique_filename'].rsplit('.', 1)[0]}.md"
                md_filepath = os.path.join(app.config['UPLOAD_FOLDER'], md_filename)
                
                # Ensure the upload directory exists
                os.makedirs(os.path.dirname(md_filepath), exist_ok=True)
                
                with open(md_filepath, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                # Create text file
                txt_content = format_as_text(job_data['filename'], all_answers)
                txt_filename = f"{job_data['unique_filename'].rsplit('.', 1)[0]}.txt"
                txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
                
                # Ensure the upload directory exists for txt file
                os.makedirs(os.path.dirname(txt_filepath), exist_ok=True)
                
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(txt_content)
                
                # Update job status
                job_data['answers'] = all_answers
                job_data['md_filename'] = md_filename
                job_data['txt_filename'] = txt_filename
                job_data['status'] = 'completed'
                
                # Send completion message
                completion_result = {
                    "completed": True,
                    "md_filename": md_filename,
                    "txt_filename": txt_filename
                }
                yield json.dumps(completion_result) + "\n"
                print(f"Job {job_id} completed successfully")
                
            except Exception as e:
                print(f"Error in file generation phase: {str(e)}")
                error_msg = {
                    "error": f"Error generating output files: {str(e)}"
                }
                yield json.dumps(error_msg) + "\n"
                
                # Even if we have an error, try to send a completion message with any files we did manage to create
                try:
                    md_filename = job_data.get('md_filename', '')
                    txt_filename = job_data.get('txt_filename', '')
                    
                    # If we have at least one file, send a completion
                    if md_filename or txt_filename:
                        completion_result = {
                            "completed": True,
                            "md_filename": md_filename,
                            "txt_filename": txt_filename,
                            "partial": True  # Flag that this is a partial completion
                        }
                        yield json.dumps(completion_result) + "\n"
                        print(f"Job {job_id} completed partially with some errors")
                except Exception as inner_e:
                    print(f"Failed to send completion message: {str(inner_e)}")
                
        except Exception as e:
            print(f"Unexpected error in generate_results: {str(e)}")
            error_msg = {
                "error": f"Unexpected error: {str(e)}"
            }
            yield json.dumps(error_msg) + "\n"
    
    return Response(stream_with_context(generate_results()), 
                    mimetype='text/event-stream')

@app.route('/download/<filename>')
def download(filename):
    """Download a processed markdown file."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": f"File {filename} not found"}), 404
            
        # Determine mimetype based on extension
        mimetype = 'text/markdown' if filename.endswith('.md') else 'text/plain'
        
        return send_file(file_path, 
                        mimetype=mimetype, 
                        download_name=filename,
                        as_attachment=True)
    except Exception as e:
        print(f"Error in download route: {str(e)}")
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500

@app.route('/download-text/<filename>')
def download_text(filename):
    """Download a processed text file."""
    try:
        base_filename = filename.rsplit('.', 1)[0]
        txt_filename = f"{base_filename}.txt"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": f"File {txt_filename} not found"}), 404
            
        return send_file(file_path, 
                        mimetype='text/plain', 
                        download_name=txt_filename,
                        as_attachment=True)
    except Exception as e:
        print(f"Error in download-text route: {str(e)}")
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500

if __name__ == '__main__':
    print(f"Starting web server with LLM model: {LLM_MODEL}")
    app.run(debug=WEB_DEBUG, host=WEB_HOST, port=WEB_PORT) 