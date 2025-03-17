#!/usr/bin/env python
"""
Interactive application for RAG queries.
"""
import os
import argparse
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline


def main():
    """
    Main function for the interactive application.
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interactive RAG application")
    parser.add_argument(
        "--pdf-dir", 
        default="data/pdfs",
        help="Directory containing PDF files (default: data/pdfs)"
    )
    parser.add_argument(
        "--vector-store-dir", 
        default="data/chroma",
        help="Directory of the vector store (default: data/chroma)"
    )
    parser.add_argument(
        "--embedding-model", 
        default="jina/jina-embeddings-v2-base-de",
        help="Name of the embedding model to use (default: jina/jina-embeddings-v2-base-de)"
    )
    parser.add_argument(
        "--llm-model", 
        default="qwq:32b",
        help="Name of the LLM model to use (default: qwq:32b)"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents before starting the application"
    )
    parser.add_argument(
        "--use-full-docs",
        action="store_true",
        help="Use full documents instead of vector search"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answers in real-time"
    )
    args = parser.parse_args()
    
    # Initialize the pipeline
    pipeline = RAGPipeline(
        pdf_directory=args.pdf_dir,
        vector_store_dir=args.vector_store_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model
    )
    
    # Ingest documents if requested
    if args.ingest:
        print(f"Ingesting documents from {args.pdf_dir}...")
        pipeline.ingest_documents()
    
    # Check if vector store exists
    if not os.path.exists(args.vector_store_dir) or not os.listdir(args.vector_store_dir):
        print("Vector store not found or empty. Ingesting documents...")
        pipeline.ingest_documents()
    
    print("\n=== PDF RAG Application ===")
    print(f"Using embedding model: {args.embedding_model}")
    print(f"Using LLM model: {args.llm_model}")
    print("Type 'exit' or 'quit' to end the application")
    print("Type 'use full' to switch to full document mode")
    print("Type 'use vector' to switch to vector search mode")
    print("Type 'list docs' to show available documents in full document mode\n")
    
    # Set initial mode
    use_full_docs = args.use_full_docs
    stream_output = args.stream
    if use_full_docs:
        print("Starting in FULL DOCUMENT mode")
    else:
        print("Starting in VECTOR SEARCH mode")
        
    if stream_output:
        print("Streaming is ENABLED")
    else:
        print("Streaming is DISABLED")
    
    print("Type 'stream on' to enable streaming")
    print("Type 'stream off' to disable streaming")
    
    # Ensure full documents are loaded if needed
    if use_full_docs:
        if not hasattr(pipeline, 'full_documents') or not pipeline.full_documents:
            pipeline._load_full_documents()
    
    # Start interactive loop
    while True:
        # Get user query
        query = input("\nEnter your query: ")
        
        # Check if user wants to exit
        if query.lower() in ["exit", "quit"]:
            print("Exiting application...")
            break
            
        # Check for mode switching commands
        if query.lower() == "use full":
            use_full_docs = True
            print("Switched to FULL DOCUMENT mode")
            if not hasattr(pipeline, 'full_documents') or not pipeline.full_documents:
                pipeline._load_full_documents()
            continue
            
        if query.lower() == "use vector":
            use_full_docs = False
            print("Switched to VECTOR SEARCH mode")
            continue
            
        if query.lower() == "stream on":
            stream_output = True
            print("Streaming is now ENABLED")
            continue
            
        if query.lower() == "stream off":
            stream_output = False
            print("Streaming is now DISABLED")
            continue
            
        if query.lower() == "list docs":
            if hasattr(pipeline, 'full_documents') and pipeline.full_documents:
                print("\nAvailable documents:")
                for idx, doc_name in enumerate(pipeline.full_documents.keys()):
                    print(f"{idx+1}. {doc_name}")
            else:
                print("No documents loaded")
            continue
        
        # Check if user wants to use a specific document
        doc_name = None
        if use_full_docs and query.lower().startswith("doc:"):
            parts = query.split(" ", 1)
            if len(parts) >= 2:
                doc_spec = parts[0].strip()[4:]  # Remove "doc:"
                query = parts[1].strip()
                
                # Check if doc_spec is a number or name
                if doc_spec.isdigit() and hasattr(pipeline, 'full_documents'):
                    idx = int(doc_spec) - 1
                    if 0 <= idx < len(pipeline.full_documents):
                        doc_name = list(pipeline.full_documents.keys())[idx]
                else:
                    doc_name = doc_spec
        
        # Generate answer
        print("Generating answer...")
        try:
            if use_full_docs:
                answer = pipeline.query_with_full_document(query, doc_name, stream=stream_output)
            else:
                answer = pipeline.query(query, stream=stream_output)
                
            # Only print the answer if not streaming (streaming already prints)
            if not stream_output:
                print("\nAnswer:")
                print(answer)
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 