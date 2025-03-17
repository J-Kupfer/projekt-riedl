#!/usr/bin/env python
"""
Script to ingest PDF documents into the vector store.
"""
import os
import argparse
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline


def main():
    """
    Main function to ingest documents.
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the vector store")
    parser.add_argument(
        "--pdf-dir", 
        default="data/pdfs",
        help="Directory containing PDF files (default: data/pdfs)"
    )
    parser.add_argument(
        "--vector-store-dir", 
        default="data/chroma",
        help="Directory to store vector embeddings (default: data/chroma)"
    )
    parser.add_argument(
        "--embedding-model", 
        default="jina/jina-embeddings-v2-base-de",
        help="Name of the embedding model to use (default: jina/jina-embeddings-v2-base-de)"
    )
    args = parser.parse_args()
    
    # Check if PDF directory exists
    if not os.path.exists(args.pdf_dir):
        print(f"Creating PDF directory: {args.pdf_dir}")
        os.makedirs(args.pdf_dir, exist_ok=True)
    
    # Initialize the pipeline
    pipeline = RAGPipeline(
        pdf_directory=args.pdf_dir,
        vector_store_dir=args.vector_store_dir,
        embedding_model=args.embedding_model
    )
    
    # Ingest documents
    pipeline.ingest_documents()


if __name__ == "__main__":
    main() 