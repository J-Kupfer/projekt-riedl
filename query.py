#!/usr/bin/env python
"""
Script to query the RAG pipeline.
"""
import argparse
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline


def main():
    """
    Main function to query the RAG pipeline.
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "query", 
        nargs="?",
        help="Query to ask"
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
        default="mistral-nemo:12b-instruct-2407-q8_0",
        help="Name of the LLM model to use (default: mistral-nemo:12b-instruct-2407-q8_0)"
    )
    parser.add_argument(
        "--use-full-doc",
        action="store_true",
        help="Use full document instead of vector search"
    )
    parser.add_argument(
        "--doc-name",
        type=str,
        help="Name of the document to use (only with --use-full-doc)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer in real-time"
    )
    args = parser.parse_args()
    
    # Initialize the pipeline
    pipeline = RAGPipeline(
        vector_store_dir=args.vector_store_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model
    )
    
    # Get the query
    if args.query:
        query = args.query
    else:
        query = input("Enter your query: ")
    
    # Generate answer
    if args.use_full_doc:
        print("Using full document for querying...")
        answer = pipeline.query_with_full_document(query, args.doc_name, stream=args.stream)
    else:
        # Check if the non-full-doc query method supports streaming
        try:
            if args.stream:
                answer = pipeline.query(query, stream=args.stream)
            else:
                answer = pipeline.query(query)
        except TypeError:
            if args.stream:
                print("Warning: Streaming is only supported with --use-full-doc. Using non-streaming query.")
            answer = pipeline.query(query)
    
    # Print the answer if not already streamed
    if not args.stream:
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main() 