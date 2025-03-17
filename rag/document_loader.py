"""
Document loader module for processing PDF files.
"""
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    """
    Handles loading and processing of PDF documents.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1500, 
        chunk_overlap: int = 300
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_and_split(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and split it into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add source metadata
            file_name = os.path.basename(pdf_path)
            for doc in documents:
                doc.metadata["source"] = file_name
                
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            return split_docs
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def format_document_with_page_numbers(self, doc: Document) -> str:
        """
        Format a document with page numbers in a structured way.
        
        Args:
            doc: Document to format
            
        Returns:
            Formatted document text with page numbers
        """
        page_number = doc.metadata.get("page", 1)
        
        formatted_text = f"""
=========
PAGE NUMBER {page_number}
=========
{doc.page_content}
=========
PAGE END
=========
"""
        return formatted_text
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all document chunks
        """
        all_docs = []
        pdf_files = Path(directory_path).glob("**/*.pdf")
        
        for pdf_path in pdf_files:
            print(f"Processing {pdf_path}")
            docs = self.load_and_split(str(pdf_path))
            all_docs.extend(docs)
            
        print(f"Processed {len(all_docs)} document chunks from PDF files")
        return all_docs
    
    def load_full_document(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file without splitting it into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of documents (one per page)
        """
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add source metadata
            file_name = os.path.basename(pdf_path)
            for doc in documents:
                doc.metadata["source"] = file_name
                
            return documents
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def load_single_document(self, pdf_path: str) -> Document:
        """
        Load a PDF file and combine all pages into a single document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Single document with all content
        """
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                return None
                
            # Combine all pages with page number formatting
            combined_text = ""
            for doc in documents:
                page_number = doc.metadata.get("page", 1)
                page_text = f"""
=========
PAGE NUMBER {page_number}
=========
{doc.page_content}
=========
PAGE END
=========
"""
                combined_text += page_text + "\n"
            
            # Use metadata from the first page
            metadata = documents[0].metadata.copy()
            metadata["source"] = os.path.basename(pdf_path)
            metadata["is_full_document"] = True
            
            # Create a single document
            return Document(page_content=combined_text, metadata=metadata)
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None 