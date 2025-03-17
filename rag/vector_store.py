"""
Vector store module for storing and retrieving document embeddings.
"""
from typing import List, Optional, Dict, Any
import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class ChromaVectorStore:
    """
    Manages the Chroma vector store for document embeddings.
    """
    
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "pdf_docs"
    ):
        """
        Initialize the Chroma vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
            embedding_model: Name of the embedding model to use with Ollama
            collection_name: Name of the collection in Chroma
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create the persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Set up the embedding function
        self.embedding_function = OllamaEmbeddings(model=embedding_model)
        
        # Initialize the vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
            collection_name=collection_name
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            print("No documents to add.")
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vectorstore.add_documents(documents)
        
        # Chroma automatically persists changes when using a persist_directory
        print("Documents added to vector store.")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """
        Perform a similarity search for a query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever for the vector store.
        
        Args:
            search_kwargs: Search parameters
            
        Returns:
            Retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 8}
            
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_vectorstore(self) -> VectorStore:
        """
        Get the underlying vector store.
        
        Returns:
            The Chroma vector store
        """
        return self.vectorstore 