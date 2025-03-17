"""
Pipeline module for orchestrating the RAG workflow using LangGraph.
"""
from typing import Dict, List, Any, TypedDict, Annotated
import json
import os
from pathlib import Path

from langgraph.graph import StateGraph
from langchain_core.documents import Document

from rag.document_loader import PDFProcessor
from rag.vector_store import ChromaVectorStore
from rag.llm import OllamaWrapper


class RAGState(TypedDict):
    """
    State definition for the RAG pipeline.
    """
    query: str
    documents: List[Document]
    answer: str


class RAGPipeline:
    """
    RAG pipeline for document ingestion and query answering.
    """
    
    def __init__(
        self,
        pdf_directory: str = "data/pdfs",
        vector_store_dir: str = "data/chroma",
        embedding_model: str = "jina/jina-embeddings-v2-base-de",
        llm_model: str = "mistral-nemo:12b-instruct-2407-q8_0"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            pdf_directory: Directory containing PDF files
            vector_store_dir: Directory to store vector embeddings
            embedding_model: Name of the embedding model
            llm_model: Name of the LLM model
        """
        self.pdf_directory = pdf_directory
        self.vector_store_dir = vector_store_dir
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.vector_store = ChromaVectorStore(
            persist_directory=vector_store_dir,
            embedding_model=embedding_model
        )
        self.ollama_llm = OllamaWrapper(model_name=llm_model)
        
        # Store full documents for direct access
        self.full_documents = {}
        
        # Initialize LangGraph
        self.graph = self._build_graph()
    
    def ingest_documents(self, store_full_docs: bool = True) -> None:
        """
        Ingest PDF documents into the vector store.
        
        Args:
            store_full_docs: Whether to also store full documents for direct access
        """
        # Process PDFs for vector store
        documents = self.pdf_processor.process_directory(self.pdf_directory)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Load and store full documents if requested
        if store_full_docs:
            self._load_full_documents()
    
    def _load_full_documents(self) -> None:
        """
        Load and store full documents for direct access.
        """
        pdf_files = Path(self.pdf_directory).glob("**/*.pdf")
        for pdf_path in pdf_files:
            str_path = str(pdf_path)
            file_name = os.path.basename(str_path)
            doc = self.pdf_processor.load_single_document(str_path)
            if doc:
                self.full_documents[file_name] = doc
                print(f"Stored full document: {file_name}")
    
    def query_with_full_document(self, query: str, doc_name: str = None, stream: bool = False) -> str:
        """
        Query using a full document instead of vector search.
        
        Args:
            query: User query
            doc_name: Name of the document to use (if None, uses the first available)
            stream: Whether to stream the output
            
        Returns:
            Generated answer
        """
        if not self.full_documents:
            self._load_full_documents()
            
        if not self.full_documents:
            return "No documents available for query."
        
        # Select the document
        if doc_name and doc_name in self.full_documents:
            document = self.full_documents[doc_name]
        else:
            # Use the first available document
            doc_name = next(iter(self.full_documents))
            document = self.full_documents[doc_name]
            
        print(f"Using full document: {doc_name}")
        
        # Create a retriever that returns the full document
        class FullDocRetriever:
            def __init__(self, doc):
                self.doc = doc
                
            def invoke(self, _query):
                # Ignore the query, just return the pre-retrieved docs
                return [self.doc]
        
        retriever = FullDocRetriever(document)
        
        if stream:
            # Stream the answer in real-time
            return self.ollama_llm.stream_answer(query, retriever)
        else:
            # Create the RAG chain
            rag_chain = self.ollama_llm.create_rag_chain(retriever)
            
            # Generate the answer
            try:
                answer = rag_chain.invoke(query)
                return answer
            except Exception as e:
                print(f"Error generating answer: {str(e)}")
                return "Sorry, I couldn't generate an answer due to an error."
    
    def _retrieval_node(self, state: RAGState) -> RAGState:
        """
        Process the user query and retrieve relevant documents.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with retrieved documents
        """
        query = state["query"]
        documents = self.vector_store.similarity_search(query)
        
        return {"query": query, "documents": documents, "answer": ""}
    
    def _generate_answer_node(self, state: RAGState) -> RAGState:
        """
        Generate an answer using the LLM.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with answer
        """
        query = state["query"]
        documents = state["documents"]
        
        # Create a retriever that returns the already retrieved documents
        class PreretrievedRetriever:
            def __init__(self, docs):
                self.docs = docs
                
            def invoke(self, _query):
                # Ignore the query, just return the pre-retrieved docs
                return self.docs
        
        retriever = PreretrievedRetriever(documents)
        
        # Create the RAG chain
        rag_chain = self.ollama_llm.create_rag_chain(retriever)
        
        # Generate the answer
        try:
            answer = rag_chain.invoke(query)
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            answer = "Sorry, I couldn't generate an answer due to an error."
        
        return {"query": query, "documents": documents, "answer": answer}
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            StateGraph instance
        """
        # Create the graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retrieval", self._retrieval_node)
        builder.add_node("generate_answer", self._generate_answer_node)
        
        # Connect nodes
        builder.add_edge("retrieval", "generate_answer")
        builder.set_entry_point("retrieval")
        builder.set_finish_point("generate_answer")
        
        # Compile the graph
        return builder.compile()
    
    def query(self, query: str) -> str:
        """
        Query the RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Generated answer
        """
        # Initialize state
        state = {"query": query, "documents": [], "answer": ""}
        
        # Run the graph
        result = self.graph.invoke(state)
        
        return result["answer"] 