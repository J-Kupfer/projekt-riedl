"""
LLM module for Ollama integration.
"""
from typing import Dict, Any, Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever


class OllamaWrapper:
    """
    Manages the integration with Ollama LLM.
    """
    
    def __init__(
        self,
        model_name: str = "mistral-nemo:12b-instruct-2407-q8_0",
        temperature: float = 0.7,
        num_ctx: int = 98304
    ):
        """
        Initialize the Ollama LLM.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            num_ctx: Context window size
        """
        self.model_name = model_name
        self.temperature = temperature
        self.num_ctx = num_ctx
        
        # Initialize the LLM
        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_ctx=num_ctx
        )
        
        # Define the standard RAG prompt template
        self.rag_prompt_template = PromptTemplate.from_template(
            """Bitte beantworte die folgende Frage basierend ausschließlich auf dem bereitgestellten Kontext. 
            Wenn die Antwort nicht im Kontext zu finden ist, antworte mit "Die Antwort ist nicht im Dokument enthalten."
            Nutze alle relevanten Informationen und sei so detailliert wie möglich.
            
            Wichtig: Beziehe dich bei deinen Antworten immer auf die Seitenzahlen im Dokument (PAGE NUMBER X). 
            Gib für wichtige Informationen und Zitate immer die Seitenzahl an, auf der sie zu finden sind.
            
            Kontext:
            {context}
            
            Frage: {query}
            
            Antwort:"""
        )
    
    def get_llm(self):
        """
        Get the LLM instance.
        
        Returns:
            The Ollama LLM instance
        """
        return self.llm
    
    def create_rag_chain(self, retriever):
        """
        Create a RAG chain with the given retriever.
        
        Args:
            retriever: Document retriever to use
            
        Returns:
            A runnable chain that can answer questions
        """
        # Define how to format documents
        def format_docs(docs):
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
            return "\n\n".join(formatted_docs)
        
        # Create a function that combines retrieval and formatting
        def retrieve_and_format(query_str):
            docs = retriever.invoke(query_str)
            return format_docs(docs)
        
        # Create the RAG chain
        rag_chain = (
            {
                "context": lambda x: retrieve_and_format(x),
                "query": RunnablePassthrough(),
            }
            | self.rag_prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain 

    def stream_answer(self, query: str, retriever) -> str:
        """
        Stream the answer in real-time to the console.
        
        Args:
            query: User query
            retriever: Document retriever
            
        Returns:
            The complete answer as a string
        """
        # Get context from retriever
        docs = retriever.invoke(query)
        
        # Format documents with page numbers
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
        
        # Prepare inputs
        inputs = {
            "context": context,
            "query": query
        }
        
        # Get prepared prompt
        prompt = self.rag_prompt_template.format_prompt(**inputs).to_string()
        
        # Set streaming=True parameter
        streaming_llm = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            streaming=True
        )
        
        # Stream the response
        response = ""
        print("\nAntwort: ", end="", flush=True)
        for chunk in streaming_llm.stream(prompt):
            print(chunk, end="", flush=True)
            response += chunk
            
        print("\n")  # Add a newline at the end
        
        return response 