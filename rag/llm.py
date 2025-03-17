"""
LLM module for Ollama integration.
"""
from typing import Dict, Any, Optional, List

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
            
            Wichtig: 
            1. Beziehe dich bei deinen Antworten immer auf die Seitenzahlen im Dokument (PAGE NUMBER X).
            2. Gib für wichtige Informationen und Zitate immer die Seitenzahl an, auf der sie zu finden sind.
            
            Kontext:
            {context}
            
            Frage: {query}
            
            Antwort:"""
        )
        
        # Define question-specific prompts
        self.question_prompts = {
            "Bitte schreibe eine zusammenfassung der Akte in ca. 250 Wörtern": PromptTemplate.from_template(
                """Bitte schreibe eine Zusammenfassung der folgenden Akte in GENAU 200-250 Wörtern. 
                Die Antwort MUSS auf Deutsch sein und die Wortzahl zwischen 200 und 250 liegen. Nicht mehr und nicht weniger.
                
                Beziehe dich dabei auf die wichtigsten Aspekte des Falles, wie Tatbestand, beteiligte Personen, Ergebnisse, 
                und rechtliche Einschätzungen. Nutze alle relevanten Informationen aus dem bereitgestellten Kontext.
                
                Wichtig: 
                1. Beziehe dich bei deinen Antworten immer auf die Seitenzahlen im Dokument (PAGE NUMBER X).
                2. Gib für wichtige Informationen und Zitate immer die Seitenzahl an, auf der sie zu finden sind.
                3. Erwähne NICHT die Wortzahl in deiner Antwort. Schreibe die Zusammenfassung einfach mit der richtigen Länge.
                
                Kontext:
                {context}
                
                Frage: {query}
                
                Antwort:"""
            ),
            "Bitte erstelle ein formatiertes Inhaltsverzeichnis mit zusammenfassenden Titeln und Seitenangaben": PromptTemplate.from_template(
                """Erstelle ein detailliertes, formatiertes Inhaltsverzeichnis für das vorliegende Dokument. 
                
                Das Inhaltsverzeichnis soll:
                - Alle wichtigen Abschnitte und Unterabschnitte des Dokuments enthalten
                - Für jeden Eintrag einen kurzen, aussagekräftigen Titel haben, der den Inhalt zusammenfasst
                - Bei jedem Eintrag die entsprechende Seitenzahl angeben
                - Eine klare hierarchische Struktur aufweisen
                
                Bitte achte besonders auf eine übersichtliche Formatierung, die die Struktur des Dokuments deutlich macht.
                Verwende die Seitenzahlen, die im Kontext als "PAGE NUMBER X" angegeben sind.
                
                Kontext:
                {context}
                
                Frage: {query}
                
                Antwort:"""
            ),
            "Bitte gib mir, formatiert, eine Ausgabe über alle Beteiligte mit Kontaktangaben, außer der Kanzlei Riedl": PromptTemplate.from_template(
                """Erstelle eine strukturierte Liste aller am Fall beteiligten Personen und Parteien mit ihren vollständigen Kontaktdaten.
                
                Die Liste soll:
                - Alle Beteiligten mit vollständigem Namen auflisten
                - Für jede Person/Partei alle verfügbaren Kontaktinformationen angeben (Adresse, Telefonnummer, E-Mail, etc.)
                - Klar formatiert und übersichtlich sein
                - Die Kanzlei Riedl und deren Mitarbeiter NICHT enthalten
                - Die Rolle jeder Person im Fall angeben (z.B. Beschuldigter, Zeuge, Geschädigter, etc.)
                
                Bitte gib für jede Information die entsprechende Seitenzahl an, auf der sie im Dokument zu finden ist.
                
                Kontext:
                {context}
                
                Frage: {query}
                
                Antwort:"""
            ),
            "Waren bei dem Fall Drogen, Alkohol, Medikamente oder Fahrerflucht im Spiel": PromptTemplate.from_template(
                """Untersuche das Dokument gründlich nach Hinweisen auf:
                1. Drogenkonsum oder -besitz
                2. Alkoholkonsum
                3. Einnahme von Medikamenten mit Auswirkung auf die Fahrtüchtigkeit
                4. Fahrerflucht
                
                Beantworte die Frage eindeutig mit Ja oder Nein für jeden dieser Aspekte und führe die entsprechenden Belege mit Seitenzahlen an.
                Falls zu einem Aspekt keine Informationen vorliegen, gib an, dass dazu keine Angaben im Dokument zu finden sind.
                
                Kontext:
                {context}
                
                Frage: {query}
                
                Antwort:"""
            ),
            "Bitte beantworte kurz und knapp wer der Schuldige in dem Fall war und wie hoch der Schade ist": PromptTemplate.from_template(
                """Beantworte präzise und kompakt folgende zwei Fragen:
                
                1. Wer trägt die Schuld in diesem Fall? (Nenne konkret die Person oder Partei)
                2. Wie hoch ist der entstandene Schaden? (Gib den exakten Betrag mit Währung an)
                
                Die Antwort soll kurz und prägnant sein, nicht mehr als 2-3 Sätze. Beziehe dich ausschließlich auf die Fakten aus dem Dokument und gib die entsprechenden Seitenzahlen an.
                
                Kontext:
                {context}
                
                Frage: {query}
                
                Antwort:"""
            )
        }
    
    def get_llm(self):
        """
        Get the LLM instance.
        
        Returns:
            The Ollama LLM instance
        """
        return self.llm
    
    def create_rag_chain(self, retriever, question=None):
        """
        Create a RAG chain with the given retriever.
        
        Args:
            retriever: Document retriever to use
            question: The question to determine which prompt to use
            
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
        
        # Select the appropriate prompt template
        prompt_template = self.rag_prompt_template
        if question and question in self.question_prompts:
            prompt_template = self.question_prompts[question]
        
        # Create the RAG chain
        rag_chain = (
            {
                "context": lambda x: retrieve_and_format(x),
                "query": RunnablePassthrough(),
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

    def stream_answer(self, query: str, retriever, question=None) -> str:
        """
        Stream the answer in real-time to the console.
        
        Args:
            query: User query
            retriever: Document retriever
            question: The question to determine which prompt to use
            
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
        
        # Select the appropriate prompt template
        prompt_template = self.rag_prompt_template
        if question and question in self.question_prompts:
            prompt_template = self.question_prompts[question]
        
        # Prepare inputs
        inputs = {
            "context": context,
            "query": query
        }
        
        # Get prepared prompt
        prompt = prompt_template.format_prompt(**inputs).to_string()
        
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