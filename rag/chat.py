import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rag import PostgresVectorStore

# Load environment variables
load_dotenv()

class DocumentChat:
    def __init__(self):
        # Initialize the PostgreSQL vector store
        self.vector_store = PostgresVectorStore()
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # System message template
        self.system_template = """You are a helpful assistant that answers questions based on the provided context. 
        Use the following context to answer the user's question. If you can't find the answer in the context, 
        say so honestly.
        
        Context:
        {context}
        
        Answer the question based only on the above context."""

    def format_context(self, results: List[Dict]) -> str:
        """Format the search results into a single context string"""
        context_parts = []
        for idx, result in enumerate(results, 1):
            content = result.get('content', '').strip()
            similarity = result.get('similarity', 0)
            context_parts.append(f"[Excerpt {idx} (Relevance: {similarity:.2f})]\n{content}\n")
        return "\n".join(context_parts)

    def generate_response(self, query: str) -> str:
        """Generate a response to the user's query"""
        try:
            # Get relevant context from the vector store
            search_results = self.vector_store.similarity_search(query, k=3)
            
            if not search_results:
                return "I couldn't find any relevant information in the documents to answer your question."
                
            context = self.format_context(search_results)
            
            # Create the messages for the LLM using LangChain message types
            messages = [
                SystemMessage(content=self.system_template.format(context=context)),
                HumanMessage(content=query)
            ]
            
            # Get the response from the LLM
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())  # This will print the full error trace
            return f"Error generating response: {str(e)}"

def main():
    chat = DocumentChat()
    
    print("\n=== Document Chat Interface ===")
    print("Welcome! Ask questions about your documents. Type 'quit' to exit.")
    print("Loading document context...")
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            if not query:
                continue
                
            print("\nSearching and generating response...")
            response = chat.generate_response(query)
            print("\nAssistant:", response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()