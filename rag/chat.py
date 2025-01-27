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
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # System message template
        # self.system_template = """You are a helpful assistant that answers questions based on the provided context. 
        # Use the following context to answer the user's question. If you can't find the answer in the context, 
        # say so honestly.
        
        # Context:
        # {context}
        
        # Answer the question based only on the above context."""
        self.system_template = """
        You are an intelligent and reliable assistant. Your primary goal is to provide clear, accurate, and concise answers based strictly on the given context.

        **Rules for Generating Responses:**
        1. **Direct Answers:**
        - If the user's question is directly answered in the context, provide the exact information without adding unrelated details.
        - Ensure your response is precise and factual.

        2. **Partial Information:**
        - If the context contains only part of the information needed to answer the question, provide a partial response and specify what is missing.

        3. **No Relevant Information:**
        - If you cannot find relevant information in the context to answer the question, state:  
            "I could not find the answer in the provided context."

        4. **Contact Information Requests:**
        - If the user asks for contact details (e.g., emails or support) and no such information is found in the context, respond with this default:  
            **CONTACT INFORMATION:**  
            - Website: Omega Solution  
            - Email: support@omega.ac  
            - Customer Support: https://support.omega.ac  

        5. **Quoting Context:**
        - Always be precise when quoting information from the context. Avoid paraphrasing unless absolutely necessary.
        

        6. **Tone:**
        - Maintain a professional, polite, and helpful tone throughout your responses.

        **Context:**
        {context}

        Now, based on the above rules, answer the user's question accurately.
        """


        

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
            search_results = self.vector_store.similarity_search(query, k=5)
            
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