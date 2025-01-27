import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


llm = ChatGroq(
    model="llama-3.2-90b-vision-preview",  
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY")  # Load API key from .env
)


