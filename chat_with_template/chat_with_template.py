import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatGroq model
llm = ChatGroq(
    model="llama-3.2-90b-vision-preview",  
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY")  # Load API key from .env
)

# Define a function to translate text
def translate_text(language: str, input_text: str) -> str:
   
    # Define the system template
    system_template = "Translate the following from English into {language}"

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{input}")
    ])

    # Invoke the prompt template with user input
    prompt = prompt_template.invoke({"language": language, "input": input_text})

    # Use the LLM to generate a response
    result = llm.invoke(prompt)

    # Return the translated text
    return result.content

# Example usage
if __name__ == "__main__":
    # Call the function with parameters
    translated_text = translate_text("spanish", "hey my name is emon")
    print(translated_text)  # Output: "Hola"