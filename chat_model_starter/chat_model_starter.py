#from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    # model="mixtral-8x7b-32768",
    model="llama-3.2-90b-vision-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


messages="I have a cat name and i want a cool name for it . Suggest me five cool names for my cat"
ai_msg = llm.invoke(messages)

print(ai_msg.content)
