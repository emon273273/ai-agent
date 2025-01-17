from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm=ChatGroq(model="llama-3.2-90b-vision-preview")

message=[
    SystemMessage("You are a expert Cse Enginner"),
    HumanMessage("Give me a c code of find prime number of 1 to 100")
]

result=llm.invoke(message)
print(result.content)