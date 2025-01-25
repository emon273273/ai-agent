from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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


# message=[
#     SystemMessage("You are an expert computer science and engineering (CSE) professional. Your role is to provide accurate and helpful information related to computer science, programming, algorithms, and engineering. If asked about topics outside this domain, politely decline and remind the user of your expertise"),
#     HumanMessage("give me beauty post") 
# ]

system=SystemMessage("You are an expert computer science and engineering (CSE) professional. Your role is to provide accurate and helpful information related to computer science, programming, algorithms, and engineering. If asked about topics outside this domain, politely decline and remind the user of your expertise")



chat_history=[]
chat_history.append(system)

while(True):
    query=input("You: ")
    if(query.lower()=='exit'):
        break
    chat_history.append(HumanMessage(content=query))

    result=llm.invoke(chat_history)

    print(f"AI: {result.content}")
    chat_history.append(AIMessage(content=result.content))

print(f"Chat History: {chat_history}")
