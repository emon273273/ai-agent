import os
import psycopg2
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables
load_dotenv()

# Verify environment variables
print("DB_NAME:", os.getenv("DB_NAME"))
print("DB_USER:", os.getenv("DB_USER"))
print("DB_HOST:", os.getenv("DB_HOST"))
print("DB_PORT:", os.getenv("DB_PORT"))

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.2-90b-vision-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Connect to PostgreSQL server (without specifying a database)
def get_server_connection():
    return psycopg2.connect(
        dbname="postgres",  # Connect to the default 'postgres' database
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )

# Connect to the database
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )

# Create database if it doesn't exist
def create_database_if_not_exists():
    try:
        conn = get_server_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # Allow database creation
        cur = conn.cursor()

        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (os.getenv("DB_NAME"),))
        if not cur.fetchone():
            # Create the database
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(os.getenv("DB_NAME"))))
            print(f"Database '{os.getenv('DB_NAME')}' created.")

        cur.close()
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")
    finally:
        if conn:
            conn.close()

# Create table if it doesn't exist
def create_table_if_not_exists():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if table exists
        cur.execute("""
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'chat_history'
        """)
        if not cur.fetchone():
            # Create the table
            cur.execute("""
                CREATE TABLE chat_history (
                    id SERIAL PRIMARY KEY,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Table 'chat_history' created.")

        conn.commit()
        cur.close()
    except psycopg2.Error as e:
        print(f"Error creating table: {e}")
    finally:
        if conn:
            conn.close()

# Save message to the database
def save_message_to_db(role, content):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = sql.SQL("INSERT INTO chat_history (role, content) VALUES (%s, %s)")
        cur.execute(query, (role, content))
        conn.commit()
        cur.close()
    except psycopg2.Error as e:
        print(f"Error saving message to database: {e}")
    finally:
        if conn:
            conn.close()

# Load chat history from the database
def load_chat_history_from_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM chat_history ORDER BY timestamp")
        rows = cur.fetchall()
        cur.close()

        chat_history = []
        for row in rows:
            role, content = row  # Unpack the row correctly
            if role == "system":
                chat_history.append(SystemMessage(content=content))
            elif role == "human":
                chat_history.append(HumanMessage(content=content))
            elif role == "ai":
                chat_history.append(AIMessage(content=content))
        return chat_history
    except psycopg2.Error as e:
        print(f"Error loading chat history: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Initialize database and table
create_database_if_not_exists()
create_table_if_not_exists()

# Initialize system message
system_message = SystemMessage(
    content="You are an expert computer science and engineering (CSE) professional. Your role is to provide accurate and helpful information related to computer science, programming, algorithms, and engineering. If asked about topics outside this domain, politely decline and remind the user of your expertise"
)

# Save system message to the database
save_message_to_db("system", system_message.content)

# Load chat history
chat_history = load_chat_history_from_db()

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Save user message to the database
    save_message_to_db("human", query)
    chat_history.append(HumanMessage(content=query))

    # Get AI response
    result = llm.invoke(chat_history)
    print(f"AI: {result.content}")

    # Save AI response to the database
    save_message_to_db("ai", result.content)
    chat_history.append(AIMessage(content=result.content))

print("Chat session ended.")