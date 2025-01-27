import json
import os
from typing import Dict, List

import numpy as np
import psycopg2
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class HuggingFaceEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

class PostgresVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.connection = None
        self.initialize_database()

    def get_db_connection(self):
        """Create a database connection"""
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )

    def initialize_database(self):
        """Initialize the database with required tables and extensions"""
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        try:
            # Create vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create tables if they don't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id),
                    embedding vector(384)  -- Dimension for 'all-MiniLM-L6-v2' model
                )
            """)
            
            # Create an index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON embeddings 
                USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            print("Database initialized successfully!")
            
        except Exception as e:
            conn.rollback()
            print(f"Error initializing database: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def store_documents(self, docs: List[Dict], document_id: str):
        """Store documents and their embeddings"""
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        try:
            for idx, doc in enumerate(docs):
                chunk_id = f"{document_id}_chunk_{idx}"
                
                # Convert metadata to JSON string
                metadata_json = json.dumps(doc.metadata)
                
                # Store document
                cur.execute("""
                    INSERT INTO documents (id, content, metadata)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE 
                    SET content = EXCLUDED.content, metadata = EXCLUDED.metadata
                """, (chunk_id, doc.page_content, metadata_json))
                
                # Generate and store embedding
                embedding = self.embeddings.embed_query(doc.page_content)
                
                cur.execute("""
                    INSERT INTO embeddings (id, document_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE 
                    SET embedding = EXCLUDED.embedding
                """, (chunk_id, chunk_id, embedding))
            
            conn.commit()
            print(f"Successfully stored {len(docs)} document chunks")
            
        except Exception as e:
            conn.rollback()
            print(f"Error storing documents: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """Search for similar documents using L2 distance"""
        query_embedding = self.embeddings.embed_query(query)
        
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT d.content, d.metadata,
                       1 / (1 + (embedding <-> %s::vector)) as similarity
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                ORDER BY embedding <-> %s::vector ASC
                LIMIT %s
            """, (query_embedding, query_embedding, k))
            
            results = cur.fetchall()
            return [{"content": r[0], 
                    "metadata": r[1] if isinstance(r[1], dict) else {}, 
                    "similarity": r[2]} 
                   for r in results]
            
        finally:
            cur.close()
            conn.close()

def process_document(file_path: str, document_id: str):
    """Process a document and store it in the vector store"""
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    print(f"\n--- Processing document: {document_id} ---")
    
    # Load and split the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    print(f"Number of document chunks: {len(docs)}")
    
    # Initialize and store in Postgres
    vector_store = PostgresVectorStore()
    vector_store.store_documents(docs, document_id)
    
    return vector_store

if __name__ == "__main__":
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "files", "OS-PRO-Local-Installation-Guide_multistore.pdf")
    document_id = "sample_document"
    
    # Process and store the document
    vector_store = process_document(file_path, document_id)
    
    # Example: Perform a similarity search
    query = "How do I install the software?"
    results = vector_store.similarity_search(query)
    
    print("\n--- Search Results ---")
    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx} (similarity: {result['similarity']:.3f}):")
        print(result['content'][:200] + "...")