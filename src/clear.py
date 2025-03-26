import pyodbc
import shutil
import os
from langchain_chroma import Chroma

def clean_public_schema(dsn="PostgresDSN"):
    """
    Deletes all tables in the public schema without dropping the schema itself.
    
    Args:
        dsn (str): The data source name for PostgreSQL connection
    """
    # Connect to PostgreSQL database
    conn = pyodbc.connect(f"DSN={dsn}")
    cursor = conn.cursor()
    
    try:
        # Find all tables in the public schema
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        
        tables = cursor.fetchall()
        
        # Drop each table
        for (table,) in tables:
            print(f"Dropping table: public.{table}")
            cursor.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')
        
        conn.commit()
        print(f"All tables in public schema dropped successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error occurred: {e}")
    finally:
        cursor.close()
        conn.close()

def clear():
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "parser_output")
    persist_directory = "./text_embeddings"

    # Delete parser_output folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted: {output_dir}")
    else:
        print(f"{output_dir} not found.")

    # Clear Chroma vector store
    if not os.path.exists(persist_directory):
        print(f"Vector database directory {persist_directory} does not exist.")
        return

    print(f"Connecting to Chroma database at {persist_directory}...")
    vectorstore = Chroma(persist_directory=persist_directory)

    count_before = vectorstore._collection.count()
    print(f"Current number of vectors: {count_before}")

    if count_before == 0:
        print("Database is already empty.")
        return

    all_ids = vectorstore._collection.get()["ids"]
    print("Deleting all vectors...")
    vectorstore.delete(ids=all_ids)

    count_after = vectorstore._collection.count()
    print(f"Vectors after deletion: {count_after}")

    if count_after == 0:
        print("Database successfully cleared.")
    else:
        print(f"Warning: {count_after} vectors remain in the database.")



if __name__ == "__main__":
    clean_public_schema()
    clear()