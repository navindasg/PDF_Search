import pyodbc
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict, Optional

def list_tables_and_columns(
    db_server: str = "localhost",
    db_database: str = "table_db",
    db_user: str = "admin",
    db_password: str = "admin",
    db_port: int = 5432,
    driver: str = "PostgreSQL"
) -> str:
    try:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={db_server};"
            f"DATABASE={db_database};"
            f"UID={db_user};"
            f"PWD={db_password};"
            f"PORT={db_port};"
        )

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Get all user tables in public schema
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)

        tables = cursor.fetchall()
        output = []

        for (table_name,) in tables:
            output.append(f"\n📄 Table: {table_name}")

            # Get column names + types
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}' AND table_schema = 'public';
            """)
            columns = cursor.fetchall()
            for col_name, data_type in columns:
                output.append(f"   - {col_name} ({data_type})")

            # Row count
            cursor.execute(f'SELECT COUNT(*) FROM public."{table_name}";')
            row_count = cursor.fetchone()[0]
            output.append(f"   → Row count: {row_count}")

        cursor.close()
        conn.close()
        return "\n".join(output) or "No tables found."

    except Exception as e:
        return f"Error listing tables and columns: {e}"


def query_db(
    query: str,
    db_server: str = "localhost",
    db_database: str = "table_db",
    db_user: str = "admin",
    db_password: str = "admin",
    db_port: int = 5432,
    driver: str = "PostgreSQL"
) -> str:
    try:
        query = query.strip()

        # Construct PostgreSQL ODBC connection string
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={db_server};"
            f"DATABASE={db_database};"
            f"UID={db_user};"
            f"PWD={db_password};"
            f"PORT={db_port};"
        )

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        cursor.execute(query)
        results = cursor.fetchall()

        # Format results as a string
        formatted = "\n".join(
            ", ".join(str(cell) for cell in row) for row in results
        )

        cursor.close()
        conn.close()
        return formatted or "Query executed with no returned rows."

    except Exception as e:
        return f"Query error: {e}"


def similarity_search(
    query: str,
    persist_directory: str = "./text_embeddings",
    k: int = 5,
    metadata_filter: Optional[Dict] = None
) -> List[Dict]:
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    if metadata_filter:
        results = db.similarity_search_with_score(
            query=query,
            k=k,
            filter=metadata_filter
        )
    else:
        results = db.similarity_search_with_score(
            query=query,
            k=k
        )
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    
    return formatted_results


if __name__ == "__main__":
    #list_tables_and_columns()
    #print(query_db("SELECT * FROM public.table_1;"))
    #print(similarity_search("eggs", k=1))
    pass