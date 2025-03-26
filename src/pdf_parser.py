import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import json
import pypdf
import fitz
import tabula
import pandas as pd
import pyodbc
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import base64
import shutil

def parse(filename):
    """
    Parse a PDF file to extract text, images, and tables.
    
    Args:
        filename (str): Path to the PDF file
        
    Returns:
        dict: Paths to the extracted files
    """
    # Create output directories if they don't exist
    output_dir = "./src/parser_output"
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    result = {
        "text_file": os.path.join(output_dir, "pdf_text.txt"),
        "tables_file": os.path.join(output_dir, "pdf_tables.json"),
        "images_dir": image_dir
    }
    
    # Extract text using pypdf
    text = ""
    with open(filename, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        print(f"Processing {filename}: {len(pdf_reader.pages)} pages")
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
    
    # Save the extracted text
    with open(result["text_file"], "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text extracted to {result['text_file']}")
    
    # Extract images using PyMuPDF
    pdf_document = fitz.open(filename)
    image_count = 0
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(image_dir, f"page{page_num+1}_img{img_index+1}.{image_ext}")
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
                image_count += 1
    
    print(f"{image_count} images extracted to {image_dir}")
    
    # Extract tables using tabula
    try:
        tables = tabula.read_pdf(filename, pages='all', multiple_tables=True)
        tables_data = []
        
        for i, table in enumerate(tables):
            # Convert DataFrame to dict
            table_dict = table.to_dict(orient='records')
            tables_data.append({
                "table_id": i + 1,
                "data": table_dict
            })
        
        # Save tables to JSON
        with open(result["tables_file"], "w", encoding="utf-8") as f:
            json.dump(tables_data, f, indent=2, default=str)
        
        print(f"{len(tables)} tables extracted to {result['tables_file']}")
    except Exception as e:
        print(f"Error extracting tables: {e}")
        
    return result

def describe_image(folder_path):
    vision_model = OllamaLLM(model="llava:13b")
    results = []
    for image_path in os.listdir(folder_path):
        image_full_path = os.path.join(folder_path, image_path)

        # Read raw file bytes and encode them directly
        with open(image_full_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Pass base64-encoded image file content
        vision_model_context = vision_model.bind(images=[image_base64])
        response = vision_model_context.invoke("Concisely describe the image. If there is any text in the image, include it as one line in the description.")
        results.append(response)
    f = open("./src/parser_output/descriptions.txt", "w")
    for result in results:
        f.write(result + "\n")
    f.close()

def embeddings(folder_path, persist_directory="text_embeddings", chunk_size=1000, chunk_overlap=200):
    # Initialize the embedding model
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Initialize text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    all_docs = []
    
    # Process each text file
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(page_content=content, metadata={"source": filename})
                chunks = text_splitter.split_documents([doc])
                all_docs.extend(chunks)
    
    db = Chroma.from_documents(all_docs, embedding_model, persist_directory=persist_directory)

def tables_to_db(json_path, dsn="PostgresDSN", schema="public"):
    # Connect to your PostgreSQL database via DSN
    conn = pyodbc.connect(f"DSN={dsn}")
    cursor = conn.cursor()

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    for table in tables:
        table_id = table.get("table_id")
        data = table.get("data")

        if not data:
            print(f"Skipping empty table {table_id}")
            continue

        df = pd.DataFrame(data)
        table_name = f"table_{table_id}"

        # Build CREATE TABLE SQL
        create_stmt = f'CREATE TABLE "{schema}"."{table_name}" ('
        for col in df.columns:
            safe_col = col.replace(" ", "_")  # sanitize column names
            create_stmt += f'"{safe_col}" TEXT, '
        create_stmt = create_stmt.rstrip(", ") + ")"

        try:
            cursor.execute(f'DROP TABLE IF EXISTS "{schema}"."{table_name}"')
            cursor.execute(create_stmt)
            print(f"Created table: {schema}.{table_name}")
        except Exception as e:
            print(f"Error creating table {table_name}: {e}")
            continue

        # Insert rows
        for _, row in df.iterrows():
            columns = ", ".join([f'"{col.replace(" ", "_")}"' for col in df.columns])
            placeholders = ", ".join(["?"] * len(df.columns))
            values = tuple(row)
            insert_sql = f'INSERT INTO "{schema}"."{table_name}" ({columns}) VALUES ({placeholders})'
            cursor.execute(insert_sql, values)

    conn.commit()
    conn.close()
    print("All tables loaded successfully.")
    print("All tables inserted.")

def main(pdf_input_directory="./src/pdf_input"):
    #find all pdfs in the input directory
    pdfs = [f for f in os.listdir(pdf_input_directory) if f.endswith(".pdf")]
    for pdf in pdfs:
        parse(os.path.join(pdf_input_directory, pdf))
        describe_image("./src/parser_output/images")
        embeddings("./src/parser_output")
        tables_to_db("./src/parser_output/pdf_tables.json")
        shutil.rmtree("./src/parser_output")
    print("All PDFs processed.")

if __name__ == "__main__":
    main()
