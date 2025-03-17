import argparse
import os
import shutil
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)

        # Extract the author from the filename.
        author = extract_author_from_filename(filename)

        if filename.endswith(".pdf"):
            document_loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            document_loader = TextLoader(file_path)
        else:
            continue  # Skip unsupported files

        docs = document_loader.load()

        # Add author information to the document metadata
        for doc in docs:
            doc.metadata["author"] = author  # Add the "author" tag

        documents.extend(docs)
    return documents


def extract_author_from_filename(filename):
    """
    A method to extract authors based on filename with multiple authors.
    For example, 'michel_foucault.pdf' would extract 'michel_foucault'.
    """
    # Remove the file extension (e.g., ".pdf")
    base_filename = filename.rsplit(".", 1)[0]

    # Split the filename by the underscore to get individual authors
    authors = base_filename.split("_")

    # Join authors with a space or keep them as a list, depending on your preference
    author_string = " ".join(authors)

    return author_string


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    """Detects if Ollama is available; otherwise, uses HuggingFace embeddings."""
    try:
        # Testa se o Ollama está rodando
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        
        # Se Ollama estiver rodando, usa "nomic-embed-text"
        print("🔹 Using Ollama (nomic-embed-text)")
        return OllamaEmbeddings(model="nomic-embed-text")

    except (requests.ConnectionError, requests.Timeout):
        # Se Ollama não estiver rodando, usa MiniLM com llama.cpp
        print("⚠️ Ollama is probably disabled. Using all-MiniLM-L6-v2 via HuggingFace.")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
