import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dxc_rag_pipeline.loaders import get_document_loader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCE_DIRECTORY = str(PROJECT_ROOT / "source_documents")
DB_PATH = str(PROJECT_ROOT / "chroma_db")

def build_database():
    documents = []
    print(f"Loading documents from {SOURCE_DIRECTORY}...")
    
    for filename in os.listdir(SOURCE_DIRECTORY):
        file_path = os.path.join(SOURCE_DIRECTORY, filename)
        try:
            loader = get_document_loader(file_path)
            if loader:
                print(f"  - Loading {filename}...")
                documents.extend(loader.load())
            else:
                print(f"  - Skipping unsupported file: {filename}")
        except Exception as e:
            print(f"Error loading file {filename}: {e}")

    if not documents:
        print("No documents were loaded. Exiting.")
        sys.exit(1)
    
    print(f"Loaded a total of {len(documents)} document sections.")

    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks.")

    print("Initializing embedding model (mxbai-embed-large:latest)...")
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")

    print("Creating embeddings and building vector store... (This may take a moment)")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print("--------------------------------------------------")
    print(f"Vector store created successfully.")
    print(f"Database has been saved to: {DB_PATH}")
    print("--------------------------------------------------")

def load_retriever():
    print("Loading existing vector store to create retriever...")
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    print("Retriever is ready.")
    return vectorstore.as_retriever()


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        print(f"Found existing database at {DB_PATH}. Deleting it to create a new one.")
        shutil.rmtree(DB_PATH)
    
    build_database()
