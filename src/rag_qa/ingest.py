"""Ingest PDF / text documents into a FAISS vector store."""
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "faiss_store"


def ingest_file(file_path: str, chunk_size: int = 512, chunk_overlap: int = 64) -> int:
    """Chunk and embed a document, saving to FAISS. Returns number of chunks."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(str(path)) if path.suffix == ".pdf" else TextLoader(str(path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    index_path = Path(INDEX_PATH)
    if index_path.exists():
        store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)

    store.save_local(INDEX_PATH)
    print(f"✅ Ingested {len(chunks)} chunks from '{path.name}' into '{INDEX_PATH}'")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_file>")
        sys.exit(1)
    ingest_file(sys.argv[1])
