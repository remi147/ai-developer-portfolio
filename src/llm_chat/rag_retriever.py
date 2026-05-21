"""FAISS-based RAG retriever for the LLM Chat Assistant."""
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

INDEX_DIR = Path("faiss_index")


class RAGRetriever:
    """Ingest documents and perform top-k semantic retrieval."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._embeddings = OpenAIEmbeddings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._store: FAISS | None = None

    def ingest(self, file_path: str) -> int:
        """Load a PDF or .txt file and index its chunks. Returns chunk count."""
        loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
        docs = loader.load()
        chunks = self._splitter.split_documents(docs)
        if self._store is None:
            self._store = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._store.add_documents(chunks)
        self._store.save_local(str(INDEX_DIR))
        return len(chunks)

    def load_index(self) -> None:
        """Load a previously saved FAISS index from disk."""
        if INDEX_DIR.exists():
            self._store = FAISS.load_local(str(INDEX_DIR), self._embeddings, allow_dangerous_deserialization=True)

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """Return top-k relevant text chunks for a query."""
        if self._store is None:
            return []
        results = self._store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
