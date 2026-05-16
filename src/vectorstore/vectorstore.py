"""Vector store module for document embedding and retrieval"""
from typing import List
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")

class VectorStore:
    """Manages vector store operations with persistent ChromaDB backend"""

    def __init__(self):
        """Initialize vector store with OpenAI embeddings"""
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create or load persistent vector store from documents.
        If a persisted store exists, loads it. Otherwise creates a new one.

        Args:
            documents: List of documents to embed
        """
        persist_path = Path(CHROMA_PERSIST_DIR)

        if persist_path.exists() and any(persist_path.iterdir()):
            print(f"📂 Loading existing vector store from {CHROMA_PERSIST_DIR}")
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embedding,
            )
        else:
            print(f"🔨 Creating new vector store at {CHROMA_PERSIST_DIR}")
            persist_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=CHROMA_PERSIST_DIR,
            )

        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """
        Get the retriever instance

        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)

    def clear(self):
        """Delete the persisted vector store to force a rebuild on next run."""
        import shutil
        persist_path = Path(CHROMA_PERSIST_DIR)
        if persist_path.exists():
            shutil.rmtree(persist_path)
            print(f"🗑️  Cleared vector store at {CHROMA_PERSIST_DIR}")
        self.vectorstore = None
        self.retriever = None