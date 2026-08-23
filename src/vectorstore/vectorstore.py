"""Vector store module with hybrid search (BM25 + semantic)"""
from typing import List
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")


class VectorStore:
    """Manages vector store operations with hybrid search (BM25 + ChromaDB)"""

    def __init__(self):
        """Initialize vector store with OpenAI embeddings"""
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.retriever = None
        self._documents: List[Document] = []

    def create_vectorstore(self, documents: List[Document]):
        """
        Create or load persistent vector store from documents.
        Builds a hybrid retriever combining BM25 + semantic search.

        Args:
            documents: List of documents to embed
        """
        self._documents = documents
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

        # Semantic retriever (ChromaDB)
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # BM25 keyword retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Hybrid: 50% BM25 + 50% semantic
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5],
        )

        print("🔀 Hybrid retriever ready (BM25 + semantic)")

    def get_retriever(self):
        """
        Get the hybrid retriever instance

        Returns:
            EnsembleRetriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search

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
        self._documents = []