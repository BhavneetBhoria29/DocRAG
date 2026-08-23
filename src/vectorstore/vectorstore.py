"""Vector store module with hybrid search (BM25 + semantic)"""
from typing import List
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.config.config import Config

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

        k = Config.RETRIEVER_K

        # Semantic retriever (ChromaDB) — retrieve wide
        semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # BM25 keyword retriever — retrieve wide
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # Hybrid: 50% BM25 + 50% semantic, fused over the wide candidate set
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5],
        )

        if Config.USE_RERANKER:
            print(f"🎯 Reranker: {Config.RERANKER_MODEL}")
            cross_encoder = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL)
            compressor = CrossEncoderReranker(model=cross_encoder, top_n=Config.RERANK_TOP_N)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
            print(f"🔀 Hybrid + rerank ready (retrieve {k} → rerank to {Config.RERANK_TOP_N})")
        else:
            self.retriever = base_retriever
            print(f"🔀 Hybrid retriever ready (BM25 + semantic, k={k}, no rerank)")

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