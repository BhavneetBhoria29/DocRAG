"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    LLM_MODEL = "openai:gpt-4o"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
        # Retrieval / reranking
    RETRIEVER_K = 10          # retrieve wide before reranking
    RERANK_TOP_N = 6          # keep this many after reranking
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        return init_chat_model(cls.LLM_MODEL)