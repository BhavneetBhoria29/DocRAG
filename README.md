# DocRAG — Agentic RAG with LangGraph

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-purple.svg)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/vector--store-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

DocRAG is an agentic Retrieval-Augmented Generation system built with LangChain, LangGraph, FAISS, and OpenAI. A **ReAct agent** reasons over your documents and Wikipedia, dynamically choosing the right tool for each query — falling back to live retrieval when documents don't cover the question rather than hallucinating an answer.

---

## Why agentic RAG?

Standard RAG always retrieves from a fixed corpus — if the answer isn't there, it guesses. DocRAG uses a ReAct agent to decide *which* tool to use per query:

- **Document retriever** → for questions within your uploaded corpus  
- **Wikipedia tool** → for general knowledge or out-of-corpus context  

This means fewer hallucinations on hybrid queries ("summarise this paper and explain the broader field context").

---

## Features

- **Multi-source ingestion** — URLs, PDFs, and plain text files
- **ReAct agent (GPT-4o)** — tool routing between FAISS retriever and Wikipedia
- **FAISS vector store** — fast local semantic search with OpenAI embeddings
- **LangGraph workflow** — stateful retrieve → reason → answer graph with Pydantic state
- **Streamlit UI** — chat interface with source document preview and search history
- **CLI mode** — example questions or interactive session from the terminal
- **UV-managed environment** — reproducible installs with `uv sync`

---

## Architecture

```
User question
     │
     ▼
┌─────────────┐      ┌──────────────────────────────────┐
│  Retriever  │─────▶│        ReAct Agent (GPT-4o)       │
│  (FAISS)    │      │  ┌────────────┐ ┌─────────────┐  │
└─────────────┘      │  │ retriever  │ │  wikipedia  │  │
                     │  │   tool     │ │    tool     │  │
                     │  └────────────┘ └─────────────┘  │
                     └──────────────────┬───────────────┘
                                        │
                                        ▼
                                     Answer
```

**Key modules:**

| Path | Responsibility |
|---|---|
| `src/config/config.py` | LLM model, chunk size, default URLs |
| `src/document_ingestion/document_processor.py` | Load URLs / PDFs / TXT and split into chunks |
| `src/vectorstore/vectorstore.py` | FAISS vector store and retriever |
| `src/graph_builder/graph_builder.py` | LangGraph workflow (retrieve → answer) |
| `src/node/reactnode.py` | ReAct agent node with retriever + Wikipedia tools |
| `src/state/rag_state.py` | Pydantic state shared across graph nodes |
| `streamlit_app.py` | Web UI |
| `main.py` | CLI entry point |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/BhavneetBhoria29/DocRAG.git
cd DocRAG
```

With `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file:

```env
OPENAI_API_KEY="sk-proj-..."
USER_AGENT="DocRAG/1.0"
```

### 3. Run the web UI

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

### 4. Run the CLI

```bash
python main.py
```

Runs a set of example questions, then optionally enters an interactive session.

---

## Loading your own documents

By default DocRAG loads two blog posts from [Lilian Weng's site](https://lilianweng.github.io). To use your own sources, edit `src/config/config.py`:

```python
DEFAULT_URLS = [
    "https://your-site.com/page",
]
```

Or create `data/urls.txt` with one URL per line — `main.py` reads it automatically.

You can also load PDFs and text files via `DocumentProcessor`:

```python
from src.document_ingestion.document_processor import DocumentProcessor

dp = DocumentProcessor()
docs = dp.load_documents([
    "path/to/file.pdf",
    "path/to/notes.txt",
    "https://example.com/article",
])
```

---

## Project structure

```
DocRAG/
├── src/
│   ├── config/              # Config and environment
│   ├── document_ingestion/  # Document loading and chunking
│   ├── vectorstore/         # FAISS vector store
│   ├── graph_builder/       # LangGraph workflow
│   ├── node/                # Graph node implementations
│   └── state/               # Shared graph state (Pydantic)
├── data/
│   └── urls.txt             # Optional list of URLs to load
├── .github/workflows/       # CI pipeline
├── streamlit_app.py         # Web UI
├── main.py                  # CLI entry point
├── pyproject.toml
└── requirements.txt
```

---

## Tech stack

| Library | Purpose |
|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | Document loading, embeddings, tool abstractions |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful agent graph / ReAct loop |
| [OpenAI](https://platform.openai.com/) | GPT-4o (LLM) + text-embedding-ada-002 (embeddings) |
| [FAISS](https://github.com/facebookresearch/faiss) | Local vector similarity search |
| [Streamlit](https://streamlit.io/) | Web UI |
| [Wikipedia](https://pypi.org/project/wikipedia/) | General knowledge fallback tool |
| [uv](https://github.com/astral-sh/uv) | Fast Python package and environment manager |

---

## Requirements

- Python 3.13+
- OpenAI API key

---

## Roadmap

- [ ] RAGAS evaluation pipeline (faithfulness, context precision, answer relevance)
- [ ] Swap FAISS for persistent vector store (ChromaDB / pgvector)
- [ ] Add hybrid search (BM25 + semantic)
- [ ] Support additional file types (DOCX, CSV, HTML)
