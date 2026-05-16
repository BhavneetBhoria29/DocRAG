# RAGent — Agentic RAG with LangGraph

RAGent is an agentic Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, FAISS, and OpenAI. It uses a **ReAct agent** to answer questions by reasoning across your own documents and Wikipedia, choosing the right tool for each query.

A Streamlit web UI is included for interactive use.

---

## Features

- **Multi-source ingestion** — load documents from URLs, PDFs, or plain text files
- **ReAct agent** — the LLM decides when to search your docs vs. query Wikipedia
- **FAISS vector store** — fast local semantic search with OpenAI embeddings
- **LangGraph workflow** — structured retrieve → reason → answer pipeline
- **Streamlit UI** — chat-style interface with source document preview and search history
- **CLI mode** — run example questions or an interactive session from the terminal

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
git clone https://github.com/<your-username>/ragent.git
cd ragent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv sync
```

### 2. Set environment variables

Create a `.env` file:

```env
OPENAI_API_KEY="sk-proj-..."
USER_AGENT="RAGent/1.0"
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

This runs a set of example questions, then optionally enters an interactive session.

---

## Loading your own documents

By default RAGent loads two blog posts from [Lilian Weng's site](https://lilianweng.github.io). To use your own sources, edit `src/config/config.py`:

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

## Tech stack

| Library | Purpose |
|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | Document loading, embeddings, tool abstractions |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful agent graph / ReAct loop |
| [OpenAI](https://platform.openai.com/) | GPT-4o (LLM) + text-embedding-ada (embeddings) |
| [FAISS](https://github.com/facebookresearch/faiss) | Local vector similarity search |
| [Streamlit](https://streamlit.io/) | Web UI |
| [Wikipedia](https://pypi.org/project/wikipedia/) | General knowledge fallback tool |

---

## Project structure

```
ragent/
├── src/
│   ├── config/           # Config and environment
│   ├── document_ingestion/  # Document loading and chunking
│   ├── vectorstore/      # FAISS vector store
│   ├── graph_builder/    # LangGraph workflow
│   ├── node/             # Graph node implementations
│   └── state/            # Shared graph state (Pydantic)
├── data/
│   └── url.txt           # Optional list of URLs to load
├── streamlit_app.py      # Web UI
├── main.py               # CLI entry point
├── pyproject.toml
└── requirements.txt
```

---

## Requirements

- Python 3.13+
- OpenAI API key
