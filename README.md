# DocRAG — Agentic RAG with LangGraph
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://docrag-wyswffymwpet3f4hxqm7ex.streamlit.app)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-purple.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/vector--store-ChromaDB-green.svg)](https://github.com/chroma-core/chroma)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Python CI](https://github.com/BhavneetBhoria29/DocRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/BhavneetBhoria29/DocRAG/actions/workflows/ci.yml)

DocRAG is an agentic Retrieval-Augmented Generation system built with LangChain, LangGraph, ChromaDB, and OpenAI. A **ReAct agent** reasons over your documents and Wikipedia, dynamically choosing the right tool for each query — falling back to live retrieval when documents don't cover the question rather than hallucinating an answer.

---

## Why agentic RAG?

Standard RAG always retrieves from a fixed corpus — if the answer isn't there, it guesses. DocRAG uses a ReAct agent to decide *which* tool to use per query:

- **Document retriever** → for questions within your uploaded corpus  
- **Wikipedia tool** → for general knowledge or out-of-corpus context  

This means fewer hallucinations on hybrid queries ("summarise this paper and explain the broader field context").

---

## Features

- **Multi-source ingestion** — URLs, PDFs, plain text, DOCX, CSV, and HTML files
- **ReAct agent (GPT-4o)** — tool routing between hybrid retriever and Wikipedia
- **Hybrid search** — BM25 + semantic search via ChromaDB with text-embedding-3-small embeddings
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
│  (Chroma)   │      │  ┌────────────┐ ┌─────────────┐  │
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
| `src/vectorstore/vectorstore.py` | ChromaDB persistent vector store + hybrid retriever |
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
    "path/to/report.docx",
    "path/to/data.csv",
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
│   ├── vectorstore/         # ChromaDB persistent vector store
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
| [OpenAI](https://platform.openai.com/) | GPT-4o (LLM) + text-embedding-3-small (embeddings) |
| [ChromaDB](https://github.com/chroma-core/chroma) | Persistent local vector store (text-embedding-3-small) |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | BM25 keyword retriever for hybrid search |
| [Streamlit](https://streamlit.io/) | Web UI |
| [Wikipedia](https://pypi.org/project/wikipedia/) | General knowledge fallback tool |
| [uv](https://github.com/astral-sh/uv) | Fast Python package and environment manager |

---

## Requirements

- Python 3.13+
- OpenAI API key

---

## Roadmap

- [x] RAGAS evaluation pipeline (faithfulness, context precision, answer relevance)
- [x] Swap FAISS for persistent vector store (ChromaDB)
- [x] Add hybrid search (BM25 + semantic)
- [x] Support additional file types (DOCX, CSV, HTML)

---
## Evaluation

Eval-first: the RAGAS harness runs the live pipeline (real retrieval and generation, not hand-written contexts) and reports a 95% bootstrap confidence interval per metric. Judge GPT-4o (temperature 0), embeddings text-embedding-3-small, 17-case golden set.

Current pipeline (hybrid retrieval + cross-encoder reranking):

| Metric            | Mean | 95% CI       |
| ----------------- | ---- | ------------ |
| Context Precision | 0.78 | [0.63, 0.90] |
| Context Recall    | 0.85 | [0.68, 1.00] |
| Faithfulness      | 0.67 | [0.55, 0.78] |
| Answer Relevancy  | 0.91 | [0.86, 0.95] |

**The reranker is the key retrieval improvement.** Adding it (retrieve 10, cross-encoder rerank to top 6) moved context precision from 0.49 to 0.78 by filtering low-relevance chunks, at a measured cost to recall (0.94 to 0.85) and faithfulness (0.86 to 0.67). This is a deliberate precision/recall tradeoff, tuned via top_n. Faithfulness recovery (chunk-size and grounding-prompt tuning) is the next milestone.

Reproducible via `python eval/evaluate.py`; per-run JSON and per-sample scores are saved under `eval/results/`. Retrieval quality is regression-tested, not asserted. 17 cases is directional; growing the golden set tightens the intervals.
