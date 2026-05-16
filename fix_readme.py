content = open('README.md').read()

replacements = [
    # Badge
    (
        '[![FAISS](https://img.shields.io/badge/vector--store-FAISS-green.svg)](https://github.com/facebookresearch/faiss)',
        '[![ChromaDB](https://img.shields.io/badge/vector--store-ChromaDB-green.svg)](https://github.com/chroma-core/chroma)'
    ),
    # Intro paragraph
    (
        'built with LangChain, LangGraph, FAISS, and OpenAI',
        'built with LangChain, LangGraph, ChromaDB, and OpenAI'
    ),
    # Features - ReAct line
    (
        '- **ReAct agent (GPT-4o)** — tool routing between FAISS retriever and Wikipedia',
        '- **ReAct agent (GPT-4o)** — tool routing between hybrid retriever and Wikipedia'
    ),
    # Features - vector store line
    (
        '- **FAISS vector store** — fast local semantic search with OpenAI embeddings',
        '- **Hybrid search** — BM25 + semantic search via ChromaDB with OpenAI embeddings'
    ),
    # Architecture diagram
    (
        '│  (FAISS)    │',
        '│  (Chroma)   │'
    ),
    # Modules table
    (
        '`src/vectorstore/vectorstore.py` | FAISS vector store and retriever',
        '`src/vectorstore/vectorstore.py` | ChromaDB persistent vector store + hybrid retriever'
    ),
    # Project structure
    (
        '├── vectorstore/         # FAISS vector store',
        '├── vectorstore/         # ChromaDB persistent vector store'
    ),
    # Loading docs example - update to show new file types
    (
        '    "path/to/notes.txt",\n    "https://example.com/article",\n])',
        '    "path/to/notes.txt",\n    "path/to/report.docx",\n    "path/to/data.csv",\n    "https://example.com/article",\n])'
    ),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"✅ Replaced: {old[:60]}...")
    else:
        print(f"⚠️  Not found: {old[:60]}...")

open('README.md', 'w').write(content)
print("\nDone.")
