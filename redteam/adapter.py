"""
redteam/adapter.py  — WIRED for DocRAG
--------------------------------------
Wires the red-team harness to DocRAG's real pipeline (VectorStore + GraphBuilder),
using the same generation path eval/evaluate.py uses: graph_builder.run(q)["answer"].

ISOLATION — why this does not touch your real index
----------------------------------------------------
DocRAG persists to a single hardcoded directory (src/vectorstore/vectorstore.py:
CHROMA_PERSIST_DIR = .../data/chroma_db) and AgenticRAG only ingests from URLs.
Indirect-injection testing needs to plant raw poison documents in the corpus, so
this adapter does NOT use AgenticRAG. Instead it:

  1. Redirects CHROMA_PERSIST_DIR to a TEMP directory for the duration of the run
     (monkeypatch), so data/chroma_db is never read or written.
  2. Builds VectorStore directly from raw Document objects (benign + poison).
  3. Builds GraphBuilder on that retriever and calls .run(q) — the same call eval
     uses — to get the real generated answer.
  4. reset_corpus() starts a clean corpus; teardown() removes the temp tree.

FRESH-DIR-PER-BUILD
-------------------
Chroma's persistent client locks its directory. Deleting and recreating the SAME
path while a prior client still holds a handle causes a stale readonly handle and
a "readonly database" write error on the next build. So every rebuild gets its
OWN unique subdirectory (build_0, build_1, ...). Old dirs are left in the temp
tree and cleaned wholesale at teardown, never deleted out from under a live
client. This is a red-team harness, not a service; disk churn in a temp dir is
fine and correctness matters more.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

# Import DocRAG internals
import src.vectorstore.vectorstore as vs_module
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.config.config import Config


@dataclass
class RetrievalResult:
    answer: str
    contexts: list[str] = field(default_factory=list)


class DocRAGAdapter:
    """Red-team wrapper over the real DocRAG pipeline, on an isolated temp store."""

    def __init__(self, use_reranker: bool = True):
        Config.USE_RERANKER = use_reranker
        self._use_reranker = use_reranker

        # Isolated temp root; DocRAG's persist constant is redirected under it.
        self._tmp_root = Path(tempfile.mkdtemp(prefix="docrag_redteam_"))
        self._orig_persist_dir = vs_module.CHROMA_PERSIST_DIR
        self._build_counter = 0

        self.llm = Config.get_llm()
        self._corpus: list[Document] = []
        self._vector_store: Optional[VectorStore] = None
        self._graph_builder: Optional[GraphBuilder] = None

    # ------------------------------------------------------------------ #
    #  Ingest attacker/benign documents as raw text                       #
    # ------------------------------------------------------------------ #
    def ingest_documents(self, docs: list[str], metadatas: Optional[list[dict]] = None) -> None:
        metadatas = metadatas or [{} for _ in docs]
        for text, meta in zip(docs, metadatas):
            self._corpus.append(Document(page_content=text, metadata=meta))
        self._rebuild()

    def seed_benign_corpus(self, docs: list[str]) -> None:
        self.ingest_documents(docs, metadatas=[{"origin": "benign"}] * len(docs))

    # ------------------------------------------------------------------ #
    #  Rebuild the retriever + graph over the current corpus              #
    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        # Drop references to any prior store/graph so the old Chroma client can
        # be garbage-collected and release its directory lock. We do NOT delete
        # the old directory here (that is what caused the readonly error); each
        # build simply uses a new path.
        self._vector_store = None
        self._graph_builder = None
        gc.collect()

        # Unique persist dir for THIS build.
        build_dir = self._tmp_root / f"build_{self._build_counter}" / "chroma_db"
        self._build_counter += 1
        vs_module.CHROMA_PERSIST_DIR = str(build_dir)

        self._vector_store = VectorStore()
        self._vector_store.create_vectorstore(self._corpus)
        self._graph_builder = GraphBuilder(
            retriever=self._vector_store.get_retriever(),
            llm=self.llm,
        )
        self._graph_builder.build()

    # ------------------------------------------------------------------ #
    #  Run one query through the full live pipeline                       #
    # ------------------------------------------------------------------ #
    def query(self, user_input: str) -> RetrievalResult:
        if self._graph_builder is None:
            raise RuntimeError("Corpus empty. Call seed_benign_corpus/ingest first.")
        docs = self._vector_store.get_retriever().invoke(user_input)
        contexts = [d.page_content for d in docs]
        answer = self._graph_builder.run(user_input)["answer"]
        return RetrievalResult(answer=answer, contexts=contexts)

    # ------------------------------------------------------------------ #
    #  Corpus lifecycle                                                   #
    # ------------------------------------------------------------------ #
    def reset_corpus(self) -> None:
        # Start a clean corpus. Do not delete old build dirs mid-run; teardown
        # clears them all at once.
        self._corpus = []
        self._vector_store = None
        self._graph_builder = None
        gc.collect()

    def teardown(self) -> None:
        """Restore the original persist dir and delete the whole temp tree."""
        vs_module.CHROMA_PERSIST_DIR = self._orig_persist_dir
        self._vector_store = None
        self._graph_builder = None
        gc.collect()
        shutil.rmtree(self._tmp_root, ignore_errors=True)

    def __del__(self):
        try:
            self.teardown()
        except Exception:
            pass