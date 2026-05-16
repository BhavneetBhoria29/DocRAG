"""
RAGAS evaluation pipeline for DocRAG.
Run: python eval/evaluate.py
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# ── Test dataset ─────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "question": "What is the role of an agent in LLM-based systems?",
        "ground_truth": "An agent uses an LLM as its reasoning engine to decide which actions to take and in what order, allowing it to complete complex tasks autonomously.",
        "contexts": [
            "In LLM-powered autonomous agents, the LLM functions as the agent's brain, complemented by several key components including planning, memory, and tool use.",
            "Agents can break down complex tasks into smaller subgoals and execute them using available tools."
        ],
        "answer": "In LLM-based systems, an agent uses the language model as a reasoning core to decide which tools to call and in what sequence, enabling it to solve multi-step tasks autonomously."
    },
    {
        "question": "What is chain-of-thought prompting?",
        "ground_truth": "Chain-of-thought prompting encourages the model to produce intermediate reasoning steps before arriving at a final answer, improving performance on complex tasks.",
        "contexts": [
            "Chain-of-thought (CoT) prompting generates a sequence of short sentences to describe reasoning logics step by step, leading to the final answer.",
            "CoT prompting has been shown to significantly improve performance on arithmetic, commonsense, and symbolic reasoning tasks."
        ],
        "answer": "Chain-of-thought prompting is a technique where the model is guided to reason step by step before producing a final answer, which improves accuracy on complex reasoning tasks."
    },
    {
        "question": "What are the main components of a ReAct agent?",
        "ground_truth": "A ReAct agent interleaves reasoning traces and task-specific actions, using tools to gather information and updating its reasoning based on observations.",
        "contexts": [
            "ReAct combines reasoning and acting in an interleaved manner. The agent generates a thought, takes an action using a tool, and observes the result before continuing.",
            "The ReAct framework allows the model to dynamically create, maintain, and adjust plans while interacting with external environments."
        ],
        "answer": "A ReAct agent consists of a reasoning step (thought), an action step (tool call), and an observation step (tool result), cycling through these until the task is complete."
    },
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "RAG combines a retrieval system with a generative model, fetching relevant documents from a corpus and conditioning the LLM response on those documents.",
        "contexts": [
            "Retrieval-Augmented Generation (RAG) retrieves relevant documents from an external knowledge base and passes them as context to the language model to generate a grounded response.",
            "RAG reduces hallucinations by grounding the model's output in retrieved evidence rather than relying solely on parametric knowledge."
        ],
        "answer": "RAG is a technique that retrieves relevant passages from a document corpus and provides them as context to a language model, producing answers grounded in evidence rather than memorised knowledge."
    },
    {
        "question": "How does FAISS perform vector similarity search?",
        "ground_truth": "FAISS uses approximate nearest neighbour algorithms to efficiently search high-dimensional embedding vectors, finding the most semantically similar documents to a query.",
        "contexts": [
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
            "It supports exact and approximate nearest neighbour search, with various index types optimised for different memory and speed trade-offs."
        ],
        "answer": "FAISS performs vector similarity search by indexing embeddings and using approximate nearest neighbour algorithms to quickly find the most similar vectors to a query embedding."
    },
]


def build_ragas_dataset(test_cases: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "question":     [t["question"]     for t in test_cases],
        "answer":       [t["answer"]       for t in test_cases],
        "contexts":     [t["contexts"]     for t in test_cases],
        "ground_truth": [t["ground_truth"] for t in test_cases],
    })


def safe_score(val) -> float:
    """Handle both scalar and list results from RAGAS."""
    if isinstance(val, list):
        cleaned = [v for v in val if v is not None]
        return round(float(np.mean(cleaned)), 4) if cleaned else 0.0
    return round(float(val), 4)


def run_evaluation() -> dict:
    print("🔍 Running RAGAS evaluation pipeline...\n")

    dataset = build_ragas_dataset(TEST_CASES)

    llm        = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ],
    )

    scores = {
        "faithfulness":      safe_score(results["faithfulness"]),
        "answer_relevancy":  safe_score(results["answer_relevancy"]),
        "context_precision": safe_score(results["context_precision"]),
        "context_recall":    safe_score(results["context_recall"]),
    }

    return scores


def save_results(scores: dict) -> Path:
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"ragas_{timestamp}.json"

    payload = {
        "timestamp":   timestamp,
        "num_samples": len(TEST_CASES),
        "scores":      scores,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_results(scores: dict) -> None:
    print("=" * 45)
    print("         RAGAS Evaluation Results")
    print("=" * 45)
    print(f"  Faithfulness       {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy   {scores['answer_relevancy']:.4f}")
    print(f"  Context Precision  {scores['context_precision']:.4f}")
    print(f"  Context Recall     {scores['context_recall']:.4f}")
    print("=" * 45)
    avg = sum(scores.values()) / len(scores)
    print(f"  Average Score      {avg:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    scores   = run_evaluation()
    out_path = save_results(scores)
    print_results(scores)
    print(f"\n✅ Results saved to {out_path}")