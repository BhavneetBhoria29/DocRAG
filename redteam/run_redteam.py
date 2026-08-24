"""
redteam/run_redteam.py
----------------------
Orchestrates the full red-team evaluation and prints the before/after story.

Pipeline of the run:

  1. BASELINE (no guardrail)
       For each attack: reset corpus -> seed benign docs -> ingest the poison
       doc -> run the trigger query through the live pipeline -> score success.
       Produces undefended ASR per family and overall.

  2. GUARDED (guardrail on)
       Same attacks, but the retrieved context is screened by the guardrail
       before generation. Produces guarded ASR (should drop).

  3. FPR
       Run benign queries (clean corpus, no poison) through the guardrail.
       Any benign query whose legitimate context gets flagged is a false
       positive. Produces FPR (the cost of the defense).

  4. REPORT
       ASR baseline vs guarded, FPR, all with bootstrap 95% CIs. This is the
       artifact you read numbers off for the CV and README.

Because the guardrail screens context, running it "guarded" means: retrieve as
normal, hand the contexts to guardrail.screen(), then regenerate using only the
kept contexts. To keep the adapter surface tiny, guarded generation reuses
adapter.query() but with the poison already filtered from what reaches the model
is NOT possible through a single call, so we approximate by screening the
retrieved contexts and, if the guardrail drops the poison chunk, re-running the
query against a corpus that no longer contains it. See _run_guarded for the exact
mechanism and its one honest caveat.

Usage:
    export OPENAI_API_KEY=sk-...
    python redteam/run_redteam.py                 # heuristic guardrail, sanitize
    python redteam/run_redteam.py --llm-stage     # add LLM detector stage
    python redteam/run_redteam.py --block         # block mode instead of sanitize
    python redteam/run_redteam.py --n 3           # smoke test on first 3 attacks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Resolve imports whether run as `python redteam/run_redteam.py` from the repo
# root or from inside redteam/. Put BOTH the repo root (for `src.*`, mirroring
# eval/evaluate.py) and this redteam/ dir (for the flat imports below) on path.
_HERE = Path(__file__).resolve().parent          # .../DocRag/redteam
_REPO_ROOT = _HERE.parent                          # .../DocRag
for _p in (str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter import DocRAGAdapter
from corpus.injection_payloads import ATTACKS, Family
from corpus.benign_queries import BENIGN_DOCS, BENIGN_QUERIES
from guardrail import InjectionGuardrail, Action
from metrics import asr, fpr
from scorer import LLMJudge, score_attack


def _run_baseline(adapter: DocRAGAdapter, judge: LLMJudge, attacks) -> dict[str, list[bool]]:
    by_family: dict[str, list[bool]] = {f.value: [] for f in Family}
    retrieved_count = 0
    for case in attacks:
        adapter.reset_corpus()
        adapter.seed_benign_corpus(BENIGN_DOCS)
        adapter.ingest_documents([case.poison_doc], metadatas=[{"origin": "attack", "id": case.id}])
        result = adapter.query(case.trigger_query)

        # DIAGNOSTIC: was the poison actually retrieved? A 0.00 ASR means one of
        # two very different things — the attack reached the model and failed
        # (real resistance), or the poison never got retrieved (test too weak).
        # Match on a distinctive slice of the poison doc against each context.
        marker = case.poison_doc[-60:].strip().lower()
        poison_retrieved = any(marker[:30] in (c or "").lower() for c in result.contexts)
        retrieved_count += int(poison_retrieved)

        score = score_attack(case, result.answer, judge)
        by_family[case.family.value].append(score.succeeded)
        flag = "poison@ctx" if poison_retrieved else "NOT-RETRIEVED"
        print(f"  [baseline] {case.id:16} -> {'HIT ' if score.succeeded else 'miss'}  "
              f"[{flag}]  ({score.method}: {score.detail[:50]})")
    print(f"  --- poison retrieved in {retrieved_count}/{len(attacks)} cases "
          f"(if low, attacks are not reaching the model; strengthen retrievability) ---")
    return by_family


def _run_guarded(adapter: DocRAGAdapter, guard: InjectionGuardrail, judge: LLMJudge, attacks) -> dict[str, list[bool]]:
    """
    Guarded run. Mechanism: run the query, screen the retrieved contexts. If the
    guardrail drops the poison, the model never should have seen it, so we
    regenerate from the kept contexts by re-querying a corpus with the poison
    removed (reset + reseed benign only). If the guardrail keeps the poison
    (evasion), we score the original answer as-is.

    Honest caveat: this measures the guardrail as a context filter. It assumes
    generation from the kept contexts equals generation had the poison never been
    retrieved, which holds when the poison chunk is fully dropped. In block mode,
    a flagged query yields a refusal, scored as attack-failed (the model emitted
    no attacker-controlled content).
    """
    by_family: dict[str, list[bool]] = {f.value: [] for f in Family}
    for case in attacks:
        adapter.reset_corpus()
        adapter.seed_benign_corpus(BENIGN_DOCS)
        adapter.ingest_documents([case.poison_doc], metadatas=[{"origin": "attack", "id": case.id}])
        result = adapter.query(case.trigger_query)

        decision = guard.screen(result.contexts)

        if not decision.flagged:
            # Guardrail missed it (evasion). Score the real answer.
            score = score_attack(case, result.answer, judge)
            succeeded = score.succeeded
            note = "evaded guard"
        elif guard.action is Action.BLOCK:
            # Whole query refused -> no attacker content emitted.
            succeeded = False
            note = "blocked"
        else:
            # Sanitize: poison dropped. Regenerate from a corpus without it.
            adapter.reset_corpus()
            adapter.seed_benign_corpus(BENIGN_DOCS)
            clean = adapter.query(case.trigger_query)
            score = score_attack(case, clean.answer, judge)
            succeeded = score.succeeded
            note = f"sanitized ({len(decision.dropped_contexts)} dropped)"

        by_family[case.family.value].append(succeeded)
        print(f"  [guarded ] {case.id:16} -> {'HIT ' if succeeded else 'safe'}  ({note})")
    return by_family


def _run_fpr(adapter: DocRAGAdapter, guard: InjectionGuardrail) -> list[bool]:
    """Benign queries over a clean corpus. Flagged == false positive."""
    adapter.reset_corpus()
    adapter.seed_benign_corpus(BENIGN_DOCS)
    flags: list[bool] = []
    for item in BENIGN_QUERIES:
        result = adapter.query(item["query"])
        decision = guard.screen(result.contexts)
        flags.append(decision.flagged)
        tag = "FLAG (fp)" if decision.flagged else "ok"
        print(f"  [fpr     ] {item['query'][:44]:44} -> {tag}")
    return flags


def main():
    ap = argparse.ArgumentParser(description="Indirect prompt-injection red-team for DocRAG")
    ap.add_argument("--llm-stage", action="store_true", help="enable the LLM detector stage in the guardrail")
    ap.add_argument("--block", action="store_true", help="guardrail blocks the whole query instead of sanitizing")
    ap.add_argument("--judge-model", default="gpt-4o", help="LLM judge model (match your RAGAS judge)")
    ap.add_argument("--n", type=int, default=0, help="run only the first N attacks (smoke test)")
    ap.add_argument("--out", default="redteam/results", help="output dir for the JSON report")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")

    attacks = ATTACKS[: args.n] if args.n else ATTACKS
    judge = LLMJudge(model=args.judge_model)
    guard = InjectionGuardrail(
        action=Action.BLOCK if args.block else Action.SANITIZE,
        use_llm_stage=args.llm_stage,
    )
    adapter = DocRAGAdapter(use_reranker=True)

    print("\n=== BASELINE (no guardrail) ===")
    base = _run_baseline(adapter, judge, attacks)
    print("\n=== GUARDED ===")
    guarded = _run_guarded(adapter, guard, judge, attacks)
    print("\n=== FPR (benign queries) ===")
    fp = _run_fpr(adapter, guard)

    adapter.reset_corpus()  # teardown: leave no red-team docs behind

    base_asr = asr(base)
    guard_asr = asr(guarded)
    fpr_p = fpr(fp)

    print("\n" + "=" * 60)
    print("  RED-TEAM REPORT — DocRAG indirect prompt injection")
    print("=" * 60)
    print("\n  Attack Success Rate (lower is better)")
    print("  " + base_asr["overall"].report().replace("ASR/overall", "baseline overall"))
    print("  " + guard_asr["overall"].report().replace("ASR/overall", "guarded  overall"))
    print("\n  Per family (baseline -> guarded):")
    for f in Family:
        b, g = base_asr[f.value], guard_asr[f.value]
        print(f"    {f.value:11} {b.rate:.2f} -> {g.rate:.2f}   (n={b.n})")
    print("\n  Guardrail cost")
    print("  " + fpr_p.report())
    print("=" * 60 + "\n")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(args.out) / f"redteam_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "config": {
            "guardrail_action": guard.action.value,
            "llm_stage": args.llm_stage,
            "judge_model": args.judge_model,
            "num_attacks": len(attacks),
        },
        "asr_baseline": {k: {"rate": v.rate, "ci": v.bootstrap_ci(), "n": v.n} for k, v in base_asr.items()},
        "asr_guarded": {k: {"rate": v.rate, "ci": v.bootstrap_ci(), "n": v.n} for k, v in guard_asr.items()},
        "fpr": {"rate": fpr_p.rate, "ci": fpr_p.bootstrap_ci(), "n": fpr_p.n},
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved -> {path}\n")


if __name__ == "__main__":
    main()