"""
redteam/metrics.py
------------------
Attack Success Rate (ASR) and False Positive Rate (FPR) with bootstrap 95%
confidence intervals.

Bootstrap CIs are used for the same reason as in the DocRAG RAGAS harness: the
sample sizes are small, a bare point estimate overstates precision, and an
interviewer will ask "how many cases, how confident". Reporting 0.25 [0.06, 0.50]
is honest; reporting 0.25 alone is not.

Definitions:
  ASR = successful_attacks / total_attacks
        (per family and overall)
  FPR = benign_queries_wrongly_flagged / total_benign_queries
        (the cost of the defense)

Both are proportions over independent binary trials, so a percentile bootstrap
over the 0/1 outcome vector gives a defensible interval without distributional
assumptions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Proportion:
    label: str
    successes: int
    n: int

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def bootstrap_ci(self, iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
        if self.n == 0:
            return (0.0, 0.0)
        outcomes = [1] * self.successes + [0] * (self.n - self.successes)
        rng = random.Random(seed)
        means = []
        for _ in range(iters):
            sample = [outcomes[rng.randrange(self.n)] for _ in range(self.n)]
            means.append(sum(sample) / self.n)
        means.sort()
        lo = means[int(0.025 * iters)]
        hi = means[int(0.975 * iters) - 1]
        return (round(lo, 3), round(hi, 3))

    def report(self) -> str:
        lo, hi = self.bootstrap_ci()
        return f"{self.label}: {self.rate:.2f} [{lo:.2f}, {hi:.2f}]  (n={self.n}, hits={self.successes})"


def asr(results_by_family: dict[str, list[bool]]) -> dict[str, Proportion]:
    """results_by_family: family name -> list of per-attack success booleans."""
    out: dict[str, Proportion] = {}
    all_success, all_n = 0, 0
    for fam, outcomes in results_by_family.items():
        s = sum(1 for x in outcomes if x)
        out[fam] = Proportion(f"ASR/{fam}", s, len(outcomes))
        all_success += s
        all_n += len(outcomes)
    out["overall"] = Proportion("ASR/overall", all_success, all_n)
    return out


def fpr(flagged: list[bool]) -> Proportion:
    """flagged: per-benign-query booleans (True = wrongly flagged by guardrail)."""
    return Proportion("FPR", sum(1 for x in flagged if x), len(flagged))
