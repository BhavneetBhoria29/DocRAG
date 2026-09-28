# EU AI Act assessment: DocRAG

I wrote this to work out what the EU AI Act would actually ask of DocRAG if it left the portfolio and went into a real deployment, and which of those asks the existing eval and red-team work already answers. It is my reading of the regulation as an engineer, not legal advice. Dates reflect the AI Act as amended by the Digital Omnibus, which entered into force on 27 July 2026.

## 1. What the system is

- A LangGraph ReAct agent (GPT-4o) that answers questions over an uploaded corpus, with hybrid BM25 + ChromaDB retrieval, a cross-encoder reranker, and a Wikipedia tool for out-of-corpus questions.
- Users are humans typing into a Streamlit chat UI or the CLI. Output is generated text plus the retrieved source chunks.
- The model is a third-party GPAI model (OpenAI). DocRAG is a system built on top of it.

## 2. Role and classification

**Role.** If I deploy DocRAG under my own name, I am the **provider** of an AI system (Art. 3(3)). OpenAI is the GPAI model provider, and its obligations (Chapter V) are OpenAI's, not mine. A company that uses DocRAG internally is a **deployer** (Art. 3(4)).

**Prohibited practices (Art. 5).** None apply. It does no manipulation, social scoring, biometric categorisation or emotion recognition.

**High-risk (Art. 6, Annex III).** As a general document Q&A tool, DocRAG is **not** high-risk. Risk tier follows the intended purpose, not the technology, so the same code becomes high-risk the moment it is put into an Annex III use. The one closest to my own background is Annex III point 5(b): creditworthiness assessment or credit scoring of natural persons. If a bank used DocRAG to answer "does this applicant's file meet our lending policy?" and that answer fed a credit decision, it would be high-risk. Section 5 works through that case because it is the realistic one in fintech.

**Transparency (Art. 50).** This is the tier DocRAG sits in today, and these obligations have applied since **2 August 2026**:

| Obligation | Applies to DocRAG? | Status |
|---|---|---|
| 50(1) Tell people they are interacting with an AI system, unless obvious | Yes, it is a chat interface | **Gap.** The UI does not say it. One line in the Streamlit header fixes it. |
| 50(2) Mark synthetic text output as AI-generated in a machine-readable way | Yes, as provider of a system generating text | **Gap.** Plan: attach provenance metadata (model, timestamp, source chunk ids) to every API response and export. Systems already on the market before 2 Aug 2026 have until 2 Dec 2026. |
| 50(4) Disclose AI-generated text published to inform the public | Only if a deployer publishes answers as-is | Deployer's duty; the provenance metadata above is what lets them meet it |

**AI literacy (Art. 4).** Applies to providers and deployers. After the Omnibus it asks for measures to support staff literacy rather than guaranteeing it. For a deployment this means a short usage guide that covers failure modes (Section 4).

## 3. What already maps to the Act

The high-risk requirements are the useful checklist even where they are not legally required, because they describe what "production-ready" means for a regulated customer.

| Requirement | Evidence already in this repo |
|---|---|
| Art. 9 risk management: identify and estimate foreseeable risks | Written threat model in `redteam/README.md` (5 attack families), measured per family |
| Art. 15(1) accuracy declared and measured | RAGAS harness in `eval/evaluate.py`, 17-sample golden set, bootstrap 95% CIs, results versioned in `eval/results/` |
| Art. 15(4) robustness to errors and inconsistencies | Reranker change measured before shipping: context precision 0.49 to 0.78, with the tradeoff kept visible (see Section 4) |
| Art. 15(5) resilience against attacks exploiting vulnerabilities, including **data poisoning** and adversarial inputs | Canary-based indirect prompt-injection harness against the live pipeline, 8 payloads across 5 families, results kept in `redteam/results/` |
| Art. 12 record-keeping | Partial. Eval and red-team runs are logged with timestamp, config and judge model. Per-request logging in the app is missing. |
| Art. 14 human oversight | Partial. Source chunks are shown next to every answer so a user can check them. There is no abstain or escalate path. |
| Art. 13 transparency to deployers | Partial. README documents architecture and eval. No instructions-for-use section with limitations. |

## 4. Residual risks, stated honestly

These are the findings a deployer would need in their instructions for use.

1. **Content poisoning works.** Command-style injection ("ignore the user") failed in every run (0% ASR). Authority-framed false content blended into document voice did not: baseline ASR was 0.38 to 0.62 across seeds (n=8, so wide CIs), driven entirely by the fact-poisoning family. Under the AI Act this is exactly the data-poisoning risk Art. 15(5) names.
2. **The current guardrail is not enough.** The keyword-screening guardrail held a 0% false-positive rate on benign queries but left most poisoning through: guarded ASR 0.25 to 0.375, with fact-poisoning still at 0.5. It catches instruction-shaped text, not plausible false statements. The fix is a semantic detector or source-trust controls, not a longer keyword list.
3. **Faithfulness dropped after reranking.** On the same 17-sample golden set, the reranker lifted context precision from 0.49 to 0.78, but faithfulness fell from 0.858 to 0.665 [0.551, 0.778] and context recall from 0.94 to 0.85. Retrieval got more precise while more answer claims went unsupported by the retrieved chunks. For a regulated use this needs a claim-level grounding check before the answer is shown, and the reranker change should be judged on both numbers, not precision alone.
4. **Small eval sets.** 17 golden samples and 8 attacks are enough to find failure classes, not to certify a rate. A high-risk deployment would need a larger, domain-specific golden set owned with the deployer.
5. **Open-web tool.** The Wikipedia tool pulls unvetted content into context, which widens the poisoning surface. In a regulated deployment I would turn it off or restrict it to an allowlist.

## 5. Scenario: DocRAG as credit-decision support (Annex III 5(b))

If a lender used DocRAG to check applicant files against lending policy, the high-risk obligations would apply from **2 December 2027** (Annex III, as moved by the Omnibus). What I would have to add:

| Area | Article | Work needed |
|---|---|---|
| Risk management | 9 | Turn the threat model and eval into a maintained risk register, re-run per release as a CI gate |
| Data governance | 10 | Document corpus provenance, access control on who can add documents (the poisoning entry point), bias checks on policy documents |
| Technical documentation | 11, Annex IV | System description, eval method, metrics with CIs, known limitations |
| Logging | 12 | Per-request trace: query, retrieved chunk ids, model version, answer, reviewer decision. Langfuse tracing, as in my PR Review Agent, covers most of this |
| Instructions for use | 13 | Section 4 of this document, written for the deployer |
| Human oversight | 14 | Answer is advisory only; low-faithfulness or low-retrieval-score answers routed to a human, like the confidence gate in bugtriage-mcp |
| Accuracy and robustness | 15 | Semantic poisoning detector, claim-level grounding check, regression thresholds that block a release |
| Quality management and conformity | 17, 43 | Process work on the provider side; internal-control conformity assessment for Annex III point 5 systems |

The deployer would also carry obligations of its own, including a fundamental rights impact assessment (Art. 27) for credit scoring.

## 6. Next steps I would take first

1. Add the AI disclosure line and response provenance metadata (closes the two Art. 50 gaps, about a day).
2. Add per-request tracing (Art. 12).
3. Replace the keyword guardrail with a semantic claim checker and re-run the red-team harness to measure it.
4. Add an abstain path when faithfulness or retrieval scores fall below a threshold.

## Sources

- Regulation (EU) 2024/1689 (AI Act): Arts. 3, 4, 5, 6, 9 to 15, 27, 43, 50 and Annex III
- AI Omnibus amendments and dates: [White & Case, "EU AI Omnibus enters into force"](https://www.whitecase.com/insight-alert/eu-ai-omnibus-enters-force-amending-ai-act)
- Numbers: `redteam/results/redteam_20260824_*.json`, `eval/results/reranked_top6.json`, `eval/results/baseline_no_reranker.json`
