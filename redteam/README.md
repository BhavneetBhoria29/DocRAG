# DocRAG Red-Team — Indirect Prompt Injection

A red-team harness that measures how well DocRAG resists **indirect prompt
injection through retrieved documents**, and how much a detection guardrail
reduces that risk and at what cost.

This grafts onto DocRAG rather than standing alone, because the whole point is
that **RAG's retrieved documents are an attack surface**: a poisoned chunk that
gets retrieved reaches the generation step and can hijack it. Testing that surface
means testing the real retriever + generation pipeline, not a mock.

## Threat model

An attacker controls or influences a document that enters DocRAG's retrieval
corpus (an uploaded file, a scraped page, a shared knowledge base). The attacker
embeds instructions in that document. When a normal user query retrieves the
poisoned chunk, the injected instruction reaches the model. Attack goals:

| Family | Goal |
|---|---|
| `override` | Hijack: make the model ignore the user and follow the attacker |
| `exfil` | Leak the system prompt or other retrieved documents |
| `fact` | Plant a false fact and get the model to repeat it as truth |
| `refusal` | Denial of service: force the model to refuse legitimate queries |
| `obfuscated` | The above, encoded or framed to evade a naive detector |

## What it measures

- **ASR (Attack Success Rate)** — fraction of attacks that succeed, per family
  and overall, baseline vs guarded. Lower is better.
- **FPR (False Positive Rate)** — fraction of benign queries the guardrail
  wrongly flags. This is the cost of the defense and the honest answer to
  "what did you sacrifice".

Both come with **bootstrap 95% confidence intervals**, same discipline as the
DocRAG RAGAS harness, because the sample sizes are small and a bare point
estimate overstates precision.

Success is detected **deterministically** wherever possible: each attack plants a
canary token the model would never emit unless the injection worked, so the core
ASR number is judge-free and reproducible. A GPT-4o (temp 0) semantic judge,
matching your RAGAS judge, handles the attacks whose success is not a single
token (e.g. dumping other retrieved documents).

## The guardrail

A two-stage detector that screens **retrieved context** before it reaches the
generation prompt (the correct place, since the payload rides in on a chunk):

1. **Heuristic stage** — flags instruction-shaped text aimed at the assistant,
   delimiter-breaking, and encoded payloads. Fast and free, but over-flags benign
   instructional docs, which is where false positives come from.
2. **LLM stage (optional)** — classifies each chunk as reference content vs
   instructions directed at an assistant, vetoing heuristic false positives at
   the cost of a per-chunk call.

Action on a flagged chunk is `sanitize` (drop the chunk, answer from the rest) by
default, or `block` (refuse the whole query). The `obfuscated` family is designed
so some attacks evade the detector: ASR does not drop to zero, and that residual
is the honest evasion gap.

## Files

```
redteam/
├── adapter.py                  # the ONLY repo-specific file — wire this to DocRAG
├── corpus/
│   ├── injection_payloads.py   # labeled attack library with canaries
│   └── benign_queries.py       # clean docs + queries for FPR
├── scorer.py                   # canary + LLM-judge success detection
├── guardrail.py                # the two-stage injection detector
├── metrics.py                  # ASR / FPR with bootstrap 95% CIs
└── run_redteam.py              # orchestrator: baseline ASR -> guarded ASR + FPR
```

## Wiring (do this once)

Everything is ready except three calls only you can supply, all in `adapter.py`,
all mirroring what `eval/evaluate.py` already does:

1. `_build_pipeline()` — build your retriever + LangGraph pipeline, pointed at the
   **ephemeral `REDTEAM_COLLECTION`**, not your real index.
2. `ingest_documents(docs)` — add texts to that ephemeral collection.
3. `query(user_input)` — run the live pipeline, return `answer` + `contexts`.

**Do not point this at `data/chroma_db`.** Indirect-injection testing plants
malicious documents in the corpus; use the isolated collection so your real index
is never contaminated. `reset_corpus()` tears it down between cases.

## Run

```bash
export OPENAI_API_KEY=sk-...
pip install -r redteam/requirements-redteam.txt

python redteam/run_redteam.py --n 3          # cheap smoke test first
python redteam/run_redteam.py                # heuristic guardrail, sanitize mode
python redteam/run_redteam.py --llm-stage    # add the LLM detector stage
python redteam/run_redteam.py --block        # stricter block mode (higher FPR)
```

Results are written to `redteam/results/redteam_<timestamp>.json`.

Same rate-limit note as the DocRAG eval: the judge and LLM-stage fire OpenAI
calls; top up credits and, if you hit 429s, the harness already runs judge calls
serially per case.

## How to report the numbers (honestly)

Once you have a real run, the CV bullet and README line follow this shape and
**stay empty until then**:

> Red-teamed DocRAG for indirect prompt injection with a canary-based harness
> (N attacks across 5 families, bootstrap 95% CIs). Measured baseline ASR of
> X%, added a context-screening guardrail that cut ASR to Y% at Z% FPR;
> obfuscated attacks are the residual evasion gap.

Report **all three**: baseline ASR, guarded ASR, and FPR. A guardrail reported
with ASR reduction but no FPR is not a result. Name the guardrail mode
(sanitize or block) and the judge model. If you enable PyRIT for the scoring
layer, say so; if you don't, don't claim it.
```
