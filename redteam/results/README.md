# Red-team run history

Two phases, kept for transparency:

- **192717, 193835, 195030** — Phase 1, naive command-style payloads
  ("ignore the user", "reveal your prompt"). Baseline ASR ~0.00: DocRAG's
  generation prompt treats retrieved chunks as data, so overt commands failed.
  This null result motivated hardening the payloads.

- **200207, 224943, 225928** — Phase 2, hardened payloads that blend into
  document voice (authority-framed false content). Baseline ASR 0.38–0.62
  across seeds, driven entirely by the data-poisoning family. These runs back
  the reported findings.

Rates vary run-to-run (n=8, non-deterministic generator); results are reported
as ranges, not point estimates.
