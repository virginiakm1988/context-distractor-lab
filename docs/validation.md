# Execution and validation

Validation date: 2026-11-16. Separate fixture, CPU preparation and provider execution evidence are retained.

- 9 offline unit tests passed; see [test output](../results/test-output.txt).
- Both interactive notebooks executed without cell errors using Python 3.12, ipywidgets 8.1.7 and nbclient 0.10.2. The saved quickstart uses explicit fixture mode; output is not presented as model inference. See [validation record](../results/validation.json).
- The package is independently installable; use one environment per repository because all teaching packages use the same import name `project_demo`.
- Live-mode CPU preparation passed with the real NVIDIA Lightning tokenizer pinned at `d468880b6ad3c6e0d21377ce7242adaea4cc884d`; file SHA-256 `623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7`. This step makes no model calls and is labeled prepared inputs.
- Real Data Designer 0.9.3 + Lightning template generation completed: 3 rows, 2 structurally valid candidates (2 unique valid templates). All candidates remain pending human review. See [actual SDK output](../results/designer-live-v2/templates.jsonl); earlier failed checks are retained in results/designer-live. These generated candidates were not used in the original NIM experiment, which predates their review.

## Real NIM small run and response-format audit

[Original run](../results/live-smoke/run_manifest.json): 12 real requests with 7 valid JSON answers and 5 plain-answer responses that failed the final-JSON parser. Original predictions and metrics are retained.

The [audited metrics](../results/live-smoke/metrics-audited.json) and [audited responses](../results/live-smoke/predictions-audited.jsonl) match every provider response to the original request SHA-256. No new model calls were made. All 12 complete answers match their oracle; final-JSON answer accuracy is 7/12. Both metrics are reported, so output-format variation is not mistaken for a reasoning degradation. The additional plain-answer metric was introduced after inspecting this run; it is a diagnostic, not a preregistered paper reproduction.

Within each question, all four distractor conditions have identical tokenizer length, gold-document start/end and question. Three small synthetic scenarios are easier than the paper corpus because verified/unverified labels are explicit. No nonlinear effect was established. Reproduce the audit with `python -m project_demo.audit_rescore --output-dir results/live-smoke`.

## Remaining validation scope

Hosted NIM inference does not require a local GPU. No GPU training is part of these three repositories. SDK compatibility on another account/runtime must be confirmed from real executions. Failures are retained and never replaced with fixtures. Independent review is recorded separately in `REVIEW.md`.

The final-JSON parser allows explanatory text before its last JSON object, or a fenced object, and rejects trailing prose. It does not require the entire response to be only JSON.

Python wheel build passed; the distribution includes its license files.
