# Independent code review

Reviewer: separate implementation agent `/root/build_speech_vision_repos`, not the author of this repository. Reviewed 2026-11-16.

## Round 1 — method, scoring and data correctness

- Fixed tokenized document slots and gold-document boundaries are verified within each question across distractor proportions. Gold answers do not enter distractors.
- Round 1 found that the parser accepts a final JSON object preceded by prose, while the metric was named strict JSON. Resolved in round 2: renamed `final_json_answer_accuracy`, documented prefix acceptance, retained original raw provider responses and separate exact-whole-answer scores.
- Round 1 found duplicate rounded hard-distractor conditions could repeat IDs and inflate samples. Resolved: preparation rejects duplicate rounded counts, with regression coverage.
- Explicit verified/unverified teaching records are easier than natural misleading evidence; the repo clearly limits claims and does not manufacture a degradation curve.

## Round 2 — readability, tutorials and reproducibility

- Re-inspected the fixes, public source links, English/繁中 instructions, CLI failure records, configuration handling, pinned resources and license files.
- Independently reran **9 tests**, all passed. Inspected both executed notebooks (01_quickstart.ipynb, 02_evaluation.ipynb): code syntax valid, no saved error outputs, mode chosen by the visible control remains intact, credentials come from environment/Colab Secrets, and evaluation can rescore saved predictions without spending API requests.
- Reviewed actual stored provider evidence and its fixture/live distinction. Provider execution itself was performed by the implementation team; this review did not independently reproduce every network request.
- No unresolved high- or medium-priority code findings remain within the reviewed small-scale scope. Public-dataset access, unrun full benchmarks, model-side nondeterminism and paper-scale reproduction remain limitations described in the README/validation report.

## Recheck command

```bash
python -m unittest discover -s tests -v
```

This is a scoped code/method review, not a claim of full paper reproduction or external peer review.
