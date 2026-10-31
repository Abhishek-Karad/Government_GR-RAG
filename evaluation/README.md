# Phase 0 evaluation harness

`verified_cases_100.jsonl` is the current 100-question benchmark. `seed_cases.jsonl` remains the original nine-case regression set. Cases use stable document IDs, GR numbers, verified evidence markers/pages, typed facts, answerability and question type; chunk IDs are optional.

Generate and verify the 100 cases from the current contextual-hybrid SQLite index with:

```bash
.venv/bin/python evaluation/generate_verified_cases.py
```

The generator writes `verified_cases_report.json` and excludes any candidate whose evidence marker, page, or fact cannot be verified against current indexed chunks.

The evaluation harness is an external FastAPI client. Start the API with evaluation mode explicitly enabled so diagnostics and retrieval-only requests are accepted:

```bash
EVAL_MODE_ENABLED=true HF_HUB_OFFLINE=1 .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Run retrieval-only evaluation with the unchanged `30/6/7` parameters:

```bash
.venv/bin/python evaluation_harness.py --mode retrieval --base-url http://127.0.0.1:8000 --warmup --runs 3
```

Run full end-to-end evaluation when Gemini is configured:

```bash
.venv/bin/python evaluation_harness.py --mode end-to-end --base-url http://127.0.0.1:8000 --runs 3
```

Reports are written to `fastapi_baseline_report.json/md` and do not overwrite the historical CLI reports. API failures are recorded separately from retrieval misses. Retrieval metrics require the explicitly enabled, same-request diagnostics response. The report includes document recall, candidate-evidence recall, strict selected-evidence recall, precision, fact coverage, numerical exact-match accuracy, citation support, abstention accuracy, latency, grouped macro/micro summaries, and per-case evidence-loss diagnostics.
