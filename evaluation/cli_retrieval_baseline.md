# Historical baseline label

The original `evaluation/baseline_report.json` and `evaluation/baseline_report.md` are preserved as the historical `cli_retrieval_baseline`. They were produced by direct imports of `gr_remote_rag.py`, not by FastAPI.

Their metrics are not directly comparable to the new `fastapi_baseline` reports because the implementations and observability differ. The historical aggregate values were document recall 1.000, document-membership evidence recall 1.000, precision@6 0.333, and raw-substring fact coverage 0.196.
