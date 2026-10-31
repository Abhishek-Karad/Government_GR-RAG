# Phase 0 architecture audit

## Actual implementation

The repository contains two serving paths:

- `gr_remote_rag.py` is the active command-line implementation. It discovers `.pdf.en.txt` files through the GitHub Git Trees API, fetches text, extracts path/regex metadata, optionally calls Gemini for enrichment, stores source text and chunks in SQLite, and writes embeddings to local Qdrant.
- `backend/` contains a FastAPI implementation split into configuration, database, embedding, retrieval, Gemini, schemas and application lifecycle modules. It shares the SQLite/Qdrant data model but duplicates some retrieval/configuration logic rather than importing the CLI path.

The current retrieval path is dense MiniLM query embedding → Qdrant candidate retrieval → optional `cross-encoder/ms-marco-MiniLM-L6-v2` reranking → duplicate removal → query-time candidate passage embeddings → coverage-aware MMR selection → optional Gemini structured answer generation. SQLite is the source of truth for document/chunk text and source provenance; Qdrant payloads carry chunk/config/version guards.

The checked-in database contains 102 documents and 1,219 chunks. The selected current v2 configuration (`structure-aware-v2`, 220 target tokens, 30 overlap, 384 dimensions) has 925 vectors; the preserved v1 configuration has 294 vectors. No Qdrant collection was deleted or modified by Phase 0.

## Configuration discrepancies

The requested benchmark baseline is `candidate_k=30`, `top_k=6`, `max_per_document=7`, `rerank=true`. The CLI defaults are currently `20/5/3` with reranking disabled; the FastAPI defaults are `20/5/3` with reranking controlled by `RERANK_ENABLED`. The Phase 0 harness passes the requested baseline explicitly and does not change either production default.

The CLI uses `structure-aware-v2` with 30-token overlap by default, while the checked-in older v1 index uses 40-token overlap. Both collections remain isolated and readable.

## Existing coverage and gaps

- The local indexed corpus currently has only 2 departments: Rural Development Department (100 documents) and Medical Education and Drugs Department (2 documents). Remote inventory of all 33 folders was not probed because network access was unavailable during the audit.
- Deterministic metadata currently includes source path, department, circular number, issue date where detected, page counts and extraction artifacts. Gemini enrichment is present for 25 documents and is persisted with status/error fields.
- Source spans and page markers are preserved for v2 chunks. Table-like text is retained as source text, but no generic structured table representation exists yet.
- Existing unit tests cover chunk token safety, source preservation, ranking/selection behavior and citation validation. There was no reusable JSONL benchmark or corpus audit before Phase 0.
- Existing code has no Phase 0 metric/reporting command and no shared benchmark runner across the CLI and FastAPI paths.

## Phase 0 decision

No retrieval behavior was changed. Phase 0 adds only a JSONL evaluation harness, verified local seed cases, a read-only corpus audit, reports and harness tests. Phase 1 requires review of these results before implementation.

## FastAPI evaluation migration

The FastAPI endpoint uses the shared `backend.retrieval_service.RetrievalService`, while the historical CLI benchmark imported and executed `gr_remote_rag.py` directly. Before this migration, `/api/v1/query` always called Gemini and exposed selected evidence plus aggregate timings, but not the original candidate set. Its request defaults were `candidate_k=20`, `top_k=5`, `max_per_document=3`, with reranking controlled by `RERANK_ENABLED` unless explicitly supplied.

The evaluation client now sends all `30/6/7` parameters explicitly over HTTP. `generate_answer` and `evaluation_diagnostics` are additive request fields gated by `EVAL_MODE_ENABLED`; ordinary API clients retain the previous defaults and answer-generation behavior. Diagnostics are collected from the same retrieval call that produced the answer and are not exposed unless explicitly requested in evaluation mode.
