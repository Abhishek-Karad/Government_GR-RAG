# Remote Maharashtra GR RAG prototype

This demo uses the public `orgpedia/mahGRs` repository **without cloning it**. It discovers files through GitHub's Git Trees API, fetches selected `.pdf.en.txt` files, extracts basic metadata, optionally generates grounded metadata with Gemini, and stores document/chunk text in SQLite. Embeddings and vector similarity search are stored separately in a persistent local Qdrant collection.

## Setup

```bash
uv sync
```

The locked environment is created in `.venv` automatically. Run the demo with
`uv run`; uv will use the lockfile and keep the environment in sync:

```bash
uv run python gr_remote_rag.py stats
```

(Optional) Create `.env` containing `GITHUB_TOKEN=...` to raise GitHub API limits and `GEMINI_API_KEY=...` to enrich metadata. When the Gemini key is present, enrichment runs automatically; `--gemini` can also be used explicitly. Never commit `.env`.

## Run

```bash
uv run python gr_remote_rag.py scan --department Rural_Development_Department --limit 10
uv run python gr_remote_rag.py ingest --department Rural_Development_Department --limit 5
uv run python gr_remote_rag.py reindex --model sentence-transformers/all-MiniLM-L6-v2 --chunk-size 220 --overlap 40 --limit 5
uv run python gr_remote_rag.py stats
uv run python gr_remote_rag.py search "What COVID precautions were advised for families?"

# reranked, coverage-aware search with full evidence
uv run python gr_remote_rag.py search "What COVID precautions were advised for families?" \
  --candidate-k 20 --top-k 6 --rerank --max-per-document 3 --show-evidence

# start the API service; models and local indexes load once at startup
HF_HUB_OFFLINE=1 uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000

# query the API
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What COVID precautions related to masks were instructed?","show_evidence":true}'
```

To explicitly request Gemini-assisted metadata extraction:

```bash
uv run python gr_remote_rag.py ingest --department Finance_Department --limit 3 --gemini
```

Gemini generates a document title/subject, faithful summary, keywords, policy category, referenced circular numbers, audience, and document actions from the file name, department, repository path, and document text. The `documents` table also stores the raw GR text, extraction source, parser-derived page count, GitHub path, official PDF URL placeholder, and government-source verification flag.

`--department` can be set to another directory under `GRs`. To traverse all departments use `--department ''`, but **always set a small `--limit` during testing**. SQLite is the source of truth for document metadata, raw GR text, and chunk text. Qdrant stores embeddings and chunk payload IDs, allowing metadata updates and vector reindexing to remain separate. The default local Qdrant path is `qdrant_storage`; override it with `QDRANT_PATH` or `--qdrant`.

## Notes

- `scan` does not download the contents of individual GRs and does not load an embedding model.
- First `ingest` downloads the sentence-transformer model weights; after caching it runs locally.
- English translations are indexed; original Marathi texts are not included in this small demo.
- Date and circular extraction may be missing for some source formats; URLs point to the GitHub file rather than a verified original PDF.
- GitHub trees are traversed non-recursively, avoiding truncation of large recursive tree responses.
- The prototype is idempotent for unchanged Git blob hashes, but not a production-grade scheduler.
- The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`; it receives raw text without E5 prefixes. E5 models receive `passage:` and `query:` prefixes. Override detection with `EMBEDDING_PREFIX_MODE=none|e5`.
- Each model/revision/chunking/prefix configuration gets its own deterministic Qdrant collection. Existing collections are preserved and same-dimension vectors from another configuration are rejected.
- Chunks are stored per index configuration in SQLite and use paragraph, heading, and numbered-clause boundaries before tokenizer-aware splitting. The effective maximum is capped by the selected model's supported sequence length.
- The active chunking strategy is `structure-aware-v2`: it uses the embedding tokenizer's offset mappings and validates the exact prefixed, special-token-inclusive input before every embedding call. Defaults are a 220-token target and 30-token overlap; with MiniLM the effective model limit is read from the loaded model (normally 256).
- `reindex` rebuilds from `documents.gr_text` and does not call GitHub or Gemini. It is safe for creating a new model/configuration index while retaining existing chunks and collections.
- `search` first retrieves and validates chunks with the embedding model, then sends the question and those chunks to Gemini for a grounded answer. Answers cite retrieved chunks as `[SOURCE N]`; set `GEMINI_ANSWER_MODEL` to use a different answer model than `GEMINI_MODEL`.
- Search can optionally rerank up to 20 validated Qdrant candidates with `cross-encoder/ms-marco-MiniLM-L6-v2` (override with `RERANKER_MODEL`). After ranking, MMR selects the final 5-8 passages using reranker relevance plus candidate-passage embedding similarity, so duplicate passages are penalized while distinct facts from one circular can remain. A Qdrant relative-score gate prevents unrelated circulars being added just to fill the budget. Omit `--rerank` for the embedding-only baseline; the same coverage selector is used in both modes for fair benchmarking. `--show-evidence` prints stable evidence IDs, Qdrant/reranker scores, document metadata, source spans, and the exact saved chunk text.
- The coverage embeddings are computed only for the retrieved candidates at query time and are not persisted or used to rebuild the GR index. Search output includes `retrieval_ms`, `ranking_ms`, and `selection_ms` so reranking-on/off comparisons can report coverage, retrieval accuracy, and latency separately. The API accepts `top_k` from 1 through 8 (5 by default).
- Changing from `structure-aware-v1` to `structure-aware-v2` creates an isolated model/index configuration. Existing vectors are retained; ingest or reindex the desired documents into the new collection before switching the active query service. Gemini enrichment metadata is preserved when document content is unchanged.
- New chunks store `source_start`, `source_end`, and `source_spans_json` in SQLite and the same span metadata in Qdrant payloads. Existing documents and collections remain readable; run `reindex` only when you need span metadata and the new exact-slice chunking for already indexed documents. This may change chunk boundaries and therefore requires new vectors in the selected isolated collection.
- Gemini answers are validated JSON with `answer`, `claims`, `evidence_ids`, `limitations`, and `answer_status`. Unknown IDs are rejected; claims without lexical support are marked unverified and the result is downgraded to `insufficient_evidence`.
- The FastAPI backend is split under `backend/`: `config.py` owns settings, `embedding_service.py` owns MiniLM, `database.py` owns SQLite access, `retrieval_service.py` owns Qdrant/filtering/reranking/diversification, `gemini_service.py` owns the reusable Gemini client, and `app.py` owns lifecycle and HTTP endpoints. The original CLI remains available.
- The API loads shared resources once during startup. SQLite connections are created per request, while model, Qdrant client, and Gemini client state are reused. Set `RERANK_ENABLED=true` to load the cross-encoder during startup; leave it false if the reranker model is not cached.
- Gemini query latency is bounded with `GEMINI_TIMEOUT_MS=15000`, one SDK attempt, and one application retry by default. Use `GEMINI_MAX_RETRIES=0` for fail-fast behavior when the API is repeatedly returning `503`; `GEMINI_SDK_ATTEMPTS` controls the SDK-side attempt count. Gemini 3.x requests use minimal thinking for this short grounded-answer task.
- If the configured Gemini model returns `503 UNAVAILABLE` or `504 DEADLINE_EXCEEDED`, the API/CLI makes one bounded attempt with `GEMINI_FALLBACK_MODEL` (default `gemini-2.5-flash-lite`). Set it to an available model or set it empty to disable fallback.
