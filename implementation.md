# Implementation Guide

## Overview

This project is a Retrieval-Augmented Generation (RAG) application for Maharashtra Government Resolutions (GRs).

The application:

1. Discovers English GR text files from the public `orgpedia/mahGRs` GitHub repository.
2. Downloads and parses the documents.
3. Stores source documents, metadata, and text chunks in SQLite.
4. Generates semantic embeddings for chunks using Sentence Transformers.
5. Stores embeddings in a local Qdrant collection.
6. Retrieves relevant passages for a user question.
7. Optionally reranks and diversifies the retrieved passages.
8. Sends the selected evidence to Gemini for a grounded answer.
9. Returns the answer together with evidence, scores, citations, and timings.

The important design rule is:

```text
SQLite = source of truth
Qdrant = vector search index
Gemini = grounded answer generator
```

## System architecture

```text
GitHub GR repository
        |
        v
Remote file discovery and download
        |
        v
Page parsing, metadata extraction, and chunking
        |
        +--------------------+
        |                    |
        v                    v
    SQLite               Embedding model
 documents/chunks             |
                               v
                            Qdrant
                         vector collection

User question
        |
        v
Question embedding
        |
        v
Qdrant candidate retrieval
        |
        v
SQLite validation
        |
        v
Deduplication and optional reranking
        |
        v
Coverage-aware evidence selection
        |
        v
Gemini grounded answer
        |
        v
FastAPI response and browser UI
```

## Repository structure

### Command-line implementation

`gr_remote_rag.py` contains the original end-to-end CLI implementation. It supports:

- `scan` — discover matching GitHub files without downloading their contents.
- `ingest` — download, chunk, embed, and index documents.
- `reindex` — rebuild an isolated vector index from text already stored in SQLite.
- `retry-enrichment` — retry failed Gemini metadata extraction.
- `search` — retrieve evidence and generate a grounded answer.
- `stats` — display document, chunk, enrichment, and department counts.

### FastAPI implementation

The `backend/` directory separates the service into smaller components:

| File | Responsibility |
| --- | --- |
| `backend/config.py` | Loads environment-based application settings. |
| `backend/database.py` | Creates SQLite connections and loads validated evidence. |
| `backend/embedding_service.py` | Loads SentenceTransformer and creates query/passage vectors. |
| `backend/index_config.py` | Creates deterministic model/chunking/index identities. |
| `backend/retrieval_service.py` | Performs Qdrant retrieval, validation, reranking, and selection. |
| `backend/gemini_service.py` | Builds grounded prompts and validates Gemini answers. |
| `backend/schemas.py` | Defines Pydantic request and response models. |
| `backend/app.py` | Creates the FastAPI application and HTTP endpoints. |

### Frontend

The `frontend/` directory contains the browser interface:

- `index.html` — page structure.
- `app.js` — submits questions and renders answers and evidence.
- `styles.css` — visual styling.

## Ingestion implementation

### GitHub discovery

The application uses the GitHub Git Trees API instead of cloning the repository. It traverses the `GRs` directory and selects files ending in `.pdf.en.txt`.

The `--department` option restricts traversal to a department. The `--limit` option should be kept small during testing because the repository can contain many documents.

### Document identity

Each document receives a deterministic ID based on its repository path. The source Git blob SHA is also stored. The SHA allows the application to detect whether a remote document has changed.

### Page parsing and source spans

The parser recognizes page markers such as:

```text
# Page 1
# Page 2
```

The application stores exact character offsets into the original `gr_text`. These offsets are retained in chunk records as `source_start`, `source_end`, and `source_spans_json`.

This allows retrieved evidence to remain traceable to the original saved text.

### Metadata extraction

Basic metadata is extracted locally using regular expressions, including:

- Circular number
- Issue date
- Department
- File name
- GitHub source URL
- Page count

When Gemini enrichment is enabled, additional fields are extracted:

- Subject and summary
- Keywords
- Policy category
- Audience and actions
- Referenced circular numbers
- Schemes and entities
- Numerical facts
- Table records

Gemini enrichment is stored separately from the raw document text. If the document content has not changed, existing enrichment is preserved.

### Structure-aware chunking

`SemanticDocumentChunker` attempts to keep related content together by using:

- Paragraph boundaries
- Numbered clauses
- Headings
- Page boundaries

If a unit is larger than the model limit, it is split using tokenizer offsets. The final formatted text is validated before it is sent to the embedding model.

The default configuration is:

```text
Target chunk size: 220 tokens
Overlap:           30 tokens
Strategy:          structure-aware-v2
```

The effective maximum is capped by the selected embedding model's actual sequence length.

## Storage model

### SQLite

SQLite stores the authoritative data:

- `documents` — document metadata and raw text.
- `chunks` — chunk text, page information, headings, and source spans.
- `index_configs` — model, revision, chunking, and prefix configuration.
- `document_index_status` — per-document indexing status for each configuration.

SQLite also stores enrichment status and errors so failed Gemini jobs can be retried.

### Qdrant

Qdrant stores normalized embedding vectors and payload metadata, including:

- `chunk_id`
- `document_id`
- Page range
- Source spans
- Document SHA
- Index version
- Embedding model
- Chunking strategy

Qdrant is not treated as the source of truth. Search validates the Qdrant payload against SQLite before using a result.

### Isolated index configurations

The collection name is derived from the embedding model, model revision, chunk size, overlap, prefix mode, and chunking strategy.

Changing the embedding model or chunking settings creates a new collection instead of mixing incompatible vectors with an existing collection.

## Query implementation

### 1. Query embedding

The user question is encoded using the configured SentenceTransformer model.

For E5-style models, the application uses `query:` and `passage:` prefixes. Other models, including MiniLM, use the original text without those prefixes.

### 2. Candidate retrieval

The API uses hybrid candidate generation:

- Qdrant performs cosine-similarity search for semantically related passages.
- SQLite performs exact-term lookup across raw chunk text, contextual retrieval text, and available metadata.

The lexical path does not require metadata. If fields such as the subject or circular number are missing, the raw chunk text is still searched. Dense and lexical candidate IDs are merged before validation and ranking.

Qdrant returns a configurable number of candidates. The API defaults are controlled by environment variables and can be overridden per request.

### 3. Candidate validation

Each Qdrant result is checked against SQLite. A result is rejected if:

- The chunk is missing.
- The document is not complete.
- The source SHA does not match.
- The index version does not match.
- The vector belongs to a different index configuration.

This prevents stale vectors from producing incorrect evidence.

### 4. Deduplication

Near-identical evidence passages are removed using normalized text comparison and sequence similarity.

### 5. Reranking

When enabled, the cross-encoder receives pairs of:

```text
(user question, contextual candidate passage)
```

The contextual passage can include the department, circular number, subject, and section heading when available. The original chunk text remains unchanged for evidence display and citation.

It produces a more question-aware relevance score than vector similarity alone.

If the reranker cannot load in the CLI, the CLI falls back to Qdrant ranking. The API normally fails startup if configured resources cannot be loaded.

### 6. Coverage-aware selection

The final selector combines relevance and redundancy reduction using an MMR-style score:

```text
MMR = relevance - redundancy penalty
```

It also applies:

- A relative dense/lexical score gate for the non-reranked baseline.
- A maximum number of passages per document.
- Optional permission for multiple passages from the same document.

When reranking is enabled, the old Qdrant relative-score gate is disabled so the reranker can rescue passages that dense similarity underrated. MMR still uses normalized reranker scores when they are available.

This is useful for questions that require several distinct facts from one circular, such as questions about precautions or instructions.

## Gemini answer generation

The selected passages are placed into a structured prompt containing:

- Evidence ID
- Department
- Circular number
- Page number
- Issue date
- Source URL
- Document metadata
- Exact source text

Gemini is instructed to answer only from the supplied evidence and to return structured JSON.

The answer schema contains:

- `answer`
- `claims`
- `evidence_ids`
- `verified`
- `limitations`
- `answer_status`

The answer validator rejects unknown evidence IDs. It also performs a lightweight lexical support check between each claim and its cited evidence. Unsupported claims cause the answer status to become `insufficient_evidence`.

## FastAPI endpoints

### Health check

```http
GET /health
```

Returns the active collection, embedding model, reranker state, and API version.

### Query endpoint

```http
POST /api/v1/query
Content-Type: application/json
```

Example:

```json
{
  "question": "What COVID precautions related to masks were instructed?",
  "candidate_k": 30,
  "top_k": 6,
  "max_per_document": 7,
  "rerank": true,
  "show_evidence": true,
  "generate_answer": true
}
```

The response includes the answer, evidence, retrieval timings, effective parameters, and optional diagnostics.

## Running the project

Install or synchronize dependencies:

```bash
uv sync
```

Scan the remote repository:

```bash
uv run python gr_remote_rag.py scan \
  --department Rural_Development_Department \
  --limit 10
```

Ingest documents:

```bash
uv run python gr_remote_rag.py ingest \
  --department Rural_Development_Department \
  --limit 5
```

Reindex existing SQLite text:

```bash
uv run python gr_remote_rag.py reindex \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --chunk-size 220 \
  --overlap 40 \
  --limit 5
```

Start the API:

```bash
HF_HUB_OFFLINE=1 uv run uvicorn backend.app:app \
  --host 0.0.0.0 \
  --port 8000
```

Then open:

```text
http://localhost:8000/
```

## Configuration

Optional `.env` settings include:

```text
GITHUB_TOKEN=...
GEMINI_API_KEY=...
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
QDRANT_PATH=qdrant_storage
GR_DB_PATH=gr_demo.sqlite3
CHUNK_SIZE=220
CHUNK_OVERLAP=30
RERANK_ENABLED=false
RERANK_PRELOAD=true
GEMINI_TIMEOUT_MS=15000
GEMINI_MAX_RETRIES=1
```

Never commit `.env` or API keys.

## Testing

Run the unit tests using the project virtual environment:

```bash
.venv/bin/python -m unittest discover -v
```

The tests cover:

- Token-safe chunking
- Page and source-span preservation
- E5 prefix handling
- Index configuration isolation
- Citation validation
- Reranking
- Deduplication
- Evidence diversity
- API request behavior
- Evaluation metrics

The live FastAPI integration test is opt-in:

```bash
RUN_FASTAPI_INTEGRATION=1 .venv/bin/python -m unittest discover -v
```

## Current limitations

This is a working prototype rather than a complete production deployment.

- The locally indexed database may not contain every remote GR.
- Only English translated text is indexed.
- Some source URLs point to GitHub instead of verified official PDF URLs.
- OCR and table formatting can be imperfect.
- Gemini answer generation requires `GEMINI_API_KEY`.
- Models must be downloaded and available locally.
- SQLite and Qdrant are local storage systems.
- There is no authentication layer.
- There is no scheduled ingestion worker.
- Some query behavior, such as allowing multiple passages from one document, uses keyword-based heuristics.

## Verification status

The current local unit test suite passes with 29 tests passing and one optional integration test skipped unless explicitly enabled.

