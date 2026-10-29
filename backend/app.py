from __future__ import annotations

import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from .config import Settings
from .database import Database
from .embedding_service import EmbeddingService
from .gemini_service import GeminiService
from .index_config import COLLECTION, get_index_config
from .retrieval_service import RetrievalService
from .schemas import EvidenceResponse, HealthResponse, QueryRequest, QueryResponse


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = Lock()
        self.embedder = None
        self.database = None
        self.qdrant = None
        self.config = None
        self.legacy = False
        self.retrieval = None
        self.gemini = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    state = AppState(settings)
    state.embedder = EmbeddingService(settings)
    state.database = Database(settings.db_path)
    from qdrant_client import QdrantClient
    state.qdrant = QdrantClient(path=settings.qdrant_path)
    conn = state.database.connection()
    try:
        config = get_index_config(conn, state.embedder.model, settings, create=False)
        if not config and state.qdrant.collection_exists(COLLECTION):
            config = {'config_id': None, 'collection_name': COLLECTION}
            state.legacy = True
        if not config or not state.qdrant.collection_exists(config['collection_name']):
            raise RuntimeError('No compatible Qdrant collection found; run ingest or reindex first')
        actual_dimension = state.qdrant.get_collection(config['collection_name']).config.params.vectors.size
        if actual_dimension != state.embedder.dimension:
            raise RuntimeError(f'Qdrant dimension {actual_dimension} does not match embedding dimension {state.embedder.dimension}')
        state.config = config
    finally:
        conn.close()
    state.retrieval = RetrievalService(settings, state.embedder, state.database, state.qdrant, state.config, state.legacy)
    state.gemini = GeminiService(settings)
    if settings.rerank_enabled:
        from .retrieval_service import reranker_model
        reranker_model(settings.reranker_model)
    app.state.rag = state
    yield
    state.qdrant.close()


app = FastAPI(title='Maharashtra GR RAG API', version='1.0.0', lifespan=lifespan)


@app.get('/health', response_model=HealthResponse)
async def health(request: Request):
    state: AppState = request.app.state.rag
    return HealthResponse(status='ok', collection=state.config['collection_name'],
                          embedding_model=state.settings.embedding_model,
                          reranker_enabled=state.settings.rerank_enabled)


def _query(state: AppState, payload: QueryRequest):
    rerank = state.settings.rerank_enabled if payload.rerank is None else payload.rerank
    with state.lock:
        evidence, timings = state.retrieval.retrieve(
            payload.question, candidate_k=payload.candidate_k, top_k=payload.top_k,
            max_per_document=payload.max_per_document, rerank=rerank)
        answer_started = time.perf_counter()
        answer = state.gemini.answer(payload.question, evidence)
        timings['gemini_ms'] = round((time.perf_counter() - answer_started) * 1000, 2)
    evidence_response = [EvidenceResponse(
        evidence_id=item['evidence_id'], document_id=item['document_id'], circular=item['circular'],
        issue_date=item['issue_date'], department=item['department'], page_label=item['page_label'],
        source_url=item['source_url'], qdrant_score=item['qdrant_score'],
        document_relevance=item['document_relevance'], reranker_score=item.get('reranker_score'),
        text=item['text'] if payload.show_evidence else None,
    ) for item in evidence]
    return QueryResponse(answer=answer, evidence=evidence_response, timings_ms=timings,
                         reranker_enabled=rerank)


@app.post('/api/v1/query', response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request):
    state: AppState = request.app.state.rag
    try:
        return await run_in_threadpool(_query, state, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

