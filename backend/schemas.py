from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=8)
    candidate_k: int = Field(default=20, ge=1, le=100)
    max_per_document: int = Field(default=3, ge=1, le=20)
    rerank: bool | None = None
    show_evidence: bool = False


class ClaimResponse(BaseModel):
    text: str
    evidence_ids: list[str]
    verified: bool


class AnswerResponse(BaseModel):
    answer: str
    claims: list[ClaimResponse]
    limitations: list[str]
    answer_status: str


class EvidenceResponse(BaseModel):
    evidence_id: str
    document_id: str
    circular: str | None = None
    issue_date: str | None = None
    department: str | None = None
    page_label: str
    source_url: str | None = None
    qdrant_score: float
    document_relevance: float
    reranker_score: float | None = None
    text: str | None = None


class QueryResponse(BaseModel):
    answer: AnswerResponse
    evidence: list[EvidenceResponse]
    timings_ms: dict[str, float]
    reranker_enabled: bool


class HealthResponse(BaseModel):
    status: str
    collection: str
    embedding_model: str
    reranker_enabled: bool

