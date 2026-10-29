from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv('GR_DB_PATH', 'gr_demo.sqlite3')
    qdrant_path: str = os.getenv('QDRANT_PATH', 'qdrant_storage')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    reranker_model: str = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L6-v2')
    gemini_model: str = os.getenv('GEMINI_ANSWER_MODEL', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite'))
    gemini_fallback_model: str = os.getenv('GEMINI_FALLBACK_MODEL', 'gemini-2.5-flash-lite')
    rerank_enabled: bool = os.getenv('RERANK_ENABLED', 'false').lower() in {'1', 'true', 'yes'}
    top_k: int = int(os.getenv('API_TOP_K', '5'))
    candidate_k: int = int(os.getenv('API_CANDIDATE_K', '20'))
    max_per_document: int = int(os.getenv('API_MAX_PER_DOCUMENT', '3'))
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '220'))
    overlap: int = int(os.getenv('CHUNK_OVERLAP', '30'))

