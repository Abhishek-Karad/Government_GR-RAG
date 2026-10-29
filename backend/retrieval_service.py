from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

from .database import Database
from .embedding_service import EmbeddingService


_RERANKER = None


def reranker_model(model_name: str):
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        print(f'[startup] Loading reranker: {model_name}')
        _RERANKER = CrossEncoder(model_name)
    return _RERANKER


def rerank_candidates(question, candidates, enabled, model_name):
    result = [dict(item, reranker_score=None) for item in candidates]
    if enabled and result:
        scores = reranker_model(model_name).predict([(question, item['text']) for item in result])
        for item, score in zip(result, scores):
            item['reranker_score'] = float(score)
        result.sort(key=lambda item: (item['reranker_score'], item['qdrant_score']), reverse=True)
    else:
        result.sort(key=lambda item: item['qdrant_score'], reverse=True)
    return result


def normalized_text(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def duplicate_evidence(left, right):
    a, b = normalized_text(left['text']), normalized_text(right['text'])
    return not a or not b or a == b or a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.88


def deduplicate_candidates(candidates):
    result = []
    for candidate in candidates:
        if not any(duplicate_evidence(candidate, existing) for existing in result):
            result.append(candidate)
    return result


def diversify_candidates(candidates, top_k, max_per_document, allow_same_document):
    if not candidates:
        return []
    floor = max(item['qdrant_score'] for item in candidates) * float(
        os.getenv('DIVERSIFICATION_MIN_RELATIVE_SCORE', '0.60'))
    relevant = [item for item in candidates if item['qdrant_score'] >= floor]
    selected, counts = [], {}
    for item in relevant:
        if counts.get(item['document_id'], 0) >= max_per_document:
            continue
        selected.append(item)
        counts[item['document_id']] = counts.get(item['document_id'], 0) + 1
        if len(selected) == top_k:
            return selected
    if allow_same_document:
        used = {item['evidence_id'] for item in selected}
        for item in relevant:
            if item['evidence_id'] not in used:
                selected.append(item)
                if len(selected) == top_k:
                    break
    return selected


def coverage_aware_select(candidates, *, top_k, max_per_document,
                          allow_same_document, passage_embeddings,
                          mmr_lambda=0.72):
    """Select relevant, non-redundant evidence with MMR.

    Candidates remain gated by the original Qdrant score so diversity cannot
    pull an unrelated circular into the answer.  MMR then uses the reranker
    score (or Qdrant score for the baseline) for relevance and embedding
    cosine similarity for redundancy.  The candidate objects are returned
    unchanged, preserving IDs, citations, pages, and scores.
    """
    if not candidates or top_k < 1:
        return []
    best_qdrant = max(item['qdrant_score'] for item in candidates)
    minimum_score = best_qdrant * float(os.getenv('DIVERSIFICATION_MIN_RELATIVE_SCORE', '0.60'))
    pool = [item for item in candidates if item['qdrant_score'] >= minimum_score]
    if not pool:
        return []

    raw_scores = [item.get('reranker_score') if item.get('reranker_score') is not None
                  else item['qdrant_score'] for item in pool]
    low, high = min(raw_scores), max(raw_scores)
    if high == low:
        relevance = {id(item): 1.0 for item in pool}
    else:
        relevance = {id(item): (score - low) / (high - low)
                     for item, score in zip(pool, raw_scores)}
    vectors = {id(item): vector for item, vector in zip(candidates, passage_embeddings or [])}

    def cosine(left, right):
        if left is None or right is None:
            return 0.0
        return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))

    selected, counts = [], {}
    remaining = list(pool)
    while remaining and len(selected) < top_k:
        eligible = [item for item in remaining
                    if counts.get(item['document_id'], 0) < max_per_document]
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda item: (
                mmr_lambda * relevance[id(item)] -
                (1.0 - mmr_lambda) * max(
                    (cosine(vectors.get(id(item)), vectors.get(id(other)))
                     for other in selected), default=0.0),
                relevance[id(item)],
            ),
        )
        selected.append(best)
        counts[best['document_id']] = counts.get(best['document_id'], 0) + 1
        remaining.remove(best)

        # When the question is not a multi-part policy question, retain the
        # existing document diversity behavior.  For broad questions, the
        # caller explicitly allows multiple distinct facts from one circular.
        if not allow_same_document:
            remaining = [item for item in remaining
                         if counts.get(item['document_id'], 0) == 0]
    return selected


class RetrievalService:
    """Retrieves, validates, reranks, deduplicates, and diversifies evidence."""

    def __init__(self, settings, embedder: EmbeddingService, database: Database, qdrant, config, legacy):
        self.settings = settings
        self.embedder = embedder
        self.database = database
        self.qdrant = qdrant
        self.config = config
        self.legacy = legacy

    def retrieve(self, question: str, *, candidate_k: int, top_k: int,
                 max_per_document: int, rerank: bool) -> tuple[list[dict[str, Any]], dict[str, float]]:
        started = time.perf_counter()
        vector = self.embedder.encode_query(question)
        encoded_ms = (time.perf_counter() - started) * 1000
        hit_records = self.qdrant.query_points(
            self.config['collection_name'], query=vector, limit=candidate_k, with_payload=True
        ).points
        qdrant_ms = (time.perf_counter() - started) * 1000
        chunk_ids = [str((hit.payload or {}).get('chunk_id')) for hit in hit_records
                     if (hit.payload or {}).get('chunk_id')]
        conn = self.database.connection()
        try:
            rows = self.database.get_evidence(
                conn, chunk_ids, config_id=self.config['config_id'], legacy=self.legacy,
                embedding_model=self.settings.embedding_model)
        finally:
            conn.close()
        validated_ms = (time.perf_counter() - started) * 1000

        candidates = []
        for hit in hit_records:
            payload = hit.payload or {}
            item = rows.get(str(payload.get('chunk_id')))
            if not item or payload.get('sha') != item['expected_sha']:
                continue
            if item['expected_version'] and payload.get('index_version') != item['expected_version']:
                continue
            page_label = str(item['page_start']) if item['page_start'] == item['page_end'] else f"{item['page_start']}-{item['page_end']}"
            try:
                spans = json.loads(item['source_spans_json']) if item['source_spans_json'] else payload.get('source_spans', [])
            except (TypeError, json.JSONDecodeError):
                spans = payload.get('source_spans', [])
            candidates.append({
                'evidence_id': f'E{len(candidates) + 1}', 'document_id': item['document_id'],
                'chunk_id': item['chunk_id'], 'text': item['text'], 'page_label': page_label,
                'page_start': item['page_start'], 'page_end': item['page_end'],
                'department': item['department'], 'circular': item['circular'],
                'issue_date': item['issue_date'], 'subject': item['subject'],
                'source_url': item['source_url'], 'source_start': item['source_start'],
                'source_end': item['source_end'], 'source_spans': spans,
                'qdrant_score': float(hit.score),
            })
        candidates = deduplicate_candidates(candidates)
        document_scores = {}
        for item in candidates:
            document_scores[item['document_id']] = max(document_scores.get(item['document_id'], float('-inf')), item['qdrant_score'])
        for item in candidates:
            item['document_relevance'] = document_scores[item['document_id']]
        ranked = rerank_candidates(question, candidates, enabled=rerank, model_name=self.settings.reranker_model)
        allow_same_document = any(word in question.casefold() for word in ('precaution', 'instruction', 'measure', 'guideline'))
        selection_started = time.perf_counter()
        passage_embeddings = self.embedder.encode_passages([item['text'] for item in ranked])
        selected = coverage_aware_select(
            ranked, top_k=top_k, max_per_document=max_per_document,
            allow_same_document=allow_same_document,
            passage_embeddings=passage_embeddings)
        return selected, {
            'embedding_ms': round(encoded_ms, 2),
            'qdrant_ms': round(qdrant_ms - encoded_ms, 2),
            'sqlite_ms': round(validated_ms - qdrant_ms, 2),
            'ranking_ms': round((time.perf_counter() - started) * 1000 - validated_ms, 2),
            'selection_ms': round((time.perf_counter() - selection_started) * 1000, 2),
            'retrieval_total_ms': round((time.perf_counter() - started) * 1000, 2),
        }

