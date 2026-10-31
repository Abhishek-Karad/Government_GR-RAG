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


def contextual_text(item):
    """Build a retrieval representation while preserving original evidence text."""
    def clipped(value, limit):
        value = str(value) if value else ''
        return value[:limit]

    labels = [
        ('Department', clipped(item.get('department'), 120)),
        ('Circular', clipped(item.get('circular'), 160)),
        ('Subject', clipped(item.get('subject'), 240)),
        ('Section', clipped(item.get('heading'), 160)),
    ]
    context = [f'{label}: {value}' for label, value in labels if value]
    context.append('Content: ' + item['text'])
    return '\n'.join(context)


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
        model = reranker_model(model_name)
        tokenizer = model.tokenizer
        max_length = min(value for value in (
            getattr(model, 'max_length', None),
            getattr(tokenizer, 'model_max_length', None),
        ) if isinstance(value, int) and 0 < value < 100000)
        question_tokens = len(tokenizer.encode(question, add_special_tokens=False, truncation=False))
        passage_budget = max(32, max_length - question_tokens - 8)
        pairs = []
        for item in result:
            passage = item.get('retrieval_text') or item['text']
            tokens = tokenizer.encode(passage, add_special_tokens=False, truncation=False)
            if len(tokens) > passage_budget:
                passage = tokenizer.decode(tokens[:passage_budget], skip_special_tokens=True)
            pairs.append((question, passage))
        scores = model.predict(pairs)
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
                          mmr_lambda=0.72, apply_score_gate=True):
    """Select relevant, non-redundant evidence with MMR.

    The non-reranked baseline uses a relative base-score gate. When a
    cross-encoder is enabled, all validated candidates reach MMR so a lower
    dense score cannot discard a passage the reranker considers relevant.
    MMR uses the normalized reranker score when available.
    """
    if not candidates or top_k < 1:
        return []
    if apply_score_gate:
        best_score = max(item.get('base_score', item['qdrant_score']) for item in candidates)
        minimum_score = best_score * float(os.getenv('DIVERSIFICATION_MIN_RELATIVE_SCORE', '0.60'))
        pool = [item for item in candidates
                if item.get('base_score', item['qdrant_score']) >= minimum_score]
    else:
        pool = list(candidates)
    if not pool:
        return []

    raw_scores = [item.get('reranker_score') if item.get('reranker_score') is not None
                  else item.get('base_score', item['qdrant_score']) for item in pool]
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

    def safe_contextual_text(self, item):
        """Use context when it fits; fall back to exact source text otherwise."""
        value = contextual_text(item)
        tokenizer = self.embedder.model.tokenizer
        limits = [value for value in (
            getattr(self.embedder.model, 'max_seq_length', None),
            getattr(tokenizer, 'model_max_length', None),
        ) if isinstance(value, int) and 0 < value < 100000]
        limit = min(limits)
        count = len(tokenizer.encode(value, add_special_tokens=True, truncation=False))
        return value if count <= limit else item['text']

    def retrieve(self, question: str, *, candidate_k: int, top_k: int,
                 max_per_document: int, rerank: bool,
                 include_diagnostics: bool = False):
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
            terms = [term for term in re.findall(r'[A-Za-z0-9][A-Za-z0-9_./%-]{2,}', question)
                     if term.casefold() not in {
                         'what', 'which', 'where', 'when', 'how', 'does', 'were', 'are',
                         'the', 'and', 'for', 'from', 'with', 'about',
                     }]
            lexical_scores = self.database.lexical_search(
                conn, terms, config_id=self.config['config_id'], legacy=self.legacy,
                embedding_model=self.settings.embedding_model, limit=max(candidate_k * 3, 50))
            all_chunk_ids = list(dict.fromkeys(chunk_ids + list(lexical_scores)))
            rows = self.database.get_evidence(
                conn, all_chunk_ids, config_id=self.config['config_id'], legacy=self.legacy,
                embedding_model=self.settings.embedding_model)
        finally:
            conn.close()
        validated_ms = (time.perf_counter() - started) * 1000

        candidates = []
        original_candidates = []
        qdrant_by_id = {
            str((hit.payload or {}).get('chunk_id')): (rank, hit)
            for rank, hit in enumerate(hit_records, 1)
            if (hit.payload or {}).get('chunk_id')
        }
        candidate_ids = list(dict.fromkeys(chunk_ids + list(lexical_scores)))
        for fallback_rank, chunk_id in enumerate(candidate_ids, 1):
            original_rank, hit = qdrant_by_id.get(chunk_id, (candidate_k + fallback_rank, None))
            payload = hit.payload or {} if hit is not None else {}
            item = rows.get(chunk_id)
            if not item:
                continue
            if hit is not None and payload.get('sha') != item['expected_sha']:
                continue
            if (hit is not None and item['expected_version'] and
                    payload.get('index_version') != item['expected_version']):
                continue
            page_label = str(item['page_start']) if item['page_start'] == item['page_end'] else f"{item['page_start']}-{item['page_end']}"
            try:
                spans = json.loads(item['source_spans_json']) if item['source_spans_json'] else payload.get('source_spans', [])
            except (TypeError, json.JSONDecodeError):
                spans = payload.get('source_spans', [])
            item = {
                'evidence_id': f'E{len(candidates) + 1}', 'document_id': item['document_id'],
                'chunk_id': item['chunk_id'], 'text': item['text'], 'page_label': page_label,
                'page_start': item['page_start'], 'page_end': item['page_end'],
                'department': item['department'], 'circular': item['circular'],
                'issue_date': item['issue_date'], 'subject': item['subject'],
                'source_url': item['source_url'], 'source_start': item['source_start'],
                'source_end': item['source_end'], 'source_spans': spans,
                'retrieval_text': self.safe_contextual_text(item),
                'qdrant_score': float(hit.score) if hit is not None else 0.0,
                'lexical_score': float(lexical_scores.get(chunk_id, 0.0)),
                'dense_retrieved': hit is not None,
                'lexical_retrieved': chunk_id in lexical_scores,
                'original_rank': original_rank,
            }
            item['base_score'] = max(item['qdrant_score'], item['lexical_score'])
            candidates.append(item)
            original_candidates.append(item)
        candidates = deduplicate_candidates(candidates)
        document_scores = {}
        for item in candidates:
            document_scores[item['document_id']] = max(document_scores.get(item['document_id'], float('-inf')), item['qdrant_score'])
        for item in candidates:
            item['document_relevance'] = document_scores[item['document_id']]
        ranked = rerank_candidates(question, candidates, enabled=rerank, model_name=self.settings.reranker_model)
        for rank, item in enumerate(ranked, 1):
            item['reranked_rank'] = rank
        allow_same_document = any(word in question.casefold() for word in ('precaution', 'instruction', 'measure', 'guideline'))
        selection_started = time.perf_counter()
        passage_embeddings = self.embedder.encode_passages(
            [item.get('retrieval_text') or item['text'] for item in ranked])
        selected = coverage_aware_select(
            ranked, top_k=top_k, max_per_document=max_per_document,
            allow_same_document=allow_same_document,
            passage_embeddings=passage_embeddings,
            apply_score_gate=not rerank)
        timings = {
            'embedding_ms': round(encoded_ms, 2),
            'qdrant_ms': round(qdrant_ms - encoded_ms, 2),
            'sqlite_ms': round(validated_ms - qdrant_ms, 2),
            'ranking_ms': round((time.perf_counter() - started) * 1000 - validated_ms, 2),
            'selection_ms': round((time.perf_counter() - selection_started) * 1000, 2),
            'retrieval_total_ms': round((time.perf_counter() - started) * 1000, 2),
        }
        if not include_diagnostics:
            return selected, timings
        selected_by_chunk = {item['chunk_id']: item for item in selected}
        ranked_by_chunk = {item['chunk_id']: item for item in ranked}
        diagnostics = []
        for original in original_candidates:
            ranked_item = ranked_by_chunk.get(original['chunk_id'])
            selected_item = selected_by_chunk.get(original['chunk_id'])
            diagnostics.append({
                'chunk_id': original['chunk_id'], 'document_id': original['document_id'],
                'circular': original['circular'], 'page_label': original['page_label'],
                'original_rank': original['original_rank'], 'qdrant_score': original['qdrant_score'],
                'reranker_score': ranked_item.get('reranker_score') if ranked_item else None,
                'reranked_rank': ranked_item.get('reranked_rank') if ranked_item else None,
                'dense_retrieved': original.get('dense_retrieved', False),
                'lexical_retrieved': original.get('lexical_retrieved', False),
                'text': original['text'],
                'selected': selected_item is not None,
                'evidence_id': selected_item.get('evidence_id') if selected_item else None,
            })
        return selected, timings, diagnostics

