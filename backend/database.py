from __future__ import annotations

import json
import sqlite3
from typing import Any


class Database:
    """Creates short-lived SQLite connections; the database file is shared safely."""

    def __init__(self, path: str):
        self.path = path

    def connection(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=30000')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY, source_path TEXT NOT NULL UNIQUE,
            sha TEXT NOT NULL, filename TEXT, department TEXT,
            circular_number TEXT, issue_date TEXT, subject TEXT, summary TEXT,
            keywords_json TEXT, policy_category TEXT, referenced_circular_numbers_json TEXT,
            audience TEXT, actions_json TEXT, gr_text TEXT, extraction_source TEXT,
            official_source_verified INTEGER NOT NULL DEFAULT 0,
            github_url TEXT, official_pdf_url TEXT, pages INTEGER,
            indexed_at TEXT, metadata_version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id),
            page INTEGER NOT NULL, text TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS index_configs (
            config_id TEXT PRIMARY KEY, embedding_model TEXT NOT NULL,
            model_revision TEXT NOT NULL, chunking_strategy TEXT NOT NULL,
            chunk_size INTEGER NOT NULL, overlap INTEGER NOT NULL,
            prefix_mode TEXT NOT NULL, collection_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_index_status (
            document_id TEXT NOT NULL REFERENCES documents(document_id),
            config_id TEXT NOT NULL REFERENCES index_configs(config_id),
            status TEXT NOT NULL, source_sha TEXT, chunk_count INTEGER DEFAULT 0,
            error TEXT, indexed_at TEXT, index_version TEXT, embedding_model TEXT,
            PRIMARY KEY(document_id, config_id)
        );
        ''')
        document_columns = {
            'keywords_json':'TEXT', 'policy_category':'TEXT',
            'referenced_circular_numbers_json':'TEXT', 'audience':'TEXT', 'actions_json':'TEXT',
            'schemes_json':'TEXT', 'entities_json':'TEXT', 'numerical_facts_json':'TEXT', 'table_records_json':'TEXT',
            'gr_text':'TEXT', 'extraction_source':'TEXT',
            'official_source_verified':'INTEGER NOT NULL DEFAULT 0',
            'metadata_version':'INTEGER NOT NULL DEFAULT 1',
            'enrichment_status':"TEXT NOT NULL DEFAULT 'pending'",
            'enrichment_error':'TEXT', 'enrichment_attempts':'INTEGER NOT NULL DEFAULT 0',
            'enriched_at':'TEXT', 'enrichment_model':'TEXT',
            'index_status':"TEXT NOT NULL DEFAULT 'completed'",
            'index_version':'TEXT', 'embedding_model':'TEXT', 'extracted_pages':'INTEGER',
            'text_artifacts_json':'TEXT',
        }
        existing = {row[1] for row in db.execute('PRAGMA table_info(documents)')}
        for column, dtype in document_columns.items():
            if column not in existing:
                db.execute(f'ALTER TABLE documents ADD COLUMN {column} {dtype}')
        chunk_columns = {row[1] for row in db.execute('PRAGMA table_info(chunks)')}
        for column, dtype in {
            'index_config_id':'TEXT', 'chunk_index':'INTEGER', 'page_start':'INTEGER',
            'page_end':'INTEGER', 'section_heading':'TEXT', 'token_count':'INTEGER',
            'embedding_model':'TEXT', 'chunking_strategy':'TEXT',
            'source_start':'INTEGER', 'source_end':'INTEGER', 'source_spans_json':'TEXT',
            'retrieval_text':'TEXT',
        }.items():
            if column not in chunk_columns:
                db.execute(f'ALTER TABLE chunks ADD COLUMN {column} {dtype}')
        status_columns = {row[1] for row in db.execute('PRAGMA table_info(document_index_status)')}
        for column in ('index_version', 'embedding_model'):
            if column not in status_columns:
                db.execute(f'ALTER TABLE document_index_status ADD COLUMN {column} TEXT')
        db.execute('CREATE INDEX IF NOT EXISTS chunks_config_idx ON chunks(index_config_id, document_id)')
        db.commit()
        return db

    def get_evidence(self, conn: sqlite3.Connection, chunk_ids: list[str], *, config_id: str | None,
                     legacy: bool, embedding_model: str) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ','.join('?' for _ in chunk_ids)
        config_clause = 'c.index_config_id IS NULL' if legacy else 'c.index_config_id=?'
        config_params = [] if legacy else [config_id]
        sql = f'''SELECT c.chunk_id,c.document_id,COALESCE(c.page_start,c.page),
                         COALESCE(c.page_end,c.page),c.text,c.section_heading,
                         c.source_start,c.source_end,c.source_spans_json,
                         c.retrieval_text,
                         d.circular_number,d.department,d.issue_date,
                         d.github_url,d.official_pdf_url,d.subject,d.summary,
                         d.keywords_json,d.policy_category,d.referenced_circular_numbers_json,
                         d.audience,d.actions_json,d.schemes_json,d.entities_json,
                         d.numerical_facts_json,d.table_records_json,
                         d.index_version,d.sha,d.embedding_model,
                         s.source_sha,s.index_version
                  FROM chunks c JOIN documents d ON c.document_id=d.document_id
                  LEFT JOIN document_index_status s
                    ON s.document_id=d.document_id AND s.config_id=?
                  WHERE c.chunk_id IN ({placeholders}) AND {config_clause}
                    AND d.index_status='completed' AND (? OR s.embedding_model=?)
                    AND (? OR EXISTS (SELECT 1 FROM document_index_status s2
                                      WHERE s2.document_id=d.document_id AND s2.config_id=?
                                        AND s2.status='completed'))'''
        params = [config_id, *chunk_ids, *config_params,
                  legacy, embedding_model, legacy, config_id]
        rows = conn.execute(sql, params).fetchall()
        evidence = {}
        for row in rows:
            (chunk_id, document_id, page_start, page_end, text, heading,
             source_start, source_end, source_spans_json, retrieval_text, circular, department,
             issue_date, github, official, subject, summary, keywords_json,
             policy_category, references_json, audience, actions_json,
             schemes_json, entities_json, numerical_facts_json, table_records_json,
             version, sha, _, status_sha, status_version) = row
            def decode_json(value, default):
                try:
                    return json.loads(value) if value else default
                except (TypeError, json.JSONDecodeError):
                    return default
            expected_sha = status_sha if not legacy else sha
            expected_version = status_version if not legacy else version
            evidence[chunk_id] = {
                'chunk_id': chunk_id, 'document_id': document_id, 'page_start': page_start,
                'page_end': page_end, 'text': text, 'heading': heading,
                'retrieval_text': retrieval_text,
                'source_start': source_start, 'source_end': source_end,
                'source_spans_json': source_spans_json, 'circular': circular,
                'department': department, 'issue_date': issue_date,
                'source_url': official or github, 'subject': subject,
                'summary': summary, 'keywords': decode_json(keywords_json, []),
                'policy_category': policy_category,
                'referenced_circular_numbers': decode_json(references_json, []),
                'audience': audience, 'actions': decode_json(actions_json, []),
                'schemes': decode_json(schemes_json, []), 'entities': decode_json(entities_json, []),
                'numerical_facts': decode_json(numerical_facts_json, {}),
                'table_records': decode_json(table_records_json, []),
                'expected_sha': expected_sha, 'expected_version': expected_version,
            }
        return evidence

    def lexical_search(self, conn: sqlite3.Connection, terms: list[str], *, config_id: str | None,
                       legacy: bool, embedding_model: str, limit: int = 100) -> dict[str, float]:
        """Find exact-term candidates without requiring extracted metadata.

        Raw chunk text is always searched. Metadata columns are searched when
        present, so enrichment improves recall but is not required.
        """
        if not terms:
            return {}
        config_clause = 'c.index_config_id IS NULL' if legacy else 'c.index_config_id=?'
        params: list[object] = []
        predicates = []
        for term in terms[:12]:
            pattern = f'%{term}%'
            predicates.append('(c.text LIKE ? OR c.retrieval_text LIKE ? OR '
                              'd.circular_number LIKE ? OR d.subject LIKE ? OR '
                              'd.department LIKE ? OR d.filename LIKE ?)')
            params.extend([pattern] * 6)
        sql = f'''SELECT c.chunk_id, c.text, COALESCE(c.retrieval_text,''),
                         d.circular_number, d.subject, d.department, d.filename
                  FROM chunks c JOIN documents d ON c.document_id=d.document_id
                  LEFT JOIN document_index_status s
                    ON s.document_id=d.document_id AND s.config_id=?
                  WHERE {' OR '.join(predicates)} AND {config_clause}
                    AND d.index_status='completed' AND (? OR s.embedding_model=? )
                    AND (? OR EXISTS (SELECT 1 FROM document_index_status s2
                                      WHERE s2.document_id=d.document_id AND s2.config_id=?
                                        AND s2.status='completed') )
                  LIMIT ?'''
        query_params = [config_id, *params, *([] if legacy else [config_id]),
                        legacy, embedding_model, legacy, config_id, limit]
        rows = conn.execute(sql, query_params).fetchall()
        scores = {}
        for chunk_id, text, retrieval_text, circular, subject, department, filename in rows:
            searchable = ' '.join(str(value or '') for value in
                                  (text, retrieval_text, circular, subject, department, filename)).casefold()
            matched = sum(1 for term in terms if term.casefold() in searchable)
            if matched:
                scores[chunk_id] = matched / max(1, len(terms))
        return scores

