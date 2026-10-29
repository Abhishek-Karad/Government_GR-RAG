from __future__ import annotations

import hashlib
import json
import os
import re

COLLECTION = os.getenv('QDRANT_COLLECTION', 'maharashtra_gr_chunks')
CHUNKING_STRATEGY = 'structure-aware-v2'


def model_slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')[-42:] or 'model'


def embedding_prefix_mode(model_name):
    configured = os.getenv('EMBEDDING_PREFIX_MODE', 'auto').strip().lower()
    if configured != 'auto':
        return configured
    return 'e5' if re.search(r'(^|[-/])e5($|[-/])', model_name.lower().replace('_', '-')) else 'none'


def effective_max_tokens(embedding_model):
    values = []
    for value in (getattr(embedding_model, 'max_seq_length', None),
                  getattr(getattr(embedding_model, 'tokenizer', None), 'model_max_length', None)):
        if isinstance(value, int) and 0 < value < 100000:
            values.append(value)
    if not values:
        raise ValueError('Embedding model does not expose a usable maximum sequence length')
    return min(values)


def get_index_config(db, embedding_model, settings, *, create=False):
    size = min(settings.chunk_size, effective_max_tokens(embedding_model))
    mode = embedding_prefix_mode(settings.embedding_model)
    revision = os.getenv('EMBEDDING_MODEL_REVISION', 'unspecified')
    identity = json.dumps({
        'model': settings.embedding_model, 'revision': revision,
        'strategy': CHUNKING_STRATEGY, 'chunk_size': size,
        'overlap': settings.overlap, 'prefix_mode': mode,
    }, sort_keys=True, separators=(',', ':'))
    config_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
    base = os.getenv('QDRANT_COLLECTION_PREFIX', COLLECTION)
    collection = f'{model_slug(base)}-{model_slug(settings.embedding_model)}-{config_id[:12]}'
    row = db.execute('SELECT config_id,collection_name FROM index_configs WHERE config_id=?', (config_id,)).fetchone()
    if row:
        return {'config_id': row[0], 'collection_name': row[1]}
    if not create:
        return None
    with db:
        db.execute('''INSERT INTO index_configs
            (config_id,embedding_model,model_revision,chunking_strategy,chunk_size,overlap,prefix_mode,collection_name,created_at)
            VALUES (?,?,?,?,?,?,?,?,datetime('now'))''',
            (config_id, settings.embedding_model, revision, CHUNKING_STRATEGY, size,
             settings.overlap, mode, collection))
    return {'config_id': config_id, 'collection_name': collection}

