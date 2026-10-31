# Evaluation report: baseline_dense_rerank_mmr_30_6_7

This Phase 0 report measures retrieval only. Gemini answer-quality metrics were not run.

## Configuration

```json
{
  "name": "baseline_dense_rerank_mmr_30_6_7",
  "candidate_k": 30,
  "top_k": 6,
  "max_per_document": 7,
  "rerank": true,
  "chunk_size": 220,
  "overlap": 30,
  "index_config_id": "21e74095355b2dbbd1185ed3",
  "config_id": "21e74095355b2dbbd1185ed3",
  "collection_name": "maharashtra-gr-chunks-sentence-transformers-all-minilm-l6-v2-21e74095355b",
  "model_revision": "unspecified",
  "prefix_mode": "none"
}
```

## Retrieval metrics

| Metric | Value |
|---|---:|
| document_recall_at_30 | 1 |
| evidence_recall_at_6 | 1 |
| mrr | 1.0 |
| precision_at_6 | 0.3333333333333333 |
| fact_coverage | 0.19583333333333333 |
| page_coverage | 1 |
| expected_evidence_coverage | unavailable |

## Latency (milliseconds)

| Stage | Mean | p50 | p95 |
|---|---:|---:|---:|
| embedding_ms | 7.54 | 7.50 | 7.78 |
| vector_ms | 1.47 | 1.38 | 1.80 |
| reranking_ms | 390.50 | 365.72 | 472.97 |
| evidence_selection_ms | 373.88 | 357.90 | 420.51 |
| total_ms | 1015.04 | 988.08 | 1079.46 |

## Per-case results

| Case | Doc R@30 | Evidence R@6 | MRR | P@6 | Fact coverage | Page coverage |
|---|---:|---:|---:|---:|---:|---:|
| eval_001 | 1 | 1 | 1.000 | 0.667 | 0.200 | n/a |
| eval_002 | 1 | 1 | 1.000 | 0.333 | 0.250 | 1 |
| eval_003 | 1 | 1 | 1.000 | 0.167 | 0.000 | 1 |
| eval_004 | 1 | 1 | 1.000 | 0.167 | 0.333 | n/a |
| eval_005 | 0 | 0 | 0.000 | 0.000 | 0.000 | n/a |

## Limitations

- Answer correctness, citation support, unsupported-claim rate, and abstention accuracy are unavailable because Gemini was not called.
- This benchmark uses the documents currently present in the local SQLite/Qdrant indexes; it does not claim full 33-department coverage.
- Cold-start model loading is not included in these per-query timings; repeated runs after warm-up are reported separately in JSON.
