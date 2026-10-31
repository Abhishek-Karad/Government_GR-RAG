# FastAPI evaluation report: retrieval

- API: `http://127.0.0.1:8000/api/v1/query`
- Cases: 9; successful requests: 9; failed requests: 0
- Warmup: True; measured runs per case: 1

## Effective configuration

```json
{
  "candidate_k": 30,
  "top_k": 6,
  "max_per_document": 7,
  "rerank": true,
  "observed_response_parameters": [
    "{\"candidate_k\": 30, \"max_per_document\": 7, \"rerank\": true, \"top_k\": 6}"
  ]
}
```

## Metric definitions

| Metric | Definition |
|---|---|
| document_recall_at_30 | Fraction of required documents in the original pre-rerank candidate diagnostics, limited to rank 30. |
| document_all_retrieved_at_30 | True only when every required document is in the original candidate set; null for empty gold. |
| evidence_recall_at_6 | Fraction of verified evidence specifications matched by selected API evidence; document membership alone is insufficient. |
| precision_at_6 | Fraction of selected API passages matching verified evidence specifications. |
| mrr | Reciprocal rank of the first required document in the original pre-rerank candidate order. |
| fact_coverage | Fraction of evidence-linked facts whose typed value appears in the matched selected passage. |

## Retrieval metrics

| Metric | Value |
|---|---:|
| document_recall_at_30 | 1.0 |
| document_all_retrieved_at_30 | 1 |
| document_partial_coverage_at_30 | 1.0 |
| evidence_recall_at_6 | 0.15 |
| evidence_all_retrieved_at_6 | 0.125 |
| precision_at_6 | 0.041666666666666664 |
| mrr | 0.6637759170653907 |
| fact_coverage | 0.15 |

## Answer metrics

| Metric | Value |
|---|---:|
| answer_fact_coverage | unavailable |
| citation_id_validity | unavailable |
| citation_factual_support | unavailable |
| unsupported_claim_rate | unavailable |
| abstention_accuracy | unavailable |

## Latency (ms)

| Stage | Mean | p50 | p95 |
|---|---:|---:|---:|
| embedding_ms | 7.582222222222223 | 6.59 | 11.893999999999998 |
| qdrant_ms | 1.7622222222222221 | 1.51 | 2.9079999999999995 |
| ranking_ms | 506.51222222222225 | 500.02 | 613.16 |
| selection_ms | 114.06444444444445 | 112.44 | 125.00999999999999 |
| gemini_ms | 0.0 | 0.0 | 0.0 |
| retrieval_total_ms | 517.6633333333333 | 508.75 | 623.102 |
| client_total_ms | 519.6490137774769 | 510.3288749996864 | 625.9127165994869 |

## Limitations

- Answer correctness is unavailable without an independent judge.
- API failures are reported separately and are not converted into retrieval misses.
- Retrieval metrics require diagnostics enabled on the FastAPI server.
