# FastAPI evaluation report: end-to-end

- API: `http://127.0.0.1:8000/api/v1/query`
- Cases: 100; successful requests: 0; failed requests: 100
- Warmup: False; measured runs per case: 1

## Effective configuration

```json
{
  "candidate_k": 30,
  "top_k": 6,
  "max_per_document": 7,
  "rerank": true,
  "observed_response_parameters": []
}
```

## Metric definitions

| Metric | Definition |
|---|---|
| document_recall_at_30 | Fraction of required documents in the original pre-rerank candidate diagnostics, limited to rank 30. |
| candidate_evidence_recall_at_30 | Fraction of verified evidence specifications matched by the fused candidate diagnostics through rank 30. |
| document_all_retrieved_at_30 | True only when every required document is in the original candidate set; null for empty gold. |
| evidence_recall_at_6 | Fraction of verified evidence specifications matched by selected API evidence; document membership alone is insufficient. |
| precision_at_6 | Fraction of selected API passages matching verified evidence specifications. |
| mrr | Reciprocal rank of the first required document in the original pre-rerank candidate order. |
| fact_coverage | Fraction of evidence-linked facts whose typed value appears in the matched selected passage. |

## Retrieval metrics

| Metric | Value |
|---|---:|
| document_recall_at_30 | unavailable |
| candidate_evidence_recall_at_30 | unavailable |
| document_all_retrieved_at_30 | unavailable |
| document_partial_coverage_at_30 | unavailable |
| evidence_recall_at_6 | unavailable |
| evidence_all_retrieved_at_6 | unavailable |
| precision_at_6 | unavailable |
| mrr | unavailable |
| fact_coverage | unavailable |

## Answer metrics

| Metric | Value |
|---|---:|
| answer_fact_coverage | unavailable |
| numerical_exact_match_accuracy | unavailable |
| citation_id_validity | unavailable |
| citation_factual_support | unavailable |
| unsupported_claim_rate | unavailable |
| abstention_accuracy | unavailable |

## Latency (ms)

| Stage | Mean | p50 | p95 |
|---|---:|---:|---:|
| embedding_ms | unavailable | unavailable | unavailable |
| qdrant_ms | unavailable | unavailable | unavailable |
| ranking_ms | unavailable | unavailable | unavailable |
| selection_ms | unavailable | unavailable | unavailable |
| gemini_ms | unavailable | unavailable | unavailable |
| retrieval_total_ms | unavailable | unavailable | unavailable |
| client_total_ms | unavailable | unavailable | unavailable |

## Metrics by department_group

| Group | Cases | Macro evidence recall@6 | Macro fact coverage | Micro evidence recall@6 |
|---|---:|---:|---:|---:|

## Metrics by question_type

| Group | Cases | Macro evidence recall@6 | Macro fact coverage | Micro evidence recall@6 |
|---|---:|---:|---:|---:|

## Limitations

- Answer correctness is unavailable without an independent judge.
- API failures are reported separately and are not converted into retrieval misses.
- Retrieval metrics require diagnostics enabled on the FastAPI server.
