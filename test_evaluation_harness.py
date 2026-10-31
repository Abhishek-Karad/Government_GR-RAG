import os
import unittest

from evaluation_harness import ApiEvaluationError, FastApiClient, metric_for_case, value_matches


def spec(eid, doc, text, page=1):
    return {"id": eid, "document_id": doc, "page": page, "text_contains": text}


def response(diagnostics, evidence):
    return {"diagnostics": diagnostics, "evidence": evidence}


def diagnostic(doc, rank=1, chunk="C1"):
    return {"chunk_id": chunk, "document_id": doc, "circular": None,
            "page_label": "1", "original_rank": rank, "qdrant_score": .9,
            "reranker_score": .1, "selected": False, "evidence_id": None}


def passage(doc, text, evidence_id="E1"):
    return {"evidence_id": evidence_id, "document_id": doc, "page_label": "1",
            "text": text, "circular": None}


class EvaluationMetricsTests(unittest.TestCase):
    def test_multiple_required_documents_reports_partial_and_all_coverage(self):
        case = {"id": "multi", "answerable": True,
                "expected_evidence": [spec("a", "D1", "alpha"), spec("b", "D2", "beta")]}
        result = metric_for_case(case, response([diagnostic("D1")], [passage("D1", "alpha")]))
        self.assertEqual(result["document_recall_at_30"], .5)
        self.assertFalse(result["document_all_retrieved_at_30"])
        self.assertEqual(result["evidence_recall_at_6"], .5)

    def test_correct_document_but_wrong_passage_is_not_evidence(self):
        case = {"id": "wrong", "answerable": True,
                "expected_evidence": [spec("x", "D1", "required fact")],
                "expected_facts": [{"value": "required fact", "type": "text", "evidence_ids": ["x"]}]}
        result = metric_for_case(case, response([diagnostic("D1")], [passage("D1", "unrelated")]))
        self.assertEqual(result["document_recall_at_30"], 1.0)
        self.assertEqual(result["evidence_recall_at_6"], 0.0)
        self.assertEqual(result["fact_coverage"], 0.0)

    def test_equivalent_numeric_and_date_formatting(self):
        self.assertTrue(value_matches("4712.98", "Rs. 4,712.98 lakh", "number"))
        self.assertTrue(value_matches("2020-10-05", "October 5, 2020", "date"))
        self.assertTrue(value_matches("2020-10-19", "19th October 2020", "date"))

    def test_missing_expected_evidence_is_unavailable(self):
        case = {"id": "missing", "answerable": True, "expected_facts": [{"value": "known", "type": "text"}]}
        result = metric_for_case(case, response([diagnostic("D1")], [passage("D1", "known")]))
        self.assertIsNone(result["evidence_recall_at_6"])
        self.assertIsNone(result["fact_coverage"])

    def test_empty_gold_document_set_is_unavailable_not_a_miss(self):
        case = {"id": "empty", "answerable": True}
        result = metric_for_case(case, response([], []))
        self.assertIsNone(result["document_recall_at_30"])
        self.assertIsNone(result["mrr"])

    def test_incomplete_evidence_for_multi_fact_question(self):
        case = {"id": "incomplete", "answerable": True,
                "expected_evidence": [spec("a", "D1", "first"), spec("b", "D1", "second")],
                "expected_facts": [{"value": "first", "type": "text", "evidence_ids": ["a"]},
                                    {"value": "second", "type": "text", "evidence_ids": ["b"]}]}
        result = metric_for_case(case, response([diagnostic("D1")], [passage("D1", "first")]))
        self.assertEqual(result["evidence_recall_at_6"], .5)
        self.assertEqual(result["fact_coverage"], .5)

    def test_unanswerable_is_excluded_and_abstention_not_scored_in_retrieval(self):
        case = {"id": "unknown", "answerable": False}
        result = metric_for_case(case, response([diagnostic("D1")], [passage("D1", "x")]))
        self.assertEqual(result["evaluation_status"], "unanswerable_excluded")
        self.assertIsNone(result["document_recall_at_30"])


class FakeResponse:
    def __init__(self, status=200, body=None, json_error=False):
        self.status_code = status; self.body = body; self.json_error = json_error

    def json(self):
        if self.json_error:
            raise ValueError("bad json")
        return self.body


class ApiClientTests(unittest.TestCase):
    def test_client_parses_successful_response_and_sends_explicit_config(self):
        client = FastApiClient("http://test", timeout=1, retries=0)
        calls = []
        client.session.post = lambda url, **kwargs: (calls.append((url, kwargs)) or FakeResponse(body={"evidence": [], "diagnostics": [], "timings_ms": {}}))
        body = client.query("Q", candidate_k=30, top_k=6, max_per_document=7, rerank=True, generate_answer=False, diagnostics=True)
        self.assertEqual(body["request_parameters"]["candidate_k"], 30)
        self.assertEqual(calls[0][1]["json"]["generate_answer"], False)

    def test_client_rejects_invalid_json_and_http_errors(self):
        client = FastApiClient("http://test", timeout=1, retries=0)
        client.session.post = lambda *a, **k: FakeResponse(json_error=True)
        with self.assertRaises(ApiEvaluationError):
            client.query("Q", candidate_k=30, top_k=6, max_per_document=7, rerank=True, generate_answer=False, diagnostics=True)
        client.session.post = lambda *a, **k: FakeResponse(status=422, body={"detail": "bad request"})
        with self.assertRaises(ApiEvaluationError):
            client.query("Q", candidate_k=30, top_k=6, max_per_document=7, rerank=True, generate_answer=False, diagnostics=True)


@unittest.skipUnless(os.getenv("RUN_FASTAPI_INTEGRATION"), "set RUN_FASTAPI_INTEGRATION=1 for live FastAPI integration")
class FastApiIntegrationTest(unittest.TestCase):
    def test_query_endpoint_is_reachable(self):
        from fastapi.testclient import TestClient
        from backend.app import app
        with TestClient(app) as client:
            response = client.post('/api/v1/query', json={
                'question': 'How many bridge works were approved in Wardha?',
                'candidate_k': 30, 'top_k': 6, 'max_per_document': 7,
                'rerank': True, 'show_evidence': True,
                'generate_answer': False, 'evaluation_diagnostics': True,
            })
            self.assertIn(response.status_code, (200, 403, 500))


if __name__ == "__main__":
    unittest.main()
