import unittest

from backend import app as app_module
from backend.schemas import QueryRequest


class FakeSettings:
    evaluation_mode_enabled = True
    rerank_enabled = False


class FakeRetrieval:
    def __init__(self):
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs))
        evidence = [{"evidence_id": "E1", "document_id": "D1", "circular": "GR-1",
                     "issue_date": None, "department": "Dept", "page_label": "2",
                     "source_url": None, "qdrant_score": .9, "document_relevance": .9,
                     "reranker_score": .2, "text": "grounded passage"}]
        timings = {"embedding_ms": 1.0, "qdrant_ms": 2.0, "sqlite_ms": 1.0,
                   "ranking_ms": 3.0, "selection_ms": 1.0, "retrieval_total_ms": 8.0}
        if kwargs["include_diagnostics"]:
            return evidence, timings, [{"chunk_id": "C1", "document_id": "D1", "circular": "GR-1",
                                        "page_label": "2", "original_rank": 1, "qdrant_score": .9,
                                        "reranker_score": .2, "selected": True, "evidence_id": "E1"}]
        return evidence, timings


class FakeGemini:
    def answer(self, question, evidence):
        return {"answer": "ok", "claims": [], "limitations": [], "answer_status": "insufficient_evidence"}


class FakeState:
    def __init__(self):
        self.settings = FakeSettings()
        self.lock = __import__("threading").Lock()
        self.retrieval = FakeRetrieval()
        self.gemini = FakeGemini()


class BackendApiTests(unittest.TestCase):
    def test_retrieval_only_uses_same_retrieval_service_and_skips_gemini(self):
        state = FakeState()
        result = app_module._query(state, QueryRequest(question="Q", candidate_k=30, top_k=6,
                                                        max_per_document=7, rerank=True,
                                                        generate_answer=False,
                                                        evaluation_diagnostics=True,
                                                        show_evidence=True))
        self.assertEqual(state.retrieval.calls[0][1]["candidate_k"], 30)
        self.assertTrue(state.retrieval.calls[0][1]["include_diagnostics"])
        self.assertEqual(result.answer.answer_status, "not_run")
        self.assertEqual(result.effective_parameters["max_per_document"], 7)
        self.assertEqual(result.diagnostics[0].original_rank, 1)

    def test_default_query_still_generates_answer_without_diagnostics(self):
        state = FakeState()
        result = app_module._query(state, QueryRequest(question="Q", show_evidence=True))
        self.assertTrue(result.answer_generation_enabled)
        self.assertIsNone(result.diagnostics)
        self.assertFalse(state.retrieval.calls[0][1]["include_diagnostics"])


if __name__ == "__main__":
    unittest.main()
