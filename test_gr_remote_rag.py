import importlib.util
import sqlite3
import sys
import types
import unittest
import time
from collections import UserDict
from argparse import Namespace
from pathlib import Path


def load_app():
    """Load the script without requiring heavyweight runtime packages for unit tests."""
    numpy = types.ModuleType('numpy')
    requests = types.ModuleType('requests')
    requests.Session = object
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = lambda: None
    sys.modules.setdefault('numpy', numpy)
    sys.modules.setdefault('requests', requests)
    sys.modules.setdefault('dotenv', dotenv)
    spec = importlib.util.spec_from_file_location('gr_remote_rag_test_target', Path(__file__).with_name('gr_remote_rag.py'))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


app = load_app()


class FakeTokenizer:
    model_max_length = 32

    def encode(self, text, add_special_tokens=True, truncation=False):
        values = text.split()
        return ([0] if add_special_tokens else []) + list(range(1, len(values) + 1)) + ([0] if add_special_tokens else [])

    def decode(self, tokens, skip_special_tokens=True):
        return ' '.join(f'token{value}' for value in tokens if value)


class FakeEmbedding:
    max_seq_length = 32
    tokenizer = FakeTokenizer()


class FakeSubwordTokenizer:
    """A fast-tokenizer-shaped fake that exposes subword/offset behavior."""
    model_max_length = 256

    @staticmethod
    def _pieces(text):
        pieces = []
        for word in text.split():
            pieces.extend([word] * max(1, (len(word) + 5) // 6))
        return pieces

    def encode(self, text, add_special_tokens=True, truncation=False):
        values = self._pieces(text)
        return ([0] if add_special_tokens else []) + list(range(1, len(values) + 1)) + ([0] if add_special_tokens else [])

    def __call__(self, text, add_special_tokens=False, truncation=False, return_offsets_mapping=False):
        offsets = []
        for match in __import__('re').finditer(r'\S+', text):
            count = max(1, (len(match.group(0)) + 5) // 6)
            offsets.extend([(match.start(), match.end())] * count)
        return UserDict(offset_mapping=offsets)


class FakeSubwordEmbedding:
    max_seq_length = 256
    tokenizer = FakeSubwordTokenizer()


class ChunkerTests(unittest.TestCase):
    def setUp(self):
        app.EMBEDDING_PREFIX_MODE = 'auto'
        app.EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

    def test_minilm_is_raw_and_e5_is_prefixed(self):
        self.assertEqual(app.embedding_input('text', 'passage'), 'text')
        app.EMBED_MODEL = 'intfloat/multilingual-e5-small'
        self.assertEqual(app.embedding_input('text', 'passage'), 'passage: text')
        self.assertEqual(app.embedding_input('question', 'query'), 'query: question')

    def test_paragraphs_and_numbered_clauses_are_preserved(self):
        chunker = app.SemanticDocumentChunker(FakeEmbedding(), max_tokens=20, overlap=2)
        chunks = chunker.chunk([(1, 'Eligibility\n\n1. First condition remains together.\n\n2. Second condition remains together.')])
        self.assertEqual(len(chunks), 1)
        self.assertIn('1. First condition', chunks[0]['text'])
        self.assertIn('2. Second condition', chunks[0]['text'])
        self.assertEqual(chunks[0]['page_start'], 1)
        self.assertLessEqual(chunks[0]['token_count'], 20)

    def test_long_clause_respects_model_limit(self):
        chunker = app.SemanticDocumentChunker(FakeEmbedding(), max_tokens=10, overlap=2)
        chunks = chunker.chunk([(3, '1. ' + 'important ' * 30)])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c['token_count'] <= 10 for c in chunks))
        self.assertTrue(all(c['page_start'] == c['page_end'] == 3 for c in chunks))

    def test_exact_punctuation_and_clause_text_are_preserved(self):
        source = '1. Keep ₹1,000.00 exactly!\n\n2. Do not rewrite—ever.'
        chunks = app.SemanticDocumentChunker(FakeEmbedding(), max_tokens=32).chunk([(2, source, 10, 10 + len(source))])
        self.assertIn('₹1,000.00', ''.join(c['text'] for c in chunks))
        self.assertIn('rewrite—ever.', ''.join(c['text'] for c in chunks))
        self.assertTrue(all(source[c['source_start'] - 10:c['source_end'] - 10] == c['text'] for c in chunks))

    def test_page_boundaries_and_spans_are_retained(self):
        source = '# Page 1\nFirst page.\n# Page 2\nSecond page.'
        pages = app.pages_with_offsets(source)
        chunks = app.SemanticDocumentChunker(FakeEmbedding(), max_tokens=32).chunk(pages)
        self.assertEqual(chunks[0]['page_start'], 1)
        self.assertEqual(chunks[-1]['page_end'], 2)
        self.assertTrue(any(c['page_end'] == 2 for c in chunks))
        for chunk in chunks:
            for start, end in chunk['source_spans']:
                self.assertIn(source[start:end].strip(), chunk['text'])

    def test_translation_and_ocr_artifacts_are_reported_without_rewriting(self):
        source = "family members ' s masks\n| table | row |\n-----"
        self.assertEqual(app.detect_text_artifacts(source), [
            'split_punctuation', 'ascii_table_or_ocr_grid', 'separator_line'
        ])
        self.assertEqual(source, "family members ' s masks\n| table | row |\n-----")

    def test_long_subword_paragraphs_are_token_safe_and_source_preserving(self):
        app.EMBEDDING_PREFIX_MODE = 'none'
        tokenizer = FakeSubwordTokenizer()
        for word_count in (100, 220, 256, 500, 750, 1300, 3000):
            source = 'governmentresolution ' * word_count
            chunker = app.SemanticDocumentChunker(
                FakeSubwordEmbedding(), max_tokens=220, overlap=30)
            chunks = chunker.chunk([(7, source, 0, len(source))])
            self.assertGreaterEqual(len(chunks), 1)
            for chunk in chunks:
                exact = app.embedding_input(chunk['embedding_text'], 'passage')
                self.assertLessEqual(app.token_count(tokenizer, exact), 220)
                self.assertEqual(source[chunk['source_start']:chunk['source_end']], chunk['text'])
                self.assertEqual(chunk['page_start'], 7)
            # The union of source spans contains every original character;
            # overlap is intentional and is bounded by the configured window.
            covered = [False] * len(source)
            for chunk in chunks:
                for position in range(chunk['source_start'], chunk['source_end']):
                    covered[position] = True
            self.assertTrue(all(covered[position] for position, value in enumerate(source)
                                if not value.isspace()))

    def test_e5_prefix_and_special_tokens_are_included_in_validation(self):
        app.EMBED_MODEL = 'intfloat/multilingual-e5-small'
        app.EMBEDDING_PREFIX_MODE = 'auto'
        source = 'governmentresolution ' * 500
        chunks = app.SemanticDocumentChunker(
            FakeSubwordEmbedding(), max_tokens=220, overlap=30).chunk([(3, source, 0, len(source))])
        for chunk in chunks:
            formatted = app.embedding_input(chunk['embedding_text'], 'passage')
            self.assertLessEqual(app.token_count(FakeSubwordTokenizer(), formatted), 256)
            self.assertTrue(formatted.startswith('passage: '))

    def test_malformed_tables_headers_pages_and_empty_documents_are_safe(self):
        app.EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
        app.EMBEDDING_PREFIX_MODE = 'none'
        source = ('# Page 1\nRepeated Header\n\n| col | value |\n|---|---|\n' +
                  ('OCR   text  with   irregular spacing.\n' * 100) +
                  '\n# Page 2\nRepeated Header\n\n1. ' + ('legal-clause ' * 300))
        pages = app.pages_with_offsets(source)
        chunks = app.SemanticDocumentChunker(
            FakeSubwordEmbedding(), max_tokens=220, overlap=30).chunk(pages)
        self.assertTrue(any(chunk['page_start'] == 1 for chunk in chunks))
        self.assertTrue(any(chunk['page_start'] == 2 for chunk in chunks))
        for chunk in chunks:
            self.assertEqual(source[chunk['source_start']:chunk['source_end']], chunk['text'])
            self.assertLessEqual(chunk['token_count'], 220)
        self.assertEqual(app.SemanticDocumentChunker(FakeSubwordEmbedding()).chunk([]), [])


class ConfigurationTests(unittest.TestCase):
    def test_same_dimension_models_get_different_configurations(self):
        db = sqlite3.connect(':memory:')
        # Create the minimum schema needed by get_index_config.
        db.execute('''CREATE TABLE index_configs (
            config_id TEXT PRIMARY KEY, embedding_model TEXT, model_revision TEXT,
            chunking_strategy TEXT, chunk_size INTEGER, overlap INTEGER,
            prefix_mode TEXT, collection_name TEXT, created_at TEXT)''')
        args = Namespace(chunk_size=220, overlap=40)
        app.EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
        first = app.get_index_config(db, FakeEmbedding(), args)
        app.EMBED_MODEL = 'intfloat/multilingual-e5-small'
        second = app.get_index_config(db, FakeEmbedding(), args)
        self.assertNotEqual(first['config_id'], second['config_id'])
        self.assertNotEqual(first['collection_name'], second['collection_name'])


class AnswerPromptTests(unittest.TestCase):
    def test_prompt_contains_question_sources_and_grounding_rules(self):
        prompt = app.build_answer_prompt('Who is eligible?', [{
            'evidence_id': 'E1',
            'text': 'Only rural households are eligible.',
            'page_label': '4',
            'department': 'Rural Development',
            'circular': 'GR-7',
            'subject': 'Housing support',
        }])
        self.assertIn('Who is eligible?', prompt)
        self.assertIn('[E1]', prompt)
        self.assertIn('Only rural households are eligible.', prompt)
        self.assertIn('Do not invent facts', prompt)

    def test_invalid_citation_is_rejected_and_unsupported_claim_is_marked(self):
        claim = types.SimpleNamespace(text='Masks were compulsory for every person.', evidence_ids=['E9'], verified=False)
        payload = types.SimpleNamespace(answer='x', claims=[claim], limitations=[], answer_status='supported')
        evidence = [{'evidence_id': 'E1', 'text': 'Masks were recommended for families.'}]
        with self.assertRaises(app.CitationValidationError):
            app.validate_answer_payload(payload, evidence)
        claim.evidence_ids = ['E1']
        validated = app.validate_answer_payload(payload, evidence)
        self.assertEqual(validated.answer_status, 'insufficient_evidence')
        self.assertFalse(claim.verified)

    def test_empty_evidence_is_insufficient(self):
        result = app.gemini_answer('What?', [])
        self.assertEqual(result['answer_status'], 'insufficient_evidence')


class RankingTests(unittest.TestCase):
    def test_cross_encoder_order_is_preserved(self):
        class FakeReranker:
            def predict(self, pairs):
                return [0.1, 0.9]
        old = app._RERANKER
        app._RERANKER = FakeReranker()
        try:
            items = app.rerank_candidates('question', [
                {'evidence_id': 'E1', 'text': 'a', 'qdrant_score': .9},
                {'evidence_id': 'E2', 'text': 'b', 'qdrant_score': .8},
            ])
            self.assertEqual([x['evidence_id'] for x in items], ['E2', 'E1'])
        finally:
            app._RERANKER = old

    def test_deduplication_and_document_diversification(self):
        items = [
            {'evidence_id': 'E1', 'document_id': 'D1', 'text': 'same text', 'qdrant_score': .9},
            {'evidence_id': 'E2', 'document_id': 'D1', 'text': 'same text', 'qdrant_score': .8},
            {'evidence_id': 'E3', 'document_id': 'D1', 'text': 'different text', 'qdrant_score': .7},
            {'evidence_id': 'E4', 'document_id': 'D2', 'text': 'other circular', 'qdrant_score': .6},
        ]
        unique = app.deduplicate_candidates(items)
        self.assertEqual(len(unique), 3)
        selected = app.diversify_candidates(unique, top_k=3, max_per_document=1, allow_same_document=True)
        self.assertEqual([x['document_id'] for x in selected], ['D1', 'D2', 'D1'])

        irrelevant = dict(items[3], evidence_id='E5', document_id='D3', qdrant_score=.1)
        self.assertNotIn(irrelevant, app.diversify_candidates(unique + [irrelevant], top_k=5))

    def test_mask_precautions_retain_coverage_after_reranking(self):
        # The first four passages are from one circular and cover distinct
        # facts.  The duplicate regular-use passage is ranked above cleaning,
        # while an unrelated circular has a low Qdrant score.
        candidates = [
            {'evidence_id': 'E1', 'document_id': 'MASK-GR',
             'text': 'Regular mask use is required in public places.',
             'qdrant_score': .95, 'reranker_score': .99},
            {'evidence_id': 'E2', 'document_id': 'MASK-GR',
             'text': 'Masks are required inside vehicles.',
             'qdrant_score': .92, 'reranker_score': .96},
            {'evidence_id': 'E3', 'document_id': 'MASK-GR',
             'text': 'Clean and reuse cloth masks after washing them.',
             'qdrant_score': .86, 'reranker_score': .78},
            {'evidence_id': 'E4', 'document_id': 'MASK-GR',
             'text': 'Do not share masks with another person.',
             'qdrant_score': .84, 'reranker_score': .82},
            {'evidence_id': 'E5', 'document_id': 'MASK-GR',
             'text': 'Regular mask use is required in public places and offices.',
             'qdrant_score': .90, 'reranker_score': .90},
            {'evidence_id': 'E6', 'document_id': 'OTHER-GR',
             'text': 'A crop subsidy application must include land records.',
             'qdrant_score': .20, 'reranker_score': .97},
        ]
        vectors = [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, 1], [.99, .01, 0, 0], [0, 0, 0, 1],
        ]
        started = time.perf_counter()
        selected = app.coverage_aware_select(
            candidates, top_k=5, max_per_document=5, allow_same_document=True,
            passage_embeddings=vectors)
        elapsed_ms = (time.perf_counter() - started) * 1000
        text = ' '.join(item['text'].casefold() for item in selected)

        self.assertIn('regular mask', text)
        self.assertIn('vehicle', text)
        self.assertIn('clean and reuse', text)
        self.assertIn('share masks', text)
        self.assertNotIn('crop subsidy', text)
        # MMR does not rewrite or renumber the evidence records.
        self.assertIn('E3', [item['evidence_id'] for item in selected])
        self.assertEqual(selected[0]['qdrant_score'], .95)
        self.assertLess(elapsed_ms, 1000)

        # A small deterministic benchmark: the same selector is run with and
        # without cross-encoder scores.  Coverage is recall of the four known
        # mask facts; elapsed times are the selection-only latency.
        baseline_candidates = [dict(item, reranker_score=None) for item in candidates]
        baseline_started = time.perf_counter()
        baseline = app.coverage_aware_select(
            baseline_candidates, top_k=5, max_per_document=5,
            allow_same_document=True, passage_embeddings=vectors)
        baseline_ms = (time.perf_counter() - baseline_started) * 1000
        expected = ('regular mask', 'vehicle', 'clean and reuse', 'share masks')
        coverage_on = sum(any(term in item['text'].casefold() for item in selected)
                          for term in expected)
        coverage_off = sum(any(term in item['text'].casefold() for item in baseline)
                           for term in expected)
        self.assertGreaterEqual(coverage_on, coverage_off)
        self.assertEqual(coverage_on, 4)
        self.assertGreater(elapsed_ms, 0)
        self.assertGreater(baseline_ms, 0)

    def test_reranking_can_be_benchmarked_without_changing_selector(self):
        candidates = [
            {'evidence_id': 'E1', 'document_id': 'D1', 'text': 'mask use',
             'qdrant_score': .9},
            {'evidence_id': 'E2', 'document_id': 'D1', 'text': 'mask cleaning',
             'qdrant_score': .8},
        ]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        old = app._RERANKER

        class FakeReranker:
            def predict(self, pairs):
                return [.9, .7]

        app._RERANKER = FakeReranker()
        try:
            ranked_on = app.rerank_candidates('mask precautions', candidates, enabled=True)
            ranked_off = app.rerank_candidates('mask precautions', candidates, enabled=False)
            selected_on = app.coverage_aware_select(
                ranked_on, top_k=2, max_per_document=2, allow_same_document=True,
                passage_embeddings=vectors)
            selected_off = app.coverage_aware_select(
                ranked_off, top_k=2, max_per_document=2, allow_same_document=True,
                passage_embeddings=vectors)
        finally:
            app._RERANKER = old
        self.assertEqual({x['evidence_id'] for x in selected_on}, {'E1', 'E2'})
        self.assertEqual({x['evidence_id'] for x in selected_off}, {'E1', 'E2'})
        self.assertTrue(all(x['reranker_score'] is None for x in ranked_off))
        self.assertTrue(all(x['reranker_score'] is not None for x in ranked_on))


if __name__ == '__main__':
    unittest.main()
