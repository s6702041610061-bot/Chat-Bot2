import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import query_cache as qc
from rag import Chunk, CurriculumIndex, Synonyms, embed


class QueryCacheTests(unittest.TestCase):
    def setUp(self):
        qc.runtime_vector.clear()
        self.synonyms = Synonyms(BASE/'synonyms.json')
        self.index = Mock(model='fake', dimensions=768, synonyms=self.synonyms)

    def tearDown(self):
        qc.runtime_vector.clear()

    def test_normalization_preserves_conditions(self):
        self.assertEqual(qc.normalize_question('  ปี ๑   เทอม ๒?! '), 'ปี 1 เทอม 2')
        self.assertNotEqual(qc.canonicalize('ปี 1 ภาคการศึกษาที่ 1 เรียนอะไร'),
                            qc.canonicalize('ปี 1 ภาคการศึกษาที่ 2 เรียนอะไร'))
        self.assertEqual(qc.canonicalize('ค่าเทอมเท่าไร?'), qc.canonicalize('ค่าเล่าเรียนเท่าไร'))

    @patch('query_cache.embed')
    def test_runtime_reuse_ignores_client_and_rejects_different_model(self, embed):
        embed.return_value = np.ones((1, 768), dtype=np.float32)
        first = qc.resolve_vector(self.index, 'คำถามใหม่ ปี ๑?', object())[1]
        second = qc.resolve_vector(self.index, '  คำถามใหม่ ปี 1  ', object())[1]
        np.testing.assert_array_equal(first, second)
        self.assertEqual(embed.call_count, 1)
        qc.resolve_vector(self.index, 'คำถามใหม่ ปี 1', None)
        self.assertEqual(embed.call_count, 1)
        self.index.model = 'changed'
        qc.resolve_vector(self.index, 'คำถามใหม่ ปี 1', object())
        self.assertEqual(embed.call_count, 2)

    @patch('query_cache.embed')
    def test_provider_errors_are_not_cached(self, embed):
        for error in (TimeoutError(), ValueError(), RuntimeError('quota'), RuntimeError('key')):
            embed.side_effect = error
            self.assertIsNone(qc.resolve_vector(self.index, 'คำถามทดสอบ', object())[1])
        self.assertEqual(embed.call_count, 4)

    @patch('query_cache.embed')
    def test_all_seed_aliases_avoid_api(self, embed):
        self.index.model = 'gemini-embedding-001'
        expected = qc.manifest(self.synonyms, self.index.model, 768)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, vectors=np.ones((50, 768), dtype=np.float32))
        real_read = Path.read_bytes
        def read(path):
            if path.name == 'query_embeddings.npz':
                return buffer.getvalue()
            if path.name == 'query_metadata.json':
                return json.dumps(expected).encode()
            return real_read(path)
        with patch.object(Path, 'read_bytes', read):
            count = 0
            for entry in qc.questions():
                for question in [entry['question'], *entry['aliases']]:
                    self.assertIsNotNone(qc.resolve_vector(self.index, question, None)[1])
                    count += 1
        self.assertEqual(count, 100)
        embed.assert_not_called()

    def test_bm25_without_vector_and_no_match(self):
        chunk = Chunk('a', 'หลักสูตร 139 หน่วยกิต', 3, '27', 31, '2569', 'main', 'หน่วยกิต', 'x.md')
        index = CurriculumIndex([chunk], None, self.synonyms, 'fake', 768)
        self.assertTrue(index.search('หน่วยกิต'))
        self.assertEqual(index.search('zzzzzz'), [])

    def test_query_provider_is_single_attempt_with_four_second_timeout(self):
        client = Mock()
        client.models.embed_content.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            embed(client, ['text'], 'gemini-embedding-001', 'RETRIEVAL_QUERY', 768)
        client.models.embed_content.assert_called_once()
        config = client.models.embed_content.call_args.kwargs['config']
        self.assertEqual(config.http_options.timeout, 4000)
        self.assertEqual(config.http_options.retry_options.attempts, 1)

    @patch('query_cache.embed')
    def test_versions_dimensions_and_expanded_text_invalidate_runtime(self, embed):
        embed.side_effect = lambda client, texts, model, task, dimensions: np.ones((1, dimensions))
        for version, dim, text in [(1, 768, 'a'), (2, 768, 'a'), (2, 384, 'a'), (2, 384, 'b')]:
            qc.runtime_vector(version, 'model', dim, qc.TASK, text, object(), {})
        self.assertEqual(embed.call_count, 4)

    def test_stale_seed_rejected(self):
        expected = qc.manifest(self.synonyms, 'model', 768)
        stale = dict(expected, model='old')
        self.assertEqual(qc.load_seed(json.dumps(expected), b'', json.dumps(stale).encode()), {})

    def test_bm25_followup_selects_official_plan(self):
        from rag import parse_curriculum
        index = CurriculumIndex(parse_curriculum(BASE/'TCT69_Curriculum.md'), None,
                                self.synonyms, 'gemini-embedding-001', 768)
        for year in range(1, 5):
            for term in (1, 2):
                result = index.search(f'ชั้นปีที่ {year} ภาคการศึกษาที่ {term} มีรายวิชาอะไรตามแผนการศึกษาในหลักสูตร พ.ศ. 2569')
                self.assertEqual(result[0][0].heading, f'ปีที่ {year} ภาคการศึกษาที่ {term}')


if __name__ == '__main__':
    unittest.main()
