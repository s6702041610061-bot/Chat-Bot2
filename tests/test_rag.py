import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil
from unittest.mock import patch
import httpx

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from chatbot import answer_question, generate_json, rewrite_question, QUERY_SCHEMA
from rag import Chunk, CurriculumIndex, Synonyms, load_index, parse_curriculum


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synonyms = Synonyms(BASE / "synonyms.json")
        cls.chunks = parse_curriculum(BASE / "TCT69_Curriculum.md")

    def test_real_curriculum_metadata_and_page_boundaries(self):
        document_pages = {c.pdf_page for c in self.chunks if c.kind != 'external'}
        self.assertEqual(document_pages, set(range(5, 189)) - {185})
        external = [c for c in self.chunks if c.kind == 'external']
        self.assertEqual({c.pdf_page for c in external}, {900})
        self.assertTrue(any('LINE Official: @sa.kmutnb' in c.text for c in external))
        self.assertEqual(len({c.id for c in self.chunks}), len(self.chunks))
        credit = next(c for c in self.chunks if '139 หน่วยกิต' in c.text)
        self.assertEqual((credit.pdf_page, credit.printed_page, credit.kind), (31, '27', 'main'))

    def test_aliases_without_english_substring_collisions(self):
        self.assertIn('ค่าเทอม', self.synonyms.terms('ค่าเล่าเรียน'))
        self.assertIn('เทคโนโลยีสารสนเทศ', self.synonyms.terms('ปวช. ไอที'))
        self.assertIn('ภาคการศึกษาที่ 2', self.synonyms.terms('ปีหนึ่งเทอมสอง'))
        self.assertNotIn('สอบตก', self.synonyms.terms('flash'))
        self.assertNotIn('ประกาศนียบัตรวิชาชีพ', self.synonyms.terms('ประกาศนียบัตรวิชาชีพชั้นสูง'))
        self.assertIn('หน่วยกิต', self.synonyms.terms('OOP กี่เครดิต'))
        self.assertNotIn('หน่วยกิตรวมตลอดหลักสูตร', self.synonyms.terms('OOP กี่เครดิต'))

    def test_historical_records_keep_both_years(self):
        comparisons = [c for c in self.chunks if c.kind == 'comparison']
        self.assertTrue(comparisons)
        self.assertTrue(any('หลักสูตรฉบับปี พ.ศ.2564' in c.text and
                            'หลักสูตรปรับปรุง พ.ศ.2569' in c.text for c in comparisons))

    @patch('rag.time.sleep')
    @patch('rag.embed')
    def test_cache_reuse_and_invalidation(self, embedding, sleep):
        embedding.side_effect = lambda client, texts, model, task, dimensions: np.ones((len(texts), dimensions), dtype=np.float32) / np.sqrt(dimensions)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.md'
            source.write_text('<!-- source-page: 005; printed-page: 1; category: 1 -->\n#### หลักสูตร\nข้อมูล ก', encoding='utf-8')
            first = load_index(source, BASE/'synonyms.json', root/'cache', object())
            calls = embedding.call_count
            second = load_index(source, BASE/'synonyms.json', root/'cache', object())
            self.assertEqual(calls, embedding.call_count)
            np.testing.assert_array_equal(first.vectors, second.vectors)
            source.write_text(source.read_text(encoding='utf-8') + '\nข้อมูล ข', encoding='utf-8')
            load_index(source, BASE/'synonyms.json', root/'cache', object())
            self.assertGreater(embedding.call_count, calls)

    @patch('rag.embed', return_value=np.array([[1., 0.]], dtype=np.float32))
    def test_history_filter_and_page_expansion(self, embedding):
        chunks = [Chunk('a', 'หน่วยกิตรวม 139', 3, '27', 31, '2569', 'main', 'หน่วยกิต', 'x.md'),
                  Chunk('b', 'วิชาเลือกเสรี 6 หน่วยกิต', 3, '27', 31, '2569', 'main', 'วิชาเลือก', 'x.md'),
                  Chunk('c', 'หน่วยกิตรวม 142', 9, '138', 142, '2564/2569', 'comparison', 'หน่วยกิต', 'x.md')]
        index = CurriculumIndex(chunks, np.array([[1.,0.],[.8,.2],[1.,0.]]), self.synonyms, 'fake', 2)
        current = index.search('หน่วยกิตรวม', 3, query_vector=np.array([1., 0.]))
        self.assertEqual(len(current), 1)
        self.assertIn('วิชาเลือกเสรี 6', current[0][0].text)
        history = index.search('หน่วยกิตรวม 2564', 3, query_vector=np.array([1., 0.]))
        self.assertTrue(any(c.kind == 'comparison' for c, _ in history))

    @patch('chatbot.generate_json')
    def test_followup_receives_history_and_topic_change(self, generate):
        generate.return_value = {'question': 'ปี 1 ภาคการศึกษาที่ 2 เรียนอะไร', 'needs_clarification': False, 'clarification': ''}
        history = [{'role': 'user', 'content': 'ปีหนึ่งเทอมแรกเรียนอะไร'}]
        result = rewrite_question(None, 'fake', 'แล้วเทอมสองล่ะ', history)
        payload = json.loads(generate.call_args.args[3])
        self.assertEqual(payload['history'], history)
        self.assertEqual(result['question'], generate.return_value['question'])

    @patch('chatbot.generate_json')
    def test_self_contained_topic_does_not_inherit_old_semester(self, generate):
        result = rewrite_question(None, 'fake', 'ค่าเล่าเรียนเท่าไร', [
            {'role':'user','content':'ปีหนึ่งเทอมสองเรียนอะไร'},
            {'role':'assistant','content':'ปีหนึ่งเทอมสองเรียนรวม 18 หน่วยกิต'}])
        self.assertEqual(result['question'], 'ค่าเล่าเรียนเท่าไร')
        generate.assert_not_called()

    @patch('chatbot.generate_json')
    def test_clarification_does_not_retrieve(self, generate):
        generate.return_value = {'question': '', 'needs_clarification': True, 'clarification': 'หมายถึงวิชาใดคะ'}
        result = answer_question(None, None, 'fake', 'วิชานั้นล่ะ', [{'role':'user','content':'สองวิชานี้'}], embedding_client=None)
        self.assertEqual(result['sources'], [])

    @patch('chatbot.generate_json')
    def test_rejects_fabricated_source_ids(self, generate):
        class Index:
            synonyms = RagTests.synonyms
            model = 'fake'
            dimensions = 768
            def search(self, *args, **kwargs):
                return [(RagTests.chunks[0], .8)]
        generate.return_value = {'answer': 'คำตอบ [S99]', 'source_ids':['S99']}
        with self.assertRaises(ValueError):
            answer_question(None, Index(), 'fake', 'คำถาม', [], embedding_client=None)

    @patch('chatbot.generate_json')
    def test_unanswered_question_adds_department_contact_once(self, generate):
        class Index:
            synonyms = RagTests.synonyms
            model = 'fake'
            dimensions = 768
            def search(self, *args, **kwargs):
                return [(RagTests.chunks[0], .8)]
        generate.return_value = {'answer': 'ไม่พบข้อมูลนี้ในเอกสารหลักสูตรที่ใช้', 'source_ids': []}
        result = answer_question(None, Index(), 'fake', 'คำถามนอกเอกสาร', [], embedding_client=None)
        self.assertEqual(result['content'].count('https://ced.kmutnb.ac.th/'), 1)
        self.assertIn('Facebook: computer Education', result['content'])
        self.assertIn('02-555-2000 ต่อ 3240', result['content'])

    @patch('chatbot.time.sleep')
    def test_temporarily_unavailable_model_retries(self, sleep):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={'choices':[{'finish_reason':'stop', 'message':{
                'content':'{"question":"หลักสูตร", "needs_clarification":false, "clarification":""}'}}]})
        with httpx.Client(base_url='https://api.deepseek.com', transport=httpx.MockTransport(handler)) as client:
            result = generate_json(client, 'deepseek-flash', 'instruction', 'payload', QUERY_SCHEMA)
        self.assertEqual(result['question'], 'หลักสูตร')
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once()

    @patch('rag.embed', side_effect=AssertionError('Document embeddings must not be recreated'))
    def test_exported_real_index_restores_missing_runtime_cache(self, embedding):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'TCT69_Curriculum.md'
            shutil.copyfile(BASE/source.name, source)
            shutil.copytree(BASE/'embeddings', root/'embeddings')
            restored = load_index(source, BASE/'synonyms.json', root/'cache', None)
            self.assertEqual(restored.vectors.shape, (len(self.chunks), 768))
            with np.load(BASE/'embeddings/curriculum_embeddings.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(restored.vectors, saved['vectors'])
            self.assertEqual(len(list((root/'cache').glob('*.npz'))), 1)
            again = load_index(source, BASE/'synonyms.json', root/'cache', None)
            np.testing.assert_array_equal(restored.vectors, again.vectors)
            embedding.assert_not_called()

    @patch('rag.embed')
    def test_exported_index_with_wrong_provenance_is_not_reused(self, embedding):
        embedding.side_effect = lambda client, texts, model, task, dimensions: np.ones(
            (len(texts), dimensions), dtype=np.float32) / np.sqrt(dimensions)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source.md'
            source.write_text('<!-- source-page: 005; printed-page: 1; category: 1 -->\n#### หลักสูตร\nข้อมูล ก', encoding='utf-8')
            load_index(source, BASE/'synonyms.json', root/'cache', object())
            # Remove only a temporary test cache, then tamper its exported provenance.
            shutil.rmtree(root/'cache')
            manifest = root/'embeddings/curriculum_metadata.json'
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
            metadata['source_sha256'] = 'incorrect'
            manifest.write_text(json.dumps(metadata), encoding='utf-8')
            load_index(source, BASE/'synonyms.json', root/'cache', object())
            self.assertEqual(embedding.call_count, 2)


if __name__ == '__main__':
    unittest.main()
