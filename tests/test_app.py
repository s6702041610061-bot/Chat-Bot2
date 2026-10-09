import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


class AppTests(unittest.TestCase):
    def tearDown(self):
        st.cache_resource.clear()

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'})
    @patch('google.genai.Client')
    @patch('rag.load_index')
    @patch('chatbot.answer_question')
    def test_chat_followup_sources_and_clear(self, answer, index, client):
        answer.return_value = {'content': '139 หน่วยกิต [S1]', 'search_question': 'หน่วยกิตรวม',
                               'sources': [{'id':'S1', 'label':'หน้าในเล่ม 27 (หน้า PDF 31)', 'text':'139 หน่วยกิต'}]}
        app = AppTest.from_file(str(BASE/'app.py')).run(timeout=15)
        self.assertFalse(app.exception)
        app.chat_input[0].set_value('เรียนกี่หน่วยกิต').run(timeout=15)
        self.assertFalse(app.exception)
        # User messages use the custom right-aligned bubble; chat_message holds assistants only.
        self.assertEqual(len(app.chat_message), 2)
        self.assertTrue(any('หน้า PDF 31' in c.value for c in app.caption))
        app.chat_input[0].set_value('แล้ววิชาเลือกเสรีล่ะ').run(timeout=15)
        self.assertEqual(answer.call_args.args[3], 'แล้ววิชาเลือกเสรีล่ะ')
        self.assertEqual(answer.call_args.args[4][0]['content'], 'เรียนกี่หน่วยกิต')
        app.sidebar.button[0].click().run(timeout=15)
        self.assertEqual(len(app.chat_message), 1)
        self.assertFalse(app.exception)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'})
    @patch('google.genai.Client')
    @patch('rag.load_index')
    @patch('chatbot.answer_question', side_effect=RuntimeError('private service detail'))
    def test_failed_request_preserves_question_and_offers_retry(self, answer, index, client):
        app = AppTest.from_file(str(BASE/'app.py')).run(timeout=15)
        app.chat_input[0].set_value('เรียนกี่หน่วยกิต').run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 1)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(any('ลองตอบคำถามล่าสุด' in b.label for b in app.button))
        self.assertNotIn('private service detail', app.error[0].value)
        self.assertIn('https://ced.kmutnb.ac.th/', app.error[0].value)
        self.assertIn('ต่อ 3240', app.error[0].value)


if __name__ == '__main__':
    unittest.main()
