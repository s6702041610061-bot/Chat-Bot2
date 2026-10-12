"""Provider routing and failure contracts without paid network requests."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatbot import ANSWER_SCHEMA, QUERY_SCHEMA, ChatServiceError, answer_question, generate_json
from rag import Chunk


def completion(content, finish_reason='stop'):
    return httpx.Response(200, json={'choices': [{
        'finish_reason': finish_reason, 'message': {'content': content}}]})


class DeepSeekTests(unittest.TestCase):
    def test_request_and_followup_route_to_distinct_clients(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(str(request.url), 'https://api.deepseek.com/chat/completions')
            self.assertEqual(request.headers['authorization'], 'Bearer chat-test')
            self.assertEqual(payload['model'], 'deepseek-flash')
            self.assertEqual(payload['thinking'], {'type': 'disabled'})
            self.assertEqual(payload['response_format'], {'type': 'json_object'})
            self.assertEqual(payload['max_tokens'], 4096)
            self.assertIn('JSON', payload['messages'][0]['content'])
            if len(requests) == 1:
                return completion(json.dumps({'question': 'อาจารย์ในภาควิชามีใครบ้าง',
                    'needs_clarification': False, 'clarification': ''}))
            return completion(json.dumps({'answer': 'มีอาจารย์ทั้งหมด 15 คน [S1]', 'source_ids': ['S1']}))
        chunk = Chunk('a', 'อาจารย์ทั้งหมด 15 คน', 5, '90', 94, '2569', 'main', 'อาจารย์', 'source.md')
        embedding_client = object()
        index = Mock()
        index.search.return_value = [(chunk, .9)]
        with patch('chatbot.resolve_vector', return_value=('อาจารย์ในภาควิชามีใครบ้าง', None)), httpx.Client(base_url='https://api.deepseek.com', headers={'Authorization': 'Bearer chat-test'},
                          transport=httpx.MockTransport(handler)) as chat_client:
            result = answer_question(chat_client, index, 'deepseek-flash', 'แล้วมีใครบ้าง',
                [{'role': 'user', 'content': 'ภาควิชามีอาจารย์กี่คน'}], embedding_client=embedding_client)
        index.search.assert_called_once_with('อาจารย์ในภาควิชามีใครบ้าง', 8, query_vector=None)
        self.assertEqual(len(requests), 2)
        self.assertNotIn('[S1]', result['content'])
        self.assertEqual(result['sources'][0]['id'], 'S1')

    @patch('chatbot.time.sleep')
    def test_invalid_key_and_empty_balance_are_immediate_safe_errors(self, sleep):
        for status, expected in ((401, 'คีย์ DeepSeek'), (402, 'เครดิต DeepSeek'), (404, 'โมเดล DeepSeek')):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, json={'error': {'message': 'private-secret-provider-detail'}})
            with self.subTest(status=status), httpx.Client(base_url='https://api.deepseek.com',
                    transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(ChatServiceError) as caught:
                    generate_json(client, 'deepseek-flash', '', '', QUERY_SCHEMA)
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn('private-secret', str(caught.exception))
                self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    @patch('chatbot.time.sleep')
    def test_rate_limit_and_server_failures_stop_after_three_attempts(self, sleep):
        for status in (429, 500, 503):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status)
            with self.subTest(status=status), httpx.Client(base_url='https://api.deepseek.com',
                    transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(ChatServiceError):
                    generate_json(client, 'deepseek-flash', '', '', ANSWER_SCHEMA)
                self.assertEqual(len(calls), 3)

    @patch('chatbot.time.sleep')
    def test_bad_json_wrong_types_and_truncated_answers_are_rejected(self, sleep):
        invalid = ['not json', '{}', '{"answer":123,"source_ids":[]}',
                   '{"answer":"test","source_ids":[2]}']
        for content, reason in [(text, 'stop') for text in invalid] + [
                ('{"answer":"partial","source_ids":[]}', 'length')]:
            calls = []
            def handler(request):
                calls.append(request)
                return completion(content, reason)
            with self.subTest(content=content, reason=reason), httpx.Client(base_url='https://api.deepseek.com',
                    transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(ChatServiceError):
                    generate_json(client, 'deepseek-flash', '', '', ANSWER_SCHEMA)
                self.assertEqual(len(calls), 3)

    @patch('chatbot.time.sleep')
    def test_transport_timeout_recovers(self, sleep):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout('private timeout detail', request=request)
            return completion('{"answer":"complete","source_ids":[]}')
        with httpx.Client(base_url='https://api.deepseek.com', transport=httpx.MockTransport(handler)) as client:
            result = generate_json(client, 'deepseek-flash', '', '', ANSWER_SCHEMA)
        self.assertEqual(result['answer'], 'complete')
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
