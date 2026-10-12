"""Opt-in integration checks using DeepSeek answers and Gemini retrieval."""
import os
import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from chatbot import DEFAULT_MODEL, answer_question, rewrite_question, create_chat_client
from rag import load_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge-cases', action='store_true', help='ตรวจเฉพาะการตีความและคำถามที่ไม่มีข้อมูล')
    parser.add_argument('--smoke', action='store_true', help='ตรวจคำถามทั่วไปหนึ่งข้อและคำถามต่อเนื่องหนึ่งข้อ')
    args = parser.parse_args()
    load_dotenv(BASE / '.env')
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY_INSURVERSE')
    chat_key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if not key or not chat_key or chat_key.startswith(('ใส่', '<', 'your')):
        raise SystemExit('ตั้งค่า DEEPSEEK_API_KEY และ GEMINI_API_KEY ก่อนตรวจสอบจริง')
    model = os.getenv('DEEPSEEK_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL
    with create_chat_client(chat_key) as client, genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=60000)) as embedding_client:
        index = load_index(BASE/'TCT69_Curriculum.md', BASE/'synonyms.json', BASE/'.rag_cache',
                           embedding_client, model=os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001'))
        history = ([{'role':'user','content':'ปีหนึ่งเทอมสองเรียนอะไร'},
                    {'role':'assistant','content':'ชั้นปีที่ 1 ภาคการศึกษาที่ 2 เรียนรวม 18 หน่วยกิต'}]
                   if args.edge_cases else [])
        cases = [
            ('เรียนทั้งหมดกี่หน่วยกิต แบ่งเป็นหมวดอะไรบ้าง', '139', (31,)),
            ('แล้ววิชาเลือกเสรีล่ะ', '6', (31, 36)),
            ('ปีหนึ่งเทอมแรกเรียนวิชาอะไรบ้าง', 'ภาษา', (37,)),
            ('แล้วเทอมสองล่ะ', 'วงจรดิจิทัล', (38,)),
            ('จบ ปวช. ไอที สมัครได้ไหม', 'เทคโนโลยีสารสนเทศ', (101,)),
        ]
        for question, required, pages in ([] if args.edge_cases else cases[:2] if args.smoke else cases):
            result = answer_question(client, index, model, question, history, embedding_client=embedding_client)
            print('QUESTION:', question)
            print('SEARCH:', result['search_question'])
            print('ANSWER:', result['content'])
            print('SOURCES:', [s['label'] for s in result['sources']])
            assert required in result['content'], f'Missing expected fact: {required}'
            assert any(f'หน้า PDF {page}' in s['label']
                       for s in result['sources'] for page in pages), 'Missing expected page'
            history.extend([{'role':'user','content':question}, {'role':'assistant','content':result['content']}])
        if args.smoke:
            print('LIVE_SMOKE_PASSED')
            return
        rewritten = rewrite_question(client, model, 'ค่าเล่าเรียนเท่าไร', history)
        assert 'เทอมสอง' not in rewritten['question'] and 'ภาคการศึกษาที่ 2' not in rewritten['question']
        print('TOPIC_CHANGE:', rewritten)
        admission = rewrite_question(client, model, 'จบ ปวช. ไอที สมัครได้ไหม', [
            {'role':'user','content':'ปีหนึ่งเทอมสองเรียนอะไรบ้าง'},
            {'role':'assistant','content':'ปีหนึ่งเทอมสองมีวงจรดิจิทัลและภาษาอังกฤษ 2'}])
        assert 'หรือคอมพิวเตอร์' not in admission['question']
        print('ADMISSION_REWRITE:', admission)
        absent = answer_question(client, index, model, 'วันเปิดรับสมัครปี 2570 คือวันที่เท่าไร', [], embedding_client=embedding_client)
        print('NO_DATA:', absent['content'])
        assert any(term in absent['content'] for term in ('ไม่พบ', 'ไม่มีข้อมูล', 'ไม่ระบุ', 'ไม่ได้ระบุ'))
        assert all(term in absent['content'] for term in (
            'https://ced.kmutnb.ac.th/', 'computer Education', '3240'))
        print('LIVE_CHECKS_PASSED')


if __name__ == '__main__':
    main()
