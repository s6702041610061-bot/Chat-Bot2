"""Opt-in integration checks using the configured Gemini API and saved index."""
import os
import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from chatbot import answer_question, rewrite_question
from rag import load_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge-cases', action='store_true', help='ตรวจเฉพาะการตีความและคำถามที่ไม่มีข้อมูล')
    args = parser.parse_args()
    load_dotenv(BASE / '.env')
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY_INSURVERSE')
    model = os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000)) as client:
        index = load_index(BASE/'TCT69_Curriculum.md', BASE/'synonyms.json', BASE/'.rag_cache', client)
        history = ([{'role':'user','content':'ปีหนึ่งเทอมสองเรียนอะไร'},
                    {'role':'assistant','content':'ชั้นปีที่ 1 ภาคการศึกษาที่ 2 เรียนรวม 18 หน่วยกิต'}]
                   if args.edge_cases else [])
        cases = [
            ('เรียนทั้งหมดกี่หน่วยกิต แบ่งเป็นหมวดอะไรบ้าง', '139', 31),
            ('แล้ววิชาเลือกเสรีล่ะ', '6', 31),
            ('ปีหนึ่งเทอมแรกเรียนวิชาอะไรบ้าง', 'ภาษา', 37),
            ('แล้วเทอมสองล่ะ', 'วงจรดิจิทัล', 38),
            ('จบ ปวช. ไอที สมัครได้ไหม', 'เทคโนโลยีสารสนเทศ', 101),
        ]
        for question, required, page in ([] if args.edge_cases else cases):
            result = answer_question(client, index, model, question, history)
            print('QUESTION:', question)
            print('SEARCH:', result['search_question'])
            print('ANSWER:', result['content'])
            print('SOURCES:', [s['label'] for s in result['sources']])
            assert required in result['content'], f'Missing expected fact: {required}'
            assert any(f'หน้า PDF {page}' in s['label'] for s in result['sources']), 'Missing expected page'
            history.extend([{'role':'user','content':question}, {'role':'assistant','content':result['content']}])
        rewritten = rewrite_question(client, model, 'ค่าเล่าเรียนเท่าไร', history)
        assert 'เทอมสอง' not in rewritten['question'] and 'ภาคการศึกษาที่ 2' not in rewritten['question']
        print('TOPIC_CHANGE:', rewritten)
        admission = rewrite_question(client, model, 'จบ ปวช. ไอที สมัครได้ไหม', [
            {'role':'user','content':'ปีหนึ่งเทอมสองเรียนอะไรบ้าง'},
            {'role':'assistant','content':'ปีหนึ่งเทอมสองมีวงจรดิจิทัลและภาษาอังกฤษ 2'}])
        assert 'หรือคอมพิวเตอร์' not in admission['question']
        print('ADMISSION_REWRITE:', admission)
        absent = answer_question(client, index, model, 'วันเปิดรับสมัครปี 2570 คือวันที่เท่าไร', [])
        print('NO_DATA:', absent['content'])
        assert any(term in absent['content'] for term in ('ไม่พบ', 'ไม่มีข้อมูล', 'ไม่ระบุ', 'ไม่ได้ระบุ'))
        print('LIVE_CHECKS_PASSED')


if __name__ == '__main__':
    main()
