"""Opt-in integration checks using DeepSeek answers and Gemini retrieval."""
import os
import argparse
import logging
from contextlib import nullcontext
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
    parser.add_argument('--no-embedding', action='store_true', help='ตรวจ seed/BM25 โดยไม่ใช้ Gemini')
    parser.add_argument('--external-only', action='store_true', help='ตรวจเฉพาะข้อมูลเพิ่มเติมนอกหลักสูตร')
    args = parser.parse_args()
    load_dotenv(BASE / '.env')
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY_INSURVERSE')
    chat_key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if (not key and not args.no_embedding) or not chat_key or chat_key.startswith(('ใส่', '<', 'your')):
        raise SystemExit('ตั้งค่า DEEPSEEK_API_KEY และ GEMINI_API_KEY ก่อนตรวจสอบจริง')
    model = os.getenv('DEEPSEEK_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL
    logging.basicConfig(level=logging.INFO)
    embedding_context = (nullcontext(None) if args.no_embedding else genai.Client(
        api_key=key, http_options=types.HttpOptions(timeout=60000)))
    with create_chat_client(chat_key) as client, embedding_context as embedding_client:
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
        for question, required, pages in ([] if args.edge_cases or args.external_only else cases[:2] if args.smoke else cases):
            result = answer_question(client, index, model, question, history,
                                     embedding_client=None if args.no_embedding else embedding_client)
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
        if not args.edge_cases:
            external_cases = [
                ('คณะครุศาสตร์อุตสาหกรรม กรุงเทพฯ ค่าเทอมภาคปกติปี 2569 เท่าไร', ('25000',)),
                ('ค่าธรรมเนียมแรกเข้าที่ไม่รวมค่าเทอมรวมเท่าไร มีรายการอะไรบ้าง', ('2700', '1000', '200', '500')),
                ('ห้องสมุดเปิดวันเสาร์ภาคการศึกษาปกติกี่โมง', ('09.00', '18.00')),
                ('ศูนย์บริการสุขภาพอยู่ที่ไหนและโทรต่ออะไร', ('79', '1126')),
                ('หอพักใน มจพ. กรุงเทพฯ รองรับนักศึกษาได้กี่คน', ('960',)),
                ('จาก MRT เตาปูนไป มจพ. ต่อรถเมล์สายอะไร ไปทิศทางไหน', ('97', 'กระทรวงสาธารณสุข')),
                ('ทุนการศึกษา เอกสารประกอบการสมัครทุนมีอะไรบ้าง', ('รายได้', 'ผลการเรียน')),
                ('กยศ. ผู้กู้รายใหม่มีใครบ้าง', ('สถานศึกษาอื่น', 'เปลี่ยนระดับ')),
                ('ใช้ Wi-Fi มหาวิทยาลัยและ eduroam ด้วยบัญชีอะไร', ('ICIT',)),
            ]
            for question, facts in external_cases:
                result = answer_question(client, index, model, question, [],
                                         embedding_client=None if args.no_embedding else embedding_client)
                print('EXTERNAL_QUESTION:', question, flush=True)
                print('ANSWER:', result['content'], flush=True)
                print('SOURCES:', [s['label'] for s in result['sources']], flush=True)
                content = result['content'].replace(',', '').replace(':', '.')
                evidence = '\n'.join(s['text'] for s in result['sources']).replace(',', '').replace(':', '.')
                for fact in facts:
                    assert fact in content, f'Missing answer fact: {fact}'
                    assert fact in evidence, f'Missing supporting fact: {fact}'
                assert any('ข้อมูลเพิ่มเติมนอกหลักสูตร' in s['label'] for s in result['sources'])
            print('EXTERNAL_CHECKS_PASSED', flush=True)
            if args.external_only:
                return
        rewritten = rewrite_question(client, model, 'ค่าเล่าเรียนเท่าไร', history)
        assert 'เทอมสอง' not in rewritten['question'] and 'ภาคการศึกษาที่ 2' not in rewritten['question']
        print('TOPIC_CHANGE:', rewritten)
        changed = answer_question(client, index, model,
                                  'คณะครุศาสตร์อุตสาหกรรม ค่าเทอมภาคปกติปี 2569 เท่าไร', history,
                                  embedding_client=None if args.no_embedding else embedding_client)
        assert 'เทอมสอง' not in changed['search_question'] and 'ภาคการศึกษาที่ 2' not in changed['search_question']
        assert '25000' in changed['content'].replace(',', '')
        assert any('ข้อมูลเพิ่มเติมนอกหลักสูตร' in s['label'] for s in changed['sources'])
        print('TOPIC_CHANGE_ANSWER:', changed['content'], flush=True)
        admission = rewrite_question(client, model, 'จบ ปวช. ไอที สมัครได้ไหม', [
            {'role':'user','content':'ปีหนึ่งเทอมสองเรียนอะไรบ้าง'},
            {'role':'assistant','content':'ปีหนึ่งเทอมสองมีวงจรดิจิทัลและภาษาอังกฤษ 2'}])
        assert 'หรือคอมพิวเตอร์' not in admission['question']
        print('ADMISSION_REWRITE:', admission)
        absent = answer_question(client, index, model, 'วันเปิดรับสมัครปี 2570 คือวันที่เท่าไร', [],
                                 embedding_client=None if args.no_embedding else embedding_client)
        print('NO_DATA:', absent['content'])
        assert any(term in absent['content'] for term in ('ไม่พบ', 'ไม่มีข้อมูล', 'ไม่ระบุ', 'ไม่ได้ระบุ'))
        assert all(term in absent['content'] for term in (
            'https://ced.kmutnb.ac.th/', 'computer Education', '3240'))
        print('LIVE_CHECKS_PASSED')


if __name__ == '__main__':
    main()
