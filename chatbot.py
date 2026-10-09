"""Conversation and answer generation, independent of the Streamlit UI."""
import json
import os
import re
import time

from google.genai import errors, types

from prompt import PROMPT_CURRICULUM, QUERY_PROMPT

DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.5-flash"
QUERY_SCHEMA = {
    "type": "object", "properties": {
        "question": {"type": "string"},
        "needs_clarification": {"type": "boolean"},
        "clarification": {"type": "string"}},
    "required": ["question", "needs_clarification", "clarification"],
}
ANSWER_SCHEMA = {
    "type": "object", "properties": {
        "answer": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "source_ids"],
}
CONTACT_FALLBACK = (
    "สามารถติดต่อสอบถามข้อมูลเพิ่มเติมได้ที่\n"
    "เว็บไซต์: https://ced.kmutnb.ac.th/\n"
    "Facebook: computer Education\n"
    "โทรศัพท์: 02-555-2000 ต่อ 3240"
)
UNANSWERED_PATTERN = re.compile(
    r"ไม่พบข้อมูล|ไม่มีข้อมูล|ไม่ระบุ(?:ไว้)?|ไม่ปรากฏในเอกสาร|"
    r"ยังยืนยันไม่ได้|ตอบ(?:คำถาม)?ไม่ได้|ไม่สามารถตอบ|ไม่ทราบ|หลักฐาน.*ไม่ครอบคลุม"
)


class ChatServiceError(RuntimeError):
    """Safe user-facing service failure, without raw provider request details."""


def add_contact_fallback(message: str) -> str:
    """Append the department contact once when a question cannot be answered."""
    message = message.strip()
    if "https://ced.kmutnb.ac.th/" in message and "ต่อ 3240" in message:
        return message
    return f"{message}\n\n{CONTACT_FALLBACK}".strip()


def generate_json(client, model_name: str, instruction: str, payload: str, schema: dict) -> dict:
    last_error = None
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", FALLBACK_MODEL).strip() or FALLBACK_MODEL
    for model in dict.fromkeys([model_name, fallback]):
        for attempt in range(3):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=instruction, temperature=0.1,
                    max_output_tokens=4096, response_mime_type="application/json",
                    response_schema=schema)
                if 'gemini-3' in model:
                    config.thinking_config = types.ThinkingConfig(thinking_level="low")
                response = client.models.generate_content(model=model, contents=payload, config=config)
                result = json.loads(response.text or "")
                if not isinstance(result, dict):
                    raise ValueError("รูปแบบคำตอบไม่ถูกต้อง")
                return result
            except errors.APIError as error:
                if error.code not in (404, 429, 500, 502, 503, 504):
                    raise
                last_error = error
                if error.code == 404:
                    break
                if error.code == 429 and 'PerDay' in str(error.details):
                    break  # Retrying cannot recover an exhausted daily quota.
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error
    message = ("โควตาของบริการตอบคำถามหมด กรุณาตรวจโควตาบัญชี Gemini หรือรอรอบโควตาใหม่"
               if isinstance(last_error, errors.APIError) and last_error.code == 429
               else "บริการตอบคำถามยังไม่พร้อม กรุณาลองใหม่อีกครั้ง")
    raise ChatServiceError(message) from last_error


def rewrite_question(client, model: str, question: str, history: list[dict]) -> dict:
    recent = [{"role": m["role"], "content": m["content"][:3000]}
              for m in history[-8:]]
    # Self-contained questions must not inherit a year/semester from another topic.
    dependent = bool(re.search(
        r"แล้ว|ล่ะ|นั้น|ดังกล่าว|ตัวนี้|อันนี้|วิชานี้|เทอมนี้|หลักสูตรนี้|"
        r"^(กี่|เท่าไร|เท่าไหร่|มีเงื่อนไข|ต้องเรียนก่อน|ต้องใช้|ได้ไหม)", question.strip()))
    if not recent or not dependent:
        return {"question": question, "needs_clarification": False, "clarification": ""}
    result = generate_json(client, model, QUERY_PROMPT,
                           json.dumps({"history": recent, "latest_question": question}, ensure_ascii=False),
                           QUERY_SCHEMA)
    if not isinstance(result.get("question"), str) or not isinstance(result.get("needs_clarification"), bool):
        raise ValueError("ไม่สามารถตีความคำถามได้ กรุณาระบุคำถามให้ครบถ้วน")
    if result["needs_clarification"] and not result.get("clarification", "").strip():
        raise ValueError("ไม่สามารถตีความคำถามได้ กรุณาระบุคำถามให้ครบถ้วน")
    if not result["needs_clarification"] and not result["question"].strip():
        raise ValueError("คำถามสำหรับค้นข้อมูลว่างเปล่า")
    return result


def answer_question(client, index, model: str, question: str, history: list[dict], top_k: int = 8) -> dict:
    rewritten = rewrite_question(client, model, question, history)
    if rewritten["needs_clarification"]:
        return {"content": rewritten["clarification"], "sources": [], "search_question": ""}
    search_question = rewritten["question"]
    retrieved = index.search(client, search_question, top_k)
    sources = {f"S{i}": chunk for i, (chunk, _) in enumerate(retrieved, 1)}
    evidence = [{"id": key, "reference": chunk.label, "content": chunk.text}
                for key, chunk in sources.items()]
    payload = json.dumps({"original_question": question, "search_question": search_question}, ensure_ascii=False)
    payload += "\n<evidence>\n" + json.dumps(evidence, ensure_ascii=False) + "\n</evidence>"
    instruction = (PROMPT_CURRICULUM +
                   "\nตอบ JSON โดย answer เป็นคำตอบที่ไม่มีรหัส S1/S2 และ source_ids "
                   "เป็นรหัสหลักฐานที่ใช้ เช่น S1 ถ้าไม่พบข้อมูลให้เป็น [] "
                   "ถ้าตอบไม่ได้หรือหลักฐานตอบได้เพียงบางส่วน ให้ระบุส่วนที่ขาด")
    result = generate_json(client, model, instruction, payload, ANSWER_SCHEMA)
    answer = result.get("answer", "").strip()
    ids = result.get("source_ids", [])
    inline = re.findall(r"\[(S\d+)\]", answer)
    if not answer or not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
        raise ValueError("รูปแบบคำตอบไม่ครบถ้วน กรุณาลองใหม่")
    if any(i not in sources for i in ids + inline):
        raise ValueError("ไม่สามารถตรวจสอบแหล่งอ้างอิงของคำตอบได้ กรุณาลองใหม่")
    ids = list(dict.fromkeys(ids + inline))
    # Keep source IDs for validation and the expandable evidence panel, while
    # removing internal retrieval labels from the user-facing answer.
    answer = re.sub(r"\s*\[S\d+\]", "", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    if not ids or UNANSWERED_PATTERN.search(answer):
        answer = add_contact_fallback(answer)
    return {"content": answer, "sources": [
        {"id": i, "label": sources[i].label, "text": sources[i].text} for i in ids],
        "search_question": search_question}
