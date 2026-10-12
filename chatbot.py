"""Conversation and answer generation, independent of the Streamlit UI."""
import json
import re
import time

import httpx

from prompt import PROMPT_CURRICULUM, QUERY_PROMPT

DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
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


def create_chat_client(api_key: str) -> httpx.Client:
    return httpx.Client(base_url=DEEPSEEK_BASE_URL, timeout=60.0,
                        headers={"Authorization": f"Bearer {api_key}"})


def validate_json_result(result: dict, schema: dict) -> None:
    if not isinstance(result, dict) or any(key not in result for key in schema["required"]):
        raise ValueError("รูปแบบคำตอบไม่ครบถ้วน")
    for key, specification in schema["properties"].items():
        value = result.get(key)
        kind = specification["type"]
        if ((kind == "string" and not isinstance(value, str)) or
                (kind == "boolean" and not isinstance(value, bool)) or
                (kind == "array" and (not isinstance(value, list) or
                                      any(not isinstance(item, str) for item in value)))):
            raise ValueError("ชนิดข้อมูลคำตอบไม่ถูกต้อง")


def generate_json(client, model_name: str, instruction: str, payload: str, schema: dict) -> dict:
    example = ({"question": "คำถามสำหรับค้นข้อมูล", "needs_clarification": False, "clarification": ""}
               if "question" in schema["properties"] else
               {"answer": "คำตอบจากหลักฐาน", "source_ids": ["S1"]})
    instruction += ("\nตอบเป็น JSON object ตาม schema นี้: " + json.dumps(schema, ensure_ascii=False)
                    + "\nตัวอย่าง JSON: " + json.dumps(example, ensure_ascii=False))
    request = {
        "model": model_name,
        "messages": [{"role": "system", "content": instruction},
                     {"role": "user", "content": payload}],
        "thinking": {"type": "disabled"}, "temperature": 0.1,
        "max_tokens": 4096, "response_format": {"type": "json_object"}, "stream": False,
    }
    last_status = None
    for attempt in range(3):
        try:
            response = client.post("/chat/completions", json=request)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("คำตอบยังไม่สมบูรณ์")
            result = json.loads(choice["message"]["content"] or "")
            validate_json_result(result, schema)
            return result
        except httpx.HTTPStatusError as error:
            last_status = error.response.status_code
            messages = {
                401: "คีย์ DeepSeek ไม่ถูกต้อง กรุณาตรวจ DEEPSEEK_API_KEY",
                402: "เครดิต DeepSeek ไม่เพียงพอ กรุณาตรวจยอดเงินหรือเติมเครดิต",
                404: "ไม่พบโมเดล DeepSeek ที่ตั้งค่าไว้ กรุณาตรวจ DEEPSEEK_MODEL",
            }
            if last_status not in (429, 500, 502, 503, 504):
                raise ChatServiceError(messages.get(last_status,
                    "บริการตอบคำถามไม่สามารถรับคำขอนี้ได้ กรุณาตรวจการตั้งค่า")) from None
        except (httpx.TransportError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            last_status = None
        if attempt < 2:
            time.sleep(2 ** (attempt + 1))
    message = ("บริการตอบคำถามได้รับคำขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่"
               if last_status == 429 else "บริการตอบคำถามยังไม่พร้อม กรุณาลองใหม่อีกครั้ง")
    raise ChatServiceError(message) from None


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


def answer_question(client, index, model: str, question: str, history: list[dict], top_k: int = 8,
                    *, embedding_client) -> dict:
    rewritten = rewrite_question(client, model, question, history)
    if rewritten["needs_clarification"]:
        return {"content": rewritten["clarification"], "sources": [], "search_question": ""}
    search_question = rewritten["question"]
    retrieved = index.search(embedding_client, search_question, top_k)
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
