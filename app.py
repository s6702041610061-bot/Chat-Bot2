"""Streamlit UI for curriculum RAG; retrieval and generation live in separate modules."""
import hashlib
import html
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st

from chatbot import DEFAULT_MODEL, ChatServiceError, add_contact_fallback, answer_question
from rag import EMBEDDING_MODEL, load_index

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "TCT69_Curriculum.md"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"
CACHE_DIR = BASE_DIR / ".rag_cache"
PROFILE_PATH = BASE_DIR / "assets" / "cyra-profile.jpg"
GREETING = ("สวัสดีค่ะ ฉันชื่อ CYRA (ไซรา) ผู้ช่วยตอบคำถามเกี่ยวกับหลักสูตร"
            "เทคโนโลยีคอมพิวเตอร์ พ.ศ. 2569 ยินดีช่วยค้นหาข้อมูลให้ค่ะ")
load_dotenv(BASE_DIR / ".env")


def apply_styles():
    """Apply the CYRA brown-and-green visual theme without changing app behavior."""
    st.markdown(
        """
        <style>
        :root {
            --cyra-forest: #214c39;
            --cyra-forest-deep: #17372b;
            --cyra-sage: #dbe7d8;
            --cyra-cream: #f5eee7;
            --cyra-paper: #fffdf8;
            --cyra-brown: #684b36;
            --cyra-caramel: #b78355;
            --cyra-line: #e4d7c8;
            --cyra-ink: #2f2a25;
            --cyra-panel: linear-gradient(165deg, var(--cyra-forest-deep), #2f5d45 70%, #654a36 130%);
            --cyra-page: radial-gradient(ellipse at 50% 12%, rgba(255, 255, 255, .56), transparent 42rem),
                         radial-gradient(ellipse at 8% 4%, rgba(183, 131, 85, .25), transparent 38rem),
                         radial-gradient(ellipse at 92% 12%, rgba(64, 108, 77, .22), transparent 34rem),
                         radial-gradient(ellipse at 48% 100%, rgba(153, 105, 75, .17), transparent 38rem),
                         linear-gradient(135deg, #f8f1e9 0%, #f1e5da 48%, #e8ece3 100%);
        }

        html, body, [class*="css"] {
            font-family: "Noto Sans Thai", "Leelawadee UI", "Tahoma", sans-serif;
        }

        .stApp {
            color: var(--cyra-ink);
            background: var(--cyra-page);
            background-attachment: fixed;
            box-shadow: inset 0 0 150px rgba(88, 62, 43, .09);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            background: transparent;
        }

        .block-container {
            max-width: 900px;
            padding-top: 2.15rem;
            padding-bottom: 7.5rem;
        }

        .st-key-cyra_hero {
            position: relative;
            overflow: hidden;
            width: 100%;
            max-width: 860px;
            margin: 0 auto 1.35rem;
            padding: 1.05rem 1.25rem;
            border: 1px solid rgba(255, 255, 255, .12);
            border-radius: 22px;
            background: var(--cyra-panel);
            box-shadow: 0 14px 34px rgba(33, 76, 57, .16);
            backdrop-filter: blur(12px);
        }

        .st-key-cyra_hero::after {
            content: "";
            position: absolute;
            right: -28px;
            bottom: -48px;
            width: 132px;
            height: 132px;
            border-radius: 50%;
            background: rgba(255, 255, 255, .045);
            pointer-events: none;
        }

        .cyra-name {
            margin: 0 !important;
            color: #fffdf8;
            font-size: clamp(2rem, 3.8vw, 2.7rem);
            font-weight: 800;
            line-height: 1;
            letter-spacing: -.04em;
        }

        .cyra-name span {
            margin-left: .45rem;
            color: #e4b987;
            font-size: .43em;
            font-weight: 700;
            letter-spacing: 0;
            vertical-align: middle;
        }

        .cyra-tagline {
            max-width: 42rem;
            margin: .18rem 0 0 !important;
            color: #edf3ed;
            font-size: .98rem;
            line-height: 1.48;
        }

        .cyra-status {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            margin-top: .42rem;
            color: #dce9dc;
            font-size: .78rem;
            font-weight: 700;
        }

        .cyra-status::before {
            content: "";
            width: .52rem;
            height: .52rem;
            border-radius: 50%;
            background: #9fcca8;
            box-shadow: 0 0 0 4px rgba(159, 204, 168, .16);
        }

        .st-key-cyra_hero [data-testid="stImage"] img {
            width: 82px !important;
            height: 82px !important;
            border: 3px solid rgba(255, 255, 255, .9);
            border-radius: 23px;
            object-fit: cover;
            box-shadow: 0 8px 19px rgba(33, 76, 57, .18);
        }

        [data-testid="stChatMessage"] {
            width: auto;
            margin-bottom: .9rem;
            color: var(--cyra-ink);
        }

        .cyra-user-row {
            display: flex;
            justify-content: flex-end;
            width: 100%;
            margin: 0 0 1rem;
        }

        .cyra-user-bubble {
            display: inline-block;
            width: auto;
            max-width: 50%;
            padding: .72rem 1rem;
            border: 1px solid rgba(104, 75, 54, .13);
            border-radius: 17px 17px 5px 17px;
            color: #3f342d;
            background: #e6d1ba;
            box-shadow: 0 10px 24px rgba(91, 59, 40, .16),
                        0 2px 6px rgba(91, 59, 40, .08),
                        inset 0 1px 0 rgba(255, 255, 255, .55);
            line-height: 1.55;
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }

        /* The row stays full width; only its inner content becomes the question bubble. */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            display: flex;
            justify-content: flex-end;
            width: 100%;
            max-width: none;
            margin: 0 0 1rem;
            padding: 0;
            border: 0;
            border-radius: 0;
            color: #3f342d;
            background: transparent;
            box-shadow: none;
            gap: 0;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
            flex: 0 1 auto;
            width: fit-content;
            min-width: 0;
            max-width: 50%;
            padding: .72rem 1rem;
            border: 1px solid rgba(104, 75, 54, .13);
            border-radius: 17px 17px 5px 17px;
            background: #e6d1ba;
            box-shadow: 0 10px 24px rgba(91, 59, 40, .16),
                        0 2px 6px rgba(91, 59, 40, .08),
                        inset 0 1px 0 rgba(255, 255, 255, .55);
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stMarkdownContainer"] {
            width: fit-content;
            max-width: 100%;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stMarkdownContainer"] p {
            margin: 0;
            overflow-wrap: anywhere;
        }

        /* CYRA responses sit naturally on the page instead of inside large cards. */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            max-width: 82%;
            margin-right: auto;
            padding: .3rem 0 1.1rem;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) li {
            color: #3f342d;
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            width: 2rem;
            height: 2rem;
            overflow: hidden;
            border: 1px solid #d8c3a9;
            background: var(--cyra-paper);
        }

        [data-testid="stChatMessageAvatarAssistant"] img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        [data-testid="stChatMessageAvatarUser"] {
            display: none;
        }

        [data-testid="stExpander"] {
            margin-top: .75rem;
            overflow: hidden;
            border: 1px solid #e1d4c4;
            border-radius: 13px;
            background: #faf6ef;
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(104, 75, 54, .1);
            border-radius: 17px;
            background: #e8ddd5;
            box-shadow: 0 8px 24px rgba(75, 55, 41, .08);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #598069;
            box-shadow: 0 0 0 3px rgba(89, 128, 105, .13),
                        0 8px 24px rgba(75, 55, 41, .08);
        }

        [data-testid="stChatInput"] textarea {
            color: var(--cyra-ink);
            caret-color: var(--cyra-forest);
        }

        [data-testid="stChatInput"] button {
            color: var(--cyra-paper);
            border-radius: 12px;
            background: var(--cyra-forest);
        }

        /* Match the fixed input dock to the page instead of Streamlit's white default. */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"],
        .stBottom,
        .stBottom > div {
            background: linear-gradient(
                to bottom,
                rgba(245, 238, 231, 0),
                rgba(245, 238, 231, .9) 28%,
                rgba(245, 238, 231, .98)
            ) !important;
            backdrop-filter: blur(5px);
        }

        [data-testid="stBottomBlockContainer"] {
            border-top: 1px solid rgba(104, 75, 54, .08);
            box-shadow: 0 -14px 32px rgba(104, 75, 54, .035);
        }

        [data-testid="stSidebar"] {
            position: fixed !important;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 1000000;
            height: 100dvh;
            border-right: 1px solid rgba(255, 255, 255, .1);
            background: var(--cyra-panel);
            box-shadow: 18px 0 45px rgba(24, 49, 38, .18);
            animation: cyra-drawer-enter .32s cubic-bezier(.22, 1, .36, 1);
            transition: transform .32s cubic-bezier(.22, 1, .36, 1),
                        visibility .32s cubic-bezier(.22, 1, .36, 1),
                        box-shadow .32s ease;
            will-change: transform;
        }

        @keyframes cyra-drawer-enter {
            from {
                transform: translateX(-100%);
                box-shadow: 0 0 0 rgba(24, 49, 38, 0);
            }
            to {
                transform: translateX(0);
                box-shadow: 18px 0 45px rgba(24, 49, 38, .18);
            }
        }

        [data-testid="stAppViewContainer"] > .main {
            width: 100% !important;
            margin-left: 0 !important;
        }

        [data-testid="stSidebarCollapseButton"] {
            position: absolute;
            top: 1rem;
            right: 1rem;
            z-index: 2;
        }

        [data-testid="stSidebarCollapsedControl"] {
            position: fixed;
            top: 1rem;
            left: 1rem;
            z-index: 1000001;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button {
            width: 2.45rem;
            height: 2.45rem;
            padding: 0;
            border: 1px solid rgba(255, 255, 255, .18);
            border-radius: 50%;
            color: #fffdf8;
            background: rgba(23, 55, 43, .88);
            box-shadow: 0 5px 16px rgba(33, 76, 57, .18);
            backdrop-filter: blur(8px);
            transition: transform .22s ease, background .22s ease, box-shadow .22s ease;
            animation: cyra-toggle-enter .24s cubic-bezier(.22, 1, .36, 1);
        }

        [data-testid="stSidebarCollapseButton"] button svg,
        [data-testid="stSidebarCollapsedControl"] button svg {
            transition: transform .28s cubic-bezier(.22, 1, .36, 1);
        }

        [data-testid="stSidebarCollapseButton"] button svg {
            transform: rotate(180deg);
        }

        @keyframes cyra-toggle-enter {
            from { opacity: 0; transform: scale(.82); }
            to { opacity: 1; transform: scale(1); }
        }

        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover {
            color: #fff;
            background: #2f5d45;
            transform: scale(1.04);
            box-shadow: 0 8px 20px rgba(33, 76, 57, .25);
        }

        [data-testid="stSidebarCollapseButton"] button:active,
        [data-testid="stSidebarCollapsedControl"] button:active {
            transform: scale(.92);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #fbf7ee;
        }

        [data-testid="stSidebar"] [data-testid="stImage"] img {
            width: 76px !important;
            height: 76px !important;
            border: 3px solid rgba(255, 255, 255, .85);
            border-radius: 22px;
            object-fit: cover;
            box-shadow: 0 10px 22px rgba(0, 0, 0, .2);
        }

        [data-testid="stSidebar"] .stButton > button {
            min-height: 2.7rem;
            border: 1px solid rgba(255, 255, 255, .35);
            border-radius: 13px;
            color: #fffdf8;
            background: rgba(255, 255, 255, .1);
            transition: transform .18s ease, background .18s ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(255, 255, 255, .65);
            color: #fff;
            background: rgba(255, 255, 255, .18);
            transform: translateY(-1px);
        }

        .cyra-side-name {
            margin: .8rem 0 .15rem;
            color: #fffdf8;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: .04em;
        }

        .cyra-side-label {
            display: inline-block;
            margin-bottom: 1.35rem;
            padding: .28rem .65rem;
            border: 1px solid rgba(255, 255, 255, .18);
            border-radius: 999px;
            color: #dce9dc;
            background: rgba(255, 255, 255, .08);
            font-size: .75rem;
        }

        .cyra-divider {
            height: 1px;
            margin: .2rem 0 1.2rem;
            background: rgba(255, 255, 255, .15);
        }

        div.stButton > button {
            border-color: #bca58e;
            border-radius: 12px;
            color: var(--cyra-brown);
            background: var(--cyra-paper);
        }

        [data-testid="stAlert"] {
            border-radius: 15px;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 1.25rem;
                padding-right: 1rem;
                padding-left: 1rem;
            }

            .st-key-cyra_hero {
                padding: 1rem;
                border-radius: 19px;
            }

            .st-key-cyra_hero [data-testid="stImage"] img {
                width: 70px !important;
                height: 70px !important;
                border-radius: 20px;
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                max-width: none;
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
                max-width: 80%;
            }

            .cyra-user-bubble {
                max-width: 80%;
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
                max-width: 94%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    with st.container(key="cyra_hero"):
        avatar_column, title_column = st.columns([.82, 5.18], vertical_alignment="center")
        with avatar_column:
            st.image(str(PROFILE_PATH), width=82)
        with title_column:
            st.markdown(
                """
                <h1 class="cyra-name">CYRA <span>ไซรา</span></h1>
                <p class="cyra-tagline">ผู้ช่วยค้นข้อมูลหลักสูตรเทคโนโลยีคอมพิวเตอร์ พ.ศ. 2569
                ถามได้อย่างเป็นธรรมชาติ พร้อมตรวจสอบข้อมูลจากเอกสารหลักสูตร</p>
                <div class="cyra-status">พร้อมช่วยค้นคำตอบ</div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))


@st.cache_resource(show_spinner=False)
def get_index(source_hash: str, synonyms_hash: str, embedding_model: str, api_key: str):
    # Hashes invalidate the in-memory cache when either source file changes.
    progress = st.progress(0, text="กำลังเตรียมฐานความรู้หลักสูตร...")
    try:
        return load_index(DATA_PATH, SYNONYMS_PATH, CACHE_DIR, get_client(api_key),
                          model=embedding_model,
                          progress=lambda done, total: progress.progress(
                              done / total, text=f"กำลังเตรียมข้อมูล {done}/{total}"))
    finally:
        progress.empty()


def render_message(message: dict):
    if message["role"] == "user":
        safe_content = html.escape(message["content"])
        st.markdown(
            f'<div class="cyra-user-row"><div class="cyra-user-bubble">{safe_content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    with st.chat_message("assistant", avatar=str(PROFILE_PATH)):
        st.caption("CYRA · คำตอบจากเอกสารหลักสูตร")
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("ดูแหล่งอ้างอิง"):
                for source in message["sources"]:
                    st.caption(source["label"])
                    st.text(source["text"])


def main():
    st.set_page_config(
        page_title="CYRA | ผู้ช่วยหลักสูตรเทคโนโลยีคอมพิวเตอร์",
        page_icon=str(PROFILE_PATH),
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    apply_styles()
    render_header()
    api_key = (os.getenv("GEMINI_API_KEY", "").strip()
               or os.getenv("GEMINI_API_KEY_INSURVERSE", "").strip())
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", EMBEDDING_MODEL).strip() or EMBEDDING_MODEL
    try:
        source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
        synonyms_hash = hashlib.sha256(SYNONYMS_PATH.read_bytes()).hexdigest()
    except OSError:
        st.error("ไม่พบไฟล์หลักสูตรหรือรายการคำที่เกี่ยวข้อง กรุณาตรวจโฟลเดอร์ workaw")
        st.stop()
    identity = source_hash + synonyms_hash + embedding_model
    if st.session_state.get("knowledge_identity") != identity:
        st.session_state["messages"] = [{"role": "assistant", "content": GREETING}]
        st.session_state["knowledge_identity"] = identity
    with st.sidebar:
        st.image(str(PROFILE_PATH), width=76)
        st.markdown('<div class="cyra-side-name">CYRA</div>', unsafe_allow_html=True)
        st.markdown('<div class="cyra-side-label">ผู้ช่วยข้อมูลหลักสูตร</div>', unsafe_allow_html=True)
        st.markdown('<div class="cyra-divider"></div>', unsafe_allow_html=True)
        st.subheader("ข้อมูลที่ใช้ตอบ")
        st.write("หลักสูตรเทคโนโลยีคอมพิวเตอร์ พ.ศ. 2569")
        st.caption("ถามต่อจากเรื่องเดิมได้ และเริ่มเรื่องใหม่ได้ทุกเมื่อ")
        if st.button("ล้างประวัติการสนทนา", use_container_width=True):
            st.session_state["messages"] = [{"role": "assistant", "content": GREETING}]
            st.session_state.pop("last_error", None)
            st.rerun()
    for message in st.session_state["messages"]:
        render_message(message)
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])
    if not api_key or api_key.startswith("ใส่_"):
        st.warning("กรุณาตั้งค่า GEMINI_API_KEY ใน workaw/.env ก่อนเริ่มสนทนา")
        st.chat_input("พิมพ์คำถามเกี่ยวกับหลักสูตร", disabled=True)
        return
    pending = st.session_state["messages"][-1]["role"] == "user"
    retry = pending and st.button("ลองตอบคำถามล่าสุดอีกครั้ง")
    question = st.chat_input("เช่น ปีหนึ่งเทอมสองเรียนวิชาอะไรบ้าง", max_chars=2000, disabled=pending)
    if question and question.strip():
        st.session_state.pop("last_error", None)
        st.session_state["messages"].append({"role": "user", "content": question.strip()})
        render_message(st.session_state["messages"][-1])
    if (question and question.strip()) or retry:
        latest = st.session_state["messages"][-1]["content"]
        history = st.session_state["messages"][1:-1]
        try:
            with st.spinner("กำลังค้นข้อมูลหลักสูตร..."):
                index = get_index(source_hash, synonyms_hash, embedding_model, api_key)
                response = answer_question(get_client(api_key), index, model, latest, history)
            st.session_state["messages"].append({"role": "assistant", **response})
            st.session_state.pop("last_error", None)
            st.rerun()
        except ChatServiceError as error:
            st.session_state["last_error"] = add_contact_fallback(str(error))
            st.rerun()
        except Exception:
            # SDK exception strings may include service/request details. Do not expose them.
            st.session_state["last_error"] = add_contact_fallback(
                "ยังตอบคำถามไม่ได้ กรุณาตรวจการเชื่อมต่อ สิทธิ์หรือโควตาของ API แล้วลองอีกครั้ง")
            st.rerun()


if __name__ == "__main__":
    main()
