"""Page-aware curriculum retrieval with persistent Gemini embeddings and BM25."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import unicodedata
from typing import Callable
from uuid import uuid4

import numpy as np
from google.genai import errors, types

PARSER_VERSION = 1
EMBEDDING_MODEL = "gemini-embedding-001"
DIMENSIONS = 768
MAX_CHARS = 1800
PAGE_PATTERN = re.compile(r"<!-- source-page: (\d+); printed-page: ([^;]+); category: (\d+) -->")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    category: int
    printed_page: str
    pdf_page: int
    status: str
    kind: str
    heading: str
    source: str

    @property
    def label(self) -> str:
        if self.kind == "external":
            return f"{self.source} — {self.heading}; ข้อมูลเพิ่มเติมนอกหลักสูตร; {self.status}"
        return (f"{self.source} — {self.heading}; หน้าในเล่ม {self.printed_page} "
                f"(หน้า PDF {self.pdf_page}); {self.status}")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().translate(
        str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))


class Synonyms:
    """Expand search terms without replacing the user's original question."""
    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("synonyms.json ต้องเป็นรายการคำหลักและคำที่เกี่ยวข้อง")
        self.groups = []
        for canonical, aliases in raw.items():
            if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
                raise ValueError("รูปแบบคำพ้องไม่ถูกต้อง")
            self.groups.append((canonical, sorted(set([canonical, *aliases]), key=len, reverse=True)))

    @staticmethod
    def matches(text: str, alias: str) -> bool:
        alias = normalize(alias)
        if re.search(r"[a-z]", alias):
            return bool(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text))
        return alias in text

    def terms(self, text: str) -> list[str]:
        normalized, matched = normalize(text), []
        for canonical, aliases in self.groups:
            for alias in aliases:
                if self.matches(normalized, alias):
                    matched.append((len(normalize(alias)), canonical, normalize(alias)))
                    break
        selected, used = [], []
        for _, canonical, alias in sorted(matched, reverse=True):
            if any(alias in previous for previous in used):
                continue
            selected.append(canonical)
            used.append(alias)
        return selected

    def expand(self, text: str) -> str:
        terms = self.terms(text)
        return text + ("\nคำค้นที่เกี่ยวข้อง: " + ", ".join(terms) if terms else "")


def parse_curriculum(path: Path) -> list[Chunk]:
    document = path.read_text(encoding="utf-8-sig")
    markers = list(PAGE_PATTERN.finditer(document))
    chunks = []
    for position, marker in enumerate(markers):
        page = int(marker.group(1))
        if page < 5:
            continue  # Cover and table of contents are not answer evidence.
        stop = markers[position + 1].start() if position + 1 < len(markers) else len(document)
        body = document[marker.end():stop]
        status_match = re.search(r"^- สถานะข้อมูล: (.+)$", body, re.M)
        status = status_match.group(1).strip() if status_match else "ข้อมูลตามเอกสาร"
        kind = ("external" if "นอกหลักสูตร" in status else
                "comparison" if "เปรียบเทียบ" in status else
                "regulation" if page >= 169 else "mou" if 122 <= page <= 136 else "main")
        page_title = re.search(r"^### (.+)$", body, re.M)
        heading = page_title.group(1) if page_title else f"องค์ประกอบที่ {marker.group(3)}"
        body = re.sub(r'<a id="[^"]+"></a>', "", body)
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        body = re.sub(r"^### .+$|^- สถานะข้อมูล: .+$|^- ลำดับระเบียนในตาราง: .+$", "", body, flags=re.M)
        body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "[ภาพประกอบในเอกสารต้นฉบับ]", body)
        body = re.sub(r"^## องค์ประกอบที่ .+$", "", body, flags=re.M)
        blocks = re.split(r"\n(?=##### ระเบียนตารางที่ |#### (?!#))", body)
        pieces, current_heading = [], heading
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            title = re.match(r"#### ([^\n]+)", block)
            if title and "ข้อความต่อเนื่อง" not in title.group(1):
                current_heading = title.group(1)
            while len(block) > MAX_CHARS:
                cut = max(block.rfind("\n", 0, MAX_CHARS), block.rfind(" ", 0, MAX_CHARS))
                if cut < MAX_CHARS // 2:
                    cut = MAX_CHARS
                pieces.append((current_heading, block[:cut]))
                block = block[cut:].strip()
            if block:
                pieces.append((current_heading, block))
        combined = []
        for title, text in pieces:
            if combined and combined[-1][0] == title and len(combined[-1][1]) + len(text) + 2 <= MAX_CHARS:
                combined[-1] = (title, combined[-1][1] + "\n\n" + text)
            else:
                combined.append((title, text))
        for i, (title, text) in enumerate(combined):
            chunks.append(Chunk(f"p{page:03d}-c{marker.group(3)}-{i:03d}", text, int(marker.group(3)),
                                marker.group(2), page, status, kind, title, path.name))
    if not chunks:
        raise ValueError("ไม่พบเนื้อหาหรือข้อมูลเลขหน้าในไฟล์หลักสูตร")
    return chunks


def embed(client, texts: list[str], model: str, task: str, dimensions: int) -> np.ndarray:
    for attempt in range(6):
        try:
            response = client.models.embed_content(
                model=model, contents=texts,
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=dimensions))
            vectors = np.asarray([item.values for item in response.embeddings or []], dtype=np.float32)
            if vectors.shape != (len(texts), dimensions) or not np.isfinite(vectors).all():
                raise ValueError("บริการ embedding ส่งข้อมูลไม่ครบหรือขนาดเวกเตอร์ไม่ตรง")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("พบ embedding ที่ไม่มีค่า")
            return vectors / norms
        except errors.APIError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            suggested = re.search(r"retry in ([\d.]+)s", error.message or "", re.I)
            delay = max(2 ** attempt, float(suggested.group(1)) + 1 if suggested else
                        30 if error.code == 429 else 0)
            time.sleep(min(delay, 60))
    raise RuntimeError("ไม่สามารถสร้าง embedding ได้")


def embedding_text(chunk: Chunk | dict) -> str:
    """Text actually embedded; lets unchanged curriculum chunks reuse old vectors."""
    if isinstance(chunk, Chunk):
        heading, status, text = chunk.heading, chunk.status, chunk.text
    else:
        heading, status, text = chunk["heading"], chunk["status"], chunk["text"]
    return f"หลักสูตร พ.ศ. 2569 | {heading} | {status}\n{text}"


def cached_chunk_vectors(cache_dir: Path, current_cache: Path, model: str,
                         dimensions: int) -> dict[str, np.ndarray]:
    """Load reusable vectors keyed by exact embedded content, across document revisions."""
    reusable = {}
    manifests = sorted(cache_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for manifest_path in manifests:
        vector_path = manifest_path.with_suffix(".npz")
        if vector_path == current_cache or not vector_path.is_file():
            continue
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = metadata.get("chunks")
            if (metadata.get("model") != model or metadata.get("dimensions") != dimensions
                    or not isinstance(records, list)):
                continue
            with np.load(vector_path, allow_pickle=False) as saved:
                vectors = saved["vectors"]
            if vectors.shape != (len(records), dimensions) or not np.isfinite(vectors).all():
                continue
            for record, vector in zip(records, vectors):
                key = hashlib.sha256(embedding_text(record).encode("utf-8")).hexdigest()
                reusable.setdefault(key, vector)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return reusable


def tokens(text: str, synonyms: Synonyms) -> Counter:
    """Thai character ngrams avoid requiring spaces between Thai words."""
    normalized = normalize(text)
    result = Counter(re.findall(r"[a-z]+|\d+", normalized))
    for run in re.findall(r"[ก-๙]+", normalized):
        for size in (2, 3):
            result.update(run[i:i + size] for i in range(len(run) - size + 1))
    result.update("syn:" + normalize(term) for term in synonyms.terms(text))
    return result


class CurriculumIndex:
    def __init__(self, chunks, vectors, synonyms, model, dimensions):
        self.chunks, self.vectors, self.synonyms = chunks, vectors, synonyms
        self.model, self.dimensions = model, dimensions
        self.documents = [tokens(c.heading + "\n" + c.text, synonyms) for c in chunks]
        self.lengths = np.asarray([sum(doc.values()) for doc in self.documents], dtype=float)
        self.average_length = max(float(self.lengths.mean()), 1)
        frequency = Counter(term for doc in self.documents for term in doc)
        self.idf = {term: math.log(1 + (len(chunks) - count + 0.5) / (count + 0.5))
                    for term, count in frequency.items()}

    def search(self, client, question: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
        query = embed(client, [self.synonyms.expand(question)], self.model, "RETRIEVAL_QUERY", self.dimensions)[0]
        semantic = self.vectors @ query
        lexical = np.zeros(len(self.chunks))
        for term in tokens(question, self.synonyms):
            frequency = np.asarray([doc.get(term, 0) for doc in self.documents], dtype=float)
            lexical += self.idf.get(term, 0) * frequency * 2.5 / (
                frequency + 1.5 * (0.25 + 0.75 * self.lengths / self.average_length))
        historical = bool(re.search(r"2564|๒๕๖๔|เปรียบเทียบ|ฉบับเดิม|หลักสูตรเก่า|ปรับปรุง.*อะไร", question))
        candidates = [i for i, c in enumerate(self.chunks) if historical or c.kind != "comparison"]
        fused = Counter()
        for scores in (semantic, lexical):
            order = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)
            for rank, index in enumerate(order[:40], 1):
                if scores is lexical and scores[index] <= 0:
                    continue
                fused[index] += 1 / (60 + rank)
        selected = sorted(fused, key=fused.get, reverse=True)[:top_k]
        # A clearly stated applicant question should use the curriculum's
        # official eligibility section as evidence, even if OCR-heavy appendices
        # receive higher semantic scores for a short query.
        query_terms = set(self.synonyms.terms(question))
        applicant_terms = {"คุณสมบัติผู้สมัคร", "มัธยมศึกษาปีที่ 6",
                           "ประกาศนียบัตรวิชาชีพ", "ประกาศนียบัตรวิชาชีพชั้นสูง"}
        applicant_intent = bool(query_terms & applicant_terms) and bool(re.search(
            r"สมัคร|เรียน|ผู้สมัคร|คุณสมบัติ|เข้า|รับ|สิทธิ์|จบ", question))
        if applicant_intent:
            eligibility = next((i for i, chunk in enumerate(self.chunks)
                                if chunk.category == 6 and
                                "คุณสมบัติของผู้เข้าศึกษา" in chunk.heading), None)
            if eligibility is not None:
                selected = [eligibility] + [i for i in selected if i != eligibility]
                selected = selected[:top_k]
        faculty_terms = {"อาจารย์ในภาควิชา", "อาจารย์ประจำหลักสูตร", "อาจารย์ผู้สอน"}
        faculty_intent = bool(query_terms & faculty_terms) or bool(re.search(
            r"อาจารย์.*(ทั้งหมด|กี่คน|รายชื่อ|มีใครบ้าง)|รายชื่อ.*อาจารย์", question))
        if faculty_intent:
            roster = next((i for i, chunk in enumerate(self.chunks)
                           if chunk.category == 5 and
                           "อาจารย์ในภาควิชา" in chunk.heading), None)
            if roster is not None:
                selected = [roster] + [i for i in selected if i != roster]
                selected = selected[:top_k]
        financial_terms = {"กยศ.", "ทุนการศึกษา", "สามารถกู้"}
        if query_terms & financial_terms:
            external = [i for i, chunk in enumerate(self.chunks) if chunk.kind == "external"]
            selected = external + [i for i in selected if i not in external]
            selected = selected[:top_k]
        # Include the full page when compact: a plan/table split into chunks must
        # not lose its remaining course rows or adjacent credit categories.
        results, seen_pages = [], set()
        for i in selected:
            chunk = self.chunks[i]
            if chunk.pdf_page in seen_pages:
                continue
            same_page = [c for c in self.chunks if c.pdf_page == chunk.pdf_page]
            page_text = "\n\n".join(c.text for c in same_page)
            if len(page_text) <= 7500:
                chunk = Chunk(chunk.id, page_text, chunk.category, chunk.printed_page,
                              chunk.pdf_page, chunk.status, chunk.kind, chunk.heading, chunk.source)
            else:
                page_text = "\n\n".join(self.chunks[j].text for j in selected
                                        if self.chunks[j].pdf_page == chunk.pdf_page)
                chunk = Chunk(chunk.id, page_text, chunk.category, chunk.printed_page,
                              chunk.pdf_page, chunk.status, chunk.kind, chunk.heading, chunk.source)
            results.append((chunk, float(semantic[i])))
            seen_pages.add(chunk.pdf_page)
        return results


def load_index(source: Path, synonyms_path: Path, cache_dir: Path, client,
               model: str = EMBEDDING_MODEL, dimensions: int = DIMENSIONS,
               progress: Callable[[int, int], None] | None = None,
               force_rebuild: bool = False) -> CurriculumIndex:
    identity = hashlib.sha256(source.read_bytes() + f"{PARSER_VERSION}:{model}:{dimensions}".encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{identity}.npz"
    rebuild_dir = cache_dir / f"{identity}.rebuild"
    chunks, vectors = parse_curriculum(source), None
    if cache.exists() and not force_rebuild:
        try:
            with np.load(cache, allow_pickle=False) as saved:
                candidate = saved["vectors"]
                if candidate.shape == (len(chunks), dimensions) and np.isfinite(candidate).all():
                    vectors = candidate
        except (ValueError, OSError, KeyError):
            pass
    if vectors is None:
        vectors = np.empty((len(chunks), dimensions), dtype=np.float32)
        filled = np.zeros(len(chunks), dtype=bool)

        if force_rebuild:
            # A full rebuild can span daily API quotas. Keep only vectors created
            # by this rebuild so the next --rebuild invocation can resume safely.
            rebuild_dir.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(chunks), 16):
                stop = min(start + 16, len(chunks))
                part_path = rebuild_dir / f"{start}.npy"
                if not part_path.exists():
                    continue
                try:
                    candidate = np.load(part_path, allow_pickle=False)
                    if candidate.shape == (stop - start, dimensions) and np.isfinite(candidate).all():
                        vectors[start:stop] = candidate
                        filled[start:stop] = True
                except (ValueError, OSError):
                    pass
        else:
            # Reuse embeddings for identical source content even when another page was edited.
            reusable = cached_chunk_vectors(cache_dir, cache, model, dimensions)
            for i, chunk in enumerate(chunks):
                vector = reusable.get(hashlib.sha256(embedding_text(chunk).encode("utf-8")).hexdigest())
                if vector is not None:
                    vectors[i] = vector
                    filled[i] = True
        if progress:
            progress(int(filled.sum()), len(chunks))

        missing = np.flatnonzero(~filled).tolist()
        interval = max(0, float(os.getenv("RAG_EMBED_BATCH_INTERVAL", "13")))
        for batch_number, start in enumerate(range(0, len(missing), 16)):
            indices = missing[start:start + 16]
            # Space requests for rate-limited accounts; no delay before the first request.
            if batch_number:
                time.sleep(interval)
            texts = [embedding_text(chunks[i]) for i in indices]
            part = embed(client, texts, model, "RETRIEVAL_DOCUMENT", dimensions)
            vectors[indices] = part
            filled[indices] = True
            if force_rebuild:
                part_path = rebuild_dir / f"{indices[0]}.npy"
                temporary_part = part_path.with_suffix(f".{uuid4().hex}.tmp")
                with temporary_part.open("wb") as file:
                    np.save(file, part, allow_pickle=False)
                temporary_part.replace(part_path)
            if progress:
                progress(int(filled.sum()), len(chunks))
        temporary = cache.with_suffix(f".{uuid4().hex}.tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, vectors=vectors)
        temporary.replace(cache)
        cache.with_suffix(".json").write_text(json.dumps({
            "model": model, "dimensions": dimensions, "parser_version": PARSER_VERSION,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "chunks": [asdict(c) for c in chunks]}, ensure_ascii=False), encoding="utf-8")
    # Keep a human-readable copy alongside the project so the generated
    # embedding is inspectable. The hashed cache remains the runtime source.
    public_dir = source.parent / "embeddings"
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache, public_dir / "curriculum_embeddings.npz")
    manifest_source = cache.with_suffix(".json")
    if manifest_source.exists():
        shutil.copyfile(manifest_source, public_dir / "curriculum_metadata.json")
    if force_rebuild and rebuild_dir.exists() and rebuild_dir.resolve().parent == cache_dir.resolve():
        shutil.rmtree(rebuild_dir)
    return CurriculumIndex(chunks, vectors, Synonyms(synonyms_path), model, dimensions)


def main():
    import os
    from dotenv import load_dotenv
    from google import genai
    base = Path(__file__).resolve().parent
    load_dotenv(base / ".env")
    parser = argparse.ArgumentParser(description="สร้าง embedding ของหลักสูตร หรือทดลองค้นข้อมูล")
    parser.add_argument("--query")
    parser.add_argument("--rebuild", action="store_true",
                        help="สร้าง embedding ใหม่ทั้งหมดโดยไม่ใช้เวกเตอร์จาก cache เดิม")
    args = parser.parse_args()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_INSURVERSE")
    if not key:
        raise SystemExit("กรุณาตั้งค่า GEMINI_API_KEY ใน workaw/.env")
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000)) as client:
        index = load_index(base / "TCT69_Curriculum.md", base / "synonyms.json", base / ".rag_cache", client,
                           model=os.getenv("GEMINI_EMBEDDING_MODEL", EMBEDDING_MODEL),
                           progress=lambda done, total: print(f"Embedding {done}/{total}", flush=True),
                           force_rebuild=args.rebuild)
        print(f"Index ready: {len(index.chunks)} chunks, {index.dimensions} dimensions")
        if args.query:
            for chunk, score in index.search(client, args.query):
                print(f"{score:.3f} {chunk.label}\n{chunk.text[:500]}\n")


if __name__ == "__main__":
    main()
