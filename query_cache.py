"""Portable FAQ vectors and Streamlit disk cache; no external database."""
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from functools import lru_cache
from uuid import uuid4

import numpy as np
import streamlit as st

from rag import embed, normalize

CACHE_VERSION = 1
TASK = "RETRIEVAL_QUERY"
BASE = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def normalize_question(question):
    return re.sub(r"\s+", " ", normalize(question)).strip().rstrip("?!！？.。… ")


def questions():
    return json.loads((BASE / "seed_questions.json").read_text(encoding="utf-8"))


def canonicalize(question):
    normalized = normalize_question(question)
    return alias_lookup((BASE/'seed_questions.json').read_text(encoding='utf-8')).get(normalized, normalized)


@lru_cache(maxsize=8)
def alias_lookup(entries_json):
    lookup = {}
    for entry in json.loads(entries_json):
        canonical = normalize_question(entry['question'])
        for form in [entry['question'], *entry['aliases']]:
            key = normalize_question(form)
            if key in lookup and lookup[key] != canonical:
                raise ValueError('Conflicting seed alias')
            lookup[key] = canonical
    return lookup


def valid_vector(vector, dimensions):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (dimensions,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Invalid query vector")
    return vector / np.linalg.norm(vector)


def manifest(synonyms, model, dimensions):
    return cached_manifest(json.dumps(synonyms.groups, ensure_ascii=False),
                           (BASE/'seed_questions.json').read_text(encoding='utf-8'), model, dimensions, CACHE_VERSION)


@lru_cache(maxsize=8)
def cached_manifest(groups_json, entries_json, model, dimensions, version):
    from rag import Synonyms
    synonyms = Synonyms.__new__(Synonyms)
    synonyms.groups = json.loads(groups_json)
    entries = json.loads(entries_json)
    return {'cache_version': version, 'model': model, 'dimensions': dimensions,
            'task': TASK, 'questions': entries,
            'texts': [synonyms.expand(normalize_question(e['question'])) for e in entries]}


@st.cache_data(show_spinner=False)
def load_seed(expected_json, vector_bytes, metadata_bytes):
    import io
    expected = json.loads(expected_json)
    if json.loads(metadata_bytes) != expected:
        return {}
    with np.load(io.BytesIO(vector_bytes), allow_pickle=False) as saved:
        vectors = saved['vectors']
    if vectors.shape != (len(expected['texts']), expected['dimensions']):
        return {}
    return {text: valid_vector(v, expected['dimensions']) for text, v in zip(expected['texts'], vectors)}


@st.cache_data(persist="disk", max_entries=2000, show_spinner=False)
def runtime_vector(version, model, dimensions, task, text, _client, _marker):
    # Underscore arguments are deliberately excluded from cache identity.
    _marker['miss'] = True
    if _client is None:
        raise ValueError('Embedding credentials unavailable')
    return valid_vector(embed(_client, [text], model, task, dimensions)[0], dimensions)


def resolve_vector(index, question, client):
    started = time.perf_counter()
    canonical = canonicalize(question)
    text = index.synonyms.expand(canonical)
    vector, origin = None, 'bm25_fallback'
    try:
        expected = manifest(index.synonyms, index.model, index.dimensions)
        vector = load_seed(json.dumps(expected, ensure_ascii=False, sort_keys=True),
                           (BASE/'embeddings/query_embeddings.npz').read_bytes(),
                           (BASE/'embeddings/query_metadata.json').read_bytes()).get(text)
        if vector is not None:
            origin = 'seed'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if vector is None:
        marker = {}
        try:
            # A hit works without a client; a miss without credentials falls back.
            vector = runtime_vector(CACHE_VERSION, index.model, index.dimensions, TASK, text, client, marker)
            origin = 'gemini' if marker.get('miss') else 'runtime'
        except Exception:
            # Provider details can contain request text; never log the exception.
            vector = None
    LOGGER.info('retrieval source=%s elapsed_ms=%.1f', origin, (time.perf_counter()-started)*1000)
    return canonical, vector


def build_seed(client, synonyms, model, dimensions):
    expected = manifest(synonyms, model, dimensions)
    destination = BASE/'embeddings'
    destination.mkdir(exist_ok=True)
    # Resume completed batches without spending their quota again.
    identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    parts = BASE/'.rag_cache'/('query-'+identity)
    parts.mkdir(parents=True, exist_ok=True)
    vectors = []
    for start in range(0, len(expected['texts']), 16):
        path = parts/f'{start}.npz'
        batch = expected['texts'][start:start+16]
        try:
            with np.load(path, allow_pickle=False) as saved:
                part = saved['vectors']
            if len(part) != len(batch):
                raise ValueError('Incomplete batch')
            part = np.stack([valid_vector(v, dimensions) for v in part])
        except (OSError, ValueError, KeyError):
            if start:
                time.sleep(13)
            part = embed(client, batch, model, TASK, dimensions)
            temporary = path.with_suffix(f'.{uuid4().hex}.tmp')
            with temporary.open('wb') as file:
                np.savez_compressed(file, vectors=part)
            temporary.replace(path)
        vectors.extend(part)
    target = destination/'query_embeddings.npz'
    temporary = target.with_suffix(f'.{uuid4().hex}.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, vectors=np.stack(vectors))
    temporary.replace(target)
    target = destination/'query_metadata.json'
    temporary = target.with_suffix(f'.{uuid4().hex}.tmp')
    temporary.write_text(json.dumps(expected, ensure_ascii=False), encoding='utf-8')
    temporary.replace(target)
    print(f'Seed ready: {len(vectors)} vectors / 100 question forms')
