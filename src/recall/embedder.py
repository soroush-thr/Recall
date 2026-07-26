"""sentence-transformers wrapper for dense (vector) embeddings.

BGE-M3 needs no query/passage instruction prefixes (unlike e5-style models), so
embed_passages and embed_query are identical calls to the underlying model.
Models are cached in-process by name so repeated calls (indexing, then search)
don't reload weights.
"""

from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_passages(texts: list[str], model_name: str) -> list[np.ndarray]:
    """Embed a batch of chunk texts. Returns one float32 vector per input text."""
    if not texts:
        return []
    model = _get_model(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def embed_query(text: str, model_name: str) -> np.ndarray:
    """Embed a single search query."""
    model = _get_model(model_name)
    vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(vector, dtype=np.float32)


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
