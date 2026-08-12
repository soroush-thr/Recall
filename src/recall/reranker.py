"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3), lazily loaded.

Follows the same lazy-model-load pattern as embedder.py: the model is only
imported/downloaded the first time `rerank()` is actually called, and cached
in-process by model name after that. Callers that never pass --rerank never
pay the load cost.
"""

from __future__ import annotations

import sys

_MODEL_CACHE: dict[str, object] = {}

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        # local_files_only avoids a network round-trip (and an unbounded hang on a
        # flaky connection) when the model is already cached; only downloads on miss.
        try:
            print(f"[recall] loading {model_name} from local cache...", file=sys.stderr, flush=True)
            model = CrossEncoder(model_name, local_files_only=True)
        except OSError:
            print(f"[recall] {model_name} not cached locally, downloading...", file=sys.stderr, flush=True)
            model = CrossEncoder(model_name)
        print(f"[recall] {model_name} ready.", file=sys.stderr, flush=True)
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def rerank(
    query: str, passages: list[str], model_name: str = DEFAULT_RERANKER_MODEL
) -> list[float]:
    """Score each (query, passage) pair. Higher score = more relevant. Order preserved."""
    if not passages:
        return []
    model = _get_model(model_name)
    pairs = [[query, passage] for passage in passages]
    scores = model.predict(pairs)
    return [float(s) for s in scores]
