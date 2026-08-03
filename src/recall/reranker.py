"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3), lazily loaded.

Follows the same lazy-model-load pattern as embedder.py: the model is only
imported/downloaded the first time `rerank()` is actually called, and cached
in-process by model name after that. Callers that never pass --rerank never
pay the load cost.
"""

from __future__ import annotations

_MODEL_CACHE: dict[str, object] = {}

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
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
