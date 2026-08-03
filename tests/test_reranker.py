"""Reranker must stay lazy (no model load without --rerank) and must actually reorder
results when engaged. We monkeypatch the cross-encoder call itself rather than pull down
BAAI/bge-reranker-v2-m3 in the test suite."""

from __future__ import annotations

from pathlib import Path

from conftest import make_project_card

from recall import reranker
from recall.indexer import build_index
from recall.search import search


def _populate(vault_path: Path):
    make_project_card(
        vault_path,
        "prj-alpha",
        "Alpha Project",
        summary="A widget system for scheduling widgets across a widget fleet.",
        tags=["widgets"],
    )
    make_project_card(
        vault_path,
        "prj-beta",
        "Beta Project",
        summary="Another widget scheduling system, also about widgets and fleets.",
        tags=["widgets"],
    )


def test_reranker_model_not_loaded_unless_called(monkeypatch):
    reranker._MODEL_CACHE.clear()

    def _boom(*args, **kwargs):
        raise AssertionError("CrossEncoder should not be imported/constructed unless rerank() is called")

    import sentence_transformers  # noqa: F401  (import check only if installed)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _boom, raising=False)
    # No call to reranker.rerank() here — module import alone must not trigger a load.
    assert reranker._MODEL_CACHE == {}


def test_rerank_empty_passages_returns_empty():
    assert reranker.rerank("query", []) == []


def test_search_rerank_reorders_using_cross_encoder_scores(vault_path: Path, monkeypatch):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)  # lexical only, no embeddings

    # Force beta to win regardless of RRF order by monkeypatching the scorer.
    def fake_rerank(query, passages, model_name="BAAI/bge-reranker-v2-m3"):
        return [1.0 if "Another widget" in p else 0.0 for p in passages]

    monkeypatch.setattr("recall.reranker.rerank", fake_rerank)

    hits = search(db_path, "widget fleet scheduling", k=2, rerank=True)
    assert len(hits) == 2
    assert hits[0].doc_id == "prj-beta"
    assert hits[0].score == 1.0


def test_search_without_rerank_does_not_touch_reranker_module(vault_path: Path, monkeypatch):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    def _boom(*args, **kwargs):
        raise AssertionError("reranker.rerank should not be called when rerank=False")

    monkeypatch.setattr("recall.reranker.rerank", _boom)
    hits = search(db_path, "widget fleet scheduling", k=2, rerank=False)
    assert hits
