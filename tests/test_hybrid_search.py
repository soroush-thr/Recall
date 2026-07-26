"""Phase 3: dense embeddings + RRF fusion. Uses a small model (not bge-m3) so the
suite doesn't pull a multi-GB download; embedder.py itself is model-agnostic."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_project_card

from recall.indexer import build_index
from recall.search import search

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

pytest.importorskip("sentence_transformers")


def _populate(vault_path: Path):
    make_project_card(
        vault_path,
        "prj-routing-2023",
        "Fleet Routing Optimizer",
        summary="A system that plans efficient delivery routes for a truck fleet using constraint solvers.",
        tags=["optimization"],
        tech=["python", "ortools"],
    )
    make_project_card(
        vault_path,
        "prj-unrelated-2022",
        "Photo Album App",
        summary="A mobile app for organizing and sharing personal photo albums.",
        tags=["mobile"],
        tech=["swift"],
    )


def test_dense_embeddings_stored_with_model_name(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    stats = build_index(vault_path, db_path, embed=True, embedding_model=TEST_MODEL)
    assert stats.embedded > 0

    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT model FROM embeddings").fetchall()
        assert rows == [(TEST_MODEL,)]
    finally:
        conn.close()


def test_dense_retrieval_finds_semantic_match_without_keyword_overlap(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path, embed=True, embedding_model=TEST_MODEL)

    # No shared keywords with the routing card's summary, only conceptual overlap.
    hits = search(db_path, "logistics shipment path planning software", embedding_model=TEST_MODEL)
    assert hits
    assert hits[0].doc_id == "prj-routing-2023"


def test_reindex_skips_unchanged_docs_with_matching_embeddings(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path, embed=True, embedding_model=TEST_MODEL)
    stats = build_index(vault_path, db_path, embed=True, embedding_model=TEST_MODEL)
    assert stats.embedded == 0
    assert stats.unchanged == 2


def test_model_change_triggers_reembed_without_touching_lexical_data(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path, embed=True, embedding_model=TEST_MODEL)
    stats = build_index(vault_path, db_path, embed=True, embedding_model="sentence-transformers/all-MiniLM-L12-v2")
    assert stats.embedded > 0
    assert stats.added == 0
    assert stats.updated == 0


def test_search_without_embeddings_falls_back_to_lexical_only(vault_path: Path):
    """No dense embeddings computed -> search still works, no model load required."""
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)  # embed=False by default

    hits = search(db_path, "delivery routes truck fleet")
    assert hits
    assert hits[0].doc_id == "prj-routing-2023"
