from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import make_project_card

from recall.db import connect
from recall.indexer import build_index


@pytest.fixture
def mcp_vault(vault_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RECALL_VAULT", str(vault_path))
    monkeypatch.delenv("RECALL_MCP_PUBLIC_ONLY", raising=False)
    make_project_card(
        vault_path,
        "prj-marl-inventory-2024",
        "Decentralized MARL Inventory Management",
        summary="A multi-agent reinforcement learning system for inventory management.",
        tags=["reinforcement-learning", "marl"],
        tech=["pytorch", "python"],
        started="2024-01-01",
    )
    make_project_card(
        vault_path,
        "prj-forecasta-2022",
        "Forecasta Energy Forecasting",
        summary="Time-series forecasting for energy demand.",
        tags=["forecasting"],
        tech=["python"],
        started="2022-01-01",
    )
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)
    return vault_path


def test_memory_search_returns_ranked_hits(mcp_vault: Path):
    from recall.mcp_server import memory_search

    results = memory_search("reinforcement learning inventory")
    assert results
    assert results[0]["doc_id"] == "prj-marl-inventory-2024"
    assert "path" in results[0] and "snippet" in results[0]


def test_memory_get_returns_full_card(mcp_vault: Path):
    from recall.mcp_server import memory_get

    text = memory_get("prj-forecasta-2022")
    assert "Forecasta Energy Forecasting" in text
    assert "Time-series forecasting" in text


def test_memory_get_unknown_id_raises(mcp_vault: Path):
    from recall.mcp_server import memory_get

    with pytest.raises(ValueError):
        memory_get("prj-does-not-exist")


def test_memory_stats_reports_counts_and_span(mcp_vault: Path):
    from recall.mcp_server import memory_stats

    stats = memory_stats()
    assert stats["total"] == 2
    assert stats["by_type"]["project"] == 2
    assert stats["earliest"] == "2022-01-01"
    assert stats["latest"] == "2024-01-01"


def test_memory_get_excludes_confidential(mcp_vault: Path):
    from recall.mcp_server import memory_get

    db_path = mcp_vault / ".recall" / "index.db"
    conn = connect(db_path)
    conn.execute(
        "UPDATE documents SET visibility = 'confidential' WHERE id = ?",
        ("prj-forecasta-2022",),
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError):
        memory_get("prj-forecasta-2022")


def test_public_only_mode_excludes_private(mcp_vault: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RECALL_MCP_PUBLIC_ONLY", "1")
    from recall.mcp_server import memory_get, memory_search, memory_stats

    with pytest.raises(ValueError):
        memory_get("prj-forecasta-2022")

    results = memory_search("forecasting")
    assert all(r["doc_id"] != "prj-forecasta-2022" for r in results)

    stats = memory_stats()
    assert stats["total"] == 0
