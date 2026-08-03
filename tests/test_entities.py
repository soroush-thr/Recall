from __future__ import annotations

from pathlib import Path

from conftest import make_project_card

from recall.db import connect
from recall.entities import merge_entities
from recall.indexer import build_index
from recall.vault import load_card


def test_merge_entities_rewrites_frontmatter_references(vault_path: Path):
    path = make_project_card(vault_path, "prj-with-person", "Project With Person")
    card = load_card(path)
    updated = card.card.model_copy(
        update={"entities": card.card.entities.model_copy(update={"people": ["person-a-dupe"]})}
    )
    from recall.vault import save_card

    save_card(vault_path, updated, card.body)

    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    rewritten = merge_entities(vault_path, db_path, "person-a-dupe", "person-a", embedding_model="BAAI/bge-m3")

    assert rewritten == ["prj-with-person"]
    reloaded = load_card(path)
    assert reloaded.card.entities.people == ["person-a"]


def test_merge_entities_merges_mentions_rows(vault_path: Path):
    make_project_card(vault_path, "prj-x", "Project X")
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    conn = connect(db_path)
    conn.execute(
        "INSERT INTO entities (id, kind, canonical, aliases) VALUES (?, ?, ?, ?)",
        ("person-a-dupe", "person", "Person A", None),
    )
    conn.execute(
        "INSERT INTO entities (id, kind, canonical, aliases) VALUES (?, ?, ?, ?)",
        ("person-a", "person", "Person A", None),
    )
    conn.execute(
        "INSERT INTO mentions (doc_id, entity_id, count) VALUES (?, ?, ?)",
        ("prj-x", "person-a-dupe", 3),
    )
    conn.execute(
        "INSERT INTO mentions (doc_id, entity_id, count) VALUES (?, ?, ?)",
        ("prj-x", "person-a", 2),
    )
    conn.commit()
    conn.close()

    merge_entities(vault_path, db_path, "person-a-dupe", "person-a", embedding_model="BAAI/bge-m3")

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT entity_id, count FROM mentions WHERE doc_id = 'prj-x'"
        ).fetchall()
        assert [dict(r) for r in rows] == [{"entity_id": "person-a", "count": 5}]
        assert conn.execute(
            "SELECT * FROM entities WHERE id = 'person-a-dupe'"
        ).fetchone() is None
    finally:
        conn.close()


def test_merge_entities_rejects_same_id(vault_path: Path):
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)
    import pytest

    with pytest.raises(ValueError):
        merge_entities(vault_path, db_path, "same", "same")
