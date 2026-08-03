"""`recall entity merge <id-a> <id-b>`: manual, deterministic entity merge. See build plan §8 —
no automated resolution, this only folds one confirmed-duplicate id into another.
"""

from __future__ import annotations

from pathlib import Path

from recall.db import connect
from recall.indexer import build_index
from recall.vault import iter_card_paths, load_card, save_card


def merge_entities(
    vault_path: Path, db_path: Path, id_a: str, id_b: str, *, embedding_model: str = "BAAI/bge-m3"
) -> list[str]:
    """Fold entity id_a into id_b.

    Rewrites `mentions` rows (merging counts on collision), deletes the id_a `entities` row,
    and rewrites any frontmatter entities.{people,orgs,projects} reference to id_a in every
    vault card so it points at id_b instead. Returns the doc ids whose card was rewritten.
    """
    if id_a == id_b:
        raise ValueError("id-a and id-b must be different")

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT doc_id, count FROM mentions WHERE entity_id = ?", (id_a,)
        ).fetchall()
        for row in rows:
            existing = conn.execute(
                "SELECT count FROM mentions WHERE doc_id = ? AND entity_id = ?",
                (row["doc_id"], id_b),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE mentions SET count = ? WHERE doc_id = ? AND entity_id = ?",
                    (existing["count"] + row["count"], row["doc_id"], id_b),
                )
                conn.execute(
                    "DELETE FROM mentions WHERE doc_id = ? AND entity_id = ?",
                    (row["doc_id"], id_a),
                )
            else:
                conn.execute(
                    "UPDATE mentions SET entity_id = ? WHERE doc_id = ? AND entity_id = ?",
                    (id_b, row["doc_id"], id_a),
                )
        conn.execute("DELETE FROM entities WHERE id = ?", (id_a,))
        conn.commit()
    finally:
        conn.close()

    rewritten: list[str] = []
    for path in iter_card_paths(vault_path):
        card = load_card(path)
        ent = card.card.entities
        changed = False
        updates = {}
        for field_name in ("people", "orgs", "projects"):
            values = getattr(ent, field_name)
            if id_a in values:
                updates[field_name] = [id_b if v == id_a else v for v in values]
                changed = True
        if changed:
            new_entities = ent.model_copy(update=updates)
            new_card = card.card.model_copy(update={"entities": new_entities})
            save_card(vault_path, new_card, card.body)
            rewritten.append(new_card.id)
            build_index(
                vault_path, db_path, doc_id=new_card.id, embed=False, embedding_model=embedding_model
            )

    return rewritten
