"""Build/update the derived SQLite index from the vault."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from recall.chunker import chunk_document
from recall.db import connect
from recall.vault import Card, iter_card_paths, load_card

SUMMARY_RE = re.compile(r"^##\s+Summary\s*$(.*?)(^##\s+|\Z)", re.MULTILINE | re.DOTALL)
EMBED_BATCH_SIZE = 32


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_summary(body: str) -> str:
    m = SUMMARY_RE.search(body)
    return m.group(1).strip() if m else ""


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    embedded: int = 0


def _existing_hash(conn: sqlite3.Connection, doc_id: str) -> str | None:
    row = conn.execute("SELECT content_hash FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return row["content_hash"] if row else None


def _stale_embedding_doc_ids(conn: sqlite3.Connection, embedding_model: str) -> set[str]:
    """Doc ids that have at least one chunk missing an embedding for embedding_model."""
    rows = conn.execute(
        """
        SELECT DISTINCT c.doc_id AS doc_id
        FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
        WHERE e.chunk_id IS NULL
        """,
        (embedding_model,),
    ).fetchall()
    return {r["doc_id"] for r in rows}


def _embed_pending(conn: sqlite3.Connection, chunk_ids: list[str], embedding_model: str) -> int:
    """Embed the given chunk_ids in batches, upserting into the embeddings table."""
    import sys

    from recall.embedder import embed_passages, vector_to_blob

    if not chunk_ids:
        return 0
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
    ).fetchall()
    by_id = {r["chunk_id"]: r["text"] for r in rows}
    ordered_ids = [cid for cid in chunk_ids if cid in by_id]
    now = _now()
    embedded = 0
    total_batches = -(-len(ordered_ids) // EMBED_BATCH_SIZE)
    for batch_num, i in enumerate(range(0, len(ordered_ids), EMBED_BATCH_SIZE), start=1):
        batch_ids = ordered_ids[i : i + EMBED_BATCH_SIZE]
        batch_texts = [by_id[cid] for cid in batch_ids]
        print(
            f"[recall] embedding batch {batch_num}/{total_batches} ({len(batch_ids)} chunks)...",
            file=sys.stderr,
            flush=True,
        )
        vectors = embed_passages(batch_texts, embedding_model)
        for chunk_id, vector in zip(batch_ids, vectors):
            conn.execute(
                """
                INSERT INTO embeddings (chunk_id, vector, model, dim, created)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                  vector=excluded.vector, model=excluded.model, dim=excluded.dim, created=excluded.created
                """,
                (chunk_id, vector_to_blob(vector), embedding_model, len(vector), now),
            )
            embedded += 1
    return embedded


def _upsert_document(conn: sqlite3.Connection, card: Card) -> None:
    c = card.card
    fm = card.frontmatter_dict
    conn.execute(
        """
        INSERT INTO documents
          (id, type, subtype, title, lang, path, content_hash, started, ended, status,
           visibility, confidence, last_verified, frontmatter, body, created, updated, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type, subtype=excluded.subtype, title=excluded.title, lang=excluded.lang,
          path=excluded.path, content_hash=excluded.content_hash, started=excluded.started,
          ended=excluded.ended, status=excluded.status, visibility=excluded.visibility,
          confidence=excluded.confidence, last_verified=excluded.last_verified,
          frontmatter=excluded.frontmatter, body=excluded.body, created=excluded.created,
          updated=excluded.updated, indexed_at=excluded.indexed_at
        """,
        (
            c.id, c.type, fm.get("subtype"), c.title, c.lang, str(card.path),
            card.content_hash(), c.started, c.ended, fm.get("status"),
            c.visibility, c.confidence,
            c.last_verified.isoformat() if c.last_verified else None,
            json.dumps(fm, ensure_ascii=False), card.body,
            c.created.isoformat(), c.updated.isoformat(), _now(),
        ),
    )

    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (c.id,))
    conn.execute("DELETE FROM tags WHERE doc_id = ?", (c.id,))

    chunks = chunk_document(
        doc_id=c.id,
        title=c.title,
        doc_type=c.type,
        subtype=fm.get("subtype"),
        started=c.started,
        ended=c.ended,
        tags=c.tags,
        tech=fm.get("tech", []) or [],
        summary=_extract_summary(card.body),
        body=card.body,
    )
    for chunk in chunks:
        chunk_id = f"{c.id}#{chunk.ordinal}"
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, section, ordinal, text, char_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, c.id, chunk.section, chunk.ordinal, chunk.header_text, len(chunk.header_text)),
        )

    for tag in c.tags:
        conn.execute(
            "INSERT OR IGNORE INTO tags (doc_id, tag) VALUES (?, ?)", (c.id, tag)
        )


def build_index(
    vault_path: Path,
    db_path: Path,
    *,
    doc_id: str | None = None,
    embed: bool = False,
    embedding_model: str = "BAAI/bge-m3",
) -> IndexStats:
    """Index the vault (or a single document) into the SQLite store.

    Skips documents whose normalized content_hash is unchanged AND whose chunks
    already have an embedding for `embedding_model` (when embed=True). Changing
    embedding_model triggers a re-embed of every document on the next run with
    embed=True, without touching lexical data.
    """
    conn = connect(db_path)
    stats = IndexStats()
    try:
        seen_ids: set[str] = set()
        stale_embedding_ids = _stale_embedding_doc_ids(conn, embedding_model) if embed else set()
        chunk_ids_to_embed: list[str] = []
        for path in iter_card_paths(vault_path):
            card = load_card(path)
            if doc_id and card.card.id != doc_id:
                continue
            seen_ids.add(card.card.id)
            existing = _existing_hash(conn, card.card.id)
            new_hash = card.content_hash()
            content_changed = existing != new_hash
            needs_reembed = embed and (content_changed or card.card.id in stale_embedding_ids)

            if not content_changed and not needs_reembed:
                stats.unchanged += 1
                continue

            if content_changed:
                _upsert_document(conn, card)
                if existing is None:
                    stats.added += 1
                else:
                    stats.updated += 1
            else:
                stats.unchanged += 1

            if needs_reembed:
                rows = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (card.card.id,)
                ).fetchall()
                chunk_ids_to_embed.extend(r["chunk_id"] for r in rows)

        if embed and chunk_ids_to_embed:
            stats.embedded = _embed_pending(conn, chunk_ids_to_embed, embedding_model)

        if doc_id is None:
            rows = conn.execute("SELECT id FROM documents").fetchall()
            stale = [r["id"] for r in rows if r["id"] not in seen_ids]
            for stale_id in stale:
                conn.execute("DELETE FROM documents WHERE id = ?", (stale_id,))
                stats.removed += 1
        conn.commit()
    finally:
        conn.close()
    return stats
