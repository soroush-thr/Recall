"""Lexical (BM25) search over the index. Hybrid retrieval lands in Phase 3."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from recall.db import connect


@dataclass
class SearchHit:
    doc_id: str
    title: str
    type: str
    started: str | None
    ended: str | None
    score: float
    snippet: str
    path: str


def search(
    db_path: Path,
    query: str,
    *,
    doc_type: str | None = None,
    tag: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    k: int = 10,
) -> list[SearchHit]:
    conn = connect(db_path)
    try:
        return _search(conn, query, doc_type=doc_type, tag=tag, date_from=date_from, date_to=date_to, k=k)
    finally:
        conn.close()


def _search(
    conn: sqlite3.Connection,
    query: str,
    *,
    doc_type: str | None,
    tag: str | None,
    date_from: str | None,
    date_to: str | None,
    k: int,
) -> list[SearchHit]:
    filters = ["d.visibility != 'confidential'"]
    params: list = []

    if doc_type:
        filters.append("d.type = ?")
        params.append(doc_type)
    if date_from:
        filters.append("d.started >= ?")
        params.append(date_from)
    if date_to:
        filters.append("d.started <= ?")
        params.append(date_to)
    if tag:
        filters.append("d.id IN (SELECT doc_id FROM tags WHERE tag = ?)")
        params.append(tag)

    where_clause = " AND ".join(filters)

    sql = f"""
        SELECT
          d.id AS doc_id, d.title AS title, d.type AS type,
          d.started AS started, d.ended AS ended, d.path AS path,
          c.text AS snippet,
          bm25(chunks_fts) AS raw_score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        JOIN documents d ON d.id = c.doc_id
        WHERE chunks_fts MATCH ? AND {where_clause}
        ORDER BY raw_score
        LIMIT 200
    """
    rows = conn.execute(sql, [_escape_fts_query(query), *params]).fetchall()

    best_per_doc: dict[str, sqlite3.Row] = {}
    doc_scores: dict[str, float] = {}
    doc_match_counts: dict[str, int] = {}
    for row in rows:
        doc_id = row["doc_id"]
        score = -row["raw_score"]  # bm25() is lower-is-better; flip so higher is better
        doc_match_counts[doc_id] = doc_match_counts.get(doc_id, 0) + 1
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score
            best_per_doc[doc_id] = row

    ranked = sorted(
        doc_scores.keys(),
        key=lambda d: doc_scores[d] + 0.1 * min(doc_match_counts[d], 5),
        reverse=True,
    )

    hits = []
    for doc_id in ranked[:k]:
        row = best_per_doc[doc_id]
        hits.append(
            SearchHit(
                doc_id=doc_id,
                title=row["title"],
                type=row["type"],
                started=row["started"],
                ended=row["ended"],
                score=doc_scores[doc_id],
                snippet=row["snippet"][:300],
                path=row["path"],
            )
        )
    return hits


def _escape_fts_query(query: str) -> str:
    """Quote each token so FTS5 treats query text as literal terms, not query syntax."""
    tokens = query.split()
    escaped = [f'"{t}"' for t in tokens if t]
    return " OR ".join(escaped) if escaped else '""'
