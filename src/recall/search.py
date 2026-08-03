"""Hybrid (lexical + dense) search over the index, fused with RRF. See build plan §7."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recall.db import connect

RRF_K = 60
TOP_N_PER_RANKER = 50
RERANK_TOP_N = 20


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
    embedding_model: str = "BAAI/bge-m3",
    rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
) -> list[SearchHit]:
    conn = connect(db_path)
    try:
        return _search(
            conn,
            query,
            doc_type=doc_type,
            tag=tag,
            date_from=date_from,
            date_to=date_to,
            k=k,
            embedding_model=embedding_model,
            rerank=rerank,
            rerank_model=rerank_model,
        )
    finally:
        conn.close()


def _doc_filter(
    *, doc_type: str | None, tag: str | None, date_from: str | None, date_to: str | None
) -> tuple[str, list]:
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
    return " AND ".join(filters), params


def _lexical_ranked_chunks(
    conn: sqlite3.Connection, query: str, where_clause: str, params: list
) -> list[sqlite3.Row]:
    sql = f"""
        SELECT
          c.chunk_id AS chunk_id, d.id AS doc_id, d.title AS title, d.type AS type,
          d.started AS started, d.ended AS ended, d.path AS path,
          c.text AS snippet,
          bm25(chunks_fts) AS raw_score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        JOIN documents d ON d.id = c.doc_id
        WHERE chunks_fts MATCH ? AND {where_clause}
        ORDER BY raw_score
        LIMIT {TOP_N_PER_RANKER}
    """
    return conn.execute(sql, [_escape_fts_query(query), *params]).fetchall()


def _dense_ranked_chunks(
    conn: sqlite3.Connection,
    query: str,
    where_clause: str,
    params: list,
    embedding_model: str,
) -> list[tuple[str, str, sqlite3.Row, float]]:
    """Returns (chunk_id, doc_id, chunk_row, cosine_score) sorted best-first, top N."""
    count_sql = f"""
        SELECT COUNT(*) AS n
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        JOIN documents d ON d.id = c.doc_id
        WHERE e.model = ? AND {where_clause}
    """
    n = conn.execute(count_sql, [embedding_model, *params]).fetchone()["n"]
    if not n:
        return []

    from recall.embedder import blob_to_vector, embed_query

    query_vec = embed_query(query, embedding_model)

    rows_sql = f"""
        SELECT
          c.chunk_id AS chunk_id, d.id AS doc_id, d.title AS title, d.type AS type,
          d.started AS started, d.ended AS ended, d.path AS path,
          c.text AS snippet, e.vector AS vector
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        JOIN documents d ON d.id = c.doc_id
        WHERE e.model = ? AND {where_clause}
    """
    rows = conn.execute(rows_sql, [embedding_model, *params]).fetchall()

    scored = []
    for row in rows:
        vec = blob_to_vector(row["vector"])
        cos = float(np.dot(query_vec, vec))  # both L2-normalized -> dot == cosine
        scored.append((row["chunk_id"], row["doc_id"], row, cos))
    scored.sort(key=lambda t: t[3], reverse=True)
    return scored[:TOP_N_PER_RANKER]


def _search(
    conn: sqlite3.Connection,
    query: str,
    *,
    doc_type: str | None,
    tag: str | None,
    date_from: str | None,
    date_to: str | None,
    k: int,
    embedding_model: str,
    rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
) -> list[SearchHit]:
    where_clause, params = _doc_filter(
        doc_type=doc_type, tag=tag, date_from=date_from, date_to=date_to
    )

    lexical_rows = _lexical_ranked_chunks(conn, query, where_clause, params)
    dense_rows = _dense_ranked_chunks(conn, query, where_clause, params, embedding_model)

    rrf_scores: dict[str, float] = {}
    chunk_meta: dict[str, sqlite3.Row] = {}
    doc_of_chunk: dict[str, str] = {}

    for rank, row in enumerate(lexical_rows, start=1):
        cid = row["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_meta.setdefault(cid, row)
        doc_of_chunk[cid] = row["doc_id"]

    for rank, (cid, doc_id, row, _cos) in enumerate(dense_rows, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_meta.setdefault(cid, row)
        doc_of_chunk[cid] = doc_id

    best_per_doc: dict[str, sqlite3.Row] = {}
    doc_scores: dict[str, float] = {}
    doc_match_counts: dict[str, int] = {}
    for cid, score in rrf_scores.items():
        doc_id = doc_of_chunk[cid]
        doc_match_counts[doc_id] = doc_match_counts.get(doc_id, 0) + 1
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score
            best_per_doc[doc_id] = chunk_meta[cid]

    ranked = sorted(
        doc_scores.keys(),
        key=lambda d: doc_scores[d] + 0.1 * min(doc_match_counts[d], 5),
        reverse=True,
    )

    final_scores = dict(doc_scores)
    if rerank and ranked:
        from recall.reranker import rerank as rerank_fn

        candidates = ranked[:RERANK_TOP_N]
        texts = [best_per_doc[d]["snippet"] for d in candidates]
        rerank_scores = rerank_fn(query, texts, rerank_model)
        for doc_id, score in zip(candidates, rerank_scores):
            final_scores[doc_id] = score
        candidates.sort(key=lambda d: final_scores[d], reverse=True)
        ranked = candidates + [d for d in ranked if d not in candidates]

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
                score=final_scores[doc_id],
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
