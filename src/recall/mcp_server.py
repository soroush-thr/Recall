"""MCP server exposing the Phase-1 index to any MCP-speaking client (e.g. Claude Code).

Three tools: memory_search (BM25), memory_get (full card by id), memory_stats
(coverage summary). Read-only — the server never writes to the vault. Public-only
mode (RECALL_MCP_PUBLIC_ONLY=1) additionally excludes visibility="private" cards,
on top of the "confidential" exclusion search() already applies unconditionally.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from recall.config import load_settings
from recall.db import connect
from recall.search import search as run_search
from recall.vault import TYPE_DIRS

mcp = FastMCP("recall")


def _public_only() -> bool:
    return os.environ.get("RECALL_MCP_PUBLIC_ONLY") == "1"


@mcp.tool()
def memory_search(
    query: str,
    type: str | None = None,
    tag: str | None = None,
    from_: str | None = None,
    to: str | None = None,
    k: int = 10,
) -> list[dict]:
    """Search personal memory (projects, people, episodes, notes, artifacts) by keyword.

    Returns ranked hits with doc_id, title, type, dates, a snippet, and the vault path.
    Use memory_get with a hit's doc_id to fetch the full card.
    """
    settings = load_settings()
    hits = run_search(settings.db_path, query, doc_type=type, tag=tag, date_from=from_, date_to=to, k=k)
    if _public_only():
        hits = [h for h in hits if _visibility(settings.db_path, h.doc_id) != "private"]
    return [
        {
            "doc_id": h.doc_id,
            "title": h.title,
            "type": h.type,
            "started": h.started,
            "ended": h.ended,
            "score": h.score,
            "snippet": h.snippet,
            "path": h.path,
        }
        for h in hits
    ]


@mcp.tool()
def memory_get(doc_id: str) -> str:
    """Fetch the full markdown card (frontmatter + body) for a document id.

    Raises if the id doesn't exist, or is confidential/private and excluded
    by the server's visibility mode.
    """
    settings = load_settings()
    for dirname in TYPE_DIRS.values():
        candidate = settings.vault_path / dirname / f"{doc_id}.md"
        if candidate.exists():
            visibility = _visibility(settings.db_path, doc_id)
            if visibility == "confidential" or (_public_only() and visibility == "private"):
                raise ValueError(f"document {doc_id!r} is not accessible in this mode")
            return candidate.read_text(encoding="utf-8")
    raise ValueError(f"no document with id {doc_id!r}")


@mcp.tool()
def memory_stats() -> dict:
    """Summary of what's in memory: counts by card type and the earliest/latest dates covered."""
    settings = load_settings()
    conn = connect(settings.db_path)
    try:
        vis_filter = "visibility != 'confidential'"
        if _public_only():
            vis_filter += " AND visibility != 'private'"
        by_type = conn.execute(
            f"SELECT type, COUNT(*) AS n FROM documents WHERE {vis_filter} GROUP BY type ORDER BY type"
        ).fetchall()
        span = conn.execute(
            f"SELECT MIN(started) AS earliest, MAX(started) AS latest FROM documents WHERE {vis_filter}"
        ).fetchone()
        total = sum(row["n"] for row in by_type)
        return {
            "total": total,
            "by_type": {row["type"]: row["n"] for row in by_type},
            "earliest": span["earliest"],
            "latest": span["latest"],
        }
    finally:
        conn.close()


def _visibility(db_path, doc_id: str) -> str | None:
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT visibility FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row["visibility"] if row else None
    finally:
        conn.close()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
