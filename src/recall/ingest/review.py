"""Stage C: the human review gate. Non-negotiable — nothing enters memory unreviewed."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from recall import db as db_module
from recall.indexer import build_index
from recall.schema import FIXED_SECTIONS, parse_card
from recall.vault import card_path

UNKNOWN_RE = re.compile(r"UNKNOWN\s*—")
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class UnknownMarkersRemain(Exception):
    def __init__(self, unknowns: list[str]):
        self.unknowns = unknowns
        super().__init__(
            f"{len(unknowns)} UNKNOWN marker(s) remain; fix them or pass --allow-unknown:\n"
            + "\n".join(f"  - {u}" for u in unknowns)
        )


@dataclass
class DraftSummary:
    sections_present: list[str] = field(default_factory=list)
    sections_missing: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    started: str | None = None
    ended: str | None = None
    tags: list[str] = field(default_factory=list)
    tech: list[str] = field(default_factory=list)


def find_unknowns(body: str) -> list[str]:
    return [line.strip() for line in body.splitlines() if UNKNOWN_RE.search(line)]


def inspect_draft(draft_path: Path) -> DraftSummary:
    post = frontmatter.loads(draft_path.read_text(encoding="utf-8"))
    doc_type = post.metadata.get("type")
    expected = FIXED_SECTIONS.get(doc_type, [])
    present = SECTION_RE.findall(post.content)
    return DraftSummary(
        sections_present=[s for s in expected if s in present],
        sections_missing=[s for s in expected if s not in present],
        unknowns=find_unknowns(post.content),
        started=post.metadata.get("started"),
        ended=post.metadata.get("ended"),
        tags=post.metadata.get("tags") or [],
        tech=post.metadata.get("tech") or [],
    )


def _git_commit_in_vault(vault_path: Path, rel_path: Path, message: str) -> bool:
    """Best-effort git add+commit in the vault repo. Returns False if vault isn't a git repo."""
    if not (vault_path / ".git").exists():
        return False
    subprocess.run(["git", "-C", str(vault_path), "add", str(rel_path)], check=False)
    subprocess.run(["git", "-C", str(vault_path), "commit", "-m", message], check=False)
    return True


def commit_draft(settings, slug: str, *, allow_unknown: bool = False, embed: bool = True) -> Path:
    """Validate a reviewed draft and commit it into the vault + index.

    Raises UnknownMarkersRemain if `UNKNOWN —` markers remain and allow_unknown is False.
    Raises whatever pydantic/ValueError the frontmatter fails schema validation with.
    """
    draft_path = settings.drafts_dir / f"{slug}.md"
    if not draft_path.exists():
        raise FileNotFoundError(f"No draft for {slug!r}; run 'recall draft' first.")

    text = draft_path.read_text(encoding="utf-8")
    post = frontmatter.loads(text)

    unknowns = find_unknowns(post.content)
    if unknowns and not allow_unknown:
        raise UnknownMarkersRemain(unknowns)

    card = parse_card(dict(post.metadata))
    if card.id != slug:
        raise ValueError(f"draft id {card.id!r} does not match slug {slug!r}")

    dest = card_path(settings.vault_path, card.type, card.id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    draft_path.unlink()

    rel_path = dest.relative_to(settings.vault_path)
    _git_commit_in_vault(settings.vault_path, rel_path, f"memory: add {card.title}")

    conn = db_module.connect(settings.db_path)
    try:
        db_module.record_ingest_status(conn, doc_id=slug, source=str(dest), status="committed")
    finally:
        conn.close()

    build_index(
        settings.vault_path,
        settings.db_path,
        doc_id=slug,
        embed=embed,
        embedding_model=settings.embedding_model,
    )
    return dest
