"""Inbox triage: deterministic type/title/tag proposals for loose captures in notes/inbox/.

No LLM calls — this is a heuristic pass over plain-text/markdown files dropped in the inbox
(e.g. by a future `recall add`), proposing how to turn them into real cards. The human always
accepts, edits, or defers; nothing is written without a decision. See RECALL-BUILD-PLAN.md §9.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from recall import db as db_module
from recall.indexer import build_index
from recall.schema import Provenance, model_for_type
from recall.vault import card_path

HASHTAG_RE = re.compile(r"(?<!\S)#([a-zA-Z][\w-]*)")
HEADING_RE = re.compile(r"^#+\s*")


def inbox_dir(vault_path: Path) -> Path:
    return vault_path / "notes" / "inbox"


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-+", "-", slug) or "untitled"


def _propose_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return HEADING_RE.sub("", line).strip() or fallback
    return fallback


def _propose_tags(text: str) -> list[str]:
    seen: list[str] = []
    for tag in HASHTAG_RE.findall(text):
        tag = tag.lower()
        if tag not in seen:
            seen.append(tag)
    return seen


@dataclass
class TriageItem:
    path: Path
    text: str
    proposed_type: str = "note"
    proposed_title: str = ""
    proposed_tags: list[str] = field(default_factory=list)
    proposed_id: str = ""


def scan_inbox(vault_path: Path) -> list[TriageItem]:
    """Walk notes/inbox/ and build a triage proposal for each file. No writes."""
    directory = inbox_dir(vault_path)
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _propose_title(text, fallback=path.stem)
        items.append(
            TriageItem(
                path=path,
                text=text,
                proposed_type="note",
                proposed_title=title,
                proposed_tags=_propose_tags(text),
                proposed_id=_slugify(title),
            )
        )
    return items


def commit_item(
    settings,
    item: TriageItem,
    *,
    doc_type: str,
    title: str,
    doc_id: str,
    tags: list[str],
) -> Path:
    """Turn a triaged inbox item into a real card, remove it from the inbox, and index it."""
    model = model_for_type(doc_type)
    today = date.today()
    fields: dict = {
        "id": doc_id,
        "title": title,
        "tags": tags,
        "provenance": Provenance(method="manual", sources=[str(item.path)], ingested_at=today),
        "created": today,
        "updated": today,
    }
    if doc_type == "project":
        fields.setdefault("subtype", "personal")
        fields.setdefault("status", "ongoing")
    elif doc_type == "person":
        fields.setdefault("relationship", "colleague")
    elif doc_type == "episode":
        fields.setdefault("significance", "medium")
    elif doc_type == "artifact":
        fields.setdefault("medium", "post")

    card = model.model_validate(fields)
    dest = card_path(settings.vault_path, card.type, card.id)
    if dest.exists():
        raise FileExistsError(f"Card already exists: {dest}")

    from recall.vault import save_card

    save_card(settings.vault_path, card, item.text.strip() + "\n")
    item.path.unlink()

    conn = db_module.connect(settings.db_path)
    try:
        db_module.record_ingest_status(conn, doc_id=card.id, source=str(item.path), status="committed")
    finally:
        conn.close()

    build_index(settings.vault_path, settings.db_path, doc_id=card.id, embedding_model=settings.embedding_model)
    return dest
