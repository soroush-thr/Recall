"""Section-aware chunking, per spec section 6."""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
MAX_SECTION_CHARS = 1200
SPLIT_TARGET_CHARS = 800
SPLIT_OVERLAP_CHARS = 100


@dataclass
class Chunk:
    section: str | None
    ordinal: int
    text: str  # raw text, for display
    header_text: str  # context-header-prepended text, for embedding/FTS


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into (section_title, section_text) pairs on '## ' headings."""
    matches = list(SECTION_RE.finditer(body))
    if not matches:
        return [("", body.strip())] if body.strip() else []
    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        sections.append((title, text))
    return sections


def _split_paragraphs(text: str, target: int, overlap: int) -> list[str]:
    """Split long text into ~target-char pieces on paragraph boundaries, with overlap."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > target and current:
            pieces.append(current)
            tail = current[-overlap:] if overlap < len(current) else current
            current = f"{tail}\n\n{para}"
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _context_header(title: str, doc_type: str, subtype: str | None, started: str | None,
                     ended: str | None, section: str) -> str:
    type_part = f"{doc_type}/{subtype}" if subtype else doc_type
    date_part = f"{started or ''}–{ended or ''}" if (started or ended) else ""
    parts = [p for p in (title, type_part, date_part, section) if p]
    return " | ".join(parts)


def chunk_document(
    *,
    doc_id: str,
    title: str,
    doc_type: str,
    subtype: str | None,
    started: str | None,
    ended: str | None,
    tags: list[str],
    tech: list[str],
    summary: str,
    body: str,
) -> list[Chunk]:
    """Produce section-aware chunks for a document, including a synthetic chunk 0."""
    chunks: list[Chunk] = []

    chunk0_text = f"{title}\n\n{summary}\n\nTags: {', '.join(tags)}\nTech: {', '.join(tech)}".strip()
    chunks.append(
        Chunk(
            section=None,
            ordinal=0,
            text=chunk0_text,
            header_text=f"{title} | {doc_type}/{subtype or ''}\n\n{chunk0_text}",
        )
    )

    ordinal = 1
    for section_title, section_text in _split_sections(body):
        if not section_text:
            continue
        header = _context_header(title, doc_type, subtype, started, ended, section_title)
        if len(section_text) <= MAX_SECTION_CHARS:
            pieces = [section_text]
        else:
            pieces = _split_paragraphs(section_text, SPLIT_TARGET_CHARS, SPLIT_OVERLAP_CHARS)
        for piece in pieces:
            chunks.append(
                Chunk(
                    section=section_title or None,
                    ordinal=ordinal,
                    text=piece,
                    header_text=f"{header}\n\n{piece}",
                )
            )
            ordinal += 1

    return chunks
