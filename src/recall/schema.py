"""Pydantic models for vault card frontmatter, all document types."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PARTIAL_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")

DocType = Literal["project", "person", "episode", "note", "artifact"]


def _validate_partial_date(v: str | date | None) -> str | None:
    if v is None:
        return v
    if isinstance(v, date):
        # YAML parses an unquoted YYYY-MM-DD as a date object; normalize back to str.
        return v.isoformat()
    if not PARTIAL_DATE_RE.match(v):
        raise ValueError(f"expected YYYY, YYYY-MM, or YYYY-MM-DD, got {v!r}")
    return v


class Entities(BaseModel):
    people: list[str] = Field(default_factory=list)
    orgs: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    method: Literal["folder-ingest", "manual", "conversation", "import"]
    sources: list[str] = Field(default_factory=list)
    evidence_ref: str | None = None
    ingested_at: date | None = None


class BaseCard(BaseModel):
    id: str
    type: DocType
    title: str
    aliases: list[str] = Field(default_factory=list)
    lang: Literal["en", "fa", "mixed"] = "en"
    started: str | date | None = None
    ended: str | date | None = None
    tags: list[str] = Field(default_factory=list)
    entities: Entities = Field(default_factory=Entities)
    visibility: Literal["private", "shareable", "public", "confidential"] = "private"
    confidence: Literal["high", "medium", "low"] = "medium"
    provenance: Provenance
    last_verified: date | None = None
    created: date
    updated: date

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError(f"id must be a kebab-case slug, got {v!r}")
        return v

    @field_validator("started", "ended")
    @classmethod
    def _partial_date(cls, v: str | None) -> str | None:
        return _validate_partial_date(v)


class ProjectCard(BaseCard):
    type: Literal["project"] = "project"
    subtype: Literal["freelance", "contest", "research", "employment", "personal", "academic"]
    status: Literal["completed", "ongoing", "abandoned", "paused"]
    role: str | None = None
    tech: list[str] = Field(default_factory=list)
    outcome: str | None = None
    client: str | None = None
    repo: str | None = None


class PersonCard(BaseCard):
    type: Literal["person"] = "person"
    relationship: Literal["client", "colleague", "supervisor", "collaborator", "friend", "mentor"]
    org: str | None = None
    first_met: str | date | None = None

    @field_validator("first_met")
    @classmethod
    def _partial_date_first_met(cls, v: str | None) -> str | None:
        return _validate_partial_date(v)


class EpisodeCard(BaseCard):
    type: Literal["episode"] = "episode"
    significance: Literal["high", "medium", "low"]
    location: str | None = None


class NoteCard(BaseCard):
    type: Literal["note"] = "note"


class ArtifactCard(BaseCard):
    type: Literal["artifact"] = "artifact"
    medium: Literal["paper", "podcast", "talk", "post", "video"]
    url: str | None = None
    venue: str | None = None


CARD_TYPES: dict[DocType, type[BaseCard]] = {
    "project": ProjectCard,
    "person": PersonCard,
    "episode": EpisodeCard,
    "note": NoteCard,
    "artifact": ArtifactCard,
}

FIXED_SECTIONS: dict[DocType, list[str]] = {
    "project": [
        "Summary",
        "Problem & Context",
        "What I Built",
        "Technical Approach",
        "Results & Impact",
        "My Role",
        "Challenges & Lessons",
        "Tech Stack",
        "Artifacts & Links",
        "Timeline",
    ],
    "episode": [
        "What Happened",
        "Context",
        "Why It Mattered",
        "People Involved",
        "What I Took From It",
    ],
    "person": [
        "Snapshot",
        "How We Worked Together",
        "Projects Together",
        "Working Style",
        "Notes",
    ],
    "note": [],
    "artifact": [],
}


def model_for_type(doc_type: str) -> type[BaseCard]:
    try:
        return CARD_TYPES[doc_type]  # type: ignore[index]
    except KeyError as e:
        raise ValueError(f"unknown document type: {doc_type!r}") from e


def parse_card(frontmatter: dict) -> BaseCard:
    """Validate a raw frontmatter dict against the schema for its declared type."""
    doc_type = frontmatter.get("type")
    if doc_type is None:
        raise ValueError("frontmatter missing required 'type' field")
    model = model_for_type(doc_type)
    return model.model_validate(frontmatter)
