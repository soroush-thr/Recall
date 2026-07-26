"""Read, write, and validate markdown cards in the vault."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import frontmatter
import yaml

from recall.schema import BaseCard, parse_card

TYPE_DIRS = {
    "project": "projects",
    "person": "people",
    "episode": "episodes",
    "note": "notes",
    "artifact": "artifacts",
}


@dataclass
class Card:
    card: BaseCard
    body: str
    path: Path

    @property
    def frontmatter_dict(self) -> dict:
        return self.card.model_dump(mode="json")

    def content_hash(self) -> str:
        """SHA-256 of normalized frontmatter (sorted keys) + stripped body.

        Deliberately NOT a hash of raw file bytes: Obsidian's Properties UI
        rewrites YAML formatting/key order on save, and hashing raw bytes
        would trigger a spurious full re-embed on every such edit.
        """
        normalized_fm = json.dumps(self.frontmatter_dict, sort_keys=True, ensure_ascii=False)
        normalized_body = self.body.rstrip()
        return hashlib.sha256((normalized_fm + "\n" + normalized_body).encode("utf-8")).digest().hex()


def type_dir(vault_path: Path, doc_type: str) -> Path:
    return vault_path / TYPE_DIRS[doc_type]


def card_path(vault_path: Path, doc_type: str, doc_id: str) -> Path:
    return type_dir(vault_path, doc_type) / f"{doc_id}.md"


def load_card(path: Path) -> Card:
    with open(path, encoding="utf-8") as f:
        post = frontmatter.load(f)
    card = parse_card(dict(post.metadata))
    expected_stem = path.stem
    if card.id != expected_stem:
        raise ValueError(
            f"id {card.id!r} does not match filename stem {expected_stem!r} ({path})"
        )
    return Card(card=card, body=post.content, path=path)


def save_card(vault_path: Path, card: BaseCard, body: str) -> Path:
    path = card_path(vault_path, card.type, card.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = card.model_dump(mode="json", exclude_none=False)
    post = frontmatter.Post(body, **fm)
    text = frontmatter.dumps(post, Dumper=_yaml_dumper())
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _yaml_dumper():
    class _Dumper(yaml.SafeDumper):
        pass

    _Dumper.add_representer(
        type(None), lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:null", "null")
    )
    return _Dumper


def iter_card_paths(vault_path: Path):
    for doc_type, dirname in TYPE_DIRS.items():
        d = vault_path / dirname
        if not d.exists():
            continue
        yield from sorted(d.glob("*.md"))


def validate_vault(vault_path: Path) -> list[tuple[Path, str]]:
    """Return a list of (path, error_message) for every card that fails validation."""
    errors = []
    for path in iter_card_paths(vault_path):
        try:
            load_card(path)
        except Exception as e:  # noqa: BLE001 - surface every validation failure
            errors.append((path, str(e)))
    return errors
