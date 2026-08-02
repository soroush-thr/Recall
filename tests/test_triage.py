from __future__ import annotations

from pathlib import Path

from recall.ingest.triage import commit_item, scan_inbox
from recall.vault import load_card


def _write_inbox_note(vault_path: Path, name: str, text: str) -> Path:
    inbox = vault_path / "notes" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(text, encoding="utf-8")
    return path


def test_scan_inbox_empty(vault_path):
    assert scan_inbox(vault_path) == []


def test_scan_inbox_proposes_title_and_tags(vault_path):
    _write_inbox_note(
        vault_path,
        "capture-1.md",
        "# Moved to Waterloo\nStarted the new job today. #career #move\n",
    )
    items = scan_inbox(vault_path)
    assert len(items) == 1
    item = items[0]
    assert item.proposed_title == "Moved to Waterloo"
    assert item.proposed_id == "moved-to-waterloo"
    assert item.proposed_tags == ["career", "move"]
    assert item.proposed_type == "note"


def test_scan_inbox_ignores_non_text_files(vault_path):
    inbox = vault_path / "notes" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "image.png").write_bytes(b"\x89PNG\r\n")
    assert scan_inbox(vault_path) == []


def test_commit_item_writes_card_and_removes_from_inbox(vault_path):
    from recall.config import load_settings

    settings = load_settings(vault_path)
    _write_inbox_note(vault_path, "capture-2.md", "A quick thought.\n")
    item = scan_inbox(vault_path)[0]

    dest = commit_item(
        settings,
        item,
        doc_type="note",
        title="A quick thought",
        doc_id="a-quick-thought",
        tags=["misc"],
    )

    assert dest.exists()
    assert not item.path.exists()
    card = load_card(dest)
    assert card.card.id == "a-quick-thought"
    assert card.card.tags == ["misc"]


def test_commit_item_refuses_to_overwrite_existing_card(vault_path):
    from recall.config import load_settings

    settings = load_settings(vault_path)
    (vault_path / "notes" / "existing.md").parent.mkdir(parents=True, exist_ok=True)
    _write_inbox_note(vault_path, "capture-3.md", "Existing.\n")
    item = scan_inbox(vault_path)[0]

    commit_item(settings, item, doc_type="note", title="X", doc_id="dup-note", tags=[])
    _write_inbox_note(vault_path, "capture-4.md", "Existing again.\n")
    item2 = scan_inbox(vault_path)[0]
    import pytest

    with pytest.raises(FileExistsError):
        commit_item(settings, item2, doc_type="note", title="X", doc_id="dup-note", tags=[])
