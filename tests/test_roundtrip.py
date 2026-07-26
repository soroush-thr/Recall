from __future__ import annotations

from pathlib import Path

from conftest import make_project_card

from recall.vault import load_card


def test_roundtrip_preserves_frontmatter_and_body(vault_path: Path):
    path = make_project_card(
        vault_path, "prj-roundtrip-test", "Roundtrip Test",
        summary="A summary line.", tags=["a", "b"],
    )
    card = load_card(path)
    assert card.card.id == "prj-roundtrip-test"
    assert card.card.title == "Roundtrip Test"
    assert card.card.tags == ["a", "b"]
    assert "A summary line." in card.body


def test_roundtrip_persian_body_no_mojibake(vault_path: Path):
    path = make_project_card(
        vault_path, "prj-farsi-test", "Farsi Test",
        summary="این یک پروژه آزمایشی است.",
    )
    card = load_card(path)
    assert "این یک پروژه آزمایشی است." in card.body

    # re-read raw bytes as utf-8 explicitly to guard against cp1252 mangling
    raw = path.read_text(encoding="utf-8")
    assert "این یک پروژه آزمایشی است." in raw
