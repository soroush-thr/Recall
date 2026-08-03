from __future__ import annotations

from pathlib import Path

import frontmatter
from conftest import make_project_card

from recall.export import export_cards
from recall.vault import load_card, save_card


def _set_visibility(vault_path: Path, path: Path, visibility: str) -> None:
    card = load_card(path)
    updated = card.card.model_copy(update={"visibility": visibility})
    save_card(vault_path, updated, card.body)


def test_export_only_includes_matching_visibility(vault_path: Path):
    p1 = make_project_card(vault_path, "prj-share", "Shareable Project")
    p2 = make_project_card(vault_path, "prj-priv", "Private Project")
    p3 = make_project_card(vault_path, "prj-conf", "Confidential Project")
    _set_visibility(vault_path, p1, "shareable")
    _set_visibility(vault_path, p2, "private")
    _set_visibility(vault_path, p3, "confidential")

    out_dir = vault_path / ".recall" / "export"
    written = export_cards(vault_path, out_dir, visibility="shareable")

    assert [p.stem for p in written] == ["prj-share"]
    assert not (out_dir / "prj-priv.md").exists()
    assert not (out_dir / "prj-conf.md").exists()


def test_export_strips_internal_fields(vault_path: Path):
    p1 = make_project_card(vault_path, "prj-share2", "Shareable Project 2")
    _set_visibility(vault_path, p1, "shareable")

    out_dir = vault_path / ".recall" / "export"
    export_cards(vault_path, out_dir, visibility="shareable")

    exported = frontmatter.loads((out_dir / "prj-share2.md").read_text(encoding="utf-8"))
    assert "provenance" not in exported.metadata
    assert "last_verified" not in exported.metadata
    assert "confidence" not in exported.metadata
    assert "visibility" not in exported.metadata
    assert exported.metadata["title"] == "Shareable Project 2"


def test_export_rejects_confidential_visibility_arg(vault_path: Path):
    import pytest

    with pytest.raises(ValueError):
        export_cards(vault_path, vault_path / ".recall" / "export", visibility="confidential")
