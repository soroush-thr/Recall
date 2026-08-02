from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from recall.config import load_settings
from recall.db import connect
from recall.ingest.review import UnknownMarkersRemain, commit_draft, inspect_draft

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _write_draft(settings, slug: str, *, with_unknown: bool) -> Path:
    tmpl = (TEMPLATES_DIR / "project.md").read_text(encoding="utf-8")
    today = date.today().isoformat()
    rendered = tmpl.format(id=slug, title="Sample Draft Project", today=today)
    rendered = rendered.replace('method: manual', 'method: folder-ingest', 1)
    summary = "UNKNOWN — client identity not in evidence" if with_unknown else "A real summary."
    rendered = rendered.replace("## Summary\n", f"## Summary\n{summary}\n", 1)
    settings.drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = settings.drafts_dir / f"{slug}.md"
    draft_path.write_text(rendered, encoding="utf-8")
    return draft_path


def test_inspect_draft_reports_sections_and_unknowns(vault_path: Path):
    settings = load_settings(vault_path)
    _write_draft(settings, "prj-review-1", with_unknown=True)
    summary = inspect_draft(settings.drafts_dir / "prj-review-1.md")
    assert "Summary" in summary.sections_present
    assert len(summary.unknowns) == 1
    assert "client identity" in summary.unknowns[0]


def test_commit_draft_refuses_with_unknown_markers(vault_path: Path):
    settings = load_settings(vault_path)
    _write_draft(settings, "prj-review-2", with_unknown=True)
    with pytest.raises(UnknownMarkersRemain):
        commit_draft(settings, "prj-review-2")
    # draft must survive a refused commit
    assert (settings.drafts_dir / "prj-review-2.md").exists()


def test_commit_draft_allow_unknown_proceeds(vault_path: Path):
    settings = load_settings(vault_path)
    _write_draft(settings, "prj-review-3", with_unknown=True)
    dest = commit_draft(settings, "prj-review-3", allow_unknown=True, embed=False)
    assert dest == vault_path / "projects" / "prj-review-3.md"
    assert dest.exists()
    assert not (settings.drafts_dir / "prj-review-3.md").exists()


def test_commit_draft_clean_commits_and_indexes(vault_path: Path):
    settings = load_settings(vault_path)
    _write_draft(settings, "prj-review-4", with_unknown=False)
    dest = commit_draft(settings, "prj-review-4", embed=False)
    assert dest.exists()

    conn = connect(settings.db_path)
    try:
        doc_row = conn.execute(
            "SELECT id, title FROM documents WHERE id = ?", ("prj-review-4",)
        ).fetchone()
        log_row = conn.execute(
            "SELECT status FROM ingest_log WHERE doc_id = ?", ("prj-review-4",)
        ).fetchone()
    finally:
        conn.close()
    assert doc_row is not None
    assert doc_row["title"] == "Sample Draft Project"
    assert log_row["status"] == "committed"


def test_commit_draft_missing_draft_raises(vault_path: Path):
    settings = load_settings(vault_path)
    with pytest.raises(FileNotFoundError):
        commit_draft(settings, "prj-does-not-exist")


def test_commit_draft_rejects_id_slug_mismatch(vault_path: Path):
    settings = load_settings(vault_path)
    _write_draft(settings, "prj-review-5", with_unknown=False)
    # simulate a slug that doesn't match the drafted card's id
    (settings.drafts_dir / "prj-wrong-slug.md").write_text(
        (settings.drafts_dir / "prj-review-5.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        commit_draft(settings, "prj-wrong-slug")
