from __future__ import annotations

import subprocess
from pathlib import Path

from recall.config import load_settings
from recall.ingest.bulk import bulk_ingest, expand_glob


class FakeBackend:
    """Returns a minimal valid project draft, no LLM/handoff involved."""

    def synthesize(self, prompt: str, *, slug: str) -> str:
        return (
            "---\n"
            f"id: {slug}\n"
            "type: project\n"
            f"title: {slug}\n"
            "subtype: personal\n"
            "status: completed\n"
            "tags: []\n"
            "visibility: private\n"
            "confidence: medium\n"
            "provenance:\n"
            "  method: folder-ingest\n"
            "  sources: []\n"
            "created: 2024-01-01\n"
            "updated: 2024-01-01\n"
            "---\n"
            "## Summary\nBody.\n"
        )


def _make_repo(root: Path, name: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=folder, check=True, capture_output=True)
    return folder


def test_expand_glob_finds_directories_only(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    found = expand_glob(str(tmp_path / "*"))
    assert [p.name for p in found] == ["a", "b"]


def test_bulk_ingest_harvests_and_drafts_each_folder(vault_path, tmp_path):
    projects_root = tmp_path / "projects"
    _make_repo(projects_root, "proj-one")
    _make_repo(projects_root, "proj-two")

    settings = load_settings(vault_path)
    results = bulk_ingest(settings, str(projects_root / "*"), "project", FakeBackend())

    assert {r.slug for r in results} == {"proj-one", "proj-two"}
    assert all(r.stage == "drafted" for r in results)
    for slug in ("proj-one", "proj-two"):
        assert (settings.evidence_dir / f"{slug}.json").exists()
        assert (settings.drafts_dir / f"{slug}.md").exists()


def test_bulk_ingest_continues_after_one_folder_fails(vault_path, tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    good = _make_repo(projects_root, "proj-good")
    _make_repo(projects_root, "proj-bad")

    settings = load_settings(vault_path)

    from recall.ingest import bulk as bulk_module

    original = bulk_module.harvest_and_store

    def flaky(settings, folder, doc_type, slug):
        if slug == "proj-bad":
            raise RuntimeError("boom")
        return original(settings, folder, doc_type, slug)

    monkeypatch.setattr(bulk_module, "harvest_and_store", flaky)

    results = bulk_ingest(settings, str(projects_root / "*"), "project", FakeBackend())
    by_slug = {r.slug: r for r in results}
    assert by_slug["proj-good"].stage == "drafted"
    assert by_slug["proj-bad"].stage == "failed"
    assert by_slug["proj-bad"].error == "boom"
