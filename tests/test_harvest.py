from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from recall.ingest.harvest import MAX_BUNDLE_CHARS, harvest, harvest_and_store


def _git(folder: Path, *args: str, env: dict | None = None) -> None:
    subprocess.run(
        ["git", "-C", str(folder), *args],
        check=True,
        capture_output=True,
        env=env,
    )


def _init_git_repo(folder: Path) -> None:
    import os

    _git(folder, "init", "-q")
    _git(folder, "config", "user.email", "test@example.com")
    _git(folder, "config", "user.name", "Test User")

    commits = [
        ("2024-01-05T10:00:00", "initial commit", "a.py"),
        ("2024-02-10T10:00:00", "add feature", "b.py"),
        ("2024-03-15T10:00:00", "final polish", "train.py"),
    ]
    for when, message, touched in commits:
        existing = (folder / touched).read_text(encoding="utf-8") if (folder / touched).exists() else ""
        (folder / touched).write_text(existing + f"# touched at {when}\n", encoding="utf-8")
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = when
        env["GIT_COMMITTER_DATE"] = when
        _git(folder, "add", "-A", env=env)
        _git(folder, "commit", "-q", "-m", message, env=env)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "# Sample Project\n\nA small fixture project for harvest tests.\n", encoding="utf-8"
    )
    (repo / "requirements.txt").write_text("numpy==1.26.0\nrequests>=2.31\n# a comment\n", encoding="utf-8")
    (repo / "a.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (repo / "b.py").write_text("import a\n\nclass Widget:\n    pass\n", encoding="utf-8")
    (repo / "train.py").write_text("def train(model):\n    return model\n", encoding="utf-8")
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Notebook heading\n"]},
            {"cell_type": "code", "source": ["print('hi')\n"], "outputs": [{"data": "should be stripped"}]},
        ]
    }
    (repo / "notebook.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    _init_git_repo(repo)
    return repo


def test_harvest_extracts_git_dates_and_deps(sample_repo: Path) -> None:
    bundle = harvest(sample_repo, doc_type="project")
    git = bundle["git"]
    assert git is not None
    assert git["first_commit_date"] == "2024-01-05"
    assert git["last_commit_date"] == "2024-03-15"
    assert git["commit_count"] == 3
    assert len(git["contributors"]) == 1
    assert git["contributors"][0]["commits"] == 3

    deps = bundle["dependencies"]["requirements.txt"]
    assert deps["numpy"] == "1.26.0"
    assert "requests" in deps


def test_harvest_strips_notebook_outputs(sample_repo: Path) -> None:
    bundle = harvest(sample_repo, doc_type="project")
    assert len(bundle["notebooks"]) == 1
    notebook = bundle["notebooks"][0]
    assert all("outputs" not in cell for cell in notebook["cells"])
    assert any("Notebook heading" in cell["source"] for cell in notebook["cells"])


def test_harvest_documentation_and_source_files(sample_repo: Path) -> None:
    bundle = harvest(sample_repo, doc_type="project")
    assert "Sample Project" in bundle["documentation"]
    paths = {f["path"] for f in bundle["source_files"]}
    assert "train.py" in paths  # keyword-matched filename scores highest


def test_harvest_denylist_and_binary_skip(tmp_path: Path) -> None:
    repo = tmp_path / "denyrepo"
    (repo / "node_modules" / "pkg").mkdir(parents=True)
    (repo / "node_modules" / "pkg" / "index.js").write_text("noise", encoding="utf-8")
    (repo / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x01binarydata")

    bundle = harvest(repo, doc_type="project")
    assert not any("node_modules" in e for e in bundle["tree"])
    assert "keep.py" in bundle["tree"]
    assert not any("blob.bin" in e for e in bundle["tree"])


def test_harvest_no_git_no_readme_degrades_gracefully(tmp_path: Path) -> None:
    repo = tmp_path / "bare_repo"
    repo.mkdir()
    (repo / "script.py").write_text("print('hi')\n", encoding="utf-8")

    bundle = harvest(repo, doc_type="project")
    assert bundle["git"] is None
    assert bundle["documentation"] == ""
    assert bundle["char_count"] <= MAX_BUNDLE_CHARS


def test_harvest_caps_bundle_size_and_reports_drops(tmp_path: Path) -> None:
    repo = tmp_path / "big_repo"
    repo.mkdir()
    # 20 keyword-matched files (main_*), each ~30KB, force the 12-file selection well over cap.
    long_line = "x = " + "1" * 200 + "\n"
    for i in range(20):
        (repo / f"main_{i}.py").write_text(long_line * 150, encoding="utf-8")
    for i in range(500):
        (repo / f"file_{i}.py").write_text(f"# file {i}\n" + ("x = 1\n" * 5), encoding="utf-8")

    bundle = harvest(repo, doc_type="project")
    assert bundle["char_count"] <= MAX_BUNDLE_CHARS
    assert len(bundle["dropped"]) > 0


def test_harvest_and_store_writes_evidence_and_logs(vault_path: Path, sample_repo: Path) -> None:
    from recall.config import load_settings
    from recall.db import connect

    settings = load_settings(vault_path)
    evidence_path = harvest_and_store(settings, sample_repo, "project", "prj-sample")
    assert evidence_path.exists()
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert data["doc_type"] == "project"
    assert "harvested_at" in data

    conn = connect(settings.db_path)
    try:
        row = conn.execute(
            "SELECT status FROM ingest_log WHERE doc_id = ?", ("prj-sample",)
        ).fetchone()
    finally:
        conn.close()
    assert row["status"] == "harvested"
