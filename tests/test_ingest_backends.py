from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from recall.ingest.backends import (
    ClaudeCodeBackend,
    HandoffRequired,
    OllamaBackend,
    get_backend,
)
from recall.ingest.synthesize import draft, render_prompt


def _evidence(doc_type: str = "project") -> dict:
    return {
        "folder": "D:/projects/sample",
        "doc_type": doc_type,
        "harvested_at": "2026-08-02",
        "tree": ["README.md", "a.py"],
        "languages": {},
        "dependencies": {},
        "documentation": "# Sample\n",
        "git": None,
        "notebooks": [],
        "source_files": [],
        "config_files": [],
        "license": None,
        "char_count": 100,
        "dropped": [],
    }


def test_render_prompt_includes_evidence_and_headings():
    prompt = render_prompt(_evidence(), "project", "prj-sample")
    assert "prj-sample" in prompt
    assert "## Summary" in prompt
    assert "\"folder\": \"D:/projects/sample\"" in prompt


def test_claude_code_backend_writes_prompt_and_raises_handoff(tmp_path: Path):
    backend = ClaudeCodeBackend(tmp_path)
    with pytest.raises(HandoffRequired) as exc_info:
        backend.synthesize("hello prompt", slug="prj-sample")
    prompt_path = tmp_path / "prj-sample.prompt.md"
    assert prompt_path.exists()
    assert prompt_path.read_text(encoding="utf-8") == "hello prompt"
    assert exc_info.value.prompt_path == prompt_path


def test_ollama_backend_posts_and_parses_response():
    backend = OllamaBackend(host="http://localhost:11434", model="qwen2.5:14b-instruct")
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps({"response": "drafted text"}).encode("utf-8")
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
        result = backend.synthesize("a prompt", slug="prj-sample")

    assert result == "drafted text"
    assert mock_urlopen.called


def test_get_backend_unknown_name_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        get_backend("nope", drafts_dir=tmp_path)


def test_draft_with_claude_backend_raises_handoff_and_leaves_status_harvested(vault_path: Path):
    from recall.config import load_settings
    from recall.db import connect

    settings = load_settings(vault_path)
    settings.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = settings.evidence_dir / "prj-sample.json"
    evidence_path.write_text(json.dumps(_evidence()), encoding="utf-8")

    conn = connect(settings.db_path)
    from recall.db import record_ingest_status

    record_ingest_status(conn, doc_id="prj-sample", source="D:/projects/sample", status="harvested")
    conn.close()

    backend = ClaudeCodeBackend(settings.drafts_dir)
    with pytest.raises(HandoffRequired):
        draft(settings, "prj-sample", backend)

    assert (settings.drafts_dir / "prj-sample.prompt.md").exists()
    assert not (settings.drafts_dir / "prj-sample.md").exists()

    conn = connect(settings.db_path)
    row = conn.execute(
        "SELECT status FROM ingest_log WHERE doc_id = ?", ("prj-sample",)
    ).fetchone()
    conn.close()
    assert row["status"] == "harvested"
