from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def make_project_card(
    vault_path: Path,
    doc_id: str,
    title: str,
    *,
    summary: str = "",
    tags: list[str] | None = None,
    tech: list[str] | None = None,
    started: str | None = None,
    subtype: str = "personal",
) -> Path:
    tmpl = (TEMPLATES_DIR / "project.md").read_text(encoding="utf-8")
    today = date.today().isoformat()
    rendered = tmpl.format(id=doc_id, title=title, today=today)
    if tags:
        rendered = rendered.replace("tags: []", f"tags: [{', '.join(tags)}]", 1)
    if tech:
        rendered = rendered.replace("tech: []", f"tech: [{', '.join(tech)}]", 1)
    if started:
        rendered = rendered.replace("started: null", f"started: {started}", 1)
    if subtype != "personal":
        rendered = rendered.replace("subtype: personal", f"subtype: {subtype}", 1)
    if summary:
        rendered = rendered.replace("## Summary\n", f"## Summary\n{summary}\n", 1)
    dest = vault_path / "projects" / f"{doc_id}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(rendered, encoding="utf-8")
    return dest


@pytest.fixture
def vault_path(tmp_path: Path) -> Path:
    from recall.config import write_default_config
    from recall.vault import TYPE_DIRS

    v = tmp_path / "vault"
    for dirname in TYPE_DIRS.values():
        (v / dirname).mkdir(parents=True, exist_ok=True)
    write_default_config(v)
    return v
