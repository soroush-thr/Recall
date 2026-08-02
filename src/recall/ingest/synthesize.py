"""Stage B: render the evidence bundle into a synthesis prompt and draft a card."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from recall import db as db_module
from recall.ingest.backends import HandoffRequired, SynthesisBackend

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def render_prompt(evidence: dict, doc_type: str, slug: str) -> str:
    template_path = PROMPTS_DIR / f"{doc_type}_card.md"
    if not template_path.exists():
        raise ValueError(f"No synthesis prompt template for doc type {doc_type!r} ({template_path})")
    template = template_path.read_text(encoding="utf-8")
    ingested_at = evidence.get("harvested_at") or date.today().isoformat()
    return template.format(
        slug=slug,
        folder=evidence.get("folder", ""),
        evidence_ref=f".recall/evidence/{slug}.json",
        ingested_at=ingested_at,
        evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False),
    )


def draft(settings, slug: str, backend: SynthesisBackend) -> Path:
    """Load the evidence bundle for `slug`, synthesize a draft card, and write it to drafts_dir.

    Raises HandoffRequired (uncaught) if the backend needs a human to run the prompt manually
    (ClaudeCodeBackend) — the ingest_log stays at 'harvested' until a draft actually lands.
    """
    evidence_path = settings.evidence_dir / f"{slug}.json"
    if not evidence_path.exists():
        raise FileNotFoundError(f"No evidence bundle for {slug!r}; run 'recall ingest' first.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    doc_type = evidence["doc_type"]

    prompt = render_prompt(evidence, doc_type, slug)

    conn = db_module.connect(settings.db_path)
    try:
        draft_text = backend.synthesize(prompt, slug=slug)
    except HandoffRequired:
        raise
    else:
        settings.drafts_dir.mkdir(parents=True, exist_ok=True)
        draft_path = settings.drafts_dir / f"{slug}.md"
        draft_path.write_text(draft_text, encoding="utf-8")
        db_module.record_ingest_status(
            conn, doc_id=slug, source=evidence.get("folder", ""), status="drafted"
        )
        return draft_path
    finally:
        conn.close()
