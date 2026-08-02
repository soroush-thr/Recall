"""Bulk folder ingestion: `recall import --glob ... --type ...`.

Harvests every folder matching a glob, drafts each, then hands the queue of drafted slugs
back to the caller (the CLI) to review one at a time. See RECALL-BUILD-PLAN.md §9.
"""

from __future__ import annotations

import glob as glob_module
import re
from dataclasses import dataclass
from pathlib import Path

from recall.ingest.backends import HandoffRequired, SynthesisBackend
from recall.ingest.harvest import harvest_and_store
from recall.ingest.synthesize import draft as run_draft


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-+", "-", slug)


@dataclass
class BulkResult:
    folder: Path
    slug: str
    stage: str  # "harvested" | "drafted" | "handoff" | "failed"
    error: str | None = None


def expand_glob(pattern: str) -> list[Path]:
    """Expand a glob pattern to existing directories, sorted for determinism."""
    matches = [Path(p) for p in glob_module.glob(pattern)]
    return sorted(p.resolve() for p in matches if p.is_dir())


def bulk_ingest(
    settings,
    pattern: str,
    doc_type: str,
    backend: SynthesisBackend,
) -> list[BulkResult]:
    """Harvest and draft every folder matching `pattern`. Never raises: failures are
    collected into the result list so one bad folder doesn't stop the batch."""
    results: list[BulkResult] = []
    for folder in expand_glob(pattern):
        slug = _slugify(folder.name)
        try:
            harvest_and_store(settings, folder, doc_type, slug)
        except Exception as e:  # noqa: BLE001 - one folder's failure must not stop the batch
            results.append(BulkResult(folder=folder, slug=slug, stage="failed", error=str(e)))
            continue

        try:
            run_draft(settings, slug, backend)
        except HandoffRequired:
            results.append(BulkResult(folder=folder, slug=slug, stage="handoff"))
        except Exception as e:  # noqa: BLE001
            results.append(BulkResult(folder=folder, slug=slug, stage="failed", error=str(e)))
        else:
            results.append(BulkResult(folder=folder, slug=slug, stage="drafted"))
    return results
