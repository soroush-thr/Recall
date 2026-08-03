"""`recall verify` (staleness/provenance re-confirmation) and `recall doctor` (integrity check).

See build plan §11. Both operate on top of vault.py/db.py rather than introducing new storage;
`doctor --fix` only repairs what's safely mechanical (reindexing drifted docs, deleting orphaned
chunk rows) — schema validation failures and duplicate ids are reported only, never auto-fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from recall.db import connect
from recall.indexer import build_index
from recall.vault import iter_card_paths, load_card, save_card, validate_vault

STALE_DAYS = 180


@dataclass
class VerifyFlag:
    doc_id: str
    path: Path
    reasons: list[str] = field(default_factory=list)


def find_stale(vault_path: Path, *, today: date | None = None) -> list[VerifyFlag]:
    """Flag every card whose last_verified is missing/older than STALE_DAYS, or whose
    provenance.sources reference a path that no longer exists on disk."""
    today = today or date.today()
    flags: list[VerifyFlag] = []
    for path in iter_card_paths(vault_path):
        card = load_card(path)
        reasons: list[str] = []
        lv = card.card.last_verified
        if lv is None:
            reasons.append("last_verified is missing")
        elif (today - lv).days > STALE_DAYS:
            reasons.append(f"last_verified {lv.isoformat()} is {(today - lv).days} days old")
        missing_sources = [s for s in card.card.provenance.sources if not Path(s).exists()]
        if missing_sources:
            reasons.append(f"missing source path(s): {', '.join(missing_sources)}")
        if reasons:
            flags.append(VerifyFlag(doc_id=card.card.id, path=path, reasons=reasons))
    return flags


def reconfirm_card(vault_path: Path, path: Path, *, today: date | None = None) -> Path:
    """Bump last_verified (and updated) to today and rewrite the card in place."""
    today = today or date.today()
    card = load_card(path)
    updated_card = card.card.model_copy(update={"last_verified": today, "updated": today})
    return save_card(vault_path, updated_card, card.body)


@dataclass
class DoctorReport:
    orphaned_chunk_ids: list[str] = field(default_factory=list)
    drifted_doc_ids: list[str] = field(default_factory=list)
    chunks_missing_embeddings: list[str] = field(default_factory=list)
    stale_model_chunk_ids: dict[str, list[str]] = field(default_factory=dict)
    duplicate_ids: dict[str, list[str]] = field(default_factory=dict)
    schema_failures: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.orphaned_chunk_ids
            or self.drifted_doc_ids
            or self.chunks_missing_embeddings
            or self.stale_model_chunk_ids
            or self.duplicate_ids
            or self.schema_failures
        )


def run_doctor(vault_path: Path, db_path: Path, *, embedding_model: str) -> DoctorReport:
    report = DoctorReport()

    conn = connect(db_path)
    try:
        report.orphaned_chunk_ids = [
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id NOT IN (SELECT id FROM documents)"
            ).fetchall()
        ]

        report.chunks_missing_embeddings = [
            r["chunk_id"]
            for r in conn.execute(
                "SELECT c.chunk_id AS chunk_id FROM chunks c "
                "LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id "
                "WHERE e.chunk_id IS NULL"
            ).fetchall()
        ]

        for row in conn.execute(
            "SELECT model, chunk_id FROM embeddings WHERE model != ?", (embedding_model,)
        ).fetchall():
            report.stale_model_chunk_ids.setdefault(row["model"], []).append(row["chunk_id"])

        for row in conn.execute("SELECT id, path, content_hash FROM documents").fetchall():
            path = Path(row["path"])
            if not path.exists():
                continue
            try:
                card = load_card(path)
            except Exception:  # noqa: BLE001 - schema failures are collected separately below
                continue
            if card.content_hash() != row["content_hash"]:
                report.drifted_doc_ids.append(row["id"])
    finally:
        conn.close()

    id_paths: dict[str, list[str]] = {}
    for path in iter_card_paths(vault_path):
        try:
            card = load_card(path)
        except Exception:
            continue
        id_paths.setdefault(card.card.id, []).append(str(path))
    report.duplicate_ids = {doc_id: paths for doc_id, paths in id_paths.items() if len(paths) > 1}

    report.schema_failures = [(p, msg) for p, msg in validate_vault(vault_path)]

    return report


def fix_doctor_report(vault_path: Path, db_path: Path, report: DoctorReport, *, embedding_model: str) -> None:
    """Repair what's mechanically safe: reindex drifted docs, delete orphaned chunk rows."""
    for doc_id in report.drifted_doc_ids:
        build_index(vault_path, db_path, doc_id=doc_id, embed=False, embedding_model=embedding_model)

    if report.orphaned_chunk_ids:
        conn = connect(db_path)
        try:
            placeholders = ",".join("?" for _ in report.orphaned_chunk_ids)
            conn.execute(
                f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
                report.orphaned_chunk_ids,
            )
            conn.commit()
        finally:
            conn.close()
