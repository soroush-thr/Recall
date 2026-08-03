from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from conftest import make_project_card

from recall.hygiene import find_stale, reconfirm_card, run_doctor, fix_doctor_report
from recall.indexer import build_index
from recall.vault import load_card, save_card


def test_find_stale_flags_old_last_verified(vault_path: Path):
    path = make_project_card(vault_path, "prj-old", "Old Project")
    card = load_card(path)
    old_date = date.today() - timedelta(days=200)
    stale_card = card.card.model_copy(update={"last_verified": old_date})
    save_card(vault_path, stale_card, card.body)

    flags = find_stale(vault_path)
    assert len(flags) == 1
    assert flags[0].doc_id == "prj-old"
    assert any("last_verified" in r for r in flags[0].reasons)


def test_find_stale_flags_missing_provenance_source(vault_path: Path):
    path = make_project_card(vault_path, "prj-missing-src", "Missing Source Project")
    card = load_card(path)
    updated = card.card.model_copy(
        update={
            "provenance": card.card.provenance.model_copy(
                update={"sources": [str(vault_path / "does-not-exist" / "folder")]}
            )
        }
    )
    save_card(vault_path, updated, card.body)

    flags = find_stale(vault_path)
    assert len(flags) == 1
    assert any("missing source" in r for r in flags[0].reasons)


def test_find_stale_fresh_card_not_flagged(vault_path: Path):
    make_project_card(vault_path, "prj-fresh", "Fresh Project")
    assert find_stale(vault_path) == []


def test_reconfirm_card_bumps_last_verified(vault_path: Path):
    path = make_project_card(vault_path, "prj-old2", "Old Project 2")
    card = load_card(path)
    old_date = date.today() - timedelta(days=200)
    stale_card = card.card.model_copy(update={"last_verified": old_date})
    save_card(vault_path, stale_card, card.body)

    reconfirm_card(vault_path, path)

    reloaded = load_card(path)
    assert reloaded.card.last_verified == date.today()
    assert find_stale(vault_path) == []


def test_doctor_clean_vault_reports_no_issues(vault_path: Path):
    make_project_card(vault_path, "prj-clean", "Clean Project")
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    # embed=False (default, no model download) means every chunk is legitimately missing an
    # embedding — that's expected here, not an integrity problem. Check everything else is clean.
    report = run_doctor(vault_path, db_path, embedding_model="BAAI/bge-m3")
    assert report.orphaned_chunk_ids == []
    assert report.drifted_doc_ids == []
    assert report.stale_model_chunk_ids == {}
    assert report.duplicate_ids == {}
    assert report.schema_failures == []


def test_doctor_detects_content_drift_and_fix_reindexes(vault_path: Path):
    path = make_project_card(vault_path, "prj-drift", "Drift Project", summary="Original summary.")
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    # Edit the vault file directly without reindexing -> db content_hash now stale.
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Original summary.", "Changed summary."), encoding="utf-8")

    report = run_doctor(vault_path, db_path, embedding_model="BAAI/bge-m3")
    assert report.drifted_doc_ids == ["prj-drift"]
    assert not report.is_clean

    fix_doctor_report(vault_path, db_path, report, embedding_model="BAAI/bge-m3")
    report2 = run_doctor(vault_path, db_path, embedding_model="BAAI/bge-m3")
    assert report2.drifted_doc_ids == []


def test_doctor_detects_orphaned_chunks_and_fix_removes_them(vault_path: Path):
    make_project_card(vault_path, "prj-orphan", "Orphan Project")
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section, ordinal, text, char_count) "
        "VALUES ('ghost#0', 'no-such-doc', NULL, 0, 'ghost text', 10)"
    )
    conn.commit()
    conn.close()

    report = run_doctor(vault_path, db_path, embedding_model="BAAI/bge-m3")
    assert "ghost#0" in report.orphaned_chunk_ids

    fix_doctor_report(vault_path, db_path, report, embedding_model="BAAI/bge-m3")
    report2 = run_doctor(vault_path, db_path, embedding_model="BAAI/bge-m3")
    assert "ghost#0" not in report2.orphaned_chunk_ids


def test_doctor_detects_schema_validation_failures(vault_path: Path):
    make_project_card(vault_path, "prj-ok", "OK Project")
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    # Add the broken card *after* indexing so build_index (which doesn't tolerate schema
    # errors) isn't itself exercised here — doctor should still catch it via validate_vault.
    bad_path = vault_path / "projects" / "prj-bad.md"
    bad_path.write_text("---\ntype: project\nid: prj-bad\n---\n\nbroken\n", encoding="utf-8")

    report = run_doctor(vault_path, db_path, embedding_model="BAAI/bge-m3")
    assert any("prj-bad" in str(p) for p, _ in report.schema_failures)
    assert not report.is_clean
