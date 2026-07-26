"""SQLite connection, schema (DDL), and migrations for the derived index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
  id            TEXT PRIMARY KEY,
  type          TEXT NOT NULL,
  subtype       TEXT,
  title         TEXT NOT NULL,
  lang          TEXT,
  path          TEXT NOT NULL UNIQUE,
  content_hash  TEXT NOT NULL,
  started       TEXT,
  ended         TEXT,
  status        TEXT,
  visibility    TEXT NOT NULL DEFAULT 'private',
  confidence    TEXT,
  last_verified TEXT,
  frontmatter   TEXT NOT NULL,
  body          TEXT NOT NULL,
  created       TEXT,
  updated       TEXT,
  indexed_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_type    ON documents(type);
CREATE INDEX IF NOT EXISTS idx_docs_started ON documents(started);
CREATE INDEX IF NOT EXISTS idx_docs_vis     ON documents(visibility);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id    TEXT PRIMARY KEY,
  doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  section     TEXT,
  ordinal     INTEGER NOT NULL,
  text        TEXT NOT NULL,
  char_count  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  content='chunks',
  content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id  TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  vector    BLOB NOT NULL,
  model     TEXT NOT NULL,
  dim       INTEGER NOT NULL,
  created   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
  doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  tag    TEXT NOT NULL,
  PRIMARY KEY (doc_id, tag)
);

CREATE TABLE IF NOT EXISTS entities (
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL,
  canonical     TEXT NOT NULL,
  aliases       TEXT,
  doc_id        TEXT REFERENCES documents(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS mentions (
  doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  count     INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (doc_id, entity_id)
);

CREATE TABLE IF NOT EXISTS ingest_log (
  run_id      TEXT PRIMARY KEY,
  source      TEXT NOT NULL,
  doc_id      TEXT,
  status      TEXT NOT NULL,
  backend     TEXT,
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  error       TEXT
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn


def reset(db_path: Path) -> None:
    """Delete the index file entirely; caller should reconnect (and reindex) after."""
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
