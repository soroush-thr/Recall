# CLAUDE.md

Notes for future Claude Code sessions working on this repo. Full spec/rationale lives in
[RECALL-BUILD-PLAN.md](RECALL-BUILD-PLAN.md); current status in [README.md](README.md). Read both
before starting a new phase.

## Where things are

- `src/recall/schema.py` — pydantic frontmatter models for the five card types.
- `src/recall/vault.py` — markdown read/write/validate, `TYPE_DIRS` mapping, content hashing.
- `src/recall/db.py` — SQLite DDL (documents, chunks, chunks_fts, embeddings, tags, entities,
  mentions, ingest_log) and `connect()`/`reset()`.
- `src/recall/chunker.py`, `src/recall/indexer.py` — section-aware chunking and index build.
- `src/recall/search.py` — BM25 lexical search (`SearchHit`, `search()`).
- `src/recall/cli.py` — Typer CLI (`recall init|new|validate|index|search|show`).
- `src/recall/mcp_server.py` — Phase 2 MCP server (`memory_search`, `memory_get`, `memory_stats`),
  entry point `recall-mcp`. Read-only; wraps `search.py` and `db.py` directly rather than
  duplicating logic.
- `tests/conftest.py` — `vault_path` fixture (empty temp vault + config) and `make_project_card`
  helper. Reuse these rather than hand-rolling vault fixtures in new test files.

## Conventions that matter

- **Windows-first, UTF-8 everywhere.** Every file open specifies `encoding="utf-8"`. Any new I/O
  must do the same — this project's whole reason for existing includes non-ASCII (Farsi) content,
  and Windows' default codepage will silently mangle it otherwise.
- **`id` must equal the filename stem.** Enforced in `vault.load_card`. Don't loosen this — it's
  what makes Obsidian wikilinks and SQLite primary keys agree (see build plan §15).
- **The index is disposable.** Never treat `.recall/index.db` as a source of truth or hand-edit it;
  it must always be reproducible via `recall index --all`. New index-derived features belong in
  `db.py`/`indexer.py`, not as new vault file formats.
- **`visibility: confidential` is excluded unconditionally**, everywhere data leaves the vault
  (search, MCP tools, future `export`). Don't add a code path that bypasses this filter.
- **No paid APIs.** Any future LLM calls (Phase 4 ingestion/drafting) go through Claude Code itself
  or a local backend — not a hosted API key.
- Config resolution order: `--vault` flag > `RECALL_VAULT` env var > `<vault>/.recall/config.yaml`
  > `Settings` defaults. `load_settings()` in `config.py` is the single place this happens.

## Testing

`.venv\Scripts\python -m pytest` runs everything. Key invariants each new feature should preserve:
idempotent indexing, byte-identical results after `--all` rebuild, and the Farsi round-trip
(`test_search_farsi_query_matches_farsi_card` / `test_roundtrip.py`) — never let non-ASCII
handling regress silently.

For MCP tool tests: call the decorated tool functions directly as plain Python callables (FastMCP's
`@mcp.tool()` doesn't wrap them opaquely), with `RECALL_VAULT` set via `monkeypatch.setenv`. See
`tests/test_mcp_server.py`. A full stdio smoke test (spin up `python -m recall.mcp_server` as a
subprocess via `mcp.client.stdio.stdio_client`) is worth rerunning by hand after touching
`mcp_server.py`, but isn't part of the automated suite.

## Phase status (keep in sync with README.md)

Done: Phase 0 (schema/vault), Phase 1 (index/search), Phase 2 (MCP server).
Next: Phase 3 — embeddings + hybrid retrieval (`embedder.py`, `bge-m3`, RRF fusion). See build plan
§12 for acceptance criteria (a Farsi query must retrieve a relevant English card).

When starting the next phase: update README's Status section and this file's Phase status line in
the same commit as the code — they're the two places that drift if skipped.
