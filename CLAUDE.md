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
  `indexer.build_index(..., embed=, embedding_model=)` batches new/changed/model-mismatched chunks
  through `embedder.py` (32 at a time) after the lexical upsert.
- `src/recall/embedder.py` — Phase 3 `sentence-transformers` wrapper: `embed_passages`,
  `embed_query`, `vector_to_blob`/`blob_to_vector` (float32 blobs in `embeddings.vector`). Models
  are cached in-process by name. BGE-M3 needs no query/passage instruction prefix, unlike e5-style
  models — don't add one if the default model ever changes to something that does need it.
- `src/recall/search.py` — hybrid search: BM25 (`chunks_fts`) + cosine similarity over `embeddings`,
  fused per-chunk with Reciprocal Rank Fusion (`SearchHit`, `search()`). The dense branch is a
  no-op (no model load, no download) whenever the candidate set has zero rows for the requested
  `embedding_model` in `embeddings` — this is what keeps lexical-only tests/usage fast.
- `src/recall/cli.py` — Typer CLI (`recall init|new|validate|index|search|show`). `recall index`
  defaults to `--embed`; pass `--no-embed` for a lexical-only rebuild with no model download.
- `src/recall/mcp_server.py` — Phase 2 MCP server (`memory_search`, `memory_get`, `memory_stats`),
  entry point `recall-mcp`. Read-only; wraps `search.py` and `db.py` directly rather than
  duplicating logic. `memory_search` uses hybrid retrieval same as the CLI.
- `src/recall/ingest/` — Phase 4 folder ingestion pipeline: `harvest.py` (deterministic folder
  walk -> evidence bundle JSON, no LLM), `backends.py` (`SynthesisBackend` protocol;
  `ClaudeCodeBackend` writes a `.prompt.md` file and raises `HandoffRequired` — file-handoff by
  design, not a `claude -p` subprocess; `OllamaBackend` posts to a local Ollama server),
  `synthesize.py` (renders `prompts/{doc_type}_card.md` with the evidence bundle, calls the
  backend), `review.py` (the human gate — `commit_draft()` blocks on any `UNKNOWN —` marker
  unless `allow_unknown=True`, then moves the draft into the vault, commits in the vault's git
  repo if one exists, and reindexes just that doc). `db.record_ingest_status()` tracks one
  `ingest_log` row per doc_id across all four stages (harvested -> drafted -> committed),
  updating in place rather than inserting a new row per stage.
- CLI commands `recall ingest|draft|review|remember` (`cli.py`) wire the above; `remember` is
  harvest -> draft -> review chained with confirmations.
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
- **`embeddings.model` tracks provenance per row.** Never assume all rows share one model — a
  config change leaves old rows around until `recall index --embed` overwrites them chunk-by-chunk.
  `indexer._stale_embedding_doc_ids()` is how re-embed-on-model-change is detected; don't bypass it
  by comparing content_hash alone.

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

`tests/test_hybrid_search.py` covers Phase 3 (embedding storage, model-change re-embed, dense
retrieval on zero keyword overlap, lexical-only fallback). It deliberately overrides
`embedding_model` to `sentence-transformers/all-MiniLM-L6-v2`/`-L12-v2` (~80MB) instead of the
production default `bge-m3` (~2GB) so the suite stays fast and offline-after-first-run —
`embedder.py` is model-agnostic so this doesn't skip anything model-specific. Existing
lexical-only tests (`test_search.py`, `test_mcp_server.py`) still call `build_index(...)` with no
`embed=` argument (defaults to `False`), so they never load a model and are unaffected by Phase 3.

## Phase status (keep in sync with README.md)

Done: Phase 0 (schema/vault), Phase 1 (index/search), Phase 2 (MCP server), Phase 3
(embeddings/hybrid retrieval — `embedder.py`, `bge-m3` default, RRF fusion in `search.py`),
Phase 4 (folder ingestion — `src/recall/ingest/`, `recall ingest|draft|review|remember`;
`ClaudeCodeBackend` is file-handoff only so far, not a live `claude -p` subprocess; person/note/
artifact synthesis prompts aren't written yet, only `project_card.md`/`episode_card.md`).
Next: Phase 5 — backfill the archive (`recall import`, `recall triage`) using the Phase 4
pipeline. See build plan §12.

When starting the next phase: update README's Status section and this file's Phase status line in
the same commit as the code — they're the two places that drift if skipped.
