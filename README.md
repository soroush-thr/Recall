# Recall

A local-first, single-user memory system that stores durable knowledge about projects I've done
and things that happened to me, retrievable from a CLI and (eventually) from inside any Claude
session via MCP.

The core idea: point it at a project folder, it explores the folder, drafts a structured memory
card, and after review commits it to permanent memory — so "have I ever built anything with X?"
has a real answer instead of relying on what I remember to mention.

Full design and rationale: [RECALL-BUILD-PLAN.md](RECALL-BUILD-PLAN.md).

## How it works

- **Vault** — markdown files with YAML frontmatter, git-versioned, human-readable, the source of
  truth. Five card types: `project`, `person`, `episode`, `note`, `artifact`.
- **Index** — a derived, disposable SQLite database (FTS5 + eventually vector search) built from
  the vault. Always rebuildable with `recall index --all`.
- **Ingestion** *(planned)* — point `recall ingest <folder>` at a real project directory; it
  harvests git history, README, dependencies, and source files into an evidence bundle, drafts a
  card from that evidence with an LLM, and blocks commit until every unsupported claim is either
  filled in or marked `UNKNOWN` by a human reviewer.

The vault lives in its own separate (private) repo — this repo is just the tool.

## Status

Building in phases per the build plan. Currently implemented:

- **Phase 0 — Foundations**: pydantic frontmatter schema for all card types, markdown templates,
  vault read/write/validate.
- **Phase 1 — Store + lexical search**: SQLite index with FTS5 full-text search, section-aware
  chunking, content-hash change detection (skips reindexing unchanged cards), BM25 search.

Not yet built: MCP server (Phase 2), embeddings/hybrid retrieval (Phase 3), automated folder
ingestion (Phase 4), backfill (Phase 5), reranking/hygiene tooling (Phase 6).

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

## Commands

```bash
recall init <vault-path>              # create a new vault (type dirs + config)
recall new <type> --title "..."       # create a card from template, open in $EDITOR
                                       #   <type>: project|person|episode|note|artifact
recall validate [--vault <path>]      # validate all cards against schema
recall index [--vault <path>] [--all] # build/update the SQLite index (--all rebuilds from scratch)
recall search "<query>" [--vault <path>] [--type project] [--tag x] [--from 2023] [--to 2025] [-k 10]
recall show <doc-id> [--vault <path>] # print a card
```

`RECALL_VAULT` env var sets the default vault path so `--vault` can be omitted.

## Testing

```bash
.venv\Scripts\python -m pytest
```

Covers schema validation, markdown/frontmatter round-tripping (including non-ASCII text), and
search correctness/idempotency/rebuild-reproducibility.

## Design principles

- Markdown is the source of truth; the index is disposable and rebuildable in one command.
- Every card records where its content came from and when it was last verified.
- An LLM may draft a card, but nothing enters memory unreviewed by a human.
- No paid APIs — local embeddings, local or Claude-Code-mediated synthesis.
- Windows-first: UTF-8 everywhere, tested against paths with spaces and non-ASCII characters.
