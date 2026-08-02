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
- **Ingestion** — point `recall ingest <folder>` at a real project directory; it harvests git
  history, README, dependencies, and source files into an evidence bundle, drafts a card from
  that evidence with an LLM, and blocks commit until every unsupported claim is either filled in
  or marked `UNKNOWN` by a human reviewer.

The vault lives in its own separate (private) repo — this repo is just the tool.

## Status

Building in phases per the build plan. Currently implemented:

- **Phase 0 — Foundations**: pydantic frontmatter schema for all card types, markdown templates,
  vault read/write/validate.
- **Phase 1 — Store + lexical search**: SQLite index with FTS5 full-text search, section-aware
  chunking, content-hash change detection (skips reindexing unchanged cards), BM25 search.
- **Phase 2 — MCP server**: `recall-mcp` exposes `memory_search`, `memory_get`, and `memory_stats`
  over stdio, so any MCP client (Claude Code, Claude Desktop) can query the vault directly from a
  chat session — no CLI switch required.
- **Phase 3 — Embeddings + hybrid retrieval**: `recall index` computes local dense embeddings
  (`BAAI/bge-m3` by default, 1024-dim, no paid API) for every chunk and stores them alongside the
  lexical index. `recall search` / `memory_search` fuse BM25 and cosine-similarity rankings with
  Reciprocal Rank Fusion, so a Farsi query can retrieve an English card (or a query with no
  keyword overlap can still find a conceptually related project) purely through the dense side.
- **Phase 4 — Folder ingestion**: `recall ingest <folder>` deterministically harvests a project
  directory (git history, README, dependencies, representative source files, notebooks) into an
  evidence bundle capped at ~50k characters; `recall draft` synthesizes a card from that evidence
  (file-handoff to Claude Code by default, or a local Ollama model); `recall review` is the
  non-negotiable human gate — it blocks committing a card into the vault while any
  `UNKNOWN — ...` marker remains, unless `--allow-unknown` is passed. `recall remember <folder>`
  chains all three stages with confirmations between them. Currently ships `project` and
  `episode` synthesis prompts; person/note/artifact prompts aren't written yet.

Not yet built: backfill across the archive (Phase 5), reranking/hygiene tooling (Phase 6).

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
recall index [--vault <path>] [--all] [--no-embed]
                                       # build/update the SQLite index (--all rebuilds from scratch)
                                       #   embeds by default (downloads the model on first run);
                                       #   --no-embed skips embedding, lexical-only, no download/model load
recall search "<query>" [--vault <path>] [--type project] [--tag x] [--from 2023] [--to 2025] [-k 10]
                                       # hybrid BM25 + dense search, fused with RRF
recall show <doc-id> [--vault <path>] # print a card

recall ingest <folder> [--type project] [--id SLUG] [--dry-run] [--vault <path>]
                                       # Stage A: harvest a folder into an evidence bundle
                                       #   --dry-run prints a summary and writes nothing
recall draft <slug> [--backend claude|ollama] [--vault <path>]
                                       # Stage B: synthesize a draft card from the evidence bundle
recall review <slug> [--allow-unknown] [--edit/--no-edit] [--vault <path>]
                                       # Stage C: the human gate — edit, validate, commit, index
recall remember <folder> [--type project] [--backend claude|ollama] [--vault <path>]
                                       # convenience wrapper: ingest -> draft -> review
```

`RECALL_VAULT` env var sets the default vault path so `--vault` can be omitted.

## MCP server (Phase 2)

`recall-mcp` runs a read-only MCP server over stdio, exposing the index to any MCP client:

- `memory_search(query, type?, tag?, from_?, to?, k?)` — hybrid BM25 + dense search (RRF-fused), same
  filters as `recall search`.
- `memory_get(doc_id)` — full markdown card (frontmatter + body) for an id.
- `memory_stats()` — card counts by type plus the earliest/latest `started` date covered.

It never writes to the vault. `confidential`-visibility cards are always excluded; set
`RECALL_MCP_PUBLIC_ONLY=1` to additionally exclude `private` cards (useful if a client is ever
shared beyond you).

[.mcp.json](.mcp.json) in this repo wires `recall-mcp` into Claude Code and is checked in as-is —
it deliberately contains **no vault path**, since that path is personal (different per machine,
and the vault itself is a private repo). Instead, each person sets `RECALL_VAULT` as their own
permanent environment variable, once, outside of git:

```powershell
setx RECALL_VAULT "C:\path\to\your\vault"    # Windows, permanent, new terminals only
```

```bash
export RECALL_VAULT=/path/to/your/vault      # macOS/Linux, add to your shell profile
```

Restart Claude Code (and your terminal) after setting it so the new environment variable is
picked up. `recall-mcp` inherits it like any other subprocess — no edits to `.mcp.json` needed,
and nothing personal ever gets committed.

## Example usage (Phases 0–2)

Everything below runs against what's built today — no ingestion or embeddings yet, just cards you
write by hand plus lexical search. Say a vault has three cards: two projects (`prj-marl-inventory-2024`,
`prj-forecasta-2022`) and one person (`per-jane-doe`, the Forecasta client), linked with wikilinks.

**1. "Have I ever built anything with reinforcement learning?"** — via CLI:

```bash
recall search "reinforcement learning inventory"
```
```
MARL Inventory Management  [project]  2024-02-01–2024-06-01  score=5.80
    MARL Inventory Management | project/research

    A multi-agent reinforcement learning system for decentralized inventory manag
    /vault/projects/prj-marl-inventory-2024.md
```

**2. Same question, from inside a Claude Code chat** — no terminal switch, Claude calls the MCP
tool directly:

```json
memory_search("inventory management reinforcement learning", k=3)
→ [{
    "doc_id": "prj-marl-inventory-2024",
    "title": "MARL Inventory Management",
    "started": "2024-02-01", "ended": "2024-06-01",
    "snippet": "...decentralized inventory management across a simulated retail supply chain...",
    "path": "/vault/projects/prj-marl-inventory-2024.md"
  }]
```
Claude can then call `memory_get("prj-marl-inventory-2024")` to pull the full card — problem,
approach, results, lessons — into its answer instead of guessing from memory of the conversation.

**3. Filtered search — "what freelance forecasting work have I done?"**

```bash
recall search "energy forecasting" --tag forecasting
```
```
Forecasta Energy Forecasting  [project]  2022-03-01–2022-09-01  score=0.59
    Time-series forecasting for energy demand for a small utility client...
    /vault/projects/prj-forecasta-2022.md
```

**4. "Who was the client on that energy job, and what came of it?"** — `recall show
prj-forecasta-2022` prints the full card, whose `client:` field wikilinks to `per-jane-doe`, whose
own card records the relationship and `Projects Together`. Two linked cards answer a question that
would otherwise depend on remembering a two-year-old email thread.

**5. "What's actually in memory right now?"**

```bash
recall search "" # or, over MCP:
memory_stats()
→ {"total": 3, "by_type": {"person": 1, "project": 2},
   "earliest": "2022-03-01", "latest": "2024-02-01"}
```

**6. Keeping Obsidian and the index in sync.** Edit a card's frontmatter in Obsidian, save, then
`recall index` (no `--all` needed) — only that card's content hash changed, so only it gets
re-chunked and re-indexed; everything else is skipped.

## Embeddings & hybrid retrieval (Phase 3)

`recall index` embeds every chunk locally with `sentence-transformers` (default model
`BAAI/bge-m3`, configurable via `embedding_model:` in `.recall/config.yaml`). Nothing is sent
anywhere — the model runs on your machine, downloaded once from Hugging Face and cached under the
usual `~/.cache/huggingface` (or `%USERPROFILE%\.cache\huggingface` on Windows).

```bash
recall index --all           # first run: downloads bge-m3 (~2GB), embeds every chunk
recall index                 # incremental: only new/changed cards get re-embedded
recall index --no-embed      # lexical-only, skips the model entirely (fast, no download)
```

`recall search` / `memory_search` always try both rankers and fuse them with Reciprocal Rank
Fusion — if a card has no embeddings yet (e.g. you ran `--no-embed`), the dense side contributes
nothing and search silently falls back to BM25-only, no error, no model load.

Changing `embedding_model` in config doesn't touch lexical data; the next `recall index` just
re-embeds every chunk under the new model name (old vectors for the previous model stay until
overwritten — `embeddings.model` tracks which model produced each row).

## Testing

```bash
.venv\Scripts\python -m pytest
```

Covers schema validation, markdown/frontmatter round-tripping (including non-ASCII text), and
search correctness/idempotency/rebuild-reproducibility. The hybrid-search tests
(`test_hybrid_search.py`) use a small model (`all-MiniLM-L6-v2`, ~80MB) instead of `bge-m3` so the
suite stays fast — `embedder.py` itself is model-agnostic, so this doesn't test anything
model-specific.

## Design principles

- Markdown is the source of truth; the index is disposable and rebuildable in one command.
- Every card records where its content came from and when it was last verified.
- An LLM may draft a card, but nothing enters memory unreviewed by a human.
- No paid APIs — local embeddings, local or Claude-Code-mediated synthesis.
- Windows-first: UTF-8 everywhere, tested against paths with spaces and non-ASCII characters.
