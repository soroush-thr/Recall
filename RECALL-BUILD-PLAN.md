# Recall — Personal Memory System

**Build specification for Claude Code**
Author: Soroush Taheri
Version: 1.0
Date: 2026-07-26

---

## 0. Purpose and scope

### What this is

A local-first, single-user memory system that stores durable knowledge about **projects I have done** and **things that happened to me**, and makes that knowledge retrievable — both from a CLI and from inside any Claude session via MCP.

The defining capability: **point it at a project folder and it explores the folder, drafts a structured memory card, and after my review commits it to permanent memory.**

### Primary use cases

1. *"Have I ever built anything with multi-agent RL?"* → returns the MARL inventory project with dates, stack, my role, outcomes.
2. *"What did I do in 2024?"* → chronological list across freelance, contests, research.
3. *"Draft a case study for the Forecasta site about a time-series project."* → Claude pulls the real project card instead of asking me to re-explain it.
4. *"Who was the client on that energy forecasting job and what was the outcome?"* → person + project cards with provenance.
5. *"What was I thinking about in the months before I applied to Waterloo?"* → episodes and notes on a timeline.

### Anti-goals (explicitly out of scope)

- Not a note-taking app. No daily notes, no journaling UI, no task management.
- Not a second brain workflow framework. No "weekly review" ceremonies.
- No web UI in v1. CLI + MCP only.
- No cloud service, no hosted vector DB, no paid API dependency.
- Not a general document store. Curated memory only — quality over volume.

### Design principles

| Principle | Consequence |
|---|---|
| **Markdown is the source of truth** | The index is disposable and rebuildable from files. No lock-in. |
| **Provenance on every claim** | Every card records where its content came from and when it was last verified. |
| **Human review gate before commit** | An LLM drafts; I approve. Nothing enters memory unreviewed. |
| **Zero paid APIs** | Local embeddings, local or Claude-Code-mediated synthesis. Consistent with TrendScribe. |
| **Bilingual by default** | Persian and English content must be searchable, including cross-lingually. |
| **Rebuildable in one command** | `recall reindex --all` reconstructs everything from the vault. |

---

## 1. Architecture

```
                    ┌─────────────────────────────┐
                    │  Interfaces                  │
                    │  • CLI (recall ...)          │
                    │  • MCP server (Claude Code)  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  Retrieval                    │
                    │  BM25 (FTS5) + dense vectors  │
                    │  → Reciprocal Rank Fusion     │
                    │  → metadata filters           │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  Index (SQLite, derived, gitignored)                 │
        │  documents · chunks · chunks_fts · embeddings ·       │
        │  entities · mentions · ingest_log                     │
        └──────────────────────────▲──────────────────────────┘
                                   │ built from
        ┌──────────────────────────┴──────────────────────────┐
        │  Vault (markdown + YAML frontmatter, git-versioned)  │
        │  projects/ people/ episodes/ notes/ artifacts/       │
        └──────────────────────────▲──────────────────────────┘
                                   │ written by
        ┌──────────────────────────┴──────────────────────────┐
        │  Ingestion                                            │
        │  Harvest (deterministic) → Synthesize (LLM)           │
        │  → Review gate (human) → Commit → Index               │
        └───────────────────────────────────────────────────────┘
```

The layering matters. The vault is durable and human-readable; the index is a cache. If the embedding model changes, or SQLite corrupts, or I abandon the tool in three years, the markdown still stands on its own.

---

## 2. Technology choices

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Matches my stack; best local-embedding ecosystem. |
| Env / packaging | `uv` | Fast, reproducible, works well on Windows. |
| CLI | `typer` | Type-hint driven, minimal boilerplate. |
| Config / validation | `pydantic` v2 | Frontmatter schema enforcement with clear errors. |
| Frontmatter parsing | `python-frontmatter` | Standard, round-trips cleanly. |
| Store | Markdown + YAML | Source of truth. Git-versioned. |
| Index | SQLite (stdlib) + FTS5 | Zero-install, single file, FTS5 ships with Python's sqlite3. |
| Vector storage | `sqlite-vec` extension | Keeps everything in one file. Fallback: numpy `.npy` + brute-force cosine (fine to ~100k chunks). |
| Embeddings | `sentence-transformers` | Local, no API. |
| Embedding model | `BAAI/bge-m3` | **Multilingual, handles Persian well, strong cross-lingual retrieval.** 1024-dim. Fallback for speed: `intfloat/multilingual-e5-base` (768-dim). Do NOT default to `all-MiniLM` or `bge-small-en` — English-only, will fail on Farsi content. |
| Reranker (optional, Phase 6) | `BAAI/bge-reranker-v2-m3` | Multilingual cross-encoder. |
| LLM for card synthesis | Claude Code (primary) or Ollama `qwen2.5:14b-instruct` (offline) | Two backends behind one interface. No paid API. |
| MCP | `mcp` Python SDK, stdio transport | Native Claude Code integration. |
| Git ignore parsing | `pathspec` | Correctly honours `.gitignore` during folder walks. |
| Testing | `pytest` | |

### Windows notes

- Use `pathlib.Path` throughout; never string-concatenate paths.
- Test against paths with spaces and drive letters (`D:\My Projects\...`).
- `sqlite-vec` ships Windows wheels; if loading the extension fails, fall back to the numpy path automatically and log a warning.
- Set `PYTHONUTF8=1` or pass `encoding="utf-8"` on every file open — Windows defaults to cp1252 and **will** mangle Persian text otherwise. This is a real failure mode, not a hypothetical.

---

## 3. Repository layout

```
recall/
├── pyproject.toml
├── README.md
├── .gitignore                  # must ignore .recall/ and vault/ if vault is separate
├── src/recall/
│   ├── __init__.py
│   ├── cli.py                  # typer app, all commands
│   ├── config.py               # pydantic Settings, loads .recall/config.yaml
│   ├── schema.py               # pydantic models for frontmatter, all doc types
│   ├── vault.py                # read/write/validate markdown cards
│   ├── db.py                   # SQLite connection, migrations, DDL
│   ├── chunker.py              # section-aware chunking
│   ├── embedder.py             # sentence-transformers wrapper, batching, cache
│   ├── indexer.py              # build/update index from vault
│   ├── search.py               # BM25 + vector + RRF + filters
│   ├── ingest/
│   │   ├── harvest.py          # deterministic folder exploration
│   │   ├── synthesize.py       # LLM card drafting
│   │   ├── backends.py         # ClaudeCodeBackend | OllamaBackend
│   │   └── review.py           # review gate
│   ├── entities.py             # entity extraction + resolution
│   ├── hygiene.py              # verify, doctor, decay
│   └── mcp_server.py           # MCP stdio server
├── prompts/
│   ├── project_card.md         # synthesis prompt + card template
│   ├── episode_card.md
│   └── triage.md
├── templates/
│   ├── project.md
│   ├── person.md
│   ├── episode.md
│   ├── note.md
│   └── artifact.md
└── tests/
    ├── fixtures/sample_repo/   # a tiny fake project for ingestion tests
    ├── test_schema.py
    ├── test_harvest.py
    ├── test_search.py
    └── test_roundtrip.py
```

The vault lives **outside** this repo by default (configurable), so the tool and the data version independently:

```
~/Documents/memory-vault/          # separate PRIVATE git repo
├── projects/
├── people/
├── episodes/
├── notes/
├── artifacts/
└── .recall/                       # gitignored
    ├── config.yaml
    ├── index.db
    ├── evidence/                  # raw harvest bundles, JSON
    ├── drafts/                    # pending review
    └── models/                    # cached embedding weights
```

---

## 4. Data model

### 4.1 Document types

| Type | Holds | Example |
|---|---|---|
| `project` | Anything I built or delivered | MARL inventory system, TrendScribe, a hackathon entry |
| `person` | People I've worked with | A client, co-founder, supervisor |
| `episode` | A thing that happened, bounded in time | Moving to Canada, a conference talk, an interview |
| `note` | An atomic idea or lesson, timeless | "Why GPU overhead can negate benefit for small networks" |
| `artifact` | A published output | A paper, podcast episode, blog post, talk |

### 4.2 Frontmatter schema

Enforced by pydantic. Common to all types:

```yaml
id: prj-marl-inventory-2024        # stable slug, immutable once assigned
type: project                      # project|person|episode|note|artifact
title: "Decentralized MARL Inventory Management"
aliases: ["MARL inventory", "the DQN inventory job"]
lang: en                           # en|fa|mixed
started: 2024-11                   # ISO date, allows YYYY / YYYY-MM / YYYY-MM-DD
ended: 2025-02                     # null if ongoing
tags: [reinforcement-learning, dqn, marl, supply-chain]
entities:                          # wikilink syntax — Obsidian resolves these as real links
  people: ["[[per-client-acme]]"]
  orgs: ["[[org-acme]]"]
  projects: []
visibility: private                # private|shareable|public
confidence: high                   # high|medium|low
provenance:
  method: folder-ingest            # folder-ingest|manual|conversation|import
  sources:
    - "D:/projects/marl-inventory"
  evidence_ref: ".recall/evidence/prj-marl-inventory-2024.json"
  ingested_at: 2026-07-26
last_verified: 2026-07-26
created: 2026-07-26
updated: 2026-07-26
```

Type-specific additions:

```yaml
# project
subtype: freelance                 # freelance|contest|research|employment|personal|academic
status: completed                  # completed|ongoing|abandoned|paused
role: "Sole ML engineer"
tech: [python, pytorch, numpy, sqlite]
outcome: "Deployed; reduced stockouts 18% in simulation"
client: per-client-acme            # optional
repo: "git@github.com:me/marl-inventory.git"

# person
relationship: client               # client|colleague|supervisor|collaborator|friend|mentor
org: org-acme
first_met: 2024-10

# episode
significance: high                 # high|medium|low
location: "Waterloo, ON"

# artifact
medium: podcast                    # paper|podcast|talk|post|video
url: "https://..."
venue: "Scientometrics"
```

### 4.3 Card body structure

**Section headings are fixed per type.** This is deliberate: predictable sections make chunking predictable, which makes retrieval predictable. The synthesis prompt must emit exactly these headings.

`project`:
```markdown
## Summary
One paragraph. What it was, for whom, what came of it.

## Problem & Context
## What I Built
## Technical Approach
## Results & Impact
## My Role
## Challenges & Lessons
## Tech Stack
## Artifacts & Links
## Timeline
```

`episode`:
```markdown
## What Happened
## Context
## Why It Mattered
## People Involved
## What I Took From It
```

`person`:
```markdown
## Snapshot
## How We Worked Together
## Projects Together
## Working Style
## Notes
```

Any fact the synthesizer cannot support from evidence must be written literally as `UNKNOWN — <what's missing>`. The review gate surfaces every `UNKNOWN` for me to fill or delete. **Never let the model guess.**

### 4.4 SQLite schema

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE documents (
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
  frontmatter   TEXT NOT NULL,        -- full YAML as JSON
  body          TEXT NOT NULL,
  created       TEXT,
  updated       TEXT,
  indexed_at    TEXT NOT NULL
);
CREATE INDEX idx_docs_type    ON documents(type);
CREATE INDEX idx_docs_started ON documents(started);
CREATE INDEX idx_docs_vis     ON documents(visibility);

CREATE TABLE chunks (
  chunk_id    TEXT PRIMARY KEY,        -- "{doc_id}#{ordinal}"
  doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  section     TEXT,
  ordinal     INTEGER NOT NULL,
  text        TEXT NOT NULL,
  char_count  INTEGER NOT NULL
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text,
  content='chunks',
  content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);
-- Triggers to keep FTS in sync on INSERT/UPDATE/DELETE of chunks.

CREATE TABLE embeddings (
  chunk_id  TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  vector    BLOB NOT NULL,             -- float32 little-endian
  model     TEXT NOT NULL,
  dim       INTEGER NOT NULL,
  created   TEXT NOT NULL
);

CREATE TABLE tags (
  doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  tag    TEXT NOT NULL,
  PRIMARY KEY (doc_id, tag)
);

CREATE TABLE entities (
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL,         -- person|org|project|tech|place
  canonical     TEXT NOT NULL,
  aliases       TEXT,                  -- JSON array
  doc_id        TEXT REFERENCES documents(id) ON DELETE SET NULL
);

CREATE TABLE mentions (
  doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  count     INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (doc_id, entity_id)
);

CREATE TABLE ingest_log (
  run_id      TEXT PRIMARY KEY,
  source      TEXT NOT NULL,
  doc_id      TEXT,
  status      TEXT NOT NULL,           -- harvested|drafted|reviewed|committed|failed
  backend     TEXT,
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  error       TEXT
);
```

**Change detection**: `content_hash` = SHA-256 of the **normalized** content — the parsed frontmatter dict serialized with sorted keys, plus the body with trailing whitespace stripped. **Do not hash raw file bytes.** Obsidian's Properties UI rewrites YAML formatting and key order on any edit; raw-byte hashing would trigger a full re-embed of a document every time I glance at it in Obsidian. On `recall index`, skip any document whose normalized hash is unchanged AND whose embeddings match the current model name.

---

## 5. Ingestion pipeline

This is the core feature. Four stages, each independently runnable and inspectable.

### Stage A — Harvest (deterministic, no LLM)

`recall ingest <folder> --type project [--dry-run]`

Walk the folder and produce a structured **evidence bundle** as JSON. No interpretation, only extraction.

**Exclusions** — honour `.gitignore` via `pathspec`, plus a built-in denylist:
```
node_modules/  .venv/  venv/  __pycache__/  .git/  dist/  build/
*.pt  *.pth  *.ckpt  *.safetensors  *.bin  *.h5  *.pkl  *.parquet
*.zip  *.tar.gz  *.mp4  *.wav  *.png  *.jpg  *.pdf
data/  datasets/  outputs/  wandb/  mlruns/  .ipynb_checkpoints/
```
Skip any file > 512 KB. Skip binary files (null-byte sniff on first 8 KB).

**Collect:**

1. **Tree** — directory structure, depth-capped at 4, max 300 entries.
2. **Language histogram** — file counts and total lines by extension.
3. **Dependencies** — parse `requirements.txt`, `pyproject.toml`, `package.json`, `environment.yml`, `Cargo.toml`, `go.mod`. Record names and pinned versions.
4. **Documentation** — full text of `README*`, `CONTRIBUTING*`, `docs/**/*.md` (up to 20 KB total).
5. **Git history** (if `.git` present):
   - first commit date, last commit date → **this is the project's real timeline**
   - total commit count, contributor names + counts
   - branch names, tag names
   - the 30 most recent commit subjects
   - `git log --stat` aggregate: most-churned files (a good proxy for "what mattered")
6. **Notebooks** — for each `.ipynb`, extract markdown cells and code cells, **strip all outputs**. Cap at 5 notebooks, 8 KB each.
7. **Representative source files** — select up to 12 by heuristic score:
   - +10 if filename matches `main|train|app|run|pipeline|model|server|index`
   - +5 if at repo root or in `src/`
   - + (import-count from other files in the repo) × 2
   - − penalty for `test_*`, `conftest`, `setup.py`
   For each selected file include: path, first 150 lines, and all top-level function/class signatures (via `ast` for Python, regex for others).
8. **Config artefacts** — `Dockerfile`, `docker-compose.yml`, `Makefile`, `.github/workflows/*`, `*.slurm`, `*.sl`.
9. **License** and any `CITATION.cff`.

**Cap the total bundle at ~50,000 characters.** If over, drop in this order: source file bodies → notebook code → commit list → tree depth. Log what was dropped.

Write to `.recall/evidence/{slug}.json`. Record a run in `ingest_log`.

`--dry-run` prints the bundle summary (what was found, what was dropped, estimated token count) and exits without calling an LLM. **Always run this first on a new repo.**

### Stage B — Synthesize (LLM)

`recall draft <slug> [--backend claude|ollama]`

Feed the evidence bundle plus `prompts/project_card.md` to the chosen backend.

**Backend interface:**
```python
class SynthesisBackend(Protocol):
    def synthesize(self, evidence: dict, template: str, doc_type: str) -> str: ...
```

- `ClaudeCodeBackend` — writes the prompt to a temp file and shells out to `claude -p @promptfile --output-format text`, or simply writes the prompt to `.recall/drafts/{slug}.prompt.md` and instructs me to run it in an open Claude Code session. **Start with the file-handoff approach; it is simpler and avoids CLI-flag churn.**
- `OllamaBackend` — POSTs to `http://localhost:11434/api/generate` with `qwen2.5:14b-instruct`, temperature 0.2, and a large `num_ctx`.

**Prompt requirements** (`prompts/project_card.md` must contain all of these):
- Emit exactly the fixed section headings for the type. No extra sections, no preamble, no markdown fences around the whole output.
- Emit valid YAML frontmatter matching the schema.
- **Every factual claim must be traceable to the evidence bundle.** Where a claim comes from a specific file, append `` `<path>` `` inline.
- Anything not derivable from evidence → write `UNKNOWN — <what is missing>`. Explicitly list: client identity, commercial outcome, contest placement, my subjective role, why the project ended. These are almost never in a repo.
- Infer `started`/`ended` from git history only. If no git history, `UNKNOWN`.
- Set `confidence: medium` by default; `low` if there is no README and no git history.
- Do not invent metrics. If the README claims a number, quote it and attribute it to the README rather than asserting it.

Write output to `.recall/drafts/{slug}.md`.

### Stage C — Review gate (human) — NON-NEGOTIABLE

`recall review <slug>`

1. Print a diff-style summary: counts of sections filled, list of every `UNKNOWN`, inferred dates, detected tech.
2. Open `.recall/drafts/{slug}.md` in `$EDITOR` (default: `notepad` on Windows, `vim` elsewhere; configurable).
3. On save, validate frontmatter against the pydantic schema. Reject and reopen on validation error with a clear message.
4. Block commit while any `UNKNOWN —` marker remains, unless `--allow-unknown` is passed.
5. On approval: move to `vault/{type}s/{slug}.md`, `git add` + `git commit -m "memory: add {title}"` in the vault repo, then index.

Rationale: a repo tells you what code exists. It does not know that this was a contest entry that placed third, that the client was difficult, or that the interesting part was a constraint you worked around. **That knowledge only exists in my head and this is the one moment to capture it.**

### Stage D — Index

`recall index [--all|--doc <id>]`

1. Parse frontmatter + body.
2. Chunk (see §6).
3. Embed new/changed chunks in batches of 32.
4. Upsert `documents`, `chunks`, `chunks_fts`, `embeddings`, `tags`.
5. Extract and resolve entities (§8).

### Convenience wrapper

`recall remember <folder>` = harvest → draft → review → commit → index, with prompts between stages.

---

## 6. Chunking

Section-aware, not fixed-window.

- One chunk per `##` section.
- If a section exceeds 1,200 characters, split on paragraph boundaries into ~800-character pieces with 100-character overlap.
- **Prepend a context header to every chunk before embedding** (but store the raw text separately for display):
  ```
  {title} | {type}/{subtype} | {started}–{ended} | {section}

  {text}
  ```
  This is a cheap, large win: it means a chunk from "Tech Stack" still retrieves on a query mentioning the project name or year.
- Always emit a synthetic chunk 0 containing title + summary + tags + tech, so every document is findable even if body sections are thin.

---

## 7. Retrieval

`recall search "<query>" [--type project] [--from 2023] [--to 2025] [--tag rl] [-k 10]`

**Algorithm:**

1. Apply metadata filters first (type, date range, tags, visibility) → candidate doc set.
2. **Lexical**: FTS5 `bm25()` over `chunks_fts`, restricted to candidates, top 50.
3. **Dense**: embed the query (with the `bge-m3` query prefix if the model requires one), cosine similarity against `embeddings` restricted to candidates, top 50.
4. **Fuse** with Reciprocal Rank Fusion:
   ```
   score(c) = Σ_over_rankers  1 / (60 + rank_in_that_ranker(c))
   ```
   RRF needs no score normalisation and is robust when one ranker fails — which matters here, because BM25 does nothing useful for a Farsi query against English cards, and the dense ranker carries it.
5. **Collapse to documents**: a document's score is the max of its chunk scores, plus 0.1 × (number of distinct matching chunks), capped. Prevents one long document from flooding results.
6. Return top *k* documents, each with its best-matching chunks and full frontmatter.
7. *(Phase 6)* Rerank top 20 chunks with `bge-reranker-v2-m3`.

**Output for CLI**: a compact table — title, type, dates, score — then the matched snippet for each hit.

**Output for MCP**: structured JSON with `id`, `title`, `type`, `dates`, `snippet`, `path`, `score`, so Claude can decide whether to fetch the full card.

---

## 8. Entities and linking

Light-touch, deterministic-first. **Do not build a knowledge graph.** Resolve identity, nothing more.

1. Entities come primarily from explicit frontmatter (`entities.people`, `client`, `org`, `tech`).
2. On index, also scan body text for known entity `canonical` names and `aliases` → record in `mentions`.
3. `recall entity list --kind person` shows all people with mention counts.
4. `recall entity merge <id-a> <id-b>` folds one into the other, rewriting references. Manual, because automated entity resolution on a personal corpus produces more errors than it saves keystrokes.
5. `recall related <doc-id>` returns documents sharing entities or tags, ranked by overlap count.

*(This is the component most directly connected to my PhD area — if I want to experiment with learned entity resolution later, this is the hook.)*

---

## 9. Capture paths

Four ways in, deliberately low-friction:

| Path | Command | Use |
|---|---|---|
| Folder ingest | `recall remember D:\projects\foo` | The main event. |
| Quick capture | `recall add "text..."` | Drops a timestamped file in `notes/inbox/`. Two seconds. No schema required. |
| Templated new | `recall new episode --title "Moved to Waterloo"` | Opens a pre-filled template in `$EDITOR`. |
| From Claude | `memory_add` MCP tool | Mid-conversation: "remember that I decided X because Y." |
| Bulk import | `recall import --glob "D:/projects/*" --type project` | Queue many folders; harvests all, drafts all, then reviews one at a time. |

**Inbox triage**: `recall triage` walks `notes/inbox/`, and for each item proposes a type, title, tags, and target file — I accept, edit, or defer. This is how loose captures become real cards without requiring discipline at capture time. Run it monthly.

---

## 10. MCP server

`src/recall/mcp_server.py`, stdio transport.

Registered in `.mcp.json`:
```json
{
  "mcpServers": {
    "recall": {
      "command": "uv",
      "args": ["run", "--directory", "C:/path/to/recall", "recall-mcp"],
      "env": { "RECALL_VAULT": "C:/Users/me/Documents/memory-vault" }
    }
  }
}
```

**Tools:**

| Tool | Signature | Notes |
|---|---|---|
| `memory_search` | `(query, type?, tag?, date_from?, date_to?, k=8)` | Returns ranked docs with snippets. |
| `memory_get` | `(id, section?)` | Full card or one section. |
| `memory_timeline` | `(date_from, date_to, type?)` | Chronological listing. |
| `memory_related` | `(id, k=5)` | Entity/tag overlap. |
| `memory_add` | `(type, title, content, tags?)` | Writes to inbox, returns new id. The only mutating tool. |
| `memory_stats` | `()` | Counts by type, date coverage, staleness summary. |

**Guardrails:**
- Read tools return only `visibility != private` when `RECALL_MCP_PUBLIC_ONLY=1`. Off by default, but available for when I'm screen-sharing or using an untrusted client.
- `memory_add` writes to inbox only — it can never overwrite an existing card. Removes the whole class of "the agent silently rewrote my memory" failure.
- Every response includes `path` so I can verify against the file.

This is where the system earns its keep. Once wired in, any Claude session can answer "what have I done with X" from ground truth rather than from what I remember to mention.

---

## 11. Hygiene

| Command | Does |
|---|---|
| `recall verify` | Flags documents where `last_verified` > 180 days, or where a path in `provenance.sources` no longer exists on disk. Interactive: re-confirm, update, or mark stale. |
| `recall doctor` | Integrity check — orphaned chunks, documents whose `content_hash` drifted, chunks missing embeddings, embeddings from a stale model, duplicate slugs, schema violations. Reports and offers `--fix`. |
| `recall reindex --all` | Nuke and rebuild the index from the vault. Must always work. This is the escape hatch. |
| `recall stats` | Coverage by year and type. Surfaces gaps — "you have nothing from 2021". |

Run `verify` quarterly. The failure mode this prevents is memory rot: cards that were true when written and quietly became wrong.

---

## 12. Build phases

Sized for ~10 hrs/week. Each phase ends in something usable — no phase is only scaffolding.

### Phase 0 — Foundations (3–4 hrs)
Repo, `uv` env, `typer` skeleton, `config.py`, `schema.py` with all pydantic models, `templates/*.md`, vault init.
**Ship**: `recall init`, `recall new <type>`, `recall validate`.
**Acceptance**: can hand-write a project card and have it validate; bad frontmatter produces a readable error.

### Phase 1 — Store + lexical search (5–6 hrs)
`db.py` with full DDL and FTS triggers, `vault.py`, `chunker.py`, `indexer.py`, BM25 search.
**Ship**: `recall index`, `recall search`, `recall show <id>`.
**Acceptance**: 10 hand-written cards, keyword search returns correct results in <200 ms; `reindex --all` reproduces the index exactly.

### Phase 2 — MCP server (4–5 hrs)
`memory_search`, `memory_get`, `memory_stats` over the Phase-1 index.
**Ship**: working `.mcp.json`, Claude Code can query my memory.
**Acceptance**: in a fresh Claude Code session, "what projects have I done with PyTorch?" returns real cards with paths.
> Deliberately early. Even keyword-only search wired into Claude is more useful than perfect search sitting in a CLI I forget to open.

### Phase 3 — Embeddings + hybrid retrieval (5–7 hrs)
`embedder.py` with `bge-m3`, batching, model-name tracking; vector storage; RRF fusion; date/type/tag filters.
**Ship**: `recall search --semantic`, hybrid by default.
**Acceptance**: a query phrased with *no* shared vocabulary with the target card still retrieves it. **A Persian query retrieves a relevant English card** — verify explicitly with a real Farsi test query.

### Phase 4 — Folder ingestion (10–14 hrs) — THE BIG ONE
`harvest.py`, `synthesize.py`, `backends.py`, `review.py`, prompts.
**Ship**: `recall ingest`, `recall draft`, `recall review`, `recall remember`.
**Acceptance**: point it at the MARL inventory repo → evidence bundle under 50k chars → draft card with correct dates from git history, correct stack, and honest `UNKNOWN`s for client and outcome → review → committed and searchable. Then repeat on a repo with **no** git history and **no** README, and confirm it degrades gracefully rather than hallucinating.

### Phase 5 — Backfill (8–12 hrs, mostly my time not coding)
Run ingestion across the archive. Add people, episodes, artifacts. Build `recall import`, `recall triage`.
**Ship**: a populated memory.
**Acceptance**: ≥ 25 project cards, ≥ 10 people, ≥ 10 episodes. `recall stats` shows continuous coverage 2019→present.

### Phase 6 — Refinement (6–10 hrs, optional)
Reranker, `recall verify` / `doctor`, entity merge, `related`, timeline view, an `export --visibility shareable` that emits portfolio-ready markdown for soroush-thr.github.io.
**Ship**: durability, plus a genuine side-benefit — the portfolio writes itself from memory.

**Total: ~40–55 hours.** Usable after Phase 2 (~13 hrs). Delivering the headline feature at Phase 4 (~28 hrs).

---

## 13. Testing requirements

- `tests/fixtures/sample_repo/` — a small fake project with a README, `requirements.txt`, three Python files, a notebook, and a real `.git` directory created in test setup with three backdated commits. Harvest tests assert exact extracted dates and dependency lists.
- **Round-trip test**: write card → index → search → retrieve → parsed frontmatter equals original. Include a card with Persian body text; assert no mojibake. This is the single most important test given Windows encoding defaults.
- **Idempotency**: running `index` twice changes nothing (assert identical row counts and hashes).
- **Rebuild**: `reindex --all` on a populated vault produces byte-identical search results.
- **Path robustness**: a vault path containing a space and a non-ASCII character.
- Harvest cap: a fixture repo with 500 files produces a bundle under the character cap and logs the drops.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **LLM invents project details** | Evidence-bundle-only prompting, mandatory `UNKNOWN` markers, inline file citations, human review gate blocking commit. |
| **Windows mangles Persian text** | `encoding="utf-8"` on every open; explicit round-trip test with Farsi content. |
| **Scope creep into a note-taking app** | Anti-goals in §0 are binding. No feature that isn't retrieval or ingestion. |
| **Embedding model changes orphan the index** | `model` column on every embedding; `doctor` detects mismatch; `reindex --all` always works. |
| **Client-confidential data leaks** | `visibility` field; MCP public-only mode; vault in a private repo; consider a separate `vault-confidential/` excluded from MCP entirely. |
| **Abandonment after Phase 2** | Phases 0–2 alone are worth having. Markdown survives regardless. |
| **Huge repos blow up harvest** | Hard caps, deny-list, `--dry-run` first. |
| **Backfill never happens** | Time-box it: five projects per sitting, oldest-first, accept `confidence: medium`. An incomplete memory beats an empty one. |

---

## 15. Obsidian as the human interface

The vault is plain markdown, so Obsidian opens it directly with no adapter. Obsidian is the **read/edit/browse** surface; Recall is the **ingest/search/MCP** surface. They share files and never conflict.

### Hard compatibility rules

1. **`id` MUST equal the filename stem.** `id: prj-marl-inventory-2024` lives at `projects/prj-marl-inventory-2024.md`. This is what makes `[[prj-marl-inventory-2024]]` resolve as a real Obsidian link while remaining a stable primary key for SQLite. Enforce in `schema.py` validation and in `recall doctor`.
2. **Entity references use wikilink syntax** (`"[[slug]]"`) in frontmatter. Recall's parser strips `[[ ]]` before lookup; Obsidian renders them as navigable links and populates its backlinks pane. One notation, both consumers.
3. **Normalized hashing** (see §4.4) — mandatory, or Obsidian edits cause spurious re-embedding.
4. **Frontmatter must stay Dataview-queryable**: flat scalars and flat lists at the top level. Nested dicts (`provenance.*`) are fine; **lists of dicts are not** — Dataview handles them poorly. Keep `provenance.sources` a flat list of strings.

### Vault additions

```
memory-vault/
├── .obsidian/          # Obsidian config — commit core config, ignore workspace churn
├── attachments/        # Obsidian drops pasted images here; Recall MUST skip this dir
└── dashboards/         # Dataview query notes (see below)
```

`.gitignore` additions:
```
.obsidian/workspace*
.obsidian/cache/
.obsidian/plugins/*/data.json
.recall/
```

Recall's indexer must skip `.obsidian/`, `attachments/`, and `dashboards/` — dashboards contain query code, not memories, and indexing them pollutes search results.

### Plugins worth installing

| Plugin | Why |
|---|---|
| **Dataview** | The big one. SQL-ish queries over frontmatter. Replaces `recall timeline`, `recall stats`, and much of `recall related`. |
| **Omnisearch** | Better lexical search than core. Still not semantic — does not replace Recall. |
| **QuickAdd** or **Templater** | Capture and templating from inside Obsidian. |

Example dashboard note (`dashboards/projects-by-year.md`):
````
```dataview
TABLE subtype, status, tech, ended
FROM "projects"
WHERE started >= "2024"
SORT started DESC
```
````

That single block replaces a Phase 6 CLI feature. **Before building any Phase 6 view command, check whether a Dataview query already does it.**

### Build impact

Drop from Phase 6: timeline view, `recall stats` coverage report, most of `recall related`. Keep: reranker, `verify`, `doctor`, `entity merge`, portfolio export. Saves roughly 4–6 hours.

---

## 16. Storage and backup posture

**Local-first is the correct call here**, for reasons specific to this data:

- Freelance work contains **client-confidential material**. Third-party hosting may breach contract terms, and at minimum expands the trust surface for no benefit.
- Local embeddings mean the content never leaves the machine — no inference-time leakage.
- Sub-100 ms search, zero cost, no vendor to outlive.

But **local is not the same as safe.** A single disk is a single point of failure, and a cross-continental relocation is exactly when laptops get stolen, dropped, or seized. Tier the storage:

| Tier | Contents | Where it lives |
|---|---|---|
| **Standard** | Personal episodes, contests, research, public projects, people | Vault → **private** GitHub repo. Git history is the backup. |
| **Confidential** | Named clients, contract terms, anything under NDA | `vault-confidential/`, git-excluded entirely. Backed up to an **encrypted** external drive (VeraCrypt or BitLocker To Go), refreshed monthly. |
| **Derived** | `.recall/` — index, embeddings, evidence, drafts | Never backed up. Rebuildable via `recall reindex --all`. |

Set `visibility: confidential` on tier-2 cards. Recall's MCP server must exclude them by default when `RECALL_MCP_PUBLIC_ONLY=1`, and `recall export` must never emit them regardless of flags.

If GitHub-as-backup is itself too much trust for the standard tier, the alternatives in order of effort: a bare git remote on an external drive (`git remote add backup /d/backups/memory.git`), a self-hosted Forgejo instance, or `git-crypt` for transparent field-level encryption on a hosted remote.

**Verify the backup actually restores.** Once, on purpose: clone to a scratch directory, run `recall reindex --all`, confirm search works. An untested backup is a hypothesis.

---

## 17. Notes for the implementing agent

- Build strictly phase by phase. Do not stub Phase 4 while building Phase 1.
- Every phase ends with passing tests and a working command. No half-wired features on `main`.
- Prefer stdlib. Every dependency added must be justified in `pyproject.toml` with a comment.
- No decorative comment separators. Conventional Commits.
- Write `README.md` incrementally as commands ship — it is the acceptance artifact.
- When a design decision is ambiguous, **ask rather than assume**; this is a system whose value depends on trusting its contents.
