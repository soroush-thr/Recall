# Recall

A local-first, single-user memory system for projects and episodes. See
[RECALL-BUILD-PLAN.md](RECALL-BUILD-PLAN.md) for the full design.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

## Commands (Phase 0 + 1)

```
recall init <vault-path>              # create a new vault
recall new project --title "..."      # create a card from template, open in $EDITOR
recall validate --vault <path>        # validate all cards against schema
recall index --vault <path> [--all]   # build/update the SQLite index
recall search "<query>" --vault <path> [--type project] [--tag x] [--from 2023] [--to 2025] [-k 10]
recall show <doc-id> --vault <path>   # print a card
```

`RECALL_VAULT` env var sets the default vault path for all commands.

## Testing

```
.venv\Scripts\python -m pytest
```
