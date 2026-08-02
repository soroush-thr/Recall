You are drafting a `project` memory card for Recall, a personal memory system, from an
evidence bundle harvested from a project folder. Follow these rules exactly.

## Output format

Emit a single markdown file: YAML frontmatter between `---` lines, followed by the body.
No preamble, no explanation, no markdown code fence around the whole output.

## Frontmatter fields (all required unless noted optional)

```yaml
id: {slug}
type: project
title: "..."
aliases: []                        # optional
lang: en                           # en|fa|mixed
started: null                      # YYYY / YYYY-MM / YYYY-MM-DD, from git history ONLY, else null
ended: null                        # same rule
tags: []                           # short kebab-case tags
entities:
  people: []
  orgs: []
  projects: []
visibility: private
confidence: medium                 # medium by default; low if no README AND no git history
provenance:
  method: folder-ingest
  sources: ["{folder}"]
  evidence_ref: "{evidence_ref}"
  ingested_at: {ingested_at}
last_verified: {ingested_at}
created: {ingested_at}
updated: {ingested_at}
subtype: research                  # freelance|contest|research|employment|personal|academic — best guess from evidence
status: completed                  # completed|ongoing|abandoned|paused — best guess from evidence
role: null                         # who built it — usually UNKNOWN unless evidence says
tech: []                           # from dependencies / language histogram
outcome: null                      # usually UNKNOWN — repos rarely state commercial/contest outcome
client: null                       # usually UNKNOWN unless evidence names a client
repo: null                         # git remote if discoverable, else null
```

## Body — exactly these headings, in this order, nothing else

```markdown
## Summary
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

## Hard rules

- **Every factual claim must be traceable to the evidence bundle.** When a claim comes from a
  specific file, append `` `<path>` `` inline right after the claim.
- **Never guess.** Anything not derivable from the evidence bundle must be written literally as
  `UNKNOWN — <what is missing>`. This almost always applies to: client identity, commercial
  outcome, contest placement, the author's subjective role, and why the project ended.
- **Dates**: infer `started`/`ended` from git history only (`first_commit_date` /
  `last_commit_date`). If the bundle has no `git` section, both are `null` and you must note
  `UNKNOWN — no git history to infer dates` somewhere relevant (e.g. Timeline).
- **Do not invent metrics.** If the README claims a number, quote it and attribute it to the
  README rather than asserting it as fact.
- Set `confidence: low` if the bundle has no documentation AND no git section; otherwise
  `medium`.

## Evidence bundle

```json
{evidence_json}
```
