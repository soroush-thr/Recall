You are drafting an `episode` memory card for Recall, a personal memory system, from an
evidence bundle harvested from a project folder. An episode is a bounded event, not a
deliverable — use this template when the folder documents something that happened rather than
something built (e.g. a conference, a move, an incident). Follow these rules exactly.

## Output format

Emit a single markdown file: YAML frontmatter between `---` lines, followed by the body.
No preamble, no explanation, no markdown code fence around the whole output.

## Frontmatter fields (all required unless noted optional)

```yaml
id: {slug}
type: episode
title: "..."
aliases: []                        # optional
lang: en                           # en|fa|mixed
started: null                      # YYYY / YYYY-MM / YYYY-MM-DD, from git history ONLY, else null
ended: null                        # same rule
tags: []
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
significance: medium               # high|medium|low — best guess from evidence
location: null                     # usually UNKNOWN unless evidence names a place
```

## Body — exactly these headings, in this order, nothing else

```markdown
## What Happened
## Context
## Why It Mattered
## People Involved
## What I Took From It
```

## Hard rules

- **Every factual claim must be traceable to the evidence bundle.** When a claim comes from a
  specific file, append `` `<path>` `` inline right after the claim.
- **Never guess.** Anything not derivable from the evidence bundle must be written literally as
  `UNKNOWN — <what is missing>`. This almost always applies to: who was involved beyond commit
  authors, why the episode mattered subjectively, and what was taken from it.
- **Dates**: infer `started`/`ended` from git history only. If the bundle has no `git` section,
  both are `null` and you must note `UNKNOWN — no git history to infer dates`.
- **Do not invent details.** Quote and attribute claims to their source file rather than
  asserting them.
- Set `confidence: low` if the bundle has no documentation AND no git section; otherwise
  `medium`.

## Evidence bundle

```json
{evidence_json}
```
