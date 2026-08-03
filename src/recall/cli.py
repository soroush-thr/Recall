"""Typer CLI app for recall."""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import typer
from pydantic import ValidationError

from recall.config import load_settings, write_default_config
from recall.db import reset as reset_db
from recall.entities import merge_entities
from recall.export import export_cards
from recall.hygiene import find_stale, reconfirm_card, run_doctor, fix_doctor_report
from recall.indexer import build_index
from recall.ingest.backends import HandoffRequired, get_backend
from recall.ingest.bulk import bulk_ingest
from recall.ingest.harvest import harvest, harvest_and_store
from recall.ingest.review import UnknownMarkersRemain, commit_draft, inspect_draft
from recall.ingest.synthesize import draft as run_draft
from recall.ingest.triage import scan_inbox, commit_item as triage_commit_item
from recall.schema import CARD_TYPES
from recall.search import search as run_search
from recall.vault import TYPE_DIRS, load_card, validate_vault

app = typer.Typer(add_completion=False, help="Recall — a local-first personal memory system.")
entity_app = typer.Typer(add_completion=False, help="Entity maintenance (manual, deterministic).")
app.add_typer(entity_app, name="entity")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


def _slugify(title: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-+", "-", slug)


@app.command()
def init(
    vault_path: Path = typer.Argument(Path.cwd, help="Directory to initialize as a memory vault."),
) -> None:
    """Initialize a new vault: create type directories and default config."""
    vault_path = vault_path.resolve()
    vault_path.mkdir(parents=True, exist_ok=True)
    for dirname in TYPE_DIRS.values():
        (vault_path / dirname).mkdir(parents=True, exist_ok=True)
    config_file = write_default_config(vault_path)
    typer.echo(f"Initialized vault at {vault_path}")
    typer.echo(f"Config: {config_file}")


@app.command()
def new(
    doc_type: str = typer.Argument(..., help=f"One of: {', '.join(CARD_TYPES.keys())}"),
    title: str = typer.Option(..., "--title", help="Card title."),
    id: str = typer.Option(None, "--id", help="Slug id; derived from title if omitted."),
) -> None:
    """Create a new card from its template and open it in $EDITOR."""
    if doc_type not in CARD_TYPES:
        typer.echo(f"Unknown type {doc_type!r}. Must be one of: {', '.join(CARD_TYPES.keys())}", err=True)
        raise typer.Exit(1)

    settings = load_settings()
    slug = id or _slugify(title)
    template_path = TEMPLATES_DIR / f"{doc_type}.md"
    template_text = template_path.read_text(encoding="utf-8")
    today = date.today().isoformat()
    rendered = template_text.format(id=slug, title=title, today=today)

    dest = settings.vault_path / TYPE_DIRS[doc_type] / f"{slug}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        typer.echo(f"Card already exists: {dest}", err=True)
        raise typer.Exit(1)
    dest.write_text(rendered, encoding="utf-8")
    typer.echo(f"Created {dest}")

    import os
    import subprocess

    editor = settings.editor
    try:
        subprocess.run([editor, str(dest)], check=False)
    except FileNotFoundError:
        typer.echo(f"Could not launch editor {editor!r}; edit {dest} manually.")


@app.command()
def validate(
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Validate every card in the vault against the schema."""
    settings = load_settings(vault_path)
    errors = validate_vault(settings.vault_path)
    if not errors:
        typer.echo("All cards valid.")
        return
    for path, message in errors:
        typer.echo(f"{path}: {message}", err=True)
    typer.echo(f"\n{len(errors)} card(s) failed validation.", err=True)
    raise typer.Exit(1)


@app.command()
def index(
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
    doc: str = typer.Option(None, "--doc", help="Index only this document id."),
    all: bool = typer.Option(False, "--all", help="Rebuild the index from scratch."),
    embed: bool = typer.Option(
        True, "--embed/--no-embed", help="Compute dense embeddings (downloads the model on first run)."
    ),
) -> None:
    """Build or update the SQLite index from the vault."""
    settings = load_settings(vault_path)
    if all:
        reset_db(settings.db_path)
    stats = build_index(
        settings.vault_path,
        settings.db_path,
        doc_id=doc,
        embed=embed,
        embedding_model=settings.embedding_model,
    )
    typer.echo(
        f"Indexed: {stats.added} added, {stats.updated} updated, "
        f"{stats.unchanged} unchanged, {stats.removed} removed, {stats.embedded} chunks embedded."
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    type: str = typer.Option(None, "--type", help="Filter by document type."),
    tag: str = typer.Option(None, "--tag", help="Filter by tag."),
    from_: str = typer.Option(None, "--from", help="Filter: started >= this date."),
    to: str = typer.Option(None, "--to", help="Filter: started <= this date."),
    k: int = typer.Option(10, "-k", help="Number of results."),
    rerank: bool = typer.Option(
        False, "--rerank", help="Rerank top candidates with a cross-encoder (downloads the model on first run)."
    ),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Search the memory index (hybrid: lexical BM25 + dense embeddings, fused with RRF)."""
    settings = load_settings(vault_path)
    hits = run_search(
        settings.db_path,
        query,
        doc_type=type,
        tag=tag,
        date_from=from_,
        date_to=to,
        k=k,
        embedding_model=settings.embedding_model,
        rerank=rerank,
    )
    if not hits:
        typer.echo("No results.")
        return
    for hit in hits:
        dates = f"{hit.started or '?'}–{hit.ended or ''}".rstrip("–")
        typer.echo(f"{hit.title}  [{hit.type}]  {dates}  score={hit.score:.2f}")
        typer.echo(f"    {hit.snippet.strip()[:150]}")
        typer.echo(f"    {hit.path}")


@app.command()
def show(
    doc_id: str = typer.Argument(..., help="Document id to display."),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Print the full card for a document id."""
    settings = load_settings(vault_path)
    for path in TYPE_DIRS.values():
        candidate = settings.vault_path / path / f"{doc_id}.md"
        if candidate.exists():
            card = load_card(candidate)
            typer.echo(candidate.read_text(encoding="utf-8"))
            return
    typer.echo(f"No document with id {doc_id!r} found.", err=True)
    raise typer.Exit(1)


@app.command()
def ingest(
    folder: Path = typer.Argument(..., help="Project folder to harvest."),
    doc_type: str = typer.Option("project", "--type", help=f"One of: {', '.join(CARD_TYPES.keys())}"),
    id: str = typer.Option(None, "--id", help="Slug id; derived from folder name if omitted."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Harvest only; print a summary, write nothing."
    ),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Harvest a project folder into an evidence bundle (Stage A of folder ingestion)."""
    if doc_type not in CARD_TYPES:
        typer.echo(f"Unknown type {doc_type!r}. Must be one of: {', '.join(CARD_TYPES.keys())}", err=True)
        raise typer.Exit(1)
    folder = folder.resolve()
    if not folder.is_dir():
        typer.echo(f"Not a directory: {folder}", err=True)
        raise typer.Exit(1)
    slug = id or _slugify(folder.name)

    if dry_run:
        bundle = harvest(folder, doc_type=doc_type)
        git = bundle["git"]
        typer.echo(f"Folder: {bundle['folder']}")
        typer.echo(f"Tree entries: {len(bundle['tree'])}")
        typer.echo(f"Languages: {bundle['languages']}")
        typer.echo(f"Dependencies: { {k: len(v) for k, v in bundle['dependencies'].items()} }")
        typer.echo(f"Documentation chars: {len(bundle['documentation'])}")
        if git:
            typer.echo(
                f"Git: {git['commit_count']} commits, "
                f"{git['first_commit_date']}–{git['last_commit_date']}, "
                f"{len(git['contributors'])} contributor(s)"
            )
        else:
            typer.echo("Git: none found")
        typer.echo(f"Notebooks: {len(bundle['notebooks'])}")
        typer.echo(f"Representative source files: {len(bundle['source_files'])}")
        typer.echo(f"Config files: {len(bundle['config_files'])}")
        typer.echo(f"License present: {bundle['license'] is not None}")
        typer.echo(f"Bundle size: {bundle['char_count']} chars")
        if bundle["dropped"]:
            typer.echo(f"Dropped to stay under cap: {len(bundle['dropped'])} item(s)")
        return

    settings = load_settings(vault_path)
    evidence_path = harvest_and_store(settings, folder, doc_type, slug)
    typer.echo(f"Harvested {folder} -> {evidence_path}")
    typer.echo(f"Next: recall draft {slug}")


@app.command()
def draft(
    slug: str = typer.Argument(..., help="Document id / evidence slug to draft a card for."),
    backend: str = typer.Option(None, "--backend", help="claude|ollama; defaults to config."),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Synthesize a draft card from a harvested evidence bundle (Stage B)."""
    settings = load_settings(vault_path)
    backend_name = backend or settings.llm_backend
    try:
        synthesis_backend = get_backend(backend_name, drafts_dir=settings.drafts_dir)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    try:
        draft_path = run_draft(settings, slug, synthesis_backend)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    except HandoffRequired as e:
        typer.echo(str(e))
        return
    typer.echo(f"Drafted {draft_path}")
    typer.echo(f"Next: recall review {slug}")


@app.command()
def review(
    slug: str = typer.Argument(..., help="Document id / draft slug to review."),
    allow_unknown: bool = typer.Option(
        False, "--allow-unknown", help="Allow committing with UNKNOWN — markers still present."
    ),
    edit: bool = typer.Option(
        True, "--edit/--no-edit", help="Open the draft in $EDITOR before validating."
    ),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Review, edit, and commit a drafted card into the vault (Stage C — the human gate)."""
    settings = load_settings(vault_path)
    draft_path = settings.drafts_dir / f"{slug}.md"
    if not draft_path.exists():
        typer.echo(f"No draft for {slug!r}; run 'recall draft {slug}' first.", err=True)
        raise typer.Exit(1)

    summary = inspect_draft(draft_path)
    typer.echo(f"Sections present: {', '.join(summary.sections_present) or '(none)'}")
    if summary.sections_missing:
        typer.echo(f"Sections missing: {', '.join(summary.sections_missing)}")
    typer.echo(f"Dates: {summary.started or '?'}–{summary.ended or '?'}")
    typer.echo(f"Tags: {summary.tags}  Tech: {summary.tech}")
    if summary.unknowns:
        typer.echo(f"UNKNOWN markers ({len(summary.unknowns)}):")
        for u in summary.unknowns:
            typer.echo(f"  - {u}")

    if edit:
        try:
            subprocess.run([settings.editor, str(draft_path)], check=False)
        except FileNotFoundError:
            typer.echo(
                f"Could not launch editor {settings.editor!r}; edit {draft_path} manually, "
                f"then rerun with --no-edit."
            )

    while True:
        try:
            dest = commit_draft(settings, slug, allow_unknown=allow_unknown)
        except UnknownMarkersRemain as e:
            typer.echo(str(e), err=True)
            if not edit or not typer.confirm("Reopen editor to fix?", default=True):
                raise typer.Exit(1)
            subprocess.run([settings.editor, str(draft_path)], check=False)
            continue
        except (ValidationError, ValueError) as e:
            typer.echo(f"Validation error: {e}", err=True)
            if not edit or not typer.confirm("Reopen editor to fix?", default=True):
                raise typer.Exit(1)
            subprocess.run([settings.editor, str(draft_path)], check=False)
            continue
        break
    typer.echo(f"Committed {dest}")


@app.command()
def remember(
    folder: Path = typer.Argument(..., help="Project folder to remember."),
    doc_type: str = typer.Option("project", "--type", help=f"One of: {', '.join(CARD_TYPES.keys())}"),
    id: str = typer.Option(None, "--id", help="Slug id; derived from folder name if omitted."),
    backend: str = typer.Option(None, "--backend", help="claude|ollama; defaults to config."),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Convenience wrapper: harvest -> draft -> review, with confirmations between stages."""
    if doc_type not in CARD_TYPES:
        typer.echo(f"Unknown type {doc_type!r}. Must be one of: {', '.join(CARD_TYPES.keys())}", err=True)
        raise typer.Exit(1)
    folder = folder.resolve()
    if not folder.is_dir():
        typer.echo(f"Not a directory: {folder}", err=True)
        raise typer.Exit(1)

    settings = load_settings(vault_path)
    slug = id or _slugify(folder.name)

    evidence_path = harvest_and_store(settings, folder, doc_type, slug)
    typer.echo(f"Harvested {folder} -> {evidence_path}")

    backend_name = backend or settings.llm_backend
    synthesis_backend = get_backend(backend_name, drafts_dir=settings.drafts_dir)
    try:
        draft_path = run_draft(settings, slug, synthesis_backend)
    except HandoffRequired as e:
        typer.echo(str(e))
        typer.echo(f"Then run: recall review {slug}")
        return
    typer.echo(f"Drafted {draft_path}")

    if not typer.confirm("Open for review now?", default=True):
        typer.echo(f"Run 'recall review {slug}' when ready.")
        return
    review(slug=slug, allow_unknown=False, edit=True, vault_path=vault_path)


@app.command(name="import")
def import_(
    glob: str = typer.Option(..., "--glob", help='Glob pattern for folders, e.g. "D:/projects/*".'),
    doc_type: str = typer.Option("project", "--type", help=f"One of: {', '.join(CARD_TYPES.keys())}"),
    backend: str = typer.Option(None, "--backend", help="claude|ollama; defaults to config."),
    review_now: bool = typer.Option(
        True, "--review/--no-review", help="Walk the drafted queue for review after harvesting/drafting."
    ),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Bulk-ingest many folders: harvest all, draft all, then review one at a time."""
    if doc_type not in CARD_TYPES:
        typer.echo(f"Unknown type {doc_type!r}. Must be one of: {', '.join(CARD_TYPES.keys())}", err=True)
        raise typer.Exit(1)

    settings = load_settings(vault_path)
    backend_name = backend or settings.llm_backend
    synthesis_backend = get_backend(backend_name, drafts_dir=settings.drafts_dir)

    results = bulk_ingest(settings, glob, doc_type, synthesis_backend)
    if not results:
        typer.echo(f"No folders matched: {glob}")
        return

    for r in results:
        typer.echo(f"{r.slug:30s} {r.stage}" + (f"  ({r.error})" if r.error else ""))

    drafted = [r.slug for r in results if r.stage == "drafted"]
    handoff = [r.slug for r in results if r.stage == "handoff"]
    failed = [r for r in results if r.stage == "failed"]
    typer.echo(
        f"\n{len(drafted)} drafted, {len(handoff)} awaiting handoff, {len(failed)} failed."
    )
    if handoff:
        typer.echo("Run 'recall draft <slug>' again for handoff items once the prompt is answered.")

    if not drafted:
        return
    if not review_now or not typer.confirm(f"Review {len(drafted)} drafted card(s) now?", default=True):
        for slug in drafted:
            typer.echo(f"Run 'recall review {slug}' when ready.")
        return

    for slug in drafted:
        typer.echo(f"\n--- Reviewing {slug} ---")
        review(slug=slug, allow_unknown=False, edit=True, vault_path=vault_path)
        if not typer.confirm("Continue to next draft?", default=True):
            break


@app.command()
def triage(
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Walk notes/inbox/ and turn loose captures into real cards (accept, edit, or defer each)."""
    settings = load_settings(vault_path)
    items = scan_inbox(settings.vault_path)
    if not items:
        typer.echo("Inbox is empty.")
        return

    for item in items:
        typer.echo(f"\n--- {item.path.name} ---")
        typer.echo(item.text.strip()[:300])
        typer.echo(
            f"\nProposed: type={item.proposed_type}  title={item.proposed_title!r}  "
            f"id={item.proposed_id}  tags={item.proposed_tags}"
        )
        action = typer.prompt("Accept, edit, or defer? [a/e/d]", default="a")
        if action.lower().startswith("d"):
            typer.echo("Deferred.")
            continue

        doc_type = item.proposed_type
        title = item.proposed_title
        doc_id = item.proposed_id
        tags = item.proposed_tags
        if action.lower().startswith("e"):
            doc_type = typer.prompt("Type", default=doc_type)
            if doc_type not in CARD_TYPES:
                typer.echo(f"Unknown type {doc_type!r}; deferring.", err=True)
                continue
            title = typer.prompt("Title", default=title)
            doc_id = typer.prompt("Id", default=doc_id)
            tags_str = typer.prompt("Tags (comma-separated)", default=", ".join(tags))
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        try:
            dest = triage_commit_item(settings, item, doc_type=doc_type, title=title, doc_id=doc_id, tags=tags)
        except (ValidationError, ValueError, FileExistsError) as e:
            typer.echo(f"Could not commit: {e}", err=True)
            continue
        typer.echo(f"Committed {dest}")


@app.command()
def verify(
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Flag cards with stale last_verified (>180d) or missing provenance source paths; walk them
    interactively (re-confirm / update in editor / skip)."""
    settings = load_settings(vault_path)
    flags = find_stale(settings.vault_path)
    if not flags:
        typer.echo("Nothing to verify — all cards are fresh.")
        return

    for flag in flags:
        typer.echo(f"\n--- {flag.doc_id} ---")
        for reason in flag.reasons:
            typer.echo(f"  - {reason}")
        action = typer.prompt("Re-confirm, update in editor, or skip? [y/u/s]", default="s")
        if action.lower().startswith("y"):
            reconfirm_card(settings.vault_path, flag.path)
            typer.echo(f"Re-confirmed {flag.doc_id}.")
        elif action.lower().startswith("u"):
            try:
                subprocess.run([settings.editor, str(flag.path)], check=False)
            except FileNotFoundError:
                typer.echo(f"Could not launch editor {settings.editor!r}; edit {flag.path} manually.")
            reconfirm_card(settings.vault_path, flag.path)
            typer.echo(f"Updated and re-confirmed {flag.doc_id}.")
        else:
            typer.echo(f"Skipped {flag.doc_id}.")


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Repair what's safely repairable."),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Integrity check over the index and vault (orphaned chunks, content drift, missing/stale
    embeddings, duplicate ids, schema validation failures)."""
    settings = load_settings(vault_path)
    report = run_doctor(settings.vault_path, settings.db_path, embedding_model=settings.embedding_model)

    if report.is_clean:
        typer.echo("No issues found.")
        return

    if report.orphaned_chunk_ids:
        typer.echo(f"Orphaned chunks ({len(report.orphaned_chunk_ids)}): {report.orphaned_chunk_ids}")
    if report.drifted_doc_ids:
        typer.echo(f"Content drift (db out of sync with vault file): {report.drifted_doc_ids}")
    if report.chunks_missing_embeddings:
        typer.echo(f"Chunks missing embeddings ({len(report.chunks_missing_embeddings)})")
    if report.stale_model_chunk_ids:
        for model, chunk_ids in report.stale_model_chunk_ids.items():
            typer.echo(f"Embeddings from stale model {model!r}: {len(chunk_ids)} chunk(s)")
    if report.duplicate_ids:
        for doc_id, paths in report.duplicate_ids.items():
            typer.echo(f"Duplicate id {doc_id!r}: {paths}")
    if report.schema_failures:
        for path, message in report.schema_failures:
            typer.echo(f"Schema validation failed: {path}: {message}", err=True)

    if fix:
        fix_doctor_report(
            settings.vault_path, settings.db_path, report, embedding_model=settings.embedding_model
        )
        typer.echo(
            f"\nFixed: reindexed {len(report.drifted_doc_ids)} drifted doc(s), "
            f"removed {len(report.orphaned_chunk_ids)} orphaned chunk row(s). "
            "Missing/stale embeddings, duplicate ids, and schema failures were not auto-fixed."
        )
    else:
        raise typer.Exit(1)


@entity_app.command(name="merge")
def entity_merge(
    id_a: str = typer.Argument(..., help="Entity id to fold away."),
    id_b: str = typer.Argument(..., help="Entity id to keep."),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Fold entity id-a into id-b: merges mentions, rewrites frontmatter references, deletes id-a."""
    settings = load_settings(vault_path)
    try:
        rewritten = merge_entities(
            settings.vault_path, settings.db_path, id_a, id_b, embedding_model=settings.embedding_model
        )
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(f"Merged {id_a!r} into {id_b!r}. Rewrote {len(rewritten)} card(s): {rewritten}")


@app.command(name="export")
def export_(
    visibility: str = typer.Option("shareable", "--visibility", help="Visibility level to export."),
    out: Path = typer.Option(None, "--out", help="Output directory; defaults to <vault>/.recall/export/."),
    vault_path: Path = typer.Option(None, "--vault", help="Vault path; defaults to config/env."),
) -> None:
    """Export cards at the given visibility level as portfolio-ready markdown (never private/confidential)."""
    settings = load_settings(vault_path)
    out_dir = out or (settings.recall_dir / "export")
    try:
        written = export_cards(settings.vault_path, out_dir, visibility=visibility)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    if not written:
        typer.echo(f"No cards with visibility={visibility!r} found.")
        return
    typer.echo(f"Exported {len(written)} card(s) to {out_dir}")


if __name__ == "__main__":
    app()
