"""Typer CLI app for recall."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from recall.config import load_settings, write_default_config
from recall.db import reset as reset_db
from recall.indexer import build_index
from recall.schema import CARD_TYPES
from recall.search import search as run_search
from recall.vault import TYPE_DIRS, load_card, validate_vault

app = typer.Typer(add_completion=False, help="Recall — a local-first personal memory system.")

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


if __name__ == "__main__":
    app()
