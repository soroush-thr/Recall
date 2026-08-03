"""`recall export --visibility shareable`: emit portfolio-ready markdown copies.

Per CLAUDE.md, `visibility: confidential` is excluded unconditionally everywhere data leaves
the vault; that filter is applied here regardless of the requested `visibility` value.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from recall.vault import iter_card_paths, load_card

# Internal bookkeeping fields stripped from the exported copy; a public portfolio page has
# no business showing provenance, verification bookkeeping, or the visibility flag itself.
INTERNAL_FIELDS = {"provenance", "last_verified", "confidence", "visibility"}


def export_cards(vault_path: Path, out_dir: Path, *, visibility: str = "shareable") -> list[Path]:
    if visibility == "confidential":
        raise ValueError("cannot export confidential cards")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in iter_card_paths(vault_path):
        card = load_card(path)
        if card.card.visibility == "confidential" or card.card.visibility != visibility:
            continue
        fm = card.frontmatter_dict
        public_fm = {k: v for k, v in fm.items() if k not in INTERNAL_FIELDS}
        post = frontmatter.Post(card.body, **public_fm)
        text = frontmatter.dumps(post)
        dest = out_dir / f"{card.card.id}.md"
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(dest)
    return written
