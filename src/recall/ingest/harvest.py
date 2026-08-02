"""Stage A: deterministic folder harvest into an evidence bundle. No LLM calls.

See RECALL-BUILD-PLAN.md §5 for the full spec this implements.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

import pathspec
import yaml

from recall import db as db_module

MAX_BUNDLE_CHARS = 50_000
MAX_FILE_BYTES = 512 * 1024
MAX_TREE_DEPTH = 4
MAX_TREE_ENTRIES = 300
MAX_DOC_CHARS = 20_000
MAX_NOTEBOOKS = 5
MAX_NOTEBOOK_CHARS = 8_000
MAX_SOURCE_FILES = 12
MAX_SOURCE_FILE_LINES = 150

DENY_DIRS = {
    "node_modules", ".venv", "venv", "__pycache__", ".git", "dist", "build",
    "data", "datasets", "outputs", "wandb", "mlruns", ".ipynb_checkpoints",
}
DENY_EXTS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".h5", ".pkl", ".parquet",
    ".zip", ".tar.gz", ".mp4", ".wav", ".png", ".jpg", ".pdf",
}

_KEYWORD_RE = re.compile(r"main|train|app|run|pipeline|model|server|index", re.IGNORECASE)
_TEST_PENALTY_RE = re.compile(r"^test_|conftest|^setup\.py$", re.IGNORECASE)
_PY_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([\w\.]+)", re.MULTILINE)
_JS_IMPORT_RE = re.compile(r"""(?:require\(|from\s+)['"]([^'"]+)['"]""")


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
    except OSError:
        return True
    return b"\x00" in chunk


def _load_gitignore_spec(folder: Path) -> pathspec.PathSpec | None:
    gitignore = folder / ".gitignore"
    if not gitignore.exists():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _is_excluded(rel_path: Path, spec: pathspec.PathSpec | None) -> bool:
    parts = rel_path.parts
    if any(part in DENY_DIRS for part in parts):
        return True
    if rel_path.suffix.lower() in DENY_EXTS or rel_path.name.endswith(".tar.gz"):
        return True
    if spec is not None and spec.match_file(rel_path.as_posix()):
        return True
    return False


def _iter_included_files(folder: Path):
    spec = _load_gitignore_spec(folder)
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        if _is_excluded(rel, spec):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if _is_binary(path):
            continue
        yield path, rel


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _build_tree(folder: Path, files: list[Path]) -> list[str]:
    entries: list[str] = []
    for path in files:
        rel = path.relative_to(folder)
        if len(rel.parts) > MAX_TREE_DEPTH:
            continue
        entries.append(rel.as_posix())
        if len(entries) >= MAX_TREE_ENTRIES:
            break
    return entries


def _language_histogram(files: list[Path]) -> dict[str, dict[str, int]]:
    hist: dict[str, dict[str, int]] = {}
    for path in files:
        ext = path.suffix.lower() or "(none)"
        bucket = hist.setdefault(ext, {"files": 0, "lines": 0})
        bucket["files"] += 1
        try:
            bucket["lines"] += _read_text(path).count("\n") + 1
        except OSError:
            pass
    return hist


def _parse_requirements_txt(path: Path) -> dict[str, str | None]:
    deps: dict[str, str | None] = {}
    for line in _read_text(path).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=|>|<)?\s*([\w.\-]*)$", line)
        if m:
            deps[m.group(1)] = m.group(3) or None
    return deps


def _parse_pyproject_toml(path: Path) -> dict[str, str | None]:
    deps: dict[str, str | None] = {}
    try:
        data = tomllib.loads(_read_text(path))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return deps
    for dep in data.get("project", {}).get("dependencies", []):
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", dep)
        if m:
            deps[m.group(1)] = m.group(2) or None
    return deps


def _parse_package_json(path: Path) -> dict[str, str | None]:
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError:
        return {}
    deps: dict[str, str | None] = {}
    for key in ("dependencies", "devDependencies"):
        deps.update(data.get(key, {}) or {})
    return deps


def _parse_environment_yml(path: Path) -> dict[str, str | None]:
    try:
        data = yaml.safe_load(_read_text(path)) or {}
    except yaml.YAMLError:
        return {}
    deps: dict[str, str | None] = {}
    for dep in data.get("dependencies", []) or []:
        if isinstance(dep, str):
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:[=<>~]+)?\s*([\w.\-]*)$", dep)
            if m:
                deps[m.group(1)] = m.group(2) or None
    return deps


def _parse_cargo_toml(path: Path) -> dict[str, str | None]:
    deps: dict[str, str | None] = {}
    try:
        data = tomllib.loads(_read_text(path))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return deps
    for name, spec in (data.get("dependencies") or {}).items():
        deps[name] = spec if isinstance(spec, str) else None
    return deps


def _parse_go_mod(path: Path) -> dict[str, str | None]:
    deps: dict[str, str | None] = {}
    for line in _read_text(path).splitlines():
        m = re.match(r"^\s*([\w./\-]+)\s+(v[\w.\-+]+)", line.strip())
        if m and not line.strip().startswith(("module", "go ")):
            deps[m.group(1)] = m.group(2)
    return deps


_MANIFEST_PARSERS = {
    "requirements.txt": _parse_requirements_txt,
    "pyproject.toml": _parse_pyproject_toml,
    "package.json": _parse_package_json,
    "environment.yml": _parse_environment_yml,
    "Cargo.toml": _parse_cargo_toml,
    "go.mod": _parse_go_mod,
}


def _collect_dependencies(folder: Path, files: list[Path]) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    names = {path.relative_to(folder).name: path for path in files}
    for filename, parser in _MANIFEST_PARSERS.items():
        path = names.get(filename)
        if path is not None:
            result[filename] = parser(path)
    return result


def _collect_documentation(folder: Path, files: list[Path]) -> str:
    parts: list[str] = []
    total = 0
    doc_files = [
        p for p in files
        if p.name.upper().startswith("README") or p.name.upper().startswith("CONTRIBUTING")
        or ("docs" in p.relative_to(folder).parts and p.suffix.lower() == ".md")
    ]
    for path in doc_files:
        text = _read_text(path)
        remaining = MAX_DOC_CHARS - total
        if remaining <= 0:
            break
        chunk = text[:remaining]
        parts.append(f"--- {path.relative_to(folder).as_posix()} ---\n{chunk}")
        total += len(chunk)
    return "\n\n".join(parts)


def _run_git(folder: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _collect_git_info(folder: Path) -> dict | None:
    if not (folder / ".git").exists():
        return None
    first_date = _run_git(folder, ["log", "--reverse", "--format=%ad", "--date=short"])
    first_date = first_date.splitlines()[0] if first_date and first_date.strip() else None
    last_date = _run_git(folder, ["log", "-1", "--format=%ad", "--date=short"])
    last_date = last_date.strip() if last_date else None
    count_out = _run_git(folder, ["rev-list", "--count", "HEAD"])
    commit_count = int(count_out.strip()) if count_out and count_out.strip().isdigit() else 0
    shortlog = _run_git(folder, ["shortlog", "-sne", "HEAD"])
    contributors = []
    if shortlog:
        for line in shortlog.splitlines():
            m = re.match(r"^\s*(\d+)\s+(.+)$", line)
            if m:
                contributors.append({"name": m.group(2).strip(), "commits": int(m.group(1))})
    branches_out = _run_git(folder, ["branch", "-a", "--format=%(refname:short)"])
    branches = [b.strip() for b in (branches_out or "").splitlines() if b.strip()]
    tags_out = _run_git(folder, ["tag"])
    tags = [t.strip() for t in (tags_out or "").splitlines() if t.strip()]
    subjects_out = _run_git(folder, ["log", "-30", "--format=%s"])
    recent_subjects = [s for s in (subjects_out or "").splitlines() if s.strip()]
    stat_out = _run_git(folder, ["log", "--format=", "--name-only"])
    churn: dict[str, int] = {}
    for line in (stat_out or "").splitlines():
        line = line.strip()
        if line:
            churn[line] = churn.get(line, 0) + 1
    most_churned = sorted(churn.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "first_commit_date": first_date,
        "last_commit_date": last_date,
        "commit_count": commit_count,
        "contributors": contributors,
        "branches": branches,
        "tags": tags,
        "recent_subjects": recent_subjects,
        "most_churned_files": [{"path": p, "changes": c} for p, c in most_churned],
    }


def _collect_notebooks(folder: Path, files: list[Path]) -> list[dict]:
    notebooks = [p for p in files if p.suffix == ".ipynb"][:MAX_NOTEBOOKS]
    result = []
    for path in notebooks:
        try:
            data = json.loads(_read_text(path))
        except json.JSONDecodeError:
            continue
        cells = []
        total = 0
        for cell in data.get("cells", []):
            source = "".join(cell.get("source", []))
            remaining = MAX_NOTEBOOK_CHARS - total
            if remaining <= 0:
                break
            source = source[:remaining]
            cells.append({"cell_type": cell.get("cell_type"), "source": source})
            total += len(source)
        result.append({"path": path.relative_to(folder).as_posix(), "cells": cells})
    return result


def _python_signatures(text: str) -> list[str]:
    sigs = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return sigs
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            sigs.append(f"def {node.name}({args})")
        elif isinstance(node, ast.ClassDef):
            sigs.append(f"class {node.name}")
    return sigs


def _regex_signatures(text: str) -> list[str]:
    sigs = []
    for m in re.finditer(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", text, re.MULTILINE):
        sigs.append(f"function {m.group(1)}(...)")
    for m in re.finditer(r"^\s*(?:export\s+)?class\s+(\w+)", text, re.MULTILINE):
        sigs.append(f"class {m.group(1)}")
    return sigs


def _import_targets(text: str, suffix: str) -> list[str]:
    if suffix == ".py":
        return _PY_IMPORT_RE.findall(text)
    if suffix in (".js", ".jsx", ".ts", ".tsx"):
        return _JS_IMPORT_RE.findall(text)
    return []


def _select_representative_files(folder: Path, files: list[Path]) -> list[dict]:
    candidates = [p for p in files if p.suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}]
    if not candidates:
        return []

    import_counts: dict[str, int] = {}
    for path in candidates:
        text = _read_text(path)
        for target in _import_targets(text, path.suffix):
            module = target.split(".")[-1]
            import_counts[module] = import_counts.get(module, 0) + 1

    scored = []
    for path in candidates:
        rel = path.relative_to(folder)
        score = 0.0
        if _KEYWORD_RE.search(path.stem):
            score += 10
        if len(rel.parts) == 1 or rel.parts[0] == "src":
            score += 5
        score += import_counts.get(path.stem, 0) * 2
        if _TEST_PENALTY_RE.search(path.name):
            score -= 20
        scored.append((score, path))

    scored.sort(key=lambda sp: sp[0], reverse=True)
    selected = scored[:MAX_SOURCE_FILES]

    result = []
    for _, path in selected:
        text = _read_text(path)
        lines = text.splitlines()[:MAX_SOURCE_FILE_LINES]
        sigs = _python_signatures(text) if path.suffix == ".py" else _regex_signatures(text)
        result.append({
            "path": path.relative_to(folder).as_posix(),
            "lines": "\n".join(lines),
            "signatures": sigs,
        })
    return result


_CONFIG_NAMES = {"Dockerfile", "docker-compose.yml", "Makefile"}
_CONFIG_SUFFIXES = {".slurm", ".sl"}


def _collect_config_files(folder: Path, files: list[Path]) -> list[dict]:
    result = []
    for path in files:
        rel = path.relative_to(folder)
        is_workflow = rel.parts[:2] == (".github", "workflows")
        if path.name in _CONFIG_NAMES or path.suffix in _CONFIG_SUFFIXES or is_workflow:
            result.append({"path": rel.as_posix(), "text": _read_text(path)})
    return result


def _collect_license(folder: Path, files: list[Path]) -> str | None:
    for path in files:
        if path.name.upper().startswith("LICENSE") or path.name == "CITATION.cff":
            return _read_text(path)
    return None


def _enforce_char_cap(bundle: dict) -> None:
    dropped: list[str] = []

    def _size() -> int:
        return len(json.dumps(bundle, ensure_ascii=False))

    if _size() <= MAX_BUNDLE_CHARS and not bundle["source_files"]:
        bundle["dropped"] = dropped
        return

    while _size() > MAX_BUNDLE_CHARS and bundle["source_files"]:
        bundle["source_files"].pop()
        dropped.append("source_file_body")

    while _size() > MAX_BUNDLE_CHARS and bundle["notebooks"]:
        bundle["notebooks"].pop()
        dropped.append("notebook_code")

    if _size() > MAX_BUNDLE_CHARS and bundle["git"]:
        while _size() > MAX_BUNDLE_CHARS and len(bundle["git"]["recent_subjects"]) > 1:
            bundle["git"]["recent_subjects"].pop()
            dropped.append("commit_list_entry")

    while _size() > MAX_BUNDLE_CHARS and len(bundle["tree"]) > 1:
        bundle["tree"].pop()
        dropped.append("tree_entry")

    bundle["dropped"] = dropped


def harvest(folder: Path, *, doc_type: str) -> dict:
    """Walk `folder` and build a deterministic evidence bundle. No LLM calls."""
    folder = folder.resolve()
    included = list(_iter_included_files(folder))
    files = [path for path, _ in included]

    bundle = {
        "folder": str(folder),
        "doc_type": doc_type,
        "tree": _build_tree(folder, files),
        "languages": _language_histogram(files),
        "dependencies": _collect_dependencies(folder, files),
        "documentation": _collect_documentation(folder, files),
        "git": _collect_git_info(folder),
        "notebooks": _collect_notebooks(folder, files),
        "source_files": _select_representative_files(folder, files),
        "config_files": _collect_config_files(folder, files),
        "license": _collect_license(folder, files),
    }
    _enforce_char_cap(bundle)
    bundle["char_count"] = len(json.dumps(bundle, ensure_ascii=False))
    return bundle


def harvest_and_store(settings, folder: Path, doc_type: str, slug: str) -> Path:
    """Harvest `folder`, write the evidence bundle under `settings.evidence_dir`, and log the run."""
    bundle = harvest(folder, doc_type=doc_type)
    bundle["harvested_at"] = date.today().isoformat()
    settings.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = settings.evidence_dir / f"{slug}.json"
    evidence_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    conn = db_module.connect(settings.db_path)
    try:
        db_module.record_ingest_status(conn, doc_id=slug, source=str(folder), status="harvested")
    finally:
        conn.close()
    return evidence_path
