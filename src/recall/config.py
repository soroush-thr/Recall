"""Settings loaded from <vault>/.recall/config.yaml, with env var overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_FILENAME = "config.yaml"


class Settings(BaseModel):
    vault_path: Path
    editor: str = "notepad"
    embedding_model: str = "BAAI/bge-m3"
    llm_backend: str = "claude"

    @property
    def recall_dir(self) -> Path:
        return self.vault_path / ".recall"

    @property
    def db_path(self) -> Path:
        return self.recall_dir / "index.db"

    @property
    def evidence_dir(self) -> Path:
        return self.recall_dir / "evidence"

    @property
    def drafts_dir(self) -> Path:
        return self.recall_dir / "drafts"

    @property
    def config_path(self) -> Path:
        return self.recall_dir / DEFAULT_CONFIG_FILENAME


def _resolve_vault_path() -> Path:
    env_path = os.environ.get("RECALL_VAULT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path.cwd()


def load_settings(vault_path: Path | None = None) -> Settings:
    """Load settings for the given vault, falling back to RECALL_VAULT env var
    or the current directory. Values in .recall/config.yaml override defaults."""
    vault_path = (vault_path or _resolve_vault_path()).resolve()
    config_file = vault_path / ".recall" / DEFAULT_CONFIG_FILENAME
    data: dict = {"vault_path": vault_path}
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        loaded.pop("vault_path", None)
        data.update(loaded)
    return Settings.model_validate(data)


def write_default_config(vault_path: Path) -> Path:
    vault_path = vault_path.resolve()
    recall_dir = vault_path / ".recall"
    recall_dir.mkdir(parents=True, exist_ok=True)
    config_file = recall_dir / DEFAULT_CONFIG_FILENAME
    if not config_file.exists():
        settings = Settings(vault_path=vault_path)
        payload = settings.model_dump(mode="json", exclude={"vault_path"})
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return config_file
