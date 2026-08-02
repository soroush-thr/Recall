"""Stage B synthesis backends: turn a rendered prompt into a draft card.

Two backends behind one interface, per RECALL-BUILD-PLAN.md §5. No paid APIs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol


class SynthesisBackend(Protocol):
    def synthesize(self, prompt: str, *, slug: str) -> str:
        """Return the drafted card's full markdown text (frontmatter + body)."""
        ...


class HandoffRequired(Exception):
    """Raised by ClaudeCodeBackend: synthesis must happen in an interactive Claude Code session."""

    def __init__(self, prompt_path: Path):
        self.prompt_path = prompt_path
        super().__init__(
            f"Prompt written to {prompt_path}. Open it in Claude Code (or paste its contents "
            f"into a session), then save the reply to the matching draft file under "
            f"'.recall/drafts/' before running 'recall review'."
        )


class ClaudeCodeBackend:
    """File-handoff backend: writes the prompt for a human to run through Claude Code.

    Starting point per the build plan — avoids depending on a `claude -p` CLI flag surface
    that may not exist or be stable. A future iteration can shell out directly once that's
    confirmed to work reliably.
    """

    def __init__(self, drafts_dir: Path):
        self.drafts_dir = drafts_dir

    def synthesize(self, prompt: str, *, slug: str) -> str:
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.drafts_dir / f"{slug}.prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        raise HandoffRequired(prompt_path)


class OllamaBackend:
    """Offline backend: POSTs to a local Ollama server running qwen2.5:14b-instruct."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5:14b-instruct"):
        self.host = host.rstrip("/")
        self.model = model

    def synthesize(self, prompt: str, *, slug: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 16384},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {self.host}: {e}") from e
        return body["response"]


def get_backend(name: str, *, drafts_dir: Path) -> SynthesisBackend:
    if name == "claude":
        return ClaudeCodeBackend(drafts_dir)
    if name == "ollama":
        return OllamaBackend()
    raise ValueError(f"Unknown backend {name!r}. Must be one of: claude, ollama")
