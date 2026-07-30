"""Folder resolution and the known-agents memory for voice-driven spawns.

STT never produces a clean filesystem path, so the model passes the user's
spoken words and code does the mapping: resolve_folder() matches against real
directories under the configured source roots, and KnownAgents remembers
name -> path pairs the user has explicitly confirmed so later spawns only
need a short confirmation.
"""

import difflib
import json
import os
import re
import time
from pathlib import Path

SRC_ROOTS_ENV = "MATE_SRC_ROOTS"  # colon-separated, default ~/src
KNOWN_AGENTS_PATH = "~/.config/matebridge/known_agents.json"


def squash(name: str) -> str:
    """Normalize a spoken or filesystem name for matching: lowercase, letters
    and digits only. "Mate Bridge" == "matebridge" == "mate-bridge"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def source_roots() -> list[Path]:
    raw = os.environ.get(SRC_ROOTS_ENV, "~/src")
    return [Path(p).expanduser() for p in raw.split(":") if p.strip()]


def resolve_folder(spoken: str, roots: list[Path] | None = None) -> list[Path]:
    """Directories under the source roots matching the spoken name. Exact
    squashed match wins; otherwise close matches (catches STT spellings like
    "herder" for herdr). Multiple results mean the caller must ask."""
    target = squash(spoken)
    if not target:
        return []
    dirs: dict[str, Path] = {}
    for root in roots if roots is not None else source_roots():
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                dirs.setdefault(squash(child.name), child)
    if target in dirs:
        return [dirs[target]]
    close = difflib.get_close_matches(target, dirs.keys(), n=3, cutoff=0.75)
    return [dirs[c] for c in close]


def speakable_path(path: Path | str) -> str:
    """Path rendered for TTS: components spoken in order, no slashes.
    /home/jonesy/src/matebridge -> "home, jonesy, src, matebridge"."""
    parts = [p for p in Path(path).parts if p != "/"]
    return ", ".join(parts)


class KnownAgents:
    """Name -> path memory for spawn targets the user explicitly confirmed
    (or sent with guardrails off). Lookup is squash-matched so the spoken
    name doesn't have to reproduce the stored spelling. Backed by a small
    JSON file; every mutation writes through immediately."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or KNOWN_AGENTS_PATH).expanduser()
        self._agents: dict[str, dict] = {}
        try:
            data = json.loads(self.path.read_text())
            self._agents = dict(data.get("agents", {}))
        except (OSError, ValueError):
            self._agents = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"agents": self._agents}, indent=2) + "\n")

    def get(self, spoken: str) -> str | None:
        """Stored path for a spoken name, or None."""
        target = squash(spoken)
        for name, entry in self._agents.items():
            if squash(name) == target:
                return entry.get("path")
        return None

    def remember(self, name: str, path: str) -> None:
        for existing in list(self._agents):
            if squash(existing) == squash(name):
                del self._agents[existing]
        self._agents[name] = {"path": str(path),
                              "last_spawned": int(time.time())}
        self._save()

    def forget(self, spoken: str) -> bool:
        target = squash(spoken)
        for name in list(self._agents):
            if squash(name) == target:
                del self._agents[name]
                self._save()
                return True
        return False

    def names(self) -> list[tuple[str, str]]:
        """(name, path) pairs, most recently spawned first. remember()
        deletes-then-appends, so dict order is oldest-to-newest already —
        no timestamp sort needed (second-resolution timestamps tie)."""
        return [(name, entry.get("path", ""))
                for name, entry in reversed(self._agents.items())]
