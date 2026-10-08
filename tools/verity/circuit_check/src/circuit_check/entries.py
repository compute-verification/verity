"""circuit-check's shared lists, one file per entry, so two changes that each add an entry touch different files: an entry's
file is named by its key (`name`), and a file whose name is not its key's is an error, so no key is in two files."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

#: the longest stem a key keeps
WIDTH = 80


def name(key: str) -> str:
    """The stem of ``key``'s file: the key itself when it is a plain name (letters, digits, ``_``, ``.``, ``-``) of at most
    `WIDTH` characters, else its runs of other characters replaced by ``_``, cut to `WIDTH`, and the first 8 hex digits of its
    SHA-256 appended."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return key if slug == key and len(key) <= WIDTH else f"{slug[:WIDTH]}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def path(directory: Path, key: str, suffix: str) -> Path:
    """Where ``key``'s entry is, or goes, in ``directory``."""
    return Path(directory) / f"{name(key)}{suffix}"


def files(directory: Path, suffix: str) -> list[Path]:
    """The entry files in ``directory``, by name; none when it is absent."""
    return sorted(Path(directory).glob(f"*{suffix}"))


def check(p: Path, key: str) -> None:
    """Refuse entry file ``p`` unless it is ``key``'s (`name`)."""
    if p.name != (want := f"{name(key)}{p.suffix}"):
        raise ValueError(f"{p}: holds the entry {key!r}, whose file is {want}")


__all__ = ["WIDTH", "check", "files", "name", "path"]
