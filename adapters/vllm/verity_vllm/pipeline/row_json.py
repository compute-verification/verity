"""A row's JSON documents, as its stages read and write them."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["field_of", "load_json", "read_json", "write_json"]


def read_json(path: str | Path) -> Any:
    """The JSON document at ``path``."""
    with open(path) as f:
        return json.load(f)


def write_json(doc: Any, path: str) -> None:
    """``doc`` as the JSON document at ``path``, the way a stage's summary is written."""
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)


def load_json(path: str | Path) -> Any:
    """The JSON document at ``path``, None when it is absent or unreadable."""
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def field_of(block: Any, key: str) -> Any:
    """``key`` of a document's ``block``; None where the block is absent, null or empty."""
    return block.get(key) if block else None
