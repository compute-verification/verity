"""A row's JSON documents, as its stages read and write them."""
from __future__ import annotations

import json
from typing import Any

__all__ = ["load_json", "read_json", "write_json"]


def read_json(path: str) -> Any:
    """The JSON document at ``path``."""
    with open(path) as f:
        return json.load(f)


def write_json(doc: Any, path: str) -> None:
    """``doc`` as the JSON document at ``path``, the way a stage's summary is written."""
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)


def load_json(path: str) -> Any:
    """The JSON document at ``path``, None when it is absent or unreadable."""
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None
