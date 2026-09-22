"""Canonical JSON (README §1): the bytes every id is computed over.

sort_keys, no whitespace, UTF-8, floats by repr, NaN/Inf rejected, -0.0 rejected (write 0.0), ints unbounded, keys must be str.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _check(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"canon: key {k!r} at {path} is {type(k).__name__}, keys must be str")
            _check(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError(f"canon: non-finite float at {path}")
        if obj == 0.0 and str(obj).startswith("-"):
            raise ValueError(f"canon: -0.0 at {path}; write 0.0")
    elif obj is not None and not isinstance(obj, (str, bool, int)):
        raise TypeError(f"canon: {type(obj).__name__} at {path} is not JSON")


def dumps(obj: Any) -> bytes:
    _check(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(obj: Any) -> str:
    return hashlib.sha256(dumps(obj)).hexdigest()
