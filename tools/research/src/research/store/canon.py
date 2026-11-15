"""Canonical JSON (README §1): the one serialisation every id is computed over.

`dumps(obj)` = `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` as UTF-8, after
rejecting what would make two equal documents serialise differently or two different documents serialise the same: NaN/Inf,
`-0.0` (use `0.0`), non-`str` keys (json would coerce `1` and `"1"` onto the same key).  Floats render via Python `repr`
(shortest round-trip), ints are unbounded, `bool` stays `true`/`false`.  Tuples serialise as lists.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

SCALARS = (str, int, float, bool, type(None))


def check(obj: Any, path: str = "$") -> None:
    """Raise ValueError naming the offending path if `obj` is not canonicalisable."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"{path}: {obj!r} is not representable in canonical JSON")
        if obj == 0.0 and math.copysign(1.0, obj) < 0:
            raise ValueError(f"{path}: -0.0 is rejected; use 0.0")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"{path}: key {k!r} is not a str")
            check(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check(v, f"{path}[{i}]")
        return
    raise ValueError(f"{path}: {type(obj).__name__} is not a JSON type")


def dumps(obj: Any) -> bytes:
    check(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def loads(data: bytes | str) -> Any:
    return json.loads(data)


def sha256(obj: Any) -> str:
    """Hex sha256 of the canonical bytes."""
    return hashlib.sha256(dumps(obj)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
