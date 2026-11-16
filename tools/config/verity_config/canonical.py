"""Canonical JSON and identities for resolved Verity configuration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


class CanonicalError(ValueError):
    """A value cannot be represented by the supported canonical JSON subset."""


def value(obj: Any, *, identity: bool = False) -> Any:
    """Return JSON data with deterministic containers and dataclass handling.

    Fields marked ``metadata={"identity": False}`` are omitted only from an
    identity. The supported scalar types are exactly null, booleans, integers,
    finite binary64 values and strings.
    """

    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            field.name: value(getattr(obj, field.name), identity=identity)
            for field in fields(obj)
            if not identity or field.metadata.get("identity", True)
        }
    if isinstance(obj, Mapping):
        if any(not isinstance(key, str) for key in obj):
            raise CanonicalError("canonical JSON object keys must be strings")
        return {key: value(child, identity=identity) for key, child in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [value(child, identity=identity) for child in obj]
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise CanonicalError("canonical JSON numbers must be finite")
        return 0 if obj == 0 else obj
    if isinstance(obj, Path):
        return obj.as_posix()
    raise CanonicalError(f"canonical JSON does not support {type(obj).__name__}")


def dumps(obj: Any, *, identity: bool = False) -> bytes:
    """RFC 8785-style UTF-8 JSON for the configuration types Verity uses."""

    return json.dumps(
        value(obj, identity=identity),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256(obj: Any, *, identity: bool = False) -> str:
    return hashlib.sha256(dumps(obj, identity=identity)).hexdigest()
