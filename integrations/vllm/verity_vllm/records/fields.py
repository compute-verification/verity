"""What every record's `load` shares: the refusal, naming the file and the field, and the two field reads."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RecordError(ValueError):
    """A record refused: `where` is the file it was read from (or what handed it over), `field` the field at fault."""

    def __init__(self, where: str, field: str, why: str):
        super().__init__(f"{where}: field {field!r} {why}")
        self.where, self.field, self.why = where, field, why


class MissingField(RecordError):
    """A required field absent, or null."""

    def __init__(self, where: str, field: str):
        super().__init__(where, field, "is missing: every writer of the record writes it")


def _kinds(kind: type | tuple[type, ...]) -> str:
    return " or ".join(k.__name__ for k in (kind if isinstance(kind, tuple) else (kind,)))


def required(doc: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...]) -> Any:
    """`doc[key]`, which must be present, not null and a `kind`."""
    if doc.get(key) is None:
        raise MissingField(where, key)
    v = doc[key]
    if not isinstance(v, kind):
        raise RecordError(where, key, f"is a {type(v).__name__}, not a {_kinds(kind)}")
    return v


def optional(doc: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...]) -> Any | None:
    """`doc[key]` when its writer wrote it, else None; a present value must be a `kind`."""
    v = doc.get(key)
    if v is not None and not isinstance(v, kind):
        raise RecordError(where, key, f"is a {type(v).__name__}, not a {_kinds(kind)}")
    return v
