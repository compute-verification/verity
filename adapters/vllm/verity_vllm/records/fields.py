"""What every record's `load` shares: the refusal, naming the file and the field, and the field reads.  A nested field is named
by its dotted path in the record (`name`, default the key)."""
from __future__ import annotations

from collections.abc import Collection, Mapping
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


class UnknownField(RecordError):
    """A field no writer of the record writes (a newer writer than this reader, or a hand edit)."""

    def __init__(self, where: str, field: str):
        super().__init__(where, field, "is not a field of the record: no writer this reader knows writes it")


def only(doc: Mapping[str, Any], keys: Collection[str], where: str, prefix: str = "") -> None:
    """Refuse a field of `doc` not in `keys` (the first, sorted), named `prefix` + its key."""
    extra = sorted(set(doc) - set(keys))
    if extra:
        raise UnknownField(where, prefix + str(extra[0]))


def _kinds(kind: type | tuple[type, ...]) -> str:
    return " or ".join(k.__name__ for k in (kind if isinstance(kind, tuple) else (kind,)))


def _typed(v: Any, where: str, name: str, kind: type | tuple[type, ...]) -> Any:
    if not isinstance(v, kind):
        raise RecordError(where, name, f"is a {type(v).__name__}, not a {_kinds(kind)}")
    return v


def required(doc: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...], name: str | None = None) -> Any:
    """`doc[key]`, which must be present, not null and a `kind`."""
    if doc.get(key) is None:
        raise MissingField(where, name or key)
    return _typed(doc[key], where, name or key, kind)


def present(doc: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...], name: str | None = None) -> Any | None:
    """`doc[key]`, which must be present, and a `kind` or null: every writer writes it, null where it has nothing to state."""
    if key not in doc:
        raise MissingField(where, name or key)
    return None if doc[key] is None else _typed(doc[key], where, name or key, kind)


def optional(doc: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...], name: str | None = None) -> Any | None:
    """`doc[key]` when its writer wrote it, else None; a present value must be a `kind`."""
    v = doc.get(key)
    return None if v is None else _typed(v, where, name or key, kind)


def objects(doc: Mapping[str, Any], key: str, where: str, name: str | None = None) -> list | tuple:
    """`doc[key]`, a required list of JSON objects."""
    v = required(doc, key, where, (list, tuple), name)
    for i, x in enumerate(v):
        _typed(x, where, f"{name or key}[{i}]", Mapping)
    return v
