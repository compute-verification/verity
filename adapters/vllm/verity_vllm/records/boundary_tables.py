"""The two messages between a window's boundary commitment and rank 0 (`pipeline.boundary_tables`), private to the node:
a class's request for its spans' history tables (`TableAssignment`, ``verity-vllm/boundary-tables/assign/v1``), which names
the engine's ids, and rank 0's answer (`TableRoots`, ``verity-vllm/boundary-tables/roots/v1``), each table's entry or why it
built none. Each `of` refuses another format, a field no writer writes, and a field missing or of another kind."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import (
    RecordError,
    objects,
    only,
    optional,
    present,
    required,
)

__all__ = ["ASSIGN", "ROOTS", "TableAssignment", "TableRoots"]

ASSIGN = "verity-vllm/boundary-tables/assign/v1"
ROOTS = "verity-vllm/boundary-tables/roots/v1"
#: an assigned span's fields, in slot order: its engine id, prompt and served lengths, and finish (null while running)
SPAN = ("engine_id", "prompt", "served", "finish")


def _format(doc: Any, schema: str, where: str) -> None:
    if not isinstance(doc, Mapping):
        raise RecordError(where, "format", f"cannot be read: the record is a {type(doc).__name__}, not a JSON object")
    if doc.get("format") != schema:
        raise RecordError(where, "format", f"is {doc.get('format')!r}, not {schema}")


@dataclass(frozen=True)
class TableAssignment:
    window: int                                        # the window's index in its serve
    run: str                                           # its run, which every table's name binds
    program: str                                       # the class's registration's program digest, which each domain binds
    prompt: int                                        # the class's prompt cap P
    outputs: int                                       # its output cap O
    layers: tuple[int, ...]                            # the tables' layers
    capacity: int                                      # each table's leaves
    spans: tuple[tuple[str, int, int, str | None], ...]  # each slot's span (`SPAN`)

    @classmethod
    def of(cls, doc: Any, where: str) -> TableAssignment:
        _format(doc, ASSIGN, where)
        only(doc, ("format", *ASSIGN_KINDS, "spans"), where)
        got = {k: required(doc, k, where, kind) for k, kind in ASSIGN_KINDS.items()}
        spans = []
        for i, s in enumerate(objects(doc, "spans", where)):
            at = f"spans[{i}]"
            only(s, SPAN, where, f"{at}.")
            spans.append((required(s, "engine_id", where, str, f"{at}.engine_id"),
                          required(s, "prompt", where, int, f"{at}.prompt"), required(s, "served", where, int, f"{at}.served"),
                          present(s, "finish", where, str, f"{at}.finish")))
        for i, x in enumerate(got["layers"]):
            if type(x) is not int:
                raise RecordError(where, f"layers[{i}]", f"is a {type(x).__name__}, not an int")
        return cls(**{**got, "layers": tuple(got["layers"]), "spans": tuple(spans)})

    def dump(self) -> dict[str, Any]:
        return {"format": ASSIGN, **{k: getattr(self, k) for k in ASSIGN_KINDS}, "layers": list(self.layers),
                "spans": [dict(zip(SPAN, s, strict=True)) for s in self.spans]}


ASSIGN_KINDS: dict[str, type | tuple[type, ...]] = {"window": int, "run": str, "program": str, "prompt": int, "outputs": int,
                                                    "layers": list, "capacity": int}


@dataclass(frozen=True)
class TableRoots:
    window: int                                        # the assignment's window, run and class
    run: str
    prompt: int
    outputs: int
    tables: tuple[Mapping[str, Any], ...] | None       # each table's entry (`history.HistoryTable.entry`), absent on error
    error: str | None                                  # why rank 0 built none, absent when it built them

    @classmethod
    def of(cls, doc: Any, where: str) -> TableRoots:
        _format(doc, ROOTS, where)
        only(doc, ("format", *ROOTS_KINDS, "tables", "error"), where)
        tables, error = optional(doc, "tables", where, list), optional(doc, "error", where, str)
        if (tables is None) == (error is None):
            raise RecordError(where, "tables", "and error: rank 0 answers exactly one of them")
        if tables is not None:
            tables = tuple(objects(doc, "tables", where))
        return cls(**{k: required(doc, k, where, kind) for k, kind in ROOTS_KINDS.items()}, tables=tables, error=error)

    def dump(self) -> dict[str, Any]:
        out = {"format": ROOTS, **{k: v for k, v in dataclasses.asdict(self).items() if v is not None}}
        if self.tables is not None:
            out["tables"] = [dict(e) for e in self.tables]
        return out


ROOTS_KINDS: dict[str, type | tuple[type, ...]] = {"window": int, "run": str, "prompt": int, "outputs": int}
