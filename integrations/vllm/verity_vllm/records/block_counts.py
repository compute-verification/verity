"""A day's inference-block counts (`verity-vllm/inference-block-counts/v1`), as `program.predict.block_counts` writes them
from that day's served windows' circuit records: per block class (its session Definition, by SHA-512) the class's shape (its
slots and steps, a block's Boolean gates, its unpriced primitives and its fresh-input bits) and its blocks padded to the
public grid, then the day's total and the largest gates and fresh bits of a class, with the gate table the gates are
counted by. It holds nothing else: no window, run, request, value, placement, time, root or partition row, and `of` refuses
a field no writer writes."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits import partition_v2 as PV2
from verity_vllm.records.fields import RecordError, objects, only, required

SCHEMA = "verity-vllm/inference-block-counts/v1"
FIELDS = ("format", "day", "table", "classes", "total", "c", "k")
CLASS = ("definition", "slots", "steps", "gates", "unpriced", "fresh", "blocks")
DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
DIGEST = re.compile(r"[0-9a-f]{128}")


def _nat(doc: Mapping[str, Any], key: str, where: str, name: str) -> int:
    v = required(doc, key, where, int, name)
    if isinstance(v, bool) or v < 0:
        raise RecordError(where, name, f"is {v!r}, not a natural number")
    return v


def _counts(doc: Any, where: str, name: str) -> dict[str, int]:
    if not isinstance(doc, Mapping) or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in doc.values()):
        raise RecordError(where, name, "is not {primitive: natural}")
    return {str(k): int(v) for k, v in sorted(doc.items())}


@dataclass(frozen=True)
class BlockClass:
    definition: str                                   # the session Definition's SHA-512 (`partition_v2.definition_digest`)
    slots: int                                        # its window class's slots: a window's blocks
    steps: int                                        # and steps
    gates: int                                        # a block's Boolean gates, by the table
    unpriced: Mapping[str, int]                       # a block's activations of each primitive the table has no count for
    fresh: int                                        # a block's fresh-input bits, |in(I) \ θ|
    blocks: int                                       # the day's blocks of the class, padded (`partition_v2.grid`)

    @classmethod
    def of(cls, doc: Any, where: str, at: str) -> BlockClass:
        only(doc, CLASS, where, f"{at}.")
        d = required(doc, "definition", where, str, f"{at}.definition")
        if not DIGEST.fullmatch(d):
            raise RecordError(where, f"{at}.definition", "isn't a SHA-512 in hex")
        n = {k: _nat(doc, k, where, f"{at}.{k}") for k in ("slots", "steps", "gates", "fresh", "blocks")}
        if n["blocks"] != PV2.grid(n["blocks"]):
            raise RecordError(where, f"{at}.blocks", f"is {n['blocks']}, not a count padded to the grid")
        return cls(definition=d, unpriced=_counts(required(doc, "unpriced", where, Mapping, f"{at}.unpriced"), where,
                                                  f"{at}.unpriced"), **n)

    def dump(self) -> dict[str, Any]:
        return {"definition": self.definition, "slots": self.slots, "steps": self.steps, "gates": self.gates,
                "unpriced": dict(self.unpriced), "fresh": self.fresh, "blocks": self.blocks}


@dataclass(frozen=True)
class BlockCounts:
    day: str                                          # the UTC day served, YYYY-MM-DD
    table: Mapping[str, int]                          # the gate table: each primitive's two-input Boolean gates
    classes: tuple[BlockClass, ...]                   # by definition
    total: int                                        # the classes' padded blocks, summed
    c: int                                            # the largest gates of a class with a block
    k: int                                            # the largest fresh bits of a class with a block

    @classmethod
    def of(cls, doc: Any, where: str) -> BlockCounts:
        """The counts as their writer wrote them, refused by field (`fields.RecordError`) where they aren't: another format, a
        field no writer writes, a count that isn't padded, or a total, c or k that isn't the classes'."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "format", f"cannot be read: the counts are a {type(doc).__name__}, not a JSON object")
        if doc.get("format") != SCHEMA:
            raise RecordError(where, "format", f"is {doc.get('format')!r}, not {SCHEMA}")
        only(doc, FIELDS, where)
        day = required(doc, "day", where, str)
        if not DAY.fullmatch(day):
            raise RecordError(where, "day", f"is {day!r}, not YYYY-MM-DD")
        classes = tuple(BlockClass.of(x, where, f"classes[{i}]") for i, x in enumerate(objects(doc, "classes", where)))
        if [x.definition for x in classes] != sorted({x.definition for x in classes}):
            raise RecordError(where, "classes", "aren't one a definition, in its order")
        st = cls(day=day, table=_counts(required(doc, "table", where, Mapping), where, "table"), classes=classes,
                 total=_nat(doc, "total", where, "total"), c=_nat(doc, "c", where, "c"), k=_nat(doc, "k", where, "k"))
        want = summary(classes)
        for f, v in want.items():
            if getattr(st, f) != v:
                raise RecordError(where, f, f"is {getattr(st, f)}, not the classes' {v}")
        return st

    def dump(self) -> dict[str, Any]:
        return {"format": SCHEMA, "day": self.day, "table": dict(self.table), "classes": [x.dump() for x in self.classes],
                "total": self.total, "c": self.c, "k": self.k}


def summary(classes: tuple[BlockClass, ...] | list[BlockClass]) -> dict[str, int]:
    """The day's ``total``, ``c`` and ``k`` from its classes: their padded blocks summed, and the largest gates and fresh
    bits of a class with a block."""
    held = [x for x in classes if x.blocks]
    return {"total": sum(x.blocks for x in classes), "c": max((x.gates for x in held), default=0),
            "k": max((x.fresh for x in held), default=0)}
