"""A day's inference-block judgment (`verity-vllm/inference-block-judgment/v1`), as `program.predict.block_counts` writes it
from that day's served windows' circuit records: whether the day's blocks are at most n (c, k)-inference blocks, against
the day's declaration of c, k and n (by SHA-512); each rule's outcome; per block class (its session Definition, by SHA-512)
its shape (its slots and steps, a block's Boolean gates, its unpriced primitives and its fresh-input bits); and the gate
table the gates are counted by. It holds no count of blocks or windows, padded or exact, and no window, run, request,
value, placement, time, root or partition row; `of` refuses a field no writer writes."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, objects, only, required

SCHEMA = "verity-vllm/inference-block-judgment/v1"
FIELDS = ("format", "day", "declaration", "c", "k", "n", "table", "classes", "checks", "verdict", "reason")
CLASS = ("definition", "slots", "steps", "gates", "unpriced", "fresh")
RULES = ("gates", "fresh", "count")
OUTCOMES = ("accept", "refuse", "undecided")
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
    definition: str                                   # the session Definition's SHA-512 (`partition_v2.id_digest`)
    slots: int                                        # its window class's slots: a window's blocks
    steps: int                                        # and steps
    gates: int                                        # a block's Boolean gates, by the table
    unpriced: Mapping[str, int]                       # a block's activations of each primitive the table has no count for
    fresh: int                                        # a block's fresh-input bits, |in(I) \ θ|

    @classmethod
    def of(cls, doc: Any, where: str, at: str) -> BlockClass:
        only(doc, CLASS, where, f"{at}.")
        d = required(doc, "definition", where, str, f"{at}.definition")
        if not DIGEST.fullmatch(d):
            raise RecordError(where, f"{at}.definition", "isn't a SHA-512 in hex")
        n = {k: _nat(doc, k, where, f"{at}.{k}") for k in ("slots", "steps", "gates", "fresh")}
        return cls(definition=d, unpriced=_counts(required(doc, "unpriced", where, Mapping, f"{at}.unpriced"), where,
                                                  f"{at}.unpriced"), **n)

    def dump(self) -> dict[str, Any]:
        return {"definition": self.definition, "slots": self.slots, "steps": self.steps, "gates": self.gates,
                "unpriced": dict(self.unpriced), "fresh": self.fresh}


def shape_checks(classes: tuple[BlockClass, ...] | list[BlockClass], c: int, k: int) -> dict[str, str]:
    """The gates and fresh rules' outcomes on ``classes`` (every one holds a block): a block over c gates or k fresh bits
    refuses; else a block whose gates are a lower bound (an unpriced primitive) leaves the gates undecided."""
    gates = ("refuse" if any(x.gates > c for x in classes) else
             "undecided" if any(x.unpriced for x in classes) else "accept")
    return {"gates": gates, "fresh": "refuse" if any(x.fresh > k for x in classes) else "accept"}


@dataclass(frozen=True)
class BlockJudgment:
    day: str                                          # the UTC day served, YYYY-MM-DD
    declaration: str                                  # the declaration's SHA-512 (its JSON, sorted keys, compact)
    c: int                                            # the declared gates a block
    k: int                                            # the declared fresh bits a block
    n: int                                            # the declared blocks the day
    table: Mapping[str, int]                          # the gate table: each primitive's two-input Boolean gates
    classes: tuple[BlockClass, ...]                   # each class with a block, by definition
    checks: Mapping[str, str]                         # each rule of `RULES`: accept, refuse or undecided
    verdict: str                                      # accept when every rule accepts, else refuse
    reason: str | None                                # the first rule that doesn't accept, naming c, k or n, never a count

    @classmethod
    def of(cls, doc: Any, where: str) -> BlockJudgment:
        """The judgment as its writer wrote it, refused by field (`fields.RecordError`) where it isn't: another format, a
        field no writer writes, a gates or fresh outcome that isn't the classes', or a verdict that isn't the checks'."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "format", f"cannot be read: the judgment is a {type(doc).__name__}, not a JSON object")
        if doc.get("format") != SCHEMA:
            raise RecordError(where, "format", f"is {doc.get('format')!r}, not {SCHEMA}")
        only(doc, FIELDS, where)
        day = required(doc, "day", where, str)
        if not DAY.fullmatch(day):
            raise RecordError(where, "day", f"is {day!r}, not YYYY-MM-DD")
        dec = required(doc, "declaration", where, str)
        if not DIGEST.fullmatch(dec):
            raise RecordError(where, "declaration", "isn't a SHA-512 in hex")
        classes = tuple(BlockClass.of(x, where, f"classes[{i}]") for i, x in enumerate(objects(doc, "classes", where)))
        if [x.definition for x in classes] != sorted({x.definition for x in classes}):
            raise RecordError(where, "classes", "aren't one a definition, in its order")
        checks = required(doc, "checks", where, Mapping)
        if set(checks) != set(RULES) or not all(v in OUTCOMES for v in checks.values()):
            raise RecordError(where, "checks", f"aren't one of {OUTCOMES} for each of {RULES}")
        c, k, n = (_nat(doc, f, where, f) for f in ("c", "k", "n"))
        for rule, got in shape_checks(classes, c, k).items():
            if checks[rule] != got:
                raise RecordError(where, f"checks.{rule}", f"is {checks[rule]}, not the classes' {got}")
        verdict = required(doc, "verdict", where, str)
        if verdict != ("accept" if all(checks[r] == "accept" for r in RULES) else "refuse"):
            raise RecordError(where, "verdict", f"is {verdict!r}, not the checks'")
        reason = doc.get("reason")
        if (reason is None) != (verdict == "accept") or not (reason is None or isinstance(reason, str)):
            raise RecordError(where, "reason", "isn't null on an accept and a sentence on a refusal")
        return cls(day=day, declaration=dec, c=c, k=k, n=n, classes=classes, verdict=verdict, reason=reason,
                   table=_counts(required(doc, "table", where, Mapping), where, "table"),
                   checks={r: checks[r] for r in RULES})

    def dump(self) -> dict[str, Any]:
        return {"format": SCHEMA, "day": self.day, "declaration": self.declaration, "c": self.c, "k": self.k, "n": self.n,
                "table": dict(self.table), "classes": [x.dump() for x in self.classes], "checks": dict(self.checks),
                "verdict": self.verdict, "reason": self.reason}
