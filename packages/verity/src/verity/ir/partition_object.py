"""`verity/partition/v1`: a Program's unit partition as an object the statement and the roots bind by digest.

    {"format": "verity/partition/v1",
     "program": "<SHA-512 of the program>",
     "cuts": {"<SHA-512 of the canonical cut>": {"owner": [<unit within the Call, per computing gate in canonical order>],
                                                 "classes": [<SHA-512 of the unit's circuit, per owner value>]}, ...},
     "calls": [{"call": <the Call activation's index in the canonical layout>, "cut": "<key into cuts>"}, ...]}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon(object))`, canon = `codec.canonical_json` (keys sorted, no whitespace, ASCII).
  A cut's key is `SHA-512(canon(cut))`: every activation of one Definition specialization shares its cut, so the object grows with
  the distinct cuts plus the Calls, not with the gates.
* **Program.** `program_sha512`: SHA-512 over the canonical descriptor `codec.program_digest` hashes (annotations excluded).
* **Calls.** One entry per Call activation (a `call` node of the Program's root body), ascending by its index; the computing gates are
  `cut.CallGraph`'s, so a cut's `owner` indexes them.
* **Units.** Not listed: a unit's global index is derived, by Call in canonical order, then owner value (the order the cut creates
  them), as a prefix sum of each Call's unit count (`len(classes)`); its class is its cut's `classes[owner value]` (`Units`).  The input
  unit is not a unit here: it is always checked, never drawn.
* **Not partition data.** Reads, widths, outputs, input reads, recomputes, gate counts and the committed set are derived: the
  program's properties from the verifier's own copy, and the committed set from them and the owners (`cut.derived_committed`: a value
  is committed exactly when a unit other than its producer's reads it, or it is a Call output).  The object never stores them
  (`validate` rejects any other key); `verify(served=...)` holds what serving commits to the derived set.  The query or cut
  policy that produced the partition is provenance, not partition data; the width rule's parameters come from the verifier's profile.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .codec import canonical_json, encode_program
from .cut import WIDTH_RULE, WIRING, CallGraph, check_cut
from .partition import CutVerdict

FORMAT = "verity/partition/v1"
_KEYS = {"format", "program", "cuts", "calls"}
_CUT_KEYS = {"owner", "classes"}
_CALL_KEYS = {"call", "cut"}
_SHA512 = re.compile(r"^[0-9a-f]{128}$")


def program_sha512(program_or_desc: Any) -> str:
    """The Program's SHA-512: over the canonical JSON of its descriptor (`codec.encode_program`) without annotations."""
    desc = program_or_desc if isinstance(program_or_desc, dict) else encode_program(program_or_desc)
    return hashlib.sha512(canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest()


def canonical(obj: Mapping[str, Any]) -> bytes:
    return canonical_json(obj)


def digest(obj: Mapping[str, Any]) -> str:
    """`SHA-512("verity/partition/v1\\0" || canon(object))`, hex."""
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical(obj)).hexdigest()


def cut_key(cut: Mapping[str, Any]) -> str:
    """A cut's key in `cuts`: `SHA-512(canon(cut))`, hex."""
    return hashlib.sha512(canonical(cut)).hexdigest()


def unit_class(definition_id: str, unit: int) -> str:
    """A unit's class when no backend supplies its circuit's digest: SHA-512 over the Definition's descriptor id and the owner value
    (one class per unit of a specialization's cut: every activation of it has the same units).  A backend names its unit circuit
    file's SHA-512 instead (`build(classes=...)`)."""
    return hashlib.sha512(f"{FORMAT}/unit\0{definition_id}\0{int(unit)}".encode()).hexdigest()


def build(program: Any, cuts: Iterable[tuple[int, Sequence[int]]], classes: Callable[[int, int], str] | None = None) -> dict[str, Any]:
    """The object of `program` from its cuts: (Call index in the canonical layout, owner per computing gate) per Call activation, e.g.
    `word.partition`'s owner.  `classes(call index, owner value)` gives each unit's circuit SHA-512 (default `unit_class` of the Call's
    Definition).  Equal cuts share one entry of `cuts`."""
    from .codec import _spec_id
    nodes = program.fn.body.nodes
    table: dict[str, dict[str, Any]] = {}
    calls = []
    for call, owner in sorted(((int(c), o) for c, o in cuts), key=lambda x: x[0]):
        own = [int(u) for u in owner]
        units = len(set(own))
        fid = _spec_id(nodes[call].fn)
        cut = {"owner": own, "classes": [classes(call, u) if classes else unit_class(fid, u) for u in range(units)]}
        key = cut_key(cut)
        table.setdefault(key, cut)
        calls.append({"call": call, "cut": key})
    return {"format": FORMAT, "program": program_sha512(program), "cuts": table, "calls": calls}


class Units:
    """The object's units in global order (by Call in canonical order, then owner value; the input unit excluded): `population`, and
    `locate(i)` -> (position in `calls`, the Call's index in the canonical layout, owner value, class) by bisection over the prefix
    sums of each Call's unit count, O(log Calls)."""

    def __init__(self, obj: Mapping[str, Any]) -> None:
        self._obj = obj
        self._start: list[int] = []
        t = 0
        for c in obj["calls"]:
            self._start.append(t)
            t += len(obj["cuts"][c["cut"]]["classes"])
        self.population = t

    def locate(self, i: int) -> tuple[int, int, int, str]:
        if not 0 <= i < self.population:
            raise IndexError(f"unit {i} outside 0..{self.population - 1}")
        k = bisect.bisect_right(self._start, i) - 1
        c = self._obj["calls"][k]
        u = i - self._start[k]
        return k, c["call"], u, self._obj["cuts"][c["cut"]]["classes"][u]


def validate(obj: Any) -> CutVerdict:
    """The object's own form, before any program is read: the four keys and no other (derived properties are never stored); a SHA-512
    program; each cut under its own key (`cut_key`), with only an owner (the unit values 0..U-1, each used) and one SHA-512 class per
    unit; Calls ascending and distinct, each naming a cut; every cut named by some Call."""
    bad: dict[str, list] = {}

    def fail(code: str, what: Any) -> None:
        bad.setdefault(code, []).append(what)

    if not isinstance(obj, Mapping):
        return CutVerdict(False, ("not-an-object",), {"not-an-object": {"n": 1, "gates": []}})
    if set(obj) != _KEYS:
        fail("keys", sorted(set(obj) ^ _KEYS))
    if obj.get("format") != FORMAT:
        fail("format", obj.get("format"))
    if not isinstance(obj.get("program"), str) or not _SHA512.match(obj["program"]):
        fail("program-digest", obj.get("program"))
    cuts = obj.get("cuts") if isinstance(obj.get("cuts"), Mapping) else {}
    for key, cut in cuts.items():
        if not isinstance(cut, Mapping) or set(cut) != _CUT_KEYS:
            fail("cut-keys", key)
            continue
        own, cls = cut["owner"], cut["classes"]
        if not isinstance(key, str) or key != cut_key(cut):
            fail("cut-key", key)
        if not isinstance(own, list) or not all(_is_int(u) and u >= 0 for u in own) or set(own) != set(range(len(set(own)))):
            fail("owner-values", key)
            continue
        if not isinstance(cls, list) or len(cls) != len(set(own)) or not all(isinstance(c, str) and _SHA512.match(c) for c in cls):
            fail("classes", key)
    calls = obj.get("calls") if isinstance(obj.get("calls"), list) else []
    prev, named = -1, set()
    for i, c in enumerate(calls):
        if not isinstance(c, Mapping) or set(c) != _CALL_KEYS:
            fail("call-keys", i)
            continue
        if not _is_int(c["call"]) or c["call"] <= prev:
            fail("calls-order", i)
        prev = c["call"] if _is_int(c["call"]) else prev
        if c["cut"] not in cuts:
            fail("call-cut", i)
        named.add(c["cut"])
    if set(cuts) - named:
        fail("cut-unused", sorted(set(cuts) - named)[:8])
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(v), "gates": v[:64]} for k, v in bad.items()})


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def verify(obj: Mapping[str, Any], program: Any, limits: tuple[int, int, int] | None = WIDTH_RULE,
           wiring: frozenset[str] = WIRING, served: Mapping[int, Iterable[int]] | None = None) -> CutVerdict:
    """The object against the program: `validate`, `program` = `program_sha512(program)`, every Call entry a `call` node of the root
    body, and per Call its cut checked on the graph derived here (`cut.CallGraph` of the Call's Definition; `cut.check_cut`: the
    invariant with the derived committed set, and the width rule unless `limits` is None), once per (Definition, cut).  `served`
    (Call index -> the computing gates serving commits for it): held to the derived set (`committed-unread`, `read-uncommitted`,
    `output-uncommitted`).  Failures by Call: {"call", "codes", "detail"}."""
    v = validate(obj)
    if not v.ok:
        return v
    bad: dict[str, list] = {}
    if obj["program"] != program_sha512(program):
        bad["program"] = [obj["program"]]
    nodes = program.fn.body.nodes
    graphs: dict[str, CallGraph] = {}
    checked: dict[tuple[str, str], CutVerdict] = {}
    for c in obj["calls"]:
        k = c["call"]
        if not 0 <= k < len(nodes) or nodes[k].form != "call" or not hasattr(nodes[k].fn, "body"):
            bad.setdefault("call-not-a-call", []).append(k)
            continue
        fn = nodes[k].fn
        g = graphs.get(fn.id)
        if g is None:
            g = graphs[fn.id] = CallGraph(fn, wiring)
        cut = obj["cuts"][c["cut"]]
        if served is not None and k in served:
            r = check_cut(g, cut["owner"], served[k], limits)
        else:
            r = checked.get((fn.id, c["cut"]))
            if r is None:
                r = checked[(fn.id, c["cut"])] = check_cut(g, cut["owner"], None, limits)
        if not r.ok:
            bad.setdefault("cut", []).append({"call": k, "codes": list(r.codes), "detail": r.detail})
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(x), "gates": x[:64]} for k, x in bad.items()})


__all__ = ["FORMAT", "program_sha512", "canonical", "digest", "cut_key", "unit_class", "build", "Units", "validate", "verify"]
