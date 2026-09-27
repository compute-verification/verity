"""`verity/partition/v1`: a Program's unit partition as an object the statement and the roots bind by digest.

    {"format": "verity/partition/v1",
     "program": "<SHA-512 of the program>",
     "calls": [{"call": <the Call activation's index in the canonical layout>,
                "owner": [<unit within the Call, per computing gate in canonical order>],
                "committed": [<computing gates committed at serving, ascending>]}, ...],
     "units": [{"call": <index into calls>, "unit": <owner value>, "class": "<SHA-512 of the unit's circuit>"}, ...]}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon(object))`, canon = `codec.canonical_json` (keys sorted, no whitespace, ASCII).
* **Program.** `program_sha512`: SHA-512 over the canonical descriptor `codec.program_digest` hashes (annotations excluded).
* **Calls.** One entry per Call activation (a `call` node of the Program's root body), ascending by its index; the computing gates are
  `cut.CallGraph`'s, so `owner` and `committed` index them.
* **Units.** A unit's global index is its position in `units`: by Call in canonical layout order, then owner value (the order the cut
  creates them).  The input unit is not a unit here: it is always checked, never drawn.
* **Not partition data.** Reads, widths, outputs, input reads, recomputes and gate counts are properties of the program: a verifier
  derives them from its own copy (`verify`), and the object never stores them (`validate` rejects any other key).  The query or cut
  policy that produced the partition is provenance, not partition data; the width rule's parameters come from the verifier's profile.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .codec import canonical_json, encode_program
from .cut import WIDTH_RULE, WIRING, CallGraph, check_cut
from .partition import CutVerdict

FORMAT = "verity/partition/v1"
_KEYS = {"format", "program", "calls", "units"}
_CALL_KEYS = {"call", "owner", "committed"}
_UNIT_KEYS = {"call", "unit", "class"}
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


def unit_class(definition_id: str, unit: int) -> str:
    """A unit's class when no backend supplies its circuit's digest: SHA-512 over the Definition's descriptor id and the owner value
    (one class per unit of a specialization's cut: every activation of it has the same units).  A backend names its unit circuit
    file's SHA-512 instead (`build(classes=...)`)."""
    return hashlib.sha512(f"{FORMAT}/unit\0{definition_id}\0{int(unit)}".encode()).hexdigest()


def build(program: Any, cuts: Iterable[tuple[int, Sequence[int], Iterable[int]]],
          classes: Callable[[int, int], str] | None = None) -> dict[str, Any]:
    """The object of `program` from its cuts: (Call index in the canonical layout, owner per computing gate, committed computing
    gates) per Call activation, e.g. `word.partition`'s owner and its heads (`head >= 0`).  `classes(call index, owner value)` gives
    each unit's circuit SHA-512 (default `unit_class` of the Call's Definition)."""
    from .codec import _spec_id
    nodes = program.fn.body.nodes
    calls, units = [], []
    for call, owner, committed in sorted(((int(c), o, m) for c, o, m in cuts), key=lambda x: x[0]):
        own = [int(u) for u in owner]
        calls.append({"call": call, "owner": own, "committed": sorted({int(g) for g in committed})})
        fid = _spec_id(nodes[call].fn)
        for u in sorted(set(own)):
            units.append({"call": len(calls) - 1, "unit": u, "class": classes(call, u) if classes else unit_class(fid, u)})
    return {"format": FORMAT, "program": program_sha512(program), "calls": calls, "units": units}


def validate(obj: Any) -> CutVerdict:
    """The object's own form, before any program is read: the four keys and no other (derived properties are never stored); a
    SHA-512 program; Calls ascending and distinct; each owner a list of unit values 0..U-1 each used; committed ascending, distinct,
    within the owner; `units` exactly the (Call, owner value) pairs in canonical order, each with a SHA-512 class."""
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
    calls = obj.get("calls") if isinstance(obj.get("calls"), list) else []
    want: list[tuple[int, int]] = []
    prev = -1
    for i, c in enumerate(calls):
        if not isinstance(c, Mapping) or set(c) != _CALL_KEYS:
            fail("call-keys", i)
            continue
        if not _is_int(c["call"]) or c["call"] <= prev:
            fail("calls-order", i)
        prev = c["call"] if _is_int(c["call"]) else prev
        own, com = c["owner"], c["committed"]
        if not isinstance(own, list) or not all(_is_int(u) and u >= 0 for u in own):
            fail("owner-values", i)
            continue
        used = set(own)
        if used != set(range(len(used))):
            fail("owner-values", i)
        if not isinstance(com, list) or not all(_is_int(g) for g in com) or com != sorted(set(com)) or \
                any(not 0 <= g < len(own) for g in com):
            fail("committed", i)
        want += [(i, u) for u in sorted(used)]
    units = obj.get("units") if isinstance(obj.get("units"), list) else []
    got = []
    for j, u in enumerate(units):
        if not isinstance(u, Mapping) or set(u) != _UNIT_KEYS or not _is_int(u["call"]) or not _is_int(u["unit"]):
            fail("unit-keys", j)
            continue
        got.append((u["call"], u["unit"]))
        if not isinstance(u["class"], str) or not _SHA512.match(u["class"]):
            fail("unit-class", j)
    if got != want:
        fail("units", {"n_units": len(got), "n_expected": len(want),
                       "first_difference": next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))})
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(v), "gates": v[:64]} for k, v in bad.items()})


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def verify(obj: Mapping[str, Any], program: Any, limits: tuple[int, int, int] | None = WIDTH_RULE,
           wiring: frozenset[str] = WIRING) -> CutVerdict:
    """The object against the program: `validate`, `program` = `program_sha512(program)`, every Call entry a `call` node of the root
    body, and per Call its cut checked on the graph derived here (`cut.CallGraph` of the Call's Definition; `cut.check_cut`: the
    invariant, and the width rule unless `limits` is None).  Failures by Call: {"call": index, "codes": [...], "detail": {...}}."""
    v = validate(obj)
    if not v.ok:
        return v
    bad: dict[str, list] = {}
    if obj["program"] != program_sha512(program):
        bad["program"] = [obj["program"]]
    nodes = program.fn.body.nodes
    graphs: dict[str, CallGraph] = {}
    for c in obj["calls"]:
        k = c["call"]
        if not 0 <= k < len(nodes) or nodes[k].form != "call" or not hasattr(nodes[k].fn, "body"):
            bad.setdefault("call-not-a-call", []).append(k)
            continue
        fn = nodes[k].fn
        g = graphs.get(fn.id)
        if g is None:
            g = graphs[fn.id] = CallGraph(fn, wiring)
        r = check_cut(g, c["owner"], c["committed"], limits)
        if not r.ok:
            bad.setdefault("cut", []).append({"call": k, "codes": list(r.codes), "detail": r.detail})
    codes = tuple(bad)
    return CutVerdict(not codes, codes, {k: {"n": len(x), "gates": x[:64]} for k, x in bad.items()})


__all__ = ["FORMAT", "program_sha512", "canonical", "digest", "unit_class", "build", "validate", "verify"]
