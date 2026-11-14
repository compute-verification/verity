"""Round 2's reader over the joined record and its ranks' check, as the verifier's `Spot.SecondWindow` and `Spot.Ranks`
define them: a two-stage run's second-stage sessions held to the joined circuit's record (its ``ranks``, ``wires`` and
``boundary_reads``, `verity.commitments.windows.interface`) and to each rank's docket, with no interior
docket. The rule, the codes and why the interior docket's checks go are the Lean's (its module docstrings).

A rank's docket as round 2 reads it is ``{"rank", "digest", "circuit", "run", "members"}``, each member
``{"key", "base", "units", "outs"}`` with ``outs`` its output ports' leaves, ``[[port, leaves], ...]``. What the Lean has
not built yet refuses here too, with its code and `NOT_BUILT`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from verity.protocols.proofs.spot.verifier import second_stage as SS

#: how a rank's registered names are its own (`Spot.Ranks.PREFIX`)
PREFIX = "r<rank>/"
#: the detail of a check whose body is not built: it refuses (`Spot.Ranks.NOT_BUILT`)
NOT_BUILT = "not yet built: refused"
#: the ranks' checks' codes, in the order they run (`Spot.Ranks.CODES`)
RANK_CODES = ("ranks-shape", "ranks-window", "ranks-run", "ranks-members", "ranks-units", "ranks-wires", "ranks-reads")
#: `Spot.SecondStage`'s per-session codes this reader reuses, read against a rank member (`Spot.SecondWindow.SESSION_CODES`)
SESSION_CODES = ("second-header", "second-population", "second-roots", "second-sample", "second-read", "second-values",
                 "second-refs", "second-tables", "second-verified", "second-table-rows")
WINDOW_WHY = "a served rank record's digest is not its rank's window, or its registration binds another digest or other bytes"
RUN_WHY = "the ranks' registrations are not one serve's, one a rank at <root>/rank<r>, each in the log before the challenge"
MEMBERS_WHY = ("a rank's members of a template don't tile its units in key order, name no template of the rank's, or "
               "disagree on an output port's leaves a unit")
UNITS_WHY = "a template's units over the ranks are not the joined record's count of it"
SHAPE_WHY = ("the ranks are not {of, prefix}, the prefix r<rank>/, at least one rank, each {window, units, wires}: 128 "
             "lowercase hex digits, naturals by template, a natural")
WIRES_WHY = ("the joined record's first wires are not each rank's own wires moved to its units, a rank's count of them, in "
             "rank order, or a wire after them has its reader units in no one rank or all its writers in its reader's rank")
READS_WHY = ("the joined record's registered declarations are not its ranks', each named r<r>/... and at its rank's units, "
             "in rank order, or a rank's record has values or histories")
WIRE_TREE_WHY = "a sampled reader member reads through one wire writer rows that lie in more than one writer member, or in none"
PARTS_WHY = "a sampled unit is in no rank member"
NO_REGISTERED = {"openings": [], "params": {}, "reads": [], "roots": [], "tables": []}
#: the `two-stage-form` refusal's detail (`Spot.SecondStage.FORM_WHY`)
FORM_WHY = ("refused: a two-stage run, its second-stage sessions over an interior registration: a statement form the "
            "soundness proof does not cover")
_HEX512 = re.compile(r"[0-9a-f]{128}")
_DIGITS = re.compile(r"[0-9]+")


def _nat(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def ranks_of(field: Any) -> list[tuple[str, dict[str, int], int]] | None:
    """`Spot.Ranks.Ranks.ofJson?`: the ranks ``(window, units, wires)`` of the record's ``ranks`` field, None for one
    ``ranks-shape`` refuses."""
    if not isinstance(field, Mapping) or sorted(field) != ["of", "prefix"] or field["prefix"] != PREFIX:
        return None
    of = field["of"]
    if not isinstance(of, list) or not of:
        return None
    out = []
    for r in of:
        if (not isinstance(r, Mapping) or sorted(r) != ["units", "window", "wires"] or not isinstance(r["window"], str)
                or not _HEX512.fullmatch(r["window"]) or not isinstance(r["units"], Mapping)
                or not all(_nat(n) for n in r["units"].values()) or not _nat(r["wires"])):
            return None
        out.append((r["window"], dict(r["units"]), r["wires"]))
    return out


def segment(ranks: Sequence[tuple[str, Mapping[str, int], int]], template: str, r: int) -> tuple[int, int]:
    """`Spot.Ranks.Ranks.segment`: rank ``r``'s units of ``template`` in the joined circuit, ``[lo, hi)``."""
    lo = sum(q[1].get(template, 0) for q in ranks[:r])
    return lo, lo + (ranks[r][1].get(template, 0) if r < len(ranks) else 0)


def index_of(template: str, key: str) -> int | None:
    """`Spot.Ranks.indexOf`: ``j`` where ``key`` is ``<template>/c<j>``."""
    rest = key[len(template) + 2:] if key.startswith(template + "/c") else ""
    return int(rest) if _DIGITS.fullmatch(rest) else None


def members_of(reg: Mapping[str, Any], template: str) -> list[tuple[int, Mapping[str, Any]]]:
    """`Spot.Ranks.membersOf`: ``reg``'s members of ``template`` with their indices, in index order."""
    ms = [(j, h) for h in reg["members"] if (j := index_of(template, h["key"])) is not None]
    return sorted(ms, key=lambda x: x[0])


def _tiles(ms: Sequence[tuple[int, Mapping[str, Any]]], n: int) -> bool:
    nxt = pos = 0
    for j, h in ms:
        if j != nxt or h["base"] != pos or h["units"] < 1:
            return False
        nxt, pos = nxt + 1, pos + h["units"]
    return pos == n


def _leaves_agree(ms: Sequence[tuple[int, Mapping[str, Any]]]) -> bool:
    if not ms:
        return True
    h0 = ms[0][1]
    return all([p for p, _ in h["outs"]] == [p for p, _ in h0["outs"]]
               and all(a[1] * h0["units"] == b[1] * h["units"] for a, b in zip(h["outs"], h0["outs"], strict=False))
               for _j, h in ms)


def _rank_of(ranks: Sequence[tuple[str, Mapping[str, int], int]], template: str, lo: int, hi: int) -> int | None:
    return next((r for r in range(len(ranks)) if segment(ranks, template, r)[0] <= lo <= hi < segment(ranks, template, r)[1]),
                None)


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(b, sort_keys=True, separators=(",", ":"))


def _moved(j: Mapping[str, Any], key: str, by: int) -> dict[str, Any]:
    return {**j, key: j[key] + by} if _nat(j.get(key)) or (key == "base" and isinstance(j.get(key), int)) else dict(j)


def _named(j: Mapping[str, Any], key: str, prefix: str) -> dict[str, Any]:
    return {**j, key: prefix + j[key]} if isinstance(j.get(key), str) else dict(j)


def shifted_wires(ranks: Sequence[tuple[str, Mapping[str, int], int]], r: int, doc: Mapping[str, Any]) -> list[dict]:
    """`Spot.Ranks.shiftedWires`: rank ``r``'s wires of its record ``doc`` moved to its units in the joined circuit."""
    return [_moved(_moved(w, "at", segment(ranks, w.get("reader", ""), r)[0]), "base",
                   segment(ranks, w.get("writer", ""), r)[0]) for w in doc.get("wires", [])]


def _join_wire_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], w: Mapping[str, Any]) -> bool:
    try:
        radix, step, (_c0, cuts) = list(w["radix"]), list(w["step"]), w.get("cut", (0, 1))
        units = math.prod(radix) // cuts
        lo = w["base"] + sum(min(0, (k - 1) * d) for k, d in zip(radix, step, strict=True))
        hi = w["base"] + sum(max(0, (k - 1) * d) for k, d in zip(radix, step, strict=True))
        r = _rank_of(ranks, w["reader"], w["at"], w["at"] + units - 1)
    except (KeyError, TypeError, ValueError):
        return False
    return r is not None and units >= 1 and not (lo >= 0 and _rank_of(ranks, w["writer"], lo, hi) == r)


def wires_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], record: Mapping[str, Any],
             records: Sequence[Mapping[str, Any]]) -> bool:
    """`Spot.Ranks.wiresOk`: ``ranks-wires``."""
    joined = record.get("wires", [])
    own = [w for r in range(len(ranks)) for w in shifted_wires(ranks, r, records[r] if r < len(records) else {})]
    return (len(records) == len(ranks) and all(len(records[r].get("wires", [])) == ranks[r][2] for r in range(len(ranks)))
            and len(own) <= len(joined) and all(_same(a, b) for a, b in zip(own, joined, strict=False))
            and all(_join_wire_ok(ranks, w) for w in joined[len(own):]))


def joined_registered(ranks: Sequence[tuple[str, Mapping[str, int], int]],
                      records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """`Spot.Ranks.joinedRegistered`: the joined record's ``registered`` as its ranks' records make it."""
    regs = [(r, (records[r] if r < len(records) else {}).get("registered") or NO_REGISTERED) for r in range(len(ranks))]
    if any(g.get("values") is not None or g.get("histories") is not None for _r, g in regs):
        return None

    def moved(r: int, j: Mapping[str, Any]) -> dict[str, Any]:
        return _moved(j, "at", segment(ranks, j.get("reader", ""), r)[0])
    out = {"openings": [moved(r, _named(o, "table", f"r{r}/")) for r, g in regs for o in g.get("openings", [])],
           "params": {f"r{r}/{k}": v for r, g in regs for k, v in g.get("params", {}).items()},
           "reads": [moved(r, _named(x, "param", f"r{r}/")) for r, g in regs for x in g.get("reads", [])],
           "roots": [moved(r, _named(x, "table", f"r{r}/")) for r, g in regs for x in g.get("roots", [])],
           "tables": [_named(_named(t, "name", f"r{r}/"), "param", f"r{r}/") for r, g in regs for t in g.get("tables", [])]}
    formats = {f"r{r}/{k}": v for r, g in regs for k, v in g.get("formats", {}).items()}
    return {**out, **({"formats": formats} if formats else {})}


def reads_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], record: Mapping[str, Any],
             records: Sequence[Mapping[str, Any]]) -> bool:
    """`Spot.Ranks.readsOk`: ``ranks-reads``."""
    g = joined_registered(ranks, records)
    return len(records) == len(ranks) and g is not None and _same(g, record.get("registered") or NO_REGISTERED)


def _reg(registrations: Sequence[Mapping[str, Any]], r: int) -> Mapping[str, Any] | None:
    return next((g for g in registrations if g["rank"] == r), None)


def member_of(ranks: Sequence[tuple[str, Mapping[str, int], int]], registrations: Sequence[Mapping[str, Any]],
              template: str, unit: int) -> tuple[int, str, int, int, int] | None:
    """`Spot.Ranks.memberOf`: the rank member ``(rank, template, j, base, units)`` of ``registrations`` that holds unit
    ``unit`` of ``template`` in the joined circuit."""
    r = next((r for r in range(len(ranks)) if segment(ranks, template, r)[0] <= unit < segment(ranks, template, r)[1]),
             None)
    reg = _reg(registrations, r) if r is not None else None
    if reg is None:
        return None
    v = unit - segment(ranks, template, r)[0]
    x = next(((j, h) for j, h in members_of(reg, template) if h["base"] <= v < h["base"] + h["units"]), None)
    return (r, template, x[0], x[1]["base"], x[1]["units"]) if x is not None else None


def rank_failures(record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], served: Sequence[tuple[str, str]],
                  registrations: Sequence[Mapping[str, Any]], root: str,
                  logged: Callable[[str], bool]) -> list[tuple[str, Any]]:
    """`Spot.Ranks.failures`: ``ranks-shape`` first, then ``ranks-window``, ``ranks-run``, ``ranks-members``,
    ``ranks-units``, ``ranks-wires`` and ``ranks-reads`` of the joined ``record`` against its ranks' records, served
    records and dockets at the serve's ``root``."""
    ranks = ranks_of(record.get("ranks"))
    if ranks is None:
        return [("ranks-shape", SHAPE_WHY)]
    n, out = len(ranks), []
    if not (len(served) == n and all((g := _reg(registrations, r)) is not None and served[r][0] == ranks[r][0]
                                     and g["digest"] == served[r][0] and g["circuit"] == served[r][1] for r in range(n))):
        out.append(("ranks-window", WINDOW_WHY))
    if not (len(registrations) == n and all(
            len(gs := [g for g in registrations if g["rank"] == r]) == 1 and gs[0]["run"] == f"{root}/rank{r}"
            and logged(gs[0]["circuit"]) for r in range(n))):
        out.append(("ranks-run", RUN_WHY))
    every = [list(ranks[r][1]) for r in range(n)]
    if not all((g := _reg(registrations, r)) is not None
               and all(any(index_of(t, h["key"]) is not None for t in every[r]) for h in g["members"])
               and all(_tiles(members_of(g, t), k)
                       and _leaves_agree([x for q in registrations for x in members_of(q, t)])
                       for t, k in ranks[r][1].items()) for r in range(n)):
        out.append(("ranks-members", MEMBERS_WHY))
    ts = record.get("templates")
    named = ([(x[0], x[1]) for x in ts if isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and _nat(x[1])]
             if isinstance(ts, list) else None)
    if (named is None or len(named) != len(ts) or not all(t in dict(named) for q in ranks for t in q[1])
            or not all(sum(q[1].get(t, 0) for q in ranks) == k for t, k in named)):
        out.append(("ranks-units", UNITS_WHY))
    if not wires_ok(ranks, record, records):
        out.append(("ranks-wires", WIRES_WHY))
    if not reads_ok(ranks, record, records):
        out.append(("ranks-reads", READS_WHY))
    return out


_WIRE_KEYS = {"reader", "port", "writer", "out", "at", "radix", "base", "step"}


def _pattern(d: Any) -> dict[str, Any] | None:
    """`Spot.Wires.Wire.ofJson`: a wire's record, or None."""
    if not isinstance(d, Mapping) or not _WIRE_KEYS <= set(d) <= {*_WIRE_KEYS, "cut"}:
        return None
    radix, step, cut = d["radix"], d["step"], d.get("cut", [0, 1])
    strs = all(isinstance(d[k], str) for k in ("reader", "port", "writer", "out"))
    ints = (isinstance(radix, list) and isinstance(step, list) and isinstance(cut, list) and len(cut) == 2
            and all(_nat(x) for x in (*radix, *cut, d["at"])) and all(isinstance(x, int) and not isinstance(x, bool)
                                                                       for x in (*step, d["base"])))
    if not (strs and ints and radix and len(radix) == len(step) and min(radix) >= 1 and cut[1] >= 1):
        return None
    return {**d, "cut": list(cut)}


def _read(d: Any) -> dict[str, Any] | None:
    """`Spot.SecondWindow.readOf?`: a read's record as a wire's, its ``param`` for the writer and no output port."""
    if not isinstance(d, Mapping) or "writer" in d or "out" in d:
        return None
    return _pattern({**{k: v for k, v in d.items() if k != "param"}, "writer": d.get("param"), "out": ""})


def _list(record: Mapping[str, Any], key: str) -> list | None:
    x = record.get(key) if isinstance(record, Mapping) else None
    return [] if x is None else x if isinstance(x, list) else None


def _all(xs: list | None, f: Callable[[Any], Any]) -> list | None:
    if xs is None:
        return None
    ys = [f(x) for x in xs]
    return None if any(y is None for y in ys) else ys


def _distinct(xs: Sequence[Any]) -> list[Any]:
    return [x for i, x in enumerate(xs) if x not in xs[:i]]


def links_of(record: Mapping[str, Any]) -> list[SS.Link] | None:
    """`Spot.SecondWindow.linksOf?`."""
    rs = _all(_list(record, "boundary_reads"), _read)
    if rs is None:
        return None
    ls = _distinct([SS.Link(r["reader"], r["port"], r["writer"]) for r in rs])
    return None if any(a.template == b.template and a.port == b.port and a.value != b.value for a in ls for b in ls) else ls


def wires_of(record: Mapping[str, Any]) -> list[SS.Wire] | None:
    """`Spot.SecondWindow.wiresOf?`."""
    ws = _all(_list(record, "wires"), _pattern)
    return None if ws is None else _distinct([SS.Wire(w["reader"], w["port"], w["writer"], w["out"]) for w in ws])


def _writer_at(p: Mapping[str, Any], i: int) -> int:
    out = p["base"]
    for r, s in zip(reversed(p["radix"]), reversed(p["step"]), strict=True):
        out, i = out + i % r * s, i // r
    return out


def _reader_units(p: Mapping[str, Any]) -> int:
    return math.prod(p["radix"]) // p["cut"][1]


def rows_of(record: Mapping[str, Any], sample: Sequence[tuple[str, int]]) -> list[tuple] | None:
    """`Spot.SecondWindow.rowsOf?`: ``(template, port, unit, row, of, out, source)`` each."""
    reads = _all(_list(record.get("registered") or {}, "reads"), _read)
    ps = [_all(_list(record, "wires"), _pattern), _all(_list(record, "boundary_reads"), _read), reads]
    if any(x is None for x in ps):
        return None
    return [(t, p["port"], u, p["cut"][0] + c, p["writer"], p["out"], _writer_at(p, (u - p["at"]) * p["cut"][1] + c))
            for p in ps[0] + ps[1] + ps[2] for t, u in sample if t == p["reader"] and p["at"] <= u < p["at"] + _reader_units(p)
            for c in range(p["cut"][1])]


def parts_of(ranks: Sequence[tuple[str, Mapping[str, int], int]], registrations: Sequence[Mapping[str, Any]],
             sample: Sequence[tuple[str, int]]) -> list[tuple[int, str, int, int, int]] | None:
    """`Spot.SecondWindow.partsOf?`."""
    ms = [member_of(ranks, registrations, t, u) for t, u in sample]
    return None if any(m is None for m in ms) else _distinct(ms)


def _box(p: Mapping[str, Any], lo: int, hi: int) -> tuple[int, int]:
    """`Spot.Wires.Wire.box`."""
    a = b = p["base"]
    radix, step = p["radix"], p["step"]
    for k in range(len(radix) - 1, -1, -1):
        r, s = radix[k], step[k]
        if lo // r == hi // r:
            q, rest = lo // r, 0
            for j in range(k - 1, -1, -1):
                rest += q % radix[j] * step[j]
                q //= radix[j]
            return a + min(lo % r * s, hi % r * s) + rest, b + max(lo % r * s, hi % r * s) + rest
        a, b, lo, hi = a + min(0, s * (r - 1)), b + max(0, s * (r - 1)), lo // r, hi // r
    return a, b


def _tree_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], registrations: Sequence[Mapping[str, Any]],
             m: tuple[int, str, int, int, int], w: Mapping[str, Any]) -> bool:
    r, t, _j, base, units = m
    lo, c, ru = segment(ranks, t, r)[0] + base, w["cut"][1], _reader_units(w)
    if t != w["reader"] or not max(lo, w["at"]) < min(lo + units, w["at"] + ru):
        return True
    a, b = (max(lo, w["at"]) - w["at"]) * c, (min(lo + units, w["at"] + ru) - w["at"]) * c - 1
    x, y = _box(w, a, b)
    p, q = (member_of(ranks, registrations, w["writer"], x), member_of(ranks, registrations, w["writer"], y)) \
        if x >= 0 else (None, None)
    return p is not None and q is not None and p[:3] == q[:3]


def wire_tree_failures(ranks: Sequence[tuple[str, Mapping[str, int], int]], registrations: Sequence[Mapping[str, Any]],
                       record: Mapping[str, Any], ms: Sequence[tuple[int, str, int, int, int]]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.wireTreeFailures`: ``ranks-wire-tree``."""
    ws = _all(_list(record, "wires"), _pattern)
    ok = ws is not None and all(_tree_ok(ranks, registrations, m, w) for m in ms for w in ws)
    return [] if ok else [("ranks-wire-tree", WIRE_TREE_WHY)]


def session_failures(record: Mapping[str, Any], statements: Sequence[Any]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.sessionFailures`: ``second-table-rows`` as `second_stage` reads it, over the joined record's
    wires; the rest refused until built."""
    ws = wires_of(record)
    own = SS.Own(sample=(), parts=None, interior=None, boundary=None, drawn=(), wires=None if ws is None else tuple(ws))
    return [(c, NOT_BUILT) for c in SESSION_CODES if c != "second-table-rows"] + SS.table_rows_failures(own, statements)


def failures(allow: bool, record: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
             served: Sequence[tuple[str, str]], registrations: Sequence[Mapping[str, Any]], root: str,
             logged: Callable[[str], bool], sample: Sequence[tuple[str, int]],
             statements: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.failures`: the ranks' failures, then ``ranks-members`` where a sampled unit is in no rank
    member, else ``ranks-wire-tree`` and the per-session failures; ``two-stage-form`` last while nothing else fails and
    the form is off."""
    found = rank_failures(record, records, served, registrations, root, logged)
    ranks = ranks_of(record.get("ranks"))
    if ranks is not None:
        ms = parts_of(ranks, registrations, sample)
        found += [("ranks-members", PARTS_WHY)] if ms is None else (
            wire_tree_failures(ranks, registrations, record, ms) + session_failures(record, statements))
    return [("two-stage-form", FORM_WHY)] if not found and not allow else found
