"""Round 2's reader over the joined record and its ranks' check, as the verifier's `Spot.SecondWindow` and `Spot.Ranks`
define them: a two-stage run's second-stage sessions held to the joined circuit's record (its ``ranks``, ``wires`` and
``boundary_reads``, `verity.commitments.periods.interface`) and to each rank's docket, with no interior
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
from dataclasses import dataclass
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
             "rank order, or a wire after them reads no reader unit, or one in no rank, or in some rank whose units it reads "
             "has all its writers of them in that rank")
READS_WHY = ("the joined record's registered declarations are not its ranks', each named r<r>/... and at its rank's units, "
             "in rank order, or a rank's record declares what the join names no rule for (values, histories, supplied "
             "parameters)")
#: the keys of a rank record's ``registered`` the join names, each by its rule (`Spot.Ranks.REGISTERED_KEYS`)
REGISTERED_KEYS = ("formats", "openings", "params", "reads", "roots", "tables")
WIRE_TREE_WHY = ("a sampled reader member reads one port through writer rows in more than one rank's window, or in none: its "
                 "session reads each wired port from one rank's committed tree")
PARTS_WHY = "a sampled proof block is in no rank member"
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


def bounds(ranks: Sequence[tuple[str, Mapping[str, int], int]], template: str) -> list[int]:
    """`Spot.Ranks.Ranks.bounds`: ``template``'s first unit of each rank in the joined circuit, then its count."""
    b = [0]
    for q in ranks:
        b.append(b[-1] + q[1].get(template, 0))
    return b


def _segment_in(b: Sequence[int], r: int) -> tuple[int, int]:
    lo = b[min(r, len(b) - 1)]
    return lo, b[r + 1] if r + 1 < len(b) else lo


def segment(ranks: Sequence[tuple[str, Mapping[str, int], int]], template: str, r: int) -> tuple[int, int]:
    """`Spot.Ranks.Ranks.segment`: rank ``r``'s units of ``template`` in the joined circuit, ``[lo, hi)``."""
    return _segment_in(bounds(ranks, template), r)


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


def _rank_in(b: Sequence[int], lo: int, hi: int) -> int | None:
    return next((r for r in range(len(b) - 1) if b[r] <= lo <= hi < b[r + 1]), None)


def _rank_of(ranks: Sequence[tuple[str, Mapping[str, int], int]], template: str, lo: int, hi: int) -> int | None:
    return _rank_in(bounds(ranks, template), lo, hi)


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


def _join_wire_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], w: Any) -> bool:
    """`Spot.Ranks.joinWireOk`: one box per wire and reader rank it reads."""
    p = _pattern(w)
    if p is None:
        return False
    units, c = _reader_units(p), p["cut"][1]
    br, bw = bounds(ranks, p["reader"]), bounds(ranks, p["writer"])
    if units < 1 or p["at"] + units > _segment_in(br, len(ranks) - 1)[1]:
        return False
    for r in range(len(ranks)):
        lo, hi = _segment_in(br, r)
        if max(lo, p["at"]) < min(hi, p["at"] + units):
            x, y = _box(p, (max(lo, p["at"]) - p["at"]) * c, (min(hi, p["at"] + units) - p["at"]) * c - 1)
            if x >= 0 and _rank_in(bw, x, y) == r:
                return False
    return True


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
    if any(set(g) - set(REGISTERED_KEYS) for _r, g in regs):
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


def _reads_in(w: Mapping[str, Any], t: str, lo: int, n: int) -> bool:
    return w["reader"] == t and max(lo, w["at"]) < min(lo + n, w["at"] + _reader_units(w))


def _bisect(n: int, p: Callable[[int], bool]) -> int:
    """`Spot.SecondWindow.bisect`: the least ``i < n`` where ``p(i)`` holds, ``p`` false and then true; ``n`` if never."""
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if p(mid) else (mid + 1, hi)
    return lo


class Index:
    """`Spot.SecondWindow.Index`: the wires sorted by reader template, port and first unit (ties in the record's order),
    and at each place the most end of its port's wires up to there."""

    def __init__(self, ws: Sequence[Mapping[str, Any]]) -> None:
        self.ws = sorted(ws, key=lambda w: (w["reader"], w["port"], w["at"]))
        self.keys = [(w["reader"], w["port"]) for w in self.ws]
        self.ends: list[int] = []
        for i, w in enumerate(self.ws):
            e = w["at"] + _reader_units(w)
            same = i > 0 and (self.ws[i - 1]["reader"], self.ws[i - 1]["port"]) == (w["reader"], w["port"])
            self.ends.append(max(self.ends[-1], e) if same else e)

    def meets(self, t: str, port: str, lo: int, n: int) -> list[Mapping[str, Any]]:
        """`Index.meets`: the wires reading ``t``'s ``port`` that meet units ``[lo, lo + n)``, by first unit."""
        keys = self.keys
        a = _bisect(len(keys), lambda i: not keys[i] < (t, port))
        b = _bisect(len(keys), lambda i: (t, port) < keys[i])
        i = a + _bisect(b - a, lambda k: lo < self.ends[a + k])
        j = a + _bisect(b - a, lambda k: lo + n <= self.ws[a + k]["at"])
        return [w for w in self.ws[i:j] if _reads_in(w, t, lo, n)]

    def ports(self, t: str) -> list[str]:
        """`Index.ports`: ``t``'s wired ports, each once, by name."""
        a = _bisect(len(self.ws), lambda i: not self.ws[i]["reader"] < t)
        b = _bisect(len(self.ws), lambda i: t < self.ws[i]["reader"])
        out: list[str] = []
        for w in self.ws[a:b]:
            if not out or out[-1] != w["port"]:
                out.append(w["port"])
        return out


def _writer_ranks(ranks: Sequence[tuple[str, Mapping[str, int], int]], m: tuple[int, str, int, int, int],
                  w: Mapping[str, Any]) -> list[int] | None:
    """`Spot.SecondWindow.writerRanks`."""
    r, t, _j, base, units = m
    lo, c, ru = segment(ranks, t, r)[0] + base, w["cut"][1], _reader_units(w)
    if t != w["reader"] or not max(lo, w["at"]) < min(lo + units, w["at"] + ru):
        return []
    a, b = (max(lo, w["at"]) - w["at"]) * c, (min(lo + units, w["at"] + ru) - w["at"]) * c - 1
    x, y = _box(w, a, b)
    if x < 0:
        return None
    p, q = _rank_of(ranks, w["writer"], x, x), _rank_of(ranks, w["writer"], y, y)
    return [p] if p is not None and p == q else None


def _tree_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], ix: Index, m: tuple[int, str, int, int, int]) -> bool:
    """`Spot.SecondWindow.treeOk`."""
    lo = segment(ranks, m[1], m[0])[0] + m[3]
    for port in ix.ports(m[1]):
        rss = [_writer_ranks(ranks, m, w) for w in ix.meets(m[1], port, lo, m[4])]
        if any(x is None for x in rss) or len({r for x in rss for r in x}) > 1:
            return False
    return True


def wire_tree_failures(ranks: Sequence[tuple[str, Mapping[str, int], int]], record: Mapping[str, Any],
                       ms: Sequence[tuple[int, str, int, int, int]]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.wireTreeFailures`: ``ranks-wire-tree``."""
    ws = _all(_list(record, "wires"), _pattern)
    ok = ws is not None and all(_tree_ok(ranks, Index(ws), m) for m in ms)
    return [] if ok else [("ranks-wire-tree", WIRE_TREE_WHY)]


TWO_WRITERS_WHY = "the member reads a port through two writers, or partly through none: no one session's header binds it"
DESIGN_WHY = "the joined record's wires or boundary reads are none: no session reads by them"


@dataclass(frozen=True)
class Given:
    """`Spot.SecondWindow.Given`: each rank's docket record in rank order, round 1's boundary record, the verdicts of
    record's sessions (`second_stage.Checked`) and the reference header and registered reads of each template."""

    dockets: tuple[Any, ...]
    boundary: Any = None
    verified: tuple[SS.Checked, ...] = ()
    refs: tuple[tuple[str, Any], ...] = ()
    ref_reads: tuple[tuple[str, Any], ...] = ()


def wire_for(ix: Index, t: str, port: str, lo: int, n: int) -> tuple[SS.Wire | None] | None:
    """`Spot.SecondWindow.wireFor`, the one rule every per-session check reads a wire by: ``(None,)`` for no wired port
    of the member's, ``(wire,)`` for its one writer's output port covering every unit, None for two writers or a
    partial cover."""
    meet = ix.meets(t, port, lo, n)
    if not meet:
        return (None,)
    w0 = meet[0]
    covered = sum(max(min(lo + n, w["at"] + _reader_units(w)) - max(lo, w["at"]), 0) for w in meet)
    if all(w["writer"] == w0["writer"] and w["out"] == w0["out"] and w["cut"][1] == 1 for w in meet) and covered == n:
        return (SS.Wire(t, port, w0["writer"], w0["out"]),)
    return None


def member_wires(ix: Index, m: tuple[int, str, int, int, int], lo: int) -> list[SS.Wire] | None:
    """`Spot.SecondWindow.memberWires`."""
    out = []
    for port in ix.ports(m[1]):
        x = wire_for(ix, m[1], port, lo, m[4])
        if x is None:
            return None
        out += [x[0]] if x[0] is not None else []
    return out


def writer_unit(ix: Index, t: str, port: str, u: int) -> tuple[str, int] | None:
    """`Spot.SecondWindow.writerUnit`."""
    w = next(iter(ix.meets(t, port, u, 1)), None)
    return None if w is None else (w["writer"], _writer_at(w, (u - w["at"]) * w["cut"][1]))


def own_for(g: Given, links: Sequence[SS.Link], m: tuple[int, str, int, int, int], drawn: Sequence[int],
            wires: Sequence[SS.Wire]) -> SS.Own:
    """`Spot.SecondWindow.ownFor`."""
    return SS.Own(sample=(0,), parts=(SS.Part(m[1], m[3], m[4]),),
                  interior=g.dockets[m[0]] if m[0] < len(g.dockets) else None, boundary=g.boundary, drawn=tuple(drawn),
                  links=tuple(x for x in links if x.template == m[1]), verified=g.verified, refs=g.refs,
                  ref_reads=g.ref_reads, wires=tuple(wires))


def _wired_read_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], ix: Index,
                   dockets: Sequence[Any], p: SS.Part, r: int, w: SS.Wire, v: str, e: Any) -> bool:
    """`Spot.SecondWindow.wiredReadOk`."""
    first = segment(ranks, p.template, r)[0]
    x0 = writer_unit(ix, p.template, w.port, first + p.lo)
    r2 = None if x0 is None or x0[1] < 0 else _rank_of(ranks, w.writer, x0[1], x0[1])
    if r2 is None:
        return False
    d = dockets[r2] if r2 < len(dockets) else None
    wfirst, x = segment(ranks, w.writer, r2)[0], SS._by_port(d).get(v)
    if x is None or not _nat(SS._get(x, "leaves")):
        return False

    def row(u: int) -> int | None:
        y = writer_unit(ix, p.template, w.port, first + u)
        return y[1] - wfirst if y is not None and wfirst <= y[1] and _rank_of(ranks, w.writer, y[1], y[1]) == r2 else None
    return (v == SS._port_of(SS._by_template(d), w.writer, w.wport) and SS._same_at(e, x)
            and SS._placed_ok(1, p, e, SS._get(x, "leaves"), row))


def window_read_ok(ranks: Sequence[tuple[str, Mapping[str, int], int]], ix: Index,
                   dockets: Sequence[Any], own: SS.Own, named: bool, p: SS.Part, r: int, port: str, e: Any) -> bool:
    """`Spot.SecondWindow.windowReadOk`: ``second-read`` in the window form."""
    lo, hi = segment(ranks, p.template, r)
    v = SS._get(e, "value")
    if not isinstance(v, str):
        return False
    if SS._link_of(own, p.template, port) is not None:
        return False
    w = SS._wire_of(own, p.template, port)
    if w is not None:
        return _wired_read_ok(ranks, ix, dockets, p, r, w, v, e)
    x = SS._by_port(own.interior).get(v)
    if x is not None:
        return (v == SS._port_of(named, p.template, port) and SS._same_at(e, x)
                and SS._rows_ok(1, p, e, hi - lo, lambda u: u))
    if v in SS._by_port(own.boundary):
        return False
    x = SS._registered_entry(own.boundary, v)
    if x is None:
        x = SS._registered_entry(own.interior, v)
    return x is not None and SS._same_at(e, x)


def position_of(registrations: Sequence[Mapping[str, Any]], m: tuple[int, str, int, int, int]) -> int:
    """`Spot.SecondWindow.positionOf`."""
    g = _reg(registrations, m[0])
    if g is None:
        return 0
    key = f"{m[1]}/c{m[2]}"
    return next((i for i, h in enumerate(g["members"]) if h["key"] == key), len(g["members"]))


def member_codes(ranks: Sequence[tuple[str, Mapping[str, int], int]], ix: Index, g: Given,
                 own: SS.Own, m: tuple[int, str, int, int, int], k: int, s: SS.Stmt) -> list[str]:
    """`Spot.SecondWindow.memberCodes`."""
    named, p = SS._by_template(own.interior), SS.Part(m[1], m[3], m[4])
    reads = SS._entries(SS._get(s.registered, "ports"))
    read = (all(window_read_ok(ranks, ix, g.dockets, own, named, p, m[0], port, e) for port, e in reads)
            and SS._linked_read(own, m[1], reads) and SS._wired_read(own, m[1], reads))
    return [c for c in SS._stmt_codes(own, named, k, p, s) if c != "second-read"] + ([] if read else ["second-read"])


def session_failures(g: Given, ranks: Sequence[tuple[str, Mapping[str, int], int]],
                     registrations: Sequence[Mapping[str, Any]], record: Mapping[str, Any],
                     ms: Sequence[tuple[int, str, int, int, int]], sample: Sequence[tuple[str, int]],
                     statements: Sequence[Any]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.sessionFailures`: the statements by position against the sampled members, then
    ``second-table-rows`` as `second_stage` reads it over the joined record's wires."""
    links, ws, out = links_of(record), _all(_list(record, "wires"), _pattern), []
    if links is None or ws is None:
        out.append(("second-read", DESIGN_WHY))
    else:
        ix = Index(ws)
        for i in range(max(len(ms), len(statements))):
            m = ms[i] if i < len(ms) else None
            if i >= len(statements):
                out.append(("second-cover", {"statement": i, "template": m[1]}))
                continue
            st = SS._stmt(statements[i])
            if m is None:
                out.append(("second-order", {"statement": i, "template": None}))
            elif st is None:
                out.append(("second-statement", {"statement": i, "template": None}))
            elif st.template != m[1]:
                out.append(("second-order", {"statement": i, "template": st.template}))
            else:
                first = segment(ranks, m[1], m[0])[0]
                lo = first + m[3]
                wires = member_wires(ix, m, lo)
                if wires is None:
                    out.append(("second-read", {"statement": i, "template": st.template, "why": TWO_WRITERS_WHY}))
                    continue
                drawn = [u - first for t, u in sample if t == m[1] and lo <= u < lo + m[4]]
                own = own_for(g, links, m, drawn, wires)
                out += [(c, {"statement": i, "template": st.template})
                        for c in member_codes(ranks, ix, g, own, m, position_of(registrations, m), st)]
    ws2 = wires_of(record)
    own = SS.Own(sample=(), parts=None, interior=None, boundary=None, drawn=(), wires=None if ws2 is None else tuple(ws2))
    return out + SS.table_rows_failures(own, statements)


def failures(allow: bool, g: Given, record: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
             served: Sequence[tuple[str, str]], registrations: Sequence[Mapping[str, Any]], root: str,
             logged: Callable[[str], bool], sample: Sequence[tuple[str, int]],
             statements: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.failures`: the ranks' failures, then ``ranks-members`` where a sampled proof block is in no rank
    member, else ``ranks-wire-tree`` and the per-session failures; ``two-stage-form`` last while nothing else fails and
    the form's switch is off."""
    found = rank_failures(record, records, served, registrations, root, logged)
    ranks = ranks_of(record.get("ranks"))
    if ranks is not None:
        ms = parts_of(ranks, registrations, sample)
        found += [("ranks-members", PARTS_WHY)] if ms is None else (
            wire_tree_failures(ranks, record, ms) + session_failures(g, ranks, registrations, record, ms, sample,
                                                                     statements))
    return [("two-stage-form", FORM_WHY)] if not found and not allow else found
