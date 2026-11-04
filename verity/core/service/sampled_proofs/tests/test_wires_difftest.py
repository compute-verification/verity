"""A window circuit's wires (`SampledProofs.Wires`): the check at the size of the description (`wires`, `Wires.check`) and the
unrolled reference (`wires_unrolled`, `Wires.checkRef`), each held to one Python reference that expands every covered
reader row, as `verity.ml.windows.interface.members` does: reader index i of a wire is unit
``at + i // C``, row ``c0 + i % C``, and its writer ``base + sum(d[k] * step[k])`` over i's mixed-radix digits, the
first digit the slowest (`Wire.writers`, copied below rather than imported: a verifier imports no `verity.ml` code).
Both pieces agreeing with it on the same inputs is the difftest of the closed form (spans, lookups, wired rows and bits,
and the overlap check pair by pair) against the expansion.

`wired_reads` (`WiredReads.ctxAt`, `WiredReads.deriveAt`) is held to a Python reference of a per-member session's wired
reads: per wired row of the member's template, each drawn instance's writer root (the writer unit's member's output, as
the registration holds it) and the writer unit's rank among its member's units.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D

PACKAGE = Path(__file__).resolve().parents[1]
PIECES = "sampled-proofs-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

WIRE_KEYS = {"reader", "port", "writer", "out", "at", "radix", "base", "step"}


def _int(x) -> bool:
    return type(x) is int


def _nat(x, what: str) -> int:
    if not (_int(x) and x >= 0):
        raise D.Reject(f"{what} is not a natural")
    return x


def _obj(x, what: str) -> dict:
    if not isinstance(x, dict):
        raise D.Reject(f"{what} is not an object")
    return x


def _list(x, what: str) -> list:
    if not isinstance(x, list):
        raise D.Reject(f"{what} is not a list")
    return x


def _str(x, what: str) -> str:
    if not isinstance(x, str):
        raise D.Reject(f"{what} is not a string")
    return x


def _template(j):
    if set(_obj(j, "a template")) != {"units", "rows", "outs"}:
        raise D.Reject("a template is {units, rows, outs}")
    units = _nat(j["units"], "units")
    rows = {}
    for p, v in _obj(j["rows"], "rows").items():
        if len(_list(v, "a row")) != 2:
            raise D.Reject("a row is [words, outer]")
        words, outer = _nat(v[0], "words"), _nat(v[1], "outer")
        if words < 1 or outer < 1:
            raise D.Reject("words and outer at least 1")
        rows[p] = (words, outer)
    outs = {o: _nat(n, "an output's words") for o, n in _obj(j["outs"], "outs").items()}
    if units < 1 or any(n < 1 for n in outs.values()):
        raise D.Reject("a template has a unit, an output a word")
    return units, rows, outs


def _wire(j):
    keys = set(_obj(j, "a wire"))
    if keys not in (WIRE_KEYS, WIRE_KEYS | {"cut"}):
        raise D.Reject("a wire's keys")
    for k in ("reader", "port", "writer", "out"):
        _str(j[k], k)
    at = _nat(j["at"], "at")
    radix = [_nat(r, "a radix") for r in _list(j["radix"], "radix")]
    step = _list(j["step"], "step")
    if not all(_int(s) for s in step) or not _int(j["base"]):
        raise D.Reject("a step or the base is not an integer")
    if not radix or len(radix) != len(step) or min(radix) < 1:
        raise D.Reject("a wire is at least one digit of radix 1 or more, and one step a digit")
    c0, c = 0, 1
    if "cut" in j:
        cut = [_nat(x, "a cut") for x in _list(j["cut"], "cut")]
        if len(cut) != 2 or cut[1] < 1:
            raise D.Reject("a wire's cut is [c0, C], C at least 1")
        c0, c = cut
    return dict(j, at=at, radix=radix, step=step, c0=c0, c=c)


def writers(w: dict, i: int) -> int:
    """V3a's `Wire.writers` at one reader index."""
    out = w["base"]
    for r, s in zip(reversed(w["radix"]), reversed(w["step"])):
        out += (i % r) * s
        i //= r
    return out


def units(w: dict) -> int:
    n = 1
    for r in w["radix"]:
        n *= r
    return n


def refusal(code: str, wire: int | None = None) -> dict:
    return {"ok": False, "code": code, "wire": wire}


def reference(j):
    if set(_obj(j, "an instance")) != {"templates", "cuts", "wires", "queries"}:
        raise D.Reject("an instance is {templates, cuts, wires, queries}")
    queries = []
    for q in _list(j["queries"], "queries"):
        if len(_list(q, "a query")) != 4:
            raise D.Reject("a query is [template, unit, port, row]")
        queries.append((_str(q[0], "template"), _nat(q[1], "unit"), _str(q[2], "port"), _nat(q[3], "row")))
    got = expand(j)
    if got.get("ok") is False:
        return got
    entries, wires = got["entries"], got["wires"]
    return {"ok": True, "wired_rows": len(entries), "wired_bits": sum(b for _, b in entries.values()),
            "spans": [[min(writers(w, x) for x in range(units(w))), max(writers(w, x) for x in range(units(w)))]
                      for w in wires],
            "lookups": [entries[q][0] if q in entries else None for q in queries]}


def expand(j):
    """The circuit's refusal, or every covered reader row with its writer and bits (``entries``) and the wires."""
    templates = {tid: _template(t) for tid, t in _obj(j["templates"], "templates").items()}
    cuts = {tid: {p: _nat(k, "a cut") for p, k in _obj(kv, "a template's cuts").items()}
            for tid, kv in _obj(j["cuts"], "cuts").items()}
    wires = [_wire(w) for w in _list(j["wires"], "wires")]

    for tid, kv in cuts.items():
        if tid not in templates:
            return refusal("cuts")
        for p, k in kv.items():
            rows = templates[tid][1]
            if p not in rows or k == 0 or rows[p][1] % k or rows[p][0] % k:
                return refusal("cuts")
    for i, w in enumerate(wires):
        if w["reader"] not in templates or w["writer"] not in templates:
            return refusal("template", i)
        (rn, rows, _), (wn, _, outs) = templates[w["reader"]], templates[w["writer"]]
        if w["port"] not in rows or w["out"] not in outs:
            return refusal("port", i)
        k = cuts.get(w["reader"], {}).get(w["port"], 1)
        if not (w["c0"] + w["c"] <= k and (w["c"] == 1 or w["radix"][-1] == w["c"])):
            return refusal("cut", i)
        if rows[w["port"]][0] // k != outs[w["out"]]:
            return refusal("width", i)
        if any(not (w["at"] + x // w["c"] < rn and 0 <= writers(w, x) < wn) for x in range(units(w))):
            return refusal("bounds", i)
    entries: dict = {}
    for w in wires:
        k = cuts.get(w["reader"], {}).get(w["port"], 1)
        for x in range(units(w)):
            row = (w["reader"], w["at"] + x // w["c"], w["port"], w["c0"] + x % w["c"])
            if row in entries:
                return refusal("overlap")
            entries[row] = ([w["writer"], writers(w, x), w["out"]], 16 * (templates[w["reader"]][1][w["port"]][0] // k))
    return {"entries": entries, "wires": wires}


def reference_drawn(j):
    """The sampled proof blocks' wired rows (`wires_drawn`): each is its writer's when both public strings are given and equal."""
    if set(_obj(j, "an instance")) != {"templates", "cuts", "wires", "drawn", "reads", "writes"}:
        raise D.Reject("an instance is {templates, cuts, wires, drawn, reads, writes}")
    drawn, reads, writes = set(), {}, {}
    for d in _list(j["drawn"], "drawn"):
        if len(_list(d, "a drawn unit")) != 2:
            raise D.Reject("a drawn unit is [template, unit]")
        drawn.add((_str(d[0], "template"), _nat(d[1], "unit")))
    for d in _list(j["reads"], "reads"):
        if len(_list(d, "a read")) != 5:
            raise D.Reject("a read is [template, unit, port, row, string]")
        reads.setdefault((_str(d[0], "t"), _nat(d[1], "u"), _str(d[2], "p"), _nat(d[3], "r")), _str(d[4], "s"))
    for d in _list(j["writes"], "writes"):
        if len(_list(d, "a write")) != 4 or not _int(d[1]):
            raise D.Reject("a write is [template, unit, out, string]")
        writes.setdefault((_str(d[0], "t"), d[1], _str(d[2], "o")), _str(d[3], "s"))
    got = expand(j)
    if got.get("ok") is False:
        return got
    rows = [(q, tuple(wr)) for q, (wr, _) in got["entries"].items() if q[:2] in drawn]
    bad = sorted(q for q, wr in rows if not (q in reads and wr in writes and reads[q] == writes[wr]))
    return {"ok": True, "accepted": not bad, "checked": len(rows), "bad": [list(q) for q in bad]}


class Refused(Exception):
    pass


def _hex(s) -> bool:
    return isinstance(s, str) and len(s) % 2 == 0 and all(c in "0123456789abcdef" for c in s)


def _within(rs, g: int) -> bool:
    return any(a <= g < b for a, b in rs)


def _rank(rs, g: int) -> int:
    return sum(max(0, min(b, g) - min(a, g)) for a, b in rs)


def _count(rs) -> int:
    return sum(max(0, b - a) for a, b in rs)


def _get(j, k):
    return j.get(k) if isinstance(j, dict) else None


def _registered(record, ports: dict) -> dict:
    """`WiredReads.writersOf`: the registration's roots by port, each with its port's leaves and domain."""
    roots = _get(record, "roots")
    seen, out = set(), {}
    for r in roots if isinstance(roots, list) else []:
        port = _get(r, "port")
        if not isinstance(port, str):
            continue
        if port in seen:
            raise Refused(f"wired reads: the registration names port {port} twice: a port is one root")
        seen.add(port)
        root, v = _get(r, "root"), ports.get(port)
        if isinstance(root, str) and port in ports:
            leaves, domain = _get(v, "leaves"), _get(v, "domain")
            if _hex(root) and _int(leaves) and leaves >= 0 and isinstance(domain, str) and _hex(domain):
                out[port] = [port, root, domain, leaves]
    return out


def _reader_units(w: dict) -> int:
    return units(w) // w["c"]


def _writer_of(wires: list, tid: str, u: int, port: str, r: int):
    """`Wires.Summary.writerOf`: the first wire covering reader row `(tid, u, port, r)`, its writer `(template, unit, out)`."""
    for w in wires:
        if (w["reader"] == tid and w["port"] == port and w["at"] <= u < w["at"] + _reader_units(w)
                and w["c0"] <= r < w["c0"] + w["c"]):
            return w["writer"], writers(w, (u - w["at"]) * w["c"] + (r - w["c0"])), w["out"]
    return None


def _derive(cuts: dict, wires: list, firsts: list, members: list, registered: dict, member, indices: list) -> list:
    """`WiredReads.deriveAt`: per wired row of the member's template, each instance's read."""
    name, rs = member
    if not rs:
        raise Refused(f"wired reads: member {name} holds no unit")
    at = next(((t, lo) for t, lo, n in firsts if lo <= rs[0][0] < lo + n), None)
    if at is None:
        raise Refused(f"wired reads: member {name}'s unit {rs[0][0]} is in no template")
    tid, first = at
    ws = [w for w in wires if w["reader"] == tid
          and any(first + w["at"] < b and a < first + w["at"] + _reader_units(w) for a, b in rs)]
    for w in ws:
        if any(a < first + w["at"] or first + w["at"] + _reader_units(w) < b for a, b in rs):
            raise Refused(f"wired reads: a wire fills port {w['port']} of only some of member {name}'s units: members "
                          "are cut at every wire's ends")
    us = []
    for g in indices:
        if not _within(rs, g):
            raise Refused(f"wired reads: the statement's unit {g} is not member {name}'s")
        us.append(max(0, g - first))
    out = []
    for w in ws:
        for r in range(w["c0"], w["c0"] + w["c"]):
            row = w["port"] if cuts.get(tid, {}).get(w["port"], 1) == 1 else f"{w['port']}#{r}"
            reads = []
            for u in us:
                wr = _writer_of(wires, tid, u, w["port"], r)
                if wr is None:
                    raise Refused(f"wired reads: no wire fills row {row} of unit {u} of {tid}")
                reads.append(_read(firsts, members, registered, row, *wr))
            out.append({"name": row, "reads": reads})
    return out


def _read(firsts, members, registered: dict, row: str, wt: str, v: int, out: str) -> list:
    """`WiredReads.readOf`: the writer unit's member's output root and the unit's rank among the member's units."""
    lo = next((lo for t, lo, _ in firsts if t == wt), None)
    if lo is None:
        raise Refused(f"wired reads: template {wt}, which writes row {row}, is not a template of the verifier's partition")
    g = lo + max(0, v)
    m = next((m for m in members if _within(m[1], g)), None)
    if m is None:
        raise Refused(f"wired reads: unit {v} of {wt}, which writes row {row}, is in no member")
    port = f"{m[0]}/{out}"
    if port not in registered:
        raise Refused(f"wired reads: the registration commits no root for {port}, which writes row {row}")
    if registered[port][3] != _count(m[1]):
        raise Refused(f"wired reads: {port}'s root has {registered[port][3]} leaves, and member {m[0]} {_count(m[1])} units")
    return [*registered[port], _rank(m[1], g)]


def reference_reads(j):
    """A per-member session's wired reads (`wired_reads`), each wired row's writer root and position per instance."""
    if set(_obj(j, "an instance")) != {"wiring", "firsts", "members", "record", "ports", "member", "indices"}:
        raise D.Reject("an instance is {wiring, firsts, members, record, ports, member, indices}")
    wiring = _obj(j["wiring"], "the wiring")
    if set(wiring) != {"templates", "cuts", "wires"}:
        raise D.Reject("the wiring is {templates, cuts, wires}")
    firsts = []
    for f in _list(j["firsts"], "firsts"):
        if len(_list(f, "a first")) != 3:
            raise D.Reject("a first is [template, first, count]")
        firsts.append((_str(f[0], "template"), _nat(f[1], "first"), _nat(f[2], "count")))
    members = []
    for m in _list(j["members"], "members"):
        if set(_obj(m, "a member")) != {"name", "units"}:
            raise D.Reject("a member is {name, units}")
        rs = []
        for u in _list(m["units"], "units"):
            if len(_list(u, "a range")) != 2:
                raise D.Reject("a range is [lo, hi]")
            rs.append((_nat(u[0], "lo"), _nat(u[1], "hi")))
        members.append((_str(m["name"], "a member's name"), rs))
    ports = _obj(j["ports"], "ports")
    name = _str(j["member"], "member")
    member = next((m for m in members if m[0] == name), None)
    if member is None:
        raise D.Reject(f"no member {name}")
    indices = [_nat(g, "an index") for g in _list(j["indices"], "indices")]
    got = expand(wiring)
    if got.get("ok") is False:
        wire = "" if got["wire"] is None else f", wire {got['wire']}"
        return {"ok": False, "reason": f"proofs: wires: the wiring is refused ({got['code']}{wire})"}
    try:
        for tid, t in sorted(wiring["templates"].items()):
            hits = [f for f in firsts if f[0] == tid]
            if len(hits) != 1:
                raise Refused(f"proofs: wires: template {tid} is not one template of the verifier's partition")
            if hits[0][2] != t["units"]:
                raise Refused(f"proofs: wires: template {tid} has {t['units']} units, and the verifier's partition "
                              f"{hits[0][2]}")
        registered = _registered(j["record"], ports)
        return {"ok": True, "ports": _derive(wiring["cuts"], got["wires"], firsts, members, registered, member, indices)}
    except Refused as e:
        return {"ok": False, "reason": str(e)}


# --- strategies ------------------------------------------------------------------------------------------------------

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text("ab/1", max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text("abk", max_size=2), inner, max_size=3),
                    max_leaves=6)
NAMES = ["A", "B", "C"]


@st.composite
def instances(draw, broken: bool = False):
    """Instances mostly well formed: two or three templates whose ports are `outer` elements of 1 or 2 words, some cut
    into `k` rows, read by up to three wires of a few digits, the later ones often on an earlier one's reader port or
    a shifted copy of it (so that some overlap), mostly from the output row of the reader row's words; at `broken` one field of one part replaced
    by arbitrary JSON."""
    names = NAMES[:draw(st.integers(2, 3))]
    templates, cuts = {}, {}
    for t in names:
        rows = {}
        for p in draw(st.sampled_from([["x"], ["x", "y"]])):
            outer = draw(st.integers(1, 4))
            rows[p] = [outer * draw(st.integers(1, 2)), outer]
            if draw(st.integers(0, 2)) == 0:
                cuts.setdefault(t, {})[p] = draw(st.integers(1, 4)) if draw(st.integers(0, 3)) == 0 else outer
        templates[t] = {"units": draw(st.integers(1, 9)), "rows": rows, "outs": {"o1": 1, "o2": 2, "o4": 4}}
    if draw(st.integers(0, 9)) == 0:
        cuts[draw(st.sampled_from(["Z", *names]))] = {draw(st.sampled_from(["x", "q"])): draw(st.integers(0, 3))}
    wires = []
    for _ in range(draw(st.integers(0, 3))):
        if wires and draw(st.integers(0, 2)) == 0:
            w = dict(wires[-1], at=max(0, wires[-1]["at"] + draw(st.integers(-2, 2))))
            if "cut" in w:
                w["cut"] = [max(0, w["cut"][0] + draw(st.integers(-1, 1))), w["cut"][1]]
            wires.append(w)
            continue
        if wires and draw(st.booleans()):
            reader, port = wires[-1]["reader"], wires[-1]["port"]
        else:
            reader = draw(st.sampled_from(names + ["Z"] * (draw(st.integers(0, 9)) == 0)))
            ports = list(templates.get(reader, {"rows": {"x": 0}})["rows"])
            port = draw(st.sampled_from(ports + ["q"] * (draw(st.integers(0, 9)) == 0)))
        k = cuts.get(reader, {}).get(port, 1)
        words = templates.get(reader, {"rows": {}})["rows"].get(port, [1])[0]
        out = (f"o{words // k}" if k and draw(st.integers(0, 3)) else draw(st.sampled_from(["o1", "o2", "o4", "o3"])))
        nd = draw(st.integers(1, 3))
        radix = draw(st.lists(st.integers(1, 3), min_size=nd, max_size=nd))
        w = {"reader": reader, "port": port, "writer": draw(st.sampled_from(names)), "out": out,
             "at": draw(st.integers(0, 4)), "radix": radix, "base": draw(st.integers(-1, 4)),
             "step": draw(st.lists(st.integers(-2, 2), min_size=nd, max_size=nd))}
        if k > 1 or draw(st.integers(0, 3)) == 0:
            c = draw(st.integers(1, max(k, 1)))
            if draw(st.integers(0, 4)):
                w["radix"] = [*radix[:-1], c]
            w["cut"] = [draw(st.integers(0, max(k - c, 0) + (draw(st.integers(0, 4)) == 0))), c]
        wires.append(w)
    queries = [[draw(st.sampled_from(names)), draw(st.integers(0, 9)), draw(st.sampled_from(["x", "y"])), draw(st.integers(0, 3))]
               for _ in range(draw(st.integers(0, 4)))]
    j = {"templates": templates, "cuts": cuts, "wires": wires, "queries": queries}
    if broken:
        part = draw(st.sampled_from(["templates", "cuts", "wires", "queries", "wire", "template"]))
        if part == "wire" and wires:
            wires[0][draw(st.sampled_from(sorted(WIRE_KEYS | {"cut", "extra"})))] = draw(JSON)
        elif part == "template":
            templates[names[0]][draw(st.sampled_from(["units", "rows", "outs", "extra"]))] = draw(JSON)
        else:
            j[part] = draw(JSON)
    return j


@st.composite
def drawn_instances(draw, broken: bool = False):
    """An instance of `instances` with some units drawn, and public strings for most of their wired rows and writers,
    mostly equal, some another string or missing."""
    j = draw(instances())
    del j["queries"]
    got = expand(j) if not broken else {"ok": False}
    drawn = [[draw(st.sampled_from(NAMES)), draw(st.integers(0, 9))] for _ in range(draw(st.integers(0, 4)))]
    reads, writes = [], []
    if got.get("ok") is not False:
        for q, (wr, _) in got["entries"].items():
            if draw(st.integers(0, 2)):
                drawn.append(list(q[:2]))
            s = draw(st.sampled_from(["s", "s", "s", "t"]))
            if draw(st.integers(0, 9)):
                reads.append([*q, s])
            if draw(st.integers(0, 9)):
                writes.append([*wr, draw(st.sampled_from(["s", s]))])
    j |= {"drawn": drawn, "reads": reads, "writes": writes}
    if broken:
        j[draw(st.sampled_from(["drawn", "reads", "writes", "wires"]))] = draw(JSON)
    return j


@st.composite
def wired_instances(draw, broken: bool = False):
    """An instance of `instances` laid out as a population: its templates' first units in a drawn order (rarely one
    missing, twice or at another count), members cut at every template's and wire's ends (some merged, so that a wire
    meets only some of a member's units, and one sometimes of two ranges), each member's outputs registered with its
    units as leaves (rarely missing, twice, at other leaves or not hex), and some of one member's units drawn, mostly a
    reader's (rarely a unit outside it)."""
    wiring = draw(instances())
    del wiring["queries"]
    templates = wiring["templates"]
    firsts, lo = [], 0
    for t in draw(st.permutations(sorted(templates))):
        firsts.append([t, lo, templates[t]["units"]])
        lo += templates[t]["units"]
    start = {t: f for t, f, _ in firsts}
    ends = {f for t, f, _ in firsts} | {lo}
    readers = []
    for w in wiring["wires"]:
        if w["reader"] in start:
            a = start[w["reader"]] + w["at"]
            readers.append((a, a + units(w) // (w["cut"][1] if "cut" in w else 1)))
            ends |= set(readers[-1])
    if draw(st.integers(0, 9)) == 0:
        k = draw(st.integers(0, len(firsts) - 1))
        change = draw(st.sampled_from(["drop", "count", "twice"]))
        if change == "drop":
            del firsts[k]
        elif change == "count":
            firsts[k] = [*firsts[k][:2], firsts[k][2] + 1]
        else:
            firsts.append(list(firsts[k]))
    cut = sorted(e for e in ends if 0 <= e <= lo)
    ranges, members = list(zip(cut, cut[1:])), []
    i = 0
    while i < len(ranges):
        a, b = ranges[i]
        while i + 1 < len(ranges) and draw(st.integers(0, 7)) == 0:
            i += 1
            b = ranges[i][1]
        members.append([[a, b]])
        i += 1
    if len(members) > 2 and draw(st.integers(0, 5)) == 0:
        k = draw(st.integers(0, len(members) - 3))
        members[k] += members.pop(k + 2)
    named = [{"name": f"m{i}", "units": rs} for i, rs in enumerate(members)]
    roots, ports = [], {}
    for m in named:
        for o in ("o1", "o2", "o4"):
            p = f"{m['name']}/{o}"
            if draw(st.integers(0, 29)) == 0:
                continue
            roots.append({"port": p, "root": "AB" if draw(st.integers(0, 29)) == 0 else
                          draw(st.sampled_from(["ab", "cd", "0f1e", ""]))})
            ports[p] = {"leaves": _count(m["units"]) + (draw(st.sampled_from([-1, 1])) if draw(st.integers(0, 29)) == 0
                                                        else 0),
                        "domain": "abc" if draw(st.integers(0, 29)) == 0 else "dd"}
    if roots and draw(st.integers(0, 29)) == 0:
        roots.append(dict(draw(st.sampled_from(roots))))
    wired = [m for m in named if any(a <= m["units"][0][0] < b for a, b in readers)]
    m = draw(st.sampled_from(wired if wired and draw(st.booleans()) else named))
    held = [g for a, b in m["units"] for g in range(a, b)]
    indices = draw(st.lists(st.sampled_from(held), max_size=4))
    if draw(st.integers(0, 9)) == 0:
        indices.append(draw(st.integers(0, lo)))
    j = {"wiring": wiring, "firsts": firsts, "members": named, "record": {"roots": roots}, "ports": ports,
         "member": m["name"], "indices": indices}
    if broken:
        part = draw(st.sampled_from([*j, "member's", "wiring's"]))
        if part == "member's":
            named[0][draw(st.sampled_from(["name", "units", "extra"]))] = draw(JSON)
        elif part == "wiring's":
            wiring[draw(st.sampled_from(["templates", "cuts", "wires", "extra"]))] = draw(JSON)
        else:
            j[part] = draw(JSON)
    return j


INPUTS = st.one_of(*[instances() for _ in range(6)], instances(broken=True), JSON)
DRAWN_INPUTS = st.one_of(*[drawn_instances() for _ in range(6)], drawn_instances(broken=True), JSON)
WIRED_INPUTS = st.one_of(*[wired_instances() for _ in range(6)], wired_instances(broken=True), JSON)
WIRES = D.Port(PIECES, "wires", reference, INPUTS, package=PACKAGE, of_record="lean")
WIRES_UNROLLED = D.Port(PIECES, "wires_unrolled", reference, INPUTS, package=PACKAGE, of_record="lean")

WIRES_DRAWN = D.Port(PIECES, "wires_drawn", reference_drawn, DRAWN_INPUTS, package=PACKAGE, of_record="lean")
WIRES_DRAWN_UNROLLED = D.Port(PIECES, "wires_drawn_unrolled", reference_drawn, DRAWN_INPUTS, package=PACKAGE,
                              of_record="lean")
WIRED_READS = D.Port(PIECES, "wired_reads", reference_reads, WIRED_INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(WIRES)


@pytest.mark.parametrize("port", [WIRES, WIRES_UNROLLED], ids=["wires", "wires_unrolled"])
def test_a_wires_piece_agrees(pieces, port):
    stats = D.run(port, examples=1500, server=pieces)
    assert stats.outputs > 1000 and stats.rejects > 0, stats


@pytest.mark.parametrize("port", [WIRES_DRAWN, WIRES_DRAWN_UNROLLED], ids=["wires_drawn", "wires_drawn_unrolled"])
def test_a_drawn_wires_piece_agrees(pieces, port):
    stats = D.run(port, examples=1500, server=pieces)
    assert stats.outputs > 1000 and stats.rejects > 0, stats


def test_the_wired_reads_piece_agrees(pieces):
    stats = D.run(WIRED_READS, examples=1500, server=pieces)
    assert stats.outputs > 1000 and stats.rejects > 0, stats


def _gather(**change) -> dict:
    """Attention's K read as a gather: reader `A` (2 units, K cut into 3 rows of 2 words) reads row c of unit u from
    writer `G`'s unit 3u + c, one wire for the whole gather."""
    j = {"templates": {"G": {"units": 6, "rows": {"x": [4, 1]}, "outs": {"o0": 2}},
                       "A": {"units": 2, "rows": {"k": [6, 3]}, "outs": {"o0": 1}}},
         "cuts": {"A": {"k": 3}},
         "wires": [{"reader": "A", "port": "k", "writer": "G", "out": "o0", "at": 0, "radix": [2, 3], "base": 0,
                    "step": [3, 1], "cut": [0, 3]}],
         "queries": [["A", 1, "k", 2], ["A", 0, "k", 1], ["A", 2, "k", 0]]}
    for path, v in change.items():
        *head, last = path.split("__")
        node = j
        for h in head:
            node = node[int(h)] if isinstance(node, list) else node[h]
        node[last] = v
    return j


@pytest.mark.parametrize("name, j, want", [
    ("a gather", _gather(), {"ok": True, "wired_rows": 6, "wired_bits": 192, "spans": [[0, 5]],
                             "lookups": [["G", 5, "o0"], ["G", 1, "o0"], None]}),
    ("a cut of no port", _gather(cuts={"A": {"q": 3}}), refusal("cuts")),
    ("a cut that doesn't divide the outer length", _gather(cuts={"A": {"k": 2}}), refusal("cuts")),
    ("a writer of no template", _gather(wires__0__writer="H"), refusal("template", 0)),
    ("an output of no row", _gather(wires__0__out="o9"), refusal("port", 0)),
    ("rows past the cut", _gather(wires__0__cut=[1, 3]), refusal("cut", 0)),
    ("a row digit that isn't the last", _gather(wires__0__radix=[3, 2], wires__0__step=[1, 3]), refusal("cut", 0)),
    ("a writer row of other words", _gather(templates__G__outs={"o0": 3}), refusal("width", 0)),
    ("a cut wire of an uncut port", _gather(cuts={}), refusal("cut", 0)),
    ("a writer past its template", _gather(wires__0__base=1), refusal("bounds", 0)),
    ("a negative writer", _gather(wires__0__base=-1), refusal("bounds", 0)),
], ids=lambda x: x if isinstance(x, str) else "")
def test_named_cases(pieces, name, j, want):
    assert reference(j) == want
    for fn in ("wires", "wires_unrolled"):
        kind, out = pieces.call(fn, j)
        assert (kind, out) == ("output", D.canonical(want)), (fn, kind, out)


def test_an_overlap_is_refused(pieces):
    j = _gather()
    j["wires"].append(dict(j["wires"][0], cut=[2, 1], radix=[1], step=[0], at=1))
    assert reference(j) == refusal("overlap")
    for fn in ("wires", "wires_unrolled"):
        assert pieces.call(fn, j) == ("output", D.canonical(refusal("overlap")))


def test_the_drawn_check_reads_only_drawn_units(pieces):
    """The gather with unit 1 drawn: its three wired rows are checked; unit 0's row that is not its writer's is not."""
    j = _gather()
    del j["queries"]
    j |= {"drawn": [["A", 1]],
          "reads": [["A", 1, "k", 0, "s3"], ["A", 1, "k", 1, "s4"], ["A", 1, "k", 2, "s5"], ["A", 0, "k", 0, "bad"]],
          "writes": [["G", u, "o0", f"s{u}"] for u in range(6)]}
    want = {"ok": True, "accepted": True, "checked": 3, "bad": []}
    assert reference_drawn(j) == want
    for fn in ("wires_drawn", "wires_drawn_unrolled"):
        assert pieces.call(fn, j) == ("output", D.canonical(want))
    j["drawn"] = [["A", 0], ["A", 1]]
    want = {"ok": True, "accepted": False, "checked": 6, "bad": [["A", 0, "k", 0], ["A", 0, "k", 1], ["A", 0, "k", 2]]}
    assert reference_drawn(j) == want
    for fn in ("wires_drawn", "wires_drawn_unrolled"):
        assert pieces.call(fn, j) == ("output", D.canonical(want))


G0, G1, A = ({"name": "g0", "units": [[0, 3]]}, {"name": "g1", "units": [[3, 6]]}, {"name": "a", "units": [[6, 8]]})


def _wiring(**change) -> dict:
    j = _gather(**change)
    del j["queries"]
    return j


def _gathered(**change) -> dict:
    """The gather as a population: `G`'s units 0 to 5 in members g0 and g1, `A`'s 6 and 7 in member a, each member's
    output registered with its units as leaves, and a's instances at units 7 and 6 drawn."""
    j = {"wiring": _wiring(), "firsts": [["G", 0, 6], ["A", 6, 2]], "members": [G0, G1, A],
         "record": {"roots": [{"port": "g0/o0", "root": "a0"}, {"port": "g1/o0", "root": "a1"},
                              {"port": "a/o0", "root": "a2"}]},
         "ports": {"g0/o0": {"leaves": 3, "domain": "dd"}, "g1/o0": {"leaves": 3, "domain": "dd"},
                   "a/o0": {"leaves": 2, "domain": "dd"}},
         "member": "a", "indices": [7, 6]}
    return j | change


def _why(reason: str) -> dict:
    return {"ok": False, "reason": reason}


GATHERED = {"ok": True, "ports": [
    {"name": f"k#{r}", "reads": [["g1/o0", "a1", "dd", 3, r], ["g0/o0", "a0", "dd", 3, r]]} for r in range(3)]}


@pytest.mark.parametrize("name, j, want", [
    ("a gather", _gathered(), GATHERED),
    ("a member no wire reads into", _gathered(member="g0", indices=[0, 2]), {"ok": True, "ports": []}),
    ("a writer member of all of G", _gathered(members=[{"name": "g", "units": [[0, 6]]}, A],
                                              record={"roots": [{"port": "g/o0", "root": "a0"}]},
                                              ports={"g/o0": {"leaves": 6, "domain": "dd"}}),
     {"ok": True, "ports": [{"name": f"k#{r}", "reads": [["g/o0", "a0", "dd", 6, 3 + r], ["g/o0", "a0", "dd", 6, r]]}
                            for r in range(3)]}),
    ("a member a wire meets only partly", _gathered(members=[G1, {"name": "a", "units": [[6, 8], [0, 3]]}]),
     _why("wired reads: a wire fills port k of only some of member a's units: members are cut at every wire's ends")),
    ("a writer's root of other leaves", _gathered(ports={**_gathered()["ports"], "g1/o0": {"leaves": 2, "domain": "dd"}}),
     _why("wired reads: g1/o0's root has 2 leaves, and member g1 3 units")),
    ("a writer's output not registered", _gathered(record={"roots": [{"port": "g1/o0", "root": "a1"}]}),
     _why("wired reads: the registration commits no root for g0/o0, which writes row k#0")),
    ("a writer's root not lowercase hex", _gathered(record={"roots": [{"port": "g0/o0", "root": "A0"},
                                                                       {"port": "g1/o0", "root": "a1"}]}),
     _why("wired reads: the registration commits no root for g0/o0, which writes row k#0")),
    ("a port registered twice", _gathered(record={"roots": [*_gathered()["record"]["roots"],
                                                            {"port": "g0/o0", "root": "b0"}]}),
     _why("wired reads: the registration names port g0/o0 twice: a port is one root")),
    ("a writer unit in no member", _gathered(members=[G0, A]),
     _why("wired reads: unit 3 of G, which writes row k#0, is in no member")),
    ("an instance outside the member", _gathered(indices=[7, 5]), _why("wired reads: the statement's unit 5 is not member a's")),
    ("a template outside the partition", _gathered(firsts=[["G", 0, 6]]),
     _why("proofs: wires: template A is not one template of the verifier's partition")),
    ("a template of other units", _gathered(firsts=[["G", 0, 6], ["A", 6, 3]]),
     _why("proofs: wires: template A has 2 units, and the verifier's partition 3")),
    ("a refused wiring", _gathered(wiring=_wiring(wires__0__base=1)),
     _why("proofs: wires: the wiring is refused (bounds, wire 0)")),
], ids=lambda x: x if isinstance(x, str) else "")
def test_wired_reads_named_cases(pieces, name, j, want):
    """Each wired row of a drawn reader opens its writer member's registered root at the writer unit's rank, or the
    derivation is refused: what `Flock.RegWired.check` then holds each instance's row to."""
    assert reference_reads(j) == want
    assert pieces.call("wired_reads", j) == ("output", D.canonical(want))
