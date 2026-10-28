"""A window circuit's wires (`SampledProofs.Wires`): the check at the size of the description (`wires`, `Wires.check`) and the
unrolled reference (`wires_unrolled`, `Wires.checkRef`), each held to one Python reference that expands every covered
reader row, as V3a's `verity.ml.windows.interface.members` does (#4111): reader index i of a wire is unit
``at + i // C``, row ``c0 + i % C``, and its writer ``base + sum(d[k] * step[k])`` over i's mixed-radix digits, the
first digit the slowest (`Wire.writers`, copied below rather than imported: a verifier imports no `verity.ml` code).
Both pieces agreeing with it on the same inputs is the difftest of the closed form (spans, lookups, wired rows and bits,
and the overlap check pair by pair) against the expansion.
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


INPUTS = st.one_of(*[instances() for _ in range(6)], instances(broken=True), JSON)
DRAWN_INPUTS = st.one_of(*[drawn_instances() for _ in range(6)], drawn_instances(broken=True), JSON)
WIRES = D.Port(PIECES, "wires", reference, INPUTS, package=PACKAGE, of_record="lean")
WIRES_UNROLLED = D.Port(PIECES, "wires_unrolled", reference, INPUTS, package=PACKAGE, of_record="lean")

WIRES_DRAWN = D.Port(PIECES, "wires_drawn", reference_drawn, DRAWN_INPUTS, package=PACKAGE, of_record="lean")
WIRES_DRAWN_UNROLLED = D.Port(PIECES, "wires_drawn_unrolled", reference_drawn, DRAWN_INPUTS, package=PACKAGE,
                              of_record="lean")

pieces = D.server_fixture(WIRES)


@pytest.mark.parametrize("port", [WIRES, WIRES_UNROLLED], ids=["wires", "wires_unrolled"])
def test_a_wires_piece_agrees(pieces, port):
    stats = D.run(port, examples=1500, server=pieces)
    assert stats.outputs > 1000 and stats.rejects > 0, stats


@pytest.mark.parametrize("port", [WIRES_DRAWN, WIRES_DRAWN_UNROLLED], ids=["wires_drawn", "wires_drawn_unrolled"])
def test_a_drawn_wires_piece_agrees(pieces, port):
    stats = D.run(port, examples=1500, server=pieces)
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
