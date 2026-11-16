"""Round 2's reader over the joined record (`Spot.SecondWindow.failures`, `Spot.Ranks.failures` inside it) held to its
Python twin (`second_window.failures`): the ``second_window`` piece answers every failure, ``[code, detail]`` in order,
of an instance, and the reference below reads the same instance into `second_window`'s types. Generated instances are
joins of one or two ranks' windows of two templates, ``A`` writing and ``B`` reading it, laid out as a join lays them
(each rank's wires moved to its units, its registered names prefixed ``r<rank>/``, then the join's own wires), with
members of a random cap, a sample, the sampled members' statements and each rank's docket; one field of most is then
changed. Malformed instances, a field of the wrong type, are rejected by both. The Python is the difftest's reference
alone: no verdict reads it.
"""

from __future__ import annotations

import json
import shutil

import pytest
from hypothesis import strategies as st

from test_wires_difftest import PACKAGE, PIECES

import verity_difftest as D
from verity.protocols.proofs.spot.verifier import second_stage as SS
from verity.protocols.proofs.spot.verifier import second_window as W

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ROOT = "serve"
HEXES = ["a" * 128, "b" * 128]
JUNK = st.one_of(st.none(), st.booleans(), st.text(max_size=3), st.integers(-3, 3), st.lists(st.integers(-2, 3), max_size=2))


def _nat(x) -> bool:
    return type(x) is int and x >= 0


def _pairs(x, ok) -> list | None:
    if not isinstance(x, list) or not all(isinstance(e, list) and len(e) == 2 and isinstance(e[0], str) and ok(e[1])
                                          for e in x):
        return None
    return [(k, v) for k, v in x]


def _held(h) -> bool:
    return (isinstance(h, dict) and isinstance(h.get("key"), str) and _nat(h.get("base")) and _nat(h.get("units"))
            and _pairs(h.get("outs"), _nat) is not None)


def _registered(g) -> bool:
    return (isinstance(g, dict) and _nat(g.get("rank")) and all(isinstance(g.get(k), str) for k in ("digest", "circuit", "run"))
            and isinstance(g.get("members"), list) and all(map(_held, g["members"])))


def reference(j) -> dict:
    """`Spot.SecondWindow.piece`'s reading of ``j``, then `second_window.failures` on it."""
    if not isinstance(j, dict) or not isinstance(j.get("allow"), bool):
        raise D.Reject("allow is not a boolean")
    g = j.get("given") if isinstance(j.get("given"), dict) else {}
    if not isinstance(g.get("dockets"), list):
        raise D.Reject("given.dockets is not a list")
    vs = g.get("verified")
    if not isinstance(vs, list) or not all(isinstance(v, dict) and isinstance(v.get("header"), str) for v in vs):
        raise D.Reject("given.verified is not a list of {header, registered, draw, tables}")
    refs, ref_reads = _pairs(g.get("refs"), lambda _: True), _pairs(g.get("ref_reads"), lambda _: True)
    if refs is None or ref_reads is None:
        raise D.Reject("given.refs or given.ref_reads is not a list of [name, x]")
    record, records = j.get("record"), j.get("records")
    if not isinstance(record, dict) or not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise D.Reject("record is not an object, or records not a list of objects")
    served = _pairs(j.get("served"), lambda v: isinstance(v, str))
    regs = j.get("registrations")
    if served is None or not isinstance(regs, list) or not all(map(_registered, regs)):
        raise D.Reject("served or registrations malformed")
    logged, sample, statements = j.get("logged"), _pairs(j.get("sample"), _nat), j.get("statements")
    if (not isinstance(j.get("root"), str) or not isinstance(logged, list) or not all(isinstance(c, str) for c in logged)
            or sample is None or not isinstance(statements, list)):
        raise D.Reject("root, logged, sample or statements malformed")
    given = W.Given(dockets=tuple(g["dockets"]), boundary=g.get("boundary"),
                    verified=tuple(SS.Checked(v["header"], v.get("registered"), v.get("draw"), v.get("tables")) for v in vs),
                    refs=tuple(refs), ref_reads=tuple(ref_reads))
    fs = W.failures(j["allow"], given, record, records, served, regs, j["root"], lambda c: c in logged, sample, statements)
    return {"failures": json.loads(json.dumps([[c, d] for c, d in fs]))}


def _tile(t: str, n: int, cap: int, out: str, per: int) -> list[dict]:
    return [{"key": f"{t}/c{j}", "base": b, "units": min(cap, n - b), "outs": [[out, per * min(cap, n - b)]]}
            for j, b in enumerate(range(0, n, cap))]


@st.composite
def joins(draw) -> dict:
    """A join of one or two ranks' windows, as the join lays it out (module docstring)."""
    n = draw(st.integers(1, 2))
    units = [{"A": draw(st.integers(1, 3)), "B": draw(st.integers(1, 3))} for _ in range(n)]
    windows = [draw(st.sampled_from(HEXES)) for _ in range(n)]
    records, dockets = [], []
    for r in range(n):
        a, b = units[r]["A"], units[r]["B"]
        wires = []
        if draw(st.booleans()):
            step = draw(st.integers(0, 1)) if b <= a else 0
            base = draw(st.integers(0, a - 1 - step * (b - 1)))
            wires.append({"reader": "B", "port": "x", "writer": "A", "out": "y", "at": 0, "radix": [b], "base": base,
                          "step": [step], **({"cut": [0, 1]} if draw(st.booleans()) else {})})
        rec = {"wires": wires}
        if draw(st.booleans()):
            rec["registered"] = {"openings": [], "params": {"w": f"p{r}"}, "roots": [], "tables": [],
                                 "reads": [{"reader": "A", "port": "w", "param": "w", "at": 0, "radix": [a], "base": 0,
                                            "step": [0]}]}
        records.append(rec)
        named = draw(st.booleans())
        dockets.append({"roots": [{"port": ("A/" if named else "") + "y", "root": f"R{r}", "domain": "d", "leaves": 2 * a},
                                  {"port": ("B/" if named else "") + "z", "root": f"S{r}", "domain": "d", "leaves": b}]})
    ranks_doc = {"of": [{"units": units[r], "window": windows[r], "wires": len(records[r]["wires"])} for r in range(n)],
                 "prefix": W.PREFIX}
    ranks = W.ranks_of(ranks_doc)
    joined = [w for r in range(n) for w in W.shifted_wires(ranks, r, records[r])]
    if n == 2 and draw(st.booleans()):
        joined.append({"reader": "B", "port": "z", "writer": "A", "out": "y", "at": W.segment(ranks, "B", 1)[0],
                       "radix": [units[1]["B"]], "base": 0, "step": [0]})
    totals = {t: sum(u[t] for u in units) for t in ("A", "B")}
    record = {"ranks": ranks_doc, "templates": [[t, k] for t, k in totals.items()], "wires": joined,
              "registered": W.joined_registered(ranks, records)}
    if draw(st.booleans()):
        record["boundary_reads"] = [{"reader": "A", "port": "t", "param": "o0", "at": 0, "radix": [1], "base": 0,
                                     "step": [0]}]
    cap = draw(st.integers(1, 3))
    regs = [{"rank": r, "digest": windows[r], "circuit": f"c{r}", "run": f"{ROOT}/rank{r}",
             "members": _tile("A", units[r]["A"], cap, "y", 2) + _tile("B", units[r]["B"], cap, "z", 1)} for r in range(n)]
    sample = [[t, draw(st.integers(0, totals[t] - 1))] for t in draw(st.lists(st.sampled_from(["A", "B"]), max_size=3))]
    ms = W.parts_of(ranks, regs, [tuple(x) for x in sample]) or []
    statements = []
    for m in ms:
        e = {"value": draw(st.sampled_from(["y", "A/y", "z", "r0/w"])), "root": draw(st.sampled_from(["R0", "R1"])),
             "domain": "d", "leaves": draw(st.sampled_from([2, 4, 6])), "reads": draw(st.lists(
                 st.one_of(st.none(), st.integers(0, 2)), max_size=2))}
        reg = draw(st.sampled_from([None, {"ports": {draw(st.sampled_from(["x", "z", "w"])): e}}]))
        statements.append({"template": m[1], "header": {}, "draw": {}, "registered": reg})
    return {"allow": draw(st.booleans()), "given": {"dockets": dockets, "boundary": None, "verified": [], "refs": [],
                                                    "ref_reads": []},
            "record": record, "records": records, "served": [[windows[r], f"c{r}"] for r in range(n)],
            "registrations": regs, "root": ROOT, "logged": [f"c{r}" for r in range(n)], "sample": sample,
            "statements": statements}


CHANGES = ["none", "units", "window", "run", "logged", "member", "wire", "join-wire", "registered", "supplied",
           "statement", "order", "sample", "docket"]


@st.composite
def instances(draw, broken: bool = False) -> dict:
    j = draw(joins())
    rec, change = j["record"], draw(st.sampled_from(CHANGES))
    if change == "units":
        q = draw(st.sampled_from(rec["ranks"]["of"]))
        q["units"]["A"] += draw(st.sampled_from([-1, 1]))
    elif change == "window":
        j["served"][0][0] = HEXES[1] if j["served"][0][0] == HEXES[0] else HEXES[0]
    elif change == "run":
        j["registrations"][-1]["run"] = f"{ROOT}/rank9"
    elif change == "logged":
        j["logged"] = j["logged"][1:]
    elif change == "member":
        h = draw(st.sampled_from([h for g in j["registrations"] for h in g["members"]]))
        h[draw(st.sampled_from(["base", "units"]))] += 1
    elif change == "wire" and rec["wires"]:
        w = draw(st.sampled_from(rec["wires"]))
        w[draw(st.sampled_from(["base", "at"]))] += 1
    elif change == "join-wire":
        rec["wires"].append({"reader": "B", "port": "x", "writer": "A", "out": "y", "at": 0, "radix": [1], "base": 0,
                             "step": [0]})
    elif change == "registered":
        rec["registered"]["params"]["extra"] = "p"
    elif change == "supplied":
        r = draw(st.integers(0, len(j["records"]) - 1))
        j["records"][r].setdefault("registered", dict(W.NO_REGISTERED))["supplied"] = ["w"]
    elif change == "statement" and j["statements"]:
        j["statements"].pop()
    elif change == "order" and len(j["statements"]) > 1:
        j["statements"].reverse()
    elif change == "sample":
        j["sample"].append(["B", sum(q["units"]["B"] for q in rec["ranks"]["of"])])
    elif change == "docket":
        j["given"]["dockets"] = j["given"]["dockets"][:-1]
    if broken:
        key = draw(st.sampled_from(["allow", "given", "record", "records", "served", "registrations", "root", "logged",
                                    "sample", "statements"]))
        j[key] = draw(JUNK)
    return j


INPUTS = st.one_of(instances(), instances(), instances(), instances(), instances(broken=True), JUNK)
SECOND_WINDOW = D.Port(PIECES, "second_window", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(SECOND_WINDOW)


def test_the_window_reader_agrees(pieces):
    stats = D.run(SECOND_WINDOW, examples=1500, server=pieces)
    assert stats.outputs > 1000 and stats.rejects > 0, stats


def test_both_sides_reach_every_code_the_generator_aims_at(pieces):
    """The generated instances reach the ranks' codes and the sessions' (a difftest that never fails a code says nothing
    about it)."""
    seen: set[str] = set()

    def counting(j):
        out = reference(j)
        seen.update(c for c, _ in out["failures"])
        return out
    port = D.Port(PIECES, "second_window", counting, INPUTS, package=PACKAGE, of_record="lean")
    D.run(port, examples=1500, server=pieces)
    assert {"ranks-window", "ranks-run", "ranks-members", "ranks-units", "ranks-wires", "ranks-reads", "second-read",
            "second-order", "second-cover", "two-stage-form"} <= seen, sorted(seen)
