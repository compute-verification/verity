"""`Q_call` v1 pinned (`qcall_vectors.json`, `verity/ml/circuits/PROTOCOL.md` §11): a word program and a Boolean program whose Calls exercise
every rule of `verity.ml.circuits.cut`'s "Q_call v1" -- a Call that fits, one that splits into sub-Calls, a batch, a scan (its carry chained,
and passed through), a pass-through output, a constant read by two units, a constant-derived gate read across units, former wiring,
a dead gate, a recompute across units and one within a unit, a gate of more than 4096 inputs, a 64-bit input -- at X = 32 and 64,
and every refusal (`gate-too-wide`, `root-form`, not topological, another program, served commits, malformed objects).  Programs
are descriptor bytes; the evaluation is from the bytes.  `python verity/ml/circuits/tests/test_qcall_vectors.py --write` rewrites
the file."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from test_qword_vectors import TAdd64, Dead, Max3, Step, TwiceSep, WideTwice, Wiring, _c, _canon, _top, registry as _qword_registry
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec, cut, partition_object as PO
from verity.ml.circuits.defs import PrimitiveDefinition, Registry
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import I32Add

PATH = Path(__file__).parent / "qcall_vectors.json"
BIT, V32, V64 = BL.BIT, Value(32), Value(64)

TLo32 = PrimitiveDefinition("TLo32", 1, (("a", V64),), V32, lambda a: a & 0xFFFFFFFF, register=False)


def registry() -> Registry:
    r = _qword_registry()
    r.add(TLo32)
    return r


# -- word Calls ---------------------------------------------------------------------------------------------------------------------

#: two sub-Calls, 64 bits out: splits into its two call nodes, each one unit
Pair = _c("CPair", (("x", Array(3, V32)), ("y", Array(3, V32))), Tuple(V32, V32), lambda B, S, x, y: tuple_of(B.call(Max3, x), B.call(Max3, y)))


def _nest(B, S, x, y, z):
    p = B.call(Pair, x, y)
    return tuple_of(p[0], p[1], B.call(I32Add, p[0], z))


#: 96 bits out: its Pair node (64 bits) splits again, into Max3 calls; paths two steps deep
Nest = _c("CNest", (("x", Array(3, V32)), ("y", Array(3, V32)), ("z", V32)), Tuple(V32, V32, V32), _nest)
#: a batch of a composite (members of 32 bits) and a batch of a primitive (members of one gate)
Rows = _c("CRows", (("rows", Array(2, Array(3, V32))), ("a", Array(2, V32)), ("b", Array(2, V32))), Tuple(Array(2, V32), Array(2, V32)),
          lambda B, S, rows, a, b: tuple_of(B.batch(Max3, rows, axes=(0,)), B.batch(I32Add, a, b, axes=(0, 0))))
#: a scan over Step: each iteration returns its sum as carry and output (one gate), read by the next iteration
Scan = _c("CScan", (("xs", Array(3, V32)), ("c", V32)), Tuple(V32, Array(3, V32)), lambda B, S, xs, c: B.scan(Step, c, xs=[xs]))
#: an iteration that passes its carry through: the final carry is the Call's parameter
StepKeep = _c("CStepKeep", (("c", V32), ("x", V32)), Tuple(V32, V32), lambda B, S, c, x: tuple_of(c, B.call(I32Add, c, x)))
ScanKeep = _c("CScanKeep", (("xs", Array(3, V32)), ("c", V32)), Tuple(V32, Array(3, V32)), lambda B, S, xs, c: B.scan(StepKeep, c, xs=[xs]))
#: a call node of no gates, returning its parameter
Ident = _c("CIdent", (("a", V32),), V32, lambda B, S, a: a)
#: returns a parameter, a sum, a call node's pass-through and another sum: 64 bits of its own, so it splits
Pass = _c("CPass", (("x", V32), ("y", V32)), Tuple(V32, V32, V32, V32),
          lambda B, S, x, y: tuple_of(x, B.call(I32Add, x, y), B.call(Ident, y), B.call(I32Add, y, y)))


def _consts(B, S, x):
    c = B.call(const(32, 7))
    d = B.call(I32Add, c, c)                                                                       # constant-derived
    return tuple_of(B.call(I32Add, x[0], c), B.call(I32Add, x[1], d), B.call(I32Add, x[0], d))


#: a constant read by two units, and a constant-derived gate (its own unit, committed) read by two
Consts = _c("CConsts", (("x", Array(2, V32)),), Tuple(V32, V32, V32), _consts)
#: a 64-bit input and a 64-bit constant, each read by a 32-bit gate: no wide gate
Lo = _c("CLo", (("w", V64),), V32, lambda B, S, w: B.call(I32Add, B.call(TLo32, w), B.call(TLo32, B.call(const(64, (1 << 40) + 3)))))


def _redundant(B, S, x):
    return B.call(I32Add, B.call(I32Add, x[0], x[1]), B.call(I32Add, x[0], x[1]))


#: a value computed twice inside one unit: redundant, not across
Redundant = _c("CRedundant", (("x", Array(2, V32)),), V32, _redundant)
WORD_CALLS = [Max3, Nest, Rows, Scan, ScanKeep, Pass, Consts, Lo, TwiceSep, Redundant, Wiring, Dead, WideTwice]

# -- Boolean Calls ------------------------------------------------------------------------------------------------------------------


def _cat(t, *cs) -> Coll:
    return Coll(t, concat([c.refs for c in cs]))


W32 = Array(32, BIT)
BXor32 = _c("CBXor32", (("a", W32), ("b", W32)), W32, lambda B, S, a, b: B.batch(BL.XOR, _cat(Array(2, W32), a, b), axes=(1,)))
BAnd32 = _c("CBAnd32", (("a", W32), ("b", W32)), W32, lambda B, S, a, b: B.batch(BL.AND, _cat(Array(2, W32), a, b), axes=(1,)))
#: two 32-bit sub-Calls, 64 bits out: one unit each
BPair = _c("CBPair", (("a", W32), ("b", W32)), Tuple(W32, W32), lambda B, S, a, b: tuple_of(B.call(BXor32, a, b), B.call(BAnd32, a, b)))
H = Array(20, BIT)


def _bconsts(B, S, x, y):
    zero = B.call(BL.constant(0))
    one = B.call(BL.NOT, zero)                                                                     # constant-derived
    ones = _cat(Array(20, BIT), *[one] * 20)
    zeros = _cat(Array(20, BIT), *[zero] * 20)
    return tuple_of(B.batch(BL.XOR, _cat(Array(2, H), x, ones), axes=(1,)), B.batch(BL.XOR, _cat(Array(2, H), y, zeros), axes=(1,)))


#: 40 bits out: Not(Const 0) is a unit, committed, read by the first batch; the constant 0 read by it and by the second batch
BConsts = _c("CBConsts", (("x", H), ("y", H)), Tuple(H, H), _bconsts)


def _full_adder(B, S, a, b, c):
    ops = BL.operands
    t = B.call(BL.XOR, ops(a, b))
    s = B.call(BL.XOR, ops(t, c))
    return _cat(Array(2, BIT), s, B.call(BL.XOR, ops(B.call(BL.AND, ops(a, b)), B.call(BL.AND, ops(t, c)))))


FullAdder = _c("CBFullAdder", (("a", BIT), ("b", BIT), ("c", BIT)), Array(2, BIT), _full_adder)


def _ripple(B, S, x, y):
    zero = B.call(BL.constant(0))
    out, carry = [], zero
    for i in range(33):
        s = B.call(FullAdder, x[i], y[i], carry)
        out.append(s[0])
        carry = s[1]
    return _cat(Array(34, BIT), *out, carry)


#: a 33-bit ripple-carry adder: 34 bits out, so its 33 full-adder calls are the units, each carry committed
Ripple = _c("CBRipple", (("x", Array(33, BIT)), ("y", Array(33, BIT))), Array(34, BIT), _ripple)
#: one gate of 5000 inputs: inputs are unbounded
BWide = _c("CBWide", (("x", Array(5000, BIT)),), BIT, lambda B, S, x: B.call(BL.and_gate(5000), x))
BOOLEAN_CALLS = [BXor32, BPair, BConsts, Ripple, BWide]

#: a 64-bit computing gate
Add64 = _c("CAdd64", (("a", Array(2, V64)),), V64, lambda B, S, a: B.call(TAdd64, a[0], a[1]))


def _const7(B, S, x):
    return tuple_of(B.call(I32Add, x[0], x[1]), B.call(const(32, 7)))


PROGRAMS = {
    "word": _top("CTopWord", WORD_CALLS),
    "boolean": _top("CTopBoolean", BOOLEAN_CALLS),
    "primitive-calls": Program(_c("CTopPrims", (("x", Array(2, V32)),), Tuple(V32, V32), _const7)),
    "too-wide": _top("CTopAdd64", [Add64]),
    "root-batch": Program(_c("CTopBatch", (("rows", Array(2, Array(3, V32))),), Array(2, V32), lambda B, S, rows: B.batch(Max3, rows, axes=(0,)))),
    "root-scan": Program(_c("CTopScan", (("xs", Array(3, V32)), ("c", V32)), Tuple(V32, Array(3, V32)), lambda B, S, xs, c: B.scan(Step, c, xs=[xs]))),
}
#: (program, the X it is evaluated at)
EVALUATED = [("word", [32, 64]), ("boolean", [32, 1]), ("primitive-calls", [32])]
#: (label, program, X): verify refuses the program's own object
REFUSED = [("a 32-bit gate at X = 16", "word", 16), ("a 64-bit gate at X = 32", "too-wide", 32), ("a root batch", "root-batch", 32),
           ("a root scan", "root-scan", 32)]
#: (label, program, definition (None: the root), node, argument, the reference sequence it is given): not topological
ORDER = [("a gate of a Call reads itself", "word", "QDead_v1", 2, 0, {"affine": ["n", 2, 0, 1]})]


def _obj(q=None):
    return {"format": PO.FORMAT, "program": "0" * 128, "query": q or PO.call_query()}


#: (label, object): validate refuses it
MALFORMED = [
    ("a W", _obj({"name": "Q_call", "version": 1, "params": {"X": 32, "W": 32}})),
    ("no X", _obj({"name": "Q_call", "version": 1, "params": {}})),
    ("X = 0", _obj({"name": "Q_call", "version": 1, "params": {"X": 0}})),
    ("X past 2^16", _obj({"name": "Q_call", "version": 1, "params": {"X": 65537}})),
    ("a float X", _obj({"name": "Q_call", "version": 1, "params": {"X": 32.0}})),
    ("version 2", _obj({"name": "Q_call", "version": 2, "params": {"X": 32}})),
]


def _sha(x) -> str:
    return hashlib.sha256(codec.canonical_json(x)).hexdigest()


def gates_dump(G: cut.CallGates) -> dict:
    return {"n": G.n, "inputs": G.inputs, "keep": G.keep, "width": G.width, "prim": G.prim, "last": G.last, "outputs": G.outputs,
            "recomputed": [list(x) for x in G.recomputed], "reads": len(G.reads), "reads_sha256": _sha([list(x) for x in G.reads])}


def _codes(v) -> dict:
    out = {"ok": v.ok, "codes": list(v.codes), "cut": [[f["call"], f["codes"]] for f in v.detail.get("cut", {}).get("gates", [])]}
    if "recomputed_across" in v.detail:
        out["recomputed_across"] = v.detail["recomputed_across"]
    if "query-inapplicable" in v.detail:
        out["inapplicable"] = v.detail["query-inapplicable"]["gates"]
    return out


def evaluation(D, X: int, gates: bool = True) -> dict:
    """The evaluation at X; `gates`: with each Call's `CallGates` and `locate` per unit (only a program's first X lists them: the
    gates do not depend on X, and `locate` is the units' order)."""
    q = PO.call_query(X)
    obj = PO.build(D, q)
    part = PO.evaluate(D, q)
    calls = []
    for p, (k, c) in enumerate(part.calls):
        G = part.graph(p)
        com = part.committed(p)
        r = part.check(p)
        e = {"call": k, "definition": c.fn.id, "units": [[lo, hi, [list(s) for s in path]] for lo, hi, path in c.units],
             "owner": part.owner(p), "committed": com, "committed_bits": sum(G.width[G.keep[g]] for g in com),
             "redundant_gates": r.detail["redundant_gates"]}
        if gates:
            e["gates"] = gates_dump(G)
        calls.append(e)
    out = {"X": X, "object": obj, "object_digest": PO.digest(obj), "population": part.population, "owners_digest": PO.owners_digest(part),
           "calls": calls, "verify": _codes(PO.verify(obj, D))}
    if gates:
        out["locate"] = [[i, list(part.locate(i)[:3]), part.locate(i)[3], list(part.gates(i))] for i in range(part.population)]
    return out


def served_cases(D) -> list:
    """`verify`'s held-to-derived check of what serving commits, on the word program (Dead's dead gate, Consts' committed constant-derived
    gate)."""
    q = PO.call_query()
    obj = PO.build(D, q)
    part = PO.evaluate(D, q)
    full = {k: part.committed(p) for p, (k, _c) in enumerate(part.calls)}
    by = {c.fn.id: (p, k) for p, (k, c) in enumerate(part.calls)}
    pc, kc = by[Consts.id]
    G = part.graph(pc)
    interior = [g for g in full[kc] if g not in G.outputs][0]
    pd, kd = by[Dead.id]
    Gd = part.graph(pd)
    dead = next(i for i, j in enumerate(Gd.keep) if Gd.last[j] == -1)
    cases = [("the derived sets", full), ("nothing", {}), ("a gate that is not an integer", full | {kc: [str(x) for x in full[kc]]}),
             ("a gate listed twice", full | {kc: full[kc] + full[kc][:1]}), ("one Call left out", {j: v for j, v in full.items() if j != kc}),
             ("a key that is no Call", full | {999: [0]}), ("a constant-derived commit left out", full | {kc: [x for x in full[kc] if x != interior]}),
             ("an output left out", full | {kc: [x for x in full[kc] if x != G.outputs[-1]]}),
             ("a dead gate served", full | {kd: sorted(full[kd] + [dead])})]
    return [[label, {str(j): v for j, v in s.items()}, _codes(PO.verify(obj, D, served=s))] for label, s in cases]


def build() -> dict:
    reg = registry()
    decoded = {k: codec.decode_program(json.loads(_canon(p)), reg) for k, p in PROGRAMS.items()}
    refused = [[label, name, X, _codes(PO.verify(PO.build(decoded[name], PO.call_query(X)), decoded[name]))] for label, name, X in REFUSED]
    other = copy.deepcopy(PO.build(decoded["word"], PO.call_query()))
    other["program"] = "f" * 128
    refused.append(["another program", "word", 32, _codes(PO.verify(other, decoded["word"]))])
    order = []
    for label, name, key, node, arg, refs in ORDER:
        d = json.loads(_canon(PROGRAMS[name]))
        d["definitions"][key or d["root"]]["body"]["nodes"][node]["args"][arg]["refs"] = refs
        D = codec.decode_program(d, reg)
        v = PO.verify(PO.build(D, PO.call_query()), D)
        order.append([label, name, key, node, arg, refs, {"codes": list(v.codes), "detail": v.detail.get("query-inapplicable", {}).get("gates")}])
    return {
        "about": "Pinned vectors of Q_call v1 (verity.ml.circuits.cut 'Q_call v1', verity.ml.circuits.partition_object, verity/ml/circuits/PROTOCOL.md §11), checked "
                 "by verity/ml/circuits/tests/test_qcall_vectors.py; written by it with --write.  Programs are descriptor bytes (TWide_v1, TAdd64_v1 and "
                 "TLo32_v1 are test primitives).  Gate indices in a Call are local (layout order, 0 = the Call's first gate); owner, "
                 "committed, outputs and recomputed number the computing gates compactly (keep).",
        "constants": {"x": cut.CALL_X, "param_max": PO.PARAM_MAX, "query": PO.call_query()},
        "descriptors": {k: _canon(p) for k, p in PROGRAMS.items()},
        "evaluations": [[name, [evaluation(decoded[name], X, i == 0) for i, X in enumerate(xs)]] for name, xs in EVALUATED],
        "verify_refusals": refused,
        "order_refusals": order,
        "served": served_cases(decoded["word"]),
        "object_refusals": [[label, obj, list(PO.validate(obj).codes)] for label, obj in MALFORMED],
    }


VECTORS = json.loads(PATH.read_text()) if PATH.exists() else {}


@pytest.fixture(scope="module")
def live() -> dict:
    return build()


def test_the_constants_of_the_version(live):
    assert live["constants"] == VECTORS["constants"] == {"x": 32, "param_max": 65536, "query": {"name": "Q_call", "version": 1, "params": {"X": 32}}}


def test_the_descriptors(live):
    assert live["descriptors"] == VECTORS["descriptors"]


@pytest.mark.parametrize("n", range(len(EVALUATED)), ids=[e[0] for e in EVALUATED])
def test_the_evaluations(live, n):
    assert live["evaluations"][n] == VECTORS["evaluations"][n]
    for e in VECTORS["evaluations"][n][1]:
        assert e["verify"]["ok"]


def _calls(name: str, X: int = 32) -> dict:
    ev = next(e for e in dict(VECTORS["evaluations"])[name] if e["X"] == X)
    return {c["definition"]: c for c in ev["calls"]}


def _paths(c: dict) -> list:
    return [p for _lo, _hi, p in c["units"]]


def test_every_computing_gate_is_in_one_unit_and_inputs_in_none():
    for name, runs in VECTORS["evaluations"]:
        gates = {c["call"]: c["gates"] for c in runs[0]["calls"]}
        for ev in runs:
            for c in ev["calls"]:
                g = gates[c["call"]]
                covered = [j for lo, hi, _p in c["units"] for j in range(lo, hi) if j in set(g["keep"])]
                assert sorted(covered) == g["keep"], (name, c["definition"])
                assert all(g["prim"][j] for j in g["keep"]) and len(c["owner"]) == len(g["keep"])
                for lo, hi, _p in c["units"]:                                                   # w_out <= X, and each unit computes
                    out = [j for j in range(lo, hi) if j in set(g["keep"]) and g["last"][j] >= hi]
                    assert sum(g["width"][j] for j in out) <= ev["X"] and any(j in set(g["keep"]) for j in range(lo, hi))
                assert c["committed_bits"] == sum(g["width"][g["keep"][x]] for x in c["committed"])


@pytest.mark.parametrize("edit,why", [(lambda us: us[:-1], "in no unit"),
                                      (lambda us: us + [us[-1]], "is in units")])
def test_evaluate_call_refuses_a_cut_that_misses_or_repeats_a_gate(monkeypatch, edit, why):
    whole = cut.call_units
    monkeypatch.setattr(cut, "call_units", lambda G, X: edit(whole(G, X)))
    with pytest.raises(ValueError, match=why):
        cut.evaluate_call(Ripple, 32)


def test_units_are_numbered_by_call_then_layout():
    for name, runs in VECTORS["evaluations"]:
        ev = runs[0]
        units = [(c["call"], u, lo, hi, path) for c in ev["calls"] for u, (lo, hi, path) in enumerate(c["units"])]
        assert len(units) == ev["population"] == len(ev["locate"])
        for (i, (p, k, u), path, (glo, ghi)), (k2, u2, lo, hi, path2) in zip(ev["locate"], units, strict=True):
            assert (k, u, path) == (k2, u2, path2) and ghi - glo == hi - lo, (name, i)
        for c in ev["calls"]:
            assert [lo for lo, _hi, _p in c["units"]] == sorted(lo for lo, _hi, _p in c["units"])


def test_the_rules_are_exercised():
    w = _calls("word")
    assert _paths(w["QMax3_v1"]) == [[]]                                                        # a Call that fits
    assert _paths(w["CNest_v1"]) == [[[0], [0]], [[0], [1]], [[1]]]                              # sub-Calls, two levels down
    assert _paths(w["CRows_v1"]) == [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]                     # batch members
    assert _paths(w["CScan_v1"]) == [[[0, 0]], [[0, 1]], [[0, 2]]]                               # scan iterations
    assert w["CScan_v1"]["committed"] == [0, 1, 2]                                               # each carry: read on, and returned
    keep = w["CScanKeep_v1"]
    assert _paths(keep) == [[[0, 0]], [[0, 1]], [[0, 2]]] and keep["gates"]["outputs"] == [0, 1, 2]   # a passed-through carry: nothing
    p = w["CPass_v1"]
    assert _paths(p) == [[[0]], [[2]]] and p["committed_bits"] == 64                             # pass-throughs: no unit, no bits
    cs = w["CConsts_v1"]
    g = cs["gates"]
    assert g["prim"][0].startswith("Const32") and 0 not in g["keep"] and _paths(cs) == [[[1]], [[2]], [[3]], [[4]]]
    assert cs["committed"] == [0, 1, 2, 3] and g["last"][1] == 4                                 # the constant-derived gate, committed
    assert _paths(w["CLo_v1"]) == [[]]                                                          # 64-bit input and constant: fine
    assert w["QTwiceSep_v1"]["gates"]["recomputed"] == [[1, 0]] and len(w["QTwiceSep_v1"]["units"]) == 2
    assert w["CRedundant_v1"]["redundant_gates"] == 1 and len(w["CRedundant_v1"]["units"]) == 1
    assert set(cut.WIRING) <= set(w["QWiring_v1"]["gates"]["prim"][j] for j in w["QWiring_v1"]["gates"]["keep"])   # former wiring computes
    dead = w["QDead_v1"]
    assert -1 in [dead["gates"]["last"][j] for j in dead["gates"]["keep"]] and len(dead["units"]) == 5
    assert w["QTWideTwice_v1"]["gates"]["recomputed"] == [[2, 1]] and _paths(w["QTWideTwice_v1"]) == [[]]
    b = _calls("boolean")
    assert _paths(b["CBXor32_v1"]) == [[]] and _paths(b["CBPair_v1"]) == [[[0]], [[1]]]
    bc = b["CBConsts_v1"]
    assert _paths(bc) == [[[1]], [[2]], [[3]]] and bc["committed"][0] == 0 and bc["gates"]["keep"][0] == 1   # Not(0), committed
    assert len(b["CBRipple_v1"]["units"]) == 33 and b["CBRipple_v1"]["committed_bits"] == 34 + 32
    assert _paths(b["CBWide_v1"]) == [[]]


def test_x_matters():
    """At X = 64 a 64-bit node is one unit (Nest's Pair, Rows' batches, TwiceSep's two sums: a recompute within one unit, so none
    across); at X = 1 a Boolean Call cuts down to its gates where its pieces return more than one bit."""
    w64, b1 = _calls("word", 64), _calls("boolean", 1)
    assert _paths(w64["CNest_v1"]) == [[[0]], [[1]]] and _paths(w64["CRows_v1"]) == [[[0]], [[1]]] and _paths(w64["QTwiceSep_v1"]) == [[]]
    assert dict(VECTORS["evaluations"])["word"][1]["verify"]["recomputed_across"] == {"n": 0, "gates": []}
    assert len(b1["CBXor32_v1"]["units"]) == 32 and len(b1["CBRipple_v1"]["units"]) == 33 * 5 and _paths(b1["CBWide_v1"]) == [[]]


def test_verify_refuses(live):
    assert live["verify_refusals"] == VECTORS["verify_refusals"]
    got = {label: v for label, _n, _x, v in VECTORS["verify_refusals"]}
    assert all(not v["ok"] for v in got.values())
    assert got["a 32-bit gate at X = 16"]["inapplicable"][0]["why"] == "gate-too-wide"
    assert got["a 64-bit gate at X = 32"]["inapplicable"][0] | {} == {"call": 1, "definition": "CAdd64_v1", "gate": 0, "primitive": "TAdd64_v1",
                                                                         "width": 64, "X": 32, "why": "gate-too-wide"}
    assert got["a root batch"]["inapplicable"][0]["why"] == got["a root scan"]["inapplicable"][0]["why"] == "root-form"
    assert got["another program"]["codes"] == ["program"]


def test_every_query_refuses_a_program_that_is_not_topological(live):
    assert live["order_refusals"] == VECTORS["order_refusals"]
    for *_x, v in VECTORS["order_refusals"]:
        assert v["codes"] == ["query-inapplicable"] and v["detail"][0]["why"] == "not topological"


def test_served(live):
    assert live["served"] == VECTORS["served"]
    got = {label: (v["codes"], [c for _k, cs in v["cut"] for c in cs]) for label, _s, v in VECTORS["served"]}
    assert got["the derived sets"] == ([], []) and got["nothing"][0] == ["served-missing"]
    assert got["a gate that is not an integer"][0] == got["a gate listed twice"][0] == ["served-malformed"]
    assert got["one Call left out"][0] == ["served-missing"] and got["a key that is no Call"][0] == ["served-unknown"]
    assert "read-uncommitted" in got["a constant-derived commit left out"][1] and "output-uncommitted" in got["an output left out"][1]
    assert got["a dead gate served"][1] == ["committed-unread"]


def test_recomputes_are_reported_never_refused():
    v = dict(VECTORS["evaluations"])["word"][0]["verify"]
    assert v["ok"] and v["recomputed_across"]["n"] == sum(r["n"] for r in v["recomputed_across"]["gates"]) > 0
    calls = {c["call"] for c in v["recomputed_across"]["gates"]}
    assert _calls("word")["QTwiceSep_v1"]["call"] in calls and _calls("word")["CRedundant_v1"]["call"] not in calls


def test_validate_refuses(live):
    assert live["object_refusals"] == VECTORS["object_refusals"]
    assert [codes for _l, _o, codes in VECTORS["object_refusals"]] == [["query-params"]] * 5 + [["query-unknown"]]


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
