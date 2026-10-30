"""`Q_word` v1's behaviour pinned beyond the six-Call vector (`qword_vectors.json`): the version's constants as data, programs that
exercise every step of `verity.ml.circuits.cut`'s algorithm (X packing narrow outputs, wide gates, returned wiring inputs, a source read
twice, the dead-unit second pass, the separable branches, primitive Calls, all three wiring primitives), every refusal of `validate`
and `verify`, and the seeded random corpus of the Lean port's conformance run (`verity/ml/flock/verifier/qword_agree.py`, same
generator and seed), as digests of its cases and of core's answers.  Programs are descriptor bytes; the evaluation is from the bytes.
`v2`: `Q_word` v2 on the same programs (v1's evaluation; a value computed in two units reported, not refused).
`python verity/ml/circuits/tests/test_qword_vectors.py --write` rewrites the file."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.ml.circuits import codec, cut, partition_object as PO
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, Registry, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.prims import Bf16ToF32, F32ToBf16Rn, const
from verity.ml.circuits.ops.scalar import BitAnd, Bf16GtStrict, F32BitsShl23, F32Fabs, F32Fmaxf, I32Add

PATH = Path(__file__).parent / "qword_vectors.json"
BIT, BF16, V32, V64 = Value(1), Value(16), Value(32), Value(64)

TWide = PrimitiveDefinition("TWide", 1, (("x", Array(cut.WIDE_FAN_IN, V32)), ("a", V32)), V32, lambda *a: sum(a) & 0xFFFFFFFF, register=False)
TNarrow = PrimitiveDefinition("TNarrow", 1, (("x", Array(4, V32)), ("a", V32)), V32, lambda *a: sum(a) & 0xFFFFFFFF, register=False)
TAdd64 = PrimitiveDefinition("TAdd64", 1, (("a", V64), ("b", V64)), V64, lambda a, b: (a + b) % (1 << 64), register=False)


def registry() -> Registry:
    r = Registry()
    r.defs.update(REGISTRY.defs)
    for p in (TWide, TNarrow, TAdd64):
        r.add(p)
    return r


def _c(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


def _max3(B, S, x):
    return B.call(F32Fmaxf, B.call(F32Fmaxf, x[0], x[1]), B.call(F32Fabs, x[2]))


Max3 = _c("QMax3", (("x", Array(3, V32)),), V32, _max3)


def _narrow(B, S, a, b):
    g = [B.call(Bf16GtStrict, a[i], b[i]) for i in range(4)]                                        # four 1-bit outputs
    widen = lambda v: B.call(Bf16ToF32, v)                                                         # noqa: E731
    h0 = B.call(F32ToBf16Rn, B.call(F32Fmaxf, widen(a[0]), widen(b[0])))                           # 16-bit outputs over 32-bit interiors
    h1 = B.call(F32ToBf16Rn, B.call(F32Fmaxf, widen(a[1]), widen(b[1])))
    h2 = B.call(F32ToBf16Rn, B.call(F32Fmaxf, widen(a[2]), widen(a[2])))                           # one input read through two wiring gates
    g4 = B.call(BitAnd, g[0], g[0])                                                                # one source read twice
    return tuple_of(*g, h0, h1, h2, g4)


#: X packs outputs: 4 x 1 bit, 3 x 16 bits, 1 bit
Narrow = _c("QNarrow", (("a", Array(4, BF16)), ("b", Array(4, BF16))), Tuple(*[BIT] * 4, BF16, BF16, BF16, BIT), _narrow)


def _wiring(B, S, x, h):
    return tuple_of(B.call(F32Fabs, x[0]), B.call(Bf16ToF32, h), B.call(F32BitsShl23, x[1]),          # wiring of inputs, returned
                    B.call(I32Add, B.call(F32Fabs, x[2]), B.call(F32BitsShl23, x[2])),               # a computing gate reading through wiring
                    B.call(F32Fabs, B.call(I32Add, x[0], x[1])))                                    # wiring of a computing gate, returned


#: all three wiring primitives; returned inputs
Wiring = _c("QWiring", (("x", Array(3, V32)), ("h", BF16)), Tuple(V32, V32, V32, V32, V32), _wiring)


def _dead(B, S, x):
    a = B.call(I32Add, x[0], x[1])
    b = B.call(I32Add, x[2], x[3])
    B.call(I32Add, a, b)                                                                           # dead, reading two output units' values
    return tuple_of(B.call(I32Add, a, x[0]), B.call(I32Add, b, x[2]))


#: the dead unit drops, its producers are promoted, and the pass repeats
Dead = _c("QDead", (("x", Array(4, V32)),), Tuple(V32, V32), _dead)


def _twice_wide(prim):
    L = prim.params[0][1].n

    def body(B, S, x, s):
        a = B.call(I32Add, s, s)
        return B.call(I32Add, B.call(prim, x, a), B.call(prim, x, a))

    return _c(f"Q{prim.name}Twice", (("x", Array(L, V32)), ("s", V32)), V32, body)


#: two identical wide gates (never a recompute; outside operands unlisted) beside the same over a 4-leaf primitive
WideTwice, NarrowTwice = _twice_wide(TWide), _twice_wide(TNarrow)

Step = _c("QStep", (("c", V32), ("x", V32)), Tuple(V32, V32), lambda B, S, c, x: (lambda s: tuple_of(s, s))(B.call(I32Add, c, x)))
#: separable with a structure-primitive node: its members are no units, and locate skips the node
SepMixed = _c("QSepMixed", (("x", Array(2, V32)), ("rows", Array(2, Array(3, V32)))), Tuple(Array(2, V32), Array(2, V32)),
              lambda B, S, x, rows: tuple_of(B.batch(F32Fabs, x, axes=(0,)), B.batch(Max3, rows, axes=(0,))))
#: a scan node: not separable
SepScan = _c("QSepScan", (("xs", Array(3, V32)), ("c", V32)), Tuple(V32, Array(3, V32)), lambda B, S, xs, c: B.scan(Step, c, xs=[xs]))
#: returns only part of what its node produces: not separable
SepPartial = _c("QSepPartial", (("rows", Array(2, Array(3, V32))),), V32, lambda B, S, rows: B.batch(Max3, rows, axes=(0,))[0])
#: a value two members compute: a recompute across units
TwiceSep = _c("QTwiceSep", (("x", Array(2, V32)),), Tuple(V32, V32),
              lambda B, S, x: tuple_of(B.call(I32Add, x[0], x[1]), B.call(I32Add, x[0], x[1])))
#: a 64-bit output: too wide at W = 32
Wide64 = _c("QWide64", (("a", Array(2, V64)),), V64, lambda B, S, a: B.call(TAdd64, a[0], a[1]))


def _wide_rows(B, S, xr, xc, s):
    a = B.call(I32Add, s, s)
    w0 = B.batch(TWide, xr, a, axes=(0, None))                                                    # wide members mapped over axis 0
    w1 = B.batch(TWide, xc, a, axes=(1, None))                                                    # and over axis 1
    return tuple_of(B.call(I32Add, w0[0], w1[1]), w0[1], w1[0])


#: wide gates inside batch nodes (member_refs' slice and take branches), in a flat body
WideRows = _c("QWideRows", (("xr", Array(2, Array(cut.WIDE_FAN_IN, V32))), ("xc", Array(cut.WIDE_FAN_IN, Array(2, V32))), ("s", V32)),
              Tuple(V32, V32, V32), _wide_rows)


def _top(name, calls):
    """A program calling each Definition (or primitive) once on its own parameters."""
    sigs = [fn.signature() if hasattr(fn, "body") else (fn.params, fn.ret) for fn in calls]
    params = tuple((f"{i}.{n}", t) for i, (ps, _r) in enumerate(sigs) for n, t in ps)

    def body(B, S, *args):
        outs, at = [], 0
        for fn, (ps, _r) in zip(calls, sigs):
            outs.append(B.call(fn, *args[at:at + len(ps)]))
            at += len(ps)
        return tuple_of(*outs)

    return Program(_c(name, params, Tuple(*[r for _p, r in sigs]), body))


def _const7(B, S, x):
    return tuple_of(B.call(I32Add, x[0], x[1]), B.call(F32Fabs, x[2]), B.call(const(32, 7)))


PROGRAMS = {
    "narrow": _top("QTopNarrow", [Narrow]),
    "wiring-and-dead": _top("QTopWiring", [Wiring, Dead]),
    "wide": _top("QTopWide", [WideTwice, NarrowTwice]),
    "wide-batched": _top("QTopWideRows", [WideRows]),
    "separable": _top("QTopSep", [SepMixed, SepScan, SepPartial]),
    "primitive-calls": Program(_c("QTopPrims", (("x", Array(3, V32)),), Tuple(V32, V32, V32), _const7)),
    "recompute": _top("QTopRecompute", [TwiceSep]),
    "too-wide": _top("QTopWide64", [Wide64]),
    "root-batch": Program(_c("QTopBatch", (("rows", Array(2, Array(3, V32))),), Array(2, V32), lambda B, S, rows: B.batch(Max3, rows, axes=(0,)))),
    "root-scan": Program(_c("QTopScan", (("xs", Array(3, V32)), ("c", V32)), Tuple(V32, Array(3, V32)),
                            lambda B, S, xs, c: B.scan(Step, c, xs=[xs]))),
}
#: (program, the queries it is evaluated at)
EVALUATED = [("narrow", [(16, 32), (32, 32), (64, 64)]), ("wiring-and-dead", [(16, 32)]), ("wide", [(16, 32)]), ("separable", [(16, 32)]),
             ("primitive-calls", [(16, 32)]), ("wide-batched", [(16, 32)])]
#: (label, program, definition (None: the root), node, argument, the reference sequence it is given): not topological
ORDER = [
    ("a gate of a Call reads itself", "wiring-and-dead", "QDead_v1", 2, 0, {"affine": ["n", 2, 0, 1]}),
    ("a root Call reads its own output", "wiring-and-dead", None, 4, 0, {"concat": [{"affine": ["n", 4, 0, 2]}, {"affine": ["n", 2, 0, 2]}]}),
    ("a root Call reads a later Call", "wiring-and-dead", None, 3, 0, {"concat": [{"affine": ["n", 4, 0, 2]}, {"affine": ["n", 0, 2, 1]}]}),
]
#: (label, program, X, W): verify refuses the program's own object
REFUSED = [("a value two units compute", "recompute", 16, 32), ("a 64-bit output at W = 32", "too-wide", 16, 32),
           ("32-bit outputs at W = 16", "wiring-and-dead", 16, 16), ("a root batch", "root-batch", 16, 32), ("a root scan", "root-scan", 16, 32)]
#: (program, the queries it is evaluated at) under Q_word v2: v1's evaluations, and the program v1 refuses only for `gate-recomputed`
V2_EVALUATED = EVALUATED + [("recompute", [(16, 32)])]


def _obj(q=None):
    return {"format": PO.FORMAT, "program": "0" * 128, "query": q or PO.query()}


#: (label, object): validate refuses it
MALFORMED = [
    ("not an object", []),
    ("an extra key", dict(_obj(), owners=[])),
    ("a non-string key", {**_obj(), 1: 2}),
    ("another format", dict(_obj(), format="verity/partition/v0")),
    ("a program that is not SHA-512 hex", dict(_obj(), program="F" * 128)),
    ("a query without params", dict(_obj(), query={"name": "Q_word", "version": 1})),
    ("an unknown version", _obj({"name": "Q_word", "version": 3, "params": {"X": 16, "W": 32}})),
    ("a missing parameter", _obj({"name": "Q_word", "version": 1, "params": {"X": 16}})),
    ("X = 0", _obj({"name": "Q_word", "version": 1, "params": {"X": 0, "W": 32}})),
    ("X past 2^16", _obj({"name": "Q_word", "version": 1, "params": {"X": 65537, "W": 32}})),
    ("a float X", _obj({"name": "Q_word", "version": 1, "params": {"X": 16.0, "W": 32}})),
    ("a boolean W", _obj({"name": "Q_word", "version": 1, "params": {"X": 16, "W": True}})),
]
#: (label, object): validate refuses it, its X and W checked as v1's
MALFORMED_V2 = [
    ("v2 without W", _obj({"name": "Q_word", "version": 2, "params": {"X": 16}})),
    ("v2 with X = 0", _obj({"name": "Q_word", "version": 2, "params": {"X": 0, "W": 32}})),
    ("v2 with an EXTRAS", _obj({"name": "Q_word", "version": 2, "params": {"X": 16, "W": 32, "EXTRAS": 0}})),
]


def graph_dump(g: cut.CallGraph) -> dict:
    return {"total": g.total, "structure": g.free, "n": g.n, "keep": g.keep.tolist(), "width": g.width.tolist(),
            "prim": [g.prim_names[x] for x in g.prim.tolist()], "external": list(g.external), "src": g.src.tolist(), "dst": g.dst.tolist(),
            "input_ids": list(g.input_ids), "input_reads": [list(x) for x in g.input_reads], "outputs": list(g.outputs),
            "returned_inputs": list(g.returned_inputs), "recomputed": [list(x) for x in g.recomputed]}


def _canon(program) -> str:
    return codec.canonical_json({k: v for k, v in codec.encode_program(program).items() if k != "annotations"}).decode()


def _codes(v) -> dict:
    out = {"ok": v.ok, "codes": list(v.codes), "cut": [[f["call"], f["codes"]] for f in v.detail.get("cut", {}).get("gates", [])]}
    if "recomputed_across" in v.detail:                                                 # Q_word v2's report
        out["recomputed_across"] = v.detail["recomputed_across"]
    return out


def evaluation(D, X: int, W: int, version: int = 1) -> dict:
    q = PO.query(X, W, version)
    obj = PO.build(D, q)
    part = PO.evaluate(D, q)
    calls = []
    for p, (k, c) in enumerate(part.calls):
        e = {"call": k, "definition": c.fn.id, "graph": graph_dump(part.graph(p)), "owner": part.owner(p), "units": c.units,
             "committed": part.committed(p)}
        if c.parts is None:
            e["kinds"] = c.kinds
        calls.append(e)
    return {"X": X, "W": W, "object": obj, "object_digest": PO.digest(obj), "population": part.population, "owners_digest": PO.owners_digest(part),
            "calls": calls, "locate": [[i, list(part.locate(i)[:3]), [list(x) for x in part.locate(i)[3]]] for i in range(part.population)],
            "verify": _codes(PO.verify(obj, D))}


def served_cases(D, version: int = 1) -> list:
    """`verify`'s held-to-derived check of what serving commits, on the wiring-and-dead program (its last Call is QDead)."""
    obj = PO.build(D, PO.query(version=version))
    part = PO.evaluate(D)
    full = {k: part.committed(p) for p, (k, _c) in enumerate(part.calls)}
    p = len(part.calls) - 1
    k, c = part.calls[p]
    g, own = part.graph(p), part.owner(p)
    dead_gate = own.index(c.kinds.index("dead"))
    interior = [x for x in full[k] if x not in g.outputs][0]
    cases = [("the derived sets", full), ("nothing", {}), ("a gate that is not an integer", full | {k: [str(x) for x in full[k]]}),
             ("a gate listed twice", full | {k: full[k] + full[k][:1]}), ("one Call left out", {j: v for j, v in full.items() if j != k}),
             ("a key that is no Call", full | {999: [0]}), ("a committed value left out", full | {k: [x for x in full[k] if x != interior]}),
             ("an output left out", full | {k: [x for x in full[k] if x != g.outputs[-1]]}),
             ("a dead gate served", full | {k: sorted(full[k] + [dead_gate])})]
    return [[label, {str(j): v for j, v in s.items()}, _codes(PO.verify(obj, D, served=s))] for label, s in cases]


# -- the Lean port's random corpus (verity/ml/flock/verifier/qword_agree.py at 34b6fb0e: the same generator, seed and answers) ---------

def _graph_json(rng, n):
    edges = [(p, q) for q in range(n) for p in range(q) if rng.random() < 2.0 / max(n, 1)]
    rng.shuffle(edges)
    outs = [rng.randrange(n) for _ in range(rng.randint(1, min(4, n)))] if n else []
    return {"n": n, "src": [p for p, _ in edges], "dst": [q for _, q in edges], "width": [rng.choice([1, 8, 16, 32]) for _ in range(n)],
            "outputs": outs, "inputs": 0, "input_reads": [], "returned_inputs": [], "total": n, "free": 0, "recomputed": []}


def _core_graph(g):
    return SimpleNamespace(n=g["n"], src=array("q", g["src"]), dst=array("q", g["dst"]), width=array("q", g["width"]), outputs=list(g["outputs"]))


def _random_def(rng, max_gates, depth=0):
    r = rng.random()
    if r < 0.2:
        return {"prim": {"params": rng.random() < 0.8, "wiring": rng.random() < 0.2}}
    if r < 0.45 and depth < 2:
        return {"separable": [[_random_def(rng, max_gates, depth + 1), rng.randint(1, 3)] for _ in range(rng.randint(1, 3))]}
    return {"flat": _graph_json(rng, rng.randint(1, max_gates))}


def _core_cut(d, X):
    if "prim" in d:
        p = d["prim"]
        return cut.DefinitionCut(None, [], []) if (not p["params"] or p["wiring"]) else cut.DefinitionCut(None, [0], ["output"])
    if "separable" in d:
        parts = []
        for sub, m in d["separable"]:
            c = _core_cut(sub, X)
            parts.append((c if c.gates else None, m))
        return cut.DefinitionCut(None, parts=parts)
    own, kinds, _ = cut.cut_word(_core_graph(d["flat"]), X)
    return cut.DefinitionCut(None, own, kinds)


def _core_locate(calls, i):
    P = PO.Partition.__new__(PO.Partition)
    P.calls, P._start, t = calls, [], 0
    for _k, c in calls:
        P._start.append(t)
        t += c.units
    P.population = t
    p, k, u, path = P.locate(i)
    return [i, [p, k, u], [list(x) for x in path]]


def _expected(case):
    X, calls, out = case["X"], [], []
    for c in case["calls"]:
        dc = _core_cut(c["def"], X)
        calls.append((c["call"], dc))
        e = {"call": c["call"], "owner": dc.owner(), "units": dc.units}
        if "graph" in c:
            e["committed"] = cut.derived_committed(_core_graph(c["graph"]), dc.owner()) if len(dc.owner()) == c["graph"]["n"] else []
        if "flat" in c["def"]:
            own, kinds, head = cut.cut_word(_core_graph(c["def"]["flat"]), X)
            e["kinds"], e["head"] = kinds, head
        out.append(e)
    pop = sum(dc.units for _, dc in calls)
    return {"calls": out, "population": pop, "locate": [_core_locate(calls, i) for i in range(pop)]}


def _random_case(rng, max_gates):
    X = rng.choice([16, 16, 32, 8, 64])
    calls = []
    for k in range(rng.randint(1, 4)):
        d = _random_def(rng, max_gates)
        c = {"call": k + rng.randint(0, 2), "def": d}
        if "flat" in d:
            c["graph"] = d["flat"]
        calls.append(c)
    return {"X": X, "calls": calls}


CORPORA = [(20260927, 500, 12), (20260927, 3000, 40)]


def corpus(seed: int, n: int, max_gates: int) -> dict:
    rng = random.Random(seed)
    cases = [_random_case(rng, max_gates) for _ in range(n)]
    want = [_expected(c) for c in cases]
    sha = lambda x: hashlib.sha512(codec.canonical_json(x)).hexdigest()                      # noqa: E731
    return {"seed": seed, "n": n, "max_gates": max_gates, "cases_sha512": sha(cases), "expected_sha512": sha(want),
            "units": sum(w["population"] for w in want), "unit_kinds": sorted({k for w in want for c in w["calls"] for k in c.get("kinds", [])}),
            "first_case": cases[0], "first_expected": want[0]}


def build() -> dict:
    reg = registry()
    decoded = {k: codec.decode_program(json.loads(_canon(p)), reg) for k, p in PROGRAMS.items()}
    refused = []
    for label, name, X, W in REFUSED:
        D = decoded[name]
        refused.append([label, name, X, W, _codes(PO.verify(PO.build(D, PO.query(X, W)), D))])
    order = []
    for label, name, key, node, arg, refs in ORDER:
        d = json.loads(_canon(PROGRAMS[name]))
        d["definitions"][key or d["root"]]["body"]["nodes"][node]["args"][arg]["refs"] = refs
        D = codec.decode_program(d, reg)
        v = PO.verify(PO.build(D), D)
        order.append([label, name, key, node, arg, refs, {"codes": list(v.codes), "detail": v.detail.get("query-inapplicable", {}).get("gates")}])
    other = copy.deepcopy(PO.build(decoded["narrow"]))
    other["program"] = "f" * 128
    refused.append(["another program", "narrow", 16, 32, _codes(PO.verify(other, decoded["narrow"]))])
    return {
        "about": "Pinned vectors of Q_word v1 (verity.ml.circuits.cut, verity.ml.circuits.partition_object) beyond partition_vectors.json, checked by "
                 "verity/ml/circuits/tests/test_qword_vectors.py; written by it with --write.  Programs are descriptor bytes (TWide_v1, TNarrow_v1 and "
                 "TAdd64_v1 are test primitives: n leaves of Value<32> or two Value<64>, one output).",
        "constants": {"wiring": sorted(cut.WIRING), "wide_fan_in": cut.WIDE_FAN_IN, "width_rule": list(cut.WIDTH_RULE), "param_max": PO.PARAM_MAX},
        "descriptors": {k: _canon(p) for k, p in PROGRAMS.items()},
        "evaluations": [[name, [evaluation(decoded[name], X, W) for X, W in qs]] for name, qs in EVALUATED],
        "verify_refusals": refused,
        "order_refusals": order,
        "served": served_cases(decoded["wiring-and-dead"]),
        "object_refusals": [[label, repr(obj) if not isinstance(obj, (dict, list)) or 1 in obj else obj, list(PO.validate(obj).codes)]
                            for label, obj in MALFORMED],
        "corpora": [corpus(*c) for c in CORPORA],
        "v2": v2(decoded),
    }


def v2(decoded: dict) -> dict:
    """Q_word v2 on the same programs: the evaluations, every verify refusal of v1 and the served cases (v1's but for
    `gate-recomputed`, and the report), and the objects validate refuses."""
    refused = [[label, name, X, W, _codes(PO.verify(PO.build(decoded[name], PO.query(X, W, 2)), decoded[name]))]
               for label, name, X, W in REFUSED]
    other = copy.deepcopy(PO.build(decoded["narrow"], PO.query(version=2)))
    other["program"] = "f" * 128
    refused.append(["another program", "narrow", 16, 32, _codes(PO.verify(other, decoded["narrow"]))])
    return {"evaluations": [[name, [evaluation(decoded[name], X, W, 2) for X, W in qs]] for name, qs in V2_EVALUATED],
            "verify_refusals": refused, "served": served_cases(decoded["wiring-and-dead"], 2),
            "object_refusals": [[label, obj, list(PO.validate(obj).codes)] for label, obj in MALFORMED_V2]}


VECTORS = json.loads(PATH.read_text()) if PATH.exists() else {}


@pytest.fixture(scope="module")
def live() -> dict:
    return build()


def test_the_constants_of_the_version(live):
    assert live["constants"] == VECTORS["constants"] == {"wiring": ["Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"], "wide_fan_in": 4096,
                                                         "width_rule": [16, 32, 0], "param_max": 65536}


def test_the_descriptors(live):
    assert live["descriptors"] == VECTORS["descriptors"]


@pytest.mark.parametrize("n", range(len(EVALUATED)), ids=[e[0] for e in EVALUATED])
def test_the_evaluations(live, n):
    assert live["evaluations"][n] == VECTORS["evaluations"][n]
    for e in VECTORS["evaluations"][n][1]:
        assert e["verify"]["ok"]


def test_x_matters_and_the_paths_are_exercised():
    ev = dict(VECTORS["evaluations"])
    narrow = [e["calls"][0]["kinds"].count("output") for e in ev["narrow"]]
    assert narrow == [5, 3, 1]                                          # runs of the returned widths at X = 16, 32, 64
    n16 = ev["narrow"][0]["calls"][0]["graph"]
    assert list(zip(n16["src"], n16["dst"])).count((0, 10)) == 1        # BitAnd(g0, g0): one edge
    assert [q for i, q in n16["input_reads"] if q == 8] == [8]          # fmaxf over two wiring gates of one input: one read
    wiring, dead = ev["wiring-and-dead"][0]["calls"]
    assert wiring["graph"]["returned_inputs"] and set(wiring["graph"]["prim"]).isdisjoint(cut.WIRING)
    assert dead["kinds"] == ["output", "output", "committed", "committed", "dead"]
    wide, narrow_ctl = ev["wide"][0]["calls"]
    assert wide["graph"]["recomputed"] == [] and narrow_ctl["graph"]["recomputed"] == [[2, 1]] and wide["graph"]["external"] == [1, 1, 1, 0]
    sep = ev["separable"][0]["calls"]
    assert "kinds" not in sep[0] and "kinds" in sep[1] and "kinds" in sep[2]    # separable; a scan; partial returns
    assert all(path[0][0] == 1 for _i, (p, _k, _u), path in ev["separable"][0]["locate"] if p == 0)   # the structure node is skipped
    prims = ev["primitive-calls"][0]
    assert [c["units"] for c in prims["calls"]] == [1, 0, 0] and prims["calls"][0]["committed"] == [0]


def test_every_wiring_primitive_is_in_a_vector():
    D = json.loads(VECTORS["descriptors"]["wiring-and-dead"])
    assert cut.WIRING <= set(D["callees"])


@pytest.mark.parametrize("n", range(len(REFUSED) + 1), ids=[r[0] for r in REFUSED] + ["another program"])
def test_verify_refuses(live, n):
    assert live["verify_refusals"][n] == VECTORS["verify_refusals"][n] and not VECTORS["verify_refusals"][n][4]["ok"]


def test_every_query_refuses_a_program_that_is_not_topological(live):
    assert live["order_refusals"] == VECTORS["order_refusals"]
    for _l, _n, _k, _node, _a, _r, v in VECTORS["order_refusals"]:
        assert v["codes"] == ["query-inapplicable"] and v["detail"][0]["why"] == "not topological"
    reg = registry()
    for name, text in VECTORS["descriptors"].items():                   # every pinned program is topological
        assert PO.order_violation(codec.decode_program(json.loads(text), reg)) is None, name


def test_served(live):
    assert live["served"] == VECTORS["served"]
    got = {label: (v["codes"], [c for _k, cs in v["cut"] for c in cs]) for label, _s, v in VECTORS["served"]}
    assert got["the derived sets"] == ([], []) and got["nothing"][0] == ["served-missing"]
    assert got["a gate that is not an integer"][0] == got["a gate listed twice"][0] == ["served-malformed"]
    assert got["one Call left out"][0] == ["served-missing"] and got["a key that is no Call"][0] == ["served-unknown"]
    assert "read-uncommitted" in got["a committed value left out"][1] and "output-uncommitted" in got["an output left out"][1]
    assert got["a dead gate served"][1] == ["committed-unread"]


def test_validate_refuses(live):
    assert live["object_refusals"] == VECTORS["object_refusals"]
    assert all(codes for _l, _o, codes in VECTORS["object_refusals"])


@pytest.mark.parametrize("n", range(len(CORPORA)), ids=[f"seed{s}-n{n}-g{g}" for s, n, g in CORPORA])
def test_the_random_corpus(live, n):
    assert live["corpora"][n] == VECTORS["corpora"][n]


def test_the_v2_vectors(live):
    assert live["v2"] == VECTORS["v2"]


def _without_report(c: dict) -> dict:
    return {k: v for k, v in c.items() if k != "recomputed_across"}


def test_v2_evaluates_as_v1():
    """v2's Calls, graphs, owners, units, committed sets, `locate` and owners digest are v1's; only the object (its version) differs.
    `verify` accepts v1's recompute-free programs at both versions (v2 reporting no recompute across units), and the program v1 refuses
    for `gate-recomputed` at v2 alone, with the same cut."""
    v1 = dict(VECTORS["evaluations"])
    rec = codec.decode_program(json.loads(VECTORS["descriptors"]["recompute"]), registry())
    v1["recompute"] = [evaluation(rec, 16, 32)]
    for name, runs in VECTORS["v2"]["evaluations"]:
        for a, b in zip(v1[name], runs, strict=True):
            assert b["object"]["query"] == dict(a["object"]["query"], version=2) and b["object_digest"] != a["object_digest"]
            same = ("X", "W", "population", "owners_digest", "calls", "locate")
            assert {k: b[k] for k in same} == {k: a[k] for k in same}, name
            assert b["verify"]["ok"], name
            if name == "recompute":
                assert a["verify"]["cut"][0][1] == ["gate-recomputed"] and b["verify"]["recomputed_across"]["n"] == 1
            else:
                assert _without_report(b["verify"]) == a["verify"] and b["verify"]["recomputed_across"] == {"n": 0, "gates": []}


def test_v2_reports_what_v1_refuses_and_refuses_the_rest_as_v1():
    """The one change: a value two members compute (each copy committed: both are returned) is v1's `gate-recomputed` and v2's report;
    every other refusal, and the served checks, are v1's word for word."""
    v1 = {r[0]: r[4] for r in VECTORS["verify_refusals"]}
    v2 = {r[0]: r[4] for r in VECTORS["v2"]["verify_refusals"]}
    assert v1.keys() == v2.keys()
    k = v1["a value two units compute"]["cut"][0][0]
    assert v1["a value two units compute"]["cut"] == [[k, ["gate-recomputed"]]]
    assert v2["a value two units compute"] == {"ok": True, "codes": [], "cut": [],
                                               "recomputed_across": {"n": 1, "gates": [{"call": k, "n": 1, "gates": [[1, 0]]}]}}
    for label in v1.keys() - {"a value two units compute"}:
        assert _without_report(v2[label]) == v1[label] and not v2[label]["ok"], label
    served = {label: v for label, _s, v in VECTORS["served"]}
    for label, _s, v in VECTORS["v2"]["served"]:
        assert _without_report(v) == served[label], label
    assert all(codes == ["query-params"] for _l, _o, codes in VECTORS["v2"]["object_refusals"])


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
