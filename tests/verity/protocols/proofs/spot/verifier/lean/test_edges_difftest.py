"""The wires a Program's root states (`Spot.Edges.edgesOf`, the `edges` piece) against its twin,
`verity.commitments.periods.edges`, on descriptors the builder writes and on their root arguments' references rewritten:
both give the same wires or the same refusal, and the named cases give the wires or refusal the rule says."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.commitments.periods.edges import _lane_wire, _parts, edges

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean")
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

XOR16 = PrimitiveDefinition("CoreEdgesXor16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: a ^ b,
                            register=False)
XOR32 = PrimitiveDefinition("CoreEdgesXor32", 1, (("a", Value(32)), ("b", Value(32))), Value(32), lambda a, b: a ^ b,
                            register=False)
LOW16 = PrimitiveDefinition("CoreEdgesLow16", 1, (("a", Value(32)),), Value(16), lambda a: a & 0xFFFF, register=False)
ROW = Array(2, Value(16))
PAIR = Tuple(Value(16), Value(16))


def _arr(name: str):
    return bind(CompositeDefinition(name, 0, (), lambda S: ((("x", ROW),), ROW),
                                    lambda B, S, x: B.batch(XOR16, x, x, axes=(0, 0)), register=False))


def _tup(name: str):
    return bind(CompositeDefinition(name, 0, (), lambda S: ((("x", PAIR),), PAIR),
                                    lambda B, S, x: tuple_of(B.call(XOR16, x[0], x[1]), B.call(XOR16, x[1], x[0])),
                                    register=False))


def _toy(name: str, ty, a, b) -> tuple[dict, str, str]:
    """A root batching `a` (2 members) over the inputs, `b` (2) over `a`'s outputs and `a` (2) over `b`'s."""
    def body(B, S, x):
        return B.batch(a, B.batch(b, B.batch(a, x, axes=(0,)), axes=(0,)), axes=(0,))

    defn = bind(CompositeDefinition(name, 0, (), lambda S: ((("x", Array(2, ty)),), Array(2, ty)), body, register=False))
    return json.loads(D.canonical(codec.encode_program(Program(defn)))), codec.descriptor_id(a), codec.descriptor_id(b)


ARR = _toy("CoreEdgesArr", ROW, _arr("CoreEdgesA"), _arr("CoreEdgesB"))
TUP = _toy("CoreEdgesTup", PAIR, _tup("CoreEdgesTA"), _tup("CoreEdgesTB"))


def _nodes(desc: dict) -> list[dict]:
    return desc["definitions"][desc["root"]]["body"]["nodes"]


def _of(toy, cuts=None, b_refs=None) -> dict:
    """The piece's input on `toy`: its ranges by the root's nodes calling either template, in order, and its wiring
    with `cuts`; `b_refs(a, a2)` rewrites `b`'s argument from `a`'s and the later `a`'s node indices."""
    desc, ta, tb = copy.deepcopy(toy[0]), toy[1], toy[2]
    ids = desc["callees"]
    at = [k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] in (ta, tb)]
    if b_refs is not None:
        _nodes(desc)[at[1]]["args"][0]["refs"] = b_refs(at[0], at[2])
    template = {"units": 4, "rows": {"x": [2, 2]}, "outs": {"out": 2}}
    return {"program": desc, "ranges": [[at[0], ta, 0, 2], [at[1], tb, 2, 2], [at[2], ta, 4, 2]],
            "wiring": {"templates": {ta: template, tb: template}, "cuts": cuts or {}, "wires": []}}


def _affine(*parts):
    return {"concat": [{"affine": ["n", k, b, c]} for k, b, c in parts]}


def _wire(r, w, offset, at, base, cut0=0) -> dict:
    return {"reader": r, "port": "x", "writer": w, "out": "out", "at": at, "radix": [2],
            "base": base, "step": [1], "cut": [cut0, 1], "offset": offset}


_, TA, TB = ARR
_, UA, UB = TUP

CASES = [
    ("A, B, A", _of(ARR), {"wires": [_wire(TB, TA, 0, 0, 0), _wire(TA, TB, 0, 2, 0)]}),
    ("B's argument offset by half a row", _of(ARR, b_refs=lambda a, a2: _affine((a, 1, 3), (a, 0, 1))),
     {"refusal": "edge-rows"}),
    ("two parts, the second continuing the first, read as one", _of(ARR, b_refs=lambda a, a2: _affine((a, 0, 2), (a, 2, 2))),
     {"wires": [_wire(TB, TA, 0, 0, 0), _wire(TA, TB, 0, 2, 0)]}),
    ("two parts, the second not continuing the first", _of(ARR, b_refs=lambda a, a2: _affine((a, 2, 2), (a, 0, 2))),
     {"wires": [{**_wire(TB, TA, 0, 0, 1), "radix": [1], "step": [0]},
                {**_wire(TB, TA, 0, 1, 0), "radix": [1], "step": [0]}, _wire(TA, TB, 0, 2, 0)]}),
    ("a strided part whose members step inside its run",
     _of(ARR, b_refs=lambda a, a2: {"strided": ["n", a, 0, 4, 0, 1, 4]}),
     {"wires": [_wire(TB, TA, 0, 0, 0), _wire(TA, TB, 0, 2, 0)]}),
    ("a row from three parts that together are one field in order",
     _of(ARR, b_refs=lambda a, a2: _affine((a, 0, 2), (a, 2, 1), (a, 3, 1))),
     {"wires": [_wire(TB, TA, 0, 0, 0), _wire(TA, TB, 0, 2, 0)]}),
    ("a row from parts of one field out of order",
     _of(ARR, b_refs=lambda a, a2: _affine((a, 0, 1), (a, 2, 1), (a, 1, 1), (a, 3, 1))), {"refusal": "edge-rows"}),
    ("an explicit part over a writer's rows, a wire a row",
     _of(ARR, b_refs=lambda a, a2: {"explicit": [["n", a, i] for i in range(4)]}),
     {"wires": [{**_wire(TB, TA, 0, 0, 0), "radix": [1], "step": [0]},
                {**_wire(TB, TA, 0, 1, 1), "radix": [1], "step": [0]}, _wire(TA, TB, 0, 2, 0)]}),
    ("an explicit row over a writer's leaves out of order",
     _of(ARR, b_refs=lambda a, a2: {"explicit": [["n", a, 1], ["n", a, 0], ["n", a, 2], ["n", a, 3]]}),
     {"refusal": "edge-rows"}),
    ("a read of a later node", _of(ARR, b_refs=lambda a, a2: {"affine": ["n", a2, 0, 4]}), {"refusal": "not-topological"}),
    ("a cut port's rows", _of(TUP, cuts={UA: {"x": 2}, UB: {"x": 2}}),
     {"wires": [_wire(UB, UA, 0, 0, 0, 0), _wire(UB, UA, 1, 0, 0, 1),
                _wire(UA, UB, 0, 2, 0, 0), _wire(UA, UB, 1, 2, 0, 1)]}),
    ("an uncut port over two fields of one unit", _of(TUP), {"refusal": "edge-run"}),
    ("rows straddling two writer units", _of(ARR, b_refs=lambda a, a2: _affine((a, 1, 2), (a, 1, 2))),
     {"refusal": "edge-units"}),
    ("a template the wiring doesn't list", {**_of(ARR), "wiring": {**_of(ARR)["wiring"], "templates": {}}},
     {"refusal": "edge-template"}),
]


def _fields_toy(name: str, fields: int, width: int, rparam, part, readers: int = 4, writers: int = 4) -> dict:
    """The piece's input on a root batching a writer (`writers` members, each `fields` fields of `width` leaves) and a reader
    (`readers` members, each one `rparam`) whose argument is `part(w)` over the writer node `w`."""
    col = Array(width, Value(16))
    writer = bind(CompositeDefinition(f"{name}W", 0, (), lambda S: ((("x", col),), Tuple(*[col] * fields)),
                                      lambda B, S, x: tuple_of(*[B.batch(XOR16, x, x, axes=(0, 0))] * fields),
                                      register=False))
    if isinstance(rparam, Value):
        reader = bind(CompositeDefinition(f"{name}R", 0, (), lambda S: ((("x", rparam),), rparam),
                                          lambda B, S, x: B.call(XOR16, x, x), register=False))
    else:
        reader = bind(CompositeDefinition(f"{name}R", 0, (), lambda S: ((("x", rparam),), rparam),
                                          lambda B, S, x: B.batch(XOR16, x, x, axes=(0, 0)), register=False))

    def body(B, S, x, y):
        return tuple_of(B.batch(writer, x, axes=(0,)), B.batch(reader, y, axes=(0,)))

    defn = bind(CompositeDefinition(name, 0, (), lambda S: ((("x", Array(writers, col)), ("y", Array(readers, rparam))),
                                                            Tuple(Array(writers, Tuple(*[col] * fields)),
                                                                  Array(readers, rparam))), body, register=False))
    desc = json.loads(D.canonical(codec.encode_program(Program(defn))))
    tw, tr = codec.descriptor_id(writer), codec.descriptor_id(reader)
    ids = desc["callees"]
    w = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tw)
    r = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tr)
    _nodes(desc)[r]["args"][0]["refs"] = part(w)
    template = {"units": 4, "rows": {"x": [1, 1]}, "outs": {"out": 1}}
    return {"program": desc, "ranges": [[w, tw, 0, writers], [r, tr, writers, readers]],
            "wiring": {"templates": {tw: template, tr: template}, "cuts": {}, "wires": []}}


ONE_FIELD = _fields_toy("CoreEdgesOneField", 4, 32, Array(32, Value(16)),
                        lambda w: {"strided": ["n", w, 0, 32, 128, 1, 128]})

ROWS32 = _fields_toy("CoreEdgesRows", 48, 64, Array(2048, Value(16)),
                     lambda w: {"strided": ["n", w, 0, 2048, 3072, 1, 8192]})

#: parts reading output fields of one writer member
FIELDS_CASES = [
    ("a part stepping inside a writer member across its fields",
     _fields_toy("CoreEdgesStep", 4, 32, Value(16), lambda w: {"strided": ["n", w, 0, 4, 0, 160, 4]}),
     {"refusal": "edge-run"}),
    ("a part whose rows are 32 fields of a writer member", ROWS32, {"refusal": "edge-run"}),
    ("a part across fields whose members step through its run, each row part of a field",
     _fields_toy("CoreEdgesRun", 4, 32, Value(16), lambda w: {"strided": ["n", w, 0, 4, 0, 160, 8]}, readers=8),
     {"refusal": "edge-run"}),
    ("a strided part over one field", ONE_FIELD,
     {"wires": [{"reader": ONE_FIELD["ranges"][1][1], "port": "x", "writer": ONE_FIELD["ranges"][0][1], "out": "out",
                 "at": 0, "radix": [4], "base": 0, "step": [1], "cut": [0, 1], "offset": 0}]}),
    ("a part over the third field of each writer unit",
     _fields_toy("CoreEdgesThird", 4, 32, Array(32, Value(16)), lambda w: {"strided": ["n", w, 64, 32, 128, 1, 128]}),
     {"wires": [{"reader": "CoreEdgesThirdR_v0", "port": "x", "writer": "CoreEdgesThirdW_v0", "out": "out",
                 "at": 0, "radix": [4], "base": 0, "step": [1], "cut": [0, 1], "offset": 64}]}),
]


def _seed(name: str, fields: int, width: int, rparam, readers: int, part: list) -> dict:
    return _fields_toy(name, fields, width, rparam, lambda w: {"strided": ["n", w, *part]}, readers=readers)


def _wire_of(j: dict, radix, step, base=0, offset=0, at=0, cut0=0) -> dict:
    return {"reader": j["ranges"][1][1], "port": "x", "writer": j["ranges"][0][1], "out": "out", "at": at,
            "radix": radix, "base": base, "step": step, "cut": [cut0, 1], "offset": offset}


SEED34 = _seed("CoreEdgesSeed34", 1, 1, Value(16), 6, [0, 3, 1, 0, 6])
SEED23 = _seed("CoreEdgesSeed23", 1, 8, Array(8, Value(16)), 4, [16, 16, 0, 1, 32])
SEED38 = _seed("CoreEdgesSeed38", 24, 1, Value(16), 9, [20, 3, 0, 24, 9])

#: the forms of the record generator's cases the reader refused (its seeds): members stepping inside a run, and an
#: explicit part mixing input-node leaves with a writer's row
SEED_CASES = [
    ("seed 34: each writer unit read by three consecutive reader units", SEED34,
     {"wires": [_wire_of(SEED34, [2, 3], [1, 0])]}),
    ("seed 23: two periods of one run over two writer units, a row a member", SEED23,
     {"wires": [_wire_of(SEED23, [2, 2], [0, 1], base=2)]}),
    ("seed 38: a strided run over one field of three writer units, read three times", SEED38,
     {"wires": [_wire_of(SEED38, [3, 3], [0, 1], offset=20)]}),
    ("seeds 82 and 503: an explicit part, input-node leaves and one writer row",
     _of(TUP, cuts={UA: {"x": 2}, UB: {"x": 2}},
         b_refs=lambda a, a2: {"explicit": [["n", 0, 0], ["n", a, 1], ["n", 0, 2], ["n", 0, 3]]}),
     {"wires": [{**_wire(UB, UA, 1, 0, 0, 1), "radix": [1], "step": [0]},
                _wire(UA, UB, 0, 2, 0, 0), _wire(UA, UB, 1, 2, 0, 1)]}),
    ("an explicit row mixing input-node leaves with a writer's",
     _of(TUP, b_refs=lambda a, a2: {"explicit": [["n", 0, 0], ["n", a, 1], ["n", 0, 2], ["n", 0, 3]]}),
     {"refusal": "edge-rows"}),
]


def _mixed_toy(name: str, widths: tuple, read: int) -> dict:
    """The piece's input on a root batching a writer (4 members, each one field a width of `widths`) and a reader (4
    members, each one 32-bit value) whose member m reads field `read` of writer unit m."""
    ret = Tuple(*[Value(b) for b in widths])
    writer = bind(CompositeDefinition(f"{name}W", 0, (), lambda S: ((("x", Value(32)),), ret),
                                      lambda B, S, x: tuple_of(*[B.call(LOW16, x) if b == 16 else B.call(XOR32, x, x)
                                                                 for b in widths]), register=False))
    reader = bind(CompositeDefinition(f"{name}R", 0, (), lambda S: ((("x", Value(32)),), Value(32)),
                                      lambda B, S, x: B.call(XOR32, x, x), register=False))

    def body(B, S, x, y):
        return tuple_of(B.batch(writer, x, axes=(0,)), B.batch(reader, y, axes=(0,)))

    defn = bind(CompositeDefinition(name, 0, (), lambda S: ((("x", Array(4, Value(32))), ("y", Array(4, Value(32)))),
                                                            Tuple(Array(4, ret), Array(4, Value(32)))), body,
                                    register=False))
    desc = json.loads(D.canonical(codec.encode_program(Program(defn))))
    tw, tr = codec.descriptor_id(writer), codec.descriptor_id(reader)
    ids = desc["callees"]
    w = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tw)
    r = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tr)
    _nodes(desc)[r]["args"][0]["refs"] = {"strided": ["n", w, read, 1, len(widths), 1, 4]}
    template = {"units": 4, "rows": {"x": [2, 1]}, "outs": {"out": sum(widths) // 16}}
    return {"program": desc, "ranges": [[w, tw, 0, 4], [r, tr, 4, 4]],
            "wiring": {"templates": {tw: template, tr: template}, "cuts": {}, "wires": []}}


MIXED_16_32 = _mixed_toy("CoreEdgesMixedA", (16, 32), 1)
MIXED_32_16_32 = _mixed_toy("CoreEdgesMixedB", (32, 16, 32), 2)

def _slots_toy(name: str) -> dict:
    """The untied and INT8 classes' token read: a writer (4 members) of four groups of fields, one 32-bit value, 28,
    one, then a pair, and a reader (4 members, each one 32-bit value) whose member m reads field 4m of writer unit m,
    the references 160 apart."""
    v, a = Value(32), Array(28, Value(32))
    ret = Tuple(*[v, a, v, Tuple(v, v)] * 4)

    def fields(B, x):
        one = B.call(XOR32, x[0], x[0])
        return [one, B.batch(XOR32, x, x, axes=(0, 0)), one, tuple_of(one, one)] * 4

    writer = bind(CompositeDefinition(f"{name}W", 0, (), lambda S: ((("x", a),), ret),
                                      lambda B, S, x: tuple_of(*fields(B, x)), register=False))
    reader = bind(CompositeDefinition(f"{name}R", 0, (), lambda S: ((("x", v),), v),
                                      lambda B, S, x: B.call(XOR32, x, x), register=False))

    def body(B, S, x, y):
        return tuple_of(B.batch(writer, x, axes=(0,)), B.batch(reader, y, axes=(0,)))

    defn = bind(CompositeDefinition(name, 0, (), lambda S: ((("x", Array(4, a)), ("y", Array(4, v))),
                                                            Tuple(Array(4, ret), Array(4, v))), body, register=False))
    desc = json.loads(D.canonical(codec.encode_program(Program(defn))))
    tw, tr = codec.descriptor_id(writer), codec.descriptor_id(reader)
    ids = desc["callees"]
    w = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tw)
    r = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tr)
    _nodes(desc)[r]["args"][0]["refs"] = {"strided": ["n", w, 0, 4, 0, 160, 4]}
    template = {"units": 4, "rows": {"x": [2, 1]}, "outs": {"out": 256}}
    return {"program": desc, "ranges": [[w, tw, 0, 4], [r, tr, 4, 4]],
            "wiring": {"templates": {tw: template, tr: template}, "cuts": {}, "wires": []}}


ROPE = _fields_toy("CoreEdgesRoPE", 48, 64, Array(64, Value(16)), lambda w: {"strided": ["n", w, 0, 2048, 3072, 1, 8192]},
                   readers=128)
SLOTS = _slots_toy("CoreEdgesSlots")
ROPE_WIDE = _fields_toy("CoreEdgesRoPEWide", 48, 64, Array(64, Value(16)),
                        lambda w: {"strided": ["n", w, 0, 2048, 3072, 1, 2**26]}, readers=2**20, writers=2**15)

#: the served classes' reads, whose field's place in the writer unit steps with a digit (the wire's `ostep`)
SERVED_CASES = [
    ("the tied class's rotary head: 32 fields of each of 4 writer units", ROPE,
     {"wires": [{**_wire_of(ROPE, [4, 32], [1, 0]), "ostep": [0, 64]}]}),
    ("the untied and INT8 classes' token: field 4m of writer unit m, 32-bit leaves", SLOTS,
     {"wires": [{**_wire_of(SLOTS, [4], [1]), "ostep": [64]}]}),
]

#: a wire's `offset` is in u16 words of the writer unit's row `out`: a 32-bit field is two words
WORD_CASES = [
    ("a 32-bit field after a 16-bit one, at word 1", MIXED_16_32, {"wires": [_wire_of(MIXED_16_32, [4], [1], offset=1)]}),
    ("a 32-bit field after a 32-bit and a 16-bit one, leaf 2 at word 3", MIXED_32_16_32,
     {"wires": [_wire_of(MIXED_32_16_32, [4], [1], offset=3)]}),
]


def _grid(*lanes, run=1):
    return lambda a, a2: {"grid": [run, [[s, a if i is None else i, b, e, lv] for s, i, b, e, lv in lanes]]}


_CUT = {UA: {"x": 2}, UB: {"x": 2}}

#: grid parts, read by the twin alone until the Lean's `Part` has the grid (`Spot.Edges.partLanes`)
GRID_CASES = [
    ("two lanes over a cut port", _of(TUP, cuts=_CUT, b_refs=_grid(("n", None, 0, 0, [[2, 2]]), ("n", None, 3, 0, [[2, -2]]))),
     {"wires": [_wire(UB, UA, 0, 0, 0, 0), {**_wire(UB, UA, 1, 0, 1, 1), "step": [-1]},
                _wire(UA, UB, 0, 2, 0, 0), _wire(UA, UB, 1, 2, 0, 1)]}),
    ("a row whose lanes read two writer units", _of(TUP, b_refs=_grid(("n", None, 0, 0, [[2, 2]]), ("n", None, 3, 0, [[2, -2]]))),
     {"refusal": "edge-units"}),
    ("a grid over an input node", _of(TUP, cuts=_CUT, b_refs=_grid(("n", 0, 0, 0, [[2, 2]]), ("n", 0, 3, 0, [[2, -2]]))),
     {"wires": [_wire(UA, UB, 0, 2, 0, 0), _wire(UA, UB, 1, 2, 0, 1)]}),
    ("lanes of two period counts", _of(TUP, cuts=_CUT, b_refs=_grid(("n", None, 0, 0, [[2, 2]]), ("n", None, 0, 0, [[4, 1]]))),
     {"refusal": "edge-rows"}),
    ("a lane past its writer node", _of(TUP, cuts=_CUT, b_refs=_grid(("n", None, 0, 0, [[2, 4]]), ("n", None, 3, 0, [[2, -2]]))),
     {"refusal": "edge-rows"}),
]


def _variants():
    """Every split of `b`'s argument into up to three affine parts over `a`'s leaves, each at an offset, and each cut."""
    out = []
    for cuts in (None, {TB: {"x": 2}}):
        for offset in (0, 1):
            for split in ((4,), (2, 2), (1, 3), (3, 1), (1, 1, 2), (2, 1, 1)):
                parts, at = [], offset
                for c in split:
                    parts.append((at % 4, c) if at % 4 + c <= 4 else (0, c))
                    at += c
                out.append(_of(ARR, cuts=cuts, b_refs=lambda a, a2, ps=tuple(parts): _affine(*((a, b, c) for b, c in ps))))
    return out


def reference(j) -> dict:
    return edges(j["program"], j["ranges"], j["wiring"])


def _code(out: dict) -> dict:
    """The output with a refusal's place dropped: the named cases name the code."""
    return {"refusal": out["refusal"].split(" at ")[0]} if "refusal" in out else out


EDGES = D.Port("spot-difftest", "edges", reference,
               st.sampled_from([c[1] for c in CASES + SEED_CASES + WORD_CASES + SERVED_CASES] + _variants()),
               package=PACKAGE, need_rejects=False, of_record="lean")
pieces = D.server_fixture(EDGES)


@pytest.mark.parametrize("name, j, want", CASES, ids=[c[0] for c in CASES])
def test_the_named_cases(pieces, name, j, want):
    """Each named case gives the rule's wires or refusal, in Lean and in the twin."""
    assert _code(got := reference(j)) == want
    assert pieces.call("edges", j) == ("output", D.canonical(got))


@pytest.mark.parametrize("name, j, want", SEED_CASES, ids=[c[0] for c in SEED_CASES])
def test_the_generators_forms(pieces, name, j, want):
    """Members stepping inside a run are a wire in closed form, and an explicit part's rows each a wire or none, in
    Lean and in the twin."""
    assert _code(got := reference(j)) == want
    assert pieces.call("edges", j) == ("output", D.canonical(got))


@pytest.mark.parametrize("name, j, want", SERVED_CASES, ids=[c[0] for c in SERVED_CASES])
def test_the_served_classes_reads(pieces, name, j, want):
    """A member's field stepping inside its writer unit is one wire whose offset steps by a word step a digit, in Lean
    and in the twin."""
    assert _code(got := reference(j)) == want
    assert pieces.call("edges", j) == ("output", D.canonical(got))


def test_the_edges_cost_the_description(pieces):
    """The rotary read with 2^20 reader members over 2^15 writer units: as many wires as with 128, its digits wider. A
    piece or twin that walked members or leaves would not return here (2^26 leaves)."""
    want = {"wires": [{**_wire_of(ROPE_WIDE, [2**15, 32], [1, 0]), "ostep": [0, 64]}]}
    assert _code(got := reference(ROPE_WIDE)) == want
    assert pieces.call("edges", ROPE_WIDE) == ("output", D.canonical(got))


@pytest.mark.parametrize("name, j, want", WORD_CASES, ids=[c[0] for c in WORD_CASES])
def test_the_offset_is_in_words(pieces, name, j, want):
    """A wire's offset is the field's first u16 word, the fields before it each their bits in words, in Lean and in
    the twin."""
    assert _code(got := reference(j)) == want
    assert pieces.call("edges", j) == ("output", D.canonical(got))


def test_a_refusal_names_its_node_argument_and_part(pieces):
    j = _of(TUP)
    b = [k for k, x in enumerate(_nodes(j["program"])) if j["program"]["callees"][x["fn"]] == UB][0]
    want = {"refusal": f"edge-run at node {b} argument 0 part 0"}
    assert reference(j) == want
    assert pieces.call("edges", j) == ("output", D.canonical(want))


@pytest.mark.parametrize("name, j, want", FIELDS_CASES, ids=[c[0] for c in FIELDS_CASES])
def test_one_run_a_row(pieces, name, j, want):
    """A row reading one whole field of one writer unit is a wire from its first word, in Lean and in the twin; one
    reading several fields or part of one is refused `edge-run`, and members whose rows are no one place in their units
    `edge-rows`."""
    assert _code(got := reference(j)) == want
    assert pieces.call("edges", j) == ("output", D.canonical(got))


@pytest.mark.parametrize("name, j, want", GRID_CASES, ids=[c[0] for c in GRID_CASES])
def test_the_grid_cases(name, j, want):
    """Each grid case gives the rule's wires or refusal in the twin."""
    assert _code(reference(j)) == want


def test_a_part_form_it_does_not_know_is_refused_by_name():
    with pytest.raises(ValueError, match="^edge-form: "):
        _parts({"rows": ["n", 1, 0, 4]}, [])


def test_two_levels_are_the_wires_radix():
    """A batch of 128 reading 64 leaves a member as four members a row, then the next row: levels `[[4, 0], [32, 64]]`
    are the wire's radix `[32, 4]` and steps `[64, 0]`, outermost first."""
    q = ("g", 64, ((False, 7, 0, 1, ((4, 0), (32, 64))),), 8192, True)
    assert _lane_wire(q, 0, 64, 64, 128) == (7, 0, [32, 4], [64, 0])
    assert _lane_wire(q, 4 * 64, 64, 64, 8) == (7, 64, [2, 4], [64, 0])


def test_the_twin_agrees_on_rewritten_arguments(pieces):
    """Lean and the twin agree on every split, offset and cut of `b`'s argument."""
    stats = D.run(EDGES, examples=200, server=pieces)
    assert stats.outputs > 0, stats
