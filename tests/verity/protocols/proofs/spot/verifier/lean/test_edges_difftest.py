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
    template = {"units": 4, "rows": {"x": [2, 2]}, "outs": {"o0": 2}}
    return {"program": desc, "ranges": [[at[0], ta, 0, 2], [at[1], tb, 2, 2], [at[2], ta, 4, 2]],
            "wiring": {"templates": {ta: template, tb: template}, "cuts": cuts or {}, "wires": []}}


def _affine(*parts):
    return {"concat": [{"affine": ["n", k, b, c]} for k, b, c in parts]}


def _wire(r, w, o, at, base, cut0=0) -> dict:
    return {"reader": r, "port": "x", "writer": w, "out": o, "at": at, "radix": [2],
            "base": base, "step": [1], "cut": [cut0, 1]}


_, TA, TB = ARR
_, UA, UB = TUP

CASES = [
    ("A, B, A", _of(ARR), {"wires": [_wire(TB, TA, "o0", 0, 0), _wire(TA, TB, "o0", 2, 0)]}),
    ("B's argument offset by half a row", _of(ARR, b_refs=lambda a, a2: _affine((a, 1, 3), (a, 0, 1))),
     {"refusal": "edge-rows"}),
    ("a two-part argument", _of(ARR, b_refs=lambda a, a2: _affine((a, 0, 2), (a, 2, 2))),
     {"wires": [{**_wire(TB, TA, "o0", 0, 0), "radix": [1], "step": [0]},
                {**_wire(TB, TA, "o0", 1, 1), "radix": [1], "step": [0]}, _wire(TA, TB, "o0", 2, 0)]}),
    ("a strided part whose members step inside its run",
     _of(ARR, b_refs=lambda a, a2: {"strided": ["n", a, 0, 4, 0, 1, 4]}),
     {"wires": [_wire(TB, TA, "o0", 0, 0), _wire(TA, TB, "o0", 2, 0)]}),
    ("an argument over the parts bound", _of(ARR, b_refs=lambda a, a2: _affine((a, 0, 2), (a, 2, 1), (a, 3, 1))),
     {"refusal": "edge-parts"}),
    ("an explicit read of a node that is no input node",
     _of(ARR, b_refs=lambda a, a2: {"explicit": [["n", a, i] for i in range(4)]}), {"refusal": "edge-rows"}),
    ("a read of a later node", _of(ARR, b_refs=lambda a, a2: {"affine": ["n", a2, 0, 4]}), {"refusal": "not-topological"}),
    ("a cut port's rows", _of(TUP, cuts={UA: {"x": 2}, UB: {"x": 2}}),
     {"wires": [_wire(UB, UA, "o0", 0, 0, 0), _wire(UB, UA, "o1", 0, 0, 1),
                _wire(UA, UB, "o0", 2, 0, 0), _wire(UA, UB, "o1", 2, 0, 1)]}),
    ("an uncut port over two fields", _of(TUP), {"refusal": "edge-fields"}),
    ("a template the wiring doesn't list", {**_of(ARR), "wiring": {**_of(ARR)["wiring"], "templates": {}}},
     {"refusal": "edge-template"}),
]


def _fields_toy(name: str, fields: int, width: int, rparam, part, readers: int = 4) -> dict:
    """The piece's input on a root batching a writer (4 members, each `fields` fields of `width` leaves) and a reader
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

    defn = bind(CompositeDefinition(name, 0, (), lambda S: ((("x", Array(4, col)), ("y", Array(readers, rparam))),
                                                            Tuple(Array(4, Tuple(*[col] * fields)),
                                                                  Array(readers, rparam))), body, register=False))
    desc = json.loads(D.canonical(codec.encode_program(Program(defn))))
    tw, tr = codec.descriptor_id(writer), codec.descriptor_id(reader)
    ids = desc["callees"]
    w = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tw)
    r = next(k for k, x in enumerate(_nodes(desc)) if ids[x["fn"]] == tr)
    _nodes(desc)[r]["args"][0]["refs"] = part(w)
    template = {"units": 4, "rows": {"x": [1, 1]}, "outs": {"o0": 1}}
    return {"program": desc, "ranges": [[w, tw, 0, 4], [r, tr, 4, readers]],
            "wiring": {"templates": {tw: template, tr: template}, "cuts": {}, "wires": []}}


ONE_FIELD = _fields_toy("CoreEdgesOneField", 4, 32, Array(32, Value(16)),
                        lambda w: {"strided": ["n", w, 0, 32, 128, 1, 128]})

#: circuits' two parts, each reading several output fields of one writer member
FIELDS_CASES = [
    ("a part stepping inside a writer member across its fields",
     _fields_toy("CoreEdgesStep", 4, 32, Value(16), lambda w: {"strided": ["n", w, 0, 4, 0, 160, 4]}),
     {"refusal": "edge-fields"}),
    ("a part whose rows are 32 fields of a writer member",
     _fields_toy("CoreEdgesRows", 48, 64, Array(2048, Value(16)), lambda w: {"strided": ["n", w, 0, 2048, 3072, 1, 8192]}),
     {"refusal": "edge-fields"}),
    ("a part across fields whose members leave its run",
     _fields_toy("CoreEdgesRun", 4, 32, Value(16), lambda w: {"strided": ["n", w, 0, 4, 0, 160, 8]}, readers=8),
     {"refusal": "edge-fields"}),
    ("a strided part over one field", ONE_FIELD,
     {"wires": [{"reader": ONE_FIELD["ranges"][1][1], "port": "x", "writer": ONE_FIELD["ranges"][0][1], "out": "o0",
                 "at": 0, "radix": [4], "base": 0, "step": [1], "cut": [0, 1]}]}),
]


def _grid(*lanes, run=1):
    return lambda a, a2: {"grid": [run, [[s, a if i is None else i, b, e, lv] for s, i, b, e, lv in lanes]]}


_CUT = {UA: {"x": 2}, UB: {"x": 2}}

#: grid parts, read by the twin alone until the Lean's `Part` has the grid (`Spot.Edges.partLanes`)
GRID_CASES = [
    ("two lanes over a cut port", _of(TUP, cuts=_CUT, b_refs=_grid(("n", None, 0, 0, [[2, 2]]), ("n", None, 3, 0, [[2, -2]]))),
     {"wires": [_wire(UB, UA, "o0", 0, 0, 0), {**_wire(UB, UA, "o1", 0, 1, 1), "step": [-1]},
                _wire(UA, UB, "o0", 2, 0, 0), _wire(UA, UB, "o1", 2, 0, 1)]}),
    ("a lane that does not fill a row", _of(TUP, b_refs=_grid(("n", None, 0, 0, [[2, 2]]), ("n", None, 3, 0, [[2, -2]]))),
     {"refusal": "edge-fields"}),
    ("a grid over an input node", _of(TUP, cuts=_CUT, b_refs=_grid(("n", 0, 0, 0, [[2, 2]]), ("n", 0, 3, 0, [[2, -2]]))),
     {"wires": [_wire(UA, UB, "o0", 2, 0, 0), _wire(UA, UB, "o1", 2, 0, 1)]}),
    ("lanes of two period counts", _of(TUP, cuts=_CUT, b_refs=_grid(("n", None, 0, 0, [[2, 2]]), ("n", None, 0, 0, [[4, 1]]))),
     {"refusal": "edge-parts"}),
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


EDGES = D.Port("spot-difftest", "edges", reference, st.sampled_from([c[1] for c in CASES] + _variants()),
               package=PACKAGE, need_rejects=False, of_record="lean")
pieces = D.server_fixture(EDGES)


@pytest.mark.parametrize("name, j, want", CASES, ids=[c[0] for c in CASES])
def test_the_named_cases(pieces, name, j, want):
    """Each named case gives the rule's wires or refusal, in Lean and in the twin."""
    assert reference(j) == want
    assert pieces.call("edges", j) == ("output", D.canonical(want))


@pytest.mark.parametrize("name, j, want", FIELDS_CASES, ids=[c[0] for c in FIELDS_CASES])
def test_one_field_a_part(pieces, name, j, want):
    """A part reading more than one output field of one writer is refused `edge-fields`, in Lean and in the twin; one
    over one field is a wire."""
    assert reference(j) == want
    assert pieces.call("edges", j) == ("output", D.canonical(want))


@pytest.mark.parametrize("name, j, want", GRID_CASES, ids=[c[0] for c in GRID_CASES])
def test_the_grid_cases(name, j, want):
    """Each grid case gives the rule's wires or refusal in the twin."""
    assert reference(j) == want


def test_a_part_form_it_does_not_know_is_refused_by_name():
    with pytest.raises(ValueError, match="^edge-parts: "):
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
