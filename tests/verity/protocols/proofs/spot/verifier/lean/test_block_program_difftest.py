"""The block's Program on bits (`Spot.BlockFront.program`, the ``block_program`` piece): each word composite a block reaches,
lifted to bits, is held to Python's lift (`versions.lift_body`, each callee its lift or its Boolean version) encoded by
the codec and read back by Lean's decoder (the ``definition`` piece), reference by reference; the block Definition, whose
parameters are the input ports and whose return is the output ports, is held to a reference over Python's layout that
names each port's bits and each unit's operand bits."""

from __future__ import annotations

import json
import shutil
from functools import cache
from pathlib import Path

import pytest

import verity_difftest as D
from verity.circuits import Array, Program, Tuple, Value, bind
from verity.circuits import partition_object as PO
from verity.circuits.boolean import bit_type
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.defs import CompositeDefinition
from verity.circuits.layout import resolve
from verity.circuits.ops.boolean import versions as BV
from verity.circuits.ops.scalar import I32Add
from verity.circuits.refs import Affine, Coll, array_of, tuple_of

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean"
PIECES = "spot-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

I32, V16, BIT = Value(32), Value(16), Value(1)


def _c(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


#: x0 + x1
Sum = _c("BSum", (("x", Array(2, I32)),), I32, lambda B, S, x: B.call(I32Add, x[0], x[1]))
#: (s + x1, s), s = x0 + z0: a parameter of two widths, its 16-bit leaf unread
Word = _c("BWord", (("x", Array(2, I32)), ("z", Tuple(I32, V16))), Tuple(I32, I32),
          lambda B, S, x, z: (lambda s: tuple_of(B.call(I32Add, s, x[1]), s))(B.call(I32Add, x[0], z[0])))
#: each row's Sum: a batch over a lifted composite
Rows = _c("BRows", (("x", Array(2, Array(2, I32))),), Array(2, I32), lambda B, S, x: B.batch(Sum, x, axes=(0,)))
OURS = {"BSum", "BWord", "BRows"}
#: I32Add's Boolean Definition, which C's Boolean map names for it
ADD = BV.boolean_version(I32Add)


def _top(B, S, a, b, c, e, f):
    u1 = B.call(Word, a, b)
    u2 = B.call(Rows, c)
    u3 = B.call(ADD, e, f)
    u4 = B.call(Sum, array_of([u1[0], u2[1]]))
    return tuple_of(u1, u2, u3, u4)


Top = _c("BTop", (("a", Array(2, I32)), ("b", Tuple(I32, V16)), ("c", Array(2, Array(2, I32))), ("e", Array(32, BIT)),
                  ("f", Array(32, BIT))), Tuple(Tuple(I32, I32), Array(2, I32), Array(32, BIT), I32), _top)


@cache
def lifted(fn):
    """``fn`` on bits as `versions.lift` makes it, each callee of ours its own lift and every other its Boolean version."""
    body = BV.lift_body(fn, override=lambda g: lifted(g) if getattr(g, "name", None) in OURS else None)
    params, ret = fn.signature()
    bparams, bret = tuple((n, bit_type(t)) for n, t in params), bit_type(ret)
    defn = CompositeDefinition(fn.name, fn.version, (*fn.defn.statics, "IR"), lambda S: (bparams, bret), body, register=False)
    return bind(defn, **fn.bindings, IR="boolean")


def text(program) -> str:
    return b"".join(descriptor_chunks(program)).decode()


@cache
def holding(fn) -> str:
    """The descriptor of a Program whose root calls ``fn`` once, so ``fn`` is a Definition of it under its own key."""
    params, ret = fn.signature()
    return text(Program(_c("BHold" + fn.name, params, ret, lambda B, S, *xs: B.call(fn, *xs))))


@cache
def top():
    program = Program(Top)
    return program, text(program)


def tyj(t) -> list:
    """`Ty.toJson`."""
    if isinstance(t, Value):
        return ["Value", t.w]
    if isinstance(t, Array):
        return ["Array", t.n, tyj(t.elem)]
    return ["Tuple", [tyj(f) for f in t.fields]]


def offsets(t):
    return BV.bit_offsets(t)


def block_reference(pieces, units: list[int]) -> dict:
    """The block Definition (`defJson`) from Python's layout: each port's gate and width, each unit's operand bits."""
    program, txt = top()
    lay, body = program.circuit, program.root.body
    calls = [(k, fn) for k, fn in PO.calls(program)]
    units = sorted(units)
    S = [(body.offsets[calls[u][0]], body.offsets[calls[u][0]] + calls[u][1].gates) for u in units]
    inside = lambda g: any(lo <= g < hi for lo, hi in S)  # noqa: E731
    def leaves(coll):
        return [(resolve(lay.root_scope, r), w) for r, w in zip(coll.refs, coll.type.leaf_widths(), strict=True)]

    ops = [[leaves(a) for a in body.nodes[calls[u][0]].args] for u in units]
    ins = sorted({gw for unit in ops for arg in unit for gw in arg if not inside(gw[0])})
    idx = {g: i for i, (g, _w) in enumerate(ins)}
    made, outs, nodes = {}, set(), []
    for t, u in enumerate(units):
        k, fn = calls[u]
        refs = []
        for arg in ops[t]:
            rs = []
            for g, w in arg:
                if g in idx:
                    rs += [["p", idx[g], b] for b in range(w)]
                else:
                    u2, o = made[g]
                    rs += [["n", u2, o + b] for b in range(w)]
            refs.append(rs)
        ret = fn.signature()[1]
        off = offsets(ret)
        for j, (g, w) in enumerate(leaves(Coll(ret, Affine("n", k, 0, ret.leaves)))):
            if inside(g):
                outs.add((g, w))
                made.setdefault(g, (t, off(j)))
        callee = descriptor_id(lifted(fn) if fn.name in OURS else fn)
        gates = json.loads(pieces.call("definition", {"descriptor": txt if fn.name not in OURS else holding(lifted(fn)),
                                                      "key": callee})[1])["gates"]
        nodes.append({"fn": callee, "form": "call", "refs": refs, "gates": gates})
    ret = [["n", made[g][0], made[g][1] + b] for g, w in sorted(outs) for b in range(w)]
    port = lambda w: tyj(bit_type(Value(w)))  # noqa: E731
    return {"params": [port(w) for _g, w in ins], "ret": ["Tuple", [port(w) for _g, w in sorted(outs)]],
            "refs": ret, "gates": sum(n["gates"] for n in nodes), "nodes": nodes}


PROGRAM = D.Port(PIECES, "block_program", None, None, package=PACKAGE, of_record="lean")
pieces = D.server_fixture(PROGRAM)

BLOCKS = [[0], [1], [2], [3], [0, 3], [3, 1], [0, 1, 2, 3]]


@pytest.mark.parametrize("units", BLOCKS, ids=lambda us: "units-" + "-".join(map(str, us)))
def test_the_block_program_agrees(pieces, units):
    _program, txt = top()
    verdict, out = pieces.call("block_program", {"descriptor": txt, "units": units})
    assert verdict == "output", out
    out = json.loads(out)
    block, want = out["block"], block_reference(pieces, units)
    assert block["key"] == "Block_v1"
    for k in ("params", "ret", "refs", "gates"):
        assert block[k] == want[k], k
    assert [(n["fn"], n["form"], [a["refs"] for a in n["args"]]) for n in block["nodes"]] == \
        [(n["fn"], n["form"], n["refs"]) for n in want["nodes"]]
    reached = {Word: {Word}, Rows: {Rows, Sum}, Sum: {Sum}, ADD: set()}
    fns = [fn for _k, fn in PO.calls(top()[0])]
    ours = set().union(*(reached[fns[u]] for u in units))
    assert set(out["lifted"]) == {descriptor_id(lifted(fn)) for fn in ours}
    for fn in ours:
        key = descriptor_id(lifted(fn))
        ref = pieces.call("definition", {"descriptor": holding(lifted(fn)), "key": key})
        assert ("output", D.canonical(out["lifted"][key])) == ref, key


@pytest.mark.parametrize("name, units", [("no unit", []), ("a unit twice", [1, 1]), ("a unit past C's", [4])],
                         ids=lambda x: x if isinstance(x, str) else "")
def test_the_block_program_refuses(pieces, name, units):
    _program, txt = top()
    assert pieces.call("block_program", {"descriptor": txt, "units": units})[0] == "reject", name


def test_the_lift_reads_cs_boolean_map(pieces):
    """The word primitive's node calls the Boolean Definition C's own map names for it."""
    _program, txt = top()
    out = json.loads(pieces.call("block_program", {"descriptor": txt, "units": [3]})[1])
    (lifted_sum,) = out["lifted"].values()
    assert json.loads(txt)["boolean"]["I32Add_v1"] == descriptor_id(ADD)
    assert [n["fn"] for n in lifted_sum["nodes"]] == [descriptor_id(ADD)]
