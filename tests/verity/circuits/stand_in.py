"""The stand-in Boolean bodies the suite's test primitives get (`stand_in_body`), registered with `verity.circuits.program` as
this module loads: the catalog's body where it has one, a stand-in for a primitive a test defines, and a refusal for a
primitive of `verity` or an adapter with none."""
from __future__ import annotations

import contextlib

from verity.circuits import boolean as BL
from verity.circuits import program as PG
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.ops.boolean import versions as V
from verity.circuits.refs import Coll, Explicit, array_of

_STAND_INS: dict[str, object] = {}


def stand_in_body(p: PrimitiveDefinition):
    """A test primitive's Boolean body for the format, layout and partition tests, which read a body's shape and never its
    function: output bit j is the Xor of bit j of every input leaf (absent past a leaf's width), so it computes the primitive
    only where that is a 1-bit sum."""
    if p.id not in _STAND_INS:
        params, ret = p.signature()
        w = ret.w

        def body(B, S, *args):
            cols: list[list[Coll]] = [[] for _ in range(w)]
            for (_n, t), a in zip(params, args):
                flat, at = list(a.refs), 0
                for lw in t.leaf_widths():
                    for j in range(min(lw, w)):
                        cols[j].append(Coll(BL.BIT, Explicit((flat[at + j],))))
                    at += lw
            out = [B.call(BL.xor_gate(len(c)), BL.operands(*c)) if len(c) > 1 else c[0] if c else B.call(BL.constant(0)) for c in cols]
            return out[0] if w == 1 else array_of(out)

        bits = [(n, BL.bit_type(t)) for n, t in params]
        _STAND_INS[p.id] = bind(CompositeDefinition(f"StandIn[{p.name}]", 1, (), lambda S: (tuple(bits), BL.bit_type(ret)),
                                                    body, register=False))
    return _STAND_INS[p.id]


def _lookup(p: PrimitiveDefinition):
    try:
        return V.boolean_version(p)
    except V.NoBooleanVersion:
        if getattr(p.evaluate, "__module__", "").split(".")[0] in ("verity", "verity_vllm", "verity_torch"):
            raise
        return stand_in_body(p)


@contextlib.contextmanager
def registered():
    """This module's lookup inside the block, and the lookup registered before it after: for a suite where another module
    registers its own."""
    saved = list(PG._BODY_LOOKUP)
    PG.register_boolean_bodies(_lookup)
    try:
        yield
    finally:
        PG._BODY_LOOKUP[:] = saved


PG.register_boolean_bodies(_lookup)
