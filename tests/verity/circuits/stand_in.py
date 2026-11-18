"""The stand-in Boolean bodies the suite's test primitives get (`stand_in_body`), registered with `verity.circuits.program` as
this module loads: the body of the lookup registered before it (else the catalog's) where it has one, a stand-in for a primitive a test defines, and a refusal for a
primitive of `verity` or an adapter with none, or one a test defines under such a primitive's id. Another suite's conftest,
or a script that writes difftest vectors, loads it by file, the file among the inputs of every suite that runs it."""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits import program as PG
from verity.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.ops.boolean import versions as V
from verity.circuits.refs import Coll, Explicit, array_of

_STAND_INS: dict[str, object] = {}
#: the packages whose primitives are the catalog's: none of them gets a stand-in
CATALOG = ("verity", "verity_vllm", "verity_torch")


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


def _of_catalog(p: PrimitiveDefinition) -> bool:
    return getattr(p.evaluate, "__module__", "").split(".")[0] in CATALOG


_BEFORE = PG._BODY_LOOKUP[0] if PG._BODY_LOOKUP else V.boolean_version


def _lookup(p: PrimitiveDefinition):
    try:
        return _BEFORE(p)
    except LookupError:
        if _of_catalog(p) or (p.id in REGISTRY and _of_catalog(REGISTRY[p.id])):
            raise
        return stand_in_body(p)


PG.register_boolean_bodies(_lookup)
