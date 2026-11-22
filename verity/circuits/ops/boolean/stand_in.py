"""The stand-in Boolean bodies test primitives get (`stand_in_body`), registered with `verity.circuits.program` by `register`:
the catalog's body where it has one, for a primitive a test defines its `exact` body or else a stand-in, and a refusal for
a primitive of `verity` or an adapter with none. Tests and the scripts that write test vectors call it; nothing that
serves or verifies does."""
from __future__ import annotations

from collections.abc import Callable

from verity.circuits import boolean as BL
from verity.circuits import program as PG
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.ops.boolean import versions as V
from verity.circuits.refs import Coll, Explicit, array_of

_STAND_INS: dict[str, object] = {}
_EXACT: dict[str, Callable[..., list[Coll]]] = {}


def exact(p: PrimitiveDefinition, bits: Callable[..., list[Coll]]) -> PrimitiveDefinition:
    """Give test primitive ``p`` the body ``bits`` computes, ``p``'s own function in place of the stand-in:
    ``bits(B, *params)`` takes each parameter's bits and returns the output's, each list least significant first."""
    _EXACT[p.id] = bits
    _STAND_INS.pop(p.id, None)
    return p


def stand_in_body(p: PrimitiveDefinition):
    """A test primitive's Boolean body: its `exact` one where a test gave it, else one for the format, layout and partition
    tests, which read a body's shape and never its function: output bit j is the Xor of bit j of every input leaf (absent
    past a leaf's width), so it computes the primitive only where that is a 1-bit sum."""
    if p.id not in _STAND_INS:
        params, ret = p.signature()
        w = ret.w
        own = _EXACT.get(p.id)

        def body(B, S, *args):
            cols: list[list[Coll]] = [[] for _ in range(w)]
            ins: list[list[Coll]] = []
            for (_n, t), a in zip(params, args):
                flat = [Coll(BL.BIT, Explicit((r,))) for r in a.refs]
                ins.append(flat)
                at = 0
                for lw in t.leaf_widths():
                    for j in range(min(lw, w)):
                        cols[j].append(flat[at + j])
                    at += lw
            if own is not None:
                out = own(B, *ins)
            else:
                out = [B.call(BL.xor_gate(len(c)), BL.operands(*c)) if len(c) > 1 else c[0] if c else B.call(BL.constant(0))
                       for c in cols]
            return out[0] if w == 1 else array_of(out)

        bits = [(n, BL.bit_type(t)) for n, t in params]
        name = f"{'Exact' if own is not None else 'StandIn'}[{p.name}]"
        _STAND_INS[p.id] = bind(CompositeDefinition(name, 1, (), lambda S: (tuple(bits), BL.bit_type(ret)), body,
                                                    register=False))
    return _STAND_INS[p.id]


def _lookup(p: PrimitiveDefinition):
    try:
        return V.boolean_version(p)
    except V.NoBooleanVersion:
        if getattr(p.evaluate, "__module__", "").split(".")[0] in ("verity", "verity_vllm", "verity_torch"):
            raise
        return stand_in_body(p)


def register() -> None:
    """Make the stand-in lookup the one `Program` builds its Boolean map with, in place of the catalog's alone."""
    PG.register_boolean_bodies(_lookup)
