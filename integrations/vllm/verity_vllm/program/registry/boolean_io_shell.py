"""`io_shell`'s Definition on bits: `IngressPrompt_v2{MODEL,LP,V}`, its word body line for line over gates (the body is
`io_shell.parse`, written once over its steps), so its word view is its `_v1` at the same statics.

A step is `boolean_dsv4_indices.GateOps`' gates for the word primitive `io_shell.WORD` names (`forms`' integer add and signed
compare, a select as `forms.mux`, an AND and a NOT of bits), and bit i of a byte as that wire.  A constant word is constant bits,
so a field's selects of constant words are ANDs with one wire and its sums fold above the field's largest value.  No table or
lookup gate.  Every byte of the body reaches ``ok``.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import io_shell as X
from verity_vllm.program.registry.boolean_dsv4_indices import GateOps

BITS = SimpleNamespace(U8=Array(8, BL.BIT), BIT=BL.BIT, I32=Array(32, BL.BIT))


def _parse(S):
    def run(C, *body):
        ok, ids, m = X.parse(GateOps(C), S, list(body))
        return [[ok], *ids, m]
    return run


IngressPromptV2 = EW.specialized("IngressPrompt", 2, ["MODEL", "LP", "V"], _parse, X.ingress_sig(BITS),
                                 doc=f"`IngressPrompt_v1`'s body on bits: {X.parse.__doc__}",
                                 word=lambda b: bind(X.IngressPrompt, **b))

BOOLEAN_IO_SHELL = [IngressPromptV2]
