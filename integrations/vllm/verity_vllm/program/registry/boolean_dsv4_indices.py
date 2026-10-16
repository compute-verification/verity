"""`dsv4_indices`' Definition on bits: `Dsv4CombineTopkSwa_v2{TOPK,RATIO,WINDOW,WIDTH}`, its word body line for line over gates
(the body is `dsv4_indices.combine`, written once over its steps), so its word view is its `_v1` at the same statics.

A step is `boolean_dsv4_indexer.GateOps`' gates for the word primitive `dsv4_indices.WORD` names (`forms`' integer add,
subtract and signed compare), a select as `forms.mux`, an AND and a NOT of bits, and bit i of a word as that wire; a constant
word is constant bits, so q's selects of constant words are wires and its sum of words with disjoint bits folds.  No table or
lookup gate.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, mux
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import boolean_dsv4_indexer as BI
from verity_vllm.program.registry import dsv4_indices as X

BITS = SimpleNamespace(I32=Array(32, BL.BIT))


class GateOps(BI.GateOps):
    """`boolean_dsv4_indexer.GateOps` with `dsv4_indices.WORD`'s ``isel``, ``band`` and ``bnot``, and ``bit(x, i)``."""

    def isel(self, c, a, b):
        return mux(self.C, c, a, b)

    def band(self, a, b):
        return self.C.AND(a, b)

    def bnot(self, a):
        return NOT(a)

    def bit(self, x, i: int):
        return x[i]


def _combine(S):
    sig = X.combine_sig(BITS)(S)
    counts = [len(BL.elements(t)) for _, t in sig[0]]

    def run(C, *elements):
        parts, i = [], 0
        for n in counts:
            parts.append(list(elements[i:i + n]))
            i += n
        return X.combine(GateOps(C), S, *parts)
    return run


Dsv4CombineTopkSwaV2 = EW.specialized("Dsv4CombineTopkSwa", 2, ["TOPK", "RATIO", "WINDOW", "WIDTH"], _combine, X.combine_sig(BITS),
                                      doc=f"`Dsv4CombineTopkSwa_v1`'s body on bits: {X.combine.__doc__}",
                                      word=lambda b: bind(X.Dsv4CombineTopkSwa, **b))

BOOLEAN_DSV4_INDICES = [Dsv4CombineTopkSwaV2]
