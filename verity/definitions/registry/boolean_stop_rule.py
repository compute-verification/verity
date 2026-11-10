"""`stop_rule`'s Definition on bits: `StopRule_v2{T, NSTOP}`, its word body line for line over gates (the body is
`stop_rule.stop`, written once over its steps), so its word view is its `_v1` at the same statics.

A step is `boolean_dsv4_indices.GateOps`' gates for the word primitive `stop_rule.WORD` names (`forms`' subtract and signed
compare, a select as `forms.mux`, an AND and a NOT of bits), an equality as `I32Eq_v2`'s (31 ANDs over the XNORs) and an OR
as `forms.OR`.  A constant word is constant bits, so the first-stop chain's selects of the constants i + 1 are ANDs only on
the low bits that T + 1 needs, and fold above them.  No table or lookup gate.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, OR, and_reduce
from verity.circuits.defs import bind
from verity.definitions.ops.boolean import elementwise as EW
from verity.circuits.types import Array
from verity.definitions.registry import boolean_dsv4_indices as BI
from verity.definitions.registry import stop_rule as X

BITS = SimpleNamespace(I32=Array(32, BL.BIT))


class GateOps(BI.GateOps):
    """`boolean_dsv4_indices.GateOps` with `stop_rule.WORD`'s ``ieq`` and ``bor``."""

    def ieq(self, a, b):
        return and_reduce(self.C, [NOT(x ^ y) for x, y in zip(a, b)])

    def bor(self, a, b):
        return OR(self.C, a, b)


def _stop(S):
    def run(C, *x):
        T, K = S.T, S.NSTOP
        n, finish = X.stop(GateOps(C), S, list(x[:T]), x[T], list(x[T + 1:T + 1 + K]), x[T + 1 + K], x[T + 2 + K])
        return [n, finish]
    return run


StopRuleV2 = EW.specialized("StopRule", 2, ["T", "NSTOP"], _stop, X.stop_sig(BITS),
                            doc=f"`StopRule_v1`'s body on bits: {X.stop.__doc__}", refuse=X.refuse,
                            word=lambda b: bind(X.StopRule, **b))

BOOLEAN_STOP_RULE = [StopRuleV2]
