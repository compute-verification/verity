"""`stop_rule`'s Definitions on bits: `StopRule_v2{T, NSTOP}` and `StopRule16_v2{T, NSTOP}`, each its word body line for
line over gates (the bodies are `stop_rule.stop` and `stop_rule.stop16`, written once over their steps), so each one's word
view is its `_v1` at the same statics.

A step is `boolean_dsv4_indices.GateOps`' gates for the word primitive `stop_rule.WORD` names (`forms`' add, subtract and
signed compare, a select as `forms.mux`, an AND and a NOT of bits), an equality as `I32Eq_v2`'s (31 ANDs over the XNORs), an
OR as `forms.OR` and a word's high half as its 16 high wires.  A constant word is constant bits, so the first-stop chain's
selects of the constants i + 1 are ANDs only on the low bits that T + 1 needs, and fold above them.  A word added to itself
is the word's wires one place up, no AND, so `StopRule16_v2`'s one output, 16 bits, adds to `stop`'s gates only the two
carries of n + finish * 8192 into bits 14 and 15, and holds none of the gates that only the high halves of n and finish
read.  No table or lookup gate.

Registered functions:
  StopRule_v2{T,NSTOP}    (ids: T x 32 bits, max_tokens: 32 bits, stops: NSTOP x 32 bits, prompt_len: 32 bits,
                          max_model_len: 32 bits) -> (n: 32 bits, finish: 32 bits)    word view: StopRule_v1
  StopRule16_v2{T,NSTOP}  the same inputs -> 16 bits                                  word view: StopRule16_v1
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

BITS = SimpleNamespace(I32=Array(32, BL.BIT), U16=Array(16, BL.BIT))


class GateOps(BI.GateOps):
    """`boolean_dsv4_indices.GateOps` with `stop_rule.WORD`'s ``ieq``, ``bor`` and ``hi16``."""

    def ieq(self, a, b):
        return and_reduce(self.C, [NOT(x ^ y) for x, y in zip(a, b)])

    def bor(self, a, b):
        return OR(self.C, a, b)

    def hi16(self, a):
        return list(a[16:])


def _stop(S):
    def run(C, *x):
        T, K = S.T, S.NSTOP
        n, finish = X.stop(GateOps(C), S, list(x[:T]), x[T], list(x[T + 1:T + 1 + K]), x[T + 1 + K], x[T + 2 + K])
        return [n, finish]
    return run


def _stop16(S):
    def run(C, *x):
        T, K = S.T, S.NSTOP
        return [X.stop16(GateOps(C), S, list(x[:T]), x[T], list(x[T + 1:T + 1 + K]), x[T + 1 + K], x[T + 2 + K])]
    return run


StopRuleV2 = EW.specialized("StopRule", 2, ["T", "NSTOP"], _stop, X.stop_sig(BITS),
                            doc=f"`StopRule_v1`'s body on bits: {X.stop.__doc__}", refuse=X.refuse,
                            word=lambda b: bind(X.StopRule, **b))
StopRule16V2 = EW.specialized("StopRule16", 2, ["T", "NSTOP"], _stop16, X.stop16_sig(BITS),
                              doc=f"`StopRule16_v1`'s body on bits: {X.stop16.__doc__}", refuse=X.refuse16,
                              word=lambda b: bind(X.StopRule16, **b))

BOOLEAN_STOP_RULE = [StopRuleV2, StopRule16V2]
