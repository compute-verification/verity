"""`io_shell`'s Definitions on bits: `IngressPrompt_v2{MODEL,LP,V}`, its word body line for line over gates (the body is
`io_shell.parse`, written once over its steps), so its word view is its `_v1` at the same statics; and
`ScanSlots_v2{NS,NP,IDB,K0,K,OCC,LP,MAX,IDS}`, the scan state row's read (`io_shell`'s module docstring), which has no word
view: its port is the row's bits.

A step is `boolean_dsv4_indices.GateOps`' gates for the word primitive `io_shell.WORD` names (`forms`' integer add and signed
compare, a select as `forms.mux`, an AND and a NOT of bits), and bit i of a byte as that wire.  A constant word is constant bits,
so a field's selects of constant words are ANDs with one wire and its sums fold above the field's largest value.  No table or
lookup gate.  Every byte of the body reaches ``ok``.

`ScanSlots_v2` copies each lane's fields out of the row as they are, its ids zero-extended, and computes one thing: the slot
its first position carries on, whose count is the prompt's length minus one (mod 2^32), a borrow chain of 30 ANDs a lane.
Every lane's is computed in every unit, so a unit's lanes but its own Program's are its cost and nobody's input.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import io_shell as X
from verity.ml.definitions.registry.boolean_dsv4_indices import GateOps

BITS = SimpleNamespace(U8=Array(8, BL.BIT), BIT=BL.BIT, I32=Array(32, BL.BIT))


def _parse(S):
    def run(C, *body):
        ok, ids, m = X.parse(GateOps(C), S, list(body))
        return [[ok], *ids, m]
    return run


IngressPromptV2 = EW.specialized("IngressPrompt", 2, ["MODEL", "LP", "V"], _parse, X.ingress_sig(BITS),
                                 doc=f"`IngressPrompt_v1`'s body on bits: {X.parse.__doc__}",
                                 word=lambda b: bind(X.IngressPrompt, **b))


def decrement(C, bits: list) -> list:
    """``bits`` (least significant first) minus one, mod 2^len: bit i flips while every bit below it is 0."""
    out, borrow = [], ONE
    for a in bits:
        out.append(a ^ borrow)
        borrow = C.AND(borrow, a.inv())
    return out


def _scan(S):
    def run(C, *state):
        x = [b for w in state for b in w]

        def word(lo: int, w: int) -> list:
            return x[lo:lo + w] + [ZERO] * (32 - w)

        out = []
        for o, lp, m, ids in zip(S.OCC, S.LP, S.MAX, S.IDS):
            at = [ids + S.IDB * X.position(S.K0, S.K, p) for p in range(S.NP)]
            out += [word(at[0], S.IDB), decrement(C, word(lp, 32)), *[word(a, S.IDB) for a in at[1:]], [ZERO] * 32,
                    word(o, 32), word(lp, 32), word(m, 32)]
        return out
    return run


ScanSlotsV2 = EW.specialized("ScanSlots", 2, ["NS", "NP", "IDB", "K0", "K", "OCC", "LP", "MAX", "IDS"], _scan, X.slots_sig,
                             doc="Each lane's first position's token and slot, occupancy word and head from the scan's state "
                                 "row (`io_shell`'s module docstring; `io_shell.scan_reference`).",
                             refuse=X.slots_refuse,
                             conformance="the network certifier's scan state row as its Programs read their slots, and each "
                                         "slot's first position's token feed")

BOOLEAN_IO_SHELL = [IngressPromptV2, ScanSlotsV2]
