"""`dsv4_decode_metadata`'s Definitions on bits, each the same function as its `_v1` at the same statics (its word view):

    Dsv4SwaWindow_v2{SW,BS,NB}          the window's body on gates                                  word view: Dsv4SwaWindow_v1
    Dsv4SwaLanes_v2{SW,BS}              the lanes' body on gates                                    word view: Dsv4SwaLanes_v1
    Dsv4SwaIndices_v2{SW,BS,NB}         the row: `dsv4_decode_metadata.swa_row` over the two        word view: Dsv4SwaIndices_v1
    Dsv4C128aSlot_v2{BS}                (x: BS x 32 bits, bn: 32 bits) -> (BS x 32 bits, BS bits)   word view: Dsv4C128aSlot_v1
    Dsv4C128aPastSlot_v2{BS}            (x: 32 bits) -> (32 bits, live)                             word view: Dsv4C128aPastSlot_v1
    Dsv4C128aLen_v2{W}                  (live: W bits, slot: 64 bits) -> 32 bits                    word view: Dsv4C128aLen_v1
    Dsv4C128aGlobalTopk_v2{W,R,BS,NB}   the row: `dsv4_decode_metadata.c128a_row` over              word view: Dsv4C128aGlobalTopk_v1
                                        `Dsv4ShortContextTopK_v2` and the three

The window and the lanes are their word bodies (`dsv4_decode_metadata.window` and `.lanes`, written once over their steps) line
for line on gates: `boolean_dsv4_indices.GateOps`' integer add, subtract, signed compare and select, the AND of a constant mask,
C's division by 2^k as (x + (x < 0 ? 2^k - 1 : 0)) >> k (arithmetic) and its remainder as x - (x / 2^k) * 2^k, the product by 2^k
as a shift, the byte's test as the OR of its 8 bits and the lookup `GatherI32x{n}_v1` as `forms`' multiplexer tree on the index's
low ceil(log2 n) bits, each bit ORed with the OR of the index's higher bits (its sign among them: the fill -1).  The tracer folds
constants and keeps only the gates that reach an output, so the window's start max(s0, 0) is AND(not s0_31, s0_i) with the sign
the constant 0, its division and remainder by BS are wires, and each of its K lookups pays only for the low 32 - log2 BS bits the
product keeps, (NB - 1 + p) * (32 - log2 BS) ANDs, p the levels at which an odd count meets the fill.  A lane's q = (r0 & (BS -
1)) + j is 0 from bit_length(BS - 1 + j) up, so its lookup is over the K words by at most two bits, folded where those are
constant, and its division and remainder are wires.

The ratio-128 slot is wiring: for x >= 0 (sign 0) x % BS is x's low k bits (BS = 2^k) and bn * BS + x % BS those under bn's low
32 - k bits, and x < 0 makes the slot -1, so every bit is ORed with x's sign: 32 ANDs a lane, its live the sign's complement.  Past
the row bn's bits are the constant ones of -1, so a lane there costs k ANDs.  The length is the popcount of the W lives
(`forms.compress_columns`), each bit ANDed with the complement of the slot's sign.  No table or lookup gate.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, compress_columns, or_reduce, sub
from verity.ml.circuits.boolean.gather import gather
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import boolean_dsv4_indexer as BI
from verity_vllm.program.registry import boolean_dsv4_indices as BX
from verity_vllm.program.registry import dsv4_decode_metadata as X

#: the length's lives are 1-bit elements, as `boolean_dsv4_decode_indices.BITS`' are, and so are a slot's (a tuple of bits)
BITS = SimpleNamespace(I32=Array(32, BL.BIT), I64=Array(64, BL.BIT), U8=Array(8, BL.BIT), BIT=BL.BIT, LIVE=Array(1, BL.BIT))


def _log2(n: int) -> int:
    if not (n >= 1 and n & (n - 1) == 0):
        raise ValueError(f"a power of two, not {n}")
    return n.bit_length() - 1


class GateOps(BX.GateOps):
    """`boolean_dsv4_indices.GateOps` with `dsv4_decode_metadata.Ops`' ``gather``, ``nonzero``, ``div``, ``rem`` and ``mul``."""

    def gather(self, idx, words):
        m = (len(words) - 1).bit_length()
        past = or_reduce(self.C, list(idx[m:]))
        return [OR(self.C, b, past) for b in gather(self.C, list(idx[:m]), [list(w) for w in words], X.FILL)]

    def nonzero(self, byte):
        return or_reduce(self.C, list(byte))

    def div(self, x, n: int):
        k = _log2(n)
        u = add(self.C, list(x), [x[31]] * k, width=32)
        return list(u[k:]) + [u[31]] * k

    def rem(self, x, n: int):
        return sub(self.C, list(x), self.mul(self.div(x, n), n), 32)[0]

    def mul(self, x, n: int):
        k = _log2(n)
        return [ZERO] * k + list(x[:32 - k])


def _window(S):
    X.check_swa_statics(S.SW, S.BS, S.NB, "Dsv4SwaWindow")

    def build(C, t, qs, qe, seq, *rest):
        r0, length, base = X.window(GateOps(C), S, t, qs, qe, seq, list(rest[:S.NB]), rest[S.NB])
        return [r0, length, *base]
    return build


def _lanes(S):
    X.check_swa_statics(S.SW, S.BS, 1, "Dsv4SwaLanes")

    def build(C, r0, length, *base):
        return X.lanes(GateOps(C), S, r0, length, list(base))
    return build


Dsv4SwaWindowV2 = EW.specialized("Dsv4SwaWindow", 2, ["SW", "BS", "NB"], _window, X.window_sig(BITS),
                                 doc=f"`Dsv4SwaWindow_v1`'s body on bits (the module docstring): {X.window.__doc__}",
                                 word=lambda b: bind(X.Dsv4SwaWindow, **b))
Dsv4SwaLanesV2 = EW.specialized("Dsv4SwaLanes", 2, ["SW", "BS"], _lanes, X.lanes_sig(BITS),
                                doc=f"`Dsv4SwaLanes_v1`'s body on bits (the module docstring): {X.lanes.__doc__}",
                                word=lambda b: bind(X.Dsv4SwaLanes, **b))


def _swa_row(B, S, t, qs, qe, seq, blocks, valid):
    """`Dsv4SwaIndices_v1`'s body on bits: `dsv4_decode_metadata.swa_row` over `Dsv4SwaWindow_v2` and `Dsv4SwaLanes_v2`."""
    return X.swa_row(B, S, BITS, Dsv4SwaWindowV2, Dsv4SwaLanesV2, t, qs, qe, seq, blocks, valid)


Dsv4SwaIndicesV2 = EW.on_bits("Dsv4SwaIndices", 2, ["SW", "BS", "NB"], X.swa_sig(BITS), _swa_row, X.Dsv4SwaIndices)


def _slot_bits(C, k: int, x, bn) -> list:
    return [OR(C, b, x[31]) for b in list(x[:k]) + list(bn[:32 - k])]


def _c128a_slot(S):
    k = X.check_c128a_slot_statics(S.BS)

    def build(C, *elements):
        xs, bn = elements[:S.BS], elements[S.BS]
        return [_slot_bits(C, k, x, bn) for x in xs] + [[NOT(x[31])] for x in xs]
    return build


def _c128a_past_slot(S):
    k = X.check_c128a_slot_statics(S.BS)

    def build(C, x):
        return [_slot_bits(C, k, x, [ONE] * 32), [NOT(x[31])]]
    return build


def _c128a_len(S):
    X.check_c128a_len_statics(S.W)
    width = S.W.bit_length()

    def build(C, *elements):
        live, slot = elements[:S.W], elements[S.W]
        count = compress_columns(C, [[b for (b,) in live]])[:width]
        ok = NOT(slot[63])
        return [C.AND(ok, b) for b in count] + [ZERO] * (32 - len(count))
    return build


Dsv4C128aSlotV2 = EW.specialized("Dsv4C128aSlot", 2, ["BS"], _c128a_slot, X.c128a_slot_sig(BITS),
                                 doc=f"`Dsv4C128aSlot_v1` on bits (the module docstring): {X.c128a_slot.__doc__}",
                                 word=lambda b: bind(X.Dsv4C128aSlot, **b))
Dsv4C128aPastSlotV2 = EW.specialized("Dsv4C128aPastSlot", 2, ["BS"], _c128a_past_slot, X.c128a_past_slot_sig(BITS),
                                     doc=f"`Dsv4C128aPastSlot_v1` on bits (the module docstring): {X.Dsv4C128aPastSlot.__doc__}",
                                     word=lambda b: bind(X.Dsv4C128aPastSlot, **b))
Dsv4C128aLenV2 = EW.specialized("Dsv4C128aLen", 2, ["W"], _c128a_len, X.c128a_len_sig(BITS),
                                doc=f"`Dsv4C128aLen_v1` on bits (the module docstring): {X.Dsv4C128aLen.__doc__}",
                                word=lambda b: bind(X.Dsv4C128aLen, **b))


def _c128a_row(B, S, position, slot, blocks):
    """`Dsv4C128aGlobalTopk_v1`'s body on bits: `dsv4_decode_metadata.c128a_row` over `Dsv4ShortContextTopK_v2`, `Dsv4C128aSlot_v2`,
    `Dsv4C128aPastSlot_v2` and `Dsv4C128aLen_v2`."""
    return X.c128a_row(B, S, BITS, BI.Dsv4ShortContextTopKV2, Dsv4C128aSlotV2, Dsv4C128aPastSlotV2, Dsv4C128aLenV2, position, slot,
                       blocks)


Dsv4C128aGlobalTopkV2 = EW.on_bits("Dsv4C128aGlobalTopk", 2, ["W", "R", "BS", "NB"], X.c128a_sig(BITS), _c128a_row,
                                   X.Dsv4C128aGlobalTopk)

BOOLEAN_DSV4_DECODE_METADATA = [Dsv4SwaWindowV2, Dsv4SwaLanesV2, Dsv4SwaIndicesV2, Dsv4C128aSlotV2, Dsv4C128aPastSlotV2,
                                Dsv4C128aLenV2, Dsv4C128aGlobalTopkV2]
