"""`dsv4_decode_indices`' Definitions on bits, each the same function as its `_v1` at the same statics (its word view):

    Dsv4GlobalTopkSlot_v2{BS,NB}       (x: 32 bits, blocks: NB words) -> (32 bits, live) word view: Dsv4GlobalTopkSlot_v1
    Dsv4GlobalTopkLen_v2{TOPK}         (live: TOPK bits, valid: 8 bits) -> 32 bits       word view: Dsv4GlobalTopkLen_v1
    Dsv4GlobalTopk_v2{TOPK,BS,NB}      the row: `dsv4_decode_indices.row` over the two   word view: Dsv4GlobalTopk_v1

The slot is wiring around one multiplexer tree.  BS = 2^k, so on a valid x (sign bit 0) x / BS is x's bits k..30 and x % BS its
low k bits, and bn * BS + x % BS is those k bits under bn's low 32 - k: the slot gathers only the blocks' low 32 - k bits, by
x's bits k..k+m-1, m = ceil(log2 NB) (`verity.ml.circuits.boolean.gather`, an index in [NB, 2^m) selecting the fill's low bits).
`GatherI32x{NB}_v1`'s fill is -1, so a block past the row makes the slot's high 32 - k bits ones, and so does a negative x, whose
slot is -1: each high bit is ORed with the OR of x's bits from k + m up (its sign among them), each low bit with the sign.  NB
words cost (NB - 1 + p) * (32 - k) ANDs, p the levels at which an odd count meets the fill, then 32 - k - m for the OR and 32 for
the ORs; its live is the sign bit's complement.  The length is the popcount of the TOPK lives (`forms.compress_columns`), each
bit ANDed with the OR of the byte's 8 bits.  No table or lookup gate.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, compress_columns, or_reduce
from verity.ml.circuits.boolean.gather import gather
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import dsv4_decode_indices as X

#: the length's lives are 1-bit elements, so that TOPK of them are TOPK 1-bit words (`BL.word_type`), not one TOPK-bit word (as
#: `boolean_marlin_moe.CUT`): `Dsv4GlobalTopkLen_v2` is reached through its row, not the switch, whose bit form of
#: Array<TOPK, bit> is that one word
BITS = SimpleNamespace(I32=Array(32, BL.BIT), U8=Array(8, BL.BIT), BIT=BL.BIT, LIVE=Array(1, BL.BIT))


def _slot(S):
    k = X.check_slot_statics(S.BS, S.NB)
    m = (S.NB - 1).bit_length()
    fill = X.FILL & ((1 << (32 - k)) - 1)

    def build(C, x, *blocks):
        bn = gather(C, x[k:k + m], [w[:32 - k] for w in blocks], fill)
        past = or_reduce(C, x[k + m:])
        return [[OR(C, b, x[31]) for b in x[:k]] + [OR(C, b, past) for b in bn], [NOT(x[31])]]
    return build


def _len(S):
    X.check_len_statics(S.TOPK)
    width = S.TOPK.bit_length()

    def build(C, *elements):
        live, valid = elements[:S.TOPK], elements[S.TOPK]
        count = compress_columns(C, [[b for (b,) in live]])[:width]
        nz = or_reduce(C, valid)
        return [C.AND(nz, b) for b in count] + [ZERO] * (32 - len(count))
    return build


Dsv4GlobalTopkSlotV2 = EW.specialized("Dsv4GlobalTopkSlot", 2, ["BS", "NB"], _slot, X.slot_sig(BITS),
                                      doc=f"`Dsv4GlobalTopkSlot_v1` on bits (the module docstring): {X.Dsv4GlobalTopkSlot.__doc__}",
                                      word=lambda b: bind(X.Dsv4GlobalTopkSlot, **b))
Dsv4GlobalTopkLenV2 = EW.specialized("Dsv4GlobalTopkLen", 2, ["TOPK"], _len, X.len_sig(BITS),
                                     doc=f"`Dsv4GlobalTopkLen_v1` on bits (the module docstring): {X.Dsv4GlobalTopkLen.__doc__}",
                                     word=lambda b: bind(X.Dsv4GlobalTopkLen, **b))


def _row(B, S, topk, blocks, valid):
    """`Dsv4GlobalTopk_v1`'s body on bits: `dsv4_decode_indices.row` over `Dsv4GlobalTopkSlot_v2` and `Dsv4GlobalTopkLen_v2`."""
    return X.row(B, S, BITS, Dsv4GlobalTopkSlotV2, Dsv4GlobalTopkLenV2, topk, blocks, valid)


Dsv4GlobalTopkV2 = EW.on_bits("Dsv4GlobalTopk", 2, ["TOPK", "BS", "NB"], X.row_sig(BITS), _row, X.Dsv4GlobalTopk)

BOOLEAN_DSV4_DECODE_INDICES = [Dsv4GlobalTopkSlotV2, Dsv4GlobalTopkLenV2, Dsv4GlobalTopkV2]
