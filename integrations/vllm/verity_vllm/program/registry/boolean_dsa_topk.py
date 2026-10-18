"""`dsa_topk`'s Definitions on bits.  These are the definitions; `dsa_topk`'s word versions are accelerators that
tests/program/test_dsa_topk.py holds equal to them.

    DsaTopKOneHot_v1{N}           (idx: 32 bits) -> N bits: bit j is idx == j (unsigned); traced, no word Definition
    DsaTopKVerdict_v1{N,TOPK}     (members: N bits, scores: N x 32 bits) -> bit; traced, no word Definition
    DsaTopKOrder_v2{N,TOPK}       (scores: N x 32 bits, order: TOPK x 32 bits) -> bit; word view DsaTopKOrder_v1
    DsaGatherRows_v2{N,TOPK,W}    (order: TOPK x 32 bits, rows: N x W x 16 bits) -> TOPK x W x 16 bits; word view DsaGatherRows_v1

`DsaTopKOrder_v2` is the selection relation in gates.  Each slot's index is decoded to N bits (`DsaTopKOneHot`, one AND per
output past a shared decoder of about sqrt(N) gates, the index's 32 - b high bits required zero, b = ceil(log2 N)), and the slots'
decodings XORed (one `Xor<TOPK>` per key, no AND) into m_j, the parity of #{s : order[s] = j}.  Then `DsaTopKVerdict`: the
popcount of m is TOPK (an adder tree), and the least member under the tie rule precedes the first non-member (two tournaments
over (key, index) records, `dsa_topk.topk_key`'s key: the f32 word XORed with its sign, its sign bit inverted, no AND).

Why that is the relation: sum_j m_j <= sum_j #{s : order[s] = j} = #{s : order[s] < N} <= TOPK, with equality throughout exactly
when every slot is in range and no two are equal.  The members are then the TOPK slots' indices, and they are the top-TOPK set
exactly when each precedes every non-member.  A slot past N, a negative one or a repeated one fails the count.

`DsaGatherRows_v2` is `boolean_gather.Embedding_v2{V=N, H=W}` per slot: W gathers of 16 bits by the slot's 32-bit index.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity_vllm.program.registry import boolean_gather as BGa
from verity_vllm.program.registry import dsa_topk as T

BIT = BL.BIT
F32 = I32 = Array(32, BIT)
BF16 = Array(16, BIT)


def index_bits(N: int) -> int:
    """b, the index bits that address N keys: ceil(log2 N), at least 1."""
    return max(1, (N - 1).bit_length())


def decode(C, bits: list, n: int, enable) -> list:
    """``enable`` AND (``bits`` == j) for j < n, ``bits`` LSB first, n <= 2^len(bits): the low half's decoding times the high
    half's, ``enable`` folded into the high half."""
    if not bits:
        return [enable]
    if len(bits) == 1:
        return [C.AND(enable, NOT(bits[0])), C.AND(enable, bits[0])][:n]
    h = len(bits) // 2
    lo = decode(C, bits[:h], min(n, 1 << h), ONE)
    hi = decode(C, bits[h:], -(-n // (1 << h)), enable)
    return [C.AND(hi[j >> h], lo[j & ((1 << h) - 1)]) for j in range(n)]


def _one_hot(S):
    """`DsaTopKOneHot_v1{N}`: bit j is (idx == j) for the unsigned 32-bit idx, its high 32 - b bits all zero."""
    T.check_statics(S.N, 1)
    b = index_bits(S.N)

    def build(C, idx):
        return decode(C, list(idx[:b]), S.N, NOT(or_reduce(C, list(idx[b:]))))
    return build


DsaTopKOneHot = EW.specialized("DsaTopKOneHot", 1, ["N"], _one_hot, lambda S: ((("idx", I32),), Array(S.N, BIT)),
                               doc=_one_hot.__doc__)


def popcount(C, bits: list) -> list:
    """The number of set ``bits`` (LSB first): an adder tree, pairs at each level."""
    nums = [[x] for x in bits]
    while len(nums) > 1:
        nxt = [add(C, nums[i], nums[i + 1], width=max(len(nums[i]), len(nums[i + 1])), carry_out=True)
               for i in range(0, len(nums) - 1, 2)]
        nums = nxt + nums[len(nums) - len(nums) % 2:]
    return nums[0]


def equals(C, x: list, value: int):
    if value >> len(x):
        return ZERO
    return and_reduce(C, [b if (value >> i) & 1 else NOT(b) for i, b in enumerate(x)])


def key(s: list) -> list:
    """`dsa_topk.topk_key` on bits: bits 0..30 XORed with the sign, the sign inverted."""
    return [x ^ s[31] for x in s[:31]] + [NOT(s[31])]


def tournament(C, recs: list, take_right) -> tuple:
    """The record (valid, key, index) the tree of `take_right` picks among ``recs``, given in index order: at each meeting the left
    holds the lower indices, and the right wins when it is valid and the left is not or ``take_right(left key, right key)``."""
    while len(recs) > 1:
        nxt = []
        for i in range(0, len(recs) - 1, 2):
            (lv, lk, li), (rv, rk, ri) = recs[i], recs[i + 1]
            s = C.AND(rv, OR(C, NOT(lv), take_right(C, lk, rk)))
            nxt.append((OR(C, lv, rv), mux(C, s, rk, lk), mux(C, s, ri, li)))
        recs = nxt + recs[len(recs) - len(recs) % 2:]
    return recs[0]


def _greater(C, lk, rk):
    """The right's key is greater: the left's first (it holds the lower indices) on a tie."""
    return sub(C, lk, rk, 32)[1]


def _not_greater(C, lk, rk):
    """The right's key is at most the left's: on a tie the right, whose higher index comes later under the tie rule."""
    return NOT(sub(C, lk, rk, 32)[1])


def _verdict(S):
    """`DsaTopKVerdict_v1{N,TOPK}`: popcount(members) = TOPK, and the last member under the tie rule (`tournament` of the members
    by `_not_greater`) precedes the first non-member (of the rest by `_greater`): its key is greater, or equal at a lower index; true
    when every key is a member."""
    T.check_statics(S.N, S.TOPK)
    b = index_bits(S.N)

    def build(C, members, *scores):
        keys = [key(list(s)) for s in scores]
        idx = [const_bits(j, b) for j in range(S.N)]
        last = tournament(C, [(m, k, i) for m, k, i in zip(members, keys, idx)], _not_greater)
        first = tournament(C, [(NOT(m), k, i) for m, k, i in zip(members, keys, idx)], _greater)
        precedes = sub(C, [NOT(x) for x in first[2]] + first[1], [NOT(x) for x in last[2]] + last[1], b + 32)[1]
        ordered = C.AND(last[0], OR(C, NOT(first[0]), precedes))
        return [C.AND(equals(C, popcount(C, list(members)), S.TOPK), ordered)]
    return build


DsaTopKVerdict = EW.specialized("DsaTopKVerdict", 1, ["N", "TOPK"], _verdict,
                                lambda S: ((("members", Array(S.N, BIT)), ("scores", Array(S.N, F32))), BIT), doc=_verdict.__doc__)


def _sig_order(S):
    return (("scores", Array(S.N, F32)), ("order", Array(S.TOPK, I32))), BIT


def _order(B, S, scores, order):
    """`DsaTopKOrder_v1` on bits: each slot's `DsaTopKOneHot_v1{N}`, the slots' decodings XORed per key (`Xor<TOPK>`), and
    `DsaTopKVerdict_v1{N,TOPK}` over the parities and the scores."""
    T.check_statics(S.N, S.TOPK)
    hits = B.batch(bind(DsaTopKOneHot, N=S.N), order, axes=(0,))
    members = hits[0] if S.TOPK == 1 else B.batch(BL.xor_gate(S.TOPK), hits, axes=(1,))
    return B.call(bind(DsaTopKVerdict, N=S.N, TOPK=S.TOPK), Coll(Array(S.N, BIT), members.refs), scores)


DsaTopKOrderV2 = EW.on_bits("DsaTopKOrder", 2, ["N", "TOPK"], _sig_order, _order, T.DsaTopKOrder)


def _sig_rows(S):
    return (("order", Array(S.TOPK, I32)), ("rows", Array(S.N, Array(S.W, BF16)))), Array(S.TOPK, Array(S.W, BF16))


def _rows(B, S, order, rows):
    """`DsaGatherRows_v1` on bits: slot s is `Embedding_v2{V=N, H=W}`(order[s], rows)."""
    return B.batch(bind(BGa.Embedding, V=S.N, H=S.W), order, rows, axes=(0, None))


DsaGatherRowsV2 = EW.on_bits("DsaGatherRows", 2, ["N", "TOPK", "W"], _sig_rows, _rows, T.DsaGatherRows)

BOOLEAN_DSA_TOPK = [DsaTopKOneHot, DsaTopKVerdict, DsaTopKOrderV2, DsaGatherRowsV2]
