"""`dsa_topk`'s Definitions on bits.  These are the definitions; `dsa_topk`'s word versions are accelerators that
tests/program/test_dsa_topk.py holds equal to them.

    DsaTopKOneHot_v1{N}           (idx: 32 bits) -> N bits: bit j is idx == j (unsigned); traced, no word Definition
    DsaTopKVerdict_v1{N,TOPK}     (members: N bits, scores: N x 32 bits) -> bit; traced, no word Definition
    DsaTopKOrder_v2{N,TOPK}       (scores: N x 32 bits, order: TOPK x 32 bits) -> bit; word view DsaTopKOrder_v1
    Xor3x16_v2, Xor3x32_v2        (a, b, c: 16 or 32 bits) -> 16 or 32 bits; word views their _v1
    DsaWaksmanSwap_v2{W,E}        (s: bit, a, b: W x E bits) -> 2 x W x E bits; word view DsaWaksmanSwap_v1
    DsaWaksmanPick_v2{W,E}        (s: bit, a, b: W x E bits) -> W x E bits; word view DsaWaksmanPick_v1
    DsaWaksmanRows_v2{N,K,W,E}    (sw: bits, rows: N x W x E bits) -> K x W x E bits; word view DsaWaksmanRows_v1
    DsaWaksmanOrder_v2{N,TOPK}    (sw: bits) -> TOPK x 32 bits; word view DsaWaksmanOrder_v1

`DsaTopKOrder_v2` is the selection relation in gates.  Each slot's index is decoded to N bits (`DsaTopKOneHot`, one AND per
output past a shared decoder of about sqrt(N) gates, the index's 32 - b high bits required zero, b = ceil(log2 N)), and the slots'
decodings XORed (one `Xor<TOPK>` per key, no AND) into m_j, the parity of #{s : order[s] = j}.  Then `DsaTopKVerdict`: the
popcount of m is TOPK (an adder tree), and the least member under the tie rule precedes the first non-member (two tournaments
over (key, index) records, `dsa_topk.topk_key`'s key: the f32 word XORed with its sign, its sign bit inverted, no AND).

Why that is the relation: sum_j m_j <= sum_j #{s : order[s] = j} = #{s : order[s] < N} <= TOPK, with equality throughout exactly
when every slot is in range and no two are equal.  The members are then the TOPK slots' indices, and they are the top-TOPK set
exactly when each precedes every non-member.  A slot past N, a negative one or a repeated one fails the count.

`DsaWaksmanRows_v2` is `dsa_topk`'s network with each switch a word-wise `SelectBf16_v2` (one AND a bit) and `Xor3x16_v2` (none):
one AND a payload bit a switch.  `DsaWaksmanOrder_v2` is the same network over the indices' constant bits.

Cost at GLM-5.3's statics (TOPK 2048, rows of kv_lora 512 + rope 64 = 576 bf16; two-input ANDs counted over the
bodies): `DsaTopKOrder_v2{N=2049}` is 4,777,016 (2,211 a slot's decode, 248,888 the verdict) and
2,150 more a key; `DsaWaksmanRows_v2{N=2049,K=2048,W=576,E=16}` is 20,493 switches x 9,216 bits = 188,863,488 and
`DsaWaksmanOrder_v2{N=2049}` 655,776.  A decode step past an I2048 prompt routes and checks the order on the 21 indexer layers and
reads its rows on all 78 (the mux it replaces, `Embedding_v2` per slot, was about N + 30 an output bit: 3.06T a step).
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.definitions.registry import dsa_topk as T

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


def _xor3(C, a, b, c):
    return [x ^ y ^ z for x, y, z in zip(a, b, c)]


Xor3x16 = definition("Xor3x16", 2, _xor3, (("a", BF16), ("b", BF16), ("c", BF16)), BF16, word=T.Xor3x16,
                     doc="a ^ b ^ c on 16 bits: Xor3x16_v1")
Xor3x32 = definition("Xor3x32", 2, _xor3, (("a", I32), ("b", I32), ("c", I32)), I32, word=T.Xor3x32,
                     doc="a ^ b ^ c on 32 bits: Xor3x32_v1")
_SELECT, _XOR3 = {16: SC.SelectBf16, 32: SC.SelectI32}, {16: Xor3x16, 32: Xor3x32}


def _row(S):
    return Array(S.W, Array(S.E, BIT))


def _sig_switch(S, two: bool):
    R = _row(S)
    return (("s", BIT), ("a", R), ("b", R)), Array(2, R) if two else R


def _swap(B, S, s, a, b):
    """`DsaWaksmanSwap_v1` on bits: `SelectBf16_v2` or `SelectI32_v2`(s, b, a) a word (one AND a bit), and `Xor3x<E>_v2`(a, b,
    that) (no AND)."""
    lo = B.batch(_SELECT[S.E], s, b, a, axes=(None, 0, 0))
    return Coll(Array(2, _row(S)), concat([lo.refs, B.batch(_XOR3[S.E], a, b, lo, axes=(0, 0, 0)).refs]))


def _pick(B, S, s, a, b):
    """`DsaWaksmanPick_v1` on bits: `SelectBf16_v2` or `SelectI32_v2`(s, b, a) a word."""
    return B.batch(_SELECT[S.E], s, b, a, axes=(None, 0, 0))


DsaWaksmanSwapV2 = EW.on_bits("DsaWaksmanSwap", 2, ["W", "E"], lambda S: _sig_switch(S, True), _swap, T.DsaWaksmanSwap)
DsaWaksmanPickV2 = EW.on_bits("DsaWaksmanPick", 2, ["W", "E"], lambda S: _sig_switch(S, False), _pick, T.DsaWaksmanPick)


def _sig_net(S):
    return (("sw", Array(T.waksman_switches(S.N, S.K), BIT)), ("rows", Array(S.N, _row(S)))), Array(S.K, _row(S))


def _net(B, S, sw, rows):
    """`DsaWaksmanRows_v1` on bits: `dsa_topk.network` of `DsaWaksmanSwap_v2` and `DsaWaksmanPick_v2`, one AND a payload bit a
    switch."""
    return T.network(B, S.N, S.K, _row(S), sw, rows, bind(DsaWaksmanSwapV2, W=S.W, E=S.E), bind(DsaWaksmanPickV2, W=S.W, E=S.E),
                     lambda n, k: bind(DsaWaksmanRowsV2, N=n, K=k, W=S.W, E=S.E))


DsaWaksmanRowsV2 = EW.on_bits("DsaWaksmanRows", 2, ["N", "K", "W", "E"], _sig_net, _net, T.DsaWaksmanRows)


def _sig_net_order(S):
    return (("sw", Array(T.waksman_switches(S.N, S.TOPK), BIT)),), Array(S.TOPK, I32)


def _net_order(B, S, sw):
    """`DsaWaksmanOrder_v1` on bits: `DsaWaksmanRows_v2{N,TOPK,1,32}` over the indices' constant bits."""
    zero, one = B.call(BL.constant(0)), B.call(BL.constant(1))
    tags = concat([(one if j >> i & 1 else zero).refs for j in range(S.N) for i in range(32)])
    return Coll(Array(S.TOPK, I32), B.call(bind(DsaWaksmanRowsV2, N=S.N, K=S.TOPK, W=1, E=32), sw,
                                           Coll(Array(S.N, Array(1, I32)), tags)).refs)


DsaWaksmanOrderV2 = EW.on_bits("DsaWaksmanOrder", 2, ["N", "TOPK"], _sig_net_order, _net_order, T.DsaWaksmanOrder)

BOOLEAN_DSA_TOPK = [DsaTopKOneHot, DsaTopKVerdict, DsaTopKOrderV2, Xor3x16, Xor3x32, DsaWaksmanSwapV2, DsaWaksmanPickV2,
                    DsaWaksmanRowsV2, DsaWaksmanOrderV2]
