"""`DsaTopKMaskCounted_v1{N,TOPK}` (and its Boolean version `_v2`) past `WHOLE_MAX` keys, as units of at most a Flock block each:
`plan` lists them with their wiring, `cut` counts them by Definition (`dsa_topk_mask.CUTS`, `boolean_dsa_topk_mask.CUTS`), and
`values` computes every unit's outputs from a row, the twin the tests hold each unit to.

A unit's block is mostly the hashes of the rows it reads, and the mask's threshold reads every key, so past about 43k keys no
unit can compute it alone.  It is computed across units instead, by `boolean_dsa_topk_mask`'s radix select taken 8 bits of the
keys at a time, over blocks of the row of `BLOCK` keys (a last block shorter than TOPK joins the one before).  Before round k
(k = 0..3) the threshold's key is known above its digit k (the prefix, its top 8k bits) and r is how many members the keys level
with it must supply (TOPK before round 0).  Round k is

* ``DsaTopKDigitCount_v1{N,AT,K}``, per block (keys AT..AT+N-1): of its keys j < n level with the prefix, how many have each of
  the 256 values of digit k (`DsaTopKDigitOneHot_v1{K}` per key, a `DsaTopKPopCount_v1{N}` per value);
* ``DsaTopKDigitPick_v1{M,K,TOPK}``, over the M blocks' counts, block b's its own port ``counts_b``: with T(v) the level keys
  at digit v or above, the threshold's
  digit is the largest v with T(v) >= r, else 0 (fewer than r level keys: the row has fewer than TOPK keys below n), and r drops
  by T(digit + 1), the level keys the set takes above it.  Those are the digit and the r the radix select's 8 passes over the
  digit's bits leave.  Round 3's pick returns the threshold's key and, per block, how many of its keys equal to that key the set
  takes: r less those of the blocks before it, at least 0.

Then ``DsaTopKMaskBlock_v1{N,AT,TOPK}``, per block: its keys j < n above the threshold's key and, of those equal to it, the first
r_b by index (`DsaRadixTies_v1`), the block's part of `DsaRadixTies_v1`'s mask over the row, since its equal keys come after
those of the blocks before it.  Every unit reads its own block of the scores and n, the Call's operands; each other input is
one whole output row of a unit before it (`Unit.wires`), each port's own: the prefix and r of the pick before, a block's counts,
the threshold's key and r_b.  The Call's mask is the mask units' outputs in block order.  Units a row: 5M + 4, M = ceil(N / `BLOCK`) blocks (44 at
131,072 keys).

Every port of a unit is whole u16 words, as a row of `verity.commitments.windows.interface.layout` is: a mask of n bits is
`words`(n), Array<ceil(n/16), Array<16, BIT>>, bit j word j // 16's bit j mod 16, and a prefix or an r is `wide` of its bits.  A
writer sets the bits past its value's to 0 and a reader ignores them.  ``DsaTopKMaskWhole_v1{N,TOPK}`` is the counted mask up to
`WHOLE_MAX` keys, a unit, in u16 words.

``DsaTopKMaskBlock_v2`` and ``DsaTopKMaskWhole_v2`` are the two mask units with every part that grows with the row as Calls
to its steps and not a gate for each key: their ties by `DsaRadixTies_v2`, their keys below n by `DsaRowValid_v2` (past a
block's start, of `DsaRowValidOffset_v1{AT}`) and the whole mask's passes by `DsaRadixPass_v2`.  ``DsaTopKDigitPick_v2`` is
the pick as Calls to its steps, a batch over the 255 digits it weighs.  The same masks and the same picks, described in a
fraction of the bytes.  `plan` and `cut` name the `_v1` units.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from types import SimpleNamespace
from typing import Any

import numpy as np

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ZERO, add, and_reduce, compress_columns, const_bits, sub
from verity.circuits.defs import CompositeDefinition, bind
from verity.definitions.ops.boolean import elementwise as EW
from verity.circuits.refs import Coll, concat, strided_view
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import boolean_dsa_topk_mask as BM
from verity.definitions.registry.boolean_dsa_topk import decode, key
from verity.definitions.registry.dsa_topk import check_statics

BIT = BL.BIT
F32 = I32 = Array(32, BIT)

#: keys a block: the most, a power of two, at which a block's count unit, its mask unit and a pick each fit Flock's block
#: (circuit-check's block rule at the sidecar's RANGES spots)
BLOCK = 16384
#: the largest N the Call is checked and proved whole at, the longest served class at which `DsaTopKMaskCounted_v2` fits
#: Flock's block whole; past it, `cut`
WHOLE_MAX = 40960
DIGIT, ROUNDS = 8, 4
BINS = 1 << DIGIT


def _rb(TOPK: int) -> int:
    return TOPK.bit_length()


def wide(bits: int) -> int:
    """``bits`` rounded up to whole u16 words."""
    return -(-bits // 16) * 16


def words(n: int) -> Array:
    """A mask of ``n`` bits as whole u16 words: Array<ceil(n/16), Array<16, BIT>>."""
    return Array(-(-n // 16), Array(16, BIT))


def low(c: Coll, bits: int) -> Coll:
    """The first ``bits`` bits of the Boolean collection ``c``, the value a reader takes from a port wider than it."""
    return Coll(Array(bits, BIT), c.refs.slice(0, bits))


def to_words(B, mask: Coll, n: int) -> Coll:
    """The mask of ``n`` bits ``mask`` (a Coll of ``B``) as `words`(n), the last word's bits past n 0."""
    pad = wide(n) - n
    if not pad:
        return Coll(words(n), mask.refs)
    zero = B.call(BL.constant(0))
    return Coll(words(n), concat([mask.refs, BM._rep(zero, pad)]))


def _pad(bits: list, width: int) -> list:
    return list(bits) + [ZERO] * (width - len(bits))


def _row_valid_from(S):
    """`DsaRowValidFrom_v1{N,AT}`: bit j is n > AT + j (n u32), a block's part of `DsaRowValid_v1`: `DsaRowValid_v1{N}`'s trie
    over n - AT, 0 when n < AT."""
    if S.AT < 1:
        raise ValueError(f"DsaRowValidFrom: AT >= 1 (AT={S.AT}); at 0 it is DsaRowValid_v1")
    trie = BM._row_valid(SimpleNamespace(N=S.N))

    def build(C, n):
        d, borrow = sub(C, list(n), const_bits(S.AT, 32), 32)
        return trie(C, [C.AND(NOT(borrow), x) for x in d])
    return build


DsaRowValidFrom = EW.specialized("DsaRowValidFrom", 1, ["N", "AT"], _row_valid_from,
                                 lambda S: ((("n", I32),), Array(S.N, BIT)), doc=_row_valid_from.__doc__)


def _valid(B, N: int, AT: int, n):
    return B.call(bind(BM.DsaRowValid, N=N), n) if AT == 0 else B.call(bind(DsaRowValidFrom, N=N, AT=AT), n)


def _row_offset(S):
    """`DsaRowValidOffset_v1{AT}`: n - AT where n >= AT, else 0 (n u32): what `DsaRowValidFrom_v1{N,AT}` runs its trie over."""
    def build(C, n):
        d, borrow = sub(C, list(n), const_bits(S.AT, 32), 32)
        return [[C.AND(NOT(borrow), x) for x in d]]
    return build


DsaRowValidOffset = BM._piece("DsaRowValidOffset", ["AT"], _row_offset, lambda S: ((("n", I32),), I32))


def _valid_calls(B, N: int, AT: int, n):
    """`_valid` as Calls: `DsaRowValid_v2{N}` of n, past the row's start of `DsaRowValidOffset_v1{AT}` of n."""
    return B.call(bind(BM.DsaRowValidV2, N=N), n if AT == 0 else B.call(bind(DsaRowValidOffset, AT=AT), n))


def _digit_one_hot(S):
    """`DsaTopKDigitOneHot_v1{K}`: of a key (`DsaTopKKey_v2` of s), bit v is (valid AND its top 8K bits are the prefix AND its
    digit K, bits 24 - 8K .. 31 - 8K, is v)."""
    lo = 32 - DIGIT * (S.K + 1)

    def build(C, s, valid, prefix=()):
        k = key(list(s))
        level = and_reduce(C, [NOT(a ^ b) for a, b in zip(k[lo + DIGIT:], prefix)])
        return [decode(C, k[lo:lo + DIGIT], BINS, C.AND(valid[0], level) if S.K else valid[0])]
    return build


DsaTopKDigitOneHot = EW.specialized(
    "DsaTopKDigitOneHot", 1, ["K"], _digit_one_hot,
    lambda S: ((("s", F32), ("valid", BIT), *((("prefix", Array(DIGIT * S.K, BIT)),) if S.K else ())), Array(BINS, BIT)),
    doc=_digit_one_hot.__doc__)


def _pop_count(S):
    """`DsaTopKPopCount_v1{N}`: how many of the N bits are set, u32 (a carry-save tree)."""
    def build(C, x):
        return [(compress_columns(C, [list(x)]) + [ZERO] * 32)[:32]]
    return build


DsaTopKPopCount = EW.specialized("DsaTopKPopCount", 1, ["N"], _pop_count, lambda S: ((("x", Array(S.N, BIT)),), I32),
                                 doc=_pop_count.__doc__)


def _count_sig(S):
    return ((("scores", Array(S.N, F32)), ("n", I32), *((("prefix", Array(wide(DIGIT * S.K), BIT)),) if S.K else ())),
            Array(BINS, I32))


def _count(B, S, scores, n, *prefix):
    """`DsaTopKDigitCount_v1{N,AT,K}`, keys AT..AT+N-1 of a row: for each v < 256, how many of them below n are level with the
    prefix (its low 8K bits) and have digit K v (`DsaTopKDigitOneHot_v1{K}` per key, `DsaTopKPopCount_v1{N}` per v)."""
    prefix = tuple(low(p, DIGIT * S.K) for p in prefix)
    valid = _valid(B, S.N, S.AT, n)
    hot = B.batch(bind(DsaTopKDigitOneHot, K=S.K), scores, valid, *prefix, axes=(0, 0, *(None for _ in prefix)))
    cols = Coll(Array(BINS, Array(S.N, BIT)), strided_view(hot, 0, S.N, 1, BINS, BINS * S.N))
    return B.batch(bind(DsaTopKPopCount, N=S.N), cols, axes=(0,))


DsaTopKDigitCount = CompositeDefinition("DsaTopKDigitCount", 1, ("N", "AT", "K"), _count_sig, _count, doc=_count.__doc__,
                                        conformance="exact-model-tested")


def _pick(S):
    """`DsaTopKDigitPick_v1{M,K,TOPK}`: round K's digit from the M blocks' counts (``counts_b`` block b's), with T(v) the level
    keys at digit v or above: the largest v with T(v) >= r, else 0; returns (prefix with the digit below it, r - T(digit + 1)),
    and at the last round (the threshold's key, r_b for each block b: that r less the keys equal to the key in blocks 0..b-1,
    at least 0).  It reads the prefix's low 8K bits and r's low RB, and writes 0 past each value's bits."""
    RB, last = _rb(S.TOPK), S.K == ROUNDS - 1
    if not 0 <= S.K < ROUNDS or S.M < 1:
        raise ValueError(f"DsaTopKDigitPick: 0 <= K < {ROUNDS} and M >= 1 (M={S.M}, K={S.K})")

    def build(C, *els):
        cnt = [[list(els[b * BINS + v]) for v in range(BINS)] for b in range(S.M)]
        prefix, r = ((list(els[S.M * BINS])[:DIGIT * S.K], list(els[S.M * BINS + 1])[:RB]) if S.K
                     else ((), const_bits(S.TOPK, RB)))
        h = [cnt[0][v] for v in range(BINS)]
        for b in range(1, S.M):
            h = [add(C, h[v], cnt[b][v], width=32) for v in range(BINS)]
        T = [None] * BINS + [[ZERO] * 32]
        for v in reversed(range(BINS)):
            T[v] = add(C, T[v + 1], h[v], width=32)
        ge = [None] + [NOT(sub(C, T[v], r, 32)[1]) for v in range(1, BINS)]
        sel = [NOT(ge[1])] + [C.AND(ge[v], NOT(ge[v + 1])) for v in range(1, BINS - 1)] + [ge[BINS - 1]]

        def pick(rows):
            out = [ZERO] * len(rows[0])
            for v in range(BINS):
                out = [o ^ C.AND(sel[v], x) for o, x in zip(out, rows[v])]
            return out

        digit = [ZERO] * DIGIT
        for v in range(BINS):
            digit = [d ^ sel[v] if v >> i & 1 else d for i, d in enumerate(digit)]
        left = sub(C, r, pick(T[1:]), RB)[0]
        if not last:
            return [_pad(digit + list(prefix), wide(DIGIT * (S.K + 1))), _pad(left, wide(RB))]
        out, before = [digit + list(prefix)], [ZERO] * 32
        for b in range(S.M):
            d, borrow = sub(C, left, before, 32)
            out.append(_pad([C.AND(NOT(borrow), x) for x in d[:RB]], wide(RB)))
            if b < S.M - 1:
                before = add(C, before, pick(cnt[b]), width=32)
        return out
    return build


def _pick_sig(S):
    RB, width = wide(_rb(S.TOPK)), wide(DIGIT * (S.K + 1))
    params = (*((f"counts_{b}", Array(BINS, I32)) for b in range(S.M)),
              *((("prefix", Array(wide(DIGIT * S.K), BIT)), ("r", Array(RB, BIT))) if S.K else ()))
    if S.K < ROUNDS - 1:
        return params, Tuple(Array(width, BIT), Array(RB, BIT))
    return params, Tuple(I32, *[Array(RB, BIT)] * S.M)


DsaTopKDigitPick = EW.specialized("DsaTopKDigitPick", 1, ["M", "K", "TOPK"], _pick, _pick_sig, doc=_pick.__doc__)


def _count_add(S):
    """`DsaTopKCountAdd_v1`: the sum of two counts, u32."""
    return lambda C, a, b: [add(C, list(a), list(b), width=32)]


def _digit_ge(S):
    """`DsaTopKDigitGe_v1{TOPK,READ}`: t >= r, t (u32) the level keys at a digit or above; r is read (RB bits) when READ,
    else TOPK."""
    return lambda C, t, *r: [[NOT(sub(C, list(t), list(r[0]) if S.READ else const_bits(S.TOPK, _rb(S.TOPK)), 32)[1])]]


def _digit_edge(S):
    """`DsaTopKDigitEdge_v1`: a AND NOT b, the pick's digit at v: T(v) >= r and T(v + 1) < r."""
    return lambda C, a, b: [[C.AND(a[0], NOT(b[0]))]]


def _digit_take(S):
    """`DsaTopKDigitTake_v1{W}`: a row of W bits where its digit is the one picked (s), else 0."""
    return lambda C, s, x: [[C.AND(s[0], b) for b in x]]


def _digit_bits(S):
    """`DsaTopKDigitBits_v1`: the digit of its one-hot bits, bit v set at digit v (no AND)."""
    def build(C, sel):
        digit = [ZERO] * DIGIT
        for v in range(BINS):
            digit = [d ^ sel[v] if v >> i & 1 else d for i, d in enumerate(digit)]
        return [digit]
    return build


def _digit_left(S):
    """`DsaTopKDigitLeft_v1{TOPK,READ}`: r less the level keys above the digit (``above``, their low RB bits), mod 2^RB; r as
    `DsaTopKDigitGe_v1` takes it."""
    RB = _rb(S.TOPK)
    return lambda C, above, *r: [sub(C, list(r[0]) if S.READ else const_bits(S.TOPK, RB), list(above), RB)[0]]


def _digit_share(S):
    """`DsaTopKDigitShare_v1{RB}`: the keys equal to the threshold's key that a block supplies: ``left`` less those of the
    blocks before it (``before``, u32), at least 0."""
    def build(C, left, before):
        d, borrow = sub(C, list(left), list(before), 32)
        return [[C.AND(NOT(borrow), x) for x in d[:S.RB]]]
    return build


def _r_sig(S):
    return ((("r", Array(_rb(S.TOPK), BIT)),) if S.READ else ())


DsaTopKCountAdd = BM._piece("DsaTopKCountAdd", [], _count_add, lambda S: ((("a", I32), ("b", I32)), I32))
DsaTopKDigitGe = BM._piece("DsaTopKDigitGe", ["TOPK", "READ"], _digit_ge, lambda S: ((("t", I32), *_r_sig(S)), BIT))
DsaTopKDigitEdge = BM._piece("DsaTopKDigitEdge", [], _digit_edge, lambda S: ((("a", BIT), ("b", BIT)), BIT))
DsaTopKDigitTake = BM._piece("DsaTopKDigitTake", ["W"], _digit_take,
                             lambda S: ((("s", BIT), ("x", Array(S.W, BIT))), Array(S.W, BIT)))
DsaTopKDigitBits = BM._piece("DsaTopKDigitBits", [], _digit_bits, lambda S: ((("sel", Array(BINS, BIT)),), Array(DIGIT, BIT)))
DsaTopKDigitLeft = BM._piece("DsaTopKDigitLeft", ["TOPK", "READ"], _digit_left,
                             lambda S: ((("above", Array(_rb(S.TOPK), BIT)), *_r_sig(S)), Array(_rb(S.TOPK), BIT)))
DsaTopKDigitShare = BM._piece("DsaTopKDigitShare", ["RB"], _digit_share,
                              lambda S: ((("left", Array(S.RB, BIT)), ("before", I32)), Array(S.RB, BIT)))


def _rows(refs, n: int, width: int) -> Coll:
    return Coll(Array(n, Array(width, BIT)), refs)


def _pick_calls(B, S, *ports):
    """`DsaTopKDigitPick_v2{M,K,TOPK}`: `DsaTopKDigitPick_v1`'s pick written as Calls to its steps.  The blocks' counts of each
    digit are summed (`DsaTopKCountAdd_v1`, a batch a block), then T(v) from digit 255 down (one a digit); a batch of
    `DsaTopKDigitGe_v1` compares each T(v) with r and one of `DsaTopKDigitEdge_v1` leaves the digit's one-hot bits;
    `DsaTopKDigitTake_v1` under those bits and XORs pick T(digit + 1) and, at the last round, each block's count of the digit;
    `DsaTopKDigitLeft_v1` and `DsaTopKDigitShare_v1` are r past the digit and a block's share of it.  Digit 0's sum and T(0),
    and T(digit + 1) above its low RB bits, are never read, and are not computed."""
    RB, M, read = _rb(S.TOPK), S.M, int(S.K > 0)
    if not 0 <= S.K < ROUNDS or M < 1:
        raise ValueError(f"DsaTopKDigitPick: 0 <= K < {ROUNDS} and M >= 1 (M={M}, K={S.K})")
    counts = ports[:M]
    prefix, rs = ([low(ports[M], DIGIT * S.K).refs], (low(ports[M + 1], RB),)) if read else ([], ())
    zero, n = B.call(BL.constant(0)), BINS - 1
    h = _rows(counts[0].refs.slice(32, 32 * BINS), n, 32)
    for c in counts[1:]:
        h = B.batch(DsaTopKCountAdd, h, _rows(c.refs.slice(32, 32 * BINS), n, 32), axes=(0, 0))
    above = [Coll(I32, h.refs.slice(32 * (n - 1), 32 * n))]
    for j in reversed(range(n - 1)):
        above.insert(0, B.call(DsaTopKCountAdd, above[0], Coll(I32, h.refs.slice(32 * j, 32 * (j + 1)))))
    ge = B.batch(bind(DsaTopKDigitGe, TOPK=S.TOPK, READ=read), _rows(concat([t.refs for t in above]), n, 32), *rs,
                 axes=(0, *(None for _ in rs)))

    def bits(lo: int, hi: int) -> Coll:
        return Coll(Array(hi - lo, BIT), ge.refs.slice(lo, hi))

    sel = Coll(Array(BINS, BIT), concat([B.call(BL.NOT, Coll(BIT, ge.refs.slice(0, 1))).refs,
                                         B.batch(DsaTopKDigitEdge, bits(0, n - 1), bits(1, n), axes=(0, 0)).refs,
                                         ge.refs.slice(n - 1, n)]))

    def pick(rows: Coll, k: int, w: int) -> Coll:
        refs = B.batch(bind(DsaTopKDigitTake, W=w), Coll(Array(k, BIT), sel.refs.slice(0, k)), rows, axes=(0, 0)).refs
        while k > 1:
            half = k // 2
            pair = Coll(Array(2, Array(half * w, BIT)), refs.slice(0, 2 * half * w))
            refs = concat([B.batch(BL.XOR, pair, axes=(1,)).refs, refs.slice(2 * half * w, k * w)])
            k -= half
        return Coll(Array(w, BIT), refs)

    def padded(refs, bits_: int) -> list:
        return [refs, *([BM._rep(zero, wide(bits_) - bits_)] if wide(bits_) > bits_ else [])]

    digit = B.call(DsaTopKDigitBits, sel)
    left = B.call(bind(DsaTopKDigitLeft, TOPK=S.TOPK, READ=read), pick(_rows(concat([t.refs.slice(0, RB) for t in above]), n, RB), n, RB),
                  *rs)
    out = [*padded(concat([digit.refs, *prefix]), DIGIT * (S.K + 1)), *padded(left.refs, RB)]
    if S.K < ROUNDS - 1:
        return Coll(_pick_sig(S)[1], concat(out))
    before = None
    for b in range(1, M):
        mine = pick(counts[b - 1], BINS, 32)
        before = mine if before is None else B.call(DsaTopKCountAdd, before, mine)
        out += padded(B.call(bind(DsaTopKDigitShare, RB=RB), left, before).refs, RB)
    return Coll(_pick_sig(S)[1], concat(out))


DsaTopKDigitPickV2 = CompositeDefinition("DsaTopKDigitPick", 2, ("M", "K", "TOPK"), _pick_sig, _pick_calls, doc=_pick_calls.__doc__,
                                         conformance="exact-model-tested: held to DsaTopKDigitPick_v1 and to the cut's twin by "
                                                     "adapters/vllm/tests/program/test_dsa_long_row_calls.py")


def _compare(S):
    """`DsaTopKKeyCompare_v1`: (valid AND key == theta, valid AND key > theta), key = `DsaTopKKey_v2` of s, both off theta + ~key
    with no carry in (a carry in would put a gate of theta alone in every key's instance): its low word is all ones exactly when
    key == theta, and it carries exactly when theta > key."""
    def build(C, s, valid, theta):
        over = add(C, list(theta), [NOT(x) for x in key(list(s))], width=32, carry_out=True)
        eq = and_reduce(C, over[:32])
        return [[C.AND(valid[0], eq), C.AND(valid[0], NOT(over[32] ^ eq))]]
    return build


DsaTopKKeyCompare = EW.specialized("DsaTopKKeyCompare", 1, [], _compare,
                                   lambda S: ((("s", F32), ("valid", BIT), ("theta", I32)), Array(2, BIT)),
                                   doc=_compare.__doc__)


def _mask_sig(S):
    return ((("scores", Array(S.N, F32)), ("n", I32), ("key", I32), ("r", Array(wide(_rb(S.TOPK)), BIT))), words(S.N))


def _mask(B, S, scores, n, theta, r, ties=BM.DsaRadixTies, valid=_valid):
    """`DsaTopKMaskBlock_v1{N,AT,TOPK}`, keys AT..AT+N-1 of a row: those below n above the threshold's key, and the first r (its
    low RB bits) of those equal to it (`DsaTopKKeyCompare_v1` per key, `DsaRadixTies_v1{N,TOPK}`), in u16 words."""
    valid = valid(B, S.N, S.AT, n)
    cmp = B.batch(DsaTopKKeyCompare, scores, valid, theta, axes=(0, 0, None))
    eq, gt = (Coll(Array(S.N, BIT), strided_view(cmp, i, 1, 2, 1, S.N)) for i in (0, 1))
    return to_words(B, B.call(bind(ties, N=S.N, TOPK=S.TOPK), eq, gt, low(r, _rb(S.TOPK))), S.N)


DsaTopKMaskBlock = CompositeDefinition("DsaTopKMaskBlock", 1, ("N", "AT", "TOPK"), _mask_sig, _mask, doc=_mask.__doc__,
                                       conformance="exact-model-tested")


def _mask_calls(B, S, scores, n, theta, r):
    """`DsaTopKMaskBlock_v2{N,AT,TOPK}`: `DsaTopKMaskBlock_v1` with its ties by `DsaRadixTies_v2{N,TOPK}`, the prefix count
    written as Calls, and its keys below n by `DsaRowValid_v2{N}` (`_valid_calls`), a batch a level of the trie."""
    return _mask(B, S, scores, n, theta, r, ties=BM.DsaRadixTiesV2, valid=_valid_calls)


DsaTopKMaskBlockV2 = CompositeDefinition("DsaTopKMaskBlock", 2, ("N", "AT", "TOPK"), _mask_sig, _mask_calls, doc=_mask_calls.__doc__,
                                         conformance="exact-model-tested")


def _whole(B, S, scores, n):
    """`DsaTopKMaskWhole_v1{N,TOPK}`: `DsaTopKMaskCounted_v2{N,TOPK}`'s mask in u16 words, the mask a unit up to `WHOLE_MAX`
    keys.  `_v2` itself keeps `_v1`'s Array<N, BIT>, as `_v1`'s Boolean version must."""
    return to_words(B, B.call(bind(BM.DsaTopKMaskCountedV2, N=S.N, TOPK=S.TOPK), scores, n), S.N)


DsaTopKMaskWhole = CompositeDefinition("DsaTopKMaskWhole", 1, ("N", "TOPK"),
                                       lambda S: ((("scores", Array(S.N, F32)), ("n", I32)), words(S.N)), _whole,
                                       doc=_whole.__doc__, conformance="exact-model-tested")


def _whole_calls(B, S, scores, n):
    """`DsaTopKMaskWhole_v2{N,TOPK}`: `DsaTopKMaskWhole_v1`'s mask, `DsaTopKMaskCounted_v2`'s radix select written out in the
    unit with each part as Calls: its keys below n by `DsaRowValid_v2{N}`, its 32 passes `DsaRadixPass_v2{N,TOPK}` and its
    ties `DsaRadixTies_v2{N,TOPK}`."""
    return to_words(B, BM._select(B, S, scores, n, ties=BM.DsaRadixTiesV2, step=BM.DsaRadixPassV2, valid=BM.DsaRowValidV2), S.N)


DsaTopKMaskWholeV2 = CompositeDefinition("DsaTopKMaskWhole", 2, ("N", "TOPK"),
                                         lambda S: ((("scores", Array(S.N, F32)), ("n", I32)), words(S.N)), _whole_calls,
                                         doc=_whole_calls.__doc__, conformance="exact-model-tested")


@dataclass(frozen=True)
class Unit:
    """One unit of the cut: ``key`` names it, ``defn`` its Definition, ``keys`` the scores it reads ((0, 0): none; a unit that
    reads scores reads n too), ``wires`` each wired parameter's (parameter, writer key, writer's output index), one writer a
    parameter, which reads that output whole; its other parameters are the Call's operands."""
    key: tuple
    defn: Any
    keys: tuple[int, int]
    wires: tuple[tuple[str, tuple, int], ...] = field(default=())


def blocks(N: int, TOPK: int, block: int = BLOCK) -> list[tuple[int, int]]:
    """The row's blocks, (first key, end): `block` keys each, a last one shorter than TOPK joined to the one before."""
    starts = list(range(0, N, block))
    if len(starts) > 1 and N - starts[-1] < TOPK:
        starts.pop()
    return [(a, starts[i + 1] if i + 1 < len(starts) else N) for i, a in enumerate(starts)]


@cache
def plan(N: int, TOPK: int, block: int = BLOCK) -> tuple[Unit, ...]:
    """The units of a row of N keys selecting TOPK, in an order that writes each before its readers."""
    check_statics(N, TOPK)
    if block < TOPK:
        raise ValueError(f"DsaTopKMaskCounted's cut: a block of {block} keys holds TOPK = {TOPK}")
    bl = blocks(N, TOPK, block)
    out: list[Unit] = []
    for k in range(ROUNDS):
        prev = ("pick", k - 1)
        for b, (lo, hi) in enumerate(bl):
            out.append(Unit(("count", k, b), bind(DsaTopKDigitCount, N=hi - lo, AT=lo, K=k), (lo, hi),
                            (("prefix", prev, 0),) if k else ()))
        w = tuple((f"counts_{b}", ("count", k, b), 0) for b in range(len(bl)))
        out.append(Unit(("pick", k), bind(DsaTopKDigitPick, M=len(bl), K=k, TOPK=TOPK), (0, 0),
                        w + ((("prefix", prev, 0), ("r", prev, 1)) if k else ())))
    last = ("pick", ROUNDS - 1)
    for b, (lo, hi) in enumerate(bl):
        out.append(Unit(("mask", b), bind(DsaTopKMaskBlock, N=hi - lo, AT=lo, TOPK=TOPK), (lo, hi),
                        (("key", last, 0), ("r", last, 1 + b))))
    return tuple(out)


def cut(t) -> list[tuple[Any, int]] | None:
    """The units a binding of `DsaTopKMaskCounted_v1{N,TOPK}` (or `_v2`) is cut into, each Definition bound and how many of it,
    or None up to `WHOLE_MAX` keys, where the Call is one unit."""
    N, TOPK = int(t.bindings["N"]), int(t.bindings["TOPK"])
    if N <= WHOLE_MAX:
        return None
    return list(Counter(u.defn for u in plan(N, TOPK)).items())


def keys_of(scores) -> np.ndarray:
    """`dsa_topk.topk_key` of each f32 word, as uint64."""
    s = np.asarray(scores, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    return np.where(s >> np.uint64(31), ~s & np.uint64(0xFFFFFFFF), s | np.uint64(0x80000000))


def values(scores, n: int, TOPK: int, block: int = BLOCK) -> dict[tuple, tuple]:
    """Each unit of `plan`(len(scores), TOPK, block)'s outputs on a row (f32 words) and its n, by unit key: a count unit's 256
    counts, a pick's (prefix, r) and the last one's (key, r_0, ..., r_{M-1}), a mask unit's bits (its block's, the words'
    padding apart)."""
    k = keys_of(scores)
    N, bl = len(k), blocks(len(k), TOPK, block)
    valid = np.arange(N) < n
    out: dict[tuple, tuple] = {}
    prefix, r = 0, TOPK
    for rnd in range(ROUNDS):
        lo_bit = 32 - DIGIT * (rnd + 1)
        level = valid & ((k >> np.uint64(lo_bit + DIGIT)) == np.uint64(prefix))
        digit = ((k >> np.uint64(lo_bit)) & np.uint64(BINS - 1)).astype(np.int64)
        counts = [np.bincount(digit[a:e][level[a:e]], minlength=BINS) for a, e in bl]
        for b, c in enumerate(counts):
            out[("count", rnd, b)] = (tuple(int(x) for x in c),)
        T = np.concatenate([np.cumsum(sum(counts)[::-1])[::-1], [0]])
        d = max((v for v in range(1, BINS) if T[v] >= r), default=0)
        r -= int(T[d + 1])
        prefix = prefix << DIGIT | d
        if rnd < ROUNDS - 1:
            out[("pick", rnd)] = (prefix, r)
    rb, before = [], 0
    for c in counts:
        rb.append(max(r - before, 0))
        before += int(c[d])
    out[("pick", ROUNDS - 1)] = (prefix, *rb)
    for b, (a, e) in enumerate(bl):
        eq = valid[a:e] & (k[a:e] == np.uint64(prefix))
        gt = valid[a:e] & (k[a:e] > np.uint64(prefix))
        out[("mask", b)] = (tuple(int(x) for x in gt | (eq & (np.cumsum(eq) - eq < rb[b]))),)
    return out


CUT_DEFINITIONS = [DsaRowValidFrom, DsaRowValidOffset, DsaTopKDigitOneHot, DsaTopKPopCount, DsaTopKDigitCount, DsaTopKDigitPick, DsaTopKCountAdd,
                   DsaTopKDigitGe, DsaTopKDigitEdge, DsaTopKDigitTake, DsaTopKDigitBits, DsaTopKDigitLeft, DsaTopKDigitShare,
                   DsaTopKDigitPickV2, DsaTopKKeyCompare, DsaTopKMaskBlock, DsaTopKMaskBlockV2, DsaTopKMaskWhole, DsaTopKMaskWholeV2]

__all__ = ["BLOCK", "WHOLE_MAX", "Unit", "blocks", "plan", "cut", "values", "keys_of", "wide", "words", "low", "to_words",
           "CUT_DEFINITIONS", "DsaRowValidFrom", "DsaRowValidOffset", "DsaTopKDigitOneHot", "DsaTopKPopCount", "DsaTopKDigitCount", "DsaTopKDigitPick",
           "DsaTopKDigitPickV2", "DsaTopKKeyCompare", "DsaTopKMaskBlock", "DsaTopKMaskBlockV2", "DsaTopKMaskWhole", "DsaTopKMaskWholeV2"]
