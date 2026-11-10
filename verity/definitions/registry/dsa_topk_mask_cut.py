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
* ``DsaTopKDigitPick_v1{M,K,TOPK}``, over the M blocks' counts: with T(v) the level keys at digit v or above, the threshold's
  digit is the largest v with T(v) >= r, else 0 (fewer than r level keys: the row has fewer than TOPK keys below n), and r drops
  by T(digit + 1), the level keys the set takes above it.  Those are the digit and the r the radix select's 8 passes over the
  digit's bits leave.  Round 3's pick returns the threshold's key and, per block, how many of its keys equal to that key the set
  takes: r less those of the blocks before it, at least 0.

Then ``DsaTopKMaskBlock_v1{N,AT,TOPK}``, per block: its keys j < n above the threshold's key and, of those equal to it, the first
r_b by index (`DsaRadixTies_v1`), the block's part of `DsaRadixTies_v1`'s mask over the row, since its equal keys come after
those of the blocks before it.  Every unit reads its own block of the scores and n, the Call's operands; each other input is an
output row of a unit before it (`Unit.wires`): the prefix and r of the pick before, a round's counts, the threshold's key and
r_b.  The Call's mask is the mask units' outputs in block order.  Units a row: 5M + 4, M = ceil(N / `BLOCK`) blocks (44 at
131,072 keys).
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
from verity.circuits.refs import Coll, strided_view
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import boolean_dsa_topk_mask as BM
from verity.definitions.registry.boolean_dsa_topk import decode, key
from verity.definitions.registry.dsa_topk import check_statics

BIT = BL.BIT
F32 = I32 = Array(32, BIT)

#: keys a block: a count unit is then 2^26 bits of Flock's 2^27 (9.6M ANDs, its scores' hashes 174 sha512x3 slots), a mask
#: unit 2^26 and a pick at most 2^24
BLOCK = 16384
#: the largest N the Call is checked and proved whole at: `DsaTopKMaskCounted_v2` is 133,431,296 of the block's 134,217,728
#: bits at 40,960 keys and 139,198,464 at 43,008; past it, `cut`
WHOLE_MAX = 40960
DIGIT, ROUNDS = 8, 4
BINS = 1 << DIGIT


def _rb(TOPK: int) -> int:
    return TOPK.bit_length()


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
    return ((("scores", Array(S.N, F32)), ("n", I32), *((("prefix", Array(DIGIT * S.K, BIT)),) if S.K else ())),
            Array(BINS, I32))


def _count(B, S, scores, n, *prefix):
    """`DsaTopKDigitCount_v1{N,AT,K}`, keys AT..AT+N-1 of a row: for each v < 256, how many of them below n are level with the
    prefix and have digit K v (`DsaTopKDigitOneHot_v1{K}` per key, `DsaTopKPopCount_v1{N}` per v)."""
    valid = _valid(B, S.N, S.AT, n)
    hot = B.batch(bind(DsaTopKDigitOneHot, K=S.K), scores, valid, *prefix, axes=(0, 0, *(None for _ in prefix)))
    cols = Coll(Array(BINS, Array(S.N, BIT)), strided_view(hot, 0, S.N, 1, BINS, BINS * S.N))
    return B.batch(bind(DsaTopKPopCount, N=S.N), cols, axes=(0,))


DsaTopKDigitCount = CompositeDefinition("DsaTopKDigitCount", 1, ("N", "AT", "K"), _count_sig, _count, doc=_count.__doc__,
                                        conformance="exact-model-tested")


def _pick(S):
    """`DsaTopKDigitPick_v1{M,K,TOPK}`: round K's digit from the M blocks' counts, with T(v) the level keys at digit v or above:
    the largest v with T(v) >= r, else 0; returns (prefix with the digit below it, r - T(digit + 1)), and at the last round (the
    threshold's key, r_b for each block b: that r less the keys equal to the key in blocks 0..b-1, at least 0)."""
    RB, last = _rb(S.TOPK), S.K == ROUNDS - 1
    if not 0 <= S.K < ROUNDS or S.M < 1:
        raise ValueError(f"DsaTopKDigitPick: 0 <= K < {ROUNDS} and M >= 1 (M={S.M}, K={S.K})")

    def build(C, *els):
        cnt = [[list(els[b * BINS + v]) for v in range(BINS)] for b in range(S.M)]
        prefix, r = (els[S.M * BINS], list(els[S.M * BINS + 1])) if S.K else ((), const_bits(S.TOPK, RB))
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
            return [digit + list(prefix), left]
        out, before = [digit + list(prefix)], [ZERO] * 32
        for b in range(S.M):
            d, borrow = sub(C, left, before, 32)
            out.append([C.AND(NOT(borrow), x) for x in d[:RB]])
            if b < S.M - 1:
                before = add(C, before, pick(cnt[b]), width=32)
        return out
    return build


def _pick_sig(S):
    RB, width = _rb(S.TOPK), DIGIT * (S.K + 1)
    params = (("counts", Array(S.M, Array(BINS, I32))),
              *((("prefix", Array(DIGIT * S.K, BIT)), ("r", Array(RB, BIT))) if S.K else ()))
    if S.K < ROUNDS - 1:
        return params, Tuple(Array(width, BIT), Array(RB, BIT))
    return params, Tuple(I32, *[Array(RB, BIT)] * S.M)


DsaTopKDigitPick = EW.specialized("DsaTopKDigitPick", 1, ["M", "K", "TOPK"], _pick, _pick_sig, doc=_pick.__doc__)


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
    return ((("scores", Array(S.N, F32)), ("n", I32), ("key", I32), ("r", Array(_rb(S.TOPK), BIT))), Array(S.N, BIT))


def _mask(B, S, scores, n, theta, r):
    """`DsaTopKMaskBlock_v1{N,AT,TOPK}`, keys AT..AT+N-1 of a row: those below n above the threshold's key, and the first r of
    those equal to it (`DsaTopKKeyCompare_v1` per key, `DsaRadixTies_v1{N,TOPK}`)."""
    valid = _valid(B, S.N, S.AT, n)
    cmp = B.batch(DsaTopKKeyCompare, scores, valid, theta, axes=(0, 0, None))
    eq, gt = (Coll(Array(S.N, BIT), strided_view(cmp, i, 1, 2, 1, S.N)) for i in (0, 1))
    return B.call(bind(BM.DsaRadixTies, N=S.N, TOPK=S.TOPK), eq, gt, r)


DsaTopKMaskBlock = CompositeDefinition("DsaTopKMaskBlock", 1, ("N", "AT", "TOPK"), _mask_sig, _mask, doc=_mask.__doc__,
                                       conformance="exact-model-tested")


@dataclass(frozen=True)
class Unit:
    """One unit of the cut: ``key`` names it, ``defn`` its Definition, ``keys`` the scores it reads ((0, 0): none; a unit that
    reads scores reads n too), ``wires`` each wired parameter's (parameter, writer key, writer's output index), a parameter
    named more than once reading those writers' outputs in turn; its other parameters are the Call's operands."""
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
        w = tuple(("counts", ("count", k, b), 0) for b in range(len(bl)))
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
    counts, a pick's (prefix, r) and the last one's (key, r_0, ..., r_{M-1}), a mask unit's bits."""
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


CUT_DEFINITIONS = [DsaRowValidFrom, DsaTopKDigitOneHot, DsaTopKPopCount, DsaTopKDigitCount, DsaTopKDigitPick, DsaTopKKeyCompare,
                   DsaTopKMaskBlock]

__all__ = ["BLOCK", "WHOLE_MAX", "Unit", "blocks", "plan", "cut", "values", "keys_of", "CUT_DEFINITIONS", "DsaRowValidFrom",
           "DsaTopKDigitOneHot", "DsaTopKPopCount", "DsaTopKDigitCount", "DsaTopKDigitPick", "DsaTopKKeyCompare",
           "DsaTopKMaskBlock"]
