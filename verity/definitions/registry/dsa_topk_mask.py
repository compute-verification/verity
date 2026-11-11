"""[GLM-5.3] The DSA indexer's top-k of a row past TOPK keys in the mask form: the selection computed in the circuit from the row's
logits, with no `run` value and no advice.

* `DsaTopKMask` is the selection: bit j is 1 exactly when key j is in `dsa_topk.topk_set`(scores, TOPK), the tie rule's top-TOPK
  set (key descending, then index ascending).
* `DsaButterflySettings` computes from that mask the settings of an order-preserving concentrator, and `DsaButterflyRows` reads a
  row's history through it: output s is the row of the s-th member in index order.  So the rows arrive in ascending index order,
  the order a canonical served row lists them in (`DsaTopKRow`), and no circuit needs the kernel's atomics order.
* `DsaTopKRow` is the long-row check: the three above over the indices 0..N-1, whose output is `sorted(topk_set(scores, TOPK))`.

These are the accelerators (`_v1`); `boolean_dsa_topk_mask` holds the definitions on bits, which circuit-check and
adapters/vllm/tests/program/test_dsa_topk_mask.py hold equal to them.

The concentrator is the inverse butterfly over P = 2^L positions, L = ceil(log2 N), position j < N holding row j and the rest
empty.  Stage t = 0..L-1 has a switch over each pair (p, p + 2^t), p's bit t zero: crossed when its setting bit is 1.  Member j,
of rank r_j = #{i < j : m_i}, leaves stage t at a position whose bits 0..t are r_j's, so after stage L-1 it is at r_j.  Two members
j < j' never contend for one switch's output: they meet at stage t only if they lie in one block of 2^(t+1) indices with
r_j = r_j' mod 2^(t+1), but 0 < r_j' - r_j <= j' - j < 2^(t+1).  A switch's setting comes from the two positions' tags
(valid, rank): s = d_lo[t] if the lower position holds a member, else (the upper holds one and d_hi[t] = 0); past the ranks' bits
(t >= kb = ceil(log2 K)) s is "the upper holds one".  The tags travel through the same switches.

Pruned (`butterfly_plan`): an output at stage t whose bits 0..t, read as a number, are K or more can reach no output below K, and a
position whose block of 2^t positions starts at N or past holds no row; a switch is kept when its lower output can reach an output
and its lower input can hold a row, a pick (its upper output dropped) when only the lower output can.  At stage t that is the
switches (p, p + 2^t) with p mod 2^t < min(2^t, K) and p < N rounded up to a multiple of 2^(t+1): `butterfly_switches`(N, K) of
them, 849,920 at N = 131,072 and K = 2,048 (Waksman's is 2,097,153 - 131,072 + 1 = 1,966,082 at that N).  Their settings are laid
out stage by stage, a stage's swaps before its picks, each set in the order (p mod 2^t, p >> (t+1)), so that every stage reads the
one before it through a few strided views (`butterfly`).

Registered functions:
  DsaTopKMask{N,TOPK}           (scores: N f32) -> N bits: bit j is j in topk_set(scores, TOPK)
  DsaTopKThresholdKey{N}x{TOPK}, DsaTopKThresholdIndex{N}x{TOPK}
                                (scores: N f32) -> i32: the TOPK-th largest ext_j = topk_key(s_j) * 2^32 + (2^32 - 1 - j) as
                                its key and its index j, one 32-bit word each; DsaTopKMask's body, with DsaTopKAtLeast per key
  DsaTopKAtLeast                (s: f32, j: i32, key: i32, at: i32) -> bit: ext(s, j) >= ext_of(key, at), the threshold's ext
  DsaTopKMaskCounted{N,TOPK}    (scores: N f32, n: i32) -> N bits: bit j is j < n and j in topk_set(scores[:n], TOPK), a row of
                                at most N keys; DsaTopKCountedThresholdKey{N}x{TOPK} and DsaTopKCountedThresholdIndex{N}x{TOPK}
                                (scores, n) -> i32, the threshold's two words with ext 0 from n on, and DsaTopKAtLeastBelow
                                (s, j, n, key, at) -> bit, j < n and ext(s, j) >= ext_of(key, at)
  DsaButterflySettings{N,K}     (mask: N bits) -> butterfly_switches(N, K) bits: the concentrator's settings for the mask's
                                members, ranks taken mod 2^kb (`butterfly_settings`; `butterfly_apply` is the network)
  DsaButterflyRows{N,K,W,E}     (sw: butterfly_switches(N, K) bits, rows: N x W x E bits) -> K x W x E bits: the concentrator at
                                sw, empty positions zero rows; at DsaButterflySettings of a mask of K members, slot s is the row of
                                the s-th member
  DsaButterflyRowsPad{T,N,K,W,E} (sw, rows: T x W x E bits, pad: (N - T) x W x E bits) -> K x W x E bits: DsaButterflyRows{N,K,W,E}
                                over rows then pad, a row of T keys at its size class N with its history an argument of its own
  DsaTopKRow{N,TOPK}            (scores: N f32) -> TOPK i32: sorted(topk_set(scores, TOPK)), DsaTopKSlot per slot
  DsaTopKSlot{N}x{TOPK}         (scores: N f32, s: i32) -> i32: sorted(topk_set(scores, TOPK))[s]
  DsaTopKKey                    (s: f32) -> i32: topk_key(s)
  DsaMaskCount, DsaMaskTag      (c: i32, m: bit) -> c + m, and the tag of a key with c members before it: 2^15 + c mod 2^15
                                when m, else 0
  DsaMaskTagStep                (c: i32, m: bit) -> (i32, 16 bits): the two, the scan step of DsaButterflySettings' tags
  DsaButterflySet               (t: i32, kb: i32, lo, hi: 16-bit tags) -> bit: the setting rule above
  DsaButterflyRoute{T,KB,TWO}   (lo, hi: 16-bit tags) -> (bit, 16 bits[, 16 bits]): a stage-T switch's setting and its tag
                                outputs (the upper one when TWO)
"""
from __future__ import annotations

import functools
import heapq

from verity.circuits.defs import PrimitiveDefinition, bind, composite, primitive
from verity.definitions.ops.prims import const
from verity.circuits.refs import Coll, broadcast, concat, strided_view, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.definitions.registry.dsa_topk import M32, DsaWaksmanPick, DsaWaksmanSwap, Xor3x16, check_statics, topk_key, topk_set
from verity.definitions.registry.prims import BF16, BIT, F32, I32, SelectBf16

_CONF = ("model-untested: dsa_topk's tie rule (`topk_set`), which dsa_topk's module docstring reads from the served kernel and "
         "holds against its orders; this form computes that set and lists it in index order, so it agrees with a served row as a "
         "set, and with its order once the row is sorted")
#: the tags' rank field: 15 bits, so K <= 2^15
RANK_BITS = 15
_THRESHOLDS: dict[tuple[int, int], tuple[PrimitiveDefinition, PrimitiveDefinition]] = {}
_COUNTED: dict[tuple[int, int], tuple[PrimitiveDefinition, PrimitiveDefinition]] = {}
_SLOTS: dict[tuple[int, int], PrimitiveDefinition] = {}


def ext(s: int, j: int) -> int:
    """The tie rule's order as one number: a larger ext precedes (key descending, then index ascending)."""
    return topk_key(s) << 32 | (M32 - (j & M32))


def rank_bits(K: int) -> int:
    """kb, the rank bits a member's tag needs: ceil(log2 K), at least 1."""
    return max(1, (K - 1).bit_length())


def check_net(N: int, K: int) -> None:
    if not 1 <= K <= N or N < 2 or K > 1 << RANK_BITS:
        raise ValueError(f"a concentrator of N >= 2 rows with 1 <= K <= min(N, 2^{RANK_BITS}) outputs (N={N}, K={K})")


def _refuse_sel(b: dict) -> str | None:
    if b.get("N") is not None and b.get("TOPK") is not None:
        try:
            check_statics(b["N"], b["TOPK"])
        except ValueError as e:
            return str(e)
    return None


def _refuse_net(b: dict) -> str | None:
    N, K = b.get("N"), b.get("K", b.get("TOPK"))
    if N is not None and K is not None:
        try:
            check_net(N, K)
        except ValueError as e:
            return str(e)
    if b.get("E") is not None and b["E"] not in (16, 32):
        return f"rows of 16- or 32-bit words (E={b['E']})"
    if b.get("W") is not None and b["W"] < 1:
        return f"rows of W >= 1 words (W={b['W']})"
    return None


@functools.lru_cache(maxsize=None)
def butterfly_plan(N: int, K: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Per stage t, (its swaps, its picks), each the switches' lower positions p (the upper is p + 2^t) in settings order: the
    module docstring's pruned inverse butterfly of N rows with K live outputs."""
    check_net(N, K)
    plan = []
    for t in range(max(1, (N - 1).bit_length())):
        blocks = -(-N >> (t + 1))
        swaps: list[int] = []
        picks: list[int] = []
        for v in range(min(1 << t, K)):
            (swaps if v + (1 << t) < K else picks).extend(h << (t + 1) | v for h in range(blocks))
        plan.append((tuple(swaps), tuple(picks)))
    return tuple(plan)


@functools.lru_cache(maxsize=None)
def butterfly_switches(N: int, K: int) -> int:
    return sum(len(s) + len(p) for s, p in butterfly_plan(N, K))


def butterfly_apply(sw, items: list, K: int, dead=0) -> list:
    """`DsaButterflyRows` at settings ``sw`` over ``items`` (N rows of any values): outputs 0..K-1, an empty position ``dead``."""
    N = len(items)
    pos = list(items) + [dead] * ((1 << max(1, (N - 1).bit_length())) - N)
    i = 0
    for t, (swaps, picks) in enumerate(butterfly_plan(N, K)):
        for ps, two in ((swaps, True), (picks, False)):
            for p in ps:
                q = p + (1 << t)
                if sw[i]:
                    pos[p], pos[q] = pos[q], pos[p]
                if not two:
                    pos[q] = dead
                i += 1
    return pos[:K]


def butterfly_settings(mask, K: int) -> list[int]:
    """The settings `DsaButterflySettings` computes from ``mask`` (N bits), switch by switch in `butterfly_plan` order."""
    N, kb = len(mask), rank_bits(K)
    P = 1 << max(1, (N - 1).bit_length())
    tags, c = [], 0
    for m in mask:
        tags.append((1, c % (1 << kb)) if m & 1 else (0, 0))
        c += m & 1
    tags += [(0, 0)] * (P - N)
    out = []
    for t, (swaps, picks) in enumerate(butterfly_plan(N, K)):
        for ps, two in ((swaps, True), (picks, False)):
            for p in ps:
                (vl, dl), (vh, dh) = tags[p], tags[p + (1 << t)]
                s = (dl >> t & 1 if vl else vh & ~dh >> t & 1) if t < kb else vh
                out.append(s)
                lo, hi = (tags[p + (1 << t)], tags[p]) if s else (tags[p], tags[p + (1 << t)])
                tags[p], tags[p + (1 << t)] = lo, hi if two else (0, 0)
    return out


def _run(X, srcs: list, dead: Coll) -> Coll:
    """The Array<len(srcs), X> of the elements ``srcs`` names, each (a Coll whose refs are one run, the element's first leaf in it)
    or None for ``dead`` (one leaf, repeated): a strided view per run of one Coll at a constant positive step."""
    w, parts, i = X.leaves, [], 0
    while i < len(srcs):
        if srcs[i] is None:
            j = i
            while j < len(srcs) and srcs[j] is None:
                j += 1
            parts.append(broadcast(dead.refs.space, dead.refs.idx, dead.refs.base, (j - i) * w))
            i = j
            continue
        c, a = srcs[i]
        j, step = i + 1, None
        while j < len(srcs) and srcs[j] is not None and srcs[j][0] is c:
            d = srcs[j][1] - srcs[j - 1][1]
            if d <= 0 or (step is not None and d != step):
                break
            step, j = d, j + 1
        parts.append(strided_view(c, a, w, step or w, 1, (j - i) * w))
        i = j
    return Coll(Array(len(srcs), X), concat(parts))


def butterfly(B, N: int, K: int, X, items: Coll, dead: Coll, switch) -> Coll:
    """The concentrator's wiring over ``items`` (Array<N, X>, one run of refs), for `DsaButterflyRows` and `DsaButterflySettings`
    and their Boolean versions: per stage, its swaps' and its picks' batch, ``switch(t, i, n, two, lo, hi)`` over the switches'
    lower and upper inputs (Array<n, X>) whose settings are bits i..i+n, which returns (its output, one run of refs; the leaves of
    one switch's output in it; the lower output's offset in those; the upper's, or None for a pick).  ``dead`` (one leaf) is an
    empty position's X.  Returns outputs 0..K-1."""
    w = X.leaves
    where: list = [(items, j * w) for j in range(N)] + [None] * ((1 << max(1, (N - 1).bit_length())) - N)
    i = 0
    for t, stage in enumerate(butterfly_plan(N, K)):
        nxt = list(where)
        for ps, two in zip(stage, (True, False)):
            if not ps:
                continue
            out, m, lo, hi = switch(t, i, len(ps), two, _run(X, [where[p] for p in ps], dead),
                                    _run(X, [where[p + (1 << t)] for p in ps], dead))
            for k, p in enumerate(ps):
                nxt[p] = (out, k * m + lo)
                nxt[p + (1 << t)] = (out, k * m + hi) if two else None
            i += len(ps)
        where = nxt
    return _run(X, where[:K], dead)


def settings(B, N: int, K: int, X, tags: Coll, dead: Coll, route) -> Coll:
    """`DsaButterflySettings`' body over its tags (Array<N, X>) and ``route(t, two)``, the stage-t switch over two tags whose
    output is (setting bit, lower tag[, upper tag])."""
    sw = []

    def switch(t, i, n, two, lo, hi):
        out = B.batch(route(t, two), lo, hi, axes=(0, 0))
        m = 1 + (2 if two else 1) * X.leaves
        sw.append(strided_view(out, 0, 1, m, 1, n))
        return out, m, 1, 1 + X.leaves if two else None

    butterfly(B, N, K, X, tags, dead, switch)
    return Coll(Array(butterfly_switches(N, K), BIT), concat(sw))


def rows(B, N: int, K: int, R, sw: Coll, items: Coll, dead: Coll, swap, pick) -> Coll:
    """`DsaButterflyRows`' body over its switches: ``swap`` and ``pick`` over two rows of type R."""

    def switch(t, i, n, two, lo, hi):
        out = B.batch(swap if two else pick, sw[i:i + n], lo, hi, axes=(0, 0, 0))
        return out, (2 if two else 1) * R.leaves, 0, R.leaves if two else None

    return butterfly(B, N, K, R, items, dead, switch)


# --- the selection -----------------------------------------------------------------------------------------------------------

@primitive("DsaTopKKey", 1, [("s", F32)], I32, conformance="exact-model-tested")
def DsaTopKKey(s: int) -> int:
    return topk_key(s)


@primitive("DsaTopKAtLeast", 1, [("s", F32), ("j", I32), ("key", I32), ("at", I32)], BIT, conformance="exact-model-tested")
def DsaTopKAtLeast(s: int, j: int, key: int, at: int) -> int:
    return int(ext(s, j) >= ext_of(key, at))


def ext_of(key: int, j: int) -> int:
    """The ext of index ``j`` at key ``key`` (`ext` of a score whose `topk_key` is ``key``)."""
    return (key & M32) << 32 | (M32 - (j & M32))


def _theta(scores: tuple, TOPK: int) -> int:
    return heapq.nlargest(TOPK, (ext(x, j) for j, x in enumerate(scores)))[-1]


def threshold(N: int, TOPK: int) -> tuple[PrimitiveDefinition, PrimitiveDefinition]:
    """`DsaTopKThresholdKey{N}x{TOPK}_v1` and `DsaTopKThresholdIndex{N}x{TOPK}_v1`: the TOPK-th largest ext over a row of N f32
    scores as its key and its index, one 32-bit word each (a proof block's output is one gate of at most 32 bits)."""
    check_statics(N, TOPK)
    p = _THRESHOLDS.get((N, TOPK))
    if p is None:
        def key(*s: int, _K=TOPK) -> int:
            return _theta(tuple(x & M32 for x in s), _K) >> 32

        def at(*s: int, _K=TOPK) -> int:
            return M32 - (_theta(tuple(x & M32 for x in s), _K) & M32)

        p = _THRESHOLDS[(N, TOPK)] = tuple(
            PrimitiveDefinition(f"DsaTopKThreshold{part}{N}x{TOPK}", 1, (("scores", Array(N, F32)),), I32, ev,
                                doc=threshold.__doc__ or "", conformance=_CONF) for part, ev in (("Key", key), ("Index", at)))
    return p


@composite("DsaTopKMask", 1, ["N", "TOPK"], lambda S: ((("scores", Array(S.N, F32)),), Array(S.N, BIT)), conformance=_CONF,
           refuse=_refuse_sel)
def DsaTopKMask(B, S, scores):
    """The top-TOPK set of a row of N keys as a mask: bit j is 1 when j is in `topk_set`(scores, TOPK), else 0.  `DsaTopKMask_v2`
    is it on bits, an exact radix select with no advice."""
    key, at = (B.call(p, scores) for p in threshold(S.N, S.TOPK))
    idx = Coll(Array(S.N, I32), concat([B.call(const(32, j)).refs for j in range(S.N)]))
    return B.batch(DsaTopKAtLeast, scores, idx, key, at, axes=(0, 0, None, None))


def _counted_theta(scores, n: int, TOPK: int) -> int:
    return heapq.nlargest(TOPK, (ext(x & M32, j) if j < n else 0 for j, x in enumerate(scores)))[-1]


def counted_threshold(N: int, TOPK: int) -> tuple[PrimitiveDefinition, PrimitiveDefinition]:
    """`DsaTopKCountedThresholdKey{N}x{TOPK}_v1` and `DsaTopKCountedThresholdIndex{N}x{TOPK}_v1`: the TOPK-th largest ext over a
    row of N f32 scores whose keys from n (u32) on rank at 0, as its key and its index, one 32-bit word each (key 0 at index
    2^32 - 1, an ext of 0, when fewer than TOPK keys are below n)."""
    check_statics(N, TOPK)
    p = _COUNTED.get((N, TOPK))
    if p is None:
        def key(*w: int, _N=N, _K=TOPK) -> int:
            return _counted_theta(w[:_N], w[_N] & M32, _K) >> 32

        def at(*w: int, _N=N, _K=TOPK) -> int:
            return M32 - (_counted_theta(w[:_N], w[_N] & M32, _K) & M32)

        p = _COUNTED[(N, TOPK)] = tuple(
            PrimitiveDefinition(f"DsaTopKCountedThreshold{part}{N}x{TOPK}", 1, (("scores", Array(N, F32)), ("n", I32)), I32, ev,
                                doc=counted_threshold.__doc__ or "", conformance=_CONF) for part, ev in (("Key", key), ("Index", at)))
    return p


@primitive("DsaTopKAtLeastBelow", 1, [("s", F32), ("j", I32), ("n", I32), ("key", I32), ("at", I32)], BIT,
           conformance="exact-model-tested")
def DsaTopKAtLeastBelow(s: int, j: int, n: int, key: int, at: int) -> int:
    return int((j & M32) < (n & M32) and ext(s, j) >= ext_of(key, at))


@composite("DsaTopKMaskCounted", 1, ["N", "TOPK"], lambda S: ((("scores", Array(S.N, F32)), ("n", I32)), Array(S.N, BIT)),
           conformance=_CONF, refuse=_refuse_sel)
def DsaTopKMaskCounted(B, S, scores, n):
    """`DsaTopKMask` over a row's first n (u32) of N keys: bit j is 1 when j < n and j is in `topk_set`(scores[:n], TOPK), every
    one of them when n <= TOPK, else 0; so one Definition of N keys serves every row of at most N.  The threshold ranks a key from
    n on at ext 0, below every key's ext, so at n >= TOPK it is the row's own.  `DsaTopKMaskCounted_v2` is it on bits."""
    key, at = (B.call(p, scores, n) for p in counted_threshold(S.N, S.TOPK))
    idx = Coll(Array(S.N, I32), concat([B.call(const(32, j)).refs for j in range(S.N)]))
    return B.batch(DsaTopKAtLeastBelow, scores, idx, n, key, at, axes=(0, 0, None, None, None))


def slot(N: int, TOPK: int) -> PrimitiveDefinition:
    """`DsaTopKSlot{N}x{TOPK}_v1`: slot s of `sorted(topk_set(scores, TOPK))` (0 when s is TOPK or more)."""
    check_statics(N, TOPK)
    p = _SLOTS.get((N, TOPK))
    if p is None:
        # The last row's members: a Call of DsaTopKRow_v1 reads one row TOPK times.  Not an lru_cache, which
        # test_declared_tables reads as a table loader.
        last: list = [((), ())]

        def ev(*w: int, _N=N, _K=TOPK) -> int:
            scores, s = tuple(x & M32 for x in w[:_N]), w[_N] & M32
            seen, got = last[0]
            if seen != scores:
                got = tuple(sorted(topk_set(scores, _K)))
                last[0] = (scores, got)
            return got[s] if s < _K else 0

        p = PrimitiveDefinition(f"DsaTopKSlot{N}x{TOPK}", 1, (("scores", Array(N, F32)), ("s", I32)), I32, ev,
                                doc=slot.__doc__ or "", conformance=_CONF)
        _SLOTS[(N, TOPK)] = p
    return p


@composite("DsaTopKRow", 1, ["N", "TOPK"], lambda S: ((("scores", Array(S.N, F32)),), Array(S.TOPK, I32)), conformance=_CONF,
           refuse=_refuse_net)
def DsaTopKRow(B, S, scores):
    """The long row's selected indices in ascending order: `sorted(topk_set(scores, TOPK))`.  `DsaTopKRow_v2` is it on bits, the
    concentrator at the mask's settings over the indices 0..N-1, so this Definition's word view is the check that the mask form
    outputs the top-TOPK set in ascending order."""
    pick = slot(S.N, S.TOPK)
    return Coll(Array(S.TOPK, I32), concat([B.call(pick, scores, B.call(const(32, s))).refs for s in range(S.TOPK)]))


# --- the concentrator --------------------------------------------------------------------------------------------------------

@primitive("DsaMaskCount", 1, [("c", I32), ("m", BIT)], I32, conformance="exact-model-tested")
def DsaMaskCount(c: int, m: int) -> int:
    return (c + (m & 1)) & M32


@primitive("DsaMaskTag", 1, [("c", I32), ("m", BIT)], BF16, conformance="exact-model-tested")
def DsaMaskTag(c: int, m: int) -> int:
    return 1 << RANK_BITS | c & ((1 << RANK_BITS) - 1) if m & 1 else 0


@composite("DsaMaskTagStep", 1, [], lambda S: ((("c", I32), ("m", BIT)), Tuple(I32, BF16)), conformance="exact-model-tested")
def DsaMaskTagStep(B, S, c, m):
    """One key of the tags' scan: the members before it, c, counted on (`DsaMaskCount`), and its tag (`DsaMaskTag`)."""
    return tuple_of(B.call(DsaMaskCount, c, m), B.call(DsaMaskTag, c, m))


@primitive("DsaButterflySet", 1, [("t", I32), ("kb", I32), ("lo", BF16), ("hi", BF16)], BIT, conformance="exact-model-tested")
def DsaButterflySet(t: int, kb: int, lo: int, hi: int) -> int:
    vl, vh = lo >> RANK_BITS & 1, hi >> RANK_BITS & 1
    if t >= kb:
        return vh
    return lo >> t & 1 if vl else vh & ~hi >> t & 1


def _sig_route(S):
    return (("lo", BF16), ("hi", BF16)), Tuple(BIT, BF16, BF16) if S.TWO else Tuple(BIT, BF16)


@composite("DsaButterflyRoute", 1, ["T", "KB", "TWO"], _sig_route, conformance="exact-model-tested")
def DsaButterflyRoute(B, S, lo, hi):
    """A stage-T switch over two tags: its setting (`DsaButterflySet` at t = T, kb = KB), its lower output `Select`(s, hi, lo) and,
    when TWO, its upper `Xor3x16`(lo, hi, lower)."""
    s = B.call(DsaButterflySet, B.call(const(32, S.T)), B.call(const(32, S.KB)), lo, hi)
    low = B.call(SelectBf16, s, hi, lo)
    return tuple_of(s, low, B.call(Xor3x16, lo, hi, low)) if S.TWO else tuple_of(s, low)


@composite("DsaButterflySettings", 1, ["N", "K"],
           lambda S: ((("mask", Array(S.N, BIT)),), Array(butterfly_switches(S.N, S.K), BIT)), conformance="exact-model-tested",
           refuse=_refuse_net)
def DsaButterflySettings(B, S, mask):
    """The concentrator's settings for ``mask``'s members (the module docstring's rule), in `butterfly_plan` order: each member's
    tag (2^15 + its rank mod 2^15, a scan of `DsaMaskTagStep`) routed through the switches as they are set (`DsaButterflyRoute`).
    At a mask of K members `DsaButterflyRows` then lists their rows in index order; at any mask they are `butterfly_settings`."""
    tags = B.scan(DsaMaskTagStep, B.call(const(32, 0)), xs=(mask,))[1]
    kb = rank_bits(S.K)
    return settings(B, S.N, S.K, BF16, tags, B.call(const(16, 0)),
                    lambda t, two: bind(DsaButterflyRoute, T=t, KB=kb, TWO=int(two)))


def _row(S):
    return Array(S.W, Value(S.E))


@composite("DsaButterflyRows", 1, ["N", "K", "W", "E"],
           lambda S: ((("sw", Array(butterfly_switches(S.N, S.K), BIT)), ("rows", Array(S.N, _row(S)))), Array(S.K, _row(S))),
           conformance="exact-model-tested", refuse=_refuse_net)
def DsaButterflyRows(B, S, sw, rows_):
    """The concentrator over N rows of W E-bit words at settings ``sw``, outputs 0..K-1 (`DsaWaksmanSwap` and `DsaWaksmanPick` its
    switches, an empty position a zero row): at `DsaButterflySettings` of a mask of K members, slot s is the s-th member's row."""
    return rows(B, S.N, S.K, _row(S), sw, rows_, B.call(const(S.E, 0)), bind(DsaWaksmanSwap, W=S.W, E=S.E),
                bind(DsaWaksmanPick, W=S.W, E=S.E))


def _sig_rows_pad(S):
    return ((("sw", Array(butterfly_switches(S.N, S.K), BIT)), ("rows", Array(S.T, _row(S))), ("pad", Array(S.N - S.T, _row(S)))),
            Array(S.K, _row(S)))


def _refuse_pad(b: dict) -> str | None:
    T, N = b.get("T"), b.get("N")
    if isinstance(T, int) and isinstance(N, int) and not 1 <= T < N:
        return f"T rows of history and N - T >= 1 of pad (T={T}, N={N})"
    return _refuse_net(b)


@composite("DsaButterflyRowsPad", 1, ["T", "N", "K", "W", "E"], _sig_rows_pad, conformance="exact-model-tested", refuse=_refuse_pad)
def DsaButterflyRowsPad(B, S, sw, rows_, pad):
    """`DsaButterflyRows_v1{N,K,W,E}` over the T rows of ``rows`` and then the N - T of ``pad``: a row of T keys read at its size
    class N. Its history is an argument of its own, a key sequence one row longer than the row before's, which the descriptor
    writes as an alias to that one plus a part, and its pad another; ahead of the history in one argument, the pad left no
    earlier argument to alias, and each row's history was written whole."""
    return B.call(bind(DsaButterflyRows, N=S.N, K=S.K, W=S.W, E=S.E), sw, Coll(Array(S.N, _row(S)), concat([rows_.refs, pad.refs])))


DSA_TOPK_MASK_COMPOSITES = [DsaTopKMask, DsaTopKMaskCounted, DsaTopKRow, DsaMaskTagStep, DsaButterflyRoute, DsaButterflySettings,
                            DsaButterflyRows, DsaButterflyRowsPad]

#: the Calls serving stages as their cut, by family: the counted mask past `dsa_topk_mask_cut.WHOLE_MAX` keys, its threshold in
#: digit rounds over blocks of keys and its mask a block a unit (`dsa_topk_mask_cut.cut`), which circuit-check holds to the block
#: in the Call's place (`checks.cut_of`).  It names the module rather than importing it: that module's units are this one's
#: Boolean versions' parts
CUTS = {"DsaTopKMaskCounted_v1": "verity.definitions.registry.dsa_topk_mask_cut"}
