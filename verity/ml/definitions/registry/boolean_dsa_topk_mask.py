"""`dsa_topk_mask`'s Definitions on bits.  These are the definitions; `dsa_topk_mask`'s word versions are accelerators that
circuit-check and integrations/vllm/tests/program/test_dsa_topk_mask.py hold equal to them.

    DsaTopKKey_v2                  (s: 32 bits) -> 32 bits; word view DsaTopKKey_v1
    DsaRadixPass_v1{N,TOPK}        (eq, gt: N bits, r: RB bits, e: N bits) -> (N bits, N bits, RB bits); traced, no word
                                   Definition (RB = TOPK's bit length)
    DsaRadixTies_v1{N,TOPK}        (eq, gt: N bits, r: RB bits) -> N bits; traced, no word Definition
    DsaTopKMask_v2{N,TOPK}         (scores: N x 32 bits) -> N bits; word view DsaTopKMask_v1
    DsaRowValid_v1{N}              (n: 32 bits) -> N bits, bit j n > j; traced, no word Definition
    DsaTopKMaskCounted_v2{N,TOPK}  (scores: N x 32 bits, n: 32 bits) -> N bits; word view DsaTopKMaskCounted_v1
    DsaButterflyTagBits_v1{N,K}    (mask: N bits) -> N x (1 + kb) bits; traced, no word Definition
    DsaButterflyCtl_v1{T,KB,TWO}   (lo, hi: 1 + KB bits) -> (bit, 1 + KB bits[, 1 + KB bits]); traced, no word Definition
    DsaButterflySettings_v2{N,K}   (mask: N bits) -> butterfly_switches(N, K) bits; word view DsaButterflySettings_v1
    DsaButterflyRows_v2{N,K,W,E}   (sw: bits, rows: N x W x E bits) -> K x W x E bits; word view DsaButterflyRows_v1
    DsaButterflyRowsPad_v2{T,N,K,W,E}  (sw: bits, rows: T x W x E bits, pad: (N - T) x W x E bits) -> K x W x E bits; word view
                                   DsaButterflyRowsPad_v1
    DsaTopKRow_v2{N,TOPK}          (scores: N x 32 bits) -> TOPK x 32 bits; word view DsaTopKRow_v1

`DsaTopKMask_v2` is an exact radix select over ext_j = (key_j, N - 1 - j) with no advice: the key's 32 bits most significant
first, a `DsaRadixPass` each, then ext's index part at once (`DsaRadixTies`).  Before a pass over key bit i, eq_j says key_j agrees
with the threshold's key above bit i and gt_j that it is greater there, and r is how many members the keys still level with it
must supply.  The pass: a_j = eq_j AND e_j, cnt = popcount(a) (a carry-save tree), p = (cnt >= r) the threshold's bit i,
r <- p ? r : r - cnt, gt_j <- gt_j OR (a_j AND NOT p), eq_j <- eq_j AND (e_j = p): four ANDs a key.  After the 32 passes eq_j says
key_j equals the threshold's key, and the r of those keys the set takes are the first r by index (a lower j a larger N - 1 - j):
m_j = gt_j OR (eq_j AND c_j < r), c_j = #{i < j : eq_i} (a Brent-Kung prefix count).

`DsaButterflySettings_v2` is `dsa_topk_mask.settings` with each tag (valid, rank mod 2^kb) as bits (`DsaButterflyTagBits`, a
Brent-Kung prefix count of the mask) and each switch's setting and tag routing traced at its stage (`DsaButterflyCtl`): a stage-t
switch routes only the rank bits past t, so its tags cost kb - t ANDs.  `DsaButterflyRows_v2` is `dsa_topk_mask.rows` of
`DsaWaksmanSwap_v2` and `DsaWaksmanPick_v2`, one AND a payload bit a switch, and `DsaTopKRow_v2` is the three over the indices'
constant bits.

ANDs (`verity_flock.ir_lower.definition_ands`) at TOPK = 2,048:

    N          mask         settings     switches   rows, W 576 E 16   DsaTopKRow
    4,096         631,138      230,892     24,576       226,492,416      1,648,462
    32,768      5,239,298    1,865,080    210,944     1,944,059,904     13,854,586
    131,072    21,483,862    7,468,176    849,920     7,832,862,720     56,149,478
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ZERO, add, compress_columns, mux, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.refs import Coll, broadcast, concat, strided_view
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import dsa_topk_mask as M
from verity.ml.definitions.registry.boolean_dsa_topk import DsaWaksmanPickV2, DsaWaksmanSwapV2, key
from verity.ml.definitions.registry.dsa_topk import check_statics

BIT = BL.BIT
F32 = I32 = Array(32, BIT)


def _rep(c: Coll, n: int):
    """n references to ``c``'s one leaf."""
    return broadcast(c.refs.space, c.refs.idx, c.refs.base, n)


def prefix_counts(C, bits: list, width: int) -> list:
    """The exclusive prefix counts of ``bits`` mod 2^width, ``width`` bits each LSB first: Brent-Kung's up and down sweeps."""
    n = len(bits)
    L = 1 << max(0, (n - 1).bit_length())
    v = [[b] for b in bits] + [[ZERO]] * (L - n)
    d = 1
    while d < L:
        for i in range(2 * d - 1, L, 2 * d):
            v[i] = add(C, v[i - d], v[i], width=min(width, max(len(v[i - d]), len(v[i])) + 1))
        d *= 2
    v[L - 1] = [ZERO] * width
    d = L // 2
    while d >= 1:
        for i in range(2 * d - 1, L, 2 * d):
            left = v[i - d]
            v[i - d] = v[i]
            v[i] = add(C, v[i], left, width=width)
        d //= 2
    return [(x + [ZERO] * width)[:width] for x in v[:n]]


DsaTopKKeyV2 = definition("DsaTopKKey", 2, lambda C, s: key(list(s)), (("s", F32),), I32, word=M.DsaTopKKey,
                          doc="`dsa_topk.topk_key` on bits (no AND): DsaTopKKey_v1")


def _rb(S) -> int:
    return S.TOPK.bit_length()


def _radix_pass(S):
    """`DsaRadixPass_v1{N,TOPK}`: one bit of the radix select (the module docstring), e_j the keys' bit; returns (eq, gt, r)."""
    check_statics(S.N, S.TOPK)
    RB = _rb(S)

    def build(C, eq, gt, r, e):
        a = [C.AND(x, y) for x, y in zip(eq, e)]
        cnt = compress_columns(C, [a])
        p = NOT(sub(C, cnt, list(r), max(len(cnt), RB))[1])
        left = mux(C, p, list(r), sub(C, list(r), cnt, RB)[0])
        pe, pa = [C.AND(p, x) for x in eq], [C.AND(p, x) for x in a]
        return [[q ^ x ^ y for q, x, y in zip(eq, a, pe)], [g ^ x ^ y for g, x, y in zip(gt, a, pa)], left]
    return build


DsaRadixPass = EW.specialized(
    "DsaRadixPass", 1, ["N", "TOPK"], _radix_pass,
    lambda S: ((("eq", Array(S.N, BIT)), ("gt", Array(S.N, BIT)), ("r", Array(_rb(S), BIT)), ("e", Array(S.N, BIT))),
               Tuple(Array(S.N, BIT), Array(S.N, BIT), Array(_rb(S), BIT))), doc=_radix_pass.__doc__)


def _radix_ties(S):
    """`DsaRadixTies_v1{N,TOPK}`: the mask from the passes' (eq, gt, r): m_j = gt_j OR (eq_j AND #{i < j : eq_i} < r)."""
    check_statics(S.N, S.TOPK)
    w, RB = max(1, (S.N - 1).bit_length()), _rb(S)

    def build(C, eq, gt, r):
        c = prefix_counts(C, list(eq), w)
        return [[g ^ C.AND(q, sub(C, cj, list(r), max(w, RB))[1]) for g, q, cj in zip(gt, eq, c)]]
    return build


DsaRadixTies = EW.specialized(
    "DsaRadixTies", 1, ["N", "TOPK"], _radix_ties,
    lambda S: ((("eq", Array(S.N, BIT)), ("gt", Array(S.N, BIT)), ("r", Array(_rb(S), BIT))), Array(S.N, BIT)),
    doc=_radix_ties.__doc__)


def _row_valid(S):
    """`DsaRowValid_v1{N}`: bit j is n > j (n u32), for a row's first n of N keys.  A binary trie over n's low L = ceil(log2 N)
    bits from the top, each node of a prefix p holding (E, G): n's bits so far are p, or exceed it; a node's children cost one AND,
    t = E AND n_i: the 0 child (E XOR t, G XOR t), the 1 child (t, G).  The root's G is n >= 2^L; a node past N is dropped."""
    L = (S.N - 1).bit_length()

    def build(C, n):
        g = or_reduce(C, n[L:])
        level = [(NOT(g), g)]
        for i in reversed(range(L)):
            nxt = []
            for e, x in level:
                t = C.AND(e, n[i])
                nxt += [(e ^ t, x ^ t), (t, x)]
            level = nxt[:-(-S.N >> i)]
        return [[x for _, x in level]]
    return build


DsaRowValid = EW.specialized("DsaRowValid", 1, ["N"], _row_valid, lambda S: ((("n", I32),), Array(S.N, BIT)),
                             doc=_row_valid.__doc__)


def _select(B, S, scores, n=None):
    """The radix select over ``scores``' keys (`DsaTopKKey_v2`): 32 `DsaRadixPass_v1` from eq all 1 (with ``n``, `DsaRowValid_v1`),
    gt all 0 and r = TOPK, and `DsaRadixTies_v1`."""
    check_statics(S.N, S.TOPK)
    N, RB = S.N, _rb(S)
    keys = B.batch(DsaTopKKeyV2, scores, axes=(0,))
    one, zero = B.call(BL.constant(1)), B.call(BL.constant(0))
    eq = Coll(Array(N, BIT), _rep(one, N)) if n is None else B.call(bind(DsaRowValid, N=N), n)
    gt = Coll(Array(N, BIT), _rep(zero, N))
    r = Coll(Array(RB, BIT), concat([(one if S.TOPK >> i & 1 else zero).refs for i in range(RB)]))
    step = bind(DsaRadixPass, N=N, TOPK=S.TOPK)
    for i in reversed(range(32)):
        out = B.call(step, eq, gt, r, Coll(Array(N, BIT), strided_view(keys, i, 1, 32, 1, N)))
        eq, gt, r = out[0], out[1], out[2]
    return B.call(bind(DsaRadixTies, N=N, TOPK=S.TOPK), eq, gt, r)


def _mask(B, S, scores):
    """`DsaTopKMask_v1` on bits: the keys (`DsaTopKKey_v2`), 32 `DsaRadixPass_v1` from eq all 1, gt all 0 and r = TOPK, and
    `DsaRadixTies_v1`."""
    return _select(B, S, scores)


DsaTopKMaskV2 = EW.on_bits("DsaTopKMask", 2, ["N", "TOPK"], lambda S: ((("scores", Array(S.N, F32)),), Array(S.N, BIT)), _mask,
                           M.DsaTopKMask)


def _counted_mask(B, S, scores, n):
    """`DsaTopKMaskCounted_v1` on bits: `DsaTopKMask_v2`'s radix select with eq starting at `DsaRowValid_v1{N}`(n), so a key from n
    on is never counted, never above the threshold and never a tie: the passes select among the first n keys, all of them when
    n <= TOPK (each pass's count then below r)."""
    return _select(B, S, scores, n)


DsaTopKMaskCountedV2 = EW.on_bits("DsaTopKMaskCounted", 2, ["N", "TOPK"],
                                  lambda S: ((("scores", Array(S.N, F32)), ("n", I32)), Array(S.N, BIT)), _counted_mask,
                                  M.DsaTopKMaskCounted)


def _tag(K: int):
    return Array(1 + M.rank_bits(K), BIT)


def _tag_bits(S):
    """`DsaButterflyTagBits_v1{N,K}`: key j's tag, (m_j, #{i < j : m_i} mod 2^kb LSB first)."""
    M.check_net(S.N, S.K)
    kb = M.rank_bits(S.K)

    def build(C, mask):
        return [[m, *c] for m, c in zip(mask, prefix_counts(C, list(mask), kb))]
    return build


DsaButterflyTagBits = EW.specialized("DsaButterflyTagBits", 1, ["N", "K"], _tag_bits,
                                     lambda S: ((("mask", Array(S.N, BIT)),), Array(S.N, _tag(S.K))), doc=_tag_bits.__doc__)


def _ctl(S):
    """`DsaButterflyCtl_v1{T,KB,TWO}`: a stage-T switch over two tags (valid, KB rank bits): its setting (d_lo[T] if the lower is
    valid, else the upper valid and d_hi[T] = 0; past the rank bits the upper valid), and the tags `Select`ed and XORed through it,
    the valid bit and the rank bits past T only (the rest 0); the upper tag when TWO."""
    T, KB = S.T, S.KB
    keep = [0, *range(2 + T, 1 + KB)]

    def build(C, lo, hi):
        s = mux(C, lo[0], [lo[1 + T]], [C.AND(hi[0], NOT(hi[1 + T]))])[0] if T < KB else hi[0]
        low, high = [ZERO] * (1 + KB), [ZERO] * (1 + KB)
        for i in keep:
            low[i] = mux(C, s, [hi[i]], [lo[i]])[0]
            high[i] = lo[i] ^ hi[i] ^ low[i]
        return [[s], low, high] if S.TWO else [[s], low]
    return build


def _sig_ctl(S):
    X = Array(1 + S.KB, BIT)
    return (("lo", X), ("hi", X)), Tuple(BIT, X, X) if S.TWO else Tuple(BIT, X)


DsaButterflyCtl = EW.specialized("DsaButterflyCtl", 1, ["T", "KB", "TWO"], _ctl, _sig_ctl, doc=_ctl.__doc__)


def _settings(B, S, mask):
    """`DsaButterflySettings_v1` on bits: `dsa_topk_mask.settings` over `DsaButterflyTagBits_v1`'s tags and `DsaButterflyCtl_v1`."""
    kb = M.rank_bits(S.K)
    tags = B.call(bind(DsaButterflyTagBits, N=S.N, K=S.K), mask)
    return M.settings(B, S.N, S.K, _tag(S.K), tags, B.call(BL.constant(0)),
                      lambda t, two: bind(DsaButterflyCtl, T=t, KB=kb, TWO=int(two)))


DsaButterflySettingsV2 = EW.on_bits(
    "DsaButterflySettings", 2, ["N", "K"],
    lambda S: ((("mask", Array(S.N, BIT)),), Array(M.butterfly_switches(S.N, S.K), BIT)), _settings, M.DsaButterflySettings)


def _row(S):
    return Array(S.W, Array(S.E, BIT))


def _rows(B, S, sw, rows):
    """`DsaButterflyRows_v1` on bits: `dsa_topk_mask.rows` of `DsaWaksmanSwap_v2` and `DsaWaksmanPick_v2`, one AND a payload bit
    a switch."""
    return M.rows(B, S.N, S.K, _row(S), sw, rows, B.call(BL.constant(0)), bind(DsaWaksmanSwapV2, W=S.W, E=S.E),
                  bind(DsaWaksmanPickV2, W=S.W, E=S.E))


DsaButterflyRowsV2 = EW.on_bits(
    "DsaButterflyRows", 2, ["N", "K", "W", "E"],
    lambda S: ((("sw", Array(M.butterfly_switches(S.N, S.K), BIT)), ("rows", Array(S.N, _row(S)))), Array(S.K, _row(S))),
    _rows, M.DsaButterflyRows)


def _rows_pad(B, S, sw, rows, pad):
    """`DsaButterflyRowsPad_v1` on bits: `DsaButterflyRows_v2{N,K,W,E}` over the T rows and then the pad."""
    return B.call(bind(DsaButterflyRowsV2, N=S.N, K=S.K, W=S.W, E=S.E), sw, Coll(Array(S.N, _row(S)), concat([rows.refs, pad.refs])))


DsaButterflyRowsPadV2 = EW.on_bits(
    "DsaButterflyRowsPad", 2, ["T", "N", "K", "W", "E"],
    lambda S: ((("sw", Array(M.butterfly_switches(S.N, S.K), BIT)), ("rows", Array(S.T, _row(S))), ("pad", Array(S.N - S.T, _row(S)))),
               Array(S.K, _row(S))),
    _rows_pad, M.DsaButterflyRowsPad)


def _row_check(B, S, scores):
    """`DsaTopKRow_v1` on bits: `DsaTopKMask_v2`, its `DsaButterflySettings_v2{N,K=TOPK}`, and `DsaButterflyRows_v2{N,TOPK,1,32}`
    over the indices' constant bits."""
    mask = B.call(bind(DsaTopKMaskV2, N=S.N, TOPK=S.TOPK), scores)
    sw = B.call(bind(DsaButterflySettingsV2, N=S.N, K=S.TOPK), mask)
    zero, one = B.call(BL.constant(0)), B.call(BL.constant(1))
    idx = concat([(one if j >> i & 1 else zero).refs for j in range(S.N) for i in range(32)])
    return Coll(Array(S.TOPK, I32), B.call(bind(DsaButterflyRowsV2, N=S.N, K=S.TOPK, W=1, E=32), sw,
                                           Coll(Array(S.N, Array(1, I32)), idx)).refs)


DsaTopKRowV2 = EW.on_bits("DsaTopKRow", 2, ["N", "TOPK"], lambda S: ((("scores", Array(S.N, F32)),), Array(S.TOPK, I32)),
                          _row_check, M.DsaTopKRow)

BOOLEAN_DSA_TOPK_MASK = [DsaTopKKeyV2, DsaRadixPass, DsaRadixTies, DsaTopKMaskV2, DsaRowValid, DsaTopKMaskCountedV2,
                         DsaButterflyTagBits, DsaButterflyCtl, DsaButterflySettingsV2, DsaButterflyRowsV2, DsaButterflyRowsPadV2,
                         DsaTopKRowV2]

#: `dsa_topk_mask.CUTS` on bits: the counted mask past `dsa_topk_mask_cut.WHOLE_MAX` keys as `dsa_topk_mask_cut`'s units, which
#: import this module
CUTS = {"DsaTopKMaskCounted_v2": "verity.ml.definitions.registry.dsa_topk_mask_cut"}
