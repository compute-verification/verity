"""`dsa_topk_mask`'s Definitions on bits.  These are the definitions; `dsa_topk_mask`'s word versions are accelerators that
circuit-check and adapters/vllm/tests/program/test_dsa_topk_mask.py hold equal to them.

    DsaTopKKey_v2                  (s: 32 bits) -> 32 bits; word view DsaTopKKey_v1
    DsaRadixPass_v1{N,TOPK}        (eq, gt: N bits, r: RB bits, e: N bits) -> (N bits, N bits, RB bits); traced, no word
                                   Definition (RB = TOPK's bit length)
    DsaRadixPass_v2{N,TOPK}        the same pass as Calls to And2_v1, DsaRadixPassCount_v1{N}, DsaCountSum_v1{W},
                                   DsaRadixPassLeft_v1{W,RB} and DsaRadixPassKey_v1; no word Definition
    DsaRadixTies_v1{N,TOPK}        (eq, gt: N bits, r: RB bits) -> N bits; traced, no word Definition
    DsaRadixTies_v2{N,TOPK}        the same mask as Calls to DsaCountSum_v1{W}, DsaCountSplit_v1{W,U} and DsaRadixTie_v1{W,RB}, the
                                   steps of its prefix count and a key's tie; no word Definition
    DsaTopKMask_v2{N,TOPK}         (scores: N x 32 bits) -> N bits; word view DsaTopKMask_v1
    DsaRowValid_v1{N}              (n: 32 bits) -> N bits, bit j n > j; traced, no word Definition
    DsaRowValid_v2{N}              the same bits as Calls to DsaRowValidRoot_v1{L}, DsaRowValidStep_v1 and DsaRowValidLast_v1, a
                                   batch a level of its trie; no word Definition
    DsaTopKMaskCounted_v2{N,TOPK}  (scores: N x 32 bits, n: 32 bits) -> N bits; word view DsaTopKMaskCounted_v1
    DsaButterflyTagBits_v1{N,K}    (mask: N bits) -> N x (1 + kb) bits; traced, no word Definition
    DsaButterflyTagsCount_v1{N,K}  (mask: N bits) -> (those tags, the mask's count: 32 bits), as Calls to DsaCountSum_v1{W},
                                   DsaCountSplit_v1{W,U} and DsaButterflyTag_v1{W,KB}; no word Definition
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

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ZERO, add, compress_columns, mux, or_reduce, sub
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import CompositeDefinition, bind
from verity.definitions.ops.boolean import elementwise as EW
from verity.circuits.refs import Coll, broadcast, concat, strided_view, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import dsa_topk_mask as M
from verity.definitions.registry.boolean_dsa import own_gates
from verity.definitions.registry.boolean_dsa_topk import DsaWaksmanPickV2, DsaWaksmanSwapV2, key
from verity.definitions.registry.dsa_topk import check_statics

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


def _piece(name: str, statics: list, build, signature):
    """A step of `DsaRadixTies_v2`, traced, with the builder that puts its own gates down wherever it is activated."""
    d = EW.specialized(name, 1, statics, build, signature, doc=build.__doc__)
    d.build = own_gates(d)
    return d


def _count_sum(S):
    """`DsaCountSum_v1{W}`: the sum of two counts of W bits, in W + 1: a step up `prefix_counts`' tree, two halves' counts to
    the whole's."""
    return lambda C, a, b: [add(C, list(a), list(b), width=S.W + 1)]


def _count_split(S):
    """`DsaCountSplit_v1{W,U}`: a step down `prefix_counts`' tree.  p (W bits) counts what comes before a node and u (U bits)
    what its left half holds: (p, p + u mod 2^W), what comes before its left half and before its right."""
    return lambda C, p, u: [list(p), add(C, list(p), list(u), width=S.W)]


def _tie(S):
    """`DsaRadixTie_v1{W,RB}`: one key of `DsaRadixTies`' mask, gt OR (eq AND c < r), c (W bits) the equal keys before it; at
    W = 0 the row's first key, with none before it and no c."""
    return lambda C, g, q, *cr: [[g[0] ^ C.AND(q[0], sub(C, [x for c in cr[:-1] for x in c], list(cr[-1]), max(S.W, S.RB))[1])]]


def _counts(n: int, width: int) -> Array:
    return Array(n, Array(width, BIT))


DsaCountSum = _piece("DsaCountSum", ["W"], _count_sum,
                     lambda S: ((("a", Array(S.W, BIT)), ("b", Array(S.W, BIT))), Array(S.W + 1, BIT)))
DsaCountSplit = _piece("DsaCountSplit", ["W", "U"], _count_split,
                       lambda S: ((("p", Array(S.W, BIT)), ("u", Array(S.U, BIT))), _counts(2, S.W)))
DsaRadixTie = _piece("DsaRadixTie", ["W", "RB"], _tie,
                     lambda S: ((("gt", BIT), ("eq", BIT), *((("c", Array(S.W, BIT)),) if S.W else ()), ("r", Array(S.RB, BIT))), BIT))


def _count_tree(B, N: int, bits: Coll, top: int | None = None) -> tuple[list[Coll], dict[int, Coll]]:
    """`prefix_counts`' tree over N bits as Calls: (the sums going up, level k's the counts of its nodes of 2^k keys that lie
    whole inside the row, k + 1 bits each; the count before each key but the first, key 2^(h-1) .. 2^h - 1's under h, in h
    bits, or in ``top`` bits mod 2^top where h is past it).  Going up, a level's nodes are summed in pairs
    (`DsaCountSum_v1{W=k}`).  Going down, a node hands the count before it to its left half and that plus its left half's own
    to its right (`DsaCountSplit_v1`); a level's first node has nothing before it and its second the first's count, by
    reference.  The count before node t of level k is below t 2^k, so nodes 2^(h-1) .. 2^h - 1 carry it in k + h bits, a
    batch for each h."""
    depth = (N - 1).bit_length()
    up = [Coll(_counts(N, 1), bits.refs)]
    for k in range(1, depth):
        t, low = N >> k, up[-1]
        up.append(B.batch(bind(DsaCountSum, W=k), *(Coll(_counts(t, k), strided_view(low, k * i, k, 2 * k, 1, t * k)) for i in (0, 1)),
                          axes=(0, 0)))
    before: dict[int, Coll] = {}
    for k in reversed(range(depth)):
        n, u, above = -(-N >> k), k + 1, before
        first = min(u, top or u)
        before = {1: Coll(_counts(1, first), up[k].refs.slice(0, first))}
        for h in range(2, (n - 1).bit_length() + 1):
            lo, w = 1 << h - 1, min(k + h, top or k + h)
            here, own = min(2 * lo, n) - lo, min(u, w)
            parts, pairs = [], here // 2
            if pairs:
                parts.append(B.batch(bind(DsaCountSplit, W=w, U=own), Coll(_counts(pairs, w), above[h - 1].refs.slice(0, w * pairs)),
                                     Coll(_counts(pairs, own), strided_view(up[k], lo * u, own, 2 * u, 1, pairs * own)),
                                     axes=(0, 0)).refs)
            if here % 2:
                parts.append(above[h - 1].refs.slice(w * pairs, w * (pairs + 1)))
            before[h] = Coll(_counts(here, w), concat(parts))
    return up, before


def _sum_counts(B, zero: Coll, nodes: list, top: int) -> tuple:
    """The sum of counts given as (references, width), as (references, width) in at most ``top`` bits: `DsaCountSum_v1` over
    them in turn, the narrower operand padded with zeros."""
    total, wide = nodes[0]
    for node, n in nodes[1:]:
        w = max(wide, n)
        x, y = (Coll(Array(w, BIT), concat([v, _rep(zero, w - m)]) if w > m else v) for v, m in ((total, wide), (node, n)))
        wide = min(w + 1, top)
        total = B.call(bind(DsaCountSum, W=w), x, y).refs.slice(0, wide)
    return total, wide


def _ties_calls(B, S, eq, gt, r):
    """`DsaRadixTies_v2{N,TOPK}`: `DsaRadixTies_v1`'s mask, its prefix count written as Calls (`_count_tree`): a few batches
    for each level of `prefix_counts`' tree and not a gate for each key, then a `DsaRadixTie_v1` per key.  `_v1` carries each
    count in the bits no constant folds away and also sums the nodes past the row's end, which nothing reads: the two are
    one function and differ in ANDs."""
    check_statics(S.N, S.TOPK)
    N, (_, before) = S.N, _count_tree(B, S.N, eq)

    def keys(lo: int, n: int) -> list[Coll]:
        return [Coll(Array(n, BIT), x.refs.slice(lo, lo + n)) for x in (gt, eq)]

    def tie(w: int):
        return bind(DsaRadixTie, W=w, RB=_rb(S))

    return Coll(Array(N, BIT), concat([
        B.batch(tie(0), *keys(0, 1), r, axes=(0, 0, None)).refs,
        *(B.batch(tie(h), *keys(1 << h - 1, len(c)), c, r, axes=(0, 0, 0, None)).refs for h, c in sorted(before.items()))]))


DsaRadixTiesV2 = CompositeDefinition(
    "DsaRadixTies", 2, ("N", "TOPK"),
    lambda S: ((("eq", Array(S.N, BIT)), ("gt", Array(S.N, BIT)), ("r", Array(_rb(S), BIT))), Array(S.N, BIT)), _ties_calls,
    doc=_ties_calls.__doc__,
    conformance="exact-model-tested: held to DsaRadixTies_v1 and to the mask's rule by adapters/vllm/tests/program/test_dsa_long_row_calls.py")

#: how many keys one `DsaRadixPassCount_v1` counts: `DsaRadixPass_v2` counts a row in chunks of this many
PASS_CHUNK = 256


def _pass_count(S):
    """`DsaRadixPassCount_v1{N}`: how many of N bits are set, in N's bit length: `DsaRadixPass_v1`'s carry-save tree."""
    w = S.N.bit_length()
    return lambda C, a: [(compress_columns(C, [list(a)]) + [ZERO] * w)[:w]]


def _pass_left(S):
    """`DsaRadixPassLeft_v1{W,RB}`: a pass's (p, r') of its count (W bits) and r (RB bits): p = (cnt >= r), the threshold's
    bit, and r' = r where p, else r - cnt."""
    def build(C, cnt, r):
        p = NOT(sub(C, list(cnt), list(r), max(S.W, S.RB))[1])
        return [[p, *mux(C, p, list(r), sub(C, list(r), list(cnt), S.RB)[0])]]
    return build


def _pass_key(S):
    """`DsaRadixPassKey_v1`: a key's (eq, gt) after a pass, of its a = eq AND e and the pass's p: eq AND (e = p) and
    gt OR (a AND NOT p), two ANDs."""
    return lambda C, q, g, a, p: [[q[0] ^ a[0] ^ C.AND(p[0], q[0]), g[0] ^ a[0] ^ C.AND(p[0], a[0])]]


DsaRadixPassCount = _piece("DsaRadixPassCount", ["N"], _pass_count,
                           lambda S: ((("a", Array(S.N, BIT)),), Array(S.N.bit_length(), BIT)))
DsaRadixPassLeft = _piece("DsaRadixPassLeft", ["W", "RB"], _pass_left,
                          lambda S: ((("cnt", Array(S.W, BIT)), ("r", Array(S.RB, BIT))), Array(1 + S.RB, BIT)))
DsaRadixPassKey = _piece("DsaRadixPassKey", [], _pass_key,
                         lambda S: ((("eq", BIT), ("gt", BIT), ("a", BIT), ("p", BIT)), Array(2, BIT)))


def _pass_calls(B, S, eq, gt, r, e):
    """`DsaRadixPass_v2{N,TOPK}`: `DsaRadixPass_v1`'s pass as Calls: a an AND a key; its count the sum of
    `DsaRadixPassCount_v1` over chunks of `PASS_CHUNK` keys, the chunks' counts summed in pairs (`DsaCountSum_v1{W}`, a batch a
    level) and what no pair takes after them; `DsaRadixPassLeft_v1` once and `DsaRadixPassKey_v1` a key.  `_v1` counts the row
    in one carry-save tree: the two are one function, and this one pays the ANDs of summing the chunks' counts."""
    check_statics(S.N, S.TOPK)
    N, RB, top = S.N, _rb(S), S.N.bit_length()
    a = B.batch(BL.AND, Coll(Array(2, Array(N, BIT)), concat([eq.refs, e.refs])), axes=(1,))
    full, rest = divmod(N, PASS_CHUNK)
    nodes = []
    if full:
        t, w = full, PASS_CHUNK.bit_length()
        level = B.batch(bind(DsaRadixPassCount, N=PASS_CHUNK), Coll(Array(full, Array(PASS_CHUNK, BIT)), a.refs.slice(0, full * PASS_CHUNK)),
                        axes=(0,))
        while True:
            if t % 2:
                nodes.insert(0, (level.refs.slice((t - 1) * w, t * w), w))
            if t < 2:
                break
            t //= 2
            level = B.batch(bind(DsaCountSum, W=w), *(Coll(_counts(t, w), strided_view(level, w * i, w, 2 * w, 1, t * w)) for i in (0, 1)),
                            axes=(0, 0))
            w += 1
    if rest:
        nodes.append((B.call(bind(DsaRadixPassCount, N=rest), Coll(Array(rest, BIT), a.refs.slice(N - rest, N))).refs, rest.bit_length()))
    cnt, w = _sum_counts(B, B.call(BL.constant(0)), nodes, top)
    out = B.call(bind(DsaRadixPassLeft, W=w, RB=RB), Coll(Array(w, BIT), cnt), r)
    keys = B.batch(DsaRadixPassKey, eq, gt, a, Coll(BIT, out.refs.slice(0, 1)), axes=(0, 0, 0, None))
    return tuple_of(*(Coll(Array(N, BIT), strided_view(keys, i, 1, 2, 1, N)) for i in (0, 1)),
                    Coll(Array(RB, BIT), out.refs.slice(1, 1 + RB)))


DsaRadixPassV2 = CompositeDefinition(
    "DsaRadixPass", 2, ("N", "TOPK"),
    lambda S: ((("eq", Array(S.N, BIT)), ("gt", Array(S.N, BIT)), ("r", Array(_rb(S), BIT)), ("e", Array(S.N, BIT))),
               Tuple(Array(S.N, BIT), Array(S.N, BIT), Array(_rb(S), BIT))), _pass_calls, doc=_pass_calls.__doc__,
    conformance="exact-model-tested: held to DsaRadixPass_v1 and to the pass's rule by adapters/vllm/tests/program/test_dsa_long_row_calls.py")


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


def _valid_root(S):
    """`DsaRowValidRoot_v1{L}`: the root of `DsaRowValid_v1`'s trie over n's low L bits, (E, G) = (n < 2^L, n >= 2^L)."""
    def build(C, n):
        g = or_reduce(C, n[S.L:])
        return [[NOT(g), g]]
    return build


def _valid_step(S):
    """`DsaRowValidStep_v1`: a node (E, G) of `DsaRowValid_v1`'s trie and n's next bit b to its children, one AND,
    t = E AND b: the 0 child (E XOR t, G XOR t), the 1 child (t, G)."""
    def build(C, node, b):
        t = C.AND(node[0], b[0])
        return [[node[0] ^ t, node[1] ^ t], [t, node[1]]]
    return build


def _valid_last(S):
    """`DsaRowValidLast_v1`: a node (E, G) over n's last bit b to its two keys' bits, (G XOR (E AND b), G)."""
    return lambda C, node, b: [[node[1] ^ C.AND(node[0], b[0]), node[1]]]


def _nodes(n: int) -> Array:
    return Array(n, Array(2, BIT))


DsaRowValidRoot = _piece("DsaRowValidRoot", ["L"], _valid_root, lambda S: ((("n", I32),), Array(2, BIT)))
DsaRowValidStep = _piece("DsaRowValidStep", [], _valid_step, lambda S: ((("node", Array(2, BIT)), ("b", BIT)), _nodes(2)))
DsaRowValidLast = _piece("DsaRowValidLast", [], _valid_last, lambda S: ((("node", Array(2, BIT)), ("b", BIT)), Array(2, BIT)))


def _valid_calls(B, S, n):
    """`DsaRowValid_v2{N}`: `DsaRowValid_v1`'s trie as Calls, a batch a level: its root (`DsaRowValidRoot_v1{L}`), a
    `DsaRowValidStep_v1` for each node that reaches a key below N, and over n's last bit a `DsaRowValidLast_v1` for each
    pair of keys.  The same ANDs as `_v1`, one a node."""
    N, L = S.N, (S.N - 1).bit_length()
    level = B.call(bind(DsaRowValidRoot, L=L), n)
    if L == 0:
        return Coll(Array(1, BIT), level.refs.slice(1, 2))
    for i in reversed(range(L)):
        t = -(-N >> i + 1)
        level = B.batch(DsaRowValidStep if i else DsaRowValidLast, Coll(_nodes(t), level.refs.slice(0, 2 * t)),
                        Coll(BIT, n.refs.slice(i, i + 1)), axes=(0, None))
    return Coll(Array(N, BIT), level.refs.slice(0, N))


DsaRowValidV2 = CompositeDefinition(
    "DsaRowValid", 2, ("N",), lambda S: ((("n", I32),), Array(S.N, BIT)), _valid_calls, doc=_valid_calls.__doc__,
    conformance="exact-model-tested: held to DsaRowValid_v1 and to n > j by adapters/vllm/tests/program/test_dsa_long_row_calls.py")


def _select(B, S, scores, n=None, ties=DsaRadixTies, step=DsaRadixPass, valid=DsaRowValid):
    """The radix select over ``scores``' keys (`DsaTopKKey_v2`): 32 `DsaRadixPass_v1` from eq all 1 (with ``n``, `DsaRowValid_v1`),
    gt all 0 and r = TOPK, and `DsaRadixTies_v1` (or ``ties``, ``step`` and ``valid``, their `_v2`)."""
    check_statics(S.N, S.TOPK)
    N, RB = S.N, _rb(S)
    keys = B.batch(DsaTopKKeyV2, scores, axes=(0,))
    one, zero = B.call(BL.constant(1)), B.call(BL.constant(0))
    eq = Coll(Array(N, BIT), _rep(one, N)) if n is None else B.call(bind(valid, N=N), n)
    gt = Coll(Array(N, BIT), _rep(zero, N))
    r = Coll(Array(RB, BIT), concat([(one if S.TOPK >> i & 1 else zero).refs for i in range(RB)]))
    step = bind(step, N=N, TOPK=S.TOPK)
    for i in reversed(range(32)):
        out = B.call(step, eq, gt, r, Coll(Array(N, BIT), strided_view(keys, i, 1, 32, 1, N)))
        eq, gt, r = out[0], out[1], out[2]
    return B.call(bind(ties, N=N, TOPK=S.TOPK), eq, gt, r)


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


def _tag_of(S):
    """`DsaButterflyTag_v1{W,KB}`: a key's tag of its mask bit and the count before it mod 2^KB, which is its W bits (no
    gate)."""
    return lambda C, m, *c: [[m[0], *(c[0] if c else ()), *[ZERO] * (S.KB - S.W)]]


DsaButterflyTag = _piece("DsaButterflyTag", ["W", "KB"], _tag_of,
                         lambda S: ((("m", BIT), *((("c", Array(S.W, BIT)),) if S.W else ())), Array(1 + S.KB, BIT)))


def _tags_count(B, S, mask):
    """`DsaButterflyTagsCount_v1{N,K}`: (`DsaButterflyTagBits_v1{N,K}`'s tags, how many of the N bits are set, u32), by one
    tree of counts written as Calls (`_count_tree`, its counts going down mod 2^kb): a `DsaButterflyTag_v1` per key, and the
    count the sum of the tree's nodes that tile the row, one for each bit set in N."""
    M.check_net(S.N, S.K)
    N, kb, depth = S.N, M.rank_bits(S.K), (S.N - 1).bit_length()
    up, before = _count_tree(B, N, mask, top=kb)
    zero = B.call(BL.constant(0))

    def keys(lo: int, n: int) -> Coll:
        return Coll(Array(n, BIT), mask.refs.slice(lo, lo + n))

    tags = concat([B.batch(bind(DsaButterflyTag, W=0, KB=kb), keys(0, 1), axes=(0,)).refs,
                   *(B.batch(bind(DsaButterflyTag, W=min(h, kb), KB=kb), keys(1 << h - 1, len(c)), c, axes=(0, 0)).refs
                     for h, c in sorted(before.items()))])
    tiles = ([(depth - 1, 0), (depth - 1, 1)] if N == 1 << depth
             else [(k, N >> k + 1 << 1) for k in reversed(range(depth)) if N >> k & 1])
    total, wide = _sum_counts(B, zero, [(up[k].refs.slice(t * (k + 1), (t + 1) * (k + 1)), k + 1) for k, t in tiles], depth + 1)
    return tuple_of(Coll(Array(N, _tag(S.K)), tags), Coll(I32, concat([total, _rep(zero, 32 - wide)])))


DsaButterflyTagsCount = CompositeDefinition(
    "DsaButterflyTagsCount", 1, ("N", "K"),
    lambda S: ((("mask", Array(S.N, BIT)),), Tuple(Array(S.N, _tag(S.K)), I32)), _tags_count, doc=_tags_count.__doc__,
    conformance="exact-model-tested: held to DsaButterflyTagBits_v1 and to the bits' count by "
                "adapters/vllm/tests/program/test_dsa_long_row_calls.py")


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

BOOLEAN_DSA_TOPK_MASK = [DsaTopKKeyV2, DsaRadixPass, DsaRadixTies, DsaCountSum, DsaCountSplit, DsaRadixTie, DsaRadixTiesV2,
                         DsaRadixPassCount, DsaRadixPassLeft, DsaRadixPassKey, DsaRadixPassV2,
                         DsaTopKMaskV2, DsaRowValid, DsaRowValidRoot, DsaRowValidStep, DsaRowValidLast, DsaRowValidV2, DsaTopKMaskCountedV2,
                         DsaButterflyTagBits, DsaButterflyTag, DsaButterflyTagsCount, DsaButterflyCtl, DsaButterflySettingsV2, DsaButterflyRowsV2, DsaButterflyRowsPadV2,
                         DsaTopKRowV2]

#: `dsa_topk_mask.CUTS` on bits: the counted mask past `dsa_topk_mask_cut.WHOLE_MAX` keys as `dsa_topk_mask_cut`'s units, which
#: import this module
CUTS = {"DsaTopKMaskCounted_v2": "verity.definitions.registry.dsa_topk_mask_cut"}
