"""Versions of `RMSNormTriton_v1`, `TokenSelect_v1`, the top-p selects and FA2's attention without dead gates: the forms the registered
compiler emits, where no gate may be one no output depends on.  Rows of record keep the v1s their pins were made from.

* `RMSNormTriton_v5`: `RMSNormTriton_v1` with each warp reduced by the adds lane 0's sum depends on (`lane0_butterfly`).  On the GPU
  every lane adds in every round of the xor butterfly, but only lane 0's sum is read: v1 has 129 dead adds a warp.
* `TokenSelect_v3`, `TopPMask_v3` and `GumbelSelectF32GivenGreedy_v2`: the v1 scan stops one lane early, and the last lane computes
  only what is read (its index, or its store).  v1's last iteration made a final best and an i + 1 that nothing reads.
* `GumbelTopPTokenSelectSharedGreedy_v2`: `GumbelTopPTokenSelectSharedGreedy_v1` over `TopPMask_v3` and
  `GumbelSelectF32GivenGreedy_v2`.
* `TokenSelect_v4`: `TokenSelect_v1` from its V-chunks of `CHUNK` logits, each one SHA-512 block of bf16 words: a chunk's
  partial by `TokenSelectChunk_v2` (`b1.TokenSelectChunk_v1` with its last lane's unread index increment gone), reduced with
  logits[0] by `TokenSelectCombine_v2` (`b1.TokenSelectCombine_v1` with each select's branches swapped, as the chunk step's
  are).  Flat, v1 and v3 lower to circuits quadratic in V; this one is linear.
* `TokenSelectGroups_v1{C,G}`: `TokenSelectChunk_v2{G}` over each group of G of its C logits, the groups' bests as one array
  and their indices as another, for `TokenSelectCombine_v2{V,G}`.  A unit's output rows then hold a word or two per group, so
  they pass the 111 bytes C-Flock needs in some row of every unit, where a chunk's partial alone is 2 and 4 bytes.
* `TopPKeepWord_v2{V,S}`: `topp_words.TopPKeepWord_v1{V,S}`'s word gates that its keep bits depend on (`live_replay`).  The keep
  word's builder applies each operation to whole vectors of lanes, and some lanes of a vector are read by nothing after it.
* `TopPMask_v4{V,S}` and `GumbelTopPTokenSelect_v3{V,S}`: `TopPMask_v2{V,S}` and `GumbelTopPTokenSelect_v2{V,S}` over
  `TopPKeepWord_v2` and `GumbelSelectF32_v3`.
* `GumbelSelectF32_v3`: `GumbelSelectF32_v1`'s scan stops one lane early, as `GumbelSelectF32GivenGreedy_v2`'s does.
* `AttentionHead_v18` and `Attention_v18`: FA2's per-iteration `Check_inf` head and attention (`fa2_check_inf`'s v5) with the first
  visited block's zero padding only where it is read.  v5 pads that block's K and V rows to BN with a zero row and fills P's
  k-steps with a zero word; when its key count is a multiple of 16 below BN neither is read, 2 dead gates a head.
* `AttentionHead_v19` and `Attention_v19`: v18 with every step a Call, so a head's key blocks can be units of their own (each
  `AttnBlock_v5` is about 9.2M ANDs at Llama's geometry, a head of 32 blocks past C-Flock's 2^27 block).  v18 writes its first
  visited block and its epilogue in its body; v19 Calls `AttnBlockFirst_v1` (that block's steps and padding, with no running
  state in, which an `AttnBlock_v5{FIRST=True}` Call would be passed and ignore) and `AttnHeadEnd_v1`.
* `AttnBlockFirst_v2{D, NVIS, BN, DOT, VS}`: `AttnBlockFirst_v1` with each key's value as D / VS slices of VS columns, so
  that a unit's v port, cut once an element, reads the value projection's output one slice a row.

Each computes its v1's function word for word, except at V = 1, where v1's zero-length scan returns a wrong carry and these return
index 0 (`tests/program/test_dead_free.py`); the attention versions compute v5's at every binding.
"""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from verity.ml.circuits.defs import Coll, PrimitiveDefinition, SpecializedDefinition, bind, composite
from verity.ml.circuits.intervals import full
from verity.ml.circuits.liveness import Liveness
from verity.ml.circuits.ops.fp32 import F32AddFtz
from verity.ml.circuits.ops.prims import ZERO32, const
from verity.ml.circuits.ops.scalar import BIT, BitAnd, BitNot, F32, F32Eq, I32Eq, SelectF32, SelectI32
from verity.ml.circuits.refs import Affine, Coll as _Coll, Concat, Explicit, Refs, Strided, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import fa2_check_inf as F2, prims as P, ref_prims as RP, sampling as SA, topp_words as TW
from verity.ml.definitions.registry.b1 import NEG_INF_BF16, ZERO16, ArgmaxStep, TokenSelectChunkStep, chunks, f32c
from verity.ml.definitions.registry.prims import BF16, I32

I64 = SA.I64
#: the logits `TokenSelect_v4` reads per chunk: 64 bf16 words, one SHA-512 block (1,024 bits)
CHUNK = 64


def lane0_butterfly(B, vals: list[Coll], masks: tuple[int, ...], add) -> Coll:
    """Lane 0's value after the rounds `v[i] = add(v[i], v[i ^ m])` (a lane whose peer `i ^ m` is past the last keeps its value), each
    round computed on the lanes the later rounds read for lane 0: those an xor of the later masks reaches.  At masks 16, 8, 4, 2, 1
    over 32 lanes, 31 adds of the butterfly's 160, the same adds in the same order as lane 0's."""
    cur = dict(enumerate(vals))
    for k, m in enumerate(masks):
        reads = {0}
        for r in masks[k + 1:]:
            reads |= {i ^ r for i in reads}
        cur = {i: B.call(add, cur[i], cur[i ^ m]) if (i ^ m) in cur else cur[i] for i in sorted(reads) if i in cur}
    return cur[0]


@composite("RMSNormTriton", 5, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormTritonV5(B, S, x, w):
    """`RMSNormTriton_v1{N, EPS}`'s function, each warp's butterfly computing lane 0's adds alone: `out_j = bf16_rne((inv * x_j) *
    w_j)`, `inv = div.full(1, sqrt.approx.ftz(fma(rcp(n), sum, eps)))`, `sum` = chunks of 1024 columns, 128 threads x 8 columns
    (`acc = x1*x1; fma(x0,x0,acc); fma(x2..x7)`), warp butterfly xor 16,8,4,2,1, CTA `(w0 + w2) + (w1 + w3)`, chunks added in order
    from +0."""
    N = S.N
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    chunk_sums = []
    for c0 in range(0, N, 1024):
        warps = []
        for wi in range(4):
            lanes = []
            for ln in range(32):
                base = c0 + 8 * (wi * 32 + ln)
                if base >= N:
                    lanes.append(B.call(ZERO32))
                    continue
                xs = [xf[base + j] for j in range(8)]
                acc = B.call(P.F32Mul, xs[1], xs[1])
                acc = B.call(P.F32Fma, xs[0], xs[0], acc)
                for j in range(2, 8):
                    acc = B.call(P.F32Fma, xs[j], xs[j], acc)
                lanes.append(acc)
            warps.append(lane0_butterfly(B, lanes, (16, 8, 4, 2, 1), P.F32Add))
        chunk_sums.append(B.call(P.F32Add, B.call(P.F32Add, warps[0], warps[2]), B.call(P.F32Add, warps[1], warps[3])))
    total = B.call(ZERO32)
    for cs in chunk_sums:
        total = B.call(P.F32Add, total, cs)
    rcp_n = B.call(P.DivFullRcp, B.call(f32c(float(N))))
    rms = B.call(P.MufuSqrtFtz, B.call(P.F32Fma, rcp_n, total, B.call(f32c(S.EPS))))
    inv = B.call(P.F32Mul, B.call(P.DivFullScaleA, B.call(f32c(1.0)), rms), B.call(P.DivFullRcp, rms))
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    sx = B.batch(P.F32Mul, inv, xf, axes=(None, 0))
    y = B.batch(P.F32Mul, sx, wf, axes=(0, 0))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite("TokenSelect", 3, ["V"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def TokenSelectV3(B, S, logits):
    """`TokenSelect_v1{V}`'s function: argmax over the bf16 logits, lowest index on ties (torch.argmax).  `ArgmaxStep_v1`'s scan to
    logits[V-2], then the last logit's test for its index alone."""
    if S.V == 1:
        return B.call(const(32, 0))
    init = tuple_of(logits[0], B.call(const(32, 0)), B.call(const(32, 1)))
    c = B.scan(ArgmaxStep, init, xs=(logits[1:S.V - 1],))[0] if S.V > 2 else init
    return B.call(P.SelectI32, B.call(P.Bf16GtStrict, logits[S.V - 1], c[0]), c[2], c[1])


@composite("TopPMask", 3, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("top_p", F32), ("splits", I32)), Array(S.V, F32)),
           conformance="exact-model-tested")
def TopPMaskV3(B, S, logits, top_p, splits):
    """`TopPMask_v1{V}`'s function: `TopPMaskStep_v1`'s scan to logits[V-2], then the last lane's store alone, `tl.where(keep, x,
    -inf)`."""
    keep = B.call(SA.topp_mask_word(S.V), logits, top_p, splits)
    i, head = B.call(const(32, 0)), []
    if S.V > 1:
        res = B.scan(bind(SA.TopPMaskStep, V=S.V), i, xs=(logits[0:S.V - 1],), shared=(keep,))
        i, head = res[0], [res[1].refs]
    last = B.call(SelectF32, B.call(SA.bit_at(S.V), keep, i), logits[S.V - 1], B.call(const(32, SA.NEG_INF_BITS)))
    return _Coll(Array(S.V, F32), concat(head + [last.refs]))


@composite("GumbelSelectF32GivenGreedy", 2, ["V"],
           lambda S: ((("logits", Array(S.V, F32)), ("seed", I64), ("pos", I64), ("greedy", BIT)), I32), conformance="exact-model-tested")
def GumbelSelectF32GivenGreedyV2(B, S, logits, seed, pos, greedy):
    """`GumbelSelectF32GivenGreedy_v1{V}`'s function: `GumbelSelectStep_v1`'s scan to logits[V-2], then the last lane's test for its
    index alone (y = x + g(key)[i] when noisy else x; its index iff y > best)."""
    if S.V == 1:
        return B.call(const(32, 0))
    key = B.call(SA.GumbelStreamKey, seed, pos)
    noisy = B.call(BitNot, greedy)
    zero = B.call(const(32, 0))
    y0 = B.call(SelectF32, noisy, B.call(F32AddFtz, logits[0], B.call(SA.GumbelNoiseLane, key, zero)), logits[0])
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    c = B.scan(SA.GumbelSelectStep, init, xs=(logits[1:S.V - 1],), shared=(key, noisy))[0] if S.V > 2 else init
    x = logits[S.V - 1]
    y = B.call(SelectF32, noisy, B.call(F32AddFtz, x, B.call(SA.GumbelNoiseLane, key, c[2])), x)
    return B.call(SelectI32, B.call(RP.F32GtStrict, y, c[0]), c[2], c[1])


@composite("GumbelTopPTokenSelectSharedGreedy", 2, ["V"],
           lambda S: ((("logits", Array(S.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32), ("splits", I32)), I32),
           conformance="exact-model-tested")
def GumbelTopPTokenSelectSharedGreedyV2(B, S, logits, top_p, seed, pos, temp, splits):
    """`GumbelTopPTokenSelectSharedGreedy_v1{V}`'s function and signature, over `TopPMask_v3` and `GumbelSelectF32GivenGreedy_v2`."""
    if S.V == 1:
        return B.call(const(32, 0))
    greedy = B.call(F32Eq, temp, B.call(const(32, 0)))
    scaled = B.call(bind(SA.TemperatureScaleGivenGreedy, V=S.V), logits, temp, greedy)
    masked = B.call(bind(TopPMaskV3, V=S.V), scaled, top_p, splits)
    return B.call(bind(GumbelSelectF32GivenGreedyV2, V=S.V), masked, seed, pos, greedy)


@composite("TokenSelectChunk", 2, ["C"], lambda S: ((("x", Array(S.C, BF16)),), Tuple(BF16, I32)), conformance="model-untested")
def TokenSelectChunkV2(B, S, x):
    """`b1.TokenSelectChunk_v1{C}`'s function: `TokenSelectChunkStep_v1`'s scan to x[C-2] from (-inf, 0, 0), then the last
    logit's test for the best and its index alone."""
    init = tuple_of(B.call(const(16, NEG_INF_BF16)), B.call(const(32, 0)), B.call(const(32, 0)))
    c = B.scan(TokenSelectChunkStep, init, xs=(x[0:S.C - 1],))[0] if S.C > 1 else init
    keep = B.call(BitNot, B.call(P.Bf16GtStrict, x[S.C - 1], c[0]))
    return tuple_of(B.call(P.SelectBf16, keep, c[0], x[S.C - 1]), B.call(SelectI32, keep, c[1], c[2]))


def _combine_type(S):
    n = len(chunks(S.V, S.C))
    return (("x0", BF16), ("best", Array(n, BF16)), ("idx", Array(n, I32))), I32


@composite("TokenSelectCombine", 2, ["V", "C"], _combine_type, conformance="model-untested")
def TokenSelectCombineV2(B, S, x0, best, idx):
    """`b1.TokenSelectCombine_v1{V,C}`'s function with each select's condition negated and its branches swapped, as
    `TokenSelectChunkStep_v1`'s are: a select's bits are its else branch's XOR one AND each, and that branch is chunk c's partial
    here, never the running (best, index), so n chunks lower flat to a circuit linear in n."""
    n = len(chunks(S.V, S.C))
    run, run_i = x0, B.call(const(32, 0))
    for c in range(n):
        keep = B.call(BitNot, B.call(P.Bf16GtStrict, best[c], run))
        at = idx[c] if c == 0 else B.call(P.I32Add, idx[c], B.call(const(32, c * S.C)))
        run_i = B.call(SelectI32, keep, run_i, at)
        if c < n - 1:
            run = B.call(P.SelectBf16, keep, run, best[c])
    return run_i


def _groups_type(S):
    if not 1 <= S.G <= S.C:
        raise ValueError(f"TokenSelectGroups_v1: groups of G = {S.G} of C = {S.C} logits (1 <= G <= C)")
    n = len(chunks(S.C, S.G))
    return (("x", Array(S.C, BF16)),), Tuple(Array(n, BF16), Array(n, I32))


@composite("TokenSelectGroups", 1, ["C", "G"], _groups_type, conformance="model-untested")
def TokenSelectGroupsV1(B, S, x):
    """`TokenSelectChunk_v2{G}` over each group of G consecutive logits of the C (`chunks(C, G)`: the last group the rest),
    each group's best a word of ``best`` and its index in the group a word of ``idx``: `TokenSelect_v4`'s chunks of G, C
    logits of them in one unit, for `TokenSelectCombine_v2{V,G}` to read a unit's groups as one row of bests and one of
    indices."""
    full = S.C // S.G
    best, idx = [], []
    if full:
        parts = B.batch(bind(TokenSelectChunkV2, C=S.G), x[0:full * S.G].reshape(Array(full, Array(S.G, BF16))), axes=(0,))
        best.append(strided_view(parts, 0, 1, 2, 0, full))
        idx.append(strided_view(parts, 1, 1, 2, 0, full))
    if S.C % S.G:
        last = B.call(bind(TokenSelectChunkV2, C=S.C % S.G), x[full * S.G:S.C])
        best.append(last[0].refs)
        idx.append(last[1].refs)
    n = len(chunks(S.C, S.G))
    return tuple_of(_Coll(Array(n, BF16), concat(best)), _Coll(Array(n, I32), concat(idx)))


@composite("TokenSelect", 4, ["V"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def TokenSelectV4(B, S, logits):
    """`TokenSelect_v1{V}`'s function from its V-chunks of `CHUNK` logits (the whole row below that): each chunk's partial by
    `TokenSelectChunk_v2`, then `TokenSelectCombine_v2` from logits[0] and the partials."""
    if S.V == 1:
        return B.call(const(32, 0))
    C = min(CHUNK, S.V)
    parts = [B.call(bind(TokenSelectChunkV2, C=hi - lo), logits[lo:hi]) for lo, hi in chunks(S.V, C)]
    best = _Coll(Array(len(parts), BF16), concat([p[0].refs for p in parts]))
    idx = _Coll(Array(len(parts), I32), concat([p[1].refs for p in parts]))
    return B.call(bind(TokenSelectCombineV2, V=S.V, C=C), logits[0], best, idx)


def _codes(r: Refs) -> np.ndarray:
    """Each reference of `r` as `topp_words`' int64 code: `idx << 32 | leaf` for a node's leaf, `-1 -` that for a parameter's."""
    if isinstance(r, Concat):
        return np.concatenate([_codes(p) for p in r.parts]) if r.parts else np.zeros(0, np.int64)
    if isinstance(r, Explicit):
        return np.array([TW._code(x) for x in r.items], np.int64)
    j = np.arange(r.count, dtype=np.int64)
    if isinstance(r, Affine):
        leaf = r.base + j
    elif isinstance(r, Strided):
        leaf = r.base + (j // r.inner) * r.outer_stride + (j % r.inner) * r.inner_stride
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")
    c = (np.int64(r.idx) << 32) | leaf
    return -1 - c if r.space == "p" else c


def live_replay(B, sd: SpecializedDefinition, args: Sequence[Coll]) -> Coll:
    """`sd`'s body emitted into `B` over `args` with only the gates its return depends on: a node nothing live reads is left out,
    and a batch keeps the members something live reads (`liveness.Liveness`, transitive, circuit-check's dead gates).  The body must
    be primitive calls and batches of primitives (`topp_words.IREngine`'s), so each kept gate is the same primitive on the same
    operands and the return is `sd`'s, word for word."""
    body = sd.body
    lv = Liveness(transitive=True).analyse(sd, full(body.ret_type.leaves))
    if lv.approximate:
        raise ValueError(f"{sd.name}: a batch's live members were over-approximated, so its replay would keep dead gates")
    params = [_codes(a.refs) for a in args]
    new: dict[int, np.ndarray] = {}

    def moved(codes: np.ndarray) -> Refs:
        out = np.empty_like(codes)
        node = codes >= 0
        key = np.where(node, codes, -1 - codes)
        idx, leaf = key >> 32, key & TW.M32
        for i in np.unique(idx[~node]).tolist():
            at = ~node & (idx == i)
            out[at] = params[i][leaf[at]]
        for i in np.unique(idx[node]).tolist():
            at = node & (idx == i)
            out[at] = new[i][leaf[at]]
        assert (out != _GONE).all(), f"{sd.name}: a live gate reads one left out"
        return TW._refs(out)

    for k, n in enumerate(body.nodes):
        live = lv.node_live[k]
        if not live:
            continue
        if not isinstance(n.fn, PrimitiveDefinition) or n.form not in ("primitive", "batch"):
            raise ValueError(f"{sd.name}: node {k} is a {n.form} of {n.fn.name}; live_replay emits primitives and their batches")
        if n.form == "primitive":
            new[k] = _codes(B.call(n.fn, *(Coll(a.type, moved(_codes(a.refs))) for a in n.args)).refs)
            continue
        R = n.fn.ret.leaves
        m = np.unique(np.concatenate([np.arange(lo // R, (hi - 1) // R + 1) for lo, hi in live]))
        colls = []
        for a, ax in zip(n.args, n.axes):
            if ax is None:
                colls.append(Coll(a.type, moved(_codes(a.refs))))
                continue
            assert ax == 0, f"{sd.name}: node {k} batches along axis {ax}"
            L = a.type.elem.leaves
            colls.append(Coll(Array(len(m), a.type.elem), moved(_codes(a.refs)[(m[:, None] * L + np.arange(L)).reshape(-1)])))
        out = np.full(n.n * R, _GONE, np.int64)
        out[(m[:, None] * R + np.arange(R)).reshape(-1)] = _codes(B.batch(n.fn, *colls, axes=n.axes, n=len(m)).refs)
        new[k] = out
    return Coll(body.ret_type, moved(_codes(body.ret)))


#: a leaf `live_replay` left out (no code `_codes` makes: a node's codes are >= 0, a parameter's > -2^63)
_GONE = np.iinfo(np.int64).min


@composite("TopPKeepWord", 2, ["V", "S"], lambda st: ((("logits", Array(st.V, F32)), ("top_p", F32)), Array(st.V, BIT)),
           conformance="exact-model-tested")
def TopPKeepWordV2(B, st, logits, top_p):
    """`TopPKeepWord_v1{V,S}`'s keep bits from the word gates they depend on: v1's body by `live_replay`."""
    return live_replay(B, bind(TW.TopPKeepWord, V=st.V, S=st.S), (logits, top_p))


@composite("TopPMask", 4, ["V", "S"],
           lambda st: ((("logits", Array(st.V, F32)), ("top_p", F32), ("splits", I32)), Array(st.V, F32)),
           conformance="exact-model-tested")
def TopPMaskV4(B, st, logits, top_p, splits):
    """`TopPMask_v2{V,S}` over `TopPKeepWord_v2{V,S}`: each lane kept or set to -inf by its keep bit, and no lane kept unless
    `splits == S`."""
    keep = B.call(bind(TopPKeepWordV2, V=st.V, S=st.S), logits, top_p)
    ok = B.call(I32Eq, splits, B.call(const(32, int(st.S))))
    keep = B.batch(BitAnd, keep, ok, axes=(0, None))
    return B.batch(SelectF32, keep, logits, B.call(const(32, SA.NEG_INF_BITS)), axes=(0, 0, None))


@composite("GumbelSelectF32", 3, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("seed", I64), ("pos", I64), ("temp", F32)), I32),
           conformance="exact-model-tested")
def GumbelSelectF32V3(B, S, logits, seed, pos, temp):
    """`GumbelSelectF32_v1{V}`'s function: `GumbelSelectStep_v1`'s scan to logits[V-2], then the last lane's test for its index
    alone (y = x + g(key)[i] when temp != 0 else x; its index iff y > best)."""
    if S.V == 1:
        return B.call(const(32, 0))
    key = B.call(SA.GumbelStreamKey, seed, pos)
    noisy = B.call(BitNot, B.call(F32Eq, temp, B.call(const(32, 0))))
    zero = B.call(const(32, 0))
    y0 = B.call(SelectF32, noisy, B.call(F32AddFtz, logits[0], B.call(SA.GumbelNoiseLane, key, zero)), logits[0])
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    c = B.scan(SA.GumbelSelectStep, init, xs=(logits[1:S.V - 1],), shared=(key, noisy))[0] if S.V > 2 else init
    x = logits[S.V - 1]
    y = B.call(SelectF32, noisy, B.call(F32AddFtz, x, B.call(SA.GumbelNoiseLane, key, c[2])), x)
    return B.call(SelectI32, B.call(RP.F32GtStrict, y, c[0]), c[2], c[1])


@composite("GumbelTopPTokenSelect", 3, ["V", "S"],
           lambda st: ((("logits", Array(st.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32), ("splits", I32)), I32),
           conformance="exact-model-tested")
def GumbelTopPTokenSelectV3(B, st, logits, top_p, seed, pos, temp, splits):
    """`GumbelTopPTokenSelect_v2{V,S}`'s function and operands, over `TopPMask_v4{V,S}` and `GumbelSelectF32_v3{V}`."""
    if st.V == 1:
        return B.call(const(32, 0))
    scaled = B.call(bind(SA.TemperatureScale, V=st.V), logits, temp)
    masked = B.call(bind(TopPMaskV4, V=st.V, S=st.S), scaled, top_p, splits)
    return B.call(bind(GumbelSelectF32V3, V=st.V), masked, seed, pos, temp)


@composite("AttentionHead", 18, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="not-yet-established: AttentionHead_v5's function")
def AttentionHeadV18(B, S, q, k, v):
    """`AttentionHead_v5{T, D, BN, DOT, INV, MASKED_FROM}` with the first visited block's padding only where read: its V rows to the
    last k-step it runs, a zero word in P only for that step, and no K rows (it reads none past its key count)."""
    return F2._check_inf_head(B, S, q, k, v, F2.AttnBlockV5, (), read_pads_only=True)


@composite("Attention", 18, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established: Attention_v5's function")
def AttentionV18(B, S, q, kc, vc):
    """`Attention_v5{T, NH, KVH, D, BN, DOT, INV, MASKED_FROM}` over `AttentionHead_v18`."""
    return F2._check_inf_attention(B, S, q, kc, vc, AttentionHeadV18, ())


@composite("AttnBlockFirst", 1, ["D", "NVIS", "BN", "DOT"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.NVIS, Array(S.D, BF16))), ("v", Array(S.NVIS, Array(S.D, BF16)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="not-yet-established: AttentionHead_v18's first visited block")
def AttnBlockFirstV1(B, S, q, k, v):
    """`AttentionHead_v18`'s first visited key block, NVIS = 1 ... BN keys: `AttnBlock_v5{D, NVIS, FIRST=True, BN, DOT, CHECK=True}`'s
    steps with no running state in (the first block reads none) and v18's padding (V rows zero past NVIS to the last k-step it
    runs, no K rows).  Returns the running state (max, lane sums, O) the next block reads."""
    return _first_block(B, S, q, k, v, "AttnBlockFirst_v1")


def _first_block(B, S, q, k, v, name):
    D, NVIS, BN = S.D, S.NVIS, S.BN
    if not 1 <= NVIS <= BN:
        raise ValueError(f"{name}: NVIS {NVIS} outside 1 ... BN = {BN}")
    if NVIS < BN and NVIS % 16:  # the last k-step's V rows past NVIS meet P's zero words
        zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
        v = _Coll(Array(NVIS + -NVIS % 16, Array(D, BF16)), concat([v.refs] + [zero_row.refs] * (-NVIS % 16)))
    first = SimpleNamespace(D=D, NVIS=NVIS, FIRST=True, BN=BN, DOT=S.DOT, CHECK=True)
    return F2._check_inf_block(B, first, q, k, v, None, None, None, None, read_pads_only=True)


def _slice_refuse(b: dict) -> str | None:
    D, vs = b.get("D"), b.get("VS")
    if vs is not None and not (type(vs) is int and vs >= 1 and (D is None or D % vs == 0)):
        return f"VS must be a positive divisor of D (got D={D!r}, VS={vs!r})"
    return None


@composite("AttnBlockFirst", 2, ["D", "NVIS", "BN", "DOT", "VS"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.NVIS, Array(S.D, BF16))),
                       ("v", Array(S.NVIS * S.D // S.VS, Array(S.VS, BF16)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="not-yet-established: AttnBlockFirst_v1's function", refuse=_slice_refuse)
def AttnBlockFirstV2(B, S, q, k, v):
    """`AttnBlockFirst_v1{D, NVIS, BN, DOT}` with each key's value as D / VS slices of VS columns: v's element
    ``t·D/VS + s`` is key t's columns [s·VS, (s+1)·VS), so a cut of v once an element reads one slice a row (a slice of the
    value projection's output), where a cut of v1's v holds at least a whole key a row.  Its leaves are v1's v in the same
    order, so the body is v1's over them, node for node."""
    return _first_block(B, S, q, k, _Coll(Array(S.NVIS, Array(S.D, BF16)), v.refs), "AttnBlockFirst_v2")


@composite("AttnHeadEnd", 1, ["D", "INV"],
           lambda S: ((("l", Array(4, F32)), ("o", Array(S.D, F32))), Array(S.D, BF16)),
           conformance="not-yet-established: AttentionHead_v18's epilogue")
def AttnHeadEndV1(B, S, l, o):
    """`AttentionHead_v18`'s epilogue on the last block's lane sums and O: `sum = (l0 + l2) + (l1 + l3)`, `inv = INV(sum)`,
    `out = bf16(o * inv)`."""
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, l[0], l[2]), B.call(P.F32AddFtz, l[1], l[3]))
    inv = B.call(S.INV, total)
    return B.batch(P.F2fpBf16, B.batch(P.F32MulFtz, o, inv, axes=(0, None)), axes=(0,))


@composite("AttentionHead", 19, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="not-yet-established: AttentionHead_v18's function")
def AttentionHeadV19(B, S, q, k, v):
    """`AttentionHead_v18{T, D, BN, DOT, INV, MASKED_FROM}` with every step a Call, so each key block can be a unit of its own: the
    first visited block `AttnBlockFirst_v1{D, NVIS, BN, DOT}` (keys (nb - 1) BN ... T - 1), block b below it
    `AttnBlock_v5{D, BN, FIRST=False, BN, DOT, CHECK = b >= MASKED_FROM}`, and the epilogue `AttnHeadEnd_v1{D, INV}`."""
    T, D, BN, DOT, MF = S.T, S.D, S.BN, S.DOT, S.MASKED_FROM
    nb = (T + BN - 1) // BN
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"AttentionHead_v19: MASKED_FROM {MF} outside blocks 0..{nb - 1} "
                         "(the first block is always a Check_inf iteration)")
    lo = (nb - 1) * BN
    st = B.call(bind(AttnBlockFirstV1, D=D, NVIS=T - lo, BN=BN, DOT=DOT), q, k[lo:T], v[lo:T])
    for b in range(nb - 2, -1, -1):
        block = bind(F2.AttnBlockV5, D=D, NVIS=BN, FIRST=False, BN=BN, DOT=DOT, CHECK=b >= MF)
        st = B.call(block, q, k[b * BN:(b + 1) * BN], v[b * BN:(b + 1) * BN], st[0], st[1], st[2])
    return B.call(bind(AttnHeadEndV1, D=D, INV=S.INV), st[1], st[2])


@composite("Attention", 19, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established: Attention_v18's function")
def AttentionV19(B, S, q, kc, vc):
    """`Attention_v18{T, NH, KVH, D, BN, DOT, INV, MASKED_FROM}` over `AttentionHead_v19`."""
    return F2._check_inf_attention(B, S, q, kc, vc, AttentionHeadV19, ())


__all__ = ["AttentionHeadV18", "AttentionHeadV19", "AttentionV18", "AttentionV19", "AttnBlockFirstV1", "AttnBlockFirstV2",
           "AttnHeadEndV1", "CHUNK",
           "GumbelSelectF32GivenGreedyV2", "GumbelSelectF32V3", "GumbelTopPTokenSelectSharedGreedyV2", "GumbelTopPTokenSelectV3",
           "RMSNormTritonV5", "TokenSelectChunkV2", "TokenSelectCombineV2", "TokenSelectGroupsV1", "TokenSelectV3", "TokenSelectV4",
           "TopPKeepWordV2", "TopPMaskV3", "TopPMaskV4", "lane0_butterfly", "live_replay"]
