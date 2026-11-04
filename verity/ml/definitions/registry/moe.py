"""The vLLM fused-MoE block (`vllm.moe_forward` -> `FusedMoEKernel(MoEPrepareAndFinalizeNoEP, TritonExperts)`, pinned
d9105ea80) as REGISTERED Definitions -- promoted from `registry.quarantine.ov_moe` with the
same names / versions, so the specialization ids the accepted OLMoE eager records resolved to are unchanged.

The block is: router top-k (`_moe_C.topk_softmax`) -> block schedule (`_moe_C.moe_align_block_size`, M*TOPK*4 > E; else the naive
assignment) -> `fused_moe_kernel(w13, MUL_ROUTED_WEIGHT=False)` -> `_C.silu_and_mul` -> `fused_moe_kernel(w2, MUL_ROUTED_WEIGHT=True)`
-> `_moe_C.moe_sum`.  Every numerical statement is per (token t, slot k) PAIR p = t*TOPK + k: the fused kernel's grouping of pairs into
expert blocks (sorted_token_ids / expert_ids / num_tokens_post_padded) is a SCHEDULE that never enters the arithmetic of an output
word -- each output word of the pair-major intermediates is one ascending-k chain of the same tensor-core steps as the dense GEMM
(`DotBf16`), then one f32 -> bf16 conversion (with the routing weight multiplied in before the conversion on the down projection).
The tile constants (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps/stages) vLLM picks per M (`try_get_optimal_moe_config`) schedule which
CTA owns which pair rows and how the k-loop is chunked; a chunk of BLOCK_SIZE_K is BLOCK_SIZE_K/16 k16 steps into the SAME carried
accumulator, so the per-coordinate chain is the tile-independent one -- checked, not assumed: these bodies were evaluated against the
real kernels at several M (different tilings, naive vs sorted schedule, different routing patterns).

Representation (the one that fits the existing operand kinds; the precedent is `Embedding_v1`): the expert index `e` is an ordinary I32
VALUE operand of the pair body -- produced by `MoeRouterTopK_v1` from the router logits, never a structural input or an observed
constant -- and the selection of the expert's weight row happens INSIDE the body as a wide-fan-in gather over the expert axis
(`GatherBf16x{E}` per weight element).  The Program's dataflow is then fixed by shapes alone (M, TOPK, E, H, I); the data-dependent part
is arithmetic on a value the Program itself produced, exactly as the token id is for the embedding.

The router's exponential is `NvExpf_v1`: the CUDA math-library `expf` sequence exactly as nvcc emits it for sm_89 (transcribed from the
PTX of a kernel calling `expf`, CUDA 12.x headers; `cvt.sat` / `fma.rm` / `ex2.approx.ftz` / exponent-field scaling) -- bit-exact against
`_moe_C::topk_softmax`'s weights (192 of 192 words; the cross-check: every routed weight of every pattern).

GLM-5.3's router (DeepSeek-V3's `noaux_tc`: sigmoid scores, `e_score_correction_bias` for selection only, one group, top-8 of 256,
renormalized, x2.5) is `MoeRouterSigmoidBias_v1`: vLLM 0.28.1rc1.dev472's `GroupedTopKRouter` -> `fused_grouped_topk` ->
`_moe_C.grouped_topk` -> `single_group_topk_{block,warp}_kernel<float, float, int, SIGMOID, 256, 8>` on the f32 logits of
`GateLinear` (bf16 weight and input, f32 output), its weights and ids returned as the kernel writes them.
"""
from __future__ import annotations

from fractions import Fraction

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.scalar import F32_NAN_PTX, BitAnd, BitNot, BitOr, F32BitsShl23, F32Eq, F32Fabs, F32Fmaxf, F32IsFinite, F32Neg, F32Sat
from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import ref_prims as R
from verity.ml.definitions.registry.prims import BF16, BIT, F32, I32, _decode, f32_of, bits_of

SEMANTIC_ID = "moe/FusedMoE@v1"

# ---------------------------------------------------------------------------------------------------------
# scalar primitives the router needs and the registry lacks
# ---------------------------------------------------------------------------------------------------------

F32GtStrict = R.F32GtStrict   # a > b on f32 values (NaN false): greedy argmax, first index wins ties
F32Sub = R.F32Sub             # RN(a - b), no ftz


def _round_f32_toward_neg_inf(x: Fraction) -> int:
    """The largest f32 <= the exact rational x (RM / floor rounding); |x| beyond the f32 range -> -inf or +MAX."""
    if x == 0:
        return 0
    neg = x < 0
    a = -x if neg else x
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if (Fraction(1 << e, 1) if e >= 0 else Fraction(1, 1 << -e)) > a:
        e -= 1
    if e < -126:
        e = -126
    scale = e - 23
    q = a * (1 << -scale) if scale < 0 else a / (1 << scale)
    n = q.numerator // q.denominator            # truncated significand (toward zero in magnitude)
    inexact = q != n
    if neg and inexact:
        n += 1                                   # toward -inf: magnitude up for negative values
    if n >= 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        return 0xFF800000 if neg else 0x7F7FFFFF
    s = 0x80000000 if neg else 0
    if n < 1 << 23:                              # subnormal (e == -126)
        return s | n
    return s | ((e + 127) << 23) | (n & 0x7FFFFF)


@primitive("F32FmaRm", 1, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32FmaRm(a: int, b: int, c: int) -> int:
    """`fma.rm.f32`: a * b + c with ONE rounding toward -infinity (exact rational arithmetic).  Non-finite operands follow
    `F32Fma_v1` (NaN propagation, inf * 0 = NaN, inf - inf = NaN); an exact zero sum is +0 unless both addends are -0."""
    ca, va = _decode(a)
    cb, vb = _decode(b)
    cc, vc = _decode(c)
    if 2 in (ca, cb, cc):
        return P.F32_NAN
    if ca == 0 and cb == 0:
        prod = va * vb
        if cc == 0:
            s = prod + vc
            if s == 0:
                prod_neg = (prod == 0) and (((a >> 31) ^ (b >> 31)) & 1) == 1
                return 0x80000000 if (prod == 0 and vc == 0 and prod_neg and (c >> 31) & 1) else 0
            return _round_f32_toward_neg_inf(s)
        return 0x7F800000 if cc == 1 else 0xFF800000
    if (ca == 0 and va == 0) or (cb == 0 and vb == 0):
        return P.F32_NAN
    sign = ((a >> 31) ^ (b >> 31)) & 1
    if cc == 0 or (cc == -1) == bool(sign):
        return 0xFF800000 if sign else 0x7F800000
    return P.F32_NAN


@primitive("F32FmaRm", 2, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32FmaRmV2(a: int, b: int, c: int) -> int:
    """`F32FmaRm_v1` with every NaN result `F32_NAN_PTX`, as `F32Fma_v2`."""
    return P.f32_gpu_nan(F32FmaRm.evaluate(a, b, c))


#: The constants of nvcc's `expf` (sm_89, CUDA 12.x math headers; PTX of `y = expf(x)`):
#:   fma.rn  t  = x * 0f3BBB989D + 0.5          (0x3BBB989D = log2(e) / 252)
#:   cvt.sat s  = clamp(t, 0, 1)                (NaN -> 0)
#:   fma.rm  jb = s * 252 + 12582913            (round toward -inf: jb = 12582913 + floor(126 + x*log2e), an integer-valued f32
#:                                              whose low 9 bits are the biased exponent 127 + floor(x*log2e), saturated to [1, 253])
#:   add.rn  j  = jb + (-12583039)              (= floor(x*log2e) as an f32, exact; -126 <= j <= 126)
#:   fma.rn  f  = x * 0f3FB8AA3B + (-j)         (log2(e) high part 1.44269502)
#:   fma.rn  f  = x * 0f32A57060 + f            (log2(e) low part 1.9259630e-8)
#:   ex2.approx.ftz.f32 e = 2^f                 (`MufuEx2Ftz_v1`, the measured sm_89 MUFU.EX2)
#:   mul.rn  y  = e * f32(bits(jb) << 23)       (= e * 2^j; `mul.f32` without .ftz: a subnormal product is kept)
C_LOG2E_OVER_252 = b1.f32c(f32_of(0x3BBB989D))
C_HALF = b1.f32c(0.5)
C_252 = b1.f32c(252.0)
C_MAGIC = b1.f32c(12582913.0)
C_NEG_MAGIC_BIAS = b1.f32c(-12583039.0)
C_LOG2E_HI = b1.f32c(f32_of(0x3FB8AA3B))
C_LOG2E_LO = b1.f32c(f32_of(0x32A57060))


@composite("NvExpf", 1, [], lambda S: ((("x", F32),), F32), conformance="not-yet-established")
def NvExpf(B, S, x):
    """CUDA math-library `expf(x)` (nvcc, sm_89, no fast-math) as the SASS computes it: saturated range reduction to
    `j = floor(x*log2e)` through the `fma.rm` magic-number trick, `f = x*log2e - j` in two fma steps (hi/lo split of log2e),
    `MUFU.EX2(f)` and an exact scaling by `2^j` built from the exponent field.  Total: x = NaN -> NaN, +inf -> +inf,
    -inf / x < ~-103.97 -> +0 (e flushes / underflows), x > ~88.72 -> +inf.  Bit-exact against `torch.exp` (libdevice
    `__nv_expf`) and `_moe_C::topk_softmax`'s weights on the recorded router cases (probe_ov_moe_router_expf.py)."""
    t = B.call(P.F32Fma, x, B.call(C_LOG2E_OVER_252), B.call(C_HALF))
    s = B.call(F32Sat, t)
    jb = B.call(F32FmaRm, s, B.call(C_252), B.call(C_MAGIC))
    j = B.call(P.F32Add, jb, B.call(C_NEG_MAGIC_BIAS))
    f = B.call(P.F32Fma, x, B.call(C_LOG2E_HI), B.call(F32Neg, j))
    f = B.call(P.F32Fma, x, B.call(C_LOG2E_LO), f)
    e = B.call(P.MufuEx2Ftz, f)
    return B.call(P.F32Mul, e, B.call(F32BitsShl23, jb))


ONE32 = b1.f32c(1.0)
NEG_10000 = b1.f32c(-10000.0)
PTX_NAN = b1.const(32, F32_NAN_PTX)


def _butterfly(B, op, vals: list[Coll]) -> list[Coll]:
    """Warp butterfly `v[t] = op(v[t], v[t ^ mask])` for mask = T/2 .. 1 (xor 4, 2, 1 at T=8). Each stage pairs threads
    that already hold identical values, so after the last stage every lane holds the same expression; lane 0 is returned
    as the row value."""
    T = len(vals)
    mask = T // 2
    while mask >= 1:
        vals = [B.call(op, vals[t], vals[t ^ mask]) for t in range(T)]
        mask //= 2
    return vals


# ---------------------------------------------------------------------------------------------------------
# router: softmax over E logits in f32, k rounds of argmax (`topkGating<VPT, E, ...>`)
# ---------------------------------------------------------------------------------------------------------


@composite("RouterArgmaxStep", 1, [], lambda S: ((("carry", Tuple(F32, I32, I32)), ("x", F32)), Tuple(Tuple(F32, I32, I32), I32)),
           conformance="model-untested")
def RouterArgmaxStep(B, S, carry, x):
    """Greedy argmax step over (best, best_index, i) on f32 probabilities: take (x, i) iff x > best (strict; the lowest
    expert wins ties — the kernel's in-thread strict `>` plus its cross-thread `(other > max) || (other == max &&
    other_expert < expert)` rule compose to exactly this global order), then i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    gt = B.call(F32GtStrict, x, best)
    nb = B.call(P.SelectF32, gt, x, best)
    ni = B.call(P.SelectI32, gt, i, best_i)
    nxt = B.call(P.I32Add, i, B.call(b1.const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("MoeRouterTopK", 1, ["E", "TOPK", "VPT"],
           lambda S: ((("logits", Array(S.E, BF16)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
           conformance="not-yet-established")
def MoeRouterTopK(B, S, logits):
    """`topk_softmax(renormalize=False)` for one token row: f32 softmax over the E bf16 logits with the kernel's reduction
    geometry (E/VPT threads x VPT values: thread-local max / sum in ascending column order, warp butterfly xor T/2..1),
    `p_i = expf(x_i - max) * (1.f / sum)` with `expf` = `NvExpf_v1` (the CUDA math-library sequence) and `1.f / sum` =
    `rcp.rn.f32` (IEEE, `F32Div_v1`), NaN/Inf -> 0, then TOPK rounds of lowest-index argmax with the winner masked to
    -10000.f.  Returns (topk_weights = the raw probabilities, topk_ids)."""
    p = _router_probs(B, S.E, S.VPT, logits)
    neg_big = B.call(NEG_10000)
    consts = [B.call(b1.const(32, i)) for i in range(S.E)]
    weights, ids = [], []
    for _ in range(S.TOPK):
        init = tuple_of(p[0], consts[0], consts[1])
        res = B.scan(RouterArgmaxStep, init, xs=(_array(p[1:]),))
        best, best_i = res[0][0], res[0][1]
        weights.append(best)
        ids.append(best_i)
        p = [B.call(P.SelectF32, B.call(P.I32Eq, consts[i], best_i), neg_big, p[i]) for i in range(S.E)]
    return tuple_of(_array(weights), _array(ids))


def _router_probs(B, E: int, VPT: int, logits: Coll, *, interior: bool = False) -> list[Coll]:
    """`topk_softmax`'s probabilities for one token row, in the kernel's reduction geometry (`MoeRouterTopK_v1`'s docstring).

    `interior`: the values the probabilities are computed from are the kernel's word for word, as a construction committing them needs
    (`MoeRouterProbs`): the row max is `fmaxf` (`F32Fmaxf_v1`: the kernel's `max(float, float)` without fast math keeps subnormals, a
    NaN operand yields the other, two NaNs give 0x7FFFFFFF), and an exponential or the reciprocal that is NaN is the GPU's one NaN word
    0x7FFFFFFF (`_ptx_nan`).  Off (`MoeRouterTopK_v1`, the record), the max is `F32Max_v1`; the probabilities are the same either way."""
    if E % VPT:
        raise ValueError("E must be a multiple of VPT")
    T = E // VPT
    fmax = F32Fmaxf if interior else P.F32Max
    x = [B.call(P.Bf16ToF32, logits[i]) for i in range(E)]
    tmax = []
    for t in range(T):
        m = x[t * VPT]
        for j in range(1, VPT):
            m = B.call(fmax, m, x[t * VPT + j])
        tmax.append(m)
    rmax = _butterfly(B, fmax, tmax)[0]
    ex = [B.call(NvExpf, B.call(F32Sub, x[i], rmax)) for i in range(E)]
    if interior:
        nan = B.call(PTX_NAN)
        ex = [_ptx_nan(B, e, nan) for e in ex]
    tsum = []
    for t in range(T):
        s = B.call(b1.ZERO32)
        for j in range(VPT):
            s = B.call(P.F32Add, s, ex[t * VPT + j])
        tsum.append(s)
    rsum = _butterfly(B, P.F32Add, tsum)[0]
    rcp = B.call(P.F32Div, B.call(ONE32), rsum)
    if interior:
        rcp = _ptx_nan(B, rcp, nan)
    zero = B.call(b1.ZERO32)
    p = []
    for i in range(E):
        q = B.call(P.F32Mul, ex[i], rcp)
        p.append(B.call(P.SelectF32, B.call(F32IsFinite, q), q, zero))
    return p


def _ptx_nan(B, v: Coll, nan: Coll) -> Coll:
    """`v`, or the GPU's canonical NaN word when `v` is a NaN: FP32 arithmetic on sm_80+ returns 0x7FFFFFFF for every NaN result, where
    `F32Add_v1` / `F32Mul_v1` / `F32Div_v1` / `F32Sub_v1` keep a NaN operand's payload or give 0x7FC00000 / 0xFFC00000."""
    return B.call(P.SelectF32, B.call(F32Eq, v, v), v, nan)


def _array(parts: list[Coll]) -> Coll:
    return array_of(list(parts))


@composite("MoeRouterTopKNorm", 1, ["E", "TOPK", "VPT"],
           lambda S: ((("logits", Array(S.E, BF16)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
           conformance="not-yet-established")
def MoeRouterTopKNorm(B, S, logits):
    """`topk_softmax(renormalize=True)` for one token row (Qwen3-MoE / Mixtral `norm_topk_prob`): the same kernel as
    `MoeRouterTopK_v1` (same softmax geometry, same TOPK argmax rounds), then the lead thread's epilogue of
    `topkGatingSoftmax` (vLLM d9105ea80 csrc/libtorch_stable/moe/topk_softmax_kernels.cu, compiled WITHOUT --use_fast_math):
    `selected_sum` = the TOPK selected probabilities added in ascending slot order from 0.f, `denom = selected_sum > 0.f ?
    selected_sum : 1.f`, `scale = 1.f / denom` (one `div.rn.f32`, `F32Div_v1`; routed_scaling_factor 1.0 is the identity),
    `w_k = w_k * scale` (one `F32Mul_v1` per slot).  Returns (normalised weights, the SAME ids)."""
    res = B.call(bind(MoeRouterTopK, E=S.E, TOPK=S.TOPK, VPT=S.VPT), logits)
    w, ids = res[0], res[1]
    acc = B.call(b1.ZERO32)
    for k in range(S.TOPK):
        acc = B.call(P.F32Add, acc, w[k])
    one = B.call(ONE32)
    denom = B.call(P.SelectF32, B.call(F32GtStrict, acc, B.call(b1.ZERO32)), acc, one)
    scale = B.call(P.F32Div, one, denom)
    return tuple_of(_array([B.call(P.F32Mul, w[k], scale) for k in range(S.TOPK)]), ids)


# ---------------------------------------------------------------------------------------------------------
# the same router in its selection order (`TargetProfile.moe_construction = "indexed-read-ordered"`, the default): the softmax once, then
# round k takes the expert after round k-1's (p, id) in the kernel's total order -- every value computed once, no masked vector carried
# ---------------------------------------------------------------------------------------------------------


@composite("MoeRouterProbs", 1, ["E", "VPT"], lambda S: ((("logits", Array(S.E, BF16)),), Array(S.E, F32)),
           conformance="not-yet-established")
def MoeRouterProbs(B, S, logits):
    """`MoeRouterTopK_v1`'s probabilities, computed from the kernel's own interior words (`_router_probs(interior=True)`: the max, the
    exponentials and the reciprocal the ordered router commits are `topkGating`'s bit for bit); +0 or positive, never NaN."""
    return _array(_router_probs(B, S.E, S.VPT, logits, interior=True))


@composite("MoeRouterFirst", 1, ["E"], lambda S: ((("p", Array(S.E, F32)),), I32), conformance="not-yet-established")
def MoeRouterFirst(B, S, p):
    """Round 0 of `MoeRouterTopK_v1`: the strict first-max argmax scan over the probabilities."""
    c0, c1 = B.call(b1.const(32, 0)), B.call(b1.const(32, 1))
    return B.scan(RouterArgmaxStep, tuple_of(p[0], c0, c1), xs=(_array([p[i] for i in range(1, S.E)]),))[0][1]


@composite("MoeRouterNext", 1, ["E"], lambda S: ((("p", Array(S.E, F32)), ("w", F32), ("e", I32)), I32), conformance="not-yet-established")
def MoeRouterNext(B, S, p, w, e):
    """Round k of `MoeRouterTopK_v1` from round k-1's selection (w = p[e], e): the experts after it in the kernel's total order
    (`p_i < w`, or `p_i == w` and `i > e`) keep their probability, the others become -10000.f, then the same first-max scan.  The
    experts chosen so far are exactly those not after (e, w) -- each round took the first of the order among the rest -- so this is the
    kernel's masked vector at round k and the same argmax."""
    neg_big = B.call(NEG_10000)
    q = []
    for i in range(S.E):
        ci = B.call(b1.const(32, i))
        after = B.call(BitOr, B.call(F32GtStrict, w, p[i]), B.call(BitAnd, B.call(F32Eq, p[i], w), B.call(BitNot, B.call(P.I32Le, ci, e))))
        q.append(B.call(P.SelectF32, after, p[i], neg_big))
    return B.scan(RouterArgmaxStep, tuple_of(q[0], B.call(b1.const(32, 0)), B.call(b1.const(32, 1))), xs=(_array(q[1:]),))[0][1]


@composite("MoeRouterPick", 1, ["E"], lambda S: ((("p", Array(S.E, F32)), ("e", I32)), F32), conformance="not-yet-established")
def MoeRouterPick(B, S, p, e):
    """`p[e]` for an id in [0, E): the weight `MoeRouterTopK_v1` returns for the slot (a round's winning value is its unmasked
    probability)."""
    return _gather(B, S.E, [p[i] for i in range(S.E)], e)


def _gather(B, E: int, p: list[Coll], e: Coll) -> Coll:
    w = p[0]
    for i in range(1, E):
        w = B.call(P.SelectF32, B.call(P.I32Eq, B.call(b1.const(32, i)), e), p[i], w)
    return w


def _ordered(B, S, logits) -> tuple[list[Coll], list[Coll]]:
    p = B.call(bind(MoeRouterProbs, E=S.E, VPT=S.VPT), logits)
    ids = [B.call(bind(MoeRouterFirst, E=S.E), p)]
    sel = [B.call(bind(MoeRouterPick, E=S.E), p, ids[0])]
    for _ in range(1, S.TOPK):
        ids.append(B.call(bind(MoeRouterNext, E=S.E), p, sel[-1], ids[-1]))
        sel.append(B.call(bind(MoeRouterPick, E=S.E), p, ids[-1]))
    return sel, ids


@composite("MoeRouterTopKOrdered", 1, ["E", "TOPK", "VPT"],
           lambda S: ((("logits", Array(S.E, BF16)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
           conformance="not-yet-established")
def MoeRouterTopKOrdered(B, S, logits):
    """`MoeRouterTopK_v1` computed once per value: one softmax, then TOPK selection rounds (`MoeRouterFirst`, `MoeRouterNext`) and
    their weights (`MoeRouterPick`).  The same function, ties, NaN, -inf and +-0 included."""
    sel, ids = _ordered(B, S, logits)
    return tuple_of(_array(sel), _array(ids))


@composite("MoeRouterTopKOrderedNorm", 1, ["E", "TOPK", "VPT"],
           lambda S: ((("logits", Array(S.E, BF16)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
           conformance="not-yet-established")
def MoeRouterTopKOrderedNorm(B, S, logits):
    """`MoeRouterTopKNorm_v1` as `MoeRouterTopKOrdered_v1`, then the same epilogue: the selected probabilities summed in slot order from
    0.f, `denom = sum > 0.f ? sum : 1.f`, `w_k * (1.f / denom)`."""
    sel, ids = _ordered(B, S, logits)
    acc = B.call(b1.ZERO32)
    for k in range(S.TOPK):
        acc = B.call(P.F32Add, acc, sel[k])
    one = B.call(ONE32)
    denom = B.call(P.SelectF32, B.call(F32GtStrict, acc, B.call(b1.ZERO32)), acc, one)
    scale = B.call(P.F32Div, one, denom)
    return tuple_of(_array([B.call(P.F32Mul, sel[k], scale) for k in range(S.TOPK)]), _array(ids))


# ---------------------------------------------------------------------------------------------------------
# the sigmoid router with a selection bias (DeepSeek-V3 / GLM `noaux_tc`, one group): `_moe_C.grouped_topk(scores, 1, 1, TOPK,
# renormalize=True, scale, bias, SIGMOID)` -> `single_group_topk_{block,warp}_kernel<float, float, int, SIGMOID, 256, 8>`
# ---------------------------------------------------------------------------------------------------------

#: `sigmoid_accurate(x) = 0.5f * tanhf(0.5f * x) + 0.5f` as nvcc emits it (CUDA 13, no fast math; the same SASS on sm_89, sm_90,
#: sm_100 and sm_120), with h = x * 0.5:
#:   FMUL   a = |h| * 0x4038AA3B (2 log2 e)       FSETP.GE big = |h| >= 0x41102CB4 (9.0109)   FSETP.GE poly = !(|h| >= 0x3F19999A (0.6))
#:   MUFU.EX2 e = 2^a    FADD d = e + 1    MUFU.RCP r = 1/d    FFMA t = r * -2 + 1    FSEL t = big ? 1 : t    LOP3 t |= sign(h)
#:   s = h * h; p = fma(s, 0x3C80F082, 0xBD563CAE); p = fma(s, p, 0x3E085941); p = fma(s, p, 0xBEAAA9ED); q = fma(s, p, +0)
#:   @poly t = fma(h, q, h)                        FFMA sig = t * 0.5 + 0.5
#: `|h| >= c` is `|h| > pred(c)` (false on NaN, as FSETP.GE is), and `t |= sign(h)` is `h < 0 ? -t : t` where it is read (t >= +0,
#: and the polynomial is taken at h = +-0 and NaN).  For |x| >= 18.02 the sigmoid is exactly 0 or 1.
C_SIG_2LOG2E = b1.const(32, 0x4038AA3B)
C_SIG_P0, C_SIG_P1, C_SIG_P2, C_SIG_P3 = (b1.const(32, c) for c in (0x3C80F082, 0xBD563CAE, 0x3E085941, 0xBEAAA9ED))
C_SIG_BELOW_06 = b1.const(32, 0x3F199999)
C_SIG_BELOW_9 = b1.const(32, 0x41102CB3)
C_NEG2 = b1.f32c(-2.0)
C_1EM20 = b1.f32c(1e-20)


@composite("NvSigmoid", 1, [], lambda S: ((("x", F32),), F32), conformance="not-yet-established")
def NvSigmoid(B, S, x):
    """`sigmoid_accurate(x)` of `grouped_topk_kernels.cu` (`0.5f * tanhf(0.5f * x) + 0.5f`) as the SASS computes it (the constants
    above): CUDA's `tanhf` of h = x/2 by `1 - 2 / (2^(2|h| log2 e) + 1)` through MUFU.EX2 and MUFU.RCP with h's sign for
    0.6 <= |h| < 9.01, 1 with h's sign beyond, an odd degree-9 polynomial below, then one FFMA.  NaN -> NaN (`F32Fma_v1`'s word)."""
    zero = B.call(b1.ZERO32)
    half, one = B.call(C_HALF), B.call(ONE32)
    h = B.call(P.F32Mul, x, half)
    ah = B.call(F32Fabs, h)
    r = B.call(P.MufuRcpFtz, B.call(P.F32Add, B.call(P.MufuEx2Ftz, B.call(P.F32Mul, ah, B.call(C_SIG_2LOG2E))), one))
    big = B.call(P.SelectF32, B.call(F32GtStrict, ah, B.call(C_SIG_BELOW_9)), one, B.call(P.F32Fma, r, B.call(C_NEG2), one))
    big = B.call(P.SelectF32, B.call(F32GtStrict, zero, h), B.call(F32Neg, big), big)
    s = B.call(P.F32Mul, h, h)
    p = B.call(P.F32Fma, s, B.call(C_SIG_P0), B.call(C_SIG_P1))
    p = B.call(P.F32Fma, s, p, B.call(C_SIG_P2))
    p = B.call(P.F32Fma, s, p, B.call(C_SIG_P3))
    q = B.call(P.F32Fma, s, p, zero)
    t = B.call(P.SelectF32, B.call(F32GtStrict, ah, B.call(C_SIG_BELOW_06)), big, B.call(P.F32Fma, h, q, h))
    return B.call(P.F32Fma, t, half, half)


@composite("MoeRouterBiasScore", 1, [], lambda S: ((("x", F32), ("sig", F32), ("b", F32)), F32), conformance="not-yet-established")
def MoeRouterBiasScore(B, S, x, sig, b):
    """`preprocess_score<float, float, SIGMOID>`'s selection score from the logit x, its sigmoid and the correction bias b:
    `biased = b + sig` (FADD), `biased == 0 ? +0 : biased`, and -inf (`InvalidScore`) when x or b is not finite.  Never NaN or -0."""
    zero = B.call(b1.ZERO32)
    biased = B.call(P.F32Add, b, sig)
    sel = B.call(P.SelectF32, B.call(F32Eq, biased, zero), zero, biased)
    ok = B.call(BitAnd, B.call(F32IsFinite, x), B.call(F32IsFinite, b))
    return B.call(P.SelectF32, ok, sel, B.call(b1.NEG_INF32))


@composite("MoeRouterSigmoidScore", 1, [], lambda S: ((("x", F32), ("b", F32)), Tuple(F32, F32)), conformance="not-yet-established")
def MoeRouterSigmoidScore(B, S, x, b):
    """One expert's `preprocess_score<float, float, SIGMOID>`: (its sigmoid `NvSigmoid_v1(x)`, its selection score
    `MoeRouterBiasScore_v1(x, sig, b)`)."""
    sig = B.call(NvSigmoid, x)
    return tuple_of(sig, B.call(MoeRouterBiasScore, x, sig, b))


@composite("RouterArgmaxOkStep", 1, [],
           lambda S: ((("carry", Tuple(BIT, F32, I32, I32)), ("ok", BIT), ("x", F32)), Tuple(Tuple(BIT, F32, I32, I32), I32)),
           conformance="model-untested")
def RouterArgmaxOkStep(B, S, carry, ok, x):
    """Greedy argmax step over (found, best, best_index, i) among the entries with ok set: take (x, i) iff ok and (nothing found
    yet or x > best; strict, so the lowest index wins ties), then i += 1."""
    found, best, best_i, i = carry[0], carry[1], carry[2], carry[3]
    take = _ok_take(B, found, best, ok, x)
    nf = B.call(BitOr, found, ok)
    nb = B.call(P.SelectF32, take, x, best)
    ni = B.call(P.SelectI32, take, i, best_i)
    return tuple_of(tuple_of(nf, nb, ni, B.call(P.I32Add, i, B.call(b1.const(32, 1)))), ni)


def _ok_take(B, found: Coll, best: Coll, ok: Coll, x: Coll) -> Coll:
    return B.call(BitAnd, ok, B.call(BitOr, B.call(BitNot, found), B.call(F32GtStrict, x, best)))


def _first_max(B, s: list[Coll]) -> Coll:
    """`MoeRouterFirst_v1(s)`: `RouterArgmaxStep_v1`'s scan over s[1 .. E-2], then the last expert's step for its index alone (a
    scan's last step would also compute a best value and an i + 1 nothing reads)."""
    c0, c1 = B.call(b1.const(32, 0)), B.call(b1.const(32, 1))
    if len(s) == 1:
        return c0
    carry = tuple_of(s[0], c0, c1)
    if len(s) > 2:
        carry = B.scan(RouterArgmaxStep, carry, xs=(_array(s[1:-1]),))[0]
    return B.call(P.SelectI32, B.call(F32GtStrict, s[-1], carry[0]), carry[2], carry[1])


@composite("MoeRouterBiasNext", 1, ["E"], lambda S: ((("s", Array(S.E, F32)), ("w", F32), ("e", I32)), I32),
           conformance="not-yet-established")
def MoeRouterBiasNext(B, S, s, w, e):
    """Slot k of the kernel's top-k from slot k-1's (w = s[e], e): the first maximum among the experts after it in the order of
    the kernel's 64-bit keys `(TwiddleIn(s) << 32) | (65535 - i)` (value descending, then lower expert first): `s_i < w`, or
    `s_i == w` and `i > e`.  The selection scores are never NaN or -0, and -inf (`InvalidScore`) ranks by index like any value,
    so the experts not after (w, e) are left out by a flag, not a masking value; each round's operands are its own (`q_i`).
    `RouterArgmaxOkStep_v1`'s scan runs over experts 1 .. E-2 and the last expert's step gives the index alone."""
    if S.E < 2:
        raise ValueError("E must be at least 2 (a slot after the first)")
    neg_inf = B.call(b1.NEG_INF32)
    ok, q = [], []
    for i in range(S.E):
        ci = B.call(b1.const(32, i))
        after = B.call(BitOr, B.call(F32GtStrict, w, s[i]), B.call(BitAnd, B.call(F32Eq, s[i], w), B.call(BitNot, B.call(P.I32Le, ci, e))))
        ok.append(after)
        q.append(B.call(P.SelectF32, after, s[i], neg_inf))
    carry = tuple_of(ok[0], q[0], B.call(b1.const(32, 0)), B.call(b1.const(32, 1)))
    if S.E > 2:
        carry = B.scan(RouterArgmaxOkStep, carry, xs=(_array(ok[1:-1]), _array(q[1:-1])))[0]
    return B.call(P.SelectI32, _ok_take(B, carry[0], carry[1], ok[-1], q[-1]), carry[3], carry[2])


@composite("MoeRouterPickPair", 1, ["E"], lambda S: ((("s", Array(S.E, F32)), ("v", Array(S.E, F32)), ("e", I32)), Tuple(F32, F32)),
           conformance="not-yet-established")
def MoeRouterPickPair(B, S, s, v, e):
    """`(s[e], v[e])` for an id in [0, E) (s[0], v[0] past the end): one comparison of e per expert, read by both gathers."""
    a, b = s[0], v[0]
    for i in range(1, S.E):
        hit = B.call(P.I32Eq, B.call(b1.const(32, i)), e)
        a, b = B.call(P.SelectF32, hit, s[i], a), B.call(P.SelectF32, hit, v[i], b)
    return tuple_of(a, b)


#: `cg::reduce(warp, x, cg::plus<float>{})` on float: shfl.bfly with lane masks 16, 8, 4, 2, 1, each lane adding the other's value to
#: its own (the SASS of both kernels on sm_89 .. sm_120).
WARP_BUTTERFLY = (16, 8, 4, 2, 1)


def _lane0_sum(B, z: list[Coll]) -> Coll:
    """Lane 0's `cg::reduce(warp, lane_unbiased, plus)` when lanes [0, len(z)) hold z and the others +0: the butterfly's adds lane 0
    depends on, an add with a +0 lane dropped (it is exact: no z is -0 or NaN), so at TOPK = 8 ``((z0 + z4) + (z2 + z6)) + ((z1 +
    z5) + (z3 + z7))``."""
    memo: dict[tuple[int, int], Coll | None] = {}

    def lane(j: int, t: int) -> Coll | None:
        if j == 0:
            return z[t] if t < len(z) else None
        if (j, t) not in memo:
            a, b = lane(j - 1, t), lane(j - 1, t ^ WARP_BUTTERFLY[j - 1])
            memo[j, t] = a if b is None else b if a is None else B.call(P.F32Add, a, b)
        return memo[j, t]

    return lane(len(WARP_BUTTERFLY), 0)


ROUTER_KERNELS = ("block", "warp")


def router_kernel(num_experts: int, num_tokens: int, topk: int) -> str | None:
    """The kernel `_moe_C.grouped_topk` launches for one group (`grouped_topk_kernels.cu` `launch`): `single_group_topk_block_kernel`
    ("block") for at least 1024 experts or for at least 256 experts and at most 1024 tokens, else `single_group_topk_warp_kernel`
    ("warp").  None where the tuned top-16 block tiers (896 or 1024 experts) could apply, which this does not pin."""
    if topk > 8 and 512 < num_experts < 1024:
        return None
    return "block" if num_experts >= 1024 or (num_experts >= 256 and num_tokens <= 1024) else "warp"


def _router_kernel(S) -> str:
    if S.KERNEL not in ROUTER_KERNELS:
        raise ValueError(f"KERNEL must be one of {ROUTER_KERNELS}, got {S.KERNEL!r}")
    if not 1 <= S.TOPK <= 32:
        raise ValueError("TOPK must be in [1, 32] (one warp lane per slot)")
    return S.KERNEL


@composite("MoeRouterSigmoidNorm", 1, ["TOPK", "SCALE", "KERNEL"],
           lambda S: ((("w", Array(S.TOPK, F32)), ("v", Array(S.TOPK, F32))), Array(S.TOPK, F32)), conformance="not-yet-established")
def MoeRouterSigmoidNorm(B, S, w, v):
    """`write_outputs(renormalize=True, routed_scaling_factor=SCALE)` from slot k's selection score w[k] and v[k]: the unbiased score
    is v[k] (`KERNEL="block"`, the sigmoid read back from shared memory) or `w[k] - v[k]` with v[k] the expert's bias (`"warp"`,
    FADD).  `finite_k = w[k] != -inf`, `z_k = finite_k ? unbiased : 0`, `sum` = lane 0's warp reduction (`_lane0_sum`), then
    `z_k * (SCALE / (sum + 1e-20f))` (FADD, `div.rn.f32`, FMUL) for a finite slot and +0 for the others, or `1.f / TOPK` in every
    slot when no slot is finite."""
    kernel = _router_kernel(S)
    zero, neg_inf = B.call(b1.ZERO32), B.call(b1.NEG_INF32)
    fin = [B.call(BitNot, B.call(F32Eq, w[k], neg_inf)) for k in range(S.TOPK)]
    u = [v[k] if kernel == "block" else B.call(F32Sub, w[k], v[k]) for k in range(S.TOPK)]
    z = [B.call(P.SelectF32, fin[k], u[k], zero) for k in range(S.TOPK)]
    scale = B.call(P.F32Div, B.call(b1.f32c(S.SCALE)), B.call(P.F32Add, _lane0_sum(B, z), B.call(C_1EM20)))
    any_fin = fin[0]
    for k in range(1, S.TOPK):
        any_fin = B.call(BitOr, any_fin, fin[k])
    even = B.call(b1.const(32, P.F32Div.evaluate(bits_of(1.0), bits_of(float(S.TOPK)))))
    return _array([B.call(P.SelectF32, any_fin, B.call(P.SelectF32, fin[k], B.call(P.F32Mul, z[k], scale), zero), even)
                   for k in range(S.TOPK)])


_CC12, _CC90 = EVIDENCE["sigmoid_router_capture"]["cc12"], EVIDENCE["sigmoid_router_capture"]["cc90"]


@composite("MoeRouterSigmoidBias", 1, ["E", "TOPK", "SCALE", "KERNEL"],
           lambda S: ((("logits", Array(S.E, F32)), ("bias", Array(S.E, F32))), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
           conformance="exact-model-tested; the RTX PRO 6000's (cc 12.0, 188 SMs) _moe_C.grouped_topk at E=256, TOPK=8, SCALE=2.5, "
                       "both launches: 10,240 rows bit-exact against this Program and every MUFU-range logit word (65,873,464) "
                       f"against its twin (tests/properties/sigmoid_router_capture_gpu.py; {_CC12['run_record']}, "
                       f"{_CC12['art']}); likewise cc 9.0's (132 SMs) at vLLM d9105ea80's defaults "
                       f"(not batch invariant, the fused grouped top-k on; {_CC90['run_record']}, {_CC90['art']})")
def MoeRouterSigmoidBias(B, S, logits, bias):
    """`_moe_C.grouped_topk(scores, n_group=1, topk_group=1, TOPK, renormalize=True, SCALE, bias, scoring_func=SIGMOID)` for one
    token row of f32 router logits (`GateLinear`'s f32 output) and the f32 `e_score_correction_bias`: per expert the sigmoid
    and the selection score (`MoeRouterSigmoidScore_v1`); the TOPK largest selection scores in the kernel's key order,
    value descending then lower expert first (`MoeRouterFirst_v1`'s first maximum, then `MoeRouterBiasNext_v1`, each slot's score
    and second value read by `MoeRouterPickPair_v1`); then `MoeRouterSigmoidNorm_v1`.  Returns (topk_weights, topk_ids) as the
    kernel writes them.

    `KERNEL` is the launch: `single_group_topk_block_kernel` (`"block"`; at E = 256 for num_tokens <= 1024) reads the unbiased
    score back from shared memory, `single_group_topk_warp_kernel` (`"warp"`, more tokens) recovers it as score - bias.  Both
    select by `reduceTopK` over 32 lanes of E/32 sorted keys; at E = 256, TOPK = 8 (the tier `<256, 8>`: eight keys per lane, at
    most seven shifted out) its result is exactly the TOPK largest keys."""
    kernel = _router_kernel(S)
    if S.TOPK > S.E:
        raise ValueError("TOPK must not exceed E")
    scores = B.batch(MoeRouterSigmoidScore, logits, bias, axes=(0, 0))
    sig, s = _array([scores[e][0] for e in range(S.E)]), _array([scores[e][1] for e in range(S.E)])
    v = sig if kernel == "block" else bias
    ids = [_first_max(B, [s[e] for e in range(S.E)])]
    pairs = [B.call(bind(MoeRouterPickPair, E=S.E), s, v, ids[0])]
    for _ in range(1, S.TOPK):
        ids.append(B.call(bind(MoeRouterBiasNext, E=S.E), s, pairs[-1][0], ids[-1]))
        pairs.append(B.call(bind(MoeRouterPickPair, E=S.E), s, v, ids[-1]))
    out = B.call(bind(MoeRouterSigmoidNorm, TOPK=S.TOPK, SCALE=S.SCALE, KERNEL=kernel), _array([p[0] for p in pairs]),
                 _array([p[1] for p in pairs]))
    return tuple_of(out, _array(ids))
# ---------------------------------------------------------------------------------------------------------


@composite("MoeExpertRow", 1, ["E", "K"], lambda S: ((("e", I32), ("slab", Array(S.E, Array(S.K, BF16)))), Array(S.K, BF16)),
           conformance="not-yet-established")
def MoeExpertRow(B, S, e, slab):
    """`wrow[k] = slab[e][k]`: one wide-fan-in gather per k over the E candidate values of this weight element (the
    `Embedding_v1` construction with the expert axis in the vocabulary's role)."""
    cols = Coll(Array(S.K, Array(S.E, BF16)), strided_view(slab, 0, S.E, 1, S.K, S.K * S.E))
    return B.batch(P.gather_bf16(S.E), e, cols, axes=(None, 0))


@composite("MoeExpertCoordinate", 1, ["E", "K"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("slab", Array(S.E, Array(S.K, BF16)))), BF16),
           conformance="not-yet-established")
def MoeExpertCoordinate(B, S, x, e, slab):
    """One output word of the up/gate projection of pair (t, k): `bf16_rne(DotBf16(x_t, W13[e][n, :]))` — the
    `fused_moe_kernel` k-chain (`tl.dot` with a carried f32 accumulator, ascending k from zero) then `acc.to(bf16)`."""
    wrow = B.call(bind(MoeExpertRow, E=S.E, K=S.K), e, slab)
    return B.call(bind(b1.GemmCoordinate, K=S.K), x, wrow)


def coordinate_major(w: Coll, E: int, N: int, K: int) -> Coll:
    """`wt[n][e][k] = w[e][n][k]` as N `Strided` parts (one per output coordinate): a reference rearrangement, no gates."""
    return Coll(Array(N, Array(E, Array(K, BF16))), concat([strided_view(w, n * K, K, N * K, 1, E * K) for n in range(N)]))


@composite("MoeExpertGemm", 1, ["E", "K", "N"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("w", Array(S.E, Array(S.N, Array(S.K, BF16))))), Array(S.N, BF16)),
           conformance="not-yet-established")
def MoeExpertGemm(B, S, x, e, w):
    """`y = x @ W[e]^T` for one (token, slot) pair: `fused_moe_kernel` with MUL_ROUTED_WEIGHT=False (the w13 launch)."""
    return B.batch(bind(MoeExpertCoordinate, E=S.E, K=S.K), x, e, coordinate_major(w, S.E, S.N, S.K), axes=(None, None, 0))


@composite("MoeExpertCoordinateW", 1, ["E", "K"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("g", F32), ("slab", Array(S.E, Array(S.K, BF16)))), BF16),
           conformance="not-yet-established")
def MoeExpertCoordinateW(B, S, x, e, g, slab):
    """One output word of the down projection of pair (t, k): `bf16_rne(f32mul(DotBf16(x, W2[e][n, :]), g))` — the routed
    weight `g = topk_weights[t, k]` is multiplied into the f32 accumulator before the single bf16 conversion
    (`accumulator = accumulator * moe_weight[:, None]; accumulator.to(compute_type)`)."""
    wrow = B.call(bind(MoeExpertRow, E=S.E, K=S.K), e, slab)
    acc = B.call(bind(b1.DotBf16, K=S.K), x, wrow)
    return B.call(P.F2fpBf16, B.call(P.F32Mul, acc, g))


@composite("MoeExpertGemmW", 1, ["E", "K", "N"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("g", F32), ("w", Array(S.E, Array(S.N, Array(S.K, BF16))))),
                      Array(S.N, BF16)),
           conformance="not-yet-established")
def MoeExpertGemmW(B, S, x, e, g, w):
    """`y = g * (x @ W[e]^T)` for one (token, slot) pair: `fused_moe_kernel` with MUL_ROUTED_WEIGHT=True (the w2 launch)."""
    return B.batch(bind(MoeExpertCoordinateW, E=S.E, K=S.K), x, e, g, coordinate_major(w, S.E, S.N, S.K),
                   axes=(None, None, None, 0))


# ---------------------------------------------------------------------------------------------------------
# the expert GEMMs with the k16 tensor-core step as a static: `fused_moe_kernel`'s `tl.dot` on a target whose step is not the
# Ampere/Ada one (`frontend.targets.moe_expert_dot`); the bodies are the `_v1` ones with `DotBf16_v2{K,DOT}` for `DotBf16_v1`
# ---------------------------------------------------------------------------------------------------------

SM120_EVIDENCE = ("RTX PRO 6000 (cc 12.0, 188 SMs), vLLM d9105ea80 fused_experts under VLLM_BATCH_INVARIANT=1 (BLOCK 64/64/32, GROUP_SIZE_M 8): "
                  f"DOT=HopperBF16WgmmaDot16_v1 reproduces every w13 and w2 output of a probe's 16 cases "
                  f"({EVIDENCE['moe_expert_cc12_probe']['run_record']}) at "
                  "E=8, TOPK=2, H=2048, I=1024, M=16 (1,048,576 words per launch) where the Ampere step differs in 428 and 286")


@composite("MoeExpertCoordinate", 2, ["E", "K", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("slab", Array(S.E, Array(S.K, BF16)))), BF16),
           conformance="exact-model-tested: " + SM120_EVIDENCE)
def MoeExpertCoordinateV2(B, S, x, e, slab):
    """`MoeExpertCoordinate_v1` with the k16 step `DOT`: `bf16_rne(DotBf16_v2{K,DOT}(x_t, W13[e][n, :]))`."""
    wrow = B.call(bind(MoeExpertRow, E=S.E, K=S.K), e, slab)
    return B.call(bind(b1.GemmCoordinateV2, K=S.K, DOT=S.DOT), x, wrow)


@composite("MoeExpertGemm", 2, ["E", "K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("w", Array(S.E, Array(S.N, Array(S.K, BF16))))), Array(S.N, BF16)),
           conformance="exact-model-tested: " + SM120_EVIDENCE)
def MoeExpertGemmV2(B, S, x, e, w):
    """`MoeExpertGemm_v1` with the k16 step `DOT` (the w13 launch)."""
    return B.batch(bind(MoeExpertCoordinateV2, E=S.E, K=S.K, DOT=S.DOT), x, e, coordinate_major(w, S.E, S.N, S.K), axes=(None, None, 0))


@composite("MoeExpertCoordinateW", 2, ["E", "K", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("g", F32), ("slab", Array(S.E, Array(S.K, BF16)))), BF16),
           conformance="exact-model-tested: " + SM120_EVIDENCE)
def MoeExpertCoordinateWV2(B, S, x, e, g, slab):
    """`MoeExpertCoordinateW_v1` with the k16 step `DOT`: `bf16_rne(f32mul(DotBf16_v2{K,DOT}(x, W2[e][n, :]), g))`."""
    wrow = B.call(bind(MoeExpertRow, E=S.E, K=S.K), e, slab)
    acc = B.call(bind(b1.DotBf16V2, K=S.K, DOT=S.DOT), x, wrow)
    return B.call(P.F2fpBf16, B.call(P.F32Mul, acc, g))


@composite("MoeExpertGemmW", 2, ["E", "K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("e", I32), ("g", F32), ("w", Array(S.E, Array(S.N, Array(S.K, BF16))))),
                      Array(S.N, BF16)),
           conformance="exact-model-tested: " + SM120_EVIDENCE)
def MoeExpertGemmWV2(B, S, x, e, g, w):
    """`MoeExpertGemmW_v1` with the k16 step `DOT` (the w2 launch)."""
    return B.batch(bind(MoeExpertCoordinateWV2, E=S.E, K=S.K, DOT=S.DOT), x, e, g, coordinate_major(w, S.E, S.N, S.K),
                   axes=(None, None, None, 0))


# ---------------------------------------------------------------------------------------------------------
# routed sum over the TOPK slots of a token (`moe_sum_vec_kernel<bf16, int, TOPK, false>`)
# ---------------------------------------------------------------------------------------------------------

_V4 = EVIDENCE["glm53_block_fp8_capture/v4"]

SUM_CONF = ("exact-model-tested at TOPK = 8, H = 6144: vLLM d9105ea80's moe_sum (moe_sum_vec_kernel<bf16, int32, 8>) on cc 9.0 "
            "(132 SMs) at vLLM's defaults, inside GLM-5.3's served FusedMoEKernel (349 token rows) and on 8 planted rows (-0s, a tie into "
            "infinity, overflow, NaN payloads, infinities, ties, bf16 subnormals, order-dependent cancellations) on which descending "
            "order, a pairwise tree, the first slot's start, a bf16 accumulator, FTZ, NaN payloads kept and truncation each differ, "
            f"0 mismatches (property glm53_block_fp8_capture/v4, {_V4['run_record']}, {_V4['art']})")


@composite("MoeSumCoordinate", 1, ["TOPK"], lambda S: ((("col", Array(S.TOPK, BF16)),), BF16), conformance=SUM_CONF)
def MoeSumCoordinate(B, S, col):
    """`out = bf16(sum_k f32(col[k]))`, ascending k from a zero f32 accumulator, one RNE conversion at the end."""
    acc = B.call(b1.ZERO32)
    for k in range(S.TOPK):
        acc = B.call(P.F32Add, acc, B.call(P.Bf16ToF32, col[k]))
    return B.call(P.F2fpBf16, acc)


@composite("MoeSum", 1, ["TOPK", "H"], lambda S: ((("y", Array(S.TOPK, Array(S.H, BF16))),), Array(S.H, BF16)),
           conformance=SUM_CONF)
def MoeSum(B, S, y):
    """`out[t, :] = sum_k cache3[t, k, :]` for one token (the TOPK weighted expert outputs of the token)."""
    cols = Coll(Array(S.H, Array(S.TOPK, BF16)), strided_view(y, 0, S.TOPK, 1, S.H, S.H * S.TOPK))
    return B.batch(bind(MoeSumCoordinate, TOPK=S.TOPK), cols, axes=(0,))


ALL = [F32Sat, F32FmaRm, F32Neg, F32BitsShl23, NvExpf, F32IsFinite, RouterArgmaxStep, MoeRouterTopK, MoeRouterTopKNorm, MoeRouterProbs,
       MoeRouterFirst, MoeRouterNext, MoeRouterPick, MoeRouterTopKOrdered, MoeRouterTopKOrderedNorm, NvSigmoid, MoeRouterBiasScore,
       MoeRouterSigmoidScore, RouterArgmaxOkStep, MoeRouterBiasNext, MoeRouterPickPair, MoeRouterSigmoidNorm, MoeRouterSigmoidBias, MoeExpertRow,
       MoeExpertCoordinate, MoeExpertGemm, MoeExpertCoordinateW, MoeExpertGemmW, MoeExpertCoordinateV2, MoeExpertGemmV2,
       MoeExpertCoordinateWV2, MoeExpertGemmWV2, MoeSumCoordinate, MoeSum]
