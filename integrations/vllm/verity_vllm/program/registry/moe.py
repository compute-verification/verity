"""[R13 moe / MOE-01] The vLLM fused-MoE block (`vllm.moe_forward` -> `FusedMoEKernel(MoEPrepareAndFinalizeNoEP, TritonExperts)`, pinned
d9105ea80) as REGISTERED Definitions -- promoted from `registry.quarantine.ov_moe` (lane ov-moe 2026-09-17, NvExpf from cov-moe) with the
same names / versions, so the specialization ids the accepted OLMoE eager records resolved to are unchanged.

The block is: router top-k (`_moe_C.topk_softmax`) -> block schedule (`_moe_C.moe_align_block_size`, M*TOPK*4 > E; else the naive
assignment) -> `fused_moe_kernel(w13, MUL_ROUTED_WEIGHT=False)` -> `_C.silu_and_mul` -> `fused_moe_kernel(w2, MUL_ROUTED_WEIGHT=True)`
-> `_moe_C.moe_sum`.  Every numerical statement is per (token t, slot k) PAIR p = t*TOPK + k: the fused kernel's grouping of pairs into
expert blocks (sorted_token_ids / expert_ids / num_tokens_post_padded) is a SCHEDULE that never enters the arithmetic of an output
word -- each output word of the pair-major intermediates is one ascending-k chain of the same tensor-core steps as the dense GEMM
(`DotBf16`), then one f32 -> bf16 conversion (with the routing weight multiplied in before the conversion on the down projection).
The tile constants (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps/stages) vLLM picks per M (`try_get_optimal_moe_config`) schedule which
CTA owns which pair rows and how the k-loop is chunked; a chunk of BLOCK_SIZE_K is BLOCK_SIZE_K/16 k16 steps into the SAME carried
accumulator, so the per-coordinate chain is the tile-independent one -- checked, not assumed: `verity_vllm/ops/moe_kernel_xcheck.py`
(cp-moe, R13) evaluates these bodies against the real kernels at several M (different tilings, naive vs sorted schedule, different
routing patterns).

Representation (the one that fits the existing operand kinds; the precedent is `Embedding_v1`): the expert index `e` is an ordinary I32
VALUE operand of the pair body -- produced by `MoeRouterTopK_v1` from the router logits, never a structural input or an observed
constant -- and the selection of the expert's weight row happens INSIDE the body as a wide-fan-in gather over the expert axis
(`GatherBf16x{E}` per weight element).  The Program's dataflow is then fixed by shapes alone (M, TOPK, E, H, I); the data-dependent part
is arithmetic on a value the Program itself produced, exactly as the token id is for the embedding.

The router's exponential is `NvExpf_v1`: the CUDA math-library `expf` sequence exactly as nvcc emits it for sm_89 (transcribed from the
PTX of a kernel calling `expf`, CUDA 12.x headers; `cvt.sat` / `fma.rm` / `ex2.approx.ftz` / exponent-field scaling) -- bit-exact against
`_moe_C::topk_softmax`'s weights (cov-moe: 192/192 words; R13 xcheck: every routed weight of every pattern).
"""
from __future__ import annotations

from fractions import Fraction

from verity.ir.defs import bind, composite, primitive
from verity.ir.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ir.types import Array, Tuple
from verity.ml.scalar import F32BitsShl23, F32IsFinite, F32Neg, F32Sat
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry import ref_prims as R
from verity_vllm.program.registry.prims import BF16, F32, I32, _decode, f32_of, bits_of

SEMANTIC_ID = "moe/FusedMoE@v1"   # (the quarantine module kept its ov-moe id)

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


#: The constants of nvcc's `expf` (sm_89, CUDA 12.x math headers; PTX of `y = expf(x)` on the pod, 2026-09-17):
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


def nv_expf_bits(x: int) -> int:
    """The same sequence evaluated directly on one f32 encoding (for probes / difftests; `NvExpf` is the Program-side
    statement of it and the two are asserted equal by the lane's tests)."""
    t = P.F32Fma.evaluate(x, 0x3BBB989D, 0x3F000000)
    s = F32Sat.evaluate(t)
    jb = F32FmaRm.evaluate(s, bits_of(252.0), bits_of(12582913.0))
    j = P.F32Add.evaluate(jb, bits_of(-12583039.0))
    f = P.F32Fma.evaluate(x, 0x3FB8AA3B, F32Neg.evaluate(j))
    f = P.F32Fma.evaluate(x, 0x32A57060, f)
    return P.F32Mul.evaluate(P.MufuEx2Ftz.evaluate(f), F32BitsShl23.evaluate(jb))


ONE32 = b1.f32c(1.0)
NEG_10000 = b1.f32c(-10000.0)


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
    E, TOPK, VPT = S.E, S.TOPK, S.VPT
    if E % VPT:
        raise ValueError("E must be a multiple of VPT")
    T = E // VPT
    x = [B.call(P.Bf16ToF32, logits[i]) for i in range(E)]
    tmax = []
    for t in range(T):
        m = x[t * VPT]
        for j in range(1, VPT):
            m = B.call(P.F32Max, m, x[t * VPT + j])
        tmax.append(m)
    rmax = _butterfly(B, P.F32Max, tmax)[0]
    ex = [B.call(NvExpf, B.call(F32Sub, x[i], rmax)) for i in range(E)]
    tsum = []
    for t in range(T):
        s = B.call(b1.ZERO32)
        for j in range(VPT):
            s = B.call(P.F32Add, s, ex[t * VPT + j])
        tsum.append(s)
    rsum = _butterfly(B, P.F32Add, tsum)[0]
    rcp = B.call(P.F32Div, B.call(ONE32), rsum)
    zero = B.call(b1.ZERO32)
    p = []
    for i in range(E):
        q = B.call(P.F32Mul, ex[i], rcp)
        p.append(B.call(P.SelectF32, B.call(F32IsFinite, q), q, zero))
    neg_big = B.call(NEG_10000)
    consts = [B.call(b1.const(32, i)) for i in range(E)]
    weights, ids = [], []
    for _ in range(TOPK):
        init = tuple_of(p[0], consts[0], consts[1])
        res = B.scan(RouterArgmaxStep, init, xs=(_array(p[1:]),))
        best, best_i = res[0][0], res[0][1]
        weights.append(best)
        ids.append(best_i)
        p = [B.call(P.SelectF32, B.call(P.I32Eq, consts[i], best_i), neg_big, p[i]) for i in range(E)]
    return tuple_of(_array(weights), _array(ids))


def _array(parts: list[Coll]) -> Coll:
    return array_of(list(parts))


@composite("MoeRouterTopKNorm", 1, ["E", "TOPK", "VPT"],
           lambda S: ((("logits", Array(S.E, BF16)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
           conformance="not-yet-established")
def MoeRouterTopKNorm(B, S, logits):
    """[R13 MOE-04] `topk_softmax(renormalize=True)` for one token row (Qwen3-MoE / Mixtral `norm_topk_prob`): the same kernel as
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
# expert GEMMs: one (token, slot) pair, the expert index as an I32 value operand, the row gather inside the body
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
# routed sum over the TOPK slots of a token (`moe_sum_vec_kernel<bf16, int, TOPK, false>`)
# ---------------------------------------------------------------------------------------------------------


@composite("MoeSumCoordinate", 1, ["TOPK"], lambda S: ((("col", Array(S.TOPK, BF16)),), BF16), conformance="not-yet-established")
def MoeSumCoordinate(B, S, col):
    """`out = bf16(sum_k f32(col[k]))`, ascending k from a zero f32 accumulator, one RNE conversion at the end."""
    acc = B.call(b1.ZERO32)
    for k in range(S.TOPK):
        acc = B.call(P.F32Add, acc, B.call(P.Bf16ToF32, col[k]))
    return B.call(P.F2fpBf16, acc)


@composite("MoeSum", 1, ["TOPK", "H"], lambda S: ((("y", Array(S.TOPK, Array(S.H, BF16))),), Array(S.H, BF16)),
           conformance="not-yet-established")
def MoeSum(B, S, y):
    """`out[t, :] = sum_k cache3[t, k, :]` for one token (the TOPK weighted expert outputs of the token)."""
    cols = Coll(Array(S.H, Array(S.TOPK, BF16)), strided_view(y, 0, S.TOPK, 1, S.H, S.H * S.TOPK))
    return B.batch(bind(MoeSumCoordinate, TOPK=S.TOPK), cols, axes=(0,))


ALL = [F32Sat, F32FmaRm, F32Neg, F32BitsShl23, NvExpf, F32IsFinite, RouterArgmaxStep, MoeRouterTopK, MoeRouterTopKNorm, MoeExpertRow,
       MoeExpertCoordinate, MoeExpertGemm, MoeExpertCoordinateW, MoeExpertGemmW, MoeSumCoordinate, MoeSum]
