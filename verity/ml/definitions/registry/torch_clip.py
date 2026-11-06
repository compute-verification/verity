"""``torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=True)`` on bf16 gradients as Boolean Definitions: the gates of torch
2.14's CUDA ``_foreach_norm`` (ForeachReduceOp.cu), ``linalg.vector_norm`` of the stacked norms (Reduce.cuh, ``NormTwoOps``),
the clip coefficient and ``_foreach_mul_`` by it. Every NaN of an f32 operation is the GPU's 0x7FFFFFFF and every rounding to
bf16 F2FP's (NaN 0x7FFF); nothing is flushed.

    Bf16SumSquares_v1{J}           (x: Array<J, BF16>) -> F32                                       J <= 65
    NormThreadSum_v1{K}            (v: Array<K, F32>) -> F32                                        2 <= K <= 64
    NormSqrtBf16_v1                (s: F32) -> BF16
    LpNormChunkSum_v1{P, ALIGNED}  (s: Array<P, F32>) -> F32                                        P <= 2048
    LpNormChunk_v1{N, ALIGNED}     (x: Array<N, BF16>) -> F32                                       N <= 65536
    LpNormCleanup_v1{C}            (o: Array<C, F32>) -> BF16                                       C <= 32768
    ForeachNorm_v1{N}              (x: Array<N, BF16>) -> BF16
    VectorNormBf16_v1{L}           (h: Array<L, BF16>) -> BF16                                      L <= 130560
    ClipCoef_v1                    (t: BF16, max_norm: F64) -> BF16
    ClipGradNorm_v1{SIZES}         (g: Tuple<Array<n, BF16> for n in SIZES>, max_norm: F64) -> Tuple<BF16, Tuple<...>>

(F32 is ``Array<32, Value<1>>``, F64 ``Array<64, Value<1>>``, BF16 ``Array<16, Value<1>>``.)

**One tensor's norm.** ``_foreach_norm(grads, 2)`` cuts each tensor into chunks of 65,536 elements, a CTA of 512 threads each
(``LpNormFunctor<BFloat16, L2>``). Chunk element 2048j + c is lane c's j-th, and a lane sums its squares in f32 from +0 in j's
order, ``acc = fma(x, x, acc)`` (`Bf16SumSquares`): the functor's sm_120 SASS has the FFMA. A bf16 word's square is exact in
f32 unless it falls below the normal range, so only there could the GPU's words tell an FFMA from an FMUL and an FADD. A lane is
one of its thread's four accumulators: on the 4-vector path, which a tensor takes when its numel is a multiple of 4 and its
pointer 8-byte aligned (ALIGNED), lane c is thread c / 4's accumulator c mod 4; otherwise thread c mod 512's accumulator
c / 512. Each thread adds its four to 0 in order (`NormThreadSum`), and ``BlockReduceSum`` adds the 512 threads' values: each
warp's halves tree (`torch_loss.WarpReduceSum_v1`), then warp 0's over the 16 warp sums and 16 zeros (`LpNormChunkSum`).
``lpnorm_cleanup`` adds the tensor's chunk partials, thread t's ``0 + o[t] + o[t + 512] + ...``, then ``BlockReduceSum``,
sqrt.rn and the bf16 store (`LpNormCleanup`, `NormSqrtBf16`). `ForeachNorm{N}` wires them for one tensor of N elements, on the
4-vector path exactly when N is a multiple of 4: the caching allocator's pointers are aligned.

Every partial is +0, positive, +inf or 0x7FFFFFFF, to which adding +0 changes nothing. So ``0 + v`` is v, an accumulator's
first element is ``fma(x, x, +0) = RN(x * x)``, an FMUL, and a lane or thread with no element is left out of a thread's sum,
or is a +0 word where a warp's tree reads it.

**The total.** ``vector_norm(stack(norms), 2)`` of the L bf16 norms is one CTA of ``reduce_kernel<512, 1>`` while
``ceil(L / block_width) < 256`` (L <= 130,560; past it torch splits the input across CTAs). Each thread has four accumulators
from +0, ``acc = fma(d, d, acc)`` (FFMA in the SASS). At L >= 128 the input is read in 4-vectors: block_width is the last power
of two of L / 4, at most 512, thread x takes the vectors x + block_width r, element i into accumulator i, then the tail element
L - L mod 4 + x into accumulator 0. Below 128, block_width is the last power of two of L, and thread x's elements
x + block_width m go into accumulator m mod 4. Each thread's ((a0 + a1) + a2) + a3, the halves tree over the block_width
threads (``block_x_reduce``: shared memory down to 32, then shfl_down by 16 .. 1), sqrt.rn and the bf16 store
(`VectorNormBf16`).

**The coefficient and the clip.** ``clamp(max_norm / (t + 1e-6), max=1.0)`` on the 0-dim bf16 total t (`ClipCoef`):
a = bf16(f32(t) + f32(1e-6)); ``max_norm / a`` is ``a.reciprocal() * max_norm``, r = bf16(1 / f32(a)) and
m = bf16(f32(r) * f32(max_norm)), the CPU scalar cast to float as cvtsd2ss does it (overflow to infinity); the clamp keeps a NaN
and gives 1.0 where m > 1. ``_foreach_mul_(grads, coef)`` is `boolean_dense.Bf16MulScalarTensor_v2` on each tensor.
`ClipGradNorm{SIZES}` is the whole call over tensors of the sizes SIZES in the parameters' order: (the total, the clipped grads).
"""
from __future__ import annotations

from typing import Any, Callable

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, and_reduce, const_bits, mux, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.ops.boolean.sqrt import f32_sqrt_rn
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Tuple, Type

from . import boolean_dense as BDN
from . import host_f64 as H
from .torch_attention import GPU_SHAPES, GPU_STEP, ON_GPU
from .torch_loss import WarpReduceSum

__all__ = ["Bf16SumSquares", "ClipCoef", "ClipGradNorm", "ForeachNorm", "LpNormChunk", "LpNormChunkSum", "LpNormCleanup",
           "NormSqrtBf16", "NormThreadSum", "VectorNormBf16", "clip_coef", "vector_norm_slots"]

BIT = BL.BIT
F32 = Array(32, BIT)
F64 = Array(64, BIT)
BF16 = Array(16, BIT)
_NAN = fp.F32_NAN_PTX
F32_ONE, BF16_ONE = 0x3F800000, 0x3F80
#: f32(1e-6), clip_grad's ``total_norm + 1e-6``
EPS_1E6 = 0x358637BD
CHUNK, THREADS, ILP, WARP = 65536, 512, 4, 32
LANES = THREADS * ILP
#: the most norms `VectorNormBf16` reduces in one CTA
VECTOR_NORM_MAX = 255 * THREADS
#: the most elements one of its accumulators takes there: 64 vectors' and a tail element
SQUARES_MAX = 65
N_MAX = 1 << 31
C_MAX = N_MAX // CHUNK
_CRAFTED = "43 tensors of 1 to 884,736 elements, on both paths, with NaN, inf, zeros and squares below f32's normal range"
_NORMS = f"every tensor's norm in {GPU_STEP} (20) and in 30 layers at {GPU_SHAPES} (272), and {_CRAFTED}"
_TOTALS = f"the total of {GPU_STEP} (L = 20) and of 30 layers at {GPU_SHAPES} (L = 272), and 32 crafted lengths of 1 to 4,096"
_IN_NORM = f"exact-model-tested; inside ForeachNorm_v1, {_NORMS}: {ON_GPU}"
_IN_BOTH = f"exact-model-tested; inside ForeachNorm_v1 and VectorNormBf16_v1, {_NORMS}, and {_TOTALS}: {ON_GPU}"


def _add(C, a, b):
    return fp.f32_add(C, a, b, nan_word=_NAN)


def _bf16(C, x):
    return fp.f32_to_bf16(C, x, W.BF16_NAN_F2FP)


def sum_squares(C, xs):
    """``acc = fma(x, x, acc)`` from +0 over the f32 words xs in order, the first ``RN(x * x)``."""
    acc = fp.f32_mul(C, xs[0], xs[0], nan_word=_NAN)
    for x in xs[1:]:
        acc = fp.f32_fma(C, x, x, acc, nan_word=_NAN)
    return acc


def _fold(C, vals):
    acc = vals[0]
    for v in vals[1:]:
        acc = _add(C, acc, v)
    return acc


def _refusal(k: str, v: Any, lo: int, hi: int) -> str | None:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        return f"{k} = {v!r} is outside {lo}..{hi}, the range of the kernel this is"
    return None


def _aligned_refusal(b: dict) -> str | None:
    a, n = b.get("ALIGNED"), b.get("N", b.get("P"))
    if a is not None and (r := _refusal("ALIGNED", a, 0, 1)):
        return r
    if a == 1 and isinstance(n, int) and n % ILP:
        return f"ALIGNED = 1 needs a multiple of 4 elements, not {n}: a tensor whose numel isn't one takes the strided path"
    return None


def _refuse(*bounds: tuple[str, int, int], aligned: bool = False):
    def refuse(b: dict) -> str | None:
        for k, lo, hi in bounds:
            if b.get(k) is not None and (r := _refusal(k, b[k], lo, hi)):
                return r
        return _aligned_refusal(b) if aligned else None
    return refuse


def _n(S: Any, k: str, lo: int, hi: int) -> int:
    v = getattr(S, k)
    if r := _refusal(k, v, lo, hi):
        raise ValueError(r)
    return v


def _aligned(S: Any, k: str) -> int:
    if r := _aligned_refusal({"ALIGNED": S.ALIGNED, k: getattr(S, k)}):
        raise ValueError(r)
    return S.ALIGNED


Bf16SumSquares = specialized(
    "Bf16SumSquares", 1, ("J",), lambda S: lambda C, *x: [sum_squares(C, [fp.bf16_to_f32(h) for h in x])],
    lambda S: ((("x", Array(_n(S, "J", 1, SQUARES_MAX), BF16)),), F32),
    doc="acc = fma(x, x, acc) in f32 from +0 over J bf16 words in order, the first RN(x * x): a lane of "
        "LpNormFunctor<BFloat16, L2> (J <= 32) or an accumulator of NormTwoOps' reduce_kernel (J <= 65)",
    refuse=_refuse(("J", 1, SQUARES_MAX)), conformance=_IN_BOTH)
NormThreadSum = specialized(
    "NormThreadSum", 1, ("K",), lambda S: lambda C, *v: [_fold(C, list(v))],
    lambda S: ((("v", Array(_n(S, "K", 2, C_MAX // THREADS), F32)),), F32),
    doc="a thread's f32 sum of its K accumulators or chunk slots in order, v0 + v1 + ... (0 + v0 is v0 on every value "
        "these take: +0, positive, +inf or 0x7FFFFFFF)",
    refuse=_refuse(("K", 2, C_MAX // THREADS)), conformance=_IN_BOTH)
NormSqrtBf16 = definition(
    "NormSqrtBf16", 1, lambda C, s: _bf16(C, f32_sqrt_rn(C, s)), (("s", F32),), BF16,
    doc="a norm's store: sqrt.rn of the f32 sum of squares, then bf16 (F2FP, NaN 0x7FFF)", conformance=_IN_BOTH)


def _zeros(B, t: Type) -> Coll:
    z = B.call(BL.constant(0))
    return Coll(t, concat([z.refs] * t.leaves))


def _pad(B, v: Coll, n: int) -> Coll:
    """The F32 words of v and +0 words after them, n in all."""
    k = len(v)
    return v if k == n else Coll(Array(n, F32), concat([v.refs, _zeros(B, Array(n - k, F32)).refs]))


def _strided(B, x: Coll, n: int, width: int, fn: Callable[[int], Any]) -> Coll:
    """Member c < min(n, width) of the n elements of x: fn(J) on its J elements c, c + width, c + 2 width, ... in order, one
    batch over the members of each J; fn(1) None takes the element itself. -> Array<min(n, width), F32>."""
    q, r = divmod(n, width)
    t = x.type.elem
    parts = []
    for J, lo, hi in ((q + 1, 0, r), (q, r, width)):
        if J == 0 or hi == lo:
            continue
        g = Coll(Array(J, Array(hi - lo, t)), concat([x[width * j + lo:width * j + hi].refs for j in range(J)]))
        f = fn(J)
        parts.append(g[0] if f is None else B.batch(f, g, axes=(1,)))
    return Coll(Array(min(n, width), F32), concat([p.refs for p in parts]))


def _thread_sum(K: int) -> Any:
    return None if K == 1 else bind(NormThreadSum, K=K)


def _block_reduce_sum(B, v: Coll) -> Coll:
    """``cuda_utils::BlockReduceSum`` at thread 0 over the first len(v) of 512 threads, the rest +0: each warp's
    `WarpReduceSum`, then warp 0's over the 16 warp sums and 16 zeros."""
    T = len(v)
    full, part = divmod(T, WARP)
    sums = []
    if full:
        sums.append(B.batch(WarpReduceSum, v[:WARP * full].reshape(Array(full, Array(WARP, F32))), axes=(0,)))
    if part:
        sums.append(B.call(WarpReduceSum, _pad(B, v[WARP * full:], WARP)).reshape(Array(1, F32)))
    return B.call(WarpReduceSum, _pad(B, Coll(Array(full + bool(part), F32), concat([s.refs for s in sums])), WARP))


def _chunk_sum(B, S, s):
    P, a = S.P, S.ALIGNED
    if a:
        threads = B.batch(bind(NormThreadSum, K=ILP), s.reshape(Array(P // ILP, Array(ILP, F32))), axes=(0,))
    else:
        threads = _strided(B, s, P, THREADS, _thread_sum)
    return _block_reduce_sum(B, threads)


def _chunk_sum_signature(S: Any):
    P = _n(S, "P", 1, LANES)
    _aligned(S, "P")
    return (("s", Array(P, F32)),), F32


LpNormChunkSum = CompositeDefinition(
    "LpNormChunkSum", 1, ("P", "ALIGNED"), _chunk_sum_signature, _chunk_sum,
    doc="one chunk's partial in LpNormFunctor<BFloat16, L2> from its first P lanes' sums, the rest +0: NormThreadSum_v1 of "
        "each thread's lanes in order (4t .. 4t + 3 at ALIGNED = 1; t, t + 512, t + 1024, t + 1536 at 0), then "
        "BlockReduceSum: WarpReduceSum_v1 on each warp, then on the 16 warp sums and 16 zeros",
    conformance=_IN_NORM, refuse=_refuse(("P", 1, LANES), aligned=True))


def _chunk(B, S, x):
    lanes = _strided(B, x, S.N, LANES, lambda J: bind(Bf16SumSquares, J=J))
    return B.call(bind(LpNormChunkSum, P=len(lanes), ALIGNED=S.ALIGNED), lanes)


def _chunk_signature(S: Any):
    N = _n(S, "N", 1, CHUNK)
    _aligned(S, "N")
    return (("x", Array(N, BF16)),), F32


LpNormChunk = CompositeDefinition(
    "LpNormChunk", 1, ("N", "ALIGNED"), _chunk_signature, _chunk,
    doc="LpNormFunctor<BFloat16, L2>'s f32 partial of one chunk of N <= 65536 elements: Bf16SumSquares_v1 on each lane "
        "(element 2048j + c is lane c's j-th), then LpNormChunkSum_v1 on the lanes",
    conformance=_IN_NORM, refuse=_refuse(("N", 1, CHUNK), aligned=True))


def _cleanup(B, S, o):
    return B.call(NormSqrtBf16, _block_reduce_sum(B, _strided(B, o, S.C, THREADS, _thread_sum)))


LpNormCleanup = CompositeDefinition(
    "LpNormCleanup", 1, ("C",), lambda S: ((("o", Array(_n(S, "C", 1, C_MAX), F32)),), BF16), _cleanup,
    doc="lpnorm_cleanup<BFloat16, L2> for one tensor of C chunks: thread t's 0 + o[t] + o[t + 512] + ... (NormThreadSum_v1), "
        "BlockReduceSum, NormSqrtBf16_v1",
    conformance=_IN_NORM, refuse=_refuse(("C", 1, C_MAX)))


def _foreach(B, S, x):
    N = S.N
    a = int(N % ILP == 0)
    F, rest = divmod(N, CHUNK)
    parts = []
    if F:
        parts.append(B.batch(bind(LpNormChunk, N=CHUNK, ALIGNED=a), x[:F * CHUNK].reshape(Array(F, Array(CHUNK, BF16))),
                             axes=(0,)))
    if rest:
        parts.append(B.call(bind(LpNormChunk, N=rest, ALIGNED=a), x[F * CHUNK:]).reshape(Array(1, F32)))
    n = F + bool(rest)
    return B.call(bind(LpNormCleanup, C=n), Coll(Array(n, F32), concat([p.refs for p in parts])))


ForeachNorm = CompositeDefinition(
    "ForeachNorm", 1, ("N",), lambda S: ((("x", Array(_n(S, "N", 1, N_MAX), BF16)),), BF16), _foreach,
    doc="torch._foreach_norm's bf16 L2 norm of one tensor of N bf16 elements at an aligned pointer: LpNormChunk_v1 on each "
        "chunk of 65536 (ALIGNED = 1 when N is a multiple of 4), then LpNormCleanup_v1",
    conformance=f"exact-model-tested; {_NORMS}: {ON_GPU}", refuse=_refuse(("N", 1, N_MAX)))


def _last_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def vector_norm_slots(L: int) -> list[list[list[int]]]:
    """The elements of each of `VectorNormBf16{L}`'s threads' four accumulators, in the order each adds them."""
    vec = L >= 128
    bw = min(_last_pow2(L // ILP if vec else L), THREADS)
    slots: list[list[list[int]]] = [[[] for _ in range(ILP)] for _ in range(bw)]
    if vec:
        for v in range(L // ILP):
            for i in range(ILP):
                slots[v % bw][i].append(ILP * v + i)
        for x, e in enumerate(range(L - L % ILP, L)):
            slots[x][0].append(e)
    else:
        for e in range(L):
            slots[e % bw][e // bw % ILP].append(e)
    return slots


def _halves(B, v: Coll) -> Coll:
    """The halves tree over the len(v) values, a power of two (``block_x_reduce``): s[i] + s[i + h] for h from half the count
    down to 1, on `WarpReduceSum` calls; the trees of the strided groups v[i + 32k] first when there are more than 32."""
    n = len(v)
    if n == 1:
        return v[0]
    if n <= WARP:
        return B.call(WarpReduceSum, _pad(B, v, WARP))
    m = n // WARP
    g = v.reshape(Array(m, Array(WARP, F32)))
    groups = Coll(Array(WARP, Array(WARP, F32)), concat([_pad(B, g.take(1, i), WARP).refs for i in range(WARP)]))
    return B.call(WarpReduceSum, B.batch(WarpReduceSum, groups, axes=(0,)))


def _vector_norm(B, S, h):
    slots = vector_norm_slots(S.L)
    by_len: dict[int, list[tuple[int, int]]] = {}
    for t, accs in enumerate(slots):
        for i, acc in enumerate(accs):
            by_len.setdefault(len(acc), []).append((t, i))
    acc_of: dict[tuple[int, int], Coll] = {}
    zero = _zeros(B, F32)
    for J, members in by_len.items():
        if J == 0:
            acc_of.update({m: zero for m in members})
            continue
        g = Coll(Array(len(members), Array(J, BF16)),
                 concat([h[e:e + 1].refs for t, i in members for e in slots[t][i]]))
        out = B.batch(bind(Bf16SumSquares, J=J), g, axes=(0,))
        acc_of.update({m: out[k] for k, m in enumerate(members)})
    accs = Coll(Array(len(slots), Array(ILP, F32)),
                concat([acc_of[(t, i)].refs for t in range(len(slots)) for i in range(ILP)]))
    return B.call(NormSqrtBf16, _halves(B, B.batch(bind(NormThreadSum, K=ILP), accs, axes=(0,))))


VectorNormBf16 = CompositeDefinition(
    "VectorNormBf16", 1, ("L",), lambda S: ((("h", Array(_n(S, "L", 1, VECTOR_NORM_MAX), BF16)),), BF16), _vector_norm,
    doc="torch.linalg.vector_norm(h, 2) of L <= 130560 contiguous bf16 words, one CTA of reduce_kernel<512, 1> with "
        "NormTwoOps<BFloat16, float, BFloat16>: Bf16SumSquares_v1 on each thread's four accumulators (4-vector reads at "
        "L >= 128), NormThreadSum_v1 of the four, the halves tree over block_width threads (WarpReduceSum_v1), NormSqrtBf16_v1",
    conformance=f"exact-model-tested; {_TOTALS}: {ON_GPU}", refuse=_refuse(("L", 1, VECTOR_NORM_MAX)))


def clip_coef(C, t, max_norm):
    """``clamp(max_norm / (t + 1e-6), max=1.0)`` on the bf16 word t and the f64 word max_norm (module docstring)."""
    a = _bf16(C, fp.f32_add(C, fp.bf16_to_f32(t), fp.f32_const(EPS_1E6), nan_word=_NAN))
    r = _bf16(C, fp.f32_div(C, fp.f32_const(F32_ONE), fp.bf16_to_f32(a), nan_word=_NAN))
    k, _ = H.f64_to_f32(C, max_norm)
    m = _bf16(C, fp.f32_mul(C, fp.bf16_to_f32(r), k, nan_word=_NAN))
    _, one_or_less = sub(C, list(m[:15]), const_bits(BF16_ONE + 1, 15), 15)
    _, not_nan = sub(C, list(m[:15]), const_bits(0x7F81, 15), 15)
    big = and_reduce(C, [NOT(m[15]), NOT(one_or_less), not_nan])
    return mux(C, big, const_bits(BF16_ONE, 16), list(m))


ClipCoef = definition(
    "ClipCoef", 1, clip_coef, (("t", BF16), ("max_norm", F64)), BF16,
    doc="clip_grad's coefficient on bits: a = bf16(f32(t) + f32(1e-6)), r = bf16(1 / f32(a)), m = bf16(f32(r) * "
        "f32(max_norm)) (cvtsd2ss), then clamp(max=1.0): 1.0 where m > 1, else m (a NaN kept)",
    conformance=f"exact-model-tested; every bf16 total at max_norm 1.0, as one tensor and as 2,034 0-dim tensors: {ON_GPU}")


def _sizes_refusal(v: Any) -> str | None:
    if not isinstance(v, tuple) or not 1 <= len(v) <= VECTOR_NORM_MAX:
        return f"SIZES is a tuple of 1 to {VECTOR_NORM_MAX} tensor sizes, not {v!r}"
    return next((r for n in v if (r := _refusal("a size", n, 1, N_MAX))), None)


def _grads(S: Any) -> Tuple:
    if r := _sizes_refusal(S.SIZES):
        raise ValueError(r)
    return Tuple(*[Array(n, BF16) for n in S.SIZES])


def _clip(B, S, g, max_norm):
    norms = [B.call(bind(ForeachNorm, N=n), g[i]) for i, n in enumerate(S.SIZES)]
    total = B.call(bind(VectorNormBf16, L=len(S.SIZES)), array_of(norms))
    coef = B.call(ClipCoef, total, max_norm).reshape(Array(1, BF16))
    return tuple_of(total, tuple_of(*[B.call(bind(BDN.Bf16MulScalarTensor, N=n), g[i], coef) for i, n in enumerate(S.SIZES)]))


ClipGradNorm = CompositeDefinition(
    "ClipGradNorm", 1, ("SIZES",),
    lambda S: ((("g", _grads(S)), ("max_norm", F64)), Tuple(BF16, _grads(S))), _clip,
    doc="torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=True) on bf16 grads of sizes SIZES on one GPU: "
        "ForeachNorm_v1 on each, VectorNormBf16_v1 of the norms, ClipCoef_v1, then Bf16MulScalarTensor_v2 of each grad by it "
        "-> (the total, the clipped grads)",
    conformance=f"exact-model-tested; the total and every clipped word of {GPU_STEP} and the total of 30 layers at "
                f"{GPU_SHAPES}: {ON_GPU}",
    refuse=lambda b: None if b.get("SIZES") is None else _sizes_refusal(b["SIZES"]))
