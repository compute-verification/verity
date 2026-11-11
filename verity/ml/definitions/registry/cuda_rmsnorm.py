"""vLLM d9105ea80's non-batch-invariant CUDA RMSNorm kernels for contiguous BF16 rows.

The public statics record the actual launch block, because both kernels switch from their
large block to 256 threads when ``num_tokens >= 256``.  ``RMSNormCudaVector8_v1`` is
``rms_norm_kernel<c10::BFloat16, 8, 2, true>``.  ``RMSNormFusedCudaVector8_v1`` is
``fused_add_rms_norm_kernel<c10::BFloat16, 8, true>`` and returns both in-place outputs.
``RMSNormFusedCudaVector8_v3``, the one served, is v1 in v1's f32 order with its sum of squares as batches, so its body has a few dozen
nodes at any N where v1 unrolls every square.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, array_of, coll_concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.prims import const
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, F32

VECTOR = 8
LARGE_BLOCK = 1024
SMALL_BLOCK = 256
SMALL_BLOCK_FROM_TOKENS = 256


def launch_block(num_tokens: int, hidden_size: int, *, fused: bool) -> int:
    """The pinned launch's ``blockDim.x`` for a contiguous BF16 row."""
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be positive, not {num_tokens}")
    if hidden_size < VECTOR or hidden_size % VECTOR:
        raise ValueError(f"hidden_size must be a positive multiple of {VECTOR}, not {hidden_size}")
    cap = LARGE_BLOCK if num_tokens < SMALL_BLOCK_FROM_TOKENS else SMALL_BLOCK
    return min(hidden_size if fused else hidden_size // VECTOR, cap)


def _check(S, *, fused: bool) -> tuple[int, int]:
    N, block = int(S.N), int(S.BLOCK)
    if N < VECTOR or N % VECTOR:
        raise ValueError(f"N must be a positive multiple of {VECTOR}, not {N}")
    limit = N if fused else N // VECTOR
    if block < 1 or block > min(limit, LARGE_BLOCK):
        raise ValueError(f"BLOCK={block} is outside 1..{min(limit, LARGE_BLOCK)} for N={N}")
    return N, block


def _cub(B, partials):
    return b1._cub_block_reduce_sum(B, list(partials))


def _arg_scale(B, S, var):
    mean = B.call(P.F32Div, var, B.call(const(32, P.bits_of(float(S.N)))))
    return B.call(P.RsqrtApprox, B.call(P.F32Add, mean, B.call(const(32, P.bits_of(float(S.EPS))))))


def _out(B, x, w, scale):
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    y = B.batch(P.F32Mul, B.batch(P.F32Mul, xf, scale, axes=(0, None)), wf, axes=(0, 0))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite(
    "RMSNormCudaVector8",
    1,
    ["N", "EPS", "BLOCK"],
    lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
    conformance="exact-model-tested",
)
def RMSNormCudaVector8(B, S, x, w):
    """The served non-fused CUDA RMSNorm vector path.

    There are ``BLOCK`` threads.  Thread ``t`` visits eight contiguous elements in
    vector ``t``, then vectors ``t + BLOCK``, ..., and accumulates one f32 FMA chain
    from +0.  CUB reduces the thread partials; the epilogue is
    ``bf16_rn((f32(x) * rsqrtf(var / N + EPS)) * f32(w))``.
    """
    N, block = _check(S, fused=False)
    vectors = N // VECTOR
    partials = []
    for t in range(block):
        els = [VECTOR * v + j for v in range(t, vectors, block) for j in range(VECTOR)]
        xf = [B.call(P.Bf16ToF32, x[e]) for e in els]
        acc = B.call(P.F32Mul, xf[0], xf[0])
        for z in xf[1:]:
            acc = B.call(P.F32Fma, z, z, acc)
        partials.append(acc)
    return _out(B, x, w, _arg_scale(B, S, _cub(B, partials)))


def _vector_sum(B, rf):
    pairs = [
        B.call(P.F32Add, B.call(P.F32Mul, rf[j], rf[j]), B.call(P.F32Mul, rf[j + 1], rf[j + 1]))
        for j in range(0, VECTOR, 2)
    ]
    acc = pairs[0]
    for pair in pairs[1:]:
        acc = B.call(P.F32Add, acc, pair)
    return acc


@composite(
    "RMSNormFusedCudaVector8",
    1,
    ["N", "EPS", "BLOCK"],
    lambda S: (
        (("x", Array(S.N, BF16)), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16))),
        Tuple(Array(S.N, BF16), Array(S.N, BF16)),
    ),
    conformance="exact-model-tested",
)
def RMSNormFusedCudaVector8V1(B, S, x, res, w):
    """The residual-add CUDA RMSNorm vector path, every square and add a node of its own.

    ``res'`` is the native BF16 add.  A vector partial is its four adjacent pair
    sums added left to right; a thread adds its later vector partials left to right.
    CUB then reduces the active partials.  Threads beyond ``N / 8`` hold +0; their
    additions are identities for these square partials and are omitted.
    """
    N, block = _check(S, fused=True)
    r = B.batch(P.Bf16AddF2fp, x, res, axes=(0, 0))
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    vectors = N // VECTOR
    partials = []
    for t in range(min(block, vectors)):
        acc = None
        for v in range(t, vectors, block):
            part = _vector_sum(B, [rf[VECTOR * v + j] for j in range(VECTOR)])
            acc = part if acc is None else B.call(P.F32Add, acc, part)
        partials.append(acc)
    scale = _arg_scale(B, S, _cub(B, partials))
    return tuple_of(_out(B, r, w, scale), r)


WARP = 32


@composite("RMSNormPairSumCuda", 1, [], lambda S: ((("p", Array(2, F32)),), F32), conformance="exact-model-tested")
def RMSNormPairSumCuda(B, S, p):
    """``p[0] + p[1]``, one ``add.f32``."""
    return B.call(P.F32Add, p[0], p[1])


@composite("RMSNormSumStepCuda", 1, [], lambda S: ((("acc", F32), ("x", F32)), Tuple(F32, F32)),
           conformance="exact-model-tested")
def RMSNormSumStepCuda(B, S, acc, x):
    """``acc + x``, carried and returned."""
    s = B.call(P.F32Add, acc, x)
    return tuple_of(s, s)


@composite("RMSNormSumInOrderCuda", 1, ["K"], lambda S: ((("xs", Array(S.K, F32)),), F32),
           conformance="exact-model-tested")
def RMSNormSumInOrderCuda(B, S, xs):
    """``((xs[0] + xs[1]) + xs[2]) + ...``: K >= 2 values added left to right, a scan of K - 1 adds."""
    K = int(S.K)
    if K < 2:
        raise ValueError(f"K={K}: a sum in order adds at least two values")
    return B.scan(RMSNormSumStepCuda, xs[0], xs=(xs[1:K],))[0]


def _pair_sums(B, a):
    """``a[2i] + a[2i + 1]`` for every i, one batch."""
    return B.batch(RMSNormPairSumCuda, a.reshape(Array(len(a) // 2, Array(2, F32))), axes=(0,))


def _sums_in_order(B, rows, k: int):
    """Each row of ``rows`` (``Array<n, Array<k, F32>>``) added left to right, one batch; a row of one is itself."""
    if k == 1:
        return rows.reshape(Array(len(rows), F32))
    return B.batch(bind(RMSNormSumInOrderCuda, K=k), rows, axes=(0,))


def _lane_partials(B, vec, block: int):
    """Thread t's vector partials t, t + BLOCK, ... added left to right: the first ``q`` of every thread's as one batch
    of sums in order, then the ``rem`` threads holding one more vector add it, one batch."""
    V = len(vec)
    if V <= block:
        return vec
    q, rem = divmod(V, block)
    lanes = _sums_in_order(B, Coll(Array(block, Array(q, F32)), strided_view(vec, 0, q, 1, block, block * q)), q)
    if not rem:
        return lanes
    head = B.batch(P.F32Add, lanes[0:rem], vec[q * block:V], axes=(0, 0))
    return coll_concat(Array(block, F32), [head, lanes[rem:block]])


def _cub_levels(B, partials):
    """`b1._cub_block_reduce_sum` with each full warp's shuffle level one batch: at offset 1, 2, ..., 16 lane i adds
    lane i + offset for every i a multiple of twice the offset, so a level is the previous level's pair sums. A last
    partial warp is reduced as CUB reduces it; the warp aggregates are then added in order from warp 0."""
    T = len(partials)
    full = T - T % WARP
    warps = []
    if full:
        level = partials[0:full]
        for _ in range(WARP.bit_length() - 1):
            level = _pair_sums(B, level)
        warps.append(level)
    if T % WARP:
        warps.append(array_of([b1._cub_block_reduce_sum(B, list(partials[full:T]))]))
    n = full // WARP + (T % WARP > 0)
    aggregates = coll_concat(Array(n, F32), warps)
    return aggregates[0] if n == 1 else B.call(bind(RMSNormSumInOrderCuda, K=n), aggregates)


@composite(
    "RMSNormFusedCudaVector8",
    3,
    ["N", "EPS", "BLOCK"],
    lambda S: (
        (("x", Array(S.N, BF16)), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16))),
        Tuple(Array(S.N, BF16), Array(S.N, BF16)),
    ),
    conformance="exact-model-tested",
)
def RMSNormFusedCudaVector8(B, S, x, res, w):
    """The served residual-add CUDA RMSNorm vector path: ``RMSNormFusedCudaVector8_v1`` in v1's f32 order, its sum of squares
    batched rather than unrolled.

    The squares, their adjacent pair sums and each vector's four pair sums added left to right are one batch each; each
    thread's vector partials added left to right are one batch of per-lane scans; each of CUB's shuffle levels over the
    full warps is one batch of pair sums, and the warp aggregates are added in order. Every add and multiply is v1's,
    on v1's operands, so each output word is v1's.
    """
    N, block = _check(S, fused=True)
    r = B.batch(P.Bf16AddF2fp, x, res, axes=(0, 0))
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    vectors = N // VECTOR
    squares = B.batch(P.F32Mul, rf, rf, axes=(0, 0))
    pairs = _pair_sums(B, squares)
    vec = _sums_in_order(B, pairs.reshape(Array(vectors, Array(VECTOR // 2, F32))), VECTOR // 2)
    scale = _arg_scale(B, S, _cub_levels(B, _lane_partials(B, vec, block)))
    return tuple_of(_out(B, r, w, scale), r)


from verity.ml.definitions.registry import boolean_norms as _boolean_norms  # noqa: E402

_boolean_norms._set_cuda_vector8_words(RMSNormCudaVector8, RMSNormFusedCudaVector8V1)


__all__ = [
    "LARGE_BLOCK",
    "RMSNormCudaVector8",
    "RMSNormFusedCudaVector8",
    "RMSNormFusedCudaVector8V1",
    "RMSNormPairSumCuda",
    "RMSNormSumInOrderCuda",
    "RMSNormSumStepCuda",
    "SMALL_BLOCK",
    "SMALL_BLOCK_FROM_TOKENS",
    "VECTOR",
    "WARP",
    "launch_block",
]
