"""vLLM d9105ea80's non-batch-invariant CUDA RMSNorm kernels for contiguous BF16 rows.

The public statics record the actual launch block, because both kernels switch from their
large block to 256 threads when ``num_tokens >= 256``.  ``RMSNormCudaVector8_v1`` is
``rms_norm_kernel<c10::BFloat16, 8, 2, true>``.  ``RMSNormFusedCudaVector8_v1`` is
``fused_add_rms_norm_kernel<c10::BFloat16, 8, true>`` and returns both in-place outputs.
"""
from __future__ import annotations

from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.prims import const
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16

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
def RMSNormFusedCudaVector8(B, S, x, res, w):
    """The served residual-add CUDA RMSNorm vector path.

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


__all__ = [
    "LARGE_BLOCK",
    "RMSNormCudaVector8",
    "RMSNormFusedCudaVector8",
    "SMALL_BLOCK",
    "SMALL_BLOCK_FROM_TOKENS",
    "VECTOR",
    "launch_block",
]
