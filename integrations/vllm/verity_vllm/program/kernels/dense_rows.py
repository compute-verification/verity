"""Kernels for host evaluation (`acquire.sources.call_boundary_plan`) of the dense chain of `registry.dense` -- GemmaRMSNorm's unfused
ATen chain and the logits soft cap -- and the Ampere k16 GEMM chain of a large Gemm row (the logits `lm_head`).

The chain's kernels are numpy, elementwise over a leading batch axis (one instance is a batch of one), registered with core
`verity.ml.circuits.evaluation` as kernel "dense" through `kernel_registry.register`, per instance and as a batch (`_claim_rows`), and
self-checked against the reference by tests/program/test_kernel_self_check.py.  Each is bit-equal to its Definition on the instances
it claims: the chain's f32 arithmetic is `F32Add_v1` / `F32Mul_v1`, which are numpy float32 RN without FTZ; `RsqrtF32` is
`derived_rows.rsqrt_approx`, `RsqrtApprox_v1`'s twin; `MeanTriton`'s reduction follows its body's register, warp-butterfly and warp
order; `Bf16Tanh` is the table of its body over the 65,536 bf16 words.  An instance with a non-finite operand or result word is
declined, and the reference computes it.

`gemm_ampere_row` is `DotBf16_v1`'s ascending chain of `tc_dot16` steps (two groups of eight products plus the running accumulator,
25-bit adder, exponent floor -132) in float64: a bf16 x bf16 product is exact there; a group's terms align as
`floor(|t| / U)` with `U = 2^(max nominal exponent - 24)`; their sum is an exact integer below 2^31; the step's word is that sum times
`U` truncated to 24 significant bits.  A coordinate whose chain meets a non-finite operand, an f32-subnormal or an overflowing step
result is recomputed by `derived_rows.dot_bf16`, so every word equals the int64 twin's.  `gemm_row` is the same chain under any
k16 step of `STEPS` (`Gemm_v2`'s DOT; Hopper's is one group of sixteen products, a 26-bit adder and floor -133, so `U = 2^(e - 25)`).
`chain_rows` runs that chain for every row of a step against one W: `cpp/k16_chain.cpp` computes it in integers (the aligned terms
are shifts of the 16-bit significand products, the same integers), reading each W row once for all the rows, with the GIL released;
where it cannot be built, `_chain` computes it in numpy over a W side (values and exponents, transposed) prepared once per weight
buffer (keyed by its address, shape and strides: a Commit's weight words do not change in place).  Either hands back the same
coordinates.  Coordinates are split over up to `threads()` threads (`MAX_THREADS`, or the cap `VERITY_DENSE_THREADS` sets when this
module is imported, `use_threads`); the words do not depend on the split or the path.
`gemm_dot_rows` is `Gemm_v2`'s batch form under kernel "rows": a step's logits rows against the one W by one `gemm_rows`.
"""
from __future__ import annotations

import ctypes
import functools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from verity_vllm.program.kernels import derived_rows as D
from verity_vllm.program.kernels.kernel_registry import Declined, register

KERNEL = "dense"
DOMAIN = "instances whose operand and result words are all finite (the reference computes the others)"
#: a Gemm row reading at least this many weight words takes the float64 chain (the logits `lm_head`); smaller rows keep the int64 twin
FAST_GEMM_WORDS = 1 << 24
MAX_THREADS = 8
_REDO_CHUNK = 4096                                  # coordinates per int64 recompute (memory-bounded, as `rows.GEMM_CHUNK`)
#: DOT primitive id -> its k16 step as (products per group, adder width, exponent floor), the parameters of `derived_rows.TC_DOT16_BY_PRIM`
AMPERE = (8, D._TC_WIDTH, D._TC_FLOOR)
STEPS = {"AmpereBF16TcDot16_v2": AMPERE, "HopperBF16WgmmaDot16_v1": (D._HOPPER_TC_GROUPS[0], D._HOPPER_TC_WIDTH, D._HOPPER_TC_FLOOR)}
_TRUNC24 = np.int64(~((1 << 29) - 1))


def _f32(u: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(u, dtype=np.uint32).view(np.float32)


def _widen(x: np.ndarray) -> np.ndarray:
    return D.bf16_to_f32_words(np.asarray(x, dtype=np.uint16))


def _finite32(u: np.ndarray) -> bool:
    return not bool(((np.asarray(u, dtype=np.uint32) & 0x7F800000) == 0x7F800000).any())


def _finite16(w: np.ndarray) -> bool:
    return not bool(((np.asarray(w, dtype=np.uint16) & 0x7F80) == 0x7F80).any())


def _claim(fn):
    """`fn(*args, **statics)` on finite words only: a non-finite operand or result word declines the instance."""
    @functools.wraps(fn)
    def kernel(*args: np.ndarray, **statics: Any) -> np.ndarray:
        for a in args:
            if not (_finite16(a) if a.dtype == np.uint16 else _finite32(a)):
                raise Declined("a non-finite operand word")
        out = fn(*args, **statics)
        if not (_finite16(out) if out.dtype == np.uint16 else _finite32(out)):
            raise Declined("a non-finite result word")
        return out
    return kernel


def _nonfinite_rows(a: np.ndarray) -> np.ndarray:
    """[n] True where instance i of the batch `a` has a non-finite word (a row broadcast over the batch is read once)."""
    a = np.asarray(a)
    if len(a) > 1 and a.strides[0] == 0:
        return np.broadcast_to(_nonfinite_rows(a[:1]), a.shape[:1])
    m, e = (0x7F80, 0x7F80) if a.dtype == np.uint16 else (0x7F800000, 0x7F800000)
    return ((a.reshape(len(a), -1) & a.dtype.type(m)) == e).any(axis=1)


def _claim_rows(fn):
    """`fn` over a batch (its leading axis), the core kernel `kernel_registry.register` takes as `batch`: (words, declined), the
    instances `_claim(fn)` declines declined.  Every function here is elementwise over that axis, so each instance's words are
    the instance form's; a declined instance's operands are read as zeros, and its words are the reference's."""
    def kernel(*args: np.ndarray, **statics: Any) -> tuple[np.ndarray, np.ndarray]:
        n = len(args[0])
        declined = np.zeros(n, dtype=bool)
        for a in args:
            declined |= _nonfinite_rows(a)
        if declined.any():
            args = tuple(np.where(declined.reshape((n,) + (1,) * (a.ndim - 1)), a.dtype.type(0), a) for a in args)
        out = fn(*args, **statics).reshape(n, -1)
        return out, declined | _nonfinite_rows(out)
    return kernel


def _f32_word(c: Any) -> np.ndarray:
    return np.float32(float(c)).reshape(1).view(np.uint32)


# ---- the norm chain -----------------------------------------------------------------------------------------------------

def square_bf16(x, **_s):
    xf = _widen(x)
    return D.mul_rn(xf, xf)


def square_f32(x, **_s):
    return D.mul_rn(x, x)


def add_widened_bf16(x, r, **_s):
    return D.add_rn(_widen(x), _widen(r))


def add_scalar_f32(x, *, C, **_s):
    return D.add_rn(np.asarray(x, dtype=np.uint32), _f32_word(C))


def add_scalar_bf16(w, *, C, **_s):
    return D.add_rn(_widen(w), _f32_word(C))


def rsqrt_f32(x, **_s):
    return D.rsqrt_approx(x)


def scale_row_bf16(x, s, **_s):
    return D.mul_rn(_widen(x), np.asarray(s, dtype=np.uint32))


def scale_row_f32(x, s, **_s):
    return D.mul_rn(np.asarray(x, dtype=np.uint32), np.asarray(s, dtype=np.uint32))


def mul_vec_f32(x, w, **_s):
    return D.mul_rn(np.asarray(x, dtype=np.uint32), np.asarray(w, dtype=np.uint32))


@functools.lru_cache(maxsize=None)
def _mean_layout(N: int) -> np.ndarray:
    """[blocks, 128 threads, registers] element index per register of `MeanTriton`'s body (N for a masked lane)."""
    from verity_vllm.program.registry import dense
    spt = dense.size_per_thread(N)
    blocks = [[[i if i < N else N for i in dense.thread_elements(c0, t, spt)] for t in range(dense.THREADS)]
              for c0 in range(0, N, dense.BLOCK_SIZE)]
    return np.asarray(blocks, dtype=np.int64)


def mean_triton(x, *, N, **_s):
    """`MeanTriton_v1{N}`: each thread's registers added in order, the warps' xor-butterfly 16, 8, 4, 2, 1, the CTA's
    `(w0 + w2) + (w1 + w3)`, the blocks' sums added in order from +0, then `div.full.f32` by N.  `x` is [..., N], one mean per row."""
    N = int(N)
    x = np.asarray(x, dtype=np.uint32)
    rows = x.reshape(-1, N)
    xs = np.concatenate([rows, np.zeros((len(rows), 1), dtype=np.uint32)], axis=1)   # the masked lanes' +0.0
    acc = np.zeros((len(rows), 1), dtype=np.uint32)
    for idx in _mean_layout(N):
        regs = xs[:, idx]                                                    # [rows, 128 threads, registers]
        lane = regs[..., 0]
        for j in range(1, regs.shape[-1]):
            lane = D.add_rn(lane, regs[..., j])
        cur = lane.reshape(-1, 4, 32)
        for m in (16, 8, 4, 2, 1):
            cur = D.add_rn(cur, cur[..., np.arange(32) ^ m])
        w = cur[..., 0]
        acc = D.add_rn(acc, D.add_rn(D.add_rn(w[:, 0:1], w[:, 2:3]), D.add_rn(w[:, 1:2], w[:, 3:4])))
    nf = _f32_word(N)
    return D.mul_rn(D.div_full_scale_a(acc, nf), D.div_full_rcp(nf)).reshape(*x.shape[:-1], 1)


# ---- the logits soft cap ---------------------------------------------------------------------------------------------------

def bf16_div_scalar(x, *, C, **_s):
    from verity_vllm.program.registry.dense import reciprocal_bits
    return D.f2fp_bf16(D.mul_rn(_widen(x), np.uint32(reciprocal_bits(float(C)))))


def bf16_mul_scalar(x, *, C, **_s):
    return D.f2fp_bf16(D.mul_rn(_widen(x), np.float32(float(C)).view(np.uint32)))


@functools.lru_cache(maxsize=1)
def _tanh_table() -> np.ndarray:
    from verity_vllm.program.registry.dense import tanh_f32_bits
    words = np.arange(1 << 16, dtype=np.uint32) << 16
    return D.f2fp_bf16(np.asarray([tanh_f32_bits(int(w)) for w in words], dtype=np.uint32))


def bf16_tanh(x, **_s):
    return _tanh_table()[np.asarray(x, dtype=np.uint16)]


# ---- the k16 GEMM chain ---------------------------------------------------------------------------------------------------------

class _Prepared:
    """One weight buffer's W side: f32 values and nominal exponent fields (`max(field, 1)`), both [K, N], and its non-finite
    coordinates.  Held with a reference to the buffer, so the address it is keyed by cannot be reused while it is cached."""

    def __init__(self, w: np.ndarray) -> None:
        self.src = w
        N, K = w.shape
        self.f = np.empty((K, N), dtype=np.float32)
        self.e = np.empty((K, N), dtype=np.uint8)
        f = self.f.view(np.uint32)

        def block(k0: int) -> None:                 # 16 columns at a time: no full-size transposed temporaries
            wT = w[:, k0:k0 + 16].T
            f[k0:k0 + 16] = wT
            f[k0:k0 + 16] <<= 16
            self.e[k0:k0 + 16] = ((wT >> 7) & 0xFF).astype(np.uint8)

        with ThreadPoolExecutor(threads()) as ex:
            list(ex.map(block, range(0, K, 16)))
        self.bad = (self.e == 0xFF).any(axis=0)
        np.maximum(self.e, 1, out=self.e)

    def key(self) -> tuple:
        return _key(self.src)


def _key(w: np.ndarray) -> tuple:
    return (w.__array_interface__["data"][0], w.shape, w.strides, w.dtype.str)


_PREPARED: list[_Prepared] = []


def prepared(w: np.ndarray) -> _Prepared:
    """The W side of `w` (N, K), prepared once per weight buffer (the most recent one is kept)."""
    if _PREPARED and _PREPARED[0].key() == _key(w):
        return _PREPARED[0]
    _PREPARED[:] = [_Prepared(w)]
    return _PREPARED[0]


def _chain(x: np.ndarray, p: _Prepared, lo: int, hi: int, step: tuple[int, int, int] = AMPERE) -> tuple[np.ndarray, np.ndarray]:
    """The k16 chain of coordinates lo..hi under `step` = (products per group, adder width, exponent floor): (f32 words, coordinates
    the float64 path does not claim)."""
    K = x.shape[0]
    G, width, floor = step
    xf = _f32(_widen(x)).astype(np.float64)
    xe = np.maximum((x >> 7) & 0xFF, 1).astype(np.int16) - 254
    a = np.zeros(hi - lo)
    bad = np.zeros(hi - lo, dtype=bool)
    with np.errstate(invalid="ignore", over="ignore"):    # a non-finite or overflowing coordinate is handed back (`bad`)
        for g in range(0, K, G):                    # each group of G products is joined by the running accumulator
            v = p.f[g:g + G, lo:hi] * xf[g:g + G, None]
            pe = np.where(v != 0, p.e[g:g + G, lo:hi] + xe[g:g + G, None], np.int16(-30000)).max(axis=0)
            ab = np.abs(a)
            a_exp = np.maximum(((ab.view(np.int64) >> 52) & 0x7FF) - 1023, -126)
            me = np.maximum(np.maximum(pe, floor), np.where(a != 0, a_exp, -(1 << 20)))
            inv = np.ldexp(1.0, (width - 1 - me).astype(np.int32))
            tot = np.copysign(np.floor(np.abs(v) * inv), v).sum(axis=0) + np.copysign(np.floor(ab * inv), a)
            y = np.abs(tot) * np.ldexp(1.0, (me - width + 1).astype(np.int32))
            bad |= (y >= 2.0 ** 128) | ((y != 0) & (y < 2.0 ** -126))
            a = np.where(tot == 0, 0.0, np.copysign((y.view(np.int64) & _TRUNC24).view(np.float64), tot))
        return a.astype(np.float32).view(np.uint32), bad


_THREADS: list[int] = [MAX_THREADS]


def use_threads(n: int | str | None) -> None:
    """Cap the chain's threads at `n` (at import, `VERITY_DENSE_THREADS`); None, or a value that is not a positive integer, is
    `MAX_THREADS`."""
    try:
        k = MAX_THREADS if n is None else int(n)
    except ValueError:
        k = MAX_THREADS
    _THREADS[0] = k if k > 0 else MAX_THREADS


use_threads(os.environ.get("VERITY_DENSE_THREADS"))


def threads() -> int:
    """The chain's threads: the cap (`use_threads`, else `MAX_THREADS`), at most this process's CPUs."""
    return max(1, min(len(os.sched_getaffinity(0)), _THREADS[0]))


def dot_ampere(x: np.ndarray, w: np.ndarray, n_threads: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`DotBf16_v1{K}` of the row `x` (K,) against every row of `w` (N, K), bf16 words: (f32 words, non-finite coordinates), equal to
    `derived_rows.dot_bf16` word for word."""
    return dot_chain(x, w, "AmpereBF16TcDot16_v2", n_threads)


def dot_chain(x: np.ndarray, w: np.ndarray, dot_id: str, n_threads: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`DotBf16_v2{K, DOT=dot_id}` of the row `x` (K,) against every row of `w` (N, K), bf16 words: (f32 words, non-finite
    coordinates), equal to `derived_rows.dot_bf16_for(dot_id)` word for word."""
    out, nonfin = chain_rows(np.asarray(x, dtype=np.uint16)[None, :], w, dot_id, n_threads)
    return out[0], nonfin[0]


_K16_SRC = Path(__file__).resolve().parent / "cpp" / "k16_chain.cpp"


@functools.lru_cache(maxsize=1)
def k16_lib() -> ctypes.CDLL | None:
    """`cpp/k16_chain.cpp`, built by `_jit` on first use; None where it cannot be built (`_chain` then computes the same words)."""
    from verity_vllm.program.kernels import _jit
    try:
        lib = ctypes.CDLL(str(_jit.atomic_build(_jit.lib_path("libk16_chain.so"), [_K16_SRC], lambda out: _jit.cxx_compile(out, _K16_SRC))))
    except (OSError, subprocess.CalledProcessError):
        return None
    lib.k16_chain.restype = ctypes.c_int
    lib.k16_chain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    return lib


def _claimed(X: np.ndarray, w: np.ndarray, step: tuple[int, int, int], n: int) -> tuple[np.ndarray, np.ndarray]:
    """The k16 chain of every row of `X` against every row of `w` over `n` coordinate spans: (f32 words [M, N], coordinates handed
    back [M, N]), by `k16_lib` where it is built, else by `_chain`."""
    (M, K), N = X.shape, w.shape[0]
    span = -(-N // n)
    lib = k16_lib()
    if lib is None:
        p = prepared(w)
        out, back = np.empty((M, N), dtype=np.uint32), np.empty((M, N), dtype=bool)

        def job(lo: int) -> None:
            for i in range(M):
                out[i, lo:lo + span], back[i, lo:lo + span] = _chain(X[i], p, lo, min(lo + span, N), step)
    else:
        w = np.ascontiguousarray(w, dtype=np.uint16)
        out, back = np.empty((M, N), dtype=np.uint32), np.zeros((M, N), dtype=bool)

        def job(lo: int) -> None:
            if lib.k16_chain(X.ctypes.data, M, w[lo:].ctypes.data, min(span, N - lo), K, *step, out[:, lo:].ctypes.data,
                             back[:, lo:].ctypes.data, N):
                raise ValueError(f"k16_chain: no {step[0]}-product groups")

    with ThreadPoolExecutor(n) as ex:
        list(ex.map(job, range(0, N, span)))
    if lib is None:
        back |= p.bad
    return out, back


def chain_rows(X: np.ndarray, w: np.ndarray, dot_id: str, n_threads: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`DotBf16_v2{K, DOT=dot_id}` of every row of `X` (M, K) against every row of `w` (N, K), bf16 words: (f32 words [M, N],
    non-finite coordinates [M, N]), row i equal to `derived_rows.dot_bf16_for(dot_id)` of `X[i]` word for word.  A coordinate the
    chain hands back, and every coordinate of a row with a non-finite word, is the int64 twin's."""
    step, redo_dot = STEPS[dot_id], D.dot_bf16_for(dot_id)
    X = np.ascontiguousarray(X, dtype=np.uint16)
    if X.ndim != 2 or X.shape[1] % 16 or w.shape[1] != X.shape[1]:
        raise ValueError(f"a k16 chain needs K a multiple of 16 and W [N, K] (X {X.shape}, W {w.shape})")
    out, redo = _claimed(X, w, step, n_threads or threads())
    redo |= (((X >> 7) & 0xFF) == 0xFF).any(axis=1, keepdims=True)
    nonfin = np.zeros(out.shape, dtype=bool)
    for r in range(len(X)):
        idx = np.flatnonzero(redo[r])
        for i in range(0, idx.size, _REDO_CHUNK):
            at = idx[i:i + _REDO_CHUNK]
            out[r, at], nonfin[r, at] = redo_dot(X[r][None, :], w[at])
    return out, nonfin


def gemm_ampere_row(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`Gemm_v1{K,N}` of one token row by the k16 chain: (bf16 words [N], non-finite coordinates)."""
    return gemm_row(x, w, "AmpereBF16TcDot16_v2")


def gemm_row(x: np.ndarray, w: np.ndarray, dot_id: str) -> tuple[np.ndarray, np.ndarray]:
    """`Gemm_v2{K,N,DOT=dot_id}` of one token row by the k16 chain: (bf16 words [N], non-finite coordinates)."""
    out, nonfin = gemm_rows(np.asarray(x, dtype=np.uint16)[None, :], w, dot_id)
    return out[0], nonfin[0]


def gemm_rows(X: np.ndarray, w: np.ndarray, dot_id: str) -> tuple[np.ndarray, np.ndarray]:
    """`Gemm_v2{K,N,DOT=dot_id}` of every token row of `X` (M, K) against one W: (bf16 words [M, N], non-finite coordinates)."""
    acc, nonfin = chain_rows(X, w, dot_id)
    return D.f2fp_bf16(acc).astype(np.uint16), nonfin


def gemm_dot_rows(dot_of: Callable[[dict], str], x: np.ndarray, w: np.ndarray, **statics: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """`Gemm_v2`'s batch form for `rows` (`kernel_registry.register`), `dot_of(statics)` its DOT primitive id: the token rows of a
    batch that reads one W past `FAST_GEMM_WORDS` words (a call boundary's logits rows, the weight field broadcast over them) by
    one `gemm_rows`, which reads W once for all of them; None for any other batch.  Declined where the row kernel declines: a
    non-finite bf16 operand word (the int64 twin flags every coordinate a non-finite W word reaches, whatever the other operand),
    accumulator or result word."""
    K, N = int(statics["K"]), int(statics["N"])
    try:
        dot_id = dot_of(statics)
    except (Declined, KeyError, ValueError):
        return None
    if not (len(x) == 1 or w.strides[0] == 0) or K * N < FAST_GEMM_WORDS or dot_id not in STEPS:
        return None
    X = np.asarray(x, dtype=np.uint16).reshape(len(x), K)
    out, nonfin = gemm_rows(X, np.asarray(w[0], dtype=np.uint16).reshape(N, K), dot_id)
    declined = nonfin.any(axis=1) | ((X & 0x7F80) == 0x7F80).any(axis=1) | ((out & 0x7F80) == 0x7F80).any(axis=1)
    return out.astype(np.uint64), declined


# ---- registration --------------------------------------------------------------------------------------------------------

def _targets():
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dense
    return {
        "SquareBf16_v1": [bind(dense.SquareBf16, N=16)], "SquareF32_v1": [bind(dense.SquareF32, N=16)],
        "MeanTriton_v1": [bind(dense.MeanTriton, N=2304), bind(dense.MeanTriton, N=20)],
        "AddWidenedBf16_v1": [bind(dense.AddWidenedBf16, N=16)], "ScaleRowBf16_v1": [bind(dense.ScaleRowBf16, N=16)],
        "AddScalarF32_v1": [bind(dense.AddScalarF32, N=1, C=1e-6), bind(dense.AddScalarF32, N=16, C=1.0)],
        "AddScalarBf16_v1": [bind(dense.AddScalarBf16, N=16, C=1.0)],
        "RsqrtF32_v1": [bind(dense.RsqrtF32, N=1), bind(dense.RsqrtF32, N=16)],
        "ScaleRowF32_v1": [bind(dense.ScaleRowF32, N=16)], "MulVecF32_v1": [bind(dense.MulVecF32, N=16)],
        "Bf16DivScalar_v1": [bind(dense.Bf16DivScalar, N=16, C=30.0)], "Bf16MulScalar_v1": [bind(dense.Bf16MulScalar, N=16, C=30.0)],
        "Bf16Tanh_v1": [bind(dense.Bf16Tanh, N=16)],
    }


KERNELS = {"SquareBf16_v1": square_bf16, "SquareF32_v1": square_f32, "MeanTriton_v1": mean_triton, "AddWidenedBf16_v1": add_widened_bf16,
           "AddScalarF32_v1": add_scalar_f32, "AddScalarBf16_v1": add_scalar_bf16, "RsqrtF32_v1": rsqrt_f32,
           "ScaleRowBf16_v1": scale_row_bf16, "ScaleRowF32_v1": scale_row_f32, "MulVecF32_v1": mul_vec_f32,
           "Bf16DivScalar_v1": bf16_div_scalar, "Bf16MulScalar_v1": bf16_mul_scalar, "Bf16Tanh_v1": bf16_tanh}


_EDGE = np.array([0x00000000, 0x80000000, 0x00010000, 0x80010000, 0x007F0000, 0x3F800000, 0xBF800000], dtype=np.uint32)


def _sample(rng: np.random.Generator, n: int, params, **_statics) -> list[np.ndarray]:
    """Self-check words: normals at a random scale per instance (2^-30 .. 2^30), one in sixteen a signed zero, a subnormal or +-1."""
    from verity.ml.circuits.evaluation.batch import type_shape
    out = []
    for pt in params:
        shape = (n, *type_shape(pt))
        v = rng.standard_normal(shape) * 2.0 ** rng.integers(-30, 30, (n,) + (1,) * (len(shape) - 1))
        u = np.where(rng.random(shape) < 1 / 16, rng.choice(_EDGE, shape), v.astype(np.float32).view(np.uint32))
        out.append(u if set(pt.leaf_widths()) == {32} else (u >> 16).astype(np.uint16))
    return out


def _sample_positive(rng: np.random.Generator, n: int, params, **statics) -> list[np.ndarray]:
    """`_sample`'s words with the sign cleared: a negative rsqrt operand is NaN, so signed words would decline every wide instance."""
    return [u & u.dtype.type(0x7FFFFFFF if u.dtype == np.uint32 else 0x7FFF) for u in _sample(rng, n, params, **statics)]


_SAMPLES = {"RsqrtF32_v1": _sample_positive}


def _register() -> None:
    for family, fn in KERNELS.items():
        register(family, _claim(fn), kernel=KERNEL, domain=DOMAIN, sample=_SAMPLES.get(family, _sample),
                 check_targets=lambda rng, f=family: _targets()[f], batch=_claim_rows(fn))


_register()

__all__ = ["KERNEL", "KERNELS", "FAST_GEMM_WORDS", "MAX_THREADS", "STEPS", "chain_rows", "dot_ampere", "dot_chain", "gemm_ampere_row",
           "gemm_dot_rows", "gemm_row", "gemm_rows", "k16_lib", "prepared", "threads", "use_threads"]
