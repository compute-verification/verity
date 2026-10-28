"""Independent CPU models of vLLM's two residual-free RMSNorm kernels: the batch-invariant Triton
``_rms_norm_kernel`` (``kernel="triton"``) and the CUDA ``vllm::rms_norm_kernel<bf16, 8>`` behind
``torch.ops._C.rms_norm`` (``kernel="cuda"``).

The arithmetic lives in ``cpp/rms_triton_model.cpp``; this module builds/loads it, reconstructs the hardware
tables it needs (``MUFU.SQRT`` and ``MUFU.RSQ`` measured by ``cuda/rms_probe.cu``, ``MUFU.RCP`` measured by
``cuda/mufu_probe.cu``; all verified against the RTX 4090 on all 2^32 inputs) and exposes

* :func:`rms_norm_rows` -- the model: BF16 rows + BF16 weight + eps -> BF16 rows (and the fp32 scalars);
* :func:`gate` -- the measured domain (kernel specialisation, dtype, n_cols family, contiguity).

Kernel identities: ``vllm/model_executor/determinism/batch_invariant.py::
_rms_norm_kernel`` specialised ``BLOCK_SIZE=1024, HAS_WEIGHT=True``, ``num_warps=4``, Triton 3.7.1 for sm_89,
launched by ``rms_norm_batch_invariant`` for every RMSNorm call without a residual (``layers.0.input_layernorm``
in the pinned Qwen2.5-1.5B profile); ``csrc/libtorch_stable/layernorm_kernels.cu::vllm::rms_norm_kernel<c10::BFloat16,
8, 2, true>`` (not launched by the pinned engine; the third rms_norm variant).
"""

from __future__ import annotations

import ctypes
import hashlib
import lzma
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verity.ml.definitions.numerics import _jit

HERE = Path(__file__).resolve().parent
CPP_SRC = HERE / "cpp" / "rms_triton_model.cpp"
LIB = _jit.lib_path("librms_triton_model.so")          # build product (the _jit cache, outside the checkout), compiled on first use

TABLE_N = 1 << 23
SQRT_T0 = 0x3F800000  # sqrt(1.0) = 1.0
RCP_R0 = 0x3F800000  # rcp(1.0) = 1.0

KERNEL = {
    "source": "vllm/model_executor/determinism/batch_invariant.py::_rms_norm_kernel (vLLM 0.28.1rc1.dev472+gd9105ea80)",
    "specialisation": {"BLOCK_SIZE": 1024, "HAS_WEIGHT": True, "num_warps": 4, "num_stages": 3,
                       "int_args_divisible_by_16": ["input_row_stride", "output_row_stride", "n_cols"]},
    "launch": "grid = (n_rows,), one CTA of 128 threads per row, 8 columns per thread per 1024-column chunk",
    "scalar_ops": "div.full.f32 -> MUFU.RCP + FFMA/FMUL with range scaling; sqrt.approx.ftz.f32 -> MUFU.SQRT",
}

VARIANTS = {
    "model": 0,
    "no_eps_fusion": 1,
    "sequential_sum": 2,
    "ieee_sqrt": 4,
    "ieee_rcp_div": 8,
    "thread_order_x0_first": 16,
    "fp64_sum": 32,
    "out_xw_first": 64,
    "unfused_squares": 128,
}

CUDA_VARIANTS = {
    "model": 0,
    "butterfly_warp_tree": 1,
    "sequential_sum": 2,
    "ieee_rsqrt": 4,
    "div_full_mean": 8,
    "fp64_sum": 32,
    "out_xw_first": 64,
    "warp_aggregates_reversed": 128,
}

CUDA_KERNEL = {
    "source": "csrc/libtorch_stable/layernorm_kernels.cu::vllm::rms_norm_kernel<c10::BFloat16, VEC_SIZE=8, NUM_DIMS=2, HasWeight=true> (vLLM d9105ea8)",
    "entry": "torch.ops._C.rms_norm(out, x, w, eps)  -- not launched by the pinned profile's engine (every RMSNorm call carries a residual "
             "or goes to the Triton kernel); modelled as the third rms_norm variant of numerics.md",
    "launch": "grid = rows, block = min(n_cols / 8, 1024) threads (192 for n_cols 1536), one 8-vector per thread per iteration",
    "reduction": "FFMA chain per vector (element order), cub WarpReduce shfl_down tree (1,2,4,8,16), cub BlockReduce<1024> warp aggregates added sequentially",
    "scalar_ops": "IEEE div.rn.f32 (MUFU.RCP + FFMA refinement + FCHK), FADD eps, rsqrt.approx.f32 -> MUFU.RSQ with 2^24 / 2^12 range scaling",
}

_lib = None


def build(force: bool = False) -> Path:
    """Compile cpp/rms_triton_model.cpp; atomic publish + builder lock (``_jit``): concurrent first users never dlopen a partial file."""
    return _jit.atomic_build(LIB, [CPP_SRC], lambda out: _jit.cxx_compile(out, CPP_SRC, cwd=CPP_SRC.parent), force=force)


def _load() -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(build()))
    except OSError:  # a stale or foreign-platform .so (e.g. copied from another machine): rebuild here
        return ctypes.CDLL(str(build(force=True)))


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load()
        _lib.rms_triton_rows.restype = None
        _lib.rms_triton_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_float,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int]
        _lib.rms_cuda_rows.restype = None
        _lib.rms_cuda_rows.argtypes = _lib.rms_triton_rows.argtypes
        _lib.rms_mufu_sqrt_words.restype = None
        _lib.rms_mufu_sqrt_words.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
        _lib.rms_mufu_rsq_words.restype = None
        _lib.rms_mufu_rsq_words.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
        _lib.rms_rsqrt_words.restype = None
        _lib.rms_rsqrt_words.argtypes = _lib.rms_mufu_rsq_words.argtypes
        _lib.rms_div_full_words.restype = None
        _lib.rms_div_full_words.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
    return _lib


# ---- hardware tables ----------------------------------------------------------------------------


def _load_delta(p: Path, n: int, t0: int) -> tuple[np.ndarray, dict]:
    raw = p.read_bytes()
    delta = np.frombuffer(lzma.decompress(raw), dtype=np.int8)
    if delta.shape[0] != n - 1:
        raise ValueError(f"{p}: expected {n - 1} deltas, got {delta.shape[0]}")
    t = np.empty(n, dtype=np.int64)
    t[0] = t0
    np.cumsum(delta.astype(np.int64), out=t[1:])
    t[1:] += t0
    return t.astype(np.uint32), {"file": p.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


RSQ_T0 = 0x3F800000  # rsqrt(1.0) = 1.0


@dataclass(frozen=True)
class RmsTables:
    sqrt: np.ndarray  # (2^24,) u32: hw sqrt.approx.ftz of 1.m (first 2^23) and of 2*1.m (last 2^23)
    rcp: np.ndarray  # (2^23,) u32: hw rcp.approx.ftz of 1.m
    source: dict
    rsq: np.ndarray | None = None  # (2^24,) u32: hw rsqrt.approx.ftz of 1.m and of 2*1.m (MUFU.RSQ, the CUDA kernel)

    @staticmethod
    def load(directory: str | Path) -> "RmsTables":
        d = Path(directory)
        sq, s1 = _load_delta(d / "mufu_sqrt_delta_int8.xz", 2 * TABLE_N, SQRT_T0)
        rc, s2 = _load_delta(d / "mufu_rcp_delta_int8.xz", TABLE_N, RCP_R0)
        src = {"sqrt": s1, "rcp": s2}
        rsq = None
        if (d / "mufu_rsq_delta_int8.xz").exists():
            rsq, s3 = _load_delta(d / "mufu_rsq_delta_int8.xz", 2 * TABLE_N, RSQ_T0)
            src["rsq"] = s3
        return RmsTables(sq, rc, src, rsq)

    @staticmethod
    def encode(table: np.ndarray, t0: int) -> bytes:
        t = table.astype(np.int64)
        if int(t[0]) != t0:
            raise ValueError(f"table[0] = {t[0]:#x} != {t0:#x}")
        delta = np.diff(t)
        if delta.min() < -128 or delta.max() > 127:
            raise ValueError("deltas do not fit int8")
        return lzma.compress(delta.astype(np.int8).tobytes(), preset=9 | lzma.PRESET_EXTREME)


def mufu_sqrt(x: np.ndarray, tables: RmsTables) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    y = np.empty_like(x)
    lib().rms_mufu_sqrt_words(x.ctypes.data, y.ctypes.data, x.size, tables.sqrt.ctypes.data)
    return y.view(np.float32)


def mufu_rsq(x: np.ndarray, tables: RmsTables) -> np.ndarray:
    """The bare ``MUFU.RSQ`` (``rsqrt.approx.ftz.f32``): subnormal inputs flush to zero -> +inf."""
    if tables.rsq is None:
        raise ValueError("no MUFU.RSQ table loaded")
    x = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    y = np.empty_like(x)
    lib().rms_mufu_rsq_words(x.ctypes.data, y.ctypes.data, x.size, tables.rsq.ctypes.data)
    return y.view(np.float32)


def rsqrt_f32(x: np.ndarray, tables: RmsTables) -> np.ndarray:
    """``rsqrtf`` as the CUDA kernel compiles it (``rsqrt.approx.f32``, no fast-math): ``MUFU.RSQ`` with the
    subnormal range scaling (``|x| < 2^-126``: ``x *= 2^24`` before, ``r *= 2^12`` after)."""
    if tables.rsq is None:
        raise ValueError("no MUFU.RSQ table loaded")
    x = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    y = np.empty_like(x)
    lib().rms_rsqrt_words(x.ctypes.data, y.ctypes.data, x.size, tables.rsq.ctypes.data)
    return y.view(np.float32)


def div_full(a: np.ndarray, b: np.ndarray, tables: RmsTables) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    b = np.ascontiguousarray(np.broadcast_to(np.asarray(b, dtype=np.float32), a.shape)).view(np.uint32)
    y = np.empty_like(a)
    lib().rms_div_full_words(a.ctypes.data, b.ctypes.data, y.ctypes.data, a.size, tables.rcp.ctypes.data)
    return y.view(np.float32)


# ---- the model ---------------------------------------------------------------------------------


def rms_norm_rows(x: np.ndarray, w: np.ndarray, eps: float, tables: RmsTables, variant: int = 0,
                  want_scalars: bool = False, want_pre: bool = False, kernel: str = "triton"):
    """``x``: (rows, n_cols) BF16 words (uint16); ``w``: (n_cols,) BF16 words; returns (rows, n_cols) BF16 words,
    followed (if requested) by the per-row fp32 scalars (triton: ``[sum_sq, mean_eps, rms, inv]``; cuda:
    ``[sum_sq, arg, mean, s]``) and by the fp32 values just before the BF16 output cast.
    ``kernel``: ``"triton"`` (the batch-invariant ``_rms_norm_kernel``) or ``"cuda"`` (``vllm::rms_norm_kernel<bf16, 8>``)."""
    x = np.ascontiguousarray(x, dtype=np.uint16)
    w = np.ascontiguousarray(w, dtype=np.uint16)
    if x.ndim != 2 or w.ndim != 1 or x.shape[1] != w.shape[0]:
        raise ValueError(f"shape mismatch: x {x.shape}, w {w.shape}")
    rows, n = x.shape
    out = np.empty_like(x)
    dbg = np.empty((rows, 4), dtype=np.float32) if want_scalars else None
    pre = np.empty((rows, n), dtype=np.float32) if want_pre else None
    if kernel == "triton":
        fn, tab2 = lib().rms_triton_rows, tables.sqrt
    elif kernel == "cuda":
        if tables.rsq is None:
            raise ValueError("the CUDA kernel model needs the MUFU.RSQ table (mufu_rsq_delta_int8.xz)")
        if n % 8 != 0:
            raise ValueError(f"n_cols={n}: only the VEC_SIZE=8 instantiation (n_cols % 8 == 0) is modelled")
        fn, tab2 = lib().rms_cuda_rows, tables.rsq
    else:
        raise ValueError(kernel)
    fn(x.ctypes.data, w.ctypes.data, rows, n, ctypes.c_float(float(eps)),
       tables.rcp.ctypes.data, tab2.ctypes.data, out.ctypes.data,
       dbg.ctypes.data if dbg is not None else None, pre.ctypes.data if pre is not None else None, int(variant))
    res = [out]
    if want_scalars:
        res.append(dbg)
    if want_pre:
        res.append(pre)
    return res[0] if len(res) == 1 else tuple(res)


class RmsUnresolved(Exception):
    """The call is outside the measured domain of the model."""


# n_cols values at which the production kernels were run against the model (natural: 1536; stress: the rest);
# the same family for both kernels (rms_stress.py --kernel triton / cuda)
MEASURED_N_COLS = (16, 64, 576, 1024, 1536, 2048, 3072, 4096)


def gate(n_cols: int, dtype: str, has_weight: bool, contiguous: bool, eps: float,
         arch: str | None = None, measured_n_cols=MEASURED_N_COLS, kernel: str = "triton") -> None:
    """Raise :class:`RmsUnresolved` unless the call is inside the measured domain: sm_89 tables, BF16 in/out,
    weighted (Triton ``HAS_WEIGHT=True`` specialisation / CUDA ``HasWeight=true`` instantiation), contiguous rows
    (row stride == n_cols; divisible by 16 for the Triton specialisation, by 8 for the CUDA ``VEC_SIZE=8``
    instantiation), ``n_cols`` in a measured family, finite eps."""
    if arch is not None and arch != "sm_89":
        raise RmsUnresolved(f"MUFU tables were measured on sm_89, not {arch}")
    if dtype not in ("torch.bfloat16", "bfloat16", "bf16"):
        raise RmsUnresolved(f"dtype {dtype}: only BF16 was measured")
    if kernel == "triton":
        if not has_weight:
            raise RmsUnresolved("HAS_WEIGHT=False is a different Triton specialisation (not measured)")
        if not contiguous:
            raise RmsUnresolved("non-contiguous input: row stride specialisation not measured")
        if n_cols % 16 != 0:
            raise RmsUnresolved(f"n_cols={n_cols} not divisible by 16: different Triton specialisation (not measured)")
    elif kernel == "cuda":
        if not has_weight:
            raise RmsUnresolved("HasWeight=false is a different vllm::rms_norm_kernel instantiation (not measured)")
        if not contiguous:
            raise RmsUnresolved("non-contiguous input: the kernel reads rows at stride n_cols (not measured otherwise)")
        if n_cols % 8 != 0:
            raise RmsUnresolved(f"n_cols={n_cols} not divisible by 8: the scalar (VEC_SIZE=1) instantiation is not modelled")
        if n_cols > 8 * 1024:
            raise RmsUnresolved(f"n_cols={n_cols} > 8192: more than one 8-vector per thread, not measured")
    else:
        raise RmsUnresolved(f"unknown kernel {kernel!r}")
    if n_cols not in measured_n_cols:
        raise RmsUnresolved(f"n_cols={n_cols} not in the measured family {list(measured_n_cols)}")
    if not np.isfinite(eps):
        raise RmsUnresolved("non-finite eps")
