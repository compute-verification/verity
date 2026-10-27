"""ctypes wrapper of ``cpp/tc_model.cpp``: bulk per-instruction evaluation and the parametrised GEMM model."""

from __future__ import annotations

import ctypes
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from verity.ml.definitions.numerics import _jit
from verity.ml.definitions.numerics.mma import TCModel

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_SRC = os.path.join(HERE, "cpp", "tc_model.cpp")
LIB = str(_jit.lib_path("libtc_model.so"))             # build product (the _jit cache, outside the checkout), compiled on first use

_lib = None


def build() -> str:
    """Compile cpp/tc_model.cpp; atomic publish + builder lock (``_jit``): no half-written .so for a concurrent dlopen."""
    from pathlib import Path

    return str(_jit.atomic_build(Path(LIB), [Path(CPP_SRC)], lambda out: _jit.cxx_compile(out, Path(CPP_SRC))))


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        _lib.tc_records.restype = None
        _lib.tc_records.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _lib.gemm_model.restype = None
        _lib.gemm_model.argtypes = (
            [ctypes.c_int]
            + [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        )
    return _lib


GPU_NAN_WORD = 0x7FFF  # cvt.rn.bf16.f32 / F2F canonical NaN (every NaN a cuBLAS/Triton kernel wrote on the A100)
TORCH_CPU_NAN_WORD = 0x7FC0  # c10::BFloat16 round_to_nearest_even (torch CPU casts, veritor.core.silicon.f32_to_bf16)


def _dtype_code(dtype: str) -> int:
    return 0 if dtype == "bf16" else 1


def _groups(model: TCModel):
    return (ctypes.c_int * len(model.groups))(*model.groups)


def tc_records(model: TCModel, acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``tc_dot`` on ``n`` records: ``acc`` (n,) u32, ``a``/``b`` (n, k) u16 -> (n,) u32."""

    n = len(acc)
    acc = np.ascontiguousarray(acc, dtype=np.uint32)
    a = np.ascontiguousarray(a, dtype=np.uint16).reshape(n, model.k)
    b = np.ascontiguousarray(b, dtype=np.uint16).reshape(n, model.k)
    d = np.zeros(n, np.uint32)
    lib().tc_records(
        _dtype_code(model.dtype),
        acc.ctypes.data,
        a.ctypes.data,
        b.ctypes.data,
        d.ctypes.data,
        n,
        _groups(model),
        len(model.groups),
        model.width,
        model.zero_exponent,
    )
    return d


@dataclass(frozen=True)
class GemmModelSpec:
    """A candidate description of how a production GEMM orders and rounds its work.

    ``nsplit``/``bounds`` describe split-K over K-tiles of ``model.k`` (``bounds`` in
    tile units, ``nsplit + 1`` entries; ``None`` = equal splits); ``reduce_mode`` how the
    FP32 partials are summed (0 sequential ascending, 1 pairwise tree, 2 sequential
    descending; 3/4/5 the same after each partial is first rounded to BF16, as cuBLASLt's
    ``splitKreduce_kernel`` does with BF16 workspace partials; 6 BF16 partials with a BF16
    rounding after every add; 7/8 CUTLASS-style serial split-K through the BF16 output in
    ascending/descending split order); ``order`` an optional
    K-tile permutation; ``bias_mode`` the epilogue
    (0 none, 1 fp32 bias add then RNE, 2 bias as the initial accumulator term, 3 RNE then
    bf16 add then RNE, 4 bias added to partial 0 before the split reduction); ``nan_word`` the
    BF16 word every FP32 -> BF16 rounding step writes for a NaN (GPU kernels: the canonical
    0x7FFF of ``cvt.rn.bf16.f32``; torch's CPU conversion: 0x7FC0).  A NaN can only arise from
    finite operands through a split/slice reduction of +inf and -inf partials (single chains
    saturate to +-inf and stay there: ``overflow_diagnosis.py``).
    """

    model_name: str
    nsplit: int = 1
    bounds: tuple[int, ...] | None = None
    reduce_mode: int = 0
    order: tuple[int, ...] | None = None
    bias_mode: int = 1
    note: str = ""
    nan_word: int = GPU_NAN_WORD

    def describe(self) -> str:
        parts = [f"per-instruction model {self.model_name}"]
        if self.nsplit == 1:
            parts.append("single chain over all K-tiles in ascending k order, zero initial accumulator")
        else:
            b = "equal" if self.bounds is None else str(list(self.bounds))
            if self.reduce_mode == 6:
                parts.append(f"split-K {self.nsplit} ({b} tile bounds), partials rounded RNE to BF16, accumulated sequentially ascending with RNE to BF16 after every add")
            elif self.reduce_mode in (7, 8):
                order = "ascending" if self.reduce_mode == 7 else "descending"
                parts.append(f"serial split-K {self.nsplit} ({b} tile bounds) through the BF16 output in {order} split order: s = bf16(p_first); s = bf16(f32(s) + p_i) with FP32 partials p_i")
            else:
                red = {0: "sequential ascending", 1: "pairwise tree", 2: "sequential descending"}[self.reduce_mode % 3]
                kind = "FP32 partials" if self.reduce_mode < 3 else "partials rounded RNE to BF16 first, then"
                parts.append(f"split-K {self.nsplit} ({b} tile bounds), {kind} reduced {red} in FP32")
        if self.order is not None:
            parts.append("K-tile permutation applied (tiles listed per split: " + ", ".join(
                str(list(self.order[self.bounds[s]:self.bounds[s + 1]][:4])) + ("..." if self.bounds[s + 1] - self.bounds[s] > 4 else "")
                for s in range(self.nsplit)) + ")" if self.bounds else "K-tile permutation applied")
        parts.append(
            {
                0: "no bias",
                1: "epilogue: fp32 acc + bf16 bias (fp32 RNE), then RNE to bf16",
                2: "bias enters as the initial accumulator (term of the first group)",
                3: "RNE to bf16, then bf16 + bias in fp32, RNE to bf16",
                4: "bias added in fp32 to split partial 0 before the split reduction",
            }[self.bias_mode]
        )
        if self.nan_word != GPU_NAN_WORD:
            parts.append(f"NaN -> 0x{self.nan_word:04x}")
        if self.note:
            parts.append(self.note)
        return "; ".join(parts)

    def to_json(self) -> dict:
        d = asdict(self)
        d["description"] = self.describe()
        return d


def gemm_model(
    model: TCModel,
    A: np.ndarray,
    Bt: np.ndarray,
    bias: np.ndarray | None,
    spec: GemmModelSpec,
    acc_init: np.ndarray | None = None,
    want_f32: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate ``spec`` on ``A`` (M,K) words, ``Bt`` (N,K) words, ``bias`` (N,) words -> (bf16 (M,N), f32 bits (M,N))."""

    M, K = A.shape
    N = Bt.shape[0]
    if K % model.k:
        raise ValueError(f"K={K} is not a multiple of the instruction K {model.k}")
    A = np.ascontiguousarray(A, dtype=np.uint16)
    Bt = np.ascontiguousarray(Bt, dtype=np.uint16)
    bias_c = None if bias is None else np.ascontiguousarray(bias, dtype=np.uint16)
    acc_c = None if acc_init is None else np.ascontiguousarray(acc_init, dtype=np.uint32)
    out_bf16 = np.zeros((M, N), np.uint16)
    out_f32 = np.zeros((M, N), np.uint32) if want_f32 else None
    bounds = None
    if spec.bounds is not None:
        bounds = (ctypes.c_int * len(spec.bounds))(*spec.bounds)
    order = None
    if spec.order is not None:
        order = (ctypes.c_int * len(spec.order))(*spec.order)
    lib().gemm_model(
        _dtype_code(model.dtype),
        A.ctypes.data,
        Bt.ctypes.data,
        None if bias_c is None else bias_c.ctypes.data,
        None if acc_c is None else acc_c.ctypes.data,
        M,
        N,
        K,
        _groups(model),
        len(model.groups),
        model.width,
        model.zero_exponent,
        spec.nsplit,
        None if bounds is None else ctypes.cast(bounds, ctypes.c_void_p),
        spec.reduce_mode,
        None if order is None else ctypes.cast(order, ctypes.c_void_p),
        spec.bias_mode if bias_c is not None else 0,
        None if out_f32 is None else out_f32.ctypes.data,
        out_bf16.ctypes.data,
        int(spec.nan_word) & 0xFFFF,
    )
    return out_bf16, out_f32
