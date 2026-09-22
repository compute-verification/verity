"""ctypes driver for the single-instruction probes in ``cuda/mma_tiles_sm80.cu`` (GPU host only).

All tensors are raw bit patterns in numpy arrays: BF16/FP16 words are ``uint16``,
FP32 accumulators/results ``uint32``.  ``mma_tiles`` executes exactly one
``mma.sync.m16n8k16`` per tile.
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_SRC = os.path.join(HERE, "cuda", "mma_tiles_sm80.cu")
LIB = os.path.join(HERE, "cuda", "libmma_tiles_sm80.so")
K = 16
MAX_TILES_PER_CALL = 1 << 16

_lib = None


def build(arch: str = "sm_80") -> str:
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(CUDA_SRC):
        subprocess.run(
            [nvcc, "-O2", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o", LIB, CUDA_SRC],
            check=True,
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name in ("mma_bf16_tiles", "mma_f16_tiles"):
            fn = getattr(_lib, name)
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
        _lib.device_info.restype = ctypes.c_int
        _lib.device_info.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
    return _lib


def device_info() -> dict:
    name = ctypes.create_string_buffer(256)
    major, minor = ctypes.c_int(), ctypes.c_int()
    rc = lib().device_info(name, 256, ctypes.byref(major), ctypes.byref(minor))
    if rc:
        raise RuntimeError(f"cudaGetDeviceProperties failed: {rc}")
    return {"name": name.value.decode(), "sm": f"sm_{major.value}{minor.value}"}


def mma_tiles(dtype: str, A: np.ndarray, Bt: np.ndarray, C: np.ndarray) -> np.ndarray:
    """One ``mma.sync`` per tile.  ``A``: (n,16,16), ``Bt``: (n,8,16) words, ``C``: (n,16,8) f32 bits -> ``D``."""

    n = A.shape[0]
    A = np.ascontiguousarray(A, dtype=np.uint16).reshape(n, 16, K)
    Bt = np.ascontiguousarray(Bt, dtype=np.uint16).reshape(n, 8, K)
    C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, 16, 8)
    D = np.zeros((n, 16, 8), dtype=np.uint32)
    fn = lib().mma_bf16_tiles if dtype == "bf16" else lib().mma_f16_tiles
    for start in range(0, n, MAX_TILES_PER_CALL):
        stop = min(n, start + MAX_TILES_PER_CALL)
        a, b, c = A[start:stop], Bt[start:stop], C[start:stop]
        d = np.zeros((stop - start, 16, 8), dtype=np.uint32)
        rc = fn(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, stop - start)
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in {dtype} tile kernel")
        D[start:stop] = d
    return D


def mma_scalar(dtype: str, acc_bits: int, a: list[int], b: list[int]) -> int:
    """One output element: place the record at D[0][0] of a single tile."""

    A = np.zeros((1, 16, K), np.uint16)
    Bt = np.zeros((1, 8, K), np.uint16)
    C = np.zeros((1, 16, 8), np.uint32)
    A[0, 0, :] = a
    Bt[0, 0, :] = b
    C[0, 0, 0] = acc_bits
    return int(mma_tiles(dtype, A, Bt, C)[0, 0, 0])


# --- encodings ------------------------------------------------------------------


def f32_bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def bits_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.uint32).view(np.float32)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 (torch's cast)."""

    u = f32_bits(x).astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_f16_bits(x: np.ndarray) -> np.ndarray:
    """FP32 -> binary16, round to nearest even (numpy's cast)."""

    return np.asarray(x, dtype=np.float32).astype(np.float16).view(np.uint16)


def f16_to_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.uint16).view(np.float16).astype(np.float32)


def to_operand(dtype: str, x: np.ndarray) -> np.ndarray:
    return to_bf16_bits(x) if dtype == "bf16" else to_f16_bits(x)


def operand_to_f32(dtype: str, x: np.ndarray) -> np.ndarray:
    return bf16_to_f32(x) if dtype == "bf16" else f16_to_f32(x)


def scalar_operand(dtype: str, v: float) -> int:
    return int(to_operand(dtype, np.array([v], np.float32))[0])
