"""P2's GPU backend: `libp2v1.so` (`p2v1.cu`) through ctypes on torch tensors, with every label's `K` and `t` expanded on the
device by the library's `p2v1_keys` with v4's framing (`p2keys4.cuh`: counter-mode SHA-512 over the setup's key prefix,
whole 128-byte SHA-512 blocks that the library compresses once into a midstate: `reference.expand_prefix`'s 256 bytes, or
the registered scheme's `schemes.p2.expand_prefix`; `reference.keys_after`) just before the kernel that uses them, and
freed after it. A library must report `p2v1_keys_version() == 4`: v1's SHAKE256 keys are retired."""
from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path
from typing import Any

from . import reference

LIB = "libp2v1.so"
CUDA_LIB = Path(__file__).resolve().parents[2] / "word_codec" / "cuda_lib.py"
_v, _u, _q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
ABI = {"p2v1_decode": [_v, _v, _v, _v, _u, _u, _v, _v], "p2v1_encode": [_v, _v, _v, _v, _u, _u, _u, _u, _u, _v, _v]}
BUILT = {   # each where the build has it: P2_FUSED's, and P2_KEYS's or P2_KEYS4's
    "p2v1_decode_fused": [_v, _v, ctypes.c_char_p, ctypes.c_int, _q, _u, _q, _u, _v, _v],
    "p2v1_keys": [ctypes.c_char_p, ctypes.c_int, _q, _u, _q, _v, _v, _v],
}
ENC_LIB = "libp2enc.so"
ENC_ABI = {"p2enc_encode": ABI["p2v1_encode"], "p2enc_ctas_per_sm": []}
ENCODERS = ("cgbn", "p2sq")
"""The encoders, each giving the same rows: `cgbn`, the library's (CGBN's root), which every caller gets unless it asks;
and `p2sq`, the root on the decode's squaring, in a library of its own beside it (`p2enc.cu`, `ENC_LIB`)."""


def exponent_split(params: reference.Params) -> tuple[int, int, int]:
    """`(a, n, r)` with `(p + 1) / 4 = 2^a (2^n − 1) + r`, `0 ≤ r < 2^a`, the form `p2v1.cu`'s root takes."""
    e = (params.p + 1) // 4
    q = (1 << (params.w - 2)) - e
    a = max(1, (q - 1).bit_length())
    r, n = (1 << a) - q, params.w - 2 - a
    assert (((1 << n) - 1) << a) + r == e and 0 <= r < 1 << a
    return a, n, r


def _cuda_lib() -> Any:
    """`benchmarks/word_codec/cuda_lib.py` by its path: the vLLM integration loads this package with `sys.path` as it is."""
    if "wc_cuda_lib" not in sys.modules:
        spec = importlib.util.spec_from_file_location("wc_cuda_lib", CUDA_LIB)
        mod = sys.modules["wc_cuda_lib"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules["wc_cuda_lib"]


class Kernels:
    """The library on the current CUDA device, for one parameter set and one setup's key prefix (`build_dir`, where v1's
    SHAKE256 extension was built, is no longer read)."""

    def __init__(self, lib_dir: str, build_dir: str, params: reference.Params, prefix: bytes) -> None:
        import torch

        self.torch = torch
        self.lib_dir, self._enc = Path(lib_dir), None
        self.lib = _cuda_lib().load(self.lib_dir / LIB, ABI, BUILT)
        if self.lib.p2v1_width_bits() != params.w:
            raise RuntimeError(f"{LIB} is built for {self.lib.p2v1_width_bits()} bits, not {params.w}")
        self.fused = hasattr(self.lib, "p2v1_decode_fused")
        self.fast_keys = hasattr(self.lib, "p2v1_keys")
        self.keys_version = self.lib.p2v1_keys_version() if hasattr(self.lib, "p2v1_keys_version") else 1
        if self.keys_version != 4 or not self.fast_keys:
            raise RuntimeError(f"{LIB} expands keys v{self.keys_version}, not v4: build it with FAST='-DP2_FAST -DP2_KEYS4 -DP2SQ_KARA'")
        self.params, self.split = params, exponent_split(params)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.prefix_bytes = bytes(prefix)
        self.flags = torch.zeros(1, dtype=torch.int32, device=self.device)

    def keys(self, first: int, count: int) -> tuple[Any, Any]:
        """`K` and `t` of labels `first .. first + count − 1`, as `[count, W/8]` big-endian rows (`t`'s first candidate; a
        second is needed with probability `c0 / 2^W`, which the kernels flag)."""
        torch = self.torch
        k, t = (torch.empty(count, self.params.width_bytes, dtype=torch.uint8, device=self.device) for _ in range(2))
        rc = self.lib.p2v1_keys(self.prefix_bytes, len(self.prefix_bytes), first, count, self.params.blocks_per_segment,
                                k.data_ptr(), t.data_ptr(), self._stream())
        if rc:
            raise RuntimeError(f"p2v1_keys: error {rc}")
        return k, t

    def _stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream

    def decode_fused(self, c: Any, first: int = 0) -> Any:
        """`decode` by the library's `p2v1_decode_fused`, which expands the keys itself, pipelined with the decode."""
        m = self.torch.empty_like(c)
        rc = self.lib.p2v1_decode_fused(c.data_ptr(), m.data_ptr(), self.prefix_bytes, len(self.prefix_bytes), first, c.shape[0],
                                        self.params.blocks_per_segment, self.params.c0, self.flags.data_ptr(), self._stream())
        if rc:
            raise RuntimeError(f"p2v1_decode_fused: error {rc}")
        return m

    def decode(self, c: Any, first: int = 0) -> Any:
        """`m` rows of the `c` rows of labels `first ..`, keys expanded for them."""
        if self.fused:
            return self.decode_fused(c, first)
        k, t = self.keys(first, c.shape[0])
        m = self.torch.empty_like(c)
        rc = self.lib.p2v1_decode(c.data_ptr(), m.data_ptr(), k.data_ptr(), t.data_ptr(), c.shape[0], self.params.c0,
                                  self.flags.data_ptr(), self._stream())
        if rc:
            raise RuntimeError(f"p2v1_decode: CUDA error {rc}")
        return m

    def enc_lib(self) -> Any:
        """`ENC_LIB` in the library's directory, loaded on first use."""
        if self._enc is None:
            if not (self.lib_dir / ENC_LIB).exists():
                raise RuntimeError(f"no {self.lib_dir / ENC_LIB}: p2enc.cu's first lines say how it is built")
            self._enc = _cuda_lib().load(self.lib_dir / ENC_LIB, ENC_ABI)
        return self._enc

    def encode(self, m: Any, first: int = 0, encoder: str = "cgbn", out: Any = None) -> Any:
        """`c` rows of the `m` rows of labels `first ..`, by one of `ENCODERS`, into `out` when given."""
        if encoder not in ENCODERS:
            raise ValueError(f"encoder is one of {ENCODERS}, not {encoder!r}")
        name, fn = ("p2v1_encode", self.lib.p2v1_encode) if encoder == "cgbn" else ("p2enc_encode", self.enc_lib().p2enc_encode)
        k, t = self.keys(first, m.shape[0])
        c = self.torch.empty_like(m) if out is None else out
        if c.shape != m.shape or c.dtype != m.dtype or not c.is_contiguous():
            raise ValueError("out is rows like m's")
        a, n, r = self.split
        rc = fn(m.data_ptr(), c.data_ptr(), k.data_ptr(), t.data_ptr(), m.shape[0], self.params.c0, a, n, r,
                self.flags.data_ptr(), self._stream())
        if rc:
            raise RuntimeError(f"{name}: CUDA error {rc}" if rc > 0 else f"{name}: the exponent's steps don't fit its chain")
        return c

    def raised(self) -> int:
        """The kernels' flags since the last call: bit 0 a tweak's first candidate at or above p, bit 1 no root."""
        f = int(self.flags.item())
        self.flags.zero_()
        return f
