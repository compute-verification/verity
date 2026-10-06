"""P2 v1's GPU backend: `libp2v1.so` (`p2v1.cu`) through ctypes on torch tensors, with every block's `K` and `t` expanded on the
device by SHAKE256 with the spec's v1 framing (`band_gpu`'s `pous_native.shake256_rows`: one shared prefix, then each block's
`u64be(segment) ‖ u64be(local block) ‖ purpose`, or the library's own `p2v1_keys` when it is built with one) just before the
kernel that uses them, and freed after it. A library built with `P2_KEYS4` expands the v4 keys instead (`reference.keys_v4`,
counter-mode SHA-512; `keys_version` 4), and has no SHAKE256 fallback."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

from . import reference

LIB = "libp2v1.so"


def exponent_split(params: reference.Params) -> tuple[int, int, int]:
    """`(a, n, r)` with `(p + 1) / 4 = 2^a (2^n − 1) + r`, `0 ≤ r < 2^a`, the form `p2v1.cu`'s root takes."""
    e = (params.p + 1) // 4
    q = (1 << (params.w - 2)) - e
    a = max(1, (q - 1).bit_length())
    r, n = (1 << a) - q, params.w - 2 - a
    assert (((1 << n) - 1) << a) + r == e and 0 <= r < 1 << a
    return a, n, r


class Kernels:
    """The library and the SHAKE256 kernel on the current CUDA device, for one parameter set, salt and model commitment."""

    def __init__(self, lib_dir: str, build_dir: str, params: reference.Params, salt: bytes, model: bytes) -> None:
        import torch

        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from band_gpu import native

        self.torch = torch
        self.native = native.load(build_dir)
        self.lib = ctypes.CDLL(str(Path(lib_dir) / LIB))
        if self.lib.p2v1_width_bits() != params.w:
            raise RuntimeError(f"{LIB} is built for {self.lib.p2v1_width_bits()} bits, not {params.w}")
        v = ctypes.c_void_p
        u = ctypes.c_uint32
        self.lib.p2v1_decode.argtypes = [v, v, v, v, u, u, v, v]
        self.lib.p2v1_encode.argtypes = [v, v, v, v, u, u, u, u, u, v, v]
        self.fused = hasattr(self.lib, "p2v1_decode_fused")
        if self.fused:
            self.lib.p2v1_decode_fused.argtypes = [v, v, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64, u, ctypes.c_uint64, u, v, v]
        self.fast_keys = hasattr(self.lib, "p2v1_keys")
        if self.fast_keys:
            q = ctypes.c_uint64
            self.lib.p2v1_keys.argtypes = [ctypes.c_char_p, ctypes.c_int, q, u, q, v, v, v]
        self.keys_version = self.lib.p2v1_keys_version() if hasattr(self.lib, "p2v1_keys_version") else 1
        if self.keys_version not in (1, 4):
            raise RuntimeError(f"{LIB} expands keys v{self.keys_version}")
        self.params, self.split = params, exponent_split(params)
        self.device = torch.device("cuda", torch.cuda.current_device())
        expand = reference.expand_prefix_v4 if self.keys_version == 4 else reference.expand_prefix
        prefix = self.prefix_bytes = expand(salt, params, model)
        self.prefix = torch.tensor(list(prefix), dtype=torch.uint8, device=self.device).view(1, -1)
        self.flags = torch.zeros(1, dtype=torch.int32, device=self.device)

    def keys(self, first: int, count: int) -> tuple[Any, Any]:
        """`K` and `t` of blocks `first .. first + count − 1`, as `[count, W/8]` big-endian rows (`t`'s first candidate; a
        second is needed with probability `c0 / 2^W`, which the kernels flag)."""
        torch = self.torch
        if self.fast_keys:
            k, t = (torch.empty(count, self.params.width_bytes, dtype=torch.uint8, device=self.device) for _ in range(2))
            rc = self.lib.p2v1_keys(self.prefix_bytes, len(self.prefix_bytes), first, count, self.params.blocks_per_segment,
                                    k.data_ptr(), t.data_ptr(), self._stream())
            if rc:
                raise RuntimeError(f"p2v1_keys: error {rc}")
            return k, t
        if self.keys_version != 1:
            raise RuntimeError(f"{LIB} reports keys v{self.keys_version} without p2v1_keys")
        idx = torch.arange(first, first + count, dtype=torch.int64, device=self.device)
        s, v = idx // self.params.blocks_per_segment, idx % self.params.blocks_per_segment
        shifts = torch.arange(56, -1, -8, dtype=torch.int64, device=self.device)
        sv = torch.cat([(s[:, None] >> shifts) & 0xFF, (v[:, None] >> shifts) & 0xFF], 1).to(torch.uint8)
        pre = self.prefix.expand(count, -1)
        out = []
        for purpose in (b"K", b"t"):
            body = torch.cat([sv, torch.full((count, 1), purpose[0], dtype=torch.uint8, device=self.device)], 1).contiguous()
            rows = torch.zeros(count, self.params.width_bytes, dtype=torch.uint8, device=self.device)
            self.native.shake256_rows(pre, body, rows, True)
            out.append(rows)
        return out[0], out[1]

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
        """`m` rows of the `c` rows of blocks `first ..`, keys expanded for them."""
        if self.fused:
            return self.decode_fused(c, first)
        k, t = self.keys(first, c.shape[0])
        m = self.torch.empty_like(c)
        rc = self.lib.p2v1_decode(c.data_ptr(), m.data_ptr(), k.data_ptr(), t.data_ptr(), c.shape[0], self.params.c0,
                                  self.flags.data_ptr(), self._stream())
        if rc:
            raise RuntimeError(f"p2v1_decode: CUDA error {rc}")
        return m

    def encode(self, m: Any, first: int = 0) -> Any:
        k, t = self.keys(first, m.shape[0])
        c = self.torch.empty_like(m)
        a, n, r = self.split
        rc = self.lib.p2v1_encode(m.data_ptr(), c.data_ptr(), k.data_ptr(), t.data_ptr(), m.shape[0], self.params.c0, a, n, r,
                                  self.flags.data_ptr(), self._stream())
        if rc:
            raise RuntimeError(f"p2v1_encode: CUDA error {rc}")
        return c

    def raised(self) -> int:
        """The kernels' flags since the last call: bit 0 a tweak's first candidate at or above p, bit 1 no root."""
        f = int(self.flags.item())
        self.flags.zero_()
        return f
