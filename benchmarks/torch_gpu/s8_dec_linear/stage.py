"""`libp2stage.so` (`p2stage.cu`) through ctypes on torch tensors: P2's decode of a tensor's blocks into its staging
buffer, the plaintext the exact INT8 GEMM reads as its weight operand.

`Stage(lib_dir).decode(c, first, prefix, base, length)` decodes the store rows `c` ([n, 2056] uint8 on the device, blocks
first .. first + n - 1) with their keys expanded in the kernel, and writes W bytes [base, base + length) into a [length]
uint8 buffer (a given `dst`, or a new one). `decode_keys` takes the keys as rows instead (`p2_v1.gpu.Kernels.keys`), the
check of the in-kernel expansion. `flags()` reads and clears the kernels' flags (`fixture.FLAG_*`).
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import fixture as F
import torch

LIB = "libp2stage.so"


class Stage:
    def __init__(self, lib_dir: str | Path, device: torch.device | None = None) -> None:
        self.lib = ctypes.CDLL(str(Path(lib_dir) / LIB))
        if self.lib.p2stage_version() != 1 or self.lib.p2stage_width_bits() != F.PARAMS.w:
            raise RuntimeError(f"{LIB}: version {self.lib.p2stage_version()} at {self.lib.p2stage_width_bits()} bits")
        v, u, q, i64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int64
        self.lib.p2stage_grid.argtypes, self.lib.p2stage_grid.restype = [ctypes.c_int, u], u
        self.lib.p2stage_scratch_bytes.argtypes, self.lib.p2stage_scratch_bytes.restype = [ctypes.c_int, u], q
        self.lib.p2stage_attributes.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                                ctypes.POINTER(ctypes.c_int)]
        self.lib.p2stage_decode.argtypes = [v, q, u, ctypes.c_char_p, ctypes.c_int, q, u, v, i64, i64, v, v, q, u, v, v]
        self.lib.p2stage_decode_keys.argtypes = [v, v, v, q, u, u, v, i64, i64, v, v, q, u, v, v]
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.warps_per_cta = self.lib.p2stage_warps_per_cta()
        self._flags = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._scratch: dict[int, Any] = {}

    def attributes(self, keyed: bool = True) -> dict[str, int]:
        r, s, o = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        rc = self.lib.p2stage_attributes(int(keyed), ctypes.byref(r), ctypes.byref(s), ctypes.byref(o))
        if rc:
            raise RuntimeError(f"p2stage_attributes: CUDA error {rc}")
        return {"regs": r.value, "smem": s.value, "ctas_per_sm": o.value, "warps_per_cta": self.warps_per_cta}

    def grid(self, n: int, keyed: bool = True) -> int:
        return int(self.lib.p2stage_grid(int(keyed), n))

    def scratch(self, grid: int, keyed: bool) -> Any:
        need = int(self.lib.p2stage_scratch_bytes(int(keyed), grid))
        have = self._scratch.get(int(keyed))
        if have is None or have.numel() < need:
            have = self._scratch[int(keyed)] = torch.empty(need, dtype=torch.uint8, device=self.device)
        return have

    def _out(self, length: int, dst: Any | None) -> Any:
        if dst is None:
            return torch.empty(length, dtype=torch.uint8, device=self.device)
        if dst.dtype != torch.uint8 or not dst.is_contiguous() or dst.numel() < length:
            raise ValueError("dst is a contiguous uint8 buffer of at least the tensor's length")
        return dst

    def _check_rows(self, c: Any, first: int, base: int, length: int) -> None:
        if c.dtype != torch.uint8 or c.dim() != 2 or c.shape[1] != F.ROW or not c.is_contiguous():
            raise ValueError(f"c is contiguous [n, {F.ROW}] uint8 rows")
        want = F.blocks_of(base, length)
        if (first, first + c.shape[0]) != (want.start, want.stop):
            raise ValueError(f"the tensor at {base} of {length} bytes is blocks {want.start} .. {want.stop - 1}, not "
                             f"{first} .. {first + c.shape[0] - 1}")

    def decode(self, c: Any, first: int, prefix: bytes, base: int, length: int, dst: Any | None = None, grid: int = 0,
               smids: Any | None = None, stream: Any | None = None) -> Any:
        self._check_rows(c, first, base, length)
        out, n = self._out(length, dst), c.shape[0]
        grid = grid or self.grid(n, True)
        s = self.scratch(grid, True)
        rc = self.lib.p2stage_decode(c.data_ptr(), first, n, prefix, len(prefix), F.PARAMS.blocks_per_segment, F.PARAMS.c0,
                                     out.data_ptr(), base, length, self._flags.data_ptr(), s.data_ptr(), s.numel(), grid,
                                     smids.data_ptr() if smids is not None else None, self._stream(stream))
        if rc:
            raise RuntimeError(f"p2stage_decode: error {rc}")
        return out

    def decode_keys(self, c: Any, k: Any, t: Any, first: int, base: int, length: int, dst: Any | None = None,
                    grid: int = 0, stream: Any | None = None) -> Any:
        self._check_rows(c, first, base, length)
        if k.shape != c.shape or t.shape != c.shape:
            raise ValueError("K and t are rows like c's")
        out, n = self._out(length, dst), c.shape[0]
        grid = grid or self.grid(n, False)
        s = self.scratch(grid, False)
        rc = self.lib.p2stage_decode_keys(c.data_ptr(), k.data_ptr(), t.data_ptr(), first, n, F.PARAMS.c0, out.data_ptr(), base,
                                          length, self._flags.data_ptr(), s.data_ptr(), s.numel(), grid, None,
                                          self._stream(stream))
        if rc:
            raise RuntimeError(f"p2stage_decode_keys: error {rc}")
        return out

    def flags(self) -> int:
        f = int(self._flags.item())
        self._flags.zero_()
        return f

    def _stream(self, stream: Any | None) -> int:
        return (stream or torch.cuda.current_stream(self.device)).cuda_stream
