"""`h1r.cu` from torch: a step's served calls as one device table, their units' receipts in two launches on the current
stream into buffers the plan owns (allocated once, at `plan`), so a CUDA graph captures `launch`.

    lib = H1R("libh1r.so")
    plan = lib.plan([Call(a, b, base, roots, live), ...])   # a [T, K], b [N, K] s8 on the device; roots (16,) int64 words
    lib.launch(plan, salt)                                  # salt (8,) int64 big-endian words on the device; version=1|2|3 (3: a unit a block)
    lib.launch(plan, salt, step_base=sb)                    # sb (1,) int64 on the device, added to every call's base
    lib.launch_j10r(plan, salt, m)                          # version 1 under J10-R's call-scale noise, Layout(m, N, K)
    plan.receipts(i)                                        # call i's (units, 2 sets) int32 view
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass

import numpy as np
import torch


class _Call(C.Structure):
    _fields_ = [("a", C.c_void_p), ("b", C.c_void_p), ("lda", C.c_longlong), ("ldb", C.c_longlong), ("roots", C.c_void_p),
                ("root_b", C.c_void_p), ("live", C.c_void_p), ("base", C.c_uint64), ("k", C.c_int), ("ncb", C.c_int), ("units", C.c_int),
                ("seed0", C.c_int), ("out0", C.c_longlong)]


@dataclass
class Call:
    a: torch.Tensor      # [T, K] int8
    b: torch.Tensor      # [N, K] int8
    base: int
    roots: torch.Tensor | int  # (16,) int64: root_a's then root_b's big-endian words; or root_a's device address
    live: torch.Tensor   # (T / 32,) uint8
    root_b: int | None = None  # root_b's device address when ROOTS is root_a's


def words(b: bytes) -> np.ndarray:
    """Bytes (a multiple of 8) as big-endian u64 words, in int64."""
    return np.frombuffer(b, dtype=">u8").astype("<u8").view("<i8").copy()


@dataclass
class Plan:
    calls: list[Call]
    table: torch.Tensor
    ks: object
    max_units: int
    seeds: torch.Tensor
    out: torch.Tensor
    offs: list[tuple[int, int, int]]   # (word offset, units, words a unit)

    def receipts(self, i: int) -> torch.Tensor:
        o, n, w = self.offs[i]
        return self.out[o:o + n * w].view(n, w)


class H1R:
    def __init__(self, path: str):
        self.lib = C.CDLL(path)
        self.lib.h1r_launch.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                        C.c_int, C.c_int, C.c_void_p]
        self.lib.h1r_n_digits.argtypes = [C.c_int]
        if self.lib.h1r_call_bytes() != C.sizeof(_Call):
            raise RuntimeError(f"H1Call is {self.lib.h1r_call_bytes()} bytes in h1r.cu, {C.sizeof(_Call)} here")

    def sets(self, k: int) -> int:
        ks = k // 32
        return ks + self.lib.h1r_n_digits(ks)

    def plan(self, calls: list[Call], seeds: torch.Tensor | None = None, out: torch.Tensor | None = None) -> Plan:
        """The calls' table, and its seed and receipt buffers: `seeds` and `out` when given (another plan's of the same
        calls' shapes, so several tables share them), else its own."""
        dev = calls[0].a.device
        tab = (_Call * len(calls))()
        seed0 = out0 = 0
        offs = []
        for i, c in enumerate(calls):
            t, k = c.a.shape
            n = c.b.shape[0]
            if c.b.shape[1] != k or t % 32 or n % 32 or k % 512:
                raise ValueError(f"call {i}: A {tuple(c.a.shape)}, B {tuple(c.b.shape)}: h1r takes 32 | T, 32 | N, 512 | K")
            units = (t // 32) * (n // 32)
            w = 2 * self.sets(k)
            ra = c.roots if isinstance(c.roots, int) else c.roots.data_ptr()
            rb = c.root_b if c.root_b is not None else ra + 64
            tab[i] = _Call(c.a.data_ptr(), c.b.data_ptr(), c.a.stride(0), c.b.stride(0), ra, rb, c.live.data_ptr(), c.base, k,
                           n // 32, units, seed0, out0)
            offs.append((out0, units, w))
            seed0 += units
            out0 += units * w
        table = torch.frombuffer(bytearray(bytes(tab)), dtype=torch.uint8).to(dev)
        ks = (C.c_int * len(calls))(*[c.a.shape[1] for c in calls])
        if (seeds is not None and seeds.numel() != seed0 * 16) or (out is not None and out.numel() != out0):
            raise ValueError(f"shared buffers of {seeds.numel() if seeds is not None else None} seed words and "
                             f"{out.numel() if out is not None else None} receipt words, not {seed0 * 16} and {out0}")
        return Plan(calls, table, ks, max(o[1] for o in offs),
                    torch.empty(seed0 * 16, dtype=torch.int64, device=dev) if seeds is None else seeds,
                    torch.empty(out0, dtype=torch.int32, device=dev) if out is None else out, offs)

    def launch(self, plan: Plan, salt: torch.Tensor, nowrite: bool = False, version: int = 2,
               step_base: torch.Tensor | None = None) -> None:
        rc = self.lib.h1r_launch(plan.table.data_ptr(), len(plan.calls), plan.max_units, plan.ks, salt.data_ptr(),
                                 None if step_base is None else step_base.data_ptr(), plan.seeds.data_ptr(),
                                 plan.out.data_ptr(), int(nowrite), int(version), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1r_launch returned {rc}")

    def launch_j10r(self, plan: Plan, salt: torch.Tensor, m: int, nowrite: bool = False,
                    step_base: torch.Tensor | None = None) -> None:
        """Version 1 under J10-R (h1r.cu's h1r_launch_j10r): every call's noise its seed_c's stream at call scale,
        Layout(m, N, K), m the calls' layout rows."""
        fn = getattr(self.lib, "h1r_launch_j10r", None)
        if fn is None:
            raise RuntimeError("this h1r build has no h1r_launch_j10r: rebuild h1r.cu (or h1o.cu) from this tree")
        fn.argtypes = [*self.lib.h1r_launch.argtypes[:-2], C.c_int, C.c_void_p]
        rc = fn(plan.table.data_ptr(), len(plan.calls), plan.max_units, plan.ks, salt.data_ptr(),
                None if step_base is None else step_base.data_ptr(), plan.seeds.data_ptr(), plan.out.data_ptr(),
                int(nowrite), int(m), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"h1r_launch_j10r returned {rc}")
