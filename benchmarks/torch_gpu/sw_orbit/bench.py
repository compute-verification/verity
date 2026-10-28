"""orbit.cu's gates and timings on one sm_90a GPU, against the plain int8 GEMM.

Per shape (N x K at M rows; default 8B's qkv, o and gate-up at M = 16,384):

- **Gates**, before any timing. Each orbit arm's O is compared with `torch._int_mm`'s, bit for bit:
  - on operands with |x|, |w| ≤ 63, where no form carries, the products' O itself (and the forms kernels count no carry);
  - on Gaussian rows quantized at their absmax (as served activations are), on heavy-tailed rows (`heavy_rows`) and,
    with `--real DIR`, on Llama 3.1 8B's layer-0 weights (weights.py's cache): the products' O plus orbit_corr's
    corrections, fused and unfused;
  - ext_gemm.h's entry (`v31_ext_*`) on Gaussian operands, and on operands one of whose half-grid rows carries past its
    list, which the entry recomputes with the plain GEMM.
  An arm that fails a gate isn't timed.
- **Timings:** 2 s items, arm by arm, interleaved over `--reps` rounds after a warm-up, each item rotating through enough
  weight copies to leave L2, on heavy-tailed activations and (with `--real`) 8B's weights. Every item records its time
  per call, the SM clock NVML reads through it and a clock64 probe right after it. The plain GEMM is the fastest of
  cuBLASLt (autotuned, s8_linear's lt.py), torch._int_mm, wgmma.linear (s8_linear's kernel, bf16 out) and CUTLASS's
  cooperative TMA GEMMs; every arm is reported as a fraction of it, as θ (MACs a second over 989.4 T at 1,830 MHz; the
  1,980 MHz reading is 1,070.5 T), as T MAC/s of the plain GEMM's count, and as MACs per SM clock (over 4,096).
- **κ:** a correction's cost per MAC in the plain GEMM's time per MAC, from the correction arms and the carries a call
  makes (each B carry adds an A column of M/2 rows, each A carry a column of F_B's N/2), and the fraction of plain the
  corrections add at the measured carry rates and at `--rho-a`'s (8B's activations).

Arms: `lt`, `int_mm`, `wgmma`, `cutlass_128x256`, `cutlass_128x256_c2`, `cutlass_128x128` (plain); `forms_a`, `forms_b`,
`batched`, `assemble`, `fused` (the orbit's parts, forms prepared); `unfused` (forms_a + batched + assemble), `orbit`
(forms_a + fused, uncorrected); `corr` (the honest corrections, full s32 updates of O), `corr_b` (B's carries only),
`corr_nostore` and `corr_b_nostore` (the same, O untouched: their compute), `corr4` (B's carries' corrections mod 4,
bit-sliced 32 rows a word: what O's bit 9 alone needs), `rmw_o` (one s32 read-modify-write of O); `exact` (forms_a +
fused + corr) and `orbit_call` (forms_b + exact: ext_gemm.h's call, whose load runs every served call; B's lists and
exact forms, built at load from forms_b's carry plane, aren't in it).

Writes OUT/bench.json (everything) and OUT/table.md.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import orbit as OR  # noqa: E402

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
PEAK_MACS = 989.4e12
MACS_PER_SM_CLK = 4096
PLAIN = ("lt", "int_mm", "wgmma", "cutlass_128x256", "cutlass_128x256_c2", "cutlass_128x128")
ORBIT_ARMS = ("forms_a", "forms_b", "batched", "assemble", "fused", "unfused", "orbit", "corr", "corr_b", "corr_nostore",
              "corr_b_nostore", "corr_b_nostore_r4", "corr4", "rmw_o", "exact", "orbit_call")
RHO_A = "qkv=0.00026,o=0.00044,gate_up=0.00175,down=0.000056"
SLOTS = len(OR.ORDER)
COPY_BYTES = 256 << 20

vp, i32, i64 = C.c_void_p, C.c_int, C.c_longlong


class Lib:
    """liborbit.so on torch's current stream."""

    SIGS = {"orbit_info": [vp], "orbit_forms": [vp, vp, vp, vp, vp, vp, i32, i32, i32, vp],
            "orbit_corr": [vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, i32, vp],
            "orbit_corr4": [vp, vp, vp, vp, vp, i32, i32, i32, vp],
            "orbit_batched": [vp, vp, vp, i32, i32, i32, vp], "orbit_assemble": [vp, vp, i32, i32, vp],
            "orbit_fused": [vp, vp, vp, i32, i32, i32, i32, i32, vp], "orbit_plain": [vp, vp, vp, i32, i32, i32, i32, vp],
            "orbit_clock": [vp, i64, vp], "v31_ext_load": [vp, vp, vp], "v31_ext_gemm": [vp, vp, vp, vp]}

    def __init__(self, path: Path):
        self.so = C.CDLL(str(path))
        for f, a in self.SIGS.items():
            getattr(self.so, f).argtypes, getattr(self.so, f).restype = a, i32
        self.so.v31_ext_init.argtypes, self.so.v31_ext_init.restype = [i32, i32, i32], vp
        self.so.v31_ext_free.argtypes, self.so.v31_ext_free.restype = [vp], None
        info = (C.c_int * 4)()
        self.so.orbit_info(info)
        self.info = {"fused_smem": info[0], "fused_stages": info[1], "fused_threads": info[2], "sms": info[3]}
        self.clk = torch.zeros(2, dtype=torch.int64, device="cuda")

    @staticmethod
    def st() -> vp:
        return vp(torch.cuda.current_stream().cuda_stream)

    def call(self, f: str, *a) -> None:
        rc = getattr(self.so, f)(*a, self.st())
        if rc:
            raise RuntimeError(f"{f} returned {rc}")

    def forms(self, x, wide, counts, carry, which: int, alist=None, acount=None) -> None:
        def ptr(t):
            return t.data_ptr() if t is not None else None
        self.call("orbit_forms", x.data_ptr(), wide.data_ptr(), counts.data_ptr(), ptr(carry), ptr(alist), ptr(acount),
                  x.shape[0], x.shape[1], which)

    def clock_mhz(self, spin: int = 4_000_000) -> float:
        self.call("orbit_clock", self.clk.data_ptr(), spin)
        c, ns = self.clk.tolist()
        return c / ns * 1e3


class Nvml:
    """The leased GPU's SM clock through NVML, sampled by a thread while an item runs."""

    def __init__(self):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2()
        self.h = vp()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
        if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)):
            self.lib.nvmlDeviceGetHandleByIndex_v2(0, C.byref(self.h))

    def sm_mhz(self) -> int:
        v = C.c_uint()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, C.byref(v))
        return v.value

    def sample(self, stop: threading.Event, out: list) -> None:
        while not stop.is_set():
            out.append(self.sm_mhz())
            time.sleep(0.02)


def gauss_rows(rows: int, k: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, k, generator=gen, device="cuda")
    s = x.abs().amax(dim=1, keepdim=True) / 127
    return torch.clamp(torch.round(x / s), -127, 127).to(torch.int8)


def small(rows: int, k: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(-63, 64, (rows, k), generator=gen, device="cuda", dtype=torch.int8)


def wrap32(x: torch.Tensor) -> torch.Tensor:
    """int64 -> int32 mod 2^32."""
    return (((x + (1 << 31)) % (1 << 32)) - (1 << 31)).to(torch.int32)


A_CAP = 1024  # orbit.cu's kAcap


def heavy_rows(rows: int, k: int, gen: torch.Generator, frac: float = 0.005, gain: float = 4.0) -> torch.Tensor:
    """Gaussian rows with a fixed `frac` of channels `gain` times larger, quantized per row at absmax 127: an activation
    stand-in whose Strassen forms carry at about 0.05%, near 8B's real 0.026-0.175% (gauss_rows carries at 0.75%)."""
    x = torch.randn(rows, k, generator=gen, device="cuda")
    ch = torch.randperm(k, generator=gen, device="cuda")[: max(1, int(frac * k))]
    x[:, ch] *= gain
    s = x.abs().amax(dim=1, keepdim=True) / 127
    return torch.clamp(torch.round(x / s), -127, 127).to(torch.int8)


def b_lists(wide: torch.Tensor, carry: torch.Tensor) -> dict:
    """B's carries as orbit_corr reads them, from forms_b's wide forms [Nh, 7 Kh] and carry plane [7, Nh, Kh]: per slot
    and column the k's of the +1 carries (bidx[bptr .. bmid)) then the -1 ones' (bidx[bmid .. bptr next)), and B's exact
    forms transposed, ft [7, Kh, Nh] s16."""
    slots, nh, kh = carry.shape
    cp, cn = (carry > 0).sum(2).to(torch.int64), (carry < 0).sum(2).to(torch.int64)
    tot = (cp + cn).reshape(-1)
    ptr = torch.zeros(slots * nh + 1, dtype=torch.int64, device=carry.device)
    ptr[1:] = torch.cumsum(tot, 0)
    start = ptr[:-1].reshape(slots, nh)
    mid = start + cp
    idx = torch.zeros(int(ptr[-1]), dtype=torch.int16, device=carry.device)
    for sel, base in ((carry > 0, start), (carry < 0, mid)):
        nz = torch.nonzero(sel)
        if nz.numel():
            s, n, k = nz[:, 0], nz[:, 1], nz[:, 2]
            row = s * nh + n
            first = torch.searchsorted(row, row, right=False)
            rank = torch.arange(row.numel(), device=row.device) - first
            idx[base.reshape(-1)[row] + rank] = k.to(torch.int16)
    bptr = torch.empty(slots, nh + 1, dtype=torch.int32, device=carry.device)
    bptr[:, :nh] = start.to(torch.int32)
    bptr[:, nh] = (start[:, -1] + cp[:, -1] + cn[:, -1]).to(torch.int32)
    f = wide.view(nh, slots, kh).to(torch.int16) + 256 * carry.permute(1, 0, 2).to(torch.int16)
    ft = f.permute(1, 2, 0).contiguous()
    return {"bptr": bptr.contiguous(), "bmid": mid.to(torch.int32).contiguous(), "bidx": idx, "ft": ft,
            "nnz": int(ptr[-1])}


class Orbit:
    """One shape's orbit buffers: A's forms (wide, counts, carry lists) and, per weight copy, B's forms and carries'
    lists (built once a copy, as a load would)."""

    def __init__(self, lib: Lib, m: int, n: int, k: int):
        self.lib, self.m, self.n, self.k = lib, m, n, k
        self.mh, self.nh, self.kh = m // 2, n // 2, k // 2
        self.a_wide = torch.empty(self.mh, SLOTS * self.kh, dtype=torch.int8, device="cuda")
        self.a_cnt = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
        self.a_list = torch.empty(self.mh, A_CAP, dtype=torch.int32, device="cuda")
        self.a_count = torch.zeros(self.mh, dtype=torch.int32, device="cuda")
        self.no_count = torch.zeros(self.mh, dtype=torch.int32, device="cuda")
        self.flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.c4 = None
        self.b_wides: list[torch.Tensor] = []
        self.b_cnts: list[torch.Tensor] = []
        self.b_lists: list[dict] = []
        self.prod = None

    def load(self, ws: list[torch.Tensor], share: bool = False) -> None:
        """B's forms and lists for each copy; `share`: the copies hold one B, so its lists are built once."""
        self.b_wides = [torch.empty(self.nh, SLOTS * self.kh, dtype=torch.int8, device="cuda") for _ in ws]
        self.b_cnts = [torch.zeros(SLOTS, dtype=torch.int64, device="cuda") for _ in ws]
        self.b_lists = []
        for i, (w, bw, bc) in enumerate(zip(ws, self.b_wides, self.b_cnts)):
            if share and i:
                self.lib.forms(w, bw, bc, None, 1)
                self.b_lists.append(self.b_lists[0])
                continue
            carry = torch.zeros(SLOTS, self.nh, self.kh, dtype=torch.int8, device="cuda")
            self.lib.forms(w, bw, bc, carry, 1)
            self.b_lists.append(b_lists(bw, carry))
            del carry

    def forms_a(self, x) -> None:
        self.lib.forms(x, self.a_wide, self.a_cnt, None, 0, self.a_list, self.a_count)

    def forms_b(self, w, i: int) -> None:
        self.lib.forms(w, self.b_wides[i], self.b_cnts[i], None, 1)

    def fused(self, o, i: int, group: int = 8) -> None:
        self.lib.call("orbit_fused", self.a_wide.data_ptr(), self.b_wides[i].data_ptr(), o.data_ptr(), self.m, self.n, self.k,
                      group, 0)

    def corr(self, o, i: int, store: bool = True, a_side: bool = True, r4: bool = False) -> None:
        """O += the corrections; `a_side` False: B's carries' only; `store` False: computed, O untouched (timing); `r4`:
        corr_kernel's 4 rows a thread rather than corr16's 16."""
        b = self.b_lists[i]
        cnt = self.a_count if a_side else self.no_count
        self.lib.call("orbit_corr", self.a_wide.data_ptr(), b["bptr"].data_ptr(), b["bmid"].data_ptr(), b["bidx"].data_ptr(),
                      self.a_list.data_ptr(), cnt.data_ptr(), b["ft"].data_ptr(), o.data_ptr(), self.flag.data_ptr(),
                      self.m, self.n, self.k, int(store) | (2 if r4 else 0))

    def corr4(self, i: int) -> None:
        if self.c4 is None:
            self.c4 = torch.empty(8, self.mh // 32, self.nh, dtype=torch.int32, device="cuda")
        b = self.b_lists[i]
        self.lib.call("orbit_corr4", self.a_wide.data_ptr(), b["bptr"].data_ptr(), b["bmid"].data_ptr(), b["bidx"].data_ptr(),
                      self.c4.data_ptr(), self.m, self.n, self.k)

    def batched(self, i: int) -> None:
        if self.prod is None:
            self.prod = torch.empty(SLOTS, self.mh, self.nh, dtype=torch.int32, device="cuda")
        self.lib.call("orbit_batched", self.a_wide.data_ptr(), self.b_wides[i].data_ptr(), self.prod.data_ptr(), self.m, self.n,
                      self.k)

    def assemble(self, o) -> None:
        self.lib.call("orbit_assemble", self.prod.data_ptr(), o.data_ptr(), self.m, self.n)

    def unfused(self, x, o, i: int) -> None:
        self.forms_a(x)
        self.batched(i)
        self.assemble(o)

    def exact(self, x, o, i: int) -> None:
        self.forms_a(x)
        self.fused(o, i)
        self.corr(o, i)

    def corr_macs(self, i: int) -> dict:
        """The corrections' MACs a call: each B carry adds one A column (Mh rows), each A carry one F_B column (Nh)."""
        nb, na = int(self.b_cnts[i].sum()), int(self.a_cnt.sum())
        return {"b_carries": nb, "a_carries": na, "b_macs": nb * self.mh, "a_macs": na * self.nh,
                "b_rate": nb / (5 * self.nh * self.kh), "a_rate": na / (5 * self.mh * self.kh)}


def gate(lib: Lib, m: int, n: int, k: int, x: torch.Tensor, w: torch.Tensor, corrected: bool) -> dict:
    """The orbit's O against torch._int_mm's on (x, w), bit for bit: fused and unfused, corrected by orbit_corr when
    `corrected` (else the operands must not carry): the fused O by corr16 where it fits, the unfused by corr_kernel."""
    ref = torch._int_mm(x, w.t())
    orb = Orbit(lib, m, n, k)
    orb.load([w])
    o = torch.empty(m, n, dtype=torch.int32, device="cuda")
    doc = {}
    for arm in ("fused", "unfused"):
        o.view(torch.uint8).fill_(0xA5)
        orb.a_cnt.zero_()
        orb.flag.zero_()
        if arm == "fused":
            orb.forms_a(x)
            orb.fused(o, 0)
        else:
            orb.unfused(x, o, 0)
        if corrected:
            torch.cuda.synchronize()
            raw = o.clone()
            orb.corr(o, 0, r4=arm == "unfused")
        torch.cuda.synchronize()
        bad = int((o != ref).sum())
        doc[arm] = {"mismatches": bad, "cells": m * n, "carries_a": orb.a_cnt.tolist(), "carries_b": orb.b_cnts[0].tolist(),
                    "list_overflow": int(orb.flag.item())}
        if corrected:
            doc[arm]["uncorrected_wrong_cells"] = int((raw != ref).sum())
            doc[arm]["corr_macs"] = orb.corr_macs(0)
            del raw
        if bad:
            idx = torch.nonzero(o != ref)[:4].tolist()
            doc[arm]["first"] = [{"at": i, "got": int(o[i[0], i[1]]), "want": int(ref[i[0], i[1]])} for i in idx]
    if corrected:
        doc["corr4_mismatches"] = corr4_check(orb)
    del orb, ref, o
    torch.cuda.empty_cache()
    return doc


def corr4_check(orb: Orbit) -> int:
    """corr4's planes against bits 8 and 9 of B's carries' corrections (orbit_corr's, on a zero O): the cells that differ."""
    mh, nh = orb.mh, orb.nh
    ob = torch.zeros(orb.m, orb.n, dtype=torch.int32, device="cuda")
    orb.corr(ob, 0, a_side=False)
    orb.corr4(0)
    torch.cuda.synchronize()
    sh = torch.arange(32, device="cuda", dtype=torch.int32).view(1, 32, 1)
    bad = 0
    for qq in range(4):
        bits = [((orb.c4[2 * qq + p].unsqueeze(1) >> sh) & 1).reshape(mh, nh) for p in range(2)]
        want = (ob[(qq >> 1) * mh:(qq >> 1) * mh + mh, (qq & 1) * nh:(qq & 1) * nh + nh] >> 8) & 3
        bad += int(((bits[0] + 2 * bits[1]) != want).sum())
    del ob
    return bad


def ext_gate(lib: Lib, m: int, n: int, k: int, x: torch.Tensor, w: torch.Tensor) -> dict:
    """ext_gemm.h's entry (init, load, gemm, free) on (x, w) against torch._int_mm, bit for bit."""
    so, st = lib.so, lib.st()
    h = so.v31_ext_init(m, n, k)
    if not h:
        return {"error": "v31_ext_init refused the shape", "mismatches": None}
    try:
        d = torch.empty(m, n, dtype=torch.int32, device="cuda")
        d.view(torch.uint8).fill_(0xA5)
        torch.cuda.synchronize()
        rc_load = so.v31_ext_load(h, w.data_ptr(), st)
        rc_gemm = so.v31_ext_gemm(h, x.data_ptr(), d.data_ptr(), st) if not rc_load else None
        torch.cuda.synchronize()
        bad = int((d != torch._int_mm(x, w.t())).sum())
    finally:
        so.v31_ext_free(h)
    del d
    torch.cuda.empty_cache()
    return {"load_rc": rc_load, "gemm_rc": rc_gemm, "mismatches": bad if not rc_load and not rc_gemm else None}


def list_overflow_rows(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Gaussian rows, but rows 0 and m/2 all 127: forms_a's slots 0, 2 and 6 carry at every k of half-grid row 0,
    past its list."""
    x = gauss_rows(m, k, gen)
    x[0] = 127
    x[m // 2] = 127
    return x


def load_lt():
    sys.path.insert(0, str(ROOT / "benchmarks" / "torch_gpu" / "s8_linear"))
    import lt as LT
    return LT


def load_wgmma(cubin: Path):
    os.environ["S8_WGMMA_CUBIN"] = str(cubin)
    for p in (ROOT / "tools" / "config", ROOT / "benchmarks" / "torch_gpu" / "s8_linear"):
        sys.path.insert(0, str(p))
    import wgmma as WG
    return WG


class Shape:
    """One (name, N, K, M): the operands, the arms' callables (each one call a weight copy) and their gate records."""

    def __init__(self, lib: Lib, name: str, n: int, k: int, m: int, cubin: Path | None, seed: int,
                 real: torch.Tensor | None = None):
        self.lib, self.name, self.n, self.k, self.m = lib, name, n, k, m
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.copies = max(1, min(8, math.ceil(COPY_BYTES / (n * k))))
        self.x = heavy_rows(m, k, gen)
        self.weights = "gauss" if real is None else "8B layer 0"
        self.ws = [gauss_rows(n, k, gen) if real is None else real.clone() for _ in range(self.copies)]
        self.z = torch.empty(m, n, dtype=torch.int32, device="cuda")
        self.ref = torch._int_mm(self.x, self.ws[0].t())
        self.orb = Orbit(lib, m, n, k)
        self.orb.load(self.ws, share=real is not None)
        self.orb.a_cnt.zero_()
        self.orb.forms_a(self.x)
        torch.cuda.synchronize()
        self.corr_macs = self.orb.corr_macs(0)
        self.gates: dict = {}
        self.cubin = cubin

    def plain_gate(self, arm: str, f1) -> bool:
        self.z.view(torch.uint8).fill_(0xA5)
        f1()
        torch.cuda.synchronize()
        bad = int((self.z != self.ref).sum())
        self.gates[arm] = {"mismatches": bad}
        return bad == 0

    def arms(self, want: tuple[str, ...], clock) -> dict:
        lib, orb, x, ws, z, m, n, k = self.lib, self.orb, self.x, self.ws, self.z, self.m, self.n, self.k
        fns = {}
        for v, arm in enumerate(("cutlass_128x256", "cutlass_128x256_c2", "cutlass_128x128")):
            if arm not in want:
                continue
            def f(i, v=v):
                lib.call("orbit_plain", x.data_ptr(), ws[i].data_ptr(), z.data_ptr(), m, n, k, v)
            try:
                if self.plain_gate(arm, lambda: f(0)):
                    fns[arm] = f
            except RuntimeError as ex:
                self.gates[arm] = {"error": str(ex)}
        if "int_mm" in want:
            def f(i):
                torch._int_mm(x, ws[i].t(), out=z)
            if self.plain_gate("int_mm", lambda: f(0)):
                fns["int_mm"] = f
        if "lt" in want:
            try:
                f = self.lt_arm(clock)
                if f is not None:
                    fns["lt"] = f
            except Exception as ex:  # noqa: BLE001 - a baseline that doesn't load is a record
                self.gates["lt"] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        if "wgmma" in want and self.cubin is not None:
            try:
                f = self.wgmma_arm()
                if f is not None:
                    fns["wgmma"] = f
            except Exception as ex:  # noqa: BLE001
                self.gates["wgmma"] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        o = z
        parts = {
            "forms_a": lambda i: orb.forms_a(x),
            "forms_b": lambda i: orb.forms_b(ws[i], i),
            "batched": lambda i: orb.batched(i),
            "assemble": lambda i: orb.assemble(o),
            "fused": lambda i: orb.fused(o, i),
            "unfused": lambda i: orb.unfused(x, o, i),
            "orbit": lambda i: (orb.forms_a(x), orb.fused(o, i)),
            "corr": lambda i: orb.corr(o, i),
            "corr_b": lambda i: orb.corr(o, i, a_side=False),
            "corr_nostore": lambda i: orb.corr(o, i, store=False),
            "corr_b_nostore": lambda i: orb.corr(o, i, store=False, a_side=False),
            "corr_b_nostore_r4": lambda i: orb.corr(o, i, store=False, a_side=False, r4=True),
            "corr4": lambda i: orb.corr4(i),
            "rmw_o": lambda i: o.add_(1),
            "exact": lambda i: orb.exact(x, o, i),
            "orbit_call": lambda i: (orb.forms_b(ws[i], i), orb.exact(x, o, i)),
        }
        orb.batched(0)
        for arm, f in parts.items():
            if arm in want:
                fns[arm] = f
        if "exact" in fns:
            orb.flag.zero_()
            if not self.plain_gate("exact", lambda: orb.exact(x, o, 0)) or int(orb.flag.item()):
                self.gates["exact"]["list_overflow"] = int(orb.flag.item())
                fns.pop("exact")
                fns.pop("orbit_call", None)
        return fns

    def lt_arm(self, clock):
        LT = load_lt()
        plan = LT.Plan(self.m, self.n, self.k)
        best, best_us, recs = None, None, {}
        for rank, h in enumerate(plan.candidates()):
            nm = LT.name(h, rank)
            try:
                ok = self.plain_gate("lt", lambda: plan.run(h, self.x, self.ws[0], self.z))
            except Exception as ex:  # noqa: BLE001
                torch.cuda.synchronize()
                recs[nm] = {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}
                continue
            if not ok:
                recs[nm] = {"mismatches": self.gates["lt"]["mismatches"]}
                continue
            us = time_once(lambda i: plan.run(h, self.x, self.ws[i], self.z), self.copies, 3)
            recs[nm] = {"us": us, **LT.config(h)}
            if best_us is None or us < best_us:
                best, best_us = h, us
        self.gates["lt"] = {"candidates": recs, "best": None if best is None else LT.name(best, 0)}
        if best is None:
            return None
        return lambda i: plan.run(best, self.x, self.ws[i], self.z)

    def wgmma_arm(self):
        WG = load_wgmma(self.cubin)
        lb = WG.lib()
        sa = torch.ones(self.m, dtype=torch.float32, device="cuda")
        sw = torch.ones(self.n, dtype=torch.float32, device="cuda")
        y = torch.empty(self.m, self.n, dtype=torch.bfloat16, device="cuda")
        cfg = WG.default_cfg(self.m, self.n, self.k, lb.sms)
        WG.scaled_mm(self.x, sa, self.ws[0], sw, y, cfg)
        torch.cuda.synchronize()
        want = self.ref.to(torch.float32).to(torch.bfloat16)
        bad = int((y.view(torch.int16) != want.view(torch.int16)).sum())
        self.gates["wgmma"] = {"cfg": str(cfg), "mismatches_bf16_of_s32": bad,
                               "note": "bf16 out with unit scales; timed as the speed baseline whatever its words"}
        return lambda i: WG.scaled_mm(self.x, sa, self.ws[i], sw, y, cfg)


def time_once(f, copies: int, calls: int) -> float:
    for i in range(copies):
        f(i)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(calls):
        for i in range(copies):
            f(i)
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (calls * copies)


def item(lib: Lib, nv: Nvml | None, f, copies: int, us_est: float, secs: float) -> dict:
    calls = max(1, int(secs * 1e6 / (us_est * copies)))
    stop, mhz = threading.Event(), []
    th = threading.Thread(target=nv.sample, args=(stop, mhz), daemon=True) if nv else None
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if th:
        th.start()
    e0.record()
    for _ in range(calls):
        for i in range(copies):
            f(i)
    e1.record()
    e1.synchronize()
    stop.set()
    if th:
        th.join()
    probe = lib.clock_mhz()
    ms = e0.elapsed_time(e1)
    tail = mhz[len(mhz) // 10:] or mhz
    return {"us": ms * 1e3 / (calls * copies), "calls": calls * copies, "s": ms / 1e3,
            "nvml_mhz": statistics.mean(tail) if tail else None, "clock64_mhz_after": probe}


def summarize(name: str, m: int, n: int, k: int, items: dict, sms: int) -> dict:
    macs = m * n * k
    med = {arm: statistics.median(r["us"] for r in rs) for arm, rs in items.items()}
    plain = {a: med[a] for a in PLAIN if a in med}
    best = min(plain, key=plain.get) if plain else None
    rows = {}
    for arm, us in med.items():
        mhz = statistics.median(r["nvml_mhz"] for r in items[arm] if r["nvml_mhz"]) if any(r["nvml_mhz"] for r in items[arm]) else None
        c64 = statistics.median(r["clock64_mhz_after"] for r in items[arm])
        rows[arm] = {"us": us, "frac_plain": us / plain[best] if best else None, "theta": macs / (us * 1e-6) / PEAK_MACS,
                     "tmacs": macs / us / 1e6, "nvml_mhz": mhz, "clock64_mhz_after": c64,
                     "macs_per_sm_clk": macs / (us * 1e-6) / ((mhz or c64) * 1e6) / sms,
                     "us_all": [round(r["us"], 2) for r in items[arm]]}
    return {"name": name, "m": m, "n": n, "k": k, "plain_best": best, "arms": rows}


def kappa(row: dict, cm: dict, rho_a: float | None) -> dict:
    """κ (a correction MAC's time over the plain GEMM's time per MAC) from the correction arms, and the fraction of plain
    the corrections add: measured, and at B's measured rate with A's at `rho_a`."""
    arms, best = row["arms"], row["plain_best"]
    if best is None:
        return {}
    t_mac = arms[best]["us"] / (row["m"] * row["n"] * row["k"])
    us = {a: arms[a]["us"] for a in ("corr", "corr_b", "corr_nostore", "corr_b_nostore", "corr_b_nostore_r4", "corr4", "rmw_o")
          if a in arms}
    b, a = cm["b_macs"], cm["a_macs"]
    out: dict = {"corr_macs": cm, "per_mac_us_plain": t_mac}
    if "corr" in us:
        out["honest_with_o_rmw"] = us["corr"] / (a + b) / t_mac
    if "corr_nostore" in us:
        out["honest_compute"] = us["corr_nostore"] / (a + b) / t_mac
    if "corr_b_nostore" in us and b:
        out["honest_b"] = us["corr_b_nostore"] / b / t_mac
        if "corr_nostore" in us and a:
            out["honest_a"] = (us["corr_nostore"] - us["corr_b_nostore"]) / a / t_mac
    if "corr_b_nostore_r4" in us and b:
        out["honest_b_r4"] = us["corr_b_nostore_r4"] / b / t_mac
    if "corr4" in us and b:
        out["mod4_b"] = us["corr4"] / b / t_mac
        if "honest_b" in out:
            out["honest_over_mod4_b"] = out["honest_b"] / out["mod4_b"]
    if "rmw_o" in us:
        out["o_rmw_frac_plain"] = us["rmw_o"] / arms[best]["us"]
    if "honest_b" in out and "honest_a" in out:
        for tag, ra in (("measured", cm["a_rate"]), ("rho_a", rho_a)):
            if ra is not None:
                out[f"corr_frac_plain_{tag}"] = 5 / 8 * (out["honest_b"] * cm["b_rate"] + out["honest_a"] * ra)
    return out


def table(rows: list[dict]) -> str:
    out = ["| shape | arm | µs | fraction of plain | θ | T MAC/s | MACs/SM/clk (of 4096) | SM MHz (NVML, clock64 after) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for arm, a in sorted(r["arms"].items(), key=lambda kv: kv[1]["us"]):
            star = " (plain)" if arm == r["plain_best"] else ""
            nv = f"{a['nvml_mhz']:.0f}" if a["nvml_mhz"] else "-"
            out.append(f"| {r['name']} {r['m']}x{r['n']}x{r['k']} | {arm}{star} | {a['us']:.1f} | {a['frac_plain']:.3f} | "
                       f"{a['theta']:.3f} | {a['tmacs']:.1f} | {a['macs_per_sm_clk']:.0f} | {nv}, {a['clock64_mhz_after']:.0f} |")
    out += ["", "| shape | weights | B rate | A rate | κ B (honest) | κ A (honest) | κ B (mod 4) | honest / mod 4 | "
            "honest with O RMW | O RMW / plain | corrections / plain (measured, at ρA) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kp = r.get("kappa", {})
        cm = kp.get("corr_macs", {})

        def f(key, fmt="{:.1f}"):
            return fmt.format(kp[key]) if key in kp else "-"
        out.append(f"| {r['name']} | {r.get('weights', '-')} | {cm.get('b_rate', 0):.4%} | {cm.get('a_rate', 0):.4%} | "
                   f"{f('honest_b')} | {f('honest_a')} | {f('mod4_b')} | {f('honest_over_mod4_b', '{:.2f}')} | "
                   f"{f('honest_with_o_rmw')} | {f('o_rmw_frac_plain', '{:.3f}')} | "
                   f"{f('corr_frac_plain_measured', '{:.4f}')}, {f('corr_frac_plain_rho_a', '{:.4f}')} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", required=True, help="build.sh's liborbit.so")
    ap.add_argument("--cubin", default=None, help="build.sh's s8_wgmma.cubin (the wgmma.linear arm)")
    ap.add_argument("--shapes", default="qkv,o,gate_up")
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--arms", default=",".join(PLAIN + ORBIT_ARMS))
    ap.add_argument("--real", default=None, help="weights.py's cache dir: gate and time on 8B's layer-0 weights")
    ap.add_argument("--rho-a", default=RHO_A, help="A's carry rate a shape, for the corrections' projected cost")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    rho_a = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.rho_a.split(",") if kv}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = Lib(Path(a.lib))
    try:
        nv = Nvml()
    except OSError:
        nv = None
    want = tuple(a.arms.split(","))
    doc = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "lib": lib.info, "m": a.m, "secs": a.secs,
           "reps": a.reps, "gates": {}, "shapes": []}
    print(json.dumps({"lib": lib.info, "device": doc["device"]}), flush=True)
    gen = torch.Generator(device="cuda").manual_seed(7)
    ok_all = True
    real = {}
    if a.real:
        for key in a.shapes.split(","):
            p = Path(a.real) / f"l0_{key}.npy"
            if p.exists():
                real[key] = torch.from_numpy(np.load(p)).cuda()
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        g = {"small": gate(lib, a.m, n, k, small(a.m, k, gen), small(n, k, gen), corrected=False),
             "gauss": gate(lib, a.m, n, k, gauss_rows(a.m, k, gen), gauss_rows(n, k, gen), corrected=True),
             "heavy": gate(lib, a.m, n, k, heavy_rows(a.m, k, gen), gauss_rows(n, k, gen), corrected=True)}
        if name in real:
            g["real_l0"] = gate(lib, a.m, n, k, gauss_rows(a.m, k, gen), real[name], corrected=True)
            g["real_l0_heavy"] = gate(lib, a.m, n, k, heavy_rows(a.m, k, gen), real[name], corrected=True)
        ok = all(v[arm]["mismatches"] == 0 for v in g.values() for arm in ("fused", "unfused"))
        ok = ok and all(v.get("corr4_mismatches", 0) == 0 for v in g.values())
        ok = ok and not any(v[arm]["list_overflow"] for v in g.values() for arm in ("fused", "unfused"))
        g["small"]["no_carries"] = not any(g["small"]["fused"]["carries_a"]) and not any(g["small"]["fused"]["carries_b"])
        ok = ok and g["small"]["no_carries"]
        ext = {"gauss": ext_gate(lib, a.m, n, k, gauss_rows(a.m, k, gen), gauss_rows(n, k, gen)),
               "list_overflow": ext_gate(lib, a.m, n, k, list_overflow_rows(a.m, k, gen), gauss_rows(n, k, gen))}
        if name in real:
            ext["real_l0"] = ext_gate(lib, a.m, n, k, heavy_rows(a.m, k, gen), real[name])
        g["ext"] = ext
        ok = ok and all(v["mismatches"] == 0 for v in ext.values())
        g["ok"] = ok
        ok_all &= ok
        doc["gates"][name] = g
        print(json.dumps({"gate": name, "ok": ok, **{kk: {arm: v[arm]["mismatches"] for arm in ("fused", "unfused")}
                                                      for kk, v in g.items() if isinstance(v, dict) and "fused" in v},
                          "corr4": {kk: v.get("corr4_mismatches") for kk, v in g.items() if isinstance(v, dict) and "fused" in v},
                          "ext": {kk: v["mismatches"] for kk, v in ext.items()}}), flush=True)
    (out / "bench.json").write_text(json.dumps(doc, indent=1) + "\n")
    if a.gate_only:
        return 0 if ok_all else 1
    rows = []
    for si, name in enumerate(a.shapes.split(",")):
        n, k = SHAPES[name]
        sh = Shape(lib, name, n, k, a.m, Path(a.cubin) if a.cubin else None, 100 + si, real.get(name))
        fns = sh.arms(want, None)
        if not doc["gates"][name]["ok"]:
            fns = {arm: f for arm, f in fns.items() if arm in PLAIN}
        est = {arm: time_once(f, sh.copies, 2) for arm, f in fns.items()}
        items: dict = {arm: [] for arm in fns}
        for _ in range(a.reps):
            for arm, f in fns.items():
                items[arm].append(item(lib, nv, f, sh.copies, est[arm], a.secs))
        row = summarize(name, a.m, n, k, items, lib.info["sms"])
        row["copies"] = sh.copies
        row["weights"] = sh.weights
        row["plain_gates"] = sh.gates
        row["kappa"] = kappa(row, sh.corr_macs, rho_a.get(name))
        rows.append(row)
        doc["shapes"] = rows
        (out / "bench.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"shape": name, "plain": row["plain_best"],
                          "arms": {arm: {"us": round(v["us"], 1), "frac": round(v["frac_plain"], 3), "theta": round(v["theta"], 3)}
                                   for arm, v in row["arms"].items()},
                          "kappa": {kk: v for kk, v in row["kappa"].items() if kk != "corr_macs"}}), flush=True)
        del sh, fns
        torch.cuda.empty_cache()
    (out / "table.md").write_text(table(rows))
    print(table(rows), flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
