"""orbit.cu's gates and timings on one sm_90a GPU, against the plain int8 GEMM.

Per shape (N x K at M rows; default 8B's qkv, o and gate-up at M = 16,384):

- **Gates**, before any timing. Each orbit arm's O is compared with `torch._int_mm`'s, bit for bit:
  - on operands with |x|, |w| ≤ 63, where no form carries, O itself (and the forms kernels count no carry);
  - on random-like operands (Gaussian rows quantized at their absmax, as served activations are) and, with `--real DIR`,
    on Llama 3.1 8B's layer-0 weights (weights.py's cache), O plus each carry's correction, which torch adds untimed from
    the forms kernel's carry planes (orbit.py's `products`): exact when the products and assembly are.
  An arm that fails a gate isn't timed.
- **Timings:** 2 s items, arm by arm, interleaved over `--reps` rounds after a warm-up, each item rotating through enough
  weight copies to leave L2. Every item records its time per call, the SM clock NVML reads through it and a clock64 probe
  right after it. The plain GEMM is the fastest of cuBLASLt (autotuned, s8_linear's lt.py), torch._int_mm, wgmma.linear
  (s8_linear's kernel, bf16 out) and CUTLASS's cooperative TMA GEMMs; every arm is reported as a fraction of it, as θ
  (MACs a second over 989.4 T) and as MACs per SM clock (over 4,096).

Arms: `lt`, `int_mm`, `wgmma`, `cutlass_128x256`, `cutlass_128x256_c2`, `cutlass_128x128` (plain); `forms_a`, `forms_b`,
`batched`, `assemble`, `fused` (the orbit's parts, forms prepared); `unfused` (forms_a + batched + assemble), `orbit`
(forms_a + fused) and `orbit_call` (forms_b + forms_a + fused: ext_gemm.h's call, whose load runs every served call). The
orbit's products are on wrapped forms, so its timed O lacks the carries' corrections; their counts are in the record.

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
SLOTS = len(OR.ORDER)
COPY_BYTES = 256 << 20

vp, i32, i64 = C.c_void_p, C.c_int, C.c_longlong


class Lib:
    """liborbit.so on torch's current stream."""

    SIGS = {"orbit_info": [vp], "orbit_forms": [vp, vp, vp, vp, i32, i32, i32, vp],
            "orbit_batched": [vp, vp, vp, i32, i32, i32, vp], "orbit_assemble": [vp, vp, i32, i32, vp],
            "orbit_fused": [vp, vp, vp, i32, i32, i32, i32, i32, vp], "orbit_plain": [vp, vp, vp, i32, i32, i32, i32, vp],
            "orbit_clock": [vp, i64, vp]}

    def __init__(self, path: Path):
        self.so = C.CDLL(str(path))
        for f, a in self.SIGS.items():
            getattr(self.so, f).argtypes, getattr(self.so, f).restype = a, i32
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

    def forms(self, x, wide, counts, carry, which: int) -> None:
        self.call("orbit_forms", x.data_ptr(), wide.data_ptr(), counts.data_ptr(), carry.data_ptr() if carry is not None else None,
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


class Orbit:
    """One shape's orbit buffers: A's forms (wide, carry plane, counts) and, per weight copy, B's."""

    def __init__(self, lib: Lib, m: int, n: int, k: int):
        self.lib, self.m, self.n, self.k = lib, m, n, k
        self.mh, self.nh, self.kh = m // 2, n // 2, k // 2
        self.a_wide = torch.empty(self.mh, SLOTS * self.kh, dtype=torch.int8, device="cuda")
        self.a_cnt = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
        self.b_wides: list[torch.Tensor] = []
        self.b_cnts: list[torch.Tensor] = []
        self.prod = None

    def load(self, ws: list[torch.Tensor]) -> None:
        self.b_wides = [torch.empty(self.nh, SLOTS * self.kh, dtype=torch.int8, device="cuda") for _ in ws]
        self.b_cnts = [torch.zeros(SLOTS, dtype=torch.int64, device="cuda") for _ in ws]
        for w, bw, bc in zip(ws, self.b_wides, self.b_cnts):
            self.lib.forms(w, bw, bc, None, 1)

    def forms_a(self, x) -> None:
        self.lib.forms(x, self.a_wide, self.a_cnt, None, 0)

    def forms_b(self, w, i: int) -> None:
        self.lib.forms(w, self.b_wides[i], self.b_cnts[i], None, 1)

    def fused(self, o, i: int, group: int = 8) -> None:
        self.lib.call("orbit_fused", self.a_wide.data_ptr(), self.b_wides[i].data_ptr(), o.data_ptr(), self.m, self.n, self.k,
                      group, 0)

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


def correction(lib: Lib, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """ΔO [M, N] int64: what each carry adds to the wrapped products' O (orbit.py's `products`, assembled), from the
    forms kernel's wide forms and carry planes, with torch's int8 GEMMs; and the carries per slot, A's and B's."""
    m, k = x.shape
    n = w.shape[0]
    mh, nh, kh = m // 2, n // 2, k // 2
    aw = torch.empty(mh, SLOTS * kh, dtype=torch.int8, device="cuda")
    bw = torch.empty(nh, SLOTS * kh, dtype=torch.int8, device="cuda")
    ac = torch.zeros(SLOTS, mh, kh, dtype=torch.int8, device="cuda")
    bc = torch.zeros(SLOTS, nh, kh, dtype=torch.int8, device="cuda")
    an = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
    bn = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
    lib.forms(x, aw, an, ac, 0)
    lib.forms(w, bw, bn, bc, 1)
    quad = {q: torch.zeros(mh, nh, dtype=torch.int64, device="cuda") for q in OR.QUADRANTS}
    for s, prod in enumerate(OR.ORDER):
        wa = aw[:, s * kh:(s + 1) * kh].contiguous()
        wb = bw[:, s * kh:(s + 1) * kh].contiguous()
        ca, cb = ac[s], bc[s]
        if not (int(an[s]) or int(bn[s])):
            continue
        d = (torch._int_mm(ca, wb.t()).to(torch.int64) + 256 * torch._int_mm(ca, cb.t()).to(torch.int64)
             + torch._int_mm(wa, cb.t()).to(torch.int64)) * 256
        for q, terms in OR.QUADRANTS.items():
            if prod in terms:
                quad[q] += terms[prod] * d
    out = torch.cat([torch.cat([quad[(0, 0)], quad[(0, 1)]], 1), torch.cat([quad[(1, 0)], quad[(1, 1)]], 1)], 0)
    counts = {"a": an.tolist(), "b": bn.tolist(), "a_rate": float(an.sum()) / (SLOTS * mh * kh),
              "b_rate": float(bn.sum()) / (SLOTS * nh * kh)}
    return out, counts


def gate(lib: Lib, m: int, n: int, k: int, x: torch.Tensor, w: torch.Tensor, corrected: bool) -> dict:
    """The orbit's O (fused and unfused) against torch._int_mm's on (x, w): bit for bit, after the torch correction
    when `corrected`."""
    ref = torch._int_mm(x, w.t())
    orb = Orbit(lib, m, n, k)
    orb.load([w])
    o = torch.empty(m, n, dtype=torch.int32, device="cuda")
    doc = {}
    delta, counts = correction(lib, x, w) if corrected else (None, None)
    for arm in ("fused", "unfused"):
        o.view(torch.uint8).fill_(0xA5)
        orb.a_cnt.zero_()
        if arm == "fused":
            orb.forms_a(x)
            orb.fused(o, 0)
        else:
            orb.unfused(x, o, 0)
        torch.cuda.synchronize()
        got = wrap32(o.to(torch.int64) + delta) if corrected else o
        bad = int((got != ref).sum())
        doc[arm] = {"mismatches": bad, "cells": m * n, "carries_a": orb.a_cnt.tolist(), "carries_b": orb.b_cnts[0].tolist()}
        if bad:
            idx = torch.nonzero(got != ref)[:4].tolist()
            doc[arm]["first"] = [{"at": i, "got": int(got[i[0], i[1]]), "want": int(ref[i[0], i[1]])} for i in idx]
    if corrected:
        doc["carries"] = counts
        doc["uncorrected_wrong_cells"] = int((o != ref).sum())
    del orb, ref, o, delta
    torch.cuda.empty_cache()
    return doc


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

    def __init__(self, lib: Lib, name: str, n: int, k: int, m: int, cubin: Path | None, seed: int):
        self.lib, self.name, self.n, self.k, self.m = lib, name, n, k, m
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.copies = max(1, min(8, math.ceil(COPY_BYTES / (n * k))))
        self.x = gauss_rows(m, k, gen)
        self.ws = [gauss_rows(n, k, gen) for _ in range(self.copies)]
        self.z = torch.empty(m, n, dtype=torch.int32, device="cuda")
        self.ref = torch._int_mm(self.x, self.ws[0].t())
        self.orb = Orbit(lib, m, n, k)
        self.orb.load(self.ws)
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
            "orbit_call": lambda i: (orb.forms_b(ws[i], i), orb.forms_a(x), orb.fused(o, i)),
        }
        orb.batched(0)
        for arm, f in parts.items():
            if arm in want:
                fns[arm] = f
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
                     "nvml_mhz": mhz, "clock64_mhz_after": c64,
                     "macs_per_sm_clk": macs / (us * 1e-6) / ((mhz or c64) * 1e6) / sms,
                     "us_all": [round(r["us"], 2) for r in items[arm]]}
    return {"name": name, "m": m, "n": n, "k": k, "plain_best": best, "arms": rows}


def table(rows: list[dict]) -> str:
    out = ["| shape | arm | µs | fraction of plain | θ | MACs/SM/clk (of 4096) | SM MHz (NVML, clock64 after) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        for arm, a in sorted(r["arms"].items(), key=lambda kv: kv[1]["us"]):
            star = " (plain)" if arm == r["plain_best"] else ""
            nv = f"{a['nvml_mhz']:.0f}" if a["nvml_mhz"] else "-"
            out.append(f"| {r['name']} {r['m']}x{r['n']}x{r['k']} | {arm}{star} | {a['us']:.1f} | {a['frac_plain']:.3f} | "
                       f"{a['theta']:.3f} | {a['macs_per_sm_clk']:.0f} | {nv}, {a['clock64_mhz_after']:.0f} |")
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
    ap.add_argument("--arms", default=",".join(PLAIN + ("forms_a", "forms_b", "batched", "assemble", "fused", "unfused", "orbit",
                                                        "orbit_call")))
    ap.add_argument("--real", default=None, help="weights.py's cache dir: gate on 8B's layer-0 weights too")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
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
             "gauss": gate(lib, a.m, n, k, gauss_rows(a.m, k, gen), gauss_rows(n, k, gen), corrected=True)}
        if name in real:
            g["real_l0"] = gate(lib, a.m, n, k, gauss_rows(a.m, k, gen), real[name], corrected=True)
        ok = all(v[arm]["mismatches"] == 0 for v in g.values() for arm in ("fused", "unfused"))
        g["small"]["no_carries"] = not any(g["small"]["fused"]["carries_a"]) and not any(g["small"]["fused"]["carries_b"])
        ok = ok and g["small"]["no_carries"]
        g["ok"] = ok
        ok_all &= ok
        doc["gates"][name] = g
        print(json.dumps({"gate": name, "ok": ok, **{kk: {arm: v[arm]["mismatches"] for arm in ("fused", "unfused")}
                                                      for kk, v in g.items() if isinstance(v, dict)}}), flush=True)
    (out / "bench.json").write_text(json.dumps(doc, indent=1) + "\n")
    if a.gate_only:
        return 0 if ok_all else 1
    rows = []
    for si, name in enumerate(a.shapes.split(",")):
        n, k = SHAPES[name]
        sh = Shape(lib, name, n, k, a.m, Path(a.cubin) if a.cubin else None, 100 + si)
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
        row["plain_gates"] = sh.gates
        row["carries_b"] = [c.tolist() for c in sh.orb.b_cnts]
        rows.append(row)
        doc["shapes"] = rows
        (out / "bench.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"shape": name, "plain": row["plain_best"],
                          "arms": {arm: {"us": round(v["us"], 1), "frac": round(v["frac_plain"], 3), "theta": round(v["theta"], 3)}
                                   for arm, v in row["arms"].items()}}), flush=True)
        del sh, fns
        torch.cuda.empty_cache()
    (out / "table.md").write_text(table(rows))
    print(table(rows), flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
