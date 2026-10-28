"""`int8_wgmma` (the ping-pong wgmma plan of `int8_fused.scaled_mm`) against `exactlin_scaled_mm_tma` at its tuned plan and
cuBLASLt int8, on one leased H200: every arm gated bit for bit before it is timed, then timed sustained and interleaved.

Gate, per shape and input kind, in this process before any timing (a failed gate times nothing, exit 4): y of every
`int8_wgmma` variant, each into a y poisoned with 0xA5 bytes, equals `exactlin_scaled_mm_tma`'s at TRITON and the reference,
an exact s32 product (float64 GEMM of the int8 operands: every partial sum an integer under 2^53) with the epilogue in torch
(float32 cvt, * s_a, * s_w, to bf16), word for word.  Input kinds: `served` (bf16 rows with log-normal channel scales and
outlier channels, quantized by `int8_fused.quant_rows`, and heavy-tailed weight rows each scaled to its absmax, so most
words are small as a checkpoint's are: the power, and so the clock, of serving), `random` (int8 over all of [-128, 127],
log-uniform scales: the most power) and
`extremes` (rows and weight rows all -128 or all 127 among random ones).  EDGES adds shapes whose M, N and K are not
multiples of any tile.

Timing, per shape: each arm's calls (one per copy of X and W, cycling over enough copies to pass the L2) captured in a
CUDA graph, warmed, then items of at least --secs seconds, the arms alternating item by item (order rotated each round).
Around every item, `stamp.cu` reads each SM's %clock64 and %globaltimer (one CTA an SM), so an item reports its declared
MACs a SM clock against 4,096 (the wgmma s8 rate a clock), its SM clock, and its rate against
theta = 989.4 T int8 MAC/s (4,096 x 132 x 1,830 MHz).  `dcgmi dmon` samples Tensor Active (1004), SM clock and power
beside the run; an item's mean is over the samples inside it.  Back to back, every arm holds the card at its power cap
(about 1,350 to 1,600 MHz); a served step's GEMMs run near 1,713 MHz between memory-bound kernels.  `--rest R` puts the SMs
to sleep after each call (`stamp.cu`'s rest, R times the call's clocks at 0.7 of the rate) and takes their rest clocks out
of the span, so an item's rate is the GEMM's at that lower mean power and higher clock.  `--plan` times
`int8_fused._plan`'s Triton config beside TRITON: what serves today where the shape has no row.

`--zy` gates the z/y-record kernels too (`int8_wgmma.scaled_mm_zy`) and times them in place of the plain ones: the same
shapes and kinds, an edge's N rounded up to whole 32-column units and its call over T = M rounded up to whole 32-row units
plus 160 slots (a row tile past M), a served shape's over T = M.  Each variant's y and records (into 0xA5-poisoned ones,
twice) equal the reference's, its y and `int8_exact.zy_records` of the exact s32 product over the T slots (zero codes under
scale 0 past M) and its dequant, and `exactlin_scaled_mm_zy_tma`'s at TRITON, byte for byte.  The Triton arms are then
`scaled_mm_zy` at TRITON and (`--plan`) at `_plan`'s config, what serves the records today.

    python wgbench.py --out OUT [--ms 8192,16384,32768] [--linears qkv,o,gate_up,down] [--variants all] [--secs 2]
                      [--rounds 2] [--no-lt] [--gate-only] [--no-edges] [--time-kinds served,random] [--plan] [--rest R]
                      [--zy]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_fused as X
from verity_vllm.engine import int8_wgmma as W

HERE = Path(__file__).resolve().parent
THETA, F_REF, PER_CLK = 989.4e12, 1.83e9, 4096
LINEARS = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192)}
TRITON = "128x128x128:w4:s3:k1:t1"  # the tuned Triton tile for Llama-3.2-1B's linears on sm_90 at M 8,192 to 32,768
EDGES = ((8193, 3072, 2048), (1000, 3000, 2064), (129, 136, 160), (8191, 2050, 8192), (300, 16384, 2048), (17, 256, 16))
KINDS = ("served", "random", "extremes")
UR, UC = E.UNIT
ZY_EDGES = tuple((m, -(-n // UC) * UC, k) for m, n, k in EDGES)


def zy_slots(m: int, edge: bool) -> int:
    return -(-m // UR) * UR + (160 if edge else 0)


def stamp_lib(tmp: Path) -> ctypes.CDLL:
    so = tmp / "libstamp.so"
    subprocess.run([W.nvcc(), "-O2", "-gencode", "arch=compute_90a,code=sm_90a", "-Xcompiler", "-fPIC", "-shared", "-o",
                    str(so), str(HERE / "stamp.cu")], check=True)
    dll = ctypes.CDLL(str(so))
    dll.stamp.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    dll.rest.argtypes = [ctypes.c_longlong, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    dll.stamp.restype = dll.rest.restype = ctypes.c_int
    return dll


class Clock:
    """Stamps of every SM's %clock64 and %globaltimer on the current stream, and the clocks each SM spent in `rest`."""

    def __init__(self, dll: ctypes.CDLL):
        self.dll = dll
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.buf = torch.zeros(2, self.sms * 3, dtype=torch.int64, device="cuda")
        self.acc = torch.zeros(1024, dtype=torch.int64, device="cuda")  # by %smid

    def mark(self, i: int) -> None:
        if i == 0:
            self.acc.zero_()
        if self.dll.stamp(self.buf[i].data_ptr(), self.sms, torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("stamp launch failed")

    def rest(self, cycles: int) -> None:
        if self.dll.rest(cycles, self.acc.data_ptr(), self.sms, torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("rest launch failed")

    def span(self) -> dict:
        a, b = (v.view(-1, 3).cpu().tolist() for v in self.buf)
        rest = self.acc.cpu().tolist()
        sa, sb = {r[0]: r for r in a}, {r[0]: r for r in b}
        common = sorted(set(sa) & set(sb))
        clk = [sb[s][1] - sa[s][1] for s in common]
        ns = [sb[s][2] - sa[s][2] for s in common]
        return {"sms": len(common), "clk_mean": sum(clk) / len(clk), "ns_mean": sum(ns) / len(ns),
                "rest_mean": sum(rest[s] for s in common) / len(common),
                "mhz": sum(c / n * 1e3 for c, n in zip(clk, ns)) / len(clk)}


class Dmon:
    """`dcgmi dmon` beside the run: (time, tensor active, SM MHz, watts) samples every 100 ms."""

    def __init__(self, gpu: int = 0):
        self.rows: list[tuple[float, float, float, float]] = []
        self.p = None
        if shutil.which("dcgmi") is None:  # items then carry no Tensor Active
            return
        self.p = subprocess.Popen(["dcgmi", "dmon", "-i", str(gpu), "-e", "1004,100,155", "-d", "100"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            parts = line.split()
            # a sample is "GPU <i> <tensor active> <MHz> <W>"; N/A fields leave it out
            if len(parts) >= 5 and parts[0] == "GPU" and all(v.replace(".", "", 1).isdigit() for v in parts[2:5]):
                self.rows.append((time.time(), float(parts[2]), float(parts[3]), float(parts[4])))

    def between(self, t0: float, t1: float) -> dict:
        got = [r for r in self.rows if t0 <= r[0] <= t1]
        if not got:
            return {}
        n = len(got)
        return {"tensor_active": sum(r[1] for r in got) / n, "dcgm_mhz": sum(r[2] for r in got) / n,
                "watts": sum(r[3] for r in got) / n, "samples": n}

    def stop(self):
        if self.p:
            self.p.terminate()


def operands(m: int, n: int, k: int, kind: str, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "served":
        ch = torch.exp(torch.randn(k, device="cuda", generator=g) * 0.8)
        ch[torch.randint(0, k, (max(1, k // 256),), device="cuda", generator=g)] *= 30.0
        x = (torch.randn(m, k, device="cuda", generator=g) * ch).bfloat16()
        xq = torch.empty(m, k, dtype=torch.int8, device="cuda")
        sa = torch.empty(m, dtype=torch.float32, device="cuda")
        X.quant_rows(x, xq, sa)
        # weights as a checkpoint's quantize: heavy-tailed rows, each scaled to its absmax (most words small)
        wf = torch.randn(n, k, device="cuda", generator=g) * torch.exp(torch.randn(n, k, device="cuda", generator=g) * 0.5)
        amax = wf.abs().amax(1, keepdim=True)
        w = (wf * (127.0 / amax)).round().to(torch.int8)
        sw = (amax[:, 0] / 127.0 * 0.02).float()
        del wf
    else:
        xq = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda", generator=g)
        w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda", generator=g)
        if kind == "extremes":
            for t, rows in ((xq, m), (w, n)):
                r = torch.arange(rows, device="cuda")
                t[r % 7 == 0] = -128
                t[r % 7 == 3] = 127
        sa = torch.exp(torch.empty(m, device="cuda").uniform_(-9, 2, generator=g)).float()
        sw = torch.exp(torch.empty(n, device="cuda").uniform_(-9, 2, generator=g)).float()
    return xq, sa, w, sw


def reference(xq, sa, w, sw) -> torch.Tensor:
    z = (xq.double() @ w.double().T).to(torch.int32)
    return ((z.float() * sa[:, None]) * sw[None, :]).to(torch.bfloat16)


def poisoned(m: int, n: int) -> torch.Tensor:
    return torch.full((m, n), 0xA5A5 - 0x10000, dtype=torch.int16, device="cuda").view(torch.bfloat16)


def gate(m, n, k, variants, kinds=KINDS, seed=0) -> list[dict]:
    rows = []
    tri = X.Cfg.parse(TRITON)
    for kind in kinds:
        xq, sa, w, sw = operands(m, n, k, kind, seed)
        ref = reference(xq, sa, w, sw).view(torch.int16)
        yt = poisoned(m, n)
        row = {"shape": [m, n, k], "kind": kind}
        try:
            X.scaled_mm(xq, sa, w, sw, yt, tri)
            row["triton_vs_ref"] = int((yt.view(torch.int16) != ref).sum())
        except Exception as e:  # noqa: BLE001  (a shape Triton's descriptors refuse is gated against the reference alone)
            row["triton_refused"] = repr(e)[:200]
        for v in variants:
            for call in (0, 1):
                y = poisoned(m, n)
                W.scaled_mm(xq, sa, w, sw, y, W.WgCfg.parse(v))
                torch.cuda.synchronize()
                bad = int((y.view(torch.int16) != ref).sum())
                row[f"{v}#{call}"] = bad
        row["ok"] = all(v == 0 for key, v in row.items() if key not in ("shape", "kind", "ok", "triton_refused"))
        rows.append(row)
        del xq, w, ref, yt
    return rows


def gate_zy(m, n, k, rows, variants, kinds=KINDS, seed=0) -> list[dict]:
    """Per kind, each arm's differing y words plus differing record bytes against the reference."""
    out = []
    tri = X.Cfg.parse(TRITON)
    units = E.zy_units(m, n, rows)
    for kind in kinds:
        xq, sa, w, sw = operands(m, n, k, kind, seed)
        z = torch.zeros(rows, n, dtype=torch.int32, device="cuda")
        z[:m] = (xq.double() @ w.double().T).to(torch.int32)
        scales = torch.zeros(rows, dtype=torch.float32, device="cuda")
        scales[:m] = sa
        yr = E.dequantize(z, scales, sw)
        ref_y, ref_zy = yr[:m].view(torch.int16).clone(), E.zy_records(z, yr)
        del z, yr

        def bad(y, zy):
            return int((y.view(torch.int16) != ref_y).sum()) + int((zy != ref_zy).sum())
        row = {"shape": [m, n, k], "rows": rows, "kind": kind, "zy": True}
        y, zy = poisoned(m, n), torch.full((units, E.ZY_BYTES), 0xA5, dtype=torch.uint8, device="cuda")
        try:
            X.scaled_mm_zy(xq, sa, w, sw, y, zy, rows, tri)
            row["triton_vs_ref"] = bad(y, zy)
        except Exception as e:  # noqa: BLE001  (as `gate`)
            row["triton_refused"] = repr(e)[:200]
        for v in variants:
            text, _, flags = v.partition("#")
            for call in (0, 1):
                y = poisoned(m, n)
                zy.fill_(0xA5)
                W.scaled_mm_zy(xq, sa, w, sw, y, zy, rows, W.WgCfg.parse(text), int(flags or 0))
                torch.cuda.synchronize()
                row[f"{v}#{call}"] = bad(y, zy)
        row["ok"] = all(v == 0 for key, v in row.items() if key not in ("shape", "rows", "kind", "zy", "ok", "triton_refused"))
        out.append(row)
        del xq, w, y, zy, ref_y, ref_zy
    return out


def arm_fn(arm: str, ops: list, ys: list, after=lambda: None, zys=None):
    """An arm's calls over the copies; the zy arms (triton-zy, plan-zy, wgzy:CFG#FLAGS) write records over T = M slots."""
    m, k = ops[0][0].shape
    n = ops[0][2].shape[0]
    if arm in ("triton", "plan", "triton-zy", "plan-zy"):
        cfg = (X.Cfg.parse(TRITON), None) if arm.startswith("triton") else X._plan(m, n, k, ops[0][0].device)
        if arm.endswith("-zy"):
            return lambda: [(X.scaled_mm_zy(xq, sa, w, sw, y, zy, m, *cfg), after())
                            for (xq, sa, w, sw), y, zy in zip(ops, ys, zys)]
        return lambda: [(X.scaled_mm(xq, sa, w, sw, y, *cfg), after()) for (xq, sa, w, sw), y in zip(ops, ys)]
    if arm == "lt":
        return lambda: [(torch._int_mm(xq, w.t()), after()) for (xq, sa, w, sw), y in zip(ops, ys)]
    if arm.startswith("wgzy:"):
        text, _, flags = arm.removeprefix("wgzy:").partition("#")
        cfg, flags = W.WgCfg.parse(text), int(flags or 0)
        return lambda: [(W.scaled_mm_zy(xq, sa, w, sw, y, zy, m, cfg, flags), after())
                        for (xq, sa, w, sw), y, zy in zip(ops, ys, zys)]
    text, _, flags = arm.removeprefix("wg:").partition("#")
    cfg, flags = W.WgCfg.parse(text), int(flags or 0)
    return lambda: [(W.scaled_mm(xq, sa, w, sw, y, cfg, flags), after()) for (xq, sa, w, sw), y in zip(ops, ys)]


def timed(m, n, k, arms, secs, rounds, clock, dmon, name, kind, rest=0.0) -> list[dict]:
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    copies = max(2, math.ceil(2 * l2 / (n * k + m * k)))
    ops = [operands(m, n, k, kind, 100 + c) for c in range(copies)]
    ys = [torch.empty(m, n, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
    zys = None
    if any(a.endswith("-zy") or a.startswith("wgzy:") for a in arms):
        zys = [torch.empty((E.zy_units(m, n, m), E.ZY_BYTES), dtype=torch.uint8, device="cuda") for _ in range(copies)]
    graphs, per_replay = {}, {}
    # a call's rest: `rest` times its clocks at 0.7 of the s8 rate
    cycles = int(rest * m * n * k / (0.7 * PER_CLK * clock.sms))
    after = (lambda: clock.rest(cycles)) if cycles else (lambda: None)
    for arm in arms:
        f = arm_fn(arm, ops, ys, after, zys)
        f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        for _ in range(3):
            g.replay()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(3):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        graphs[arm], per_replay[arm] = g, e0.elapsed_time(e1) / 3e3
    items = []
    macs_call = m * n * k
    for r in range(rounds):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for arm in order:
            g = graphs[arm]
            reps = max(1, math.ceil(secs / per_replay[arm]))
            torch.cuda.synchronize()
            t0 = time.time()
            clock.mark(0)
            for _ in range(reps):
                g.replay()
            clock.mark(1)
            torch.cuda.synchronize()
            t1 = time.time()
            sp = clock.span()
            calls = reps * copies
            # with rests, a call's time is the span's clocks less the SMs' rest, at the span's mean clock
            busy = sp["clk_mean"] - sp["rest_mean"]
            sec = sp["ns_mean"] / 1e9 * busy / sp["clk_mean"]
            macs = macs_call * calls
            it = {"linear": name, "shape": [m, n, k], "kind": kind, "arm": arm, "round": r, "calls": calls, "secs": sec,
                  "us_per_call": sec / calls * 1e6, "theta_frac": macs / (sec * THETA), "sm_mhz": sp["mhz"],
                  "per_clk_frac": macs / (busy * sp["sms"] * PER_CLK), "sms": sp["sms"],
                  "rest_frac": sp["rest_mean"] / sp["clk_mean"]}
            it.update(dmon.between(t0 + 0.2, t1 - 0.1))
            items.append(it)
            print(json.dumps(it), flush=True)
    del graphs
    return items


def summary(items: list[dict]) -> str:
    out = ["| linear | M | data | arm | us a call | of theta (1,830 MHz) | SM MHz | MACs/SM-clk / 4096 | Tensor Active | W |",
           "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    key = {}
    for it in items:
        key.setdefault((it["linear"], it["shape"][0], it["kind"], it["arm"]), []).append(it)
    for (lin, m, kind, arm), its in key.items():
        def mean(f, its=its):
            got = [i[f] for i in its if f in i]
            return sum(got) / len(got) if got else None
        ta, watts = mean("tensor_active"), mean("watts")
        out.append(f"| {lin} | {m} | {kind} | {arm} | {mean('us_per_call'):.1f} | {mean('theta_frac'):.4f} | {mean('sm_mhz'):.0f} | "
                   f"{mean('per_clk_frac'):.4f} | {'' if ta is None else f'{ta:.3f}'} | {'' if watts is None else f'{watts:.0f}'} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default="8192,16384,32768")
    ap.add_argument("--linears", default="gate_up,qkv,o,down")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-lt", action="store_true")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--time-kinds", default="served", help="input kinds timed (served, random)")
    ap.add_argument("--probes", default="", help="extra timed arms CFG#FLAGS (int8_wgmma.scaled_mm's flags), not gated")
    ap.add_argument("--plan", action="store_true", help="time `int8_fused._plan`'s Triton config too (what serves today)")
    ap.add_argument("--rest", type=float, default=0.0,
                    help="the SMs asleep after each call for this many times its clocks at 0.7 of the rate (a served step's clock)")
    ap.add_argument("--zy-variants", default="", help="the zy kernels gated and timed, CFG#FLAGS (default --variants)")
    ap.add_argument("--zy", action="store_true",
                    help="gate the z/y-record kernels too and time them (probes CFG#FLAGS too, wg:CFG#FLAGS the plain kernel)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = [f"{v}:g8" for v in W.VARIANTS] if a.variants == "all" else a.variants.split(",")
    zy_variants = a.zy_variants.split(",") if a.zy_variants else variants
    lib = W.lib()
    info = {"gpu": torch.cuda.get_device_name(0), "layout": W.layout(), "triton": TRITON, "variants": variants,
            "ptxas": (Path(lib._name).with_name("ptxas.txt")).read_text()}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    ms = [int(v) for v in a.ms.split(",")]
    shapes = [(lin, m, *LINEARS[lin]) for m in ms for lin in a.linears.split(",")]
    gates = []
    for m, n, k in () if a.no_edges else EDGES:
        gates += gate(m, n, k, variants)
    for _, m, n, k in shapes:
        gates += gate(m, n, k, variants, seed=m)
    if a.zy:
        for m, n, k in () if a.no_edges else ZY_EDGES:
            gates += gate_zy(m, n, k, zy_slots(m, True), zy_variants)
            torch.cuda.empty_cache()
        for _, m, n, k in shapes:
            gates += gate_zy(m, n, k, zy_slots(m, False), zy_variants, seed=m)
            torch.cuda.empty_cache()
    with (out / "gates.jsonl").open("w") as f:
        for g in gates:
            f.write(json.dumps(g) + "\n")
    bad = [g for g in gates if not g["ok"]]
    print(f"gates: {len(gates) - len(bad)}/{len(gates)} ok", flush=True)
    if bad:
        print(json.dumps(bad, indent=1))
        return 4
    if a.gate_only:
        return 0
    tmp = Path(tempfile.mkdtemp())
    clock, dmon = Clock(stamp_lib(tmp)), Dmon()
    if a.zy:
        arms = (["triton-zy"] + (["plan-zy"] if a.plan else []) + [f"wgzy:{v}" for v in zy_variants]
                + [v if v.startswith("wg:") else f"wgzy:{v}" for v in a.probes.split(",") if v]
                + ([] if a.no_lt else ["lt"]))
    else:
        arms = (["triton"] + (["plan"] if a.plan else []) + [f"wg:{v}" for v in variants]
                + [f"wg:{v}" for v in a.probes.split(",") if v] + ([] if a.no_lt else ["lt"]))
    items = []
    try:
        for lin, m, n, k in shapes:
            for kind in a.time_kinds.split(","):
                items += timed(m, n, k, arms, a.secs, a.rounds, clock, dmon, lin, kind, a.rest)
                torch.cuda.empty_cache()
    finally:
        dmon.stop()
        with (out / "items.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        (out / "summary.md").write_text(summary(items))
        (out / "dmon.jsonl").write_text("\n".join(json.dumps(r) for r in dmon.rows) + "\n")
    print(summary(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
