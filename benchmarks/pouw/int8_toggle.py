"""int8_toggle: does an int8 GEMM run faster on narrow operands? Sustained `torch._int_mm` throughput, SM clock and power
on one GPU for the operand patterns a zero instance feeds the tensor pipe: honest's
wide operands, CO-Dither's wide U against a bit (B), both-sided dither's bit against a bit (A), and zeros (record's clean
product). A pattern that runs faster than `wide` is ε_clock a cheater gets from it on the binding facet.

    python benchmarks/pouw/int8_toggle.py --out DIR [--n 8192] [--seconds 8] [--rounds 3]

Run it under a gpu-lease (`GPU_LEASE_UUID` names the GPU `nvidia-smi` samples every 100 ms). Clocks are read, never set.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

#: pattern: (left operand, right operand)
PATTERNS = {"full": ("full", "full"), "wide": ("wide", "wide"), "u-bit": ("wide", "bit"), "bit": ("bit", "bit"),
            "zero": ("zero", "zero")}
#: operand: [lo, hi) of its uniform int8 entries
RANGES = {"full": (-128, 128), "wide": (-64, 64), "bit": (0, 2), "zero": (0, 1)}
SMI = ("clocks.sm", "power.draw", "clocks_throttle_reasons.active")


def operand(kind: str, shape: tuple[int, int], gen: torch.Generator, device: str) -> torch.Tensor:
    lo, hi = RANGES[kind]
    return torch.randint(lo, hi, shape, generator=gen, device=device, dtype=torch.int8)


class Sampler:
    """`nvidia-smi -lms 100` in the background, each line stamped on arrival."""

    def __init__(self, gpu: str):
        self.rows: list[tuple[float, list[str]]] = []
        self.p = subprocess.Popen(["nvidia-smi", f"--query-gpu={','.join(SMI)}", "--format=csv,noheader,nounits",
                                   "-i", gpu, "-lms", "100"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.rows.append((time.monotonic(), [x.strip() for x in line.split(",")]))

    def window(self, t0: float, t1: float) -> dict:
        got = [r for t, r in self.rows if t0 <= t <= t1]
        num = [(float(r[0]), float(r[1])) for r in got
               if len(r) > 1 and all(x.replace(".", "", 1).isdigit() for x in r[:2])]
        return {"samples": len(got), "clock_mhz": statistics.median(c for c, _ in num) if num else None,
                "power_w": statistics.median(p for _, p in num) if num else None,
                "throttle": sorted({r[2] for r in got if len(r) > 2})}

    def stop(self):
        self.p.terminate()


def phase(a: torch.Tensor, b: torch.Tensor, seconds: float, smi: Sampler) -> dict:
    """Back-to-back GEMMs for `seconds`: the first half settles clocks and power, the second is timed."""
    n = a.shape[0]
    torch._int_mm(a, b)
    torch.cuda.synchronize()
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end - seconds / 2:
        for _ in range(8):
            torch._int_mm(a, b)
        torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0, reps = time.monotonic(), 0
    e0.record()
    while time.monotonic() < t_end:
        for _ in range(8):
            torch._int_mm(a, b)
        reps += 8
        torch.cuda.synchronize()
    e1.record()
    torch.cuda.synchronize()
    sec = e0.elapsed_time(e1) / 1e3
    return {"reps": reps, "seconds": sec, "tops": 2 * n**3 * reps / sec / 1e12, **smi.window(t0, time.monotonic())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get("GPU_LEASE_UUID", "0")
    info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit", "--format=csv",
                           "-i", gpu], capture_output=True, text=True, check=False).stdout
    gen = torch.Generator(device="cuda").manual_seed(0)
    ops = {k: operand(k, (a.n, a.n), gen, "cuda") for k in RANGES}
    smi = Sampler(gpu)
    rows = []
    for r in range(a.rounds):
        for name, (lk, rk) in PATTERNS.items():
            row = {"round": r, "pattern": name, **phase(ops[lk], ops[rk].t().contiguous().t(), a.seconds, smi)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    smi.stop()
    med = {p: {k: statistics.median(x[k] for x in rows if x["pattern"] == p and x[k] is not None)
               for k in ("tops", "clock_mhz", "power_w")} for p in PATTERNS}
    for p in PATTERNS:
        med[p]["vs_wide"] = med[p]["tops"] / med["wide"]["tops"]
    doc = {"kind": "int8-toggle/v1", "n": a.n, "seconds": a.seconds, "rounds": a.rounds, "gpu": info.strip(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "patterns": PATTERNS, "ranges": RANGES,
           "median": med, "rows": rows}
    (out / "result.json").write_text(json.dumps(doc, indent=1))
    lines = [f"int8 GEMM {a.n}³, `torch._int_mm`, {a.rounds} rounds × {a.seconds:g} s (median)\n", info.strip() + "\n",
             "| pattern | operands | TOPS | vs wide | SM clock MHz | power W |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {p} | {PATTERNS[p][0]} × {PATTERNS[p][1]} | {m['tops']:.1f} | {m['vs_wide']:.4f} | "
              f"{m['clock_mhz']:.0f} | {m['power_w']:.0f} |" for p, m in med.items()]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
