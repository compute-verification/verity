#!/usr/bin/env python3
"""The units' diagnostic timing on one card (information only: it compares no words), from the unit modules beside it or
in MODULES (another ship's, so two builds are timed by one ruler in one lease).

At each B of TIME_B, each unit and torch's nearest op, interleaved rep by rep, under two rulers:
- median_ms: the GPU's time.  A torch.cuda._sleep ahead of the start event keeps the stream busy while the host queues
  the launch, so the events bracket the kernels and nothing of the host; ``covered`` says the start event was still
  queued when the host had queued the end event (if not, the host's time leaked in);
- host_ms: the start event on an idle stream, then the launch (the ruler of the first H200 run, which counts the
  host's ~0.05 ms of a launch).
Then a step row twice: ``summed_ms``, the units' medians times their counts at a decode step (STEP), and ``measured_ms``,
that step's launches queued in a decode step's order (per layer: RMSNorm, quantizer of 16384, softmax at n = 32,
quantizer, residual add, RMSNorm, quantizer, quantizer of 65536 with ReLU, residual add; then the final RMSNorm,
quantizer and argmax) and timed as one, without the embedding or any matmul.

  python3 bench.py [MODULES]   (env: TIME_B=4096,16384, REPS=20, STEP_REPS=5)
"""
from __future__ import annotations

import json
import os
import statistics
import sys

from verity_config.settings import of

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))

import torch

import add as AD
import argmax as AM
import lib as L
import quant as Q
import relu as RL
import rmsnorm as RN
import softmax as SM

HEADS = 256
LAYERS = 52
settings = of(__file__)
STEP = {"rmsnorm 16384 (units)": 2 * LAYERS + 1, "quant 16384 (units)": 3 * LAYERS + 1, "quant 65536 relu (units)": LAYERS,
        "softmax n=32 (units)": LAYERS, "add 16384": 2 * LAYERS, "argmax 256 (units)": 1}
#: the prelude's cycles: an arm's host time is well under a millisecond, a step's ~500 launches well under 100 ms
ARM_SLEEP, STEP_SLEEP = 2_000_000, 200_000_000


def gpu_ms(f, cycles: int) -> tuple[float, bool]:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(cycles)
    e0.record()
    f()
    e1.record()
    covered = not e0.query()
    e1.synchronize()
    return e0.elapsed_time(e1), covered


def host_ms(f) -> float:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    f()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1)


def timing(dev, B: int, reps: int, step_reps: int) -> list[dict]:
    g = torch.Generator(device=dev).manual_seed(7)
    rnd = lambda *s: torch.randn(*s, generator=g, device=dev).to(torch.bfloat16)  # noqa: E731
    x16, w16, x64 = rnd(B, 16384), rnd(16384), torch.relu(rnd(B, 65536))
    s = rnd(B * HEADS, 32)
    lens32 = torch.full((B,), 32, dtype=torch.int32, device=dev)
    lmix = torch.randint(1, 33, (B,), generator=g, device=dev, dtype=torch.int32)
    logits = rnd(B, 256)
    o16, p = torch.empty_like(x16), torch.empty_like(s)
    c16, c64 = torch.empty(B, 16384, dtype=torch.int8, device=dev), torch.empty(B, 65536, dtype=torch.int8, device=dev)
    tok = torch.empty(B, dtype=torch.int16, device=dev)
    y64 = torch.empty_like(x64)
    r16 = rnd(B, 16384)
    arms = {
        "rmsnorm 16384 (units)": (lambda: RN.gpu(x16, w16, out=o16, units=True), 2 * x16.numel() * 2),
        "torch rms_norm 16384": (lambda: torch.nn.functional.rms_norm(x16, (16384,), w16, 1e-5), 2 * x16.numel() * 2),
        "quant 16384 (units)": (lambda: Q.gpu(x16, False, codes=c16, units=True), x16.numel() * 3),
        "quant 65536 relu (units)": (lambda: Q.gpu(x64, True, codes=c64, units=True), x64.numel() * 3),
        "add 16384": (lambda: AD.gpu(x16, r16, out=o16), 3 * x16.numel() * 2),
        "torch add 16384": (lambda: torch.add(x16, r16, out=o16), 3 * x16.numel() * 2),
        "softmax n=32 (units)": (lambda: SM.gpu(s, lens32, HEADS, out=p, units=True, check_lens=False), 2 * s.numel() * 2),
        "softmax mixed n (units)": (lambda: SM.gpu(s, lmix, HEADS, out=p, units=True, check_lens=False), 2 * s.numel() * 2),
        "torch softmax f32 n=32": (lambda: torch.softmax(s.float(), -1).to(torch.bfloat16), 2 * s.numel() * 2),
        "argmax 256 (units)": (lambda: AM.gpu(logits, out=tok, units=True), logits.numel() * 2),
        "torch argmax 256": (lambda: torch.argmax(logits, -1), logits.numel() * 2),
        "relu 65536": (lambda: RL.gpu(x64, out=y64), 2 * x64.numel() * 2),
        "torch relu 65536": (lambda: torch.relu(x64), 2 * x64.numel() * 2),
    }
    if not hasattr(torch.nn.functional, "rms_norm"):
        del arms["torch rms_norm 16384"]
    for f, _ in arms.values():
        f()
    torch.cuda.synchronize()
    gpu_t: dict[str, list[float]] = {k: [] for k in arms}
    host_t: dict[str, list[float]] = {k: [] for k in arms}
    uncovered = dict.fromkeys(arms, 0)
    for _ in range(reps):
        for name, (f, _) in arms.items():
            ms, covered = gpu_ms(f, ARM_SLEEP)
            gpu_t[name].append(ms)
            uncovered[name] += not covered
            host_t[name].append(host_ms(f))
    rows = []
    for name, (_, nbytes) in arms.items():
        med = statistics.median(gpu_t[name])
        rows.append({"part": "timing", "arm": name, "B": B, "reps": reps, "median_ms": med, "min_ms": min(gpu_t[name]),
                     "reps_uncovered": uncovered[name], "host_ms": statistics.median(host_t[name]), "bytes": nbytes,
                     "GBps_at_median": nbytes / med / 1e6})
    med = {r["arm"]: r["median_ms"] for r in rows}
    hmed = {r["arm"]: r["host_ms"] for r in rows}

    def step() -> None:
        for _ in range(LAYERS):
            RN.gpu(x16, w16, out=o16, units=True)
            Q.gpu(o16, False, codes=c16, units=True)
            SM.gpu(s, lens32, HEADS, out=p, units=True, check_lens=False)
            Q.gpu(x16, False, codes=c16, units=True)
            AD.gpu(x16, r16, out=o16)
            RN.gpu(x16, w16, out=o16, units=True)
            Q.gpu(o16, False, codes=c16, units=True)
            Q.gpu(x64, True, codes=c64, units=True)
            AD.gpu(x16, r16, out=o16)
        RN.gpu(x16, w16, out=o16, units=True)
        Q.gpu(o16, False, codes=c16, units=True)
        AM.gpu(logits, out=tok, units=True)

    step()
    torch.cuda.synchronize()
    st, cov = [], 0
    for _ in range(step_reps):
        ms, covered = gpu_ms(step, STEP_SLEEP)
        st.append(ms)
        cov += covered
    rows.append({"part": "step", "B": B, "layers": LAYERS, "counts": STEP,
                 "summed_ms": sum(n * med[arm] for arm, n in STEP.items()),
                 "summed_host_ms": sum(n * hmed[arm] for arm, n in STEP.items()),
                 "measured_ms": statistics.median(st), "measured_min_ms": min(st), "step_reps": step_reps,
                 "step_reps_uncovered": step_reps - cov})
    return rows


def summary(row: dict) -> str:
    if row["part"] == "step":
        return (f"step B={row['B']}: {row['summed_ms']:.2f} ms summed ({row['summed_host_ms']:.2f} ms on the host-inclusive "
                f"ruler), {row['measured_ms']:.2f} ms measured ({row['layers']} layers)")
    return (f"timing B={row['B']} {row['arm']}: {row['median_ms']:.3f} ms ({row['GBps_at_median']:.0f} GB/s; host-inclusive "
            f"{row['host_ms']:.3f} ms)")


def runs() -> list[int]:
    return list(settings.TIME_B)


def main() -> int:
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    lib = L.lib()
    print(json.dumps({"part": "bench", "modules": os.path.dirname(os.path.abspath(L.__file__)), "cubin": str(lib.path),
                      "sha256": lib.sha256, "name": props.name,
                      "uuid": str(getattr(props, "uuid", "")).lower().removeprefix("gpu-")}), flush=True)
    for B in runs():
        for row in timing(dev, B, settings.REPS, settings.STEP_REPS):
            print(json.dumps(row), flush=True)
            print(summary(row), file=sys.stderr)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
