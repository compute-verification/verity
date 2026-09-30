"""ue4m3_scale_gpu: build, SASS-count, gate and time Pearl-C4's block-scale step on sm_120a (``ue4m3_scale.cu``).

    python benchmarks/pouw/kernels/ue4m3_scale_gpu.py build        # no GPU: nvcc, the loop's SASS per variant, the CPU checks
    python benchmarks/pouw/kernels/ue4m3_scale_gpu.py run [--expect-uuid GPU-...] [--seconds 0.2] [--reps 5] [--clock-mhz 2100]

``run`` needs the card. It
  1. gates: every variant's reciprocal on the 126 valid bytes against ``__frcp_rn`` (``ref_approx``'s count is information:
     whether MUFU's ``rcp.approx`` happens to be exact there), lut256's addressing from every lane, and on every finite
     non-negative amax the LUT path and the integer encode against F2FP + ``__frcp_rn``. Any nonzero count but
     ``ref_approx``'s fails the run and times nothing;
  2. times each variant on one wave (blocks = SMs x 3, 256 threads), ``--reps`` times, and prices it against ``base`` (the
     1/6 multiply, the floor and the folds): W1 FP8 units per scale = UNIT / (scales per SM per clock) over the difference.

Under ``research run`` it writes a ``research/result/v0.1`` document.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_result  # noqa: E402
import ue4m3_scale as U  # noqa: E402

SOURCE = HERE / "ue4m3_scale.cu"
ARCH = "sm_120a"
VARIANTS = ("ref_rn", "ref_approx", "lut", "lut_lane", "exp_mant", "newton", "ienc_lut", "lut256", "base")
UNIT = 1023.98                      # FP8 MACs per SM per clock (GPU 0, locked 2,100 MHz)
SCALES_PER_ITER = 16


def nvcc_bin() -> str | None:
    for c in (shutil.which("nvcc"), "/usr/local/cuda-12.9/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).exists():
            return c
    return None


def loop_sass(so: Path, bindir: Path) -> dict:
    """Opcode counts in each time_<variant> kernel's loop body (the longest backward branch), per iteration of 16 scales."""

    text = subprocess.run([str(bindir / "cuobjdump"), "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    out = {}
    for f in re.split(r"\n\s*Function : ", text)[1:]:
        name = f.split("\n", 1)[0].strip()
        if not name.startswith("time_"):
            continue
        ins = []
        for line in f.splitlines():
            m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);", line)
            if m:
                ins.append((int(m.group(1), 16), m.group(3), m.group(4)))
        back = [(int(t.group(1), 16), a) for a, op, args in ins if op.startswith("BRA")
                for t in [re.search(r"0x([0-9a-f]+)", args)] if t and int(t.group(1), 16) < a]
        lo, hi = max(back, key=lambda x: x[1] - x[0])
        out[name[5:]] = dict(collections.Counter(op.split(".")[0] for a, op, _ in ins if lo <= a <= hi))
    return out


def build(out_dir: Path) -> dict:
    nvcc = nvcc_bin()
    if nvcc is None:
        raise FileNotFoundError("nvcc not found (PATH, /usr/local/cuda-12.9/bin or /usr/local/cuda/bin)")
    out_dir.mkdir(parents=True, exist_ok=True)
    so = out_dir / "ue4m3_scale.so"
    cmd = [nvcc, "-O3", f"-gencode=arch=compute_{ARCH[3:]},code={ARCH}", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(SOURCE)]
    subprocess.run(cmd, check=True)
    cpu = {"recip": {k: sum(U.bits(f(b)) != U.bits(U.REF[b]) for b in U.BYTES)
                     for k, f in (("lut", lambda b: U.LUT[b]), ("exp_mant", U.exp_mant), ("newton", U.newton))}}
    return {"so": str(so), "cmd": " ".join(cmd), "loop_sass": loop_sass(so, Path(nvcc).parent), "cpu": cpu}


def tables() -> tuple[np.ndarray, np.ndarray]:
    lut = np.array([U.bits(x) for x in U.LUT], np.uint32)
    tab = np.array([U.bits(x) for x in U.MANT] + [U.bits(x) for x in U.SUB], np.uint32)
    return lut, tab


def run(so: str, args) -> dict:
    lib = ctypes.CDLL(so)
    lut, tab = tables()
    sms, ma, mi = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    uuid = ctypes.create_string_buffer(16)
    rc = lib.ue_setup(lut.ctypes.data_as(ctypes.c_void_p), tab.ctypes.data_as(ctypes.c_void_p),
                      ctypes.byref(sms), ctypes.byref(ma), ctypes.byref(mi), uuid)
    if rc:
        raise RuntimeError(f"ue_setup failed ({rc})")
    raw = uuid.raw
    u = f"GPU-{raw[:4].hex()}-{raw[4:6].hex()}-{raw[6:8].hex()}-{raw[8:10].hex()}-{raw[10:16].hex()}"
    if args.expect_uuid and args.expect_uuid != u:
        raise RuntimeError(f"device {u} is not the leased {args.expect_uuid}")
    bad = np.zeros(len(VARIANTS) + 3, np.uint64)
    if lib.ue_gates(bad.ctypes.data_as(ctypes.c_void_p)):
        raise RuntimeError("ue_gates failed")
    gates = {f"recip_{v}": int(bad[i]) for i, v in enumerate(VARIANTS[:-1]) if v not in ("ref_rn", "lut_lane", "ienc_lut")}
    gates.update(pipeline_lut=int(bad[len(VARIANTS)]), pipeline_ienc_byte=int(bad[len(VARIANTS) + 1]),
                 pipeline_ienc_lut=int(bad[len(VARIANTS) + 2]))
    ok = all(n == 0 for k, n in gates.items() if k != "recip_ref_approx")
    res = {"device": u, "sms": sms.value, "cc": f"{ma.value}.{mi.value}", "gates": gates, "ok": ok, "timing": {}}
    if not ok:
        return res
    ms, khz = ctypes.c_double(), ctypes.c_double()
    blocks = sms.value * 3
    per_iter = {}
    for i, v in enumerate(VARIANTS):
        iters = 64
        while True:
            if lib.ue_time(i, blocks, iters, ctypes.byref(ms), ctypes.byref(khz)):
                raise RuntimeError(f"ue_time {v} failed")
            if ms.value > 1e3 * args.seconds / 4 or iters > 1 << 26:
                break
            iters *= 4
        times = []
        for _ in range(args.reps):
            lib.ue_time(i, blocks, iters, ctypes.byref(ms), ctypes.byref(khz))
            times.append(ms.value)
        f_hz = (args.clock_mhz * 1e6) if args.clock_mhz else khz.value * 1e3
        clk_per_iter = statistics.median(times) * 1e-3 * f_hz / iters          # SM clocks per iteration (every SM busy)
        per_iter[v] = clk_per_iter
        res["timing"][v] = {"iters": iters, "ms": times, "sm_clocks_per_iter": clk_per_iter}
    threads_per_sm = blocks * 256 / sms.value
    for v in VARIANTS[:-1]:
        d = (per_iter[v] - per_iter["base"]) / (threads_per_sm * SCALES_PER_ITER)   # SM clocks per scale, net of base
        res["timing"][v]["w1_fp8_per_scale"] = UNIT * d
        res["timing"][v]["fp4_per_element"] = 2 * UNIT * d / 16
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "run"))
    ap.add_argument("--expect-uuid", default=None)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--clock-mhz", type=float, default=0.0, help="the locked SM clock (default: the device's reported rate)")
    ap.add_argument("--out-dir", default=os.environ.get("RESEARCH_RUN_DIR", "/tmp/ue4m3_scale"))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    b = build(out)
    res = {"build": b}
    ok = all(n == 0 for n in b["cpu"]["recip"].values())
    if args.cmd == "run":
        res["run"] = run(b["so"], args)
        ok = ok and res["run"]["ok"]
    path = out / "ue4m3_scale.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    meas = [(f"{v}_fp4_per_element", t["fp4_per_element"], "FP4 units")
            for v, t in res.get("run", {}).get("timing", {}).items() if "fp4_per_element" in t]
    run_result.write(path, ok, "ue4m3_scale_gpu", json.dumps(res.get("run", {}).get("gates", b["cpu"])), meas)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
