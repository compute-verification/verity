#!/usr/bin/env python3
"""The card's characterization, on one GPU: what it is, its sustainable lock clock for GEMM work, its tensor-core rates per
precision (FP8 with FP32 against FP16 accumulation among them), its DRAM read and copy bandwidth, its L2's size and
bandwidth, and how much the operands' entropy moves a GEMM's time and power.

  python characterize.py --gpu 5|lease --expect-uuid GPU-...|--server-md FILE [--pod] [--clock-label L] --out char.json
         [--sections identity mma dram l2 entropy hold [lock]] [--lock-candidates 2600 2500 ...] [--sustain-s 30]

The dense GEMM rates per precision (autotuned cuBLASLt and CUTLASS, gated) are `bench.py`'s baselines at the headline
shapes, without arms. Every GEMM this script times is gated first, like the bench's. `hold` holds an FP8 and an NVFP4 GEMM
for `--sustain-s` each at the machine's clocks and records them. `lock`, only when asked, locks this GPU only
(`nvidia-smi -i <UUID> -lgc F,F`), holds an FP8 and an NVFP4 GEMM for `--sustain-s` at each candidate clock from the top
down, and publishes the highest clock both hold with no power, thermal or board throttle and the SM clock at the lock for at
least 99% of the samples; it resets the clock (`-rgc`) at exit. Where the clock can't be locked (a container) it records
the unlocked sustained clocks instead, labelled so.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import baselines as B
import bench as H
import identity as I
from arm import Buffer, Shape

SECTIONS = ("identity", "mma", "dram", "l2", "entropy", "hold", "lock")
DEFAULT_SECTIONS = SECTIONS[:-1]  # `lock` changes the GPU's clocks: only where that's allowed, and only when asked
MMA_KINDS = {0: "bf16.f32", 1: "f16.f16", 2: "e4m3.f32", 3: "e4m3.f16", 4: "f8f6f4.e4m3.f32", 5: "f8f6f4.e2m1.f32",
             6: "mxf8f6f4.e4m3.ue8m0.f32", 7: "mxf4nvf4.e2m1.ue4m3.f32 (nvfp4)", 8: "mxf4nvf4.e2m1.ue8m0.f32 (mxfp4)",
             9: "s8.s32"}
LOAD_SHAPE = Shape("m8192-n8192-k8192", 8192, 8192, 8192)
L2_SIZES_MB = (4, 8, 16, 24, 32, 48, 64, 80, 96, 104, 112, 120, 128, 136, 144, 160, 192, 256, 512)
AT_LOCK_SHARE = 0.99


def mma_rates(dev, nvml, reps: int = 5, target_ms: float = 100.0) -> dict:
    """Each MMA kind's peak rate: sms x 4 blocks of 8 warps, CHAINS independent MMAs per warp per iteration."""
    blocks, threads = dev.info["sms"] * 4, 256
    scratch = dev.alloc(blocks * threads * 4)
    out = {}
    try:
        for kind, name in MMA_KINDS.items():
            per_iter = blocks * threads // 32 * dev.lib.ph_mma_chains() * dev.mma_flops(kind)
            ms = dev.mma_peak(kind, blocks, threads, 1000, scratch)
            iters = max(1000, int(1000 * target_ms / max(ms, 1e-3)))
            runs = []
            for _ in range(reps):
                ms = dev.mma_peak(kind, blocks, threads, iters, scratch)
                s = nvml.sample()
                runs.append({"ms": ms, "tflops": per_iter * iters / (ms * 1e-3) / 1e12, "sensors": s,
                             "flops_per_sm_per_clk": per_iter * iters / (ms * 1e-3) / (dev.info["sms"] * s["sm_mhz"] * 1e6)
                             if s.get("sm_mhz") else None})
            med = statistics.median(r["tflops"] for r in runs)
            out[name] = {"kind": kind, "iters": iters, "tflops_median": med, "runs": runs,
                         "flops_per_sm_per_clk_median": statistics.median(r["flops_per_sm_per_clk"] for r in runs
                                                                          if r["flops_per_sm_per_clk"])}
    finally:
        dev.free(scratch)
    f32, f16 = out["e4m3.f32"]["tflops_median"], out["e4m3.f16"]["tflops_median"]
    out["fp8_f32_accumulate_over_f16"] = f32 / f16
    out["fp8_f32_over_bf16"] = f32 / out["bf16.f32"]["tflops_median"]
    return out


def timed(dev, body, reps: int) -> list[float]:
    e0, e1 = dev.event(), dev.event()
    xs = []
    for _ in range(reps):
        dev.flush()
        dev.record(e0)
        body()
        dev.record(e1)
        xs.append(dev.elapsed_ms(e0, e1))
    return xs


def dram(dev, nvml, gib: int = 8, reps: int = 5) -> dict:
    free, _ = dev.mem_info()
    n = min(gib << 30, int(free * 0.4)) & ~0xFFFF
    a, b = dev.alloc(n), dev.alloc(n // 2)
    try:
        dev.memset(a.ptr, 0, n)
        dev.memset(b.ptr, 0, n // 2)
        dev.sync()
        rd = timed(dev, lambda: dev.read_bw(a, 2), reps)
        half = Buffer(a.ptr, n // 2)
        cp = timed(dev, lambda: dev.copy(b, half), reps)
        return {"bytes": n, "read_gb_s": [2 * n / (t * 1e-3) / 1e9 for t in rd],
                "copy_gb_s": [2 * (n // 2) / (t * 1e-3) / 1e9 for t in cp], "sensors": nvml.sample()}
    finally:
        dev.free(a)
        dev.free(b)


def l2(dev, nvml, total_gb: float = 16.0, reps: int = 3) -> dict:
    top = max(L2_SIZES_MB) << 20
    buf = dev.alloc(top)
    try:
        dev.memset(buf.ptr, 0, top)
        rows = []
        for mb in L2_SIZES_MB:
            size = mb << 20
            passes = max(2, math.ceil(total_gb * 1e9 / size))
            sub = Buffer(buf.ptr, size)
            xs = timed(dev, lambda sub=sub, passes=passes: dev.read_bw(sub, passes), reps)
            rows.append({"mb": mb, "passes": passes, "gb_s": statistics.median(size * passes / (t * 1e-3) / 1e9 for t in xs)})
        return {"device_l2_bytes": dev.l2, "sweep": rows, "sensors": nvml.sample()}
    finally:
        dev.free(buf)


def gated_gemm(bench, dev, fam_name: str, shape: Shape, seed: int, lt_ws, kernels):
    """The fastest gated kernel of one family at `shape` on its timing sets: (entry, operands, gated, burst)."""
    fam = B.FAMILIES[fam_name]
    plan = H.shape_plan(shape, [fam], (), dev.l2)
    ok, _ = B.cutlass_candidates(dev, fam, shape, kernels)
    sf = {fam.name: (max([c.sf_bytes[0] for c in ok], default=0), max([c.sf_bytes[1] for c in ok], default=0))}
    so = B.allocate(dev, shape, [fam], plan["sets"], seed, sf)
    gated = H.gate_baselines(bench, dev, shape, [fam], so, kernels, lt_ws, False, 8, seed)
    entries = H.tune_baselines(bench, dev, shape, [fam], so, gated, lt_ws, plan["burst"], 3, 1)
    if not entries:
        raise RuntimeError(f"no {fam_name} kernel passed its gate at {shape.label}")
    return entries[0], so, gated, plan["burst"]


def entropy(bench, dev, nvml, lt_ws, kernels, seed: int, reps: int = 20) -> dict:
    """The same kernel on the ternary gate set and on full-entropy operands, the same operand set repeated (hot in L2), so
    the difference is the datapath's: time and power."""
    out = {}
    for fam_name in ("fp8-e4m3", "nvfp4", "bf16"):
        entry, so, gated, _ = gated_gemm(bench, dev, fam_name, LOAD_SHAPE, seed, lt_ws, kernels)
        fam = B.FAMILIES[fam_name]
        cand = next(c for c in gated[fam_name]["gated"] if c.name == entry.key[-1])
        row = {"kernel": cand.name}
        bench.ledger.require(entry.key)
        try:
            for label, s in (("ternary", -1), ("full_entropy", 0)):
                call = B.BaselineCall(dev, cand, LOAD_SHAPE, so.ops(fam, s), lt_ws)
                e = H.Entry(("entropy", fam_name, label), "baseline", [call])
                bench.timer.ensure_graph(e, 16)
                xs, pw = [], []
                for _ in range(reps):
                    t = bench.timer.time(e, "graph", 16)
                    xs.append(t["ms_per_call"])
                    pw.append(t["sensors"]["after"]["power_w"])
                row[label] = {"ms_per_call_median": statistics.median(xs), "power_w_median": statistics.median(pw),
                              "ms": xs, "power_w": pw}
                H.free_entries(bench, [e])
            row["full_over_ternary_time"] = row["full_entropy"]["ms_per_call_median"] / row["ternary"]["ms_per_call_median"]
        finally:
            H.free_entries(bench, [entry])
            B.release_plans(dev, gated[fam_name]["all"])
            so.free(dev)
        out[fam_name] = row
    return out


def sustain(bench, dev, entry, burst: int, seconds: float, lock_mhz: int | None) -> dict:
    """Replays the entry's graph for `seconds` (wall clock), an NVML sample after each; the verdict reads the last 2/3."""
    bench.ledger.require(entry.key)
    samples, ms = [], []
    e0, e1 = dev.event(), dev.event()
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        dev.record(e0)
        dev.graph_launch(entry.graph)
        dev.record(e1)
        ms.append(dev.elapsed_ms(e0, e1) / burst)
        samples.append(bench.timer.sensors.sample())
    tail = samples[len(samples) // 3:]
    bad = sum(1 for s in tail if H.bad_throttle(s))
    at = sum(1 for s in tail if lock_mhz and s.get("sm_mhz") == lock_mhz)
    clocks = [s["sm_mhz"] for s in tail if s.get("sm_mhz")]
    return {"seconds": seconds, "calls": len(ms) * burst, "ms_per_call_median": statistics.median(ms[len(ms) // 3:]),
            "sm_mhz_min": min(clocks, default=None), "sm_mhz_median": statistics.median(clocks) if clocks else None,
            "power_w_max": max(s["power_w"] for s in tail), "temp_c_max": max(s["temp_c"] or 0 for s in tail),
            "bad_throttle_samples": bad, "samples_in_tail": len(tail),
            "share_at_lock": at / len(tail) if lock_mhz and tail else None,
            "throttle_seen": sorted({s["throttle"] for s in tail})}


def hold(bench, dev, lt_ws, kernels, seed: int, seconds: float) -> dict:
    """An FP8 and an NVFP4 GEMM, each held for `seconds` at whatever clocks the machine sets (node 2 fixes them node-wide);
    nothing here changes a clock."""
    out = {}
    for fam_name in ("fp8-e4m3", "nvfp4"):
        entry, so, gated, burst = gated_gemm(bench, dev, fam_name, LOAD_SHAPE, seed, lt_ws, kernels)
        try:
            sustain(bench, dev, entry, burst, min(5.0, seconds / 4), None)
            out[fam_name] = {"kernel": entry.key[-1], **sustain(bench, dev, entry, burst, seconds, None)}
        finally:
            H.free_entries(bench, [entry])
            B.release_plans(dev, gated[fam_name]["all"])
            so.free(dev)
    return out


def lock_sweep(bench, dev, nvml, lt_ws, kernels, seed: int, candidates: list[int], seconds: float) -> dict:
    import native as N

    loads = {}
    for fam_name in ("fp8-e4m3", "nvfp4"):
        loads[fam_name] = gated_gemm(bench, dev, fam_name, LOAD_SHAPE, seed, lt_ws, kernels)
    out: dict = {"candidates": candidates, "rows": [], "chosen_mhz": None}
    try:
        for f in candidates:
            r = N.smi(nvml.uuid, "-lgc", f"{f},{f}")
            if r["returncode"] != 0:
                out["cannot_lock"] = r
                break
            row = {"mhz": f, "loads": {}}
            for fam_name, (entry, _, _, burst) in loads.items():
                sustain(bench, dev, entry, burst, min(5.0, seconds / 4), f)
                row["loads"][fam_name] = sustain(bench, dev, entry, burst, seconds, f)
            row["holds"] = all(x["bad_throttle_samples"] == 0 and (x["share_at_lock"] or 0) >= AT_LOCK_SHARE
                               for x in row["loads"].values())
            out["rows"].append(row)
            if row["holds"]:
                out["chosen_mhz"] = f
                break
        if "cannot_lock" in out:
            out["unlocked"] = {fam_name: sustain(bench, dev, entry, burst, seconds, None)
                               for fam_name, (entry, _, _, burst) in loads.items()}
    finally:
        out["reset"] = N.smi(nvml.uuid, "-rgc")
        for fam_name, (entry, so, gated, _) in loads.items():
            H.free_entries(bench, [entry])
            B.release_plans(dev, gated[fam_name]["all"])
            so.free(dev)
    return out


def default_candidates(nvml, floor_share: float = 0.7, step: int = 90) -> list[int]:
    clocks = nvml.supported_clocks()
    if not clocks:
        return []
    top_mem = max(clocks)
    got, last = [], None
    for c in sorted(clocks[top_mem], reverse=True):
        if c < floor_share * max(clocks[top_mem]):
            break
        if last is None or last - c >= step:
            got.append(c)
            last = c
    return got


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gpu", required=True, help="an index, or 'lease': the one index a lease set")
    p.add_argument("--expect-uuid")
    p.add_argument("--server-md", type=Path, help="read --expect-uuid from this copy of server.md's table")
    p.add_argument("--pod", action="store_true")
    p.add_argument("--sections", nargs="*", default=list(DEFAULT_SECTIONS), choices=SECTIONS)
    p.add_argument("--lock-candidates", nargs="*", type=int)
    p.add_argument("--sustain-s", type=float, default=30.0)
    p.add_argument("--clock-label", help="the clocks' label where the machine fixes them (node 2: locked-2100)")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    out: dict = {"argv": sys.argv if argv is None else argv, "started": H.now(), "sections": {},
                 "clock_label": args.clock_label or "unlocked"}
    code = H.EXIT_DEVICE

    def log(s: str) -> None:
        print(s, file=sys.stderr, flush=True)

    try:
        import native as N

        dev = N.CudaDevice()
        gpu, expected = I.expected_for(args.gpu, args.expect_uuid,
                                       args.server_md.read_text() if args.server_md else None, dict(os.environ))
        ident = I.check(dict(os.environ), gpu, dev.device_count, dev.info["uuid"], expected, pod=args.pod)
        nvml = N.Nvml(ident["uuid"])
        nid = nvml.identity()
        out["identity"] = {**ident, "card": I.check_card(dev.info["name"], dev.info["capability"], dev.info["sms"],
                                                         nid["driver"]), "nvml": nid, "cuda": dev.info}
        out["versions"] = N.versions(dev)
        bench = H.Bench(dev, nvml, H.S.MIN_REPS, ("graph",), log)
        lt_ws = dev.alloc(B.LT_WORKSPACE)
        kernels = B.cutlass_kernels(dev)
        for sec in args.sections:
            log(f"section {sec}")
            try:
                if sec == "identity":
                    r = {"supported_clocks": nvml.supported_clocks(), "smi_q": N.smi(nvml.uuid, "-q"),
                         "smi_clocks": N.smi(nvml.uuid, "-q", "-d", "CLOCK,POWER,PERFORMANCE,TEMPERATURE")}
                elif sec == "mma":
                    r = mma_rates(dev, nvml)
                elif sec == "dram":
                    r = dram(dev, nvml)
                elif sec == "l2":
                    r = l2(dev, nvml)
                elif sec == "entropy":
                    r = entropy(bench, dev, nvml, lt_ws, kernels, args.seed)
                elif sec == "hold":
                    r = hold(bench, dev, lt_ws, kernels, args.seed, args.sustain_s)
                else:
                    r = lock_sweep(bench, dev, nvml, lt_ws, kernels, args.seed,
                                   args.lock_candidates or default_candidates(nvml), args.sustain_s)
            except H.NegativeControlFailed:
                raise
            except Exception as ex:  # noqa: BLE001 - one section's failure is recorded; the others still run
                r = {"error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
            out["sections"][sec] = r
        out["gates"] = [{"key": list(k), **v} for k, v in bench.ledger.records.items()]
        failed = [s for s, r in out["sections"].items() if "error" in r]
        code = H.EXIT_DEVICE if failed else H.EXIT_OK
    except I.IdentityError as ex:
        code, out["error"] = H.EXIT_USAGE, f"identity: {ex}"
    except Exception as ex:  # noqa: BLE001
        code, out["error"], out["traceback"] = H.EXIT_DEVICE, f"{type(ex).__name__}: {ex}", traceback.format_exc()
    finally:
        out["finished"], out["exit"] = H.now(), code
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, default=str))
        log(f"exit {code} -> {args.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
