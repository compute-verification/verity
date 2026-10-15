#!/usr/bin/env python3
"""Committing a fresh n x n FP8 operand on one RTX PRO 6000 against the n x n x n FP8 GEMM it feeds (CMP-8, Daniel's #8).

  python fresh_operand.py gemm --out GEMM.json [--sizes 4096,8192,16384]
  python fresh_operand.py commit --rows-cubin hm96_rows.cubin --out COMMIT.json [--sizes 4096,8192,16384]

``gemm``: torch._scaled_mm (cuBLASLt) on E4M3 operands from a unit normal, per-tensor scales, BF16 out, median of REPS
CUDA-event times after a warm-up, with the derived TFLOPS against the card's measured E4M3 mma.sync peak.
``commit``: the operand's n rows of n E4M3 codes (n bytes each) committed as the served statement commits rows, hm96-sha512
row-seg/v1 leaves of a frame-v3-sha512 tree (`served_zk/hm96_rows.cu` through rows_bench's `Commit`), with fresh salts:
``segments`` (every 1 KiB segment's SHA-512, a lane a segment), ``finish`` (each row's digest over its segments' digests,
hm96's commit string and leaf, the frame leaf) and ``levels`` (the tree), timed apart and together, and the unsalted
digests alone (mode 0). Before timing, at every size, the device's root from poisoned outputs equals Python's
(verity.ml.commitments) and the no-write control's doesn't.
"""
from __future__ import annotations

import argparse
import ctypes as C
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: E4M3 dense mma.sync MACs per SM per clock on the RTX PRO 6000 (kb/sm120/gotchas.md, r20260930-063213-c94a)
MACS_PER_SM_CLK = 1024


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gemm(args) -> dict:
    import torch
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    out: dict = {"gpu": props.name, "sms": props.multi_processor_count, "torch": torch.__version__, "sizes": {}}
    one = torch.tensor(1.0, device=dev)
    for n in args.sizes:
        a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t()
        f = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # noqa: E731
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        ms = []
        for _ in range(args.reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            f()
            e.record()
            e.synchronize()
            ms.append(s.elapsed_time(e))
        med = statistics.median(ms)
        out["sizes"][str(n)] = {"ms": ms, "median_ms": med, "tflops": 2 * n ** 3 / med / 1e9}
        del a, b
        torch.cuda.empty_cache()
        print(json.dumps({n: round(med, 4)}), flush=True)
    return out


def commit(args) -> dict:
    RB = load("rows_bench", HERE.parent / "served_zk" / "rows_bench.py")
    R = RB.load("gpu1_run", HERE.parent / "pearl_c_sm120" / "run.py")
    image = Path(args.rows_cubin).read_bytes()
    h = R.Dev(image)
    h.open()
    nv = R.Nvml(h.uuid)
    out: dict = {"device": h.name, "uuid": h.uuid, "sms": h.sms, "clk_before": nv.read(), "sizes": {}}
    consts = RB.constants(h)
    u64 = C.c_uint64
    for n in args.sizes:
        t0 = time.perf_counter()
        rows = os.urandom(n * n)
        data = h.put(rows)
        cm = RB.Commit(h, h, consts, f"fresh-operand/{n}", data, n, n, n, RB.SEG, prefix="saturation/fresh-operand")
        setup_s = time.perf_counter() - t0
        want = cm.reference([rows[r * n:(r + 1) * n] for r in range(n)])
        gate = {}
        for run, name, same in ((lambda: cm.run(2, one_cta=False), "root", True), (lambda: None, "control", False)):
            for p_, n_ in ((cm.x, 64 * cm.rows), (cm.bc, 128 * cm.rows), (cm.tree, cm.tree_bytes)):
                h.ck("cuMemsetD8_v2", u64(p_), RB.POISON, n_)
            cm.write_pads()
            run()
            h.sync()
            root = h.read(cm.tree + cm.tree_bytes - 64, 64)
            gate[name] = (root == want["root"]) == same
        rec: dict = {"bytes": n * n, "segments_per_row": cm.s, "setup_s": round(setup_s, 2), "gate": gate}
        if all(gate.values()):
            reps = args.reps
            rec["segments_ms"] = statistics.median(h.timed(cm.segments, reps))
            rec["finish_levels_ms"] = statistics.median(h.timed(lambda: cm.finish(2, one_cta=False), reps))
            rec["total_ms"] = statistics.median(h.timed(lambda: cm.run(2, one_cta=False), reps))
            rec["digests_only_ms"] = statistics.median(h.timed(lambda: cm.run(0, one_cta=False), reps))
            rec["segments_gbps"] = n * n / rec["segments_ms"] / 1e6
            rec["total_gbps"] = n * n / rec["total_ms"] / 1e6
            comp = n * cm.s * 9
            rec["segment_compressions"] = comp
            rec["segment_gcomp_per_s"] = comp / rec["segments_ms"] / 1e6
        out["sizes"][str(n)] = rec
        for p_ in (data, cm.d_salts, cm.leaf_heads, cm.node_heads, cm.digests, cm.x, cm.bc, cm.tree):
            h.free(p_)
        print(json.dumps({n: {k: rec.get(k) for k in ("gate", "segments_ms", "total_ms", "digests_only_ms")}}), flush=True)
    out["clk_after"] = nv.read()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("gemm", "commit"))
    ap.add_argument("--sizes", default="4096,8192,16384")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rows-cubin")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.sizes = [int(x) for x in args.sizes.split(",")]
    doc = {"schema": f"verity-pouw/saturation-fresh-operand-{args.what}/v1", "run": os.environ.get("RESEARCH_RUN_ID"),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "macs_per_sm_clk": MACS_PER_SM_CLK,
           **(gemm(args) if args.what == "gemm" else commit(args))}
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if args.what == "gemm" or all(all(r["gate"].values()) for r in doc["sizes"].values()) else 4


if __name__ == "__main__":
    sys.exit(main())
