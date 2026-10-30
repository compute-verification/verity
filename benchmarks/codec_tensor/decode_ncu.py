"""Where the weights' decoders spend their time, under Nsight Compute: one launch each of the fused GEMM's untagged
kernel (one linear at M, at a DECODERSxSTAGES config), wt.cu's decode_kernel and kv.cu's over the same bytes, then
the sketched kernel with two sketch warps at each `--sketch` ROWS:DxS (a WT_RECEIPT build's sketch-sha512 receipts
at k_s = ROWS, written), at each k, with the occupancy, scheduler, warp-state, instruction and source sections. Writes
OUT/decode-ncu.ncu-rep, its details page (decode-ncu.txt), its SASS source page with the stall reasons an
instruction (decode-ncu-sass.csv) and, from a build with -lineinfo (step.sh's NVFLAGS), the same by CUDA source line
(decode-ncu-cuda.csv) and correlated (decode-ncu-cuda-sass.csv). The launches' order is the DECODE-NCU line's.

    python decode_ncu.py --wt-lib wt.so --kv-lib step.so --out OUT [--linear gate_up --m 8 --fused 10x6 --ks 2,3
                         --sketch 16:8x7,64:8x8]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench as BW  # noqa: E402

sys.path.insert(0, str(HERE.parent / "word_codec"))
import kv_bench as KB  # noqa: E402

SECTIONS = ("LaunchStats", "Occupancy", "SpeedOfLight", "ComputeWorkloadAnalysis", "InstructionStats",
            "SchedulerStats", "WarpStateStats", "SourceCounters")


def sketch_specs(spec: str) -> list[tuple[int, int, int]]:
    """`--sketch`'s ROWS:DxS entries as (rows, D, stages)."""
    out = []
    for e in (v for v in spec.split(",") if v):
        rows, cfg = e.split(":")
        d, s = map(int, cfg.split("x"))
        out.append((int(rows), d, s))
    return out


def launches(a) -> list[dict]:
    """The child: at each k the fused kernel, then wt.cu's decoder, then kv.cu's, then each sketched kernel, once."""
    import torch

    wt = BW.Lib(a.wt_lib)
    kv = KB.Lib(wt.field.name, a.kv_lib)
    n, kd = BW.MODELS[a.model][1][a.linear]
    st, sms = BW.stream(torch), torch.cuda.get_device_properties(0).multi_processor_count
    store = torch.empty(n * kd // wt.pb * wt.s, dtype=torch.uint8, device="cuda").random_(0, 256)
    x = torch.randn(a.m, kd, device="cuda").to(torch.float8_e4m3fn)
    y = torch.zeros(a.m, n, dtype=torch.float32, device="cuda")
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    d, s = map(int, a.fused.split("x"))
    words = store.numel() // wt.s
    nch, sketched, frags = kd // 512, sketch_specs(a.sketch), {}
    if sketched:
        BW.set_sketch_tag(wt)
    for rows in sorted({r for r, _, _ in sketched}):
        BW.set_pass(wt, 8, rows=rows)
        bits = BW.sketch_bits(wt, BW.window_seed(wt), rows, BW.SKT.vector_bytes(wt.sb, 8))
        signs = torch.from_numpy(BW.pass_signs(wt, bits, 8).view(np.int32)).cuda()
        frags[rows] = BW.sign_frags(wt, torch, signs, 8, st)
    if sketched:
        BW.set_sketch_tag(wt)
    out = []
    for k in map(int, a.ks.split(",")):
        key = BW.lib_key(wt, k)
        BW.set_key(wt, torch, key)
        KB.set_key(kv, KB.lib_key(kv, k))
        launched = ctypes.c_int(0)
        wt.call("wt_fused_sketch", k, None, None, None, a.m, n, kd, 0, 0, d, s, 0, 0, None, None, 0, 0, None, None, 8,
                0, ctypes.byref(launched), st)
        gr = launched.value // (kd // 512)
        steps = BW.tsteps(wt, torch, key, d, gr, kd // 512)
        wt.call("wt_fused_sketch", k, BW.ptr(store), BW.ptr(x.view(torch.uint8)), BW.ptr(y), a.m, n, kd, 0, 0, d, s,
                gr, 0, BW.ptr(steps), None, wt.jbase, 0, None, None, 8, 0, None, st)
        wt.call("wt_decode", BW.ptr(store), None, words, 0, k, BW.ptr(xr), sms * a.grid_mult, st)
        kv.call("kv_decode", BW.ptr(store), None, words, 0, k, sms * a.grid_mult, BW.ptr(xr), st)
        order = [f"fused_k{k}_d{d}s{s}", "wt_decode", "kv_decode"]
        sk = []
        for rows, sd, ss in sketched:
            wt.call("wt_fused_sketch", k, None, None, None, a.m, n, kd, 0, 0, sd, ss, 0, 0, None, None, 0, rows, None,
                    None, 8, 2, ctypes.byref(launched), st)
            sgr = launched.value // nch
            ssteps = BW.tsteps(wt, torch, key, sd, sgr, nch)
            nvec = wt.so.wt_sketch_vectors(n, kd, sgr, 8)
            tags = torch.empty(nvec * wt.tag_bytes, dtype=torch.uint8, device="cuda")
            wt.call("wt_fused_sketch", k, BW.ptr(store), BW.ptr(x.view(torch.uint8)), BW.ptr(y), a.m, n, kd, 0, 0, sd,
                    ss, sgr, 0, BW.ptr(ssteps), BW.ptr(tags), wt.jbase, rows, BW.ptr(frags[rows]), None, 8, 2, None, st)
            order.append(f"fused_k{k}_d{sd}s{ss}w2sk{rows}")
            sk.append({"rows": rows, "fused": [sd, ss], "groups": sgr, "vectors": nvec})
        torch.cuda.synchronize()
        out.append({"k": k, "linear": a.linear, "m": a.m, "fused": [d, s], "groups": gr, "words": words,
                    "grid": sms * a.grid_mult, "sketched": sk, "order": order})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wt-lib", required=True)
    ap.add_argument("--kv-lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="70b", choices=sorted(BW.MODELS))
    ap.add_argument("--linear", default="gate_up")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--fused", default="10x6", help="the untagged fused kernel's DECODERSxSTAGES")
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--grid-mult", type=int, default=8, help="the decoders' CTAs an SM in the grid")
    ap.add_argument("--sketch", default="", help="sketched kernels with two sketch warps, ROWS:DxS,... (a WT_RECEIPT "
                                                 "build)")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        print("DECODE-NCU " + json.dumps(launches(a)), flush=True)
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ncu = shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    rep = out / "decode-ncu"
    cmd = [ncu, "--clock-control", "none", "--import-source", "yes", "-k", "regex:fused_kernel|decode_kernel", "-f",
           "-o", str(rep)] + [x for sec in SECTIONS for x in ("--section", sec)] + [
        sys.executable, __file__, "--child"] + [f"--{k.replace('_', '-')}={v}" for k, v in vars(a).items()
                                                 if k != "child"]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (out / "decode-ncu.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        print(p.stdout[-4000:], p.stderr[-4000:])
        return 7
    for page, name, extra in (("details", "decode-ncu.txt", []),
                              ("source", "decode-ncu-sass.csv", ["--csv", "--print-source", "sass"]),
                              ("source", "decode-ncu-cuda.csv", ["--csv", "--print-source", "cuda"]),
                              ("source", "decode-ncu-cuda-sass.csv", ["--csv", "--print-source", "cuda,sass"])):
        with (out / name).open("w") as fh:
            subprocess.run([ncu, "--import", f"{rep}.ncu-rep", "--page", page] + extra, stdout=fh,
                           stderr=subprocess.STDOUT, check=False)
    print(next((ln for ln in p.stdout.splitlines() if ln.startswith("DECODE-NCU ")), "no launches"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
