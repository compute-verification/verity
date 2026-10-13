"""Where the weights' decoders spend their time, under Nsight Compute: one launch each of the fused GEMM's untagged
kernel (one linear at M, at a DECODERSxSTAGES config), wt.cu's decode_kernel and kv.cu's over the same bytes, at each
k, with the occupancy, scheduler, warp-state, instruction and source sections. Writes OUT/decode-ncu.ncu-rep, its
details page (decode-ncu.txt) and its SASS source page with the stall reasons an instruction (decode-ncu-sass.csv).

    python decode_ncu.py --wt-lib wt.so --kv-lib step.so --out OUT [--linear gate_up --m 8 --fused 10x6 --ks 2,3]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench as BW  # noqa: E402  (before kv_bench, as in step.py)

sys.path.insert(0, str(HERE.parent / "word_codec"))
import kv_bench as KB  # noqa: E402

SECTIONS = ("LaunchStats", "Occupancy", "SpeedOfLight", "ComputeWorkloadAnalysis", "InstructionStats",
            "SchedulerStats", "WarpStateStats", "SourceCounters")


def launches(a) -> list[dict]:
    """The child: at each k the fused kernel, then wt.cu's decoder, then kv.cu's, each once."""
    import torch

    wt = BW.Lib(a.wt_lib)
    kv = KB.Lib(wt.field.name, a.kv_lib)
    n, kd = BW.MODELS[a.model][1][a.linear]
    st, sms = BW.stream(torch), torch.cuda.get_device_properties(0).multi_processor_count
    BW.set_sketch_tag(wt)
    store = torch.empty(n * kd // wt.pb * wt.s, dtype=torch.uint8, device="cuda").random_(0, 256)
    x = torch.randn(a.m, kd, device="cuda").to(torch.float8_e4m3fn)
    y = torch.zeros(a.m, n, dtype=torch.float32, device="cuda")
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    d, s = map(int, a.fused.split("x"))
    words = store.numel() // wt.s
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
                gr, 0, BW.ptr(steps), None, BW.TAG_BASE, 0, None, None, 8, 0, None, st)
        wt.call("wt_decode", BW.ptr(store), None, words, 0, k, BW.ptr(xr), sms * a.grid_mult, st)
        kv.call("kv_decode", BW.ptr(store), None, words, 0, k, sms * a.grid_mult, BW.ptr(xr), st)
        torch.cuda.synchronize()
        out.append({"k": k, "linear": a.linear, "m": a.m, "fused": [d, s], "groups": gr, "words": words,
                    "grid": sms * a.grid_mult})
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
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        print("DECODE-NCU " + json.dumps(launches(a)), flush=True)
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ncu = shutil.which("ncu") or "/usr/local/cuda-13.0/bin/ncu"
    rep = out / "decode-ncu"
    cmd = [ncu, "--clock-control", "none", "--import-source", "no", "-k", "regex:fused_kernel|decode_kernel", "-f",
           "-o", str(rep)] + [x for sec in SECTIONS for x in ("--section", sec)] + [
        sys.executable, __file__, "--child"] + [f"--{k.replace('_', '-')}={v}" for k, v in vars(a).items()
                                                 if k not in ("child", "out")]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (out / "decode-ncu.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        print(p.stdout[-4000:], p.stderr[-4000:])
        return 7
    for page, name, extra in (("details", "decode-ncu.txt", []),
                              ("source", "decode-ncu-sass.csv", ["--csv", "--print-source", "sass"])):
        with (out / name).open("w") as fh:
            subprocess.run([ncu, "--import", f"{rep}.ncu-rep", "--page", page] + extra, stdout=fh,
                           stderr=subprocess.STDOUT, check=False)
    print(next((ln for ln in p.stdout.splitlines() if ln.startswith("DECODE-NCU ")), "no launches"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
