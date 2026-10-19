"""Nsight Compute's view of h1u.cu's unit at the served shapes: why a decode segment costs about three times its modeled
issue bound. Run under ncu (job.sh's prof mode):

    ncu --profile-from-start off --clock-control none --set full -k regex:'h1u|h1s_noise' -o OUT/prof \
        python hb_prof.py --lib NAME=PATH [--lib ...] --out OUT [--ms 32,256] [--linears qkv,down]
    python hb_prof.py --pages OUT/prof.ncu-rep --ncu NCU --out OUT     # the report's text pages, under OUT/prof/

Each build (and its two-launch kh_unit_ws, arm NAME-ws) passes h1ubench.py's gate on each shape first (0xA5 poison, its
receipts against receipts.py's on sampled units, the no-write control refused); a build whose gate fails is not
profiled. It then runs once warm and once between torch.cuda.profiler.start and stop, which is all ncu profiles; an NVTX
range names arm, linear and M. Nothing here is timed: ncu replays each kernel under its own counters.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
import sys
import time
from pathlib import Path

PAGES = {"details.txt": ["--page", "details", "--print-details", "all"],
         "raw.csv": ["--page", "raw", "--csv"],
         "source-sass.csv": ["--page", "source", "--print-source", "sass", "--csv"]}


def pages(rep: str, ncu: str, out: Path) -> int:
    d = out / "prof"
    d.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name, flags in PAGES.items():
        r = subprocess.run([ncu, "-i", rep, *flags], capture_output=True)
        body = r.stdout if r.returncode == 0 else r.stdout + r.stderr
        if name.endswith(".csv"):
            (d / f"{name}.gz").write_bytes(gzip.compress(body))
        else:
            (d / name).write_bytes(body)
        print(f"hb_prof: {name} rc {r.returncode} ({len(body)} bytes)", flush=True)
        rc = rc or r.returncode
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--no-ws", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default="32,256")
    ap.add_argument("--linears", default="qkv,down")
    ap.add_argument("--picks", type=int, default=8)
    ap.add_argument("--pages", metavar="REP")
    ap.add_argument("--ncu", default="ncu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.pages:
        return pages(args.pages, args.ncu, out)

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from h1ubench import Call, Lib, gate_shape, shape_ref
    from hb_kbench import LINEARS, inputs, w_seed, x_seed

    libs: dict = {}
    for v in args.lib:
        name, _, path = v.partition("=")
        libs[name] = Lib(path)
        if not args.no_ws and Lib.has_ws(path):
            libs[f"{name}-ws"] = Lib(path, ws=True)
    rng = random.Random(0x4831F2)
    doc = {"schema": "dither-hb/prof/v1", "device": torch.cuda.get_device_name(0),
           "libs": {a: v for a, v in (x.partition("=")[::2] for x in args.lib)},
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ranges": []}
    ms = [int(v) for v in args.ms.split(",")]
    bad = 0
    for name in args.linears.split(","):
        n, k = LINEARS[name]
        xh = inputs(x_seed(k), max(ms), k)
        b = torch.from_numpy(inputs(w_seed(name), n, k)).to("cuda")
        for m in ms:
            x = torch.from_numpy(xh[:m]).to("cuda")
            a = (x if m >= 32 else torch.cat([x, x.new_zeros(32 - m, k)])).contiguous()
            arms = list(libs)
            first = Call(libs[arms[0]], a, b, rng.randbytes(64), rng.randbytes(64), rng.randbytes(64),
                         rng.getrandbits(40))
            calls = {arm: first if i == 0 else first.twin(libs[arm]) for i, arm in enumerate(arms)}
            picks = sorted({0, first.units - 1, *rng.sample(range(first.units), min(first.units, args.picks))})
            ref_t = shape_ref(first, picks)
            for arm, c in calls.items():
                g = gate_shape(c, picks, ref_t)
                r = {"arm": arm, "linear": name, "m": m, "units": c.units, "nvtx": f"{arm}/{name}/m{m}",
                     "gate": {x: g[x] for x in g if x != "picks"}}
                doc["ranges"].append(r)
                print(json.dumps(r), flush=True)
                if not g["ok"]:
                    bad += 1
                    continue
                c.run()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(r["nvtx"])
                torch.cuda.profiler.start()
                c.run()
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                torch.cuda.nvtx.range_pop()
            del calls, first
            torch.cuda.empty_cache()
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "prof.json").write_text(json.dumps(doc, indent=1))
    return 4 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
