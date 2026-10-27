"""J10-R's proof blocks under a profiler: what bounds an h1u build's call on the H200.

    python prof.py run --fixtures DIR --scratch S --h1u NAME=PATH [--sm] [--ms 16384] [--linears q,kv,o,up,down]
                       --out OUT
    python prof.py summary RAW.csv [--out prof-summary.json]

`run`: one call a linear and M on bench.py's operands (`shape.whole`, every row live), once warm and then once between
cudaProfilerStart and cudaProfilerStop, each in an NVTX range LINEAR-M, so `ncu --profile-from-start off` profiles that
one call's kernels (kh_seeds_j10r's, then kh_units_multi_j10r's or with --sm _sm's: h1jr_rows, h1jr_x, h1umj or
h1umj_sm). OUT/order.json lists the calls in launch order with their proof blocks and proof-block segments.

`summary`: ncu's raw page (`--page raw --csv`) as one record a kernel: its duration, cycles, the pipes' utilisation (every
`*pipe*pct_of_peak_sustained_active` column), issue activity, occupancy and its limits, registers, memory throughput, the
warp stall ratios and the instruction counts by pipe.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: raw-page columns kept by name pattern (ncu 2025's metric names at sm_90)
KEEP = [r"^Kernel Name$", r"^ID$", r"^NVTX", r"^Grid Size$", r"^Block Size$", r"^gpu__time_duration\.sum$",
        r"^sm__cycles_elapsed\.(avg|max)$", r"^smsp__cycles_active\.avg$", r"^sm__throughput\.avg\.pct",
        r"pipe.*pct_of_peak_sustained_active$", r"^smsp__issue_active\.avg\.pct", r"^smsp__inst_executed\.sum$",
        r"^sm__inst_executed_pipe_[a-z_]+\.sum$", r"^smsp__inst_executed_pipe_[a-z_]+\.sum$",
        r"^sm__warps_active\.avg\.pct", r"^launch__occupancy_limit", r"^launch__registers_per_thread$",
        r"^launch__shared_mem_per_block", r"^launch__waves_per_multiprocessor$", r"^launch__occupancy_per_block_size",
        r"^sm__maximum_warps_per_active_cycle_pct$", r"^dram__throughput\.avg\.pct", r"^dram__bytes_(read|write)\.sum$",
        r"^lts__throughput\.avg\.pct", r"^l1tex__throughput\.avg\.pct", r"^lts__t_bytes\.sum$",
        r"^smsp__average_warp(s|_latency)_issue_stalled_.*_per_issue_active\.ratio$",
        r"^smsp__thread_inst_executed_per_inst_executed\.ratio$", r"^sm__sass_thread_inst_executed_op_.*\.sum$",
        r"^smsp__sass_inst_executed_op_.*\.sum$", r"^sm__pipe_.*cycles_active\.(avg|max|sum)$",
        r"^device__attribute_(max_gpu_frequency_khz|display_name|multiprocessor_count)$",
        r"^sm__cycles_active\.avg$", r"^gpc__cycles_elapsed\.max$"]


def _num(v: str):
    t = v.replace(",", "").strip()
    try:
        return int(t)
    except ValueError:
        try:
            return float(t)
        except ValueError:
            return v


def summary(raw: Path) -> list[dict]:
    rows = list(csv.reader(raw.open(newline="")))
    if len(rows) < 3:
        return []
    head, units, data = rows[0], rows[1], rows[2:]
    pats = [re.compile(p) for p in KEEP]
    keep = [i for i, h in enumerate(head) if any(p.search(h) for p in pats)]
    out = []
    for r in data:
        rec = {}
        for i in keep:
            if i < len(r) and r[i] != "":
                u = units[i] if i < len(units) else ""
                rec[head[i] + (f" [{u}]" if u else "")] = _num(r[i])
        out.append(rec)
    return out


def run(a) -> int:
    import numpy as np
    import torch

    import fixtures as F
    import shape as S
    import units as UN

    out, scratch = Path(a.out), Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    doc_f, _ = F.load(Path(a.fixtures))
    libs = {}
    for spec in a.h1u:
        name, _, path = spec.partition("=")
        libs[name] = UN.H1u(path, sm=a.sm and UN.H1u.has_sm(path))
    order = []
    calls = []
    for m in [int(x) for x in a.ms.split(",")]:
        for lin in a.linears.split(","):
            key = f"{lin}-{m}"
            c = doc_f["calls"].get(key) or S.call(lin, m)
            ta = torch.from_numpy(np.ascontiguousarray(S.whole("a", lin, m, scratch, a.jobs))).cuda()
            tb = torch.from_numpy(np.ascontiguousarray(S.whole("b", lin, m, scratch, a.jobs))).cuda()
            roots = (bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]))
            for nm, lib in libs.items():
                cl = UN.H1uCall(lib, ta, tb, S.SALT, roots, c["base"])
                cl.run()
                torch.cuda.synchronize()
                calls.append((f"{key}/{nm}", cl))
                ks = c["k"] // S.T
                order.append({"call": key, "arm": nm, "linear": lin, "m": m, "n": c["n"], "k": c["k"],
                              "units": cl.units, "unit_segments": cl.units * ks, "sm": lib.sm})
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for name, cl in calls:
        torch.cuda.nvtx.range_push(name)
        cl.run()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    (out / "order.json").write_text(json.dumps(order, indent=1))
    print(json.dumps(order))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fixtures", required=True)
    r.add_argument("--scratch", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--h1u", action="append", default=[], metavar="NAME=PATH")
    r.add_argument("--sm", action="store_true")
    r.add_argument("--ms", default="16384")
    r.add_argument("--linears", default="q,kv,o,up,down")
    r.add_argument("--jobs", type=int, default=32)
    s = sub.add_parser("summary")
    s.add_argument("raw")
    s.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "run":
        return run(a)
    recs = summary(Path(a.raw))
    text = json.dumps(recs, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0 if recs else 3


if __name__ == "__main__":
    sys.exit(main())
