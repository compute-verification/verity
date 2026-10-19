"""The kernels the dither gate ran, from its nsys capture's SQLite export: only the kernels of the process that pushed the
gate's NVTX range (`GATE_RANGE`, sbench.py's dither worker) that started and ended inside it, so another arm's engine in its
own process (the exact arm's `_int_mm`) or the gate's own CPU-side checks after the range never count.  Each name's
instances, the fused linear's (`h1f_k`) and every other kernel whose name looks like a GEMM, per step (a step is 128
`h1f_k` launches, one per served linear), so a second GEMM beside the fused one shows by name.  `processes` lists every
captured process's kernel count, for context.

    python kernels.py gate.sqlite
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys

GEMM = re.compile(r"gemm|cutlass|cublas|nvjet|_int_mm|igemm|scaled_mm|matmul|mma|xmma|wgmma|triton_.*(mm|dot)", re.I)
FUSED = "h1f_k"
GATE_RANGE = "verity/dither/gate"


def _pid(global_id: int) -> int:
    return (int(global_id) >> 24) & 0xFFFFFF


def main(path: str) -> int:
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges = []
    for start, end, text, tid, text_id in db.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        if (text if text is not None else strings.get(text_id)) == GATE_RANGE:
            ranges.append((int(start), int(end), _pid(tid)))
    if len(ranges) != 1:
        print(json.dumps({"error": f"{len(ranges)} NVTX ranges named {GATE_RANGE!r}, want 1"}))
        return 2
    start, end, pid = ranges[0]
    kern: dict[str, list[int]] = {}
    procs: dict[int, int] = {}
    for name_id, gpid, s, e in db.execute("SELECT demangledName, globalPid, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        p = _pid(gpid)
        procs[p] = procs.get(p, 0) + 1
        if p == pid and start <= s and e <= end:
            k = kern.setdefault(strings.get(name_id, str(name_id)), [0, 0])
            k[0] += 1
            k[1] += int(e) - int(s)
    rows = [{"name": n, "instances": c, "total_ns": t} for n, (c, t) in kern.items()]
    fused = sum(k["instances"] for k in rows if FUSED in k["name"])
    steps = fused / 128 if fused else 0
    other = [k for k in rows if FUSED not in k["name"] and GEMM.search(k["name"])]
    doc = {"range": {"name": GATE_RANGE, "pid": pid, "ms": round((end - start) / 1e6, 3)},
           "processes": {str(p): n for p, n in sorted(procs.items())},
           "fused_instances": fused, "steps": steps, "other_gemm": other,
           "other_gemm_per_step": round(sum(k["instances"] for k in other) / steps, 3) if steps else None,
           "kernels": sorted(rows, key=lambda k: -k["total_ns"])[:40]}
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
