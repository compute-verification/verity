"""The kernels each dither arm's gate ran, from its nsys capture's SQLite export: per NVTX range ``GATE_RANGE/<arm>``
(sbench.py's dither workers push one each), only the kernels of the process that pushed it that started and ended inside
it, so another arm's engine in its own process (the exact arm's `_int_mm`) or the gate's own checks after the range never
count.  Per arm: each name's instances, the fused linear's (`h1f_k`) and every other kernel whose name looks like a GEMM,
per step (a step is 128 `h1f_k` launches, one per served linear), so a second GEMM beside the fused one shows by name.
`processes` lists every captured process's kernel count, for context.

    python kernels.py gate.sqlite
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys

GEMM = re.compile(r"gemm|cutlass|cublas|nvjet|_int_mm|igemm|scaled_mm|matmul|mma|xmma|wgmma|triton_.*(mm|dot)", re.I)
FUSED = "h1f_k"
GATE_RANGE = "verity/dither/gate/"


def _pid(global_id: int) -> int:
    return (int(global_id) >> 24) & 0xFFFFFF


def main(path: str) -> int:
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges = {}
    for start, end, text, tid, text_id in db.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text if text is not None else strings.get(text_id)
        if name and name.startswith(GATE_RANGE):
            if name in ranges:
                print(json.dumps({"error": f"two NVTX ranges named {name!r}"}))
                return 2
            ranges[name] = (int(start), int(end), _pid(tid))
    if not ranges:
        print(json.dumps({"error": f"no NVTX range named {GATE_RANGE}<arm>"}))
        return 2
    kern: dict[str, dict[str, list[int]]] = {n: {} for n in ranges}
    procs: dict[int, int] = {}
    for name_id, gpid, s, e in db.execute("SELECT demangledName, globalPid, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        p = _pid(gpid)
        procs[p] = procs.get(p, 0) + 1
        for rn, (start, end, pid) in ranges.items():
            if p == pid and start <= s and e <= end:
                k = kern[rn].setdefault(strings.get(name_id, str(name_id)), [0, 0])
                k[0] += 1
                k[1] += int(e) - int(s)
    doc: dict = {"processes": {str(p): n for p, n in sorted(procs.items())}, "arms": {}}
    for rn, (start, end, pid) in ranges.items():
        rows = [{"name": n, "instances": c, "total_ns": t} for n, (c, t) in kern[rn].items()]
        fused = sum(k["instances"] for k in rows if FUSED in k["name"])
        steps = fused / 128 if fused else 0
        other = [k for k in rows if FUSED not in k["name"] and GEMM.search(k["name"])]
        doc["arms"][rn.removeprefix(GATE_RANGE)] = {
            "range": {"name": rn, "pid": pid, "ms": round((end - start) / 1e6, 3)},
            "fused_instances": fused, "steps": steps, "other_gemm": other,
            "other_gemm_per_step": round(sum(k["instances"] for k in other) / steps, 3) if steps else None,
            "kernels": sorted(rows, key=lambda k: -k["total_ns"])[:40]}
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
