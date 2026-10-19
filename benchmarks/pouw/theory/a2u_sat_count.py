"""Count `a2u_sat.cu`'s loop body per kernel in `cuobjdump -sass` (`cpw_sass_count.parse`): its instructions, the uniform
datapath's (opcodes U*), the FFMA stream's, and any spill (LDL/STL). With `--device FILE` (the binary's output on a
device) each kernel's measured SMSP-cycles a warp-loop is set beside its count: `total_ipc` (every instruction) and
`vector_ipc` (all but U*) a SMSP-cycle. A2u (every class 1 instruction an issue slot) says `total_ipc` <= 1; an l_
kernel (its chains on the lanes) above 1 would be vector dual issue. Prints one JSON object."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpw_sass_count import CC_ARCH, parse  # noqa: E402

KERNEL = re.compile(r"^([slu])_(\d+)_(\d+)(?:_(\d+))?$")


def count(path: str) -> dict[str, dict[str, dict]]:
    res: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for (arch, k), (ops, where, branches) in parse(path).items():
        m = KERNEL.match(k)
        if not m:
            continue
        spans = [(where[t], i) for i, t in branches if t in where and where[t] < i]
        lo, hi = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(ops) - 1)
        body = ops[lo:hi + 1]
        uni = [op for op in body if op.startswith("U")]
        res[arch][k] = {"nf": int(m.group(2)), "k": int(m.group(3)), "body": len(body), "uniform": len(uni),
                        "vector": len(body) - len(uni), "ffma": sum(op.split(".")[0] == "FFMA" for op in body),
                        "spill": sum(op.split(".")[0] in ("LDL", "STL") for op in body),
                        "opcodes": dict(collections.Counter(body).most_common())}
    return dict(res)


def device(path: str) -> tuple[str | None, dict[str, list[float]]]:
    arch, rows = None, collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        m = re.match(r"device cc=(\S+)", line)
        if m:
            arch = CC_ARCH.get(m.group(1))
        m = re.match(r"a2u (\S+) (?:w=(\d+) )?rep=\d+ smsp_cyc_per_warp_loop=([0-9.]+)", line)
        if m:
            rows[(m.group(1), int(m.group(2) or 8))].append(float(m.group(3)))
    return arch, dict(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--device", help="the binary's output on one device")
    a = ap.parse_args()
    doc: dict = {"sass": count(a.sass)}
    if a.device:
        arch, rows = device(a.device)
        doc["device"] = {"arch": arch, "rates": {}}
        for (k, w), cyc in rows.items():
            c = doc["sass"].get(arch, {}).get(k)
            if not c or not cyc:
                continue
            best = min(cyc)
            doc["device"]["rates"][f"{k}@w{w}"] = {"warps_a_smsp": w, "smsp_cyc_per_warp_loop": cyc,
                                                   "total_ipc": round(c["body"] / best, 4),
                                                   "vector_ipc": round(c["vector"] / best, 4),
                                                   "a2u_holds": c["body"] / best <= 1.0}
    print(json.dumps(doc, indent=1))
