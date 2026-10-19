"""The kernels nsys saw in the dither gate's profiler range (`nsys stats --report cuda_gpu_kern_sum --format csv`): each
name's instances, the fused linear's (`h1f_k`) and every other kernel whose name looks like a GEMM, per step (a step is 128
`h1f_k` launches, one per served linear), so a second GEMM beside the fused one shows by name.

    python kernels.py gate_cuda_gpu_kern_sum.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys

GEMM = re.compile(r"gemm|cutlass|cublas|nvjet|_int_mm|igemm|scaled_mm|matmul|mma|xmma|wgmma|triton_.*(mm|dot)", re.I)
FUSED = "h1f_k"


def main(path: str) -> int:
    rows = list(csv.DictReader(open(path)))
    kern = [{"name": r["Name"], "instances": int(r["Instances"]), "total_ns": int(float(r["Total Time (ns)"]))} for r in rows]
    fused = sum(k["instances"] for k in kern if FUSED in k["name"])
    steps = fused / 128 if fused else 0
    other = [k for k in kern if FUSED not in k["name"] and GEMM.search(k["name"])]
    doc = {"fused_instances": fused, "steps": steps, "other_gemm": other,
           "other_gemm_per_step": round(sum(k["instances"] for k in other) / steps, 3) if steps else None,
           "kernels": sorted(kern, key=lambda k: -k["total_ns"])[:40]}
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
