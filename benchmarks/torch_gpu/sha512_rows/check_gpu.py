"""The SHA-512 rows engine's checks against the CPU reference (`verity/commitments/cuda/sha512_rows/check.py`'s
``report``) on the current CUDA device, or ``--device host``, with the benchmarks' build directory and variant
(`torch_engine.py`), and on the device one layer of commit_slots' frozen H200 layout (``bench.h200_rows``):
``python check_gpu.py --out DIR`` writes DIR/check.json and exits 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bench
import torch_engine as T

from verity.commitments.cuda.sha512_rows import check as C


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layout-slots", type=int, nargs="*", default=[4096, 16384])
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    doc = C.report(T.Engine(a.device), a.layout_slots, bench.h200_rows())
    (a.out / "check.json").write_text(json.dumps(doc, indent=1, default=int) + "\n")
    print(json.dumps({"all_mismatches": doc["all_mismatches"], "seconds": round(doc["seconds"], 1)}))
    return 0 if doc["all_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
