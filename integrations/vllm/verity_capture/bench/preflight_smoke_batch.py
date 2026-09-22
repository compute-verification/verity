#!/usr/bin/env python3
"""R10 owner 10: the cheapest proof batch for a worker's GPU proof smoke -- ONE F2fpBf16_v1 obligation (one `f2fp_bf16` gate,
FP32 -> BF16 RNE) over a one-commitment tree with a genuine Merkle opening, built with `zk/bench/a56_approved_batches.build`
(the R9 PU lanes' builder; "f2fp primitive B=1" was the R9 floor row).  The operand is a fixed FP32 word, not a W11 fixture, so
no fixture files are needed.  Writes <out>/f2fp-bf16_n1/{statement.bin,witness-fmt2.bin,plan.bin,census.json}; the transparent
checker accepts the batch before anything is written (build() asserts it).

    /workspace/venv-cu129/bin/python verity_capture/bench/preflight_smoke_batch.py --out /workspace/cp/proof-smoke
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# runs from a copy in /workspace/cp on the pod: the repo is the cwd (or $VERITOR_REPO), not two levels above this file
REPO = Path(os.environ.get("VERITOR_REPO") or os.getcwd()).resolve()
if not (REPO / "src" / "veritor").is_dir():
    REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "zk" / "bench", REPO / "zk" / "campaign", REPO / "vllm-poc"):
    sys.path.insert(0, str(p))

from a56_approved_batches import build, f2fp_subcircuit  # noqa: E402
from veritor.core.reduce_gates import make_reduce_mufu_gate_set  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--value", type=float, default=1.2345678, help="the FP32 operand (default 1.2345678)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    word = struct.unpack("<I", struct.pack("<f", a.value))[0]
    rm = make_reduce_mufu_gate_set()
    entries = []
    for n in a.sizes:
        entries.append(build("f2fp-bf16", f2fp_subcircuit(), rm, lambda i: [word], n, out,
                             {"kind": "F2fpBf16_v1|primitive", "note": "R10 preflight proof smoke: one f2fp_bf16 gate of a fixed FP32 word",
                              "operands": f"synthetic f32 {a.value} = 0x{word:08x}", "guest": "666e937f (zk host, unmodified)"}))
    (out / "manifest.json").write_text(json.dumps({"schema": "veritor/a56/approved-batches/v1", "batches": entries}, indent=1) + "\n")
    print(json.dumps({"batches": [e["dir"] for e in entries], "operand_hex": f"0x{word:08x}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
