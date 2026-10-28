"""Store declaration for the native-peak GEMM microbench (research.store README §6.1), for runs of

    python benchmarks/native_peak/measure.py --dtype bf16

Key params follow measure.py's argparse: what is measured (`dtype`, `sizes`, `warmup`, `reps`, `sustain_s`).  `--out`, `--device`
and `--no-sass` never enter the derivation: the first names where the answer goes, the device is a condition (`hw.family`), and the
SASS capture is an annotation of the same measurement.  Nothing here imports torch: the declaration must load on the control machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from research.store.tool import Tool

SCRIPT = "measure.py"


def parse(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == SCRIPT), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--sizes", default="8192,16384")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sustain-s", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-sass", action="store_true")
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    if out.get("sizes"):
        out["sizes"] = ",".join(str(int(s)) for s in sorted({int(x) for x in out["sizes"].split(",") if x.strip()}))
    if unknown:
        out["_unknown"] = unknown
    return out


NATIVE_PEAK = Tool(
    name="native_peak", version="1",
    command=["python", "benchmarks/native_peak/measure.py"],
    closure=["benchmarks/native_peak/**"],
    key_params=["dtype", "sizes", "warmup", "reps", "sustain_s"],
    key_conditions=["hw.family"],
    produces=["native-peak/v1", "run-files/v1"],
    parse=parse,
)
