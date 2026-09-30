"""Store declarations for the sm_120 FP8 rows (``fp8.py``) and the W1 price microbenchmarks (``w1.py``) (research.store README §6.1).

    research run --on SERVER --project verity --campaign pouw --tool tc_probe_fp8 --source . --env CUDA_VISIBLE_DEVICES=0 -- \
        python tools/tc_probe_fp4/fp8.py --row chain --out $RESEARCH_RUN_DIR --expect-gpu-uuid GPU-...
    research run --on SERVER --project verity --campaign pouw --tool w1_prices --source . --env CUDA_VISIBLE_DEVICES=0 -- \
        python tools/tc_probe_fp4/w1.py --out $RESEARCH_RUN_DIR --expect-gpu-uuid GPU-...

Key params follow each script's argparse; ``--out`` and ``--expect-gpu-uuid`` never enter the derivation (the UUID is recorded in
the result, and the attempt's conditions carry the device).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from research.store.tool import Tool

_CLOSURE = ["tools/tc_probe_fp4/**", "tools/tc_probe/tc_probe.py", "packages/verity/src/verity/ml/**"]


def _argv_after(argv: list[str], script: str) -> list[str]:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == script), -1)
    return argv[idx + 1:]


def parse_fp8(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--row", default=None)
    ap.add_argument("--instruction", default="f8_e4m3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--n-random", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=2048)
    ap.add_argument("--chain-tiles", type=int, default=16)
    ap.add_argument("--families", default=None)
    ap.add_argument("--expect-gpu-uuid", default=None)
    ap.add_argument("--expect-model", default=None)
    ap.add_argument("--lease-max-min", type=int, default=None)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--target-cycles", type=int, default=1 << 27)
    ap.add_argument("--clocks", default="unlocked")
    ns, unknown = ap.parse_known_args(_argv_after(argv, "fp8.py"))
    out = vars(ns)
    out["n"] = out.pop("n_random")
    if out["row"] != "step":
        out["instruction"] = None
    if out["row"] != "sparse":
        for k in ("expect_model", "reps", "target_cycles", "clocks"):
            out[k] = None
    if out.get("families"):
        out["families"] = ",".join(sorted(f for f in out["families"].split(",") if f))
    if unknown:
        out["_unknown"] = unknown
    return out


def parse_w1(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--target-cycles", type=int, default=1 << 27)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--clocks", default="unlocked")
    ap.add_argument("--expect-gpu-uuid", default=None)
    ap.add_argument("--ops", default=None)
    ap.add_argument("--prebuilt", default=None)
    ns, unknown = ap.parse_known_args(_argv_after(argv, "w1.py"))
    out = vars(ns)
    if unknown:
        out["_unknown"] = unknown
    return out


TC_PROBE_FP8 = Tool(
    name="tc_probe_fp8", version="1",
    command=["python", "tools/tc_probe_fp4/fp8.py"],
    closure=_CLOSURE,
    key_params=["row", "instruction", "families", "seed", "n", "steps", "chain_tiles", "expect_model", "reps", "target_cycles", "clocks"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "tc-compile-check/v1", "run-files/v1"],
    parse=parse_fp8,
)

W1_PRICES = Tool(
    name="w1_prices", version="1",
    command=["python", "tools/tc_probe_fp4/w1.py"],
    closure=_CLOSURE,
    key_params=["seed", "reps", "target_cycles", "compile_only", "clocks", "ops"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_w1,
)
