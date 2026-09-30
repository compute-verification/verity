"""Store declaration for the Blackwell FP4 tensor-core probe (research.store README §6.1), lane fp4-5090.

    research run --on vy-5090 --project verity --campaign r21-silicon --tool tc_probe_fp4 --source <worktree> \
        --send autoproof/probes/results/probe3_1783788526.json -- \
        python tools/tc_probe_fp4/probe.py --instruction nvf4 --out $RESEARCH_RUN_DIR --sass-check --sweep --n-random 4096 \
            --replay-autoproof $RESEARCH_RUN_DIR/inputs/probe3_1783788526.json

Key params follow probe.py's argparse; `--out` and `--procs` never enter the derivation.  Replay files are content-resolved.

`tc_price_fp4` is the W1 price microbenchmark beside it (w1_price.py, lane pouw rtx-pro GPU 4); `--out`, `--expect-uuid` and
`--max-seconds` are guards and paths, not part of the answer:

    research run --on vy-nebius-2 --project verity --campaign pouw --tool tc_price_fp4 --source . --cwd source -- \
        gpu-lease 1 -- uv run --no-project --with numpy python tools/tc_probe_fp4/w1_price.py --out $RESEARCH_RUN_DIR \
            --expect-uuid GPU-...,GPU-... --locked-mhz 2100
"""
from __future__ import annotations

import argparse
from pathlib import Path

from research.store.tool import Tool

SCRIPT = "probe.py"


def parse(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == SCRIPT), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--instruction", default="nvf4")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replay-autoproof", action="append", default=[])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-random", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None)
    ap.add_argument("--sass-check", action="store_true")
    ap.add_argument("--chain2", action="store_true")            # lane tcgen05: K=128 chains on the TMEM accumulator
    ap.add_argument("--smem-strides", default=None)             # lane tcgen05: shared-memory descriptor strides
    ap.add_argument("--chain", type=int, default=1)             # mma.sync: in-register chains of L dependent MMAs
    ap.add_argument("--expect-model", default=None)             # the model whose mismatches gate the run
    ap.add_argument("--corrupt-word", action="store_true")      # the gate's self-test: one flipped word must fail the run
    ap.add_argument("--expect-uuid", default=None)              # a guard (stops a run on the wrong GPU), not part of the answer
    ap.add_argument("--fake-silicon", default=None)             # dry run; probe.py refuses it under a research run
    ap.add_argument("--sass-only", action="store_true")
    ap.add_argument("--hot-bits", default="14,15,16")            # sp_nvf4_t1: the hot accumulators' h
    ap.add_argument("--model-sample", type=int, default=4096)   # sp_nvf4_t1: words checked against the pinned models
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    if out["arch"] is None:  # probe.py's default: sm_120a for the mma.sync instructions, sm_100a for the tcgen05 ones
        out["arch"] = "sm_100a" if str(out["instruction"]).startswith("tcgen05") else "sm_120a"
    out["n"] = out.pop("n_random")
    out.pop("expect_uuid")
    out["replay_autoproof"] = sorted(out["replay_autoproof"])
    if unknown:
        out["_unknown"] = unknown
    if out.get("families"):
        out["families"] = ",".join(sorted(f for f in out["families"].split(",") if f))
    return out


def parse_price(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == "w1_price.py"), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--target-ms", type=float, default=100.0)
    ap.add_argument("--sustain-s", type=float, default=3.0)
    ap.add_argument("--warps-per-sm", type=int, default=16)
    ap.add_argument("--ops", default=None)
    ap.add_argument("--op-warps-per-sm", type=int, default=None)
    ap.add_argument("--latency-iters", type=int, default=20000)
    ap.add_argument("--no-scale-update", action="store_true")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=900.0)   # a guard on the plan's length, not part of the answer
    ap.add_argument("--expect-uuid", default=None)                # a guard (stops a run on the wrong GPU)
    ap.add_argument("--locked-mhz", type=float, default=None)
    ap.add_argument("--corrupt-word", action="store_true")
    ap.add_argument("--fake-silicon", action="store_true")
    ap.add_argument("--sass-only", action="store_true")
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    out.pop("out")
    out.pop("max_seconds")
    out.pop("expect_uuid")
    for k in ("rows", "ops"):
        if out[k]:
            out[k] = ",".join(sorted(r for r in out[k].split(",") if r))
    if unknown:
        out["_unknown"] = unknown
    return out


TC_PRICE_FP4 = Tool(
    name="tc_price_fp4", version="2",
    command=["python", "tools/tc_probe_fp4/w1_price.py"],
    closure=["tools/tc_probe_fp4/**", "packages/verity/src/verity/ml/tc/**"],
    key_params=["rows", "ops", "reps", "target_ms", "sustain_s", "warps_per_sm", "op_warps_per_sm", "latency_iters", "no_scale_update",
                "seed", "locked_mhz", "corrupt_word", "fake_silicon", "sass_only"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_price,
)


TC_PROBE_FP4 = Tool(
    name="tc_probe_fp4", version="1",
    command=["python", "tools/tc_probe_fp4/probe.py"],
    closure=["tools/tc_probe_fp4/**", "packages/verity/src/verity/ml/tc/**"],
    key_params=["instruction", "arch", "families", "seed", "n", "group", "sweep", "golden_per_family", "replay_autoproof", "sass_check",
                "chain2", "smem_strides", "chain", "expect_model", "corrupt_word", "fake_silicon", "sass_only", "hot_bits",
                "model_sample"],
    key_conditions=["hw.family"],
    param_artifacts=["replay_autoproof"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse,
)
