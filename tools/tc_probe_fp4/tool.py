"""Store declaration for the Blackwell FP4 tensor-core probe (research.store README §6.1), lane fp4-5090.

    research run --on vy-5090 --project verity --campaign r21-silicon --tool tc_probe_fp4 --source <worktree> \
        --send autoproof/probes/results/probe3_1783788526.json -- \
        python tools/tc_probe_fp4/probe.py --instruction nvf4 --out $RESEARCH_RUN_DIR --sass-check --sweep --n-random 4096 \
            --replay-autoproof $RESEARCH_RUN_DIR/inputs/probe3_1783788526.json

Key params follow probe.py's argparse; `--out` and `--procs` never enter the derivation.  Replay files are content-resolved.
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
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    if out["arch"] is None:  # probe.py's default: sm_120a for the mma.sync instructions, sm_100a for the tcgen05 ones
        out["arch"] = "sm_100a" if str(out["instruction"]).startswith("tcgen05") else "sm_120a"
    out["n"] = out.pop("n_random")
    out["replay_autoproof"] = sorted(out["replay_autoproof"])
    if unknown:
        out["_unknown"] = unknown
    if out.get("families"):
        out["families"] = ",".join(sorted(f for f in out["families"].split(",") if f))
    return out


TC_PROBE_FP4 = Tool(
    name="tc_probe_fp4", version="1",
    command=["python", "tools/tc_probe_fp4/probe.py"],
    closure=["tools/tc_probe_fp4/**", "packages/verity/src/verity/ml/tc/**"],
    key_params=["instruction", "arch", "families", "seed", "n", "group", "sweep", "golden_per_family", "replay_autoproof", "sass_check",
                "chain2", "smem_strides"],
    key_conditions=["hw.family"],
    param_artifacts=["replay_autoproof"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse,
)
