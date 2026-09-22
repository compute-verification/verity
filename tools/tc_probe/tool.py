"""Store declaration for the tensor-core probe (research.store README §6.1).

    research run --tool tc_probe --project verity -- python tools/tc_probe/tc_probe.py --arch sm_89 --out $RESEARCH_RUN_DIR --sweep ...

Key params follow tc_probe.py's argparse (`--instruction` is parsed when present and simply absent otherwise, so a probe
extended with it and one without hash differently only when it is given).  `--out` and `--procs` never enter the derivation:
one is the run dir, the other is parallelism.  Path-valued key params are content-resolved by the store, never recorded by name.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from research.store.tool import Tool

SCRIPT = "tc_probe.py"


def parse(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == SCRIPT), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pipeline", default="ampere_bf16_m16n8k16")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--replay-golden", default=None)
    ap.add_argument("--replay-special", default=None)
    ap.add_argument("--replay-instances", default=None)
    ap.add_argument("--instances-manifest", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-random", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None)
    ap.add_argument("--canary", action="store_true")
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    if ns.instruction is None:
        out.pop("instruction")
    if not ns.canary:  # absent unless given: runs recorded before the flag existed keep their derivation hash
        out.pop("canary")
    if unknown:
        out["_unknown"] = unknown
    if out.get("families"):
        out["families"] = ",".join(sorted(f for f in out["families"].split(",") if f))
    return out


TC_PROBE = Tool(
    name="tc_probe", version="1",
    command=["python", "tools/tc_probe/tc_probe.py"],
    closure=["tools/tc_probe/**", "packages/verity/src/verity/ml/tc/**"],
    key_params=["instruction", "arch", "pipeline", "sweep", "families", "seed", "n_random", "golden_per_family",
                "replay_golden", "replay_special", "replay_instances", "instances_manifest", "canary"],
    key_conditions=["hw.family"],
    param_artifacts=["replay_golden", "replay_special", "replay_instances", "instances_manifest"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse,
)
