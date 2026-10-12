"""Store declaration for the tensor-core probe (research.store README §6.1).

    research run --tool tc_probe --project verity -- python benchmarks/tc_probe/tc_probe.py --arch sm_89 --out $RESEARCH_RUN_DIR --sweep ...

Key params follow tc_probe.py's argparse (`--instruction` is parsed when present and simply absent otherwise, so a probe
extended with it and one without hash differently only when it is given).  `--out` and `--procs` never enter the derivation:
one is the run dir, the other is parallelism.  Path-valued key params are content-resolved by the store, never recorded by name.
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from research.pythonpath import package_paths, workspace_root
from research.store.tool import Tool

TREE = workspace_root(__file__)

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
    command=["python", "benchmarks/tc_probe/tc_probe.py"],
    closure=["benchmarks/tc_probe/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["instruction", "arch", "pipeline", "sweep", "families", "seed", "n_random", "golden_per_family",
                "replay_golden", "replay_special", "replay_instances", "instances_manifest", "canary"],
    key_conditions=["hw.family"],
    param_artifacts=["replay_golden", "replay_special", "replay_instances", "instances_manifest"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse,
)

# --- instances_hw: frozen input sets replayed on silicon (lane instances-hw) --------------------------------------------------------

INSTANCES_SCRIPT = "instances_hw.py"


def _script_flags(argv: list[str], script: str) -> list[str]:
    """The tokens after `script` in `argv` -- also when the command is a shell string (`bash -c 'export ...; python .../script --x'`):
    the string is split and the flags run to the end of the script's own command."""
    idx = next((i for i, t in enumerate(argv) if Path(t).name == script), None)
    if idx is not None:
        return argv[idx + 1:]
    for t in argv:
        if script in t and " " in t:
            toks = shlex.split(t)
            j = next((i for i, x in enumerate(toks) if Path(x).name == script), None)
            if j is None:
                continue
            flags: list[str] = []
            for x in toks[j + 1:]:
                if x in (";", "&&", "||", "|"):
                    break
                if x.endswith(";"):
                    flags.append(x[:-1])
                    break
                flags.append(x)
            return flags
    return argv[1:]


def parse_instances(argv: list[str]) -> dict:
    """`instances_hw.py`'s flags.  `--root` (the built bench-instances arrays) and `--manifest` are content-resolved; `--out` and
    `--procs` never enter the derivation."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--range", default=None)
    ap.add_argument("--only-ids", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--sass-check", action="store_true")
    ap.add_argument("--out", default=None)
    ns, unknown = ap.parse_known_args(_script_flags(argv, INSTANCES_SCRIPT))
    out = vars(ns)
    for k in ("instruction", "arch", "root", "manifest", "range", "only_ids"):
        if out[k] is None:
            out.pop(k)
    if unknown:
        out["_unknown"] = unknown
    return out


INSTANCES_HW = Tool(
    name="instances_hw", version="1",
    command=["python", "benchmarks/tc_probe/instances_hw.py"],
    closure=["benchmarks/tc_probe/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**", "targets/target.py"),
             "benchmarks/numerical/python/verity_numerical/bench/instances.py", "benchmarks/numerical/python/verity_numerical/bench/instances_gen.py",
             "fixtures/bench-instances/v1/manifest.json"],
    key_params=["instances", "instruction", "arch", "root", "manifest", "range", "only_ids", "repeat", "sass_check"],
    key_conditions=["hw.family"],
    param_artifacts=["root", "manifest"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse_instances,
)
