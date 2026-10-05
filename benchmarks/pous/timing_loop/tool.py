"""Store declaration for the timing-loop measurement (research.store README §6.1).

    research run --on vy-nebius-2 --tool benchmarks.pous.timing_loop.tool:TIMING_LOOP --campaign memory-accounting \\
        --cwd source --declared-output 'out/*' -- bash benchmarks/pous/timing_loop/node.sh [bench.py args]

Key params: the responder (GPU or CPU), the rounds, the loads, the loops and the limit.  Imports nothing but
``research.store.tool``.
"""

from __future__ import annotations

import argparse

from research.pythonpath import package_paths, workspace_root
from research.store.tool import Tool

TREE = workspace_root(__file__)


def parse(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--cpu-responder", action="store_true")
    ap.add_argument("--rounds", type=int, default=100_000)
    ap.add_argument("--loads", default="idle,load")
    ap.add_argument("--loops", default="python,lean-drive,lean-own,lean-own-2cpu")
    ap.add_argument("--limit-ns", type=int, default=850_000)
    ns, _ = ap.parse_known_args(argv)
    return {"responder": "cpu" if ns.cpu_responder else "gpu", "rounds": ns.rounds, "loads": ns.loads, "loops": ns.loops,
            "limit_ns": ns.limit_ns}


TIMING_LOOP = Tool(
    name="pous_timing_loop",
    version="1",
    command=["bash", "benchmarks/pous/timing_loop/node.sh"],
    closure=["benchmarks/pous/timing_loop/**", "benchmarks/pous/band_gpu/**", "benchmarks/pous/weight_sets.json",
             "backends/flock/verifier/lean/Flock/Hash.lean", "backends/flock/verifier/lean/lakefile.toml",
             *package_paths(TREE, "verity", "protocols/accounting/space/pous/**")],
    key_params=["responder", "rounds", "loads", "loops", "limit_ns"],
    produces=["run-files/v1"],
    parse=parse,
)
