"""Store declaration for the commitment-scheme commit-cost benchmark (research.store README §6.1).

    research run --tool commit_cost --project verity -- python benchmarks/commitments/commit_cost.py --scheme all --leaves 1048576

Key params: the schemes, the leaf count, the leaf size.  ``--reps`` and ``--out`` say how often and where, and do not
enter the derivation.  Imports nothing but ``research.store.tool``.
"""

from __future__ import annotations

import argparse

from research.store.tool import Tool


def parse(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--scheme", action="append")
    ap.add_argument("--leaves", type=int, default=1 << 16)
    ap.add_argument("--leaf-bytes", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ns, _ = ap.parse_known_args(argv)
    schemes = sorted({"frame-v3", "vllm-production"} if not ns.scheme or "all" in ns.scheme else set(ns.scheme))
    return {"scheme": ",".join(schemes), "leaves": ns.leaves, "leaf_bytes": ns.leaf_bytes, "reps": ns.reps}


COMMIT_COST = Tool(
    name="commit_cost",
    version="1",
    command=["python", "benchmarks/commitments/commit_cost.py"],
    closure=["benchmarks/commitments/**", "packages/verity/src/verity/commitments/**", "packages/verity/src/verity/errors.py"],
    key_params=["scheme", "leaves", "leaf_bytes"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)
