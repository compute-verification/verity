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
    ap.add_argument("--impl", default="python")
    ap.add_argument("--out", default=None)
    ns, _ = ap.parse_known_args(argv)
    default = {"frame-v3", "frame-v3-sha256-row", "vllm-v1"}
    every = default | {"frame-v3-blake3-row"}
    schemes = sorted(default if not ns.scheme else every if "all" in ns.scheme else set(ns.scheme))
    out = {"scheme": ",".join(schemes), "leaves": ns.leaves, "leaf_bytes": ns.leaf_bytes, "reps": ns.reps}
    if ns.impl != "python":            # the plain references keep their derivation keys
        out["impl"] = ns.impl
    return out


COMMIT_COST = Tool(
    name="commit_cost",
    version="1",
    command=["python", "benchmarks/commitments/commit_cost.py"],
    closure=["benchmarks/commitments/**", "packages/verity/src/verity/commitments/**", "packages/verity/src/verity/errors.py",
             "backends/shared/hash_gpu/**"],
    key_params=["scheme", "leaves", "leaf_bytes", "impl"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)
