"""Store declarations for the commitment benchmarks (research.store README §6.1).

    research run --tool commit_cost --project verity -- python benchmarks/commitments/commit_cost.py --scheme all --leaves 1048576
    research run --tool hiding_leaf_cost --campaign salted-leaves -- python benchmarks/commitments/hiding_leaf_cost.py
    research run --tool serving_commit_cost --campaign C -- python benchmarks/commitments/serving_commit_cost.py --input-set DIR --n 1024

Key params: for commit_cost the schemes, the leaf count and the leaf size; for hiding_leaf_cost the row sizes and the row
count; for serving_commit_cost the input set, the instance count and the workers.  ``--reps`` and ``--out`` say how often and where, and do not enter the derivation.  Imports nothing but
``research.store.tool``.
"""

from __future__ import annotations

import argparse

from research.pythonpath import package_paths, workspace_root
from research.store.tool import Tool

TREE = workspace_root(__file__)


def parse(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--scheme", action="append")
    ap.add_argument("--leaves", type=int, default=1 << 16)
    ap.add_argument("--leaf-bytes", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ns, _ = ap.parse_known_args(argv)
    default = {"frame-v3", "frame-v3-sha256-row", "vllm-v1"}
    every = default | {"frame-v3-blake3-row", "frame-v3-sha512", "frame-v3-sha512-row", "vllm-v1-sha512"}
    schemes = sorted(default if not ns.scheme else every if "all" in ns.scheme else set(ns.scheme))
    return {"scheme": ",".join(schemes), "leaves": ns.leaves, "leaf_bytes": ns.leaf_bytes, "reps": ns.reps}


COMMIT_COST = Tool(
    name="commit_cost",
    version="1",
    command=["python", "benchmarks/commitments/commit_cost.py"],
    closure=["benchmarks/commitments/**", *package_paths(TREE, "verity.ml.commitments", "**"), *package_paths(TREE, "verity", "errors.py")],
    key_params=["scheme", "leaves", "leaf_bytes"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)


def parse_hiding(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--row-bytes", type=int, action="append")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    ns, _ = ap.parse_known_args(argv)
    return {"row_bytes": ",".join(map(str, ns.row_bytes or (4, 256, 1536, 3072, 4096, 8192))), "rows": ns.rows}


HIDING_LEAF_COST = Tool(
    name="hiding_leaf_cost",
    version="1",
    command=["python", "benchmarks/commitments/hiding_leaf_cost.py"],
    closure=["benchmarks/commitments/**", *package_paths(TREE, "verity.ml.commitments", "**"), *package_paths(TREE, "verity", "errors.py"),
             "verity/ml/windows/hiding.py", "integrations/vllm/verity_vllm/commit/scheme.py"],
    key_params=["row_bytes", "rows"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_hiding,
)


def parse_serving(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--input-set", default=None)
    ap.add_argument("--input-set-art", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ns, _ = ap.parse_known_args(argv)
    return {"input_set": ns.input_set_art or ns.input_set, "n": ns.n, "workers": ns.workers}


SERVING_COMMIT_COST = Tool(
    name="serving_commit_cost",
    version="1",
    command=["python", "benchmarks/commitments/serving_commit_cost.py"],
    closure=["benchmarks/commitments/**", *package_paths(TREE, "verity.ml.commitments", "**"), *package_paths(TREE, "verity", "errors.py"),
             "verity/ml/windows/rows.py", "verity/ml/windows/hiding.py",
             "benchmarks/numerical/python/verity_numerical/bench/input_sets.py"],
    key_params=["input_set", "n", "workers"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_serving,
)
