#!/usr/bin/env python3
"""Commit cost of each named commitment scheme: hash N leaves of B bytes, then build the tree over them.

    python benchmarks/commitments/commit_cost.py --leaves 65536 --leaf-bytes 256 --reps 5
    research run --tool commit_cost --project verity -- python benchmarks/commitments/commit_cost.py --scheme all --leaves 1048576

Schemes (``--scheme``, repeatable, or ``all``):

    frame-v3            ``verity.ml.commitments.frame_v3``, word leaves: leaf = ``domain.leaf(rank, position, schema, value)``,
                        tree = domain-, level- and index-bound nodes, padded to a power of two
    frame-v3-sha256-row the same tree over ``sha256/row/v1`` row leaves (the value is one row of 16-bit words)
    frame-v3-blake3-row the same tree over ``blake3-keyed/row/v2`` row leaves (plain-Python BLAKE3: slow by design)
    frame-v3-sha512     frame-v3 word leaves through SHA-512 (64-byte domain id, leaves, pads, nodes and root)
    frame-v3-sha512-row the SHA-512 frame over ``sha512/row/v1`` row leaves

Every scheme runs through ``CommitmentScheme``: the leaf phase is ``leaf_layout(...).digest(value)`` then
``tree_leaf`` (what a prover's digest plus the verifier's native step cost together), the tree phase is ``root``.
All run the plain-Python references, so the numbers are the cost of the *framing* on one CPU core under
``hashlib``, not of any serving fast path (GPU trees, thread pools).  Each rep times the leaf phase and the tree
phase separately; the reported value is the median.  After timing, one opening per scheme is produced and checked
through the scheme's ``verify_path``, and the validation status says whether all of them verified.

Writes ``result.json`` (``research/result/v0.1``) to ``$RESEARCH_RUN_DIR`` or ``--out`` when either is set; prints a
one-line summary per scheme either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import ROLE_X
from verity.ml.commitments.scheme import CommitmentScheme

SCHEMES = ("frame-v3", "frame-v3-sha256-row", "frame-v3-blake3-row", "frame-v3-sha512", "frame-v3-sha512-row")
DEFAULT_SCHEMES = ("frame-v3", "frame-v3-sha256-row")
RESULT_SCHEMA = "research/result/v0.1"


def values(n: int, nbytes: int) -> list[bytes]:
    """``n`` distinct deterministic leaf values of ``nbytes`` bytes each (the counter in the first 8 bytes)."""

    block = hashlib.sha256(b"commit-cost").digest() * (nbytes // 32 + 1)
    return [(i.to_bytes(8, "little") + block)[:nbytes] for i in range(n)]


def setup(scheme: str, n: int, nbytes: int) -> CommitmentScheme:
    """The scheme configured with one port ``"v"`` of ``n`` leaves of ``nbytes`` bytes."""

    digest = hashlib.sha256(f"commit-cost/{scheme}".encode()).digest()
    frame = "sha512" if "sha512" in scheme else "sha256"
    if scheme.startswith("frame-v3"):
        domain = CommitmentDomain(digest, 0, RangeIndexedDomain(n), hash=frame)
        if not scheme.endswith("-row"):
            return FrameV3({"v": Port(domain, WordLeaf(f"u8[{nbytes}]", nbytes))})
        if nbytes % 2 or not nbytes:
            raise ValueError("row leaves need a positive even --leaf-bytes (16-bit words)")
        return FrameV3({"v": Port(domain, RowLeaf(scheme.split("-")[2], ROLE_X, 16, nbytes // 2))})
    raise ValueError(f"unknown scheme {scheme!r}; known: {', '.join(SCHEMES)}")


def measure(scheme: str, n: int, nbytes: int, reps: int) -> dict[str, Any]:
    vs = values(n, nbytes)
    impl = setup(scheme, n, nbytes)
    layouts = [impl.leaf_layout("v", i) for i in range(n)]
    leaf_s, tree_s = [], []
    root = b""
    for _ in range(reps):
        t0 = time.perf_counter()
        leaves = [impl.tree_leaf("v", i, lay.digest(v)) for i, (lay, v) in enumerate(zip(layouts, vs))]
        t1 = time.perf_counter()
        root = impl.root("v", leaves)
        t2 = time.perf_counter()
        leaf_s.append(t1 - t0)
        tree_s.append(t2 - t1)
    index = n // 2
    verified = impl.verify_path("v", root, n, index, leaves[index], impl.path("v", leaves, index))
    leaf, tree = statistics.median(leaf_s), statistics.median(tree_s)
    return {"scheme": scheme, "hashes": list(impl.hashes), "leaves": n, "leaf_bytes": nbytes, "reps": reps, "leaf_s": leaf, "tree_s": tree,
            "commit_s": leaf + tree, "root": root.hex(), "opening_verified": verified}


def measurements(row: dict[str, Any]) -> list[dict[str, Any]]:
    key = row["scheme"]
    total = row["commit_s"]
    out = [
        {"name": f"{key}.leaf_s", "value": row["leaf_s"], "unit": "s"},
        {"name": f"{key}.tree_s", "value": row["tree_s"], "unit": "s"},
        {"name": f"{key}.commit_s", "value": total, "unit": "s"},
        {"name": f"{key}.leaves_per_s", "value": row["leaves"] / total if total else None, "unit": "leaf/s"},
        {"name": f"{key}.bytes_per_s", "value": row["leaves"] * row["leaf_bytes"] / total if total else None, "unit": "B/s"},
    ]
    return out


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scheme", action="append", choices=(*SCHEMES, "all"), help=f"repeatable; default {', '.join(DEFAULT_SCHEMES)}; all = every scheme")
    ap.add_argument("--leaves", type=int, default=1 << 16)
    ap.add_argument("--leaf-bytes", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None, help="directory for result.json (default $RESEARCH_RUN_DIR)")
    ns = ap.parse_args(argv)
    if ns.leaves <= 0 or ns.leaf_bytes < 0 or ns.reps <= 0:
        ap.error("--leaves and --reps must be positive, --leaf-bytes nonnegative")
    ns.schemes = list(DEFAULT_SCHEMES) if not ns.scheme else list(SCHEMES) if "all" in ns.scheme else list(dict.fromkeys(ns.scheme))
    return ns


def main(argv: list[str] | None = None) -> int:
    ns = parse(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    rows = [measure(s, ns.leaves, ns.leaf_bytes, ns.reps) for s in ns.schemes]
    for row in rows:
        print(json.dumps({k: row[k] for k in ("scheme", "leaves", "leaf_bytes", "leaf_s", "tree_s", "commit_s", "opening_verified")}))
    out = ns.out or os.environ.get("RESEARCH_RUN_DIR")
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "commit_cost.json").write_text(json.dumps(rows, indent=1) + "\n")
        ok = all(r["opening_verified"] for r in rows)
        result = {
            "schema": RESULT_SCHEMA,
            "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
            "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "commit_cost", "schemes": [r["scheme"] for r in rows], "leaves": ns.leaves,
                                     "leaf_bytes": ns.leaf_bytes, "reps": ns.reps,
                                     "implementation": "python-reference", "python": sys.version.split()[0]},
            "validation": {"status": "passed" if ok else "failed",
                           "checker": "one opening per scheme through verify_path",
                           "detail": ", ".join(f"{r['scheme']}={'ok' if r['opening_verified'] else 'FAILED'}" for r in rows), "evidence": []},
            "measurements": [m for r in rows for m in measurements(r)] + [{"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["commit_cost.json"],
            "artifacts": [],
        }
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, d / "result.json")
    return 0 if all(r["opening_verified"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
