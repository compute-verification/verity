#!/usr/bin/env python3
"""CPERF 2026-09-17 §5: host-side opening generation / verification cost and opening sizes on the pinned store.

    PYTHONPATH=. python scripts/cprof_openings.py --store <commit dir> [--store-v2 <commit-v2 dir>]

Production API only (``TreeCache.open`` / ``MerkleTree.open`` / ``verify_opening`` / ``open_range``): per RU tree, the
tree build (production, trusted index) then ONE opening per leaf for every leaf of a decode RU and a prefill RU, the
opening JSON size, verification time, and a RangeOpening over one position's contiguous rows.  Weights v2: one
whole-tensor RangeOpening (the v2 path the prover actually uses) and one row opening.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

from verity_vllm import identity  # noqa: E402
from verity_vllm.capture import CaptureReader  # noqa: E402
from verity_vllm.commit import MerkleTree, TreeCache, verify_opening, verify_range_opening  # noqa: E402


def bench_tree(reader: CaptureReader, rid: str, step: int) -> dict:
    ru = reader.ru(rid, step)
    t0 = time.perf_counter()
    tree = MerkleTree(ru.leaves(recompute=False), ru.tree_id)
    t_build = time.perf_counter() - t0
    ids = identity.tree_sorted([e.canonical_id for e in ru.index.entries])
    values = {c: reader.value(c) for c in ids}
    t0 = time.perf_counter()
    ops = [tree.open(values[c]) for c in ids]
    t_open = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok = all(verify_opening(tree.root, tree.leaf_count, o) for o in ops)
    t_verify = time.perf_counter() - t0
    js = [len(json.dumps(o.to_json(), separators=(",", ":"))) for o in ops[: min(64, len(ops))]]
    path_b = [32 * sum(1 for p in o.path if p is not None) for o in ops]
    # one position's rows = a contiguous run in tree order? (ids sort by t then op) -> take the first 30 ids of position t0
    t_first = identity.parse(ids[0]).t
    run = [c for c in ids if identity.parse(c).t == t_first]
    rng = None
    try:
        lo = tree.index_of(run[0])
        hi = tree.index_of(run[-1]) + 1
        if hi - lo == len(run):
            t0 = time.perf_counter()
            rp = tree.range_path(lo, hi)
            t_rp = time.perf_counter() - t0
            rng = {"rows": len(run), "range_path_bytes": 32 * sum((l is not None) + (r is not None) for l, r in rp), "range_path_s": t_rp,
                   "individual_paths_bytes": sum(path_b[lo:hi]), "value_bytes": sum(len(values[c].data) for c in run)}
    except Exception as e:  # noqa: BLE001
        rng = {"error": repr(e)}
    return {"tree": ru.tree_id, "phase": ru.index.phase, "leaves": tree.leaf_count, "depth": tree.depth, "build_trusted_s": t_build,
            "open_all_s": t_open, "open_us_per_leaf": t_open / len(ids) * 1e6, "verify_all_s": t_verify, "verify_us_per_leaf": t_verify / len(ids) * 1e6,
            "verified": ok, "path_bytes_median": statistics.median(path_b), "path_bytes_max": max(path_b),
            "opening_json_bytes_median": statistics.median(js), "value_bytes_median": statistics.median(len(values[c].data) for c in ids),
            "one_position_range": rng}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--store-v2", default=None)
    args = ap.parse_args()
    reader = CaptureReader(args.store)
    keys = reader.ru_keys()
    decode = next(k for k in keys if reader.ru(*k).index.phase == "decode")
    prefill = next(k for k in keys if reader.ru(*k).index.phase == "prefill")
    out = {"schema": "veritor/cperf/openings/v1", "store": args.store, "decode_ru": bench_tree(reader, *decode), "prefill_ru": bench_tree(reader, *prefill)}
    # TreeCache.open end to end (build + open) for one id in a decode RU, cold cache
    cid = reader.ru(*decode).index.entries[min(100, len(reader.ru(*decode).index.entries) - 1)].canonical_id
    t0 = time.perf_counter()
    tc = TreeCache(reader, recompute_leaves=True)
    o = tc.open(cid)
    out["treecache_open_one_decode_row_cold_recompute_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    o = tc.open(cid)
    out["treecache_open_one_decode_row_warm_s"] = time.perf_counter() - t0
    out["one_row_opening_json_bytes"] = len(json.dumps(o.to_json(), separators=(",", ":")))
    for label, store in (("weights_v1", args.store), ("weights_v2", args.store_v2)):
        if not store:
            continue
        r2 = CaptureReader(store)
        w = r2.weights
        t0 = time.perf_counter()
        tc2 = TreeCache(r2, recompute_leaves=False)
        tree = tc2.tree("weights")
        t_build = time.perf_counter() - t0
        # a mid-size tensor (first row-block tensor if v2, else the first tensor)
        entry = next((e for e in w.index.entries if e.layout.kind != "whole"), w.index.entries[0])
        t0 = time.perf_counter()
        op = tc2.open(entry.canonical_id)
        t_open = time.perf_counter() - t0
        d = {"format_version": w.index.format_version, "leaves": tree.leaf_count, "depth": tree.depth, "build_trusted_s": t_build,
             "tensor": entry.canonical_id.rsplit("/", 1)[-1], "tensor_bytes": entry.nbytes, "open_s": t_open, "kind": type(op).__name__}
        if hasattr(op, "path_bytes"):
            d["range_path_bytes"] = op.path_bytes()
            d["range_leaves"] = len(op.leaves)
            t0 = time.perf_counter()
            d["verified"] = verify_range_opening(tree.root, tree.leaf_count, op)
            d["verify_s"] = time.perf_counter() - t0
        else:
            d["path_bytes"] = 32 * sum(1 for p in op.path if p is not None)
            t0 = time.perf_counter()
            d["verified"] = verify_opening(tree.root, tree.leaf_count, op)
            d["verify_s"] = time.perf_counter() - t0
        if w.index.format_version == 2:
            # one single row leaf of that tensor
            row_id = entry.block_id(0)
            t0 = time.perf_counter()
            ro = tree.open(w.block_values(entry.canonical_id)[0])
            d["one_row_open_s"] = time.perf_counter() - t0
            d["one_row_path_bytes"] = 32 * sum(1 for p in ro.path if p is not None)
            d["one_row_json_bytes"] = len(json.dumps(ro.to_json(), separators=(",", ":")))
            d["row_id"] = row_id
        out[label] = d
    print("CPERF-OPENINGS-JSON " + json.dumps(out, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
