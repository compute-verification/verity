#!/usr/bin/env python3
"""CPROF: instrumented profile of the CURRENT commitment path over a capture store (§32.11).

For every tree of the store (RU trees, boundary trees, weights tree) the three phases of
``verity_vllm.commit`` are timed separately, calling the exact same functions the production
path calls:

    read  -- bytes out of ``values.bin`` (memmap -> bytes), what ``RUReader.read`` does
    hash  -- ``hashing.leaf_hash`` over each value (what ``Value.leaf()`` does)
    fold  -- ``MerkleTree(leaves)`` from the leaf hashes (id parse + sort + node/lift hashes)

plus the un-split end-to-end ``build_commitments(recompute_leaves=True/False)`` wall
(the number ``replay.json commitments.seconds`` reports), and GPU utilisation samples during
that wall (the commitment path runs on the CPU; the samples document the idle GPU).

Reps: the first rep is reported apart (page-cache cold-ish); the median of the remaining
reps is the steady-state number.

Usage:
    python scripts/cprof_store_profile.py --store <commit dir> \
        --out docs/data/commitment-prof/store-profile.json [--reps 5] [--gpu-sample]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
for p in (str(POC),):
    if p not in sys.path:
        sys.path.insert(0, p)

from verity_vllm import hashing  # noqa: E402
from verity_vllm.capture import CaptureReader  # noqa: E402
from verity_vllm.commit import MerkleTree, boundary_tree, build_commitments  # noqa: E402


def node_count(leaf_count: int) -> int:
    """Internal hashes (node + lift) of a tree with ``leaf_count`` leaves: exactly L - 1... no:
    pairs + lifts per level; a lift is one hash too."""
    total, width = 0, leaf_count
    while width > 1:
        total += width // 2 + (width % 2)
        width = (width + 1) // 2
    return total


class GpuSampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.samples: list[dict] = []
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().split(",")
                self.samples.append({"util_pct": float(out[0]), "power_w": float(out[1]),
                                     "mem_mib": float(out[2]), "t": time.time()})
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        utils = [s["util_pct"] for s in self.samples]
        return {"samples": len(self.samples), "util_mean_pct": statistics.mean(utils),
                "util_max_pct": max(utils),
                "power_mean_w": statistics.mean(s["power_w"] for s in self.samples)}


def profile_ru(reader: CaptureReader, request_id: str, step: int, reps: int) -> dict:
    row: dict = {"tree": f"ru/{request_id}/{step}", "kind": "ru"}
    phases: dict[str, list[float]] = {"read": [], "hash": [], "fold": []}
    for _ in range(reps):
        ru = reader.ru(request_id, step)  # fresh RUReader (fresh memmap) every rep
        entries = ru.index.entries
        t0 = time.perf_counter()
        raws = [(e, ru.read(e.canonical_id)) for e in entries]
        t1 = time.perf_counter()
        leaves = {e.canonical_id: hashing.leaf_hash(e.canonical_id, e.dtype, e.shape, raw)
                  for e, raw in raws}
        t2 = time.perf_counter()
        tree = MerkleTree(leaves, ru.tree_id)
        t3 = time.perf_counter()
        phases["read"].append(t1 - t0)
        phases["hash"].append(t2 - t1)
        phases["fold"].append(t3 - t2)
        row.setdefault("leaves", len(leaves))
        row.setdefault("bytes", sum(len(raw) for _, raw in raws))
        row.setdefault("depth", tree.depth)
        row.setdefault("phase", ru.index.phase)
        row.setdefault("positions", len(ru.index.positions))
    row["node_hashes"] = node_count(row["leaves"])
    for k, v in phases.items():
        row[f"{k}_first_s"] = v[0]
        row[f"{k}_s"] = statistics.median(v[1:]) if len(v) > 1 else v[0]
    row["total_s"] = row["read_s"] + row["hash_s"] + row["fold_s"]
    return row


def profile_weights(reader: CaptureReader, reps: int) -> dict:
    w = reader.weights
    row: dict = {"tree": "weights", "kind": "weights",
                 "format_version": w.index.format_version,
                 "tensors": len(w.index.entries), "leaves": w.index.leaf_count()}
    phases: dict[str, list[float]] = {"read": [], "hash": [], "fold": []}
    for _ in range(reps):
        t0 = time.perf_counter()
        raws = [(e, w.read(e.canonical_id)) for e in w.index.entries]
        t1 = time.perf_counter()
        leaves: dict[str, bytes] = {}
        for e, raw in raws:
            if e.layout.kind == "whole":
                leaves[e.canonical_id] = hashing.leaf_hash(e.canonical_id, e.dtype, e.shape, raw)
            else:
                rb = e.row_nbytes
                for start in e.block_starts():
                    end = min(start + e.layout.rows_per_leaf, e.rows)
                    leaves[e.block_id(start)] = hashing.leaf_hash(
                        e.block_id(start), e.dtype, e.block_shape(start, end), raw[start * rb:end * rb])
        t2 = time.perf_counter()
        MerkleTree(leaves, "weights")
        t3 = time.perf_counter()
        phases["read"].append(t1 - t0)
        phases["hash"].append(t2 - t1)
        phases["fold"].append(t3 - t2)
        row.setdefault("bytes", sum(len(raw) for _, raw in raws))
    row["node_hashes"] = node_count(row["leaves"])
    for k, v in phases.items():
        row[f"{k}_first_s"] = v[0]
        row[f"{k}_s"] = statistics.median(v[1:]) if len(v) > 1 else v[0]
    row["total_s"] = row["read_s"] + row["hash_s"] + row["fold_s"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu-sample", action="store_true")
    args = ap.parse_args()

    store = Path(args.store)
    reader = CaptureReader(store)
    doc: dict = {
        "schema": "veritor/cprof/store-profile/v1",
        "store": str(store),
        "profile_id": reader.profile_id,
        "reps": args.reps,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": ("per-phase perf_counter around the exact production functions "
                   "(RUReader.read / hashing.leaf_hash / MerkleTree); first rep apart, median of the rest"),
    }

    trees: list[dict] = []
    for request_id, step in reader.ru_keys():
        trees.append(profile_ru(reader, request_id, step, args.reps))

    for request_id in reader.request_ids():
        if not reader.has_boundary(request_id):
            continue
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            tree = boundary_tree(reader, request_id)
            times.append(time.perf_counter() - t0)
        b = reader.boundary(request_id)
        values = b.leaves(reader.profile_id)
        trees.append({"tree": f"boundary/{request_id}", "kind": "boundary",
                      "leaves": tree.leaf_count, "bytes": sum(len(v.data) for v in values),
                      "node_hashes": node_count(tree.leaf_count), "depth": tree.depth,
                      "total_first_s": times[0],
                      "total_s": statistics.median(times[1:]) if len(times) > 1 else times[0]})

    if reader.has_weights():
        trees.append(profile_weights(reader, args.reps))

    doc["trees"] = trees
    agg: dict[str, dict] = {}
    for t in trees:
        a = agg.setdefault(t["kind"], {"trees": 0, "leaves": 0, "bytes": 0, "total_s": 0.0,
                                       "read_s": 0.0, "hash_s": 0.0, "fold_s": 0.0})
        a["trees"] += 1
        a["leaves"] += t.get("leaves", 0)
        a["bytes"] += t.get("bytes", 0)
        a["total_s"] += t["total_s"]
        for k in ("read_s", "hash_s", "fold_s"):
            a[k] += t.get(k, 0.0)
    doc["by_kind"] = agg

    # end-to-end build_commitments, as replay.py times it (fresh reader each rep)
    for recompute in (True, False):
        times = []
        gpu = GpuSampler() if (args.gpu_sample and recompute) else None
        ctx = gpu if gpu is not None else _null()
        with ctx:
            for _ in range(args.reps):
                fresh = CaptureReader(store)
                t0 = time.perf_counter()
                build_commitments(fresh, recompute_leaves=recompute, write=False)
                times.append(time.perf_counter() - t0)
        key = "build_commitments_recompute" if recompute else "build_commitments_trusted_index"
        doc[key] = {"first_s": times[0],
                    "median_s": statistics.median(times[1:]) if len(times) > 1 else times[0],
                    "all_s": times}
        if gpu is not None:
            doc["gpu_during_commit"] = gpu.summary()

    # the production run's own recorded timings, for the same-store cross-check
    replay_json = store / "replay.json"
    if replay_json.exists():
        r = json.loads(replay_json.read_text())
        doc["store_recorded"] = {
            "commitments_seconds": (r.get("commitments") or {}).get("seconds"),
            "weights_seconds": ((r.get("timing") or {}).get("weights") or {}).get("seconds"),
            "capture": r.get("capture"),
            "capture_timing": (r.get("timing") or {}).get("capture"),
            "run_s": (r.get("timing") or {}).get("run_s"),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("by_kind", "build_commitments_recompute",
                                          "build_commitments_trusted_index") if k in doc}, indent=1))
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
