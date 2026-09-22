#!/usr/bin/env python3
"""C1 pilot driver: write the packed plan/pack for a commit store (or a synthetic store of a given shape), run the Rust
engine `vce`, compare roots with commitments.json, and time the Python packed prototype on the same host/session.

    PYTHONPATH=. python scripts/cprof_c1_plan.py --store <commit dir> --vce <path to vce> --out plan_dir
    PYTHONPATH=. python scripts/cprof_c1_plan.py --synthetic 64x364+4x11586 --vce ... --out plan_dir

Plan: "VCE1" u32 ntrees { u32 nleaves { u32 hlen header u64 offset u64 nbytes } }; pack: values in tree order.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

import numpy as np  # noqa: E402

from verity_vllm import identity  # noqa: E402
from verity_vllm.commit_bench import fasttree as ft  # noqa: E402
from verity_vllm.commit_bench.prov import provenance  # noqa: E402
from verity_vllm.commit_bench.semantic import POS_LEAF_TAG  # noqa: E402


def write_plan(out: Path, trees: list[list[tuple[bytes, bytes]]]) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    plan, pack = out / "plan.bin", out / "pack.bin"
    off = 0
    with plan.open("wb") as fp, pack.open("wb") as fk:
        fp.write(b"VCE1" + len(trees).to_bytes(4, "big"))
        for leaves in trees:
            fp.write(len(leaves).to_bytes(4, "big"))
            for h, v in leaves:
                fp.write(len(h).to_bytes(4, "big") + h + off.to_bytes(8, "big") + len(v).to_bytes(8, "big"))
                fk.write(v)
                off += len(v)
    return plan, pack


def run_vce(vce: str, plan: Path, pack: Path, threads: int, pos: bool, reps: int) -> tuple[list[str], dict]:
    cmd = [vce, "--plan", str(plan), "--pack", str(pack), "--threads", str(threads), "--reps", str(reps)] + (["--pos"] if pos else [])
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    roots = [line.split()[1] for line in r.stdout.strip().splitlines()]
    kv = dict(tok.split("=") for tok in r.stderr.strip().split()[1:])
    return roots, kv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store")
    ap.add_argument("--synthetic", help="e.g. 60x364+4x11586 (trees x leaves of 1152 B)")
    ap.add_argument("--vce", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    expected: list[str] | None = None
    if a.store:
        from verity_vllm.capture import CaptureReader

        reader = CaptureReader(a.store)
        tdoc = reader.commitments["trees"]
        trees, expected = [], []
        for rid, step in reader.ru_keys():
            ru = reader.ru(rid, step)
            by = {e.canonical_id: e for e in ru.index.entries}
            mm = np.memmap(ru.dir / "values.bin", dtype=np.uint8, mode="r")
            ids = identity.tree_sorted(list(by))
            trees.append([(ft.leaf_header(c, by[c].dtype, by[c].shape), bytes(mm[by[c].offset : by[c].offset + by[c].nbytes])) for c in ids])
            expected.append(tdoc[ru.tree_id]["root"])
        source = a.store
    else:
        rng = random.Random(5)
        trees = []
        P = "ec44fc6dd686"
        for part in a.synthetic.split("+"):
            nt, nl = (int(x) for x in part.split("x"))
            for t in range(nt):
                ids = sorted(f"p={P}/r=472ba20e-f4a0-5b64-90fe-ddb4971deff7/t={i // 362}/ph=decode/l={i % 30}/op=layers.{i % 30}.mlp.down_proj.gemm/port={i}" for i in range(nl))
                trees.append([(ft.leaf_header(c, "bfloat16", (576,)), rng.randbytes(1152)) for c in ids])
        source = f"synthetic:{a.synthetic}"
    leaves = sum(len(t) for t in trees)
    nbytes = sum(len(v) for t in trees for _, v in t)
    plan, pack = write_plan(out, trees)

    # Python packed prototype, same host/session (headers precomputed, values in memory)
    py = []
    for _ in range(min(a.reps, 3)):
        t0 = time.perf_counter()
        py_roots = [ft.fold([ft.leaf_digest(h, v) for h, v in t]) for t in trees]
        py.append(time.perf_counter() - t0)
    py_hex = [r.hex() for r in py_roots]
    t0 = time.perf_counter()
    pos_py = [ft.fold([ft.leaf_digest(POS_LEAF_TAG, v) for _, v in t]) for t in trees]
    py_pos_s = time.perf_counter() - t0
    # contiguous SHA-256 rate on this host (hash throughput floor input)
    import hashlib

    blob = b"".join(v for t in trees for _, v in t)
    t0 = time.perf_counter()
    hashlib.sha256(blob).digest()
    sha_s = time.perf_counter() - t0
    # memory bandwidth proxy: numpy copy of the pack
    arr = np.frombuffer(blob, dtype=np.uint8)
    t0 = time.perf_counter()
    arr.copy()
    copy_s = time.perf_counter() - t0

    results = {}
    for th in (int(x) for x in a.threads.split(",")):
        for pos in (False, True):
            roots, kv = run_vce(a.vce, plan, pack, th, pos, a.reps)
            key = f"vce_{'pos' if pos else 'prod'}_t{th}"
            results[key] = {"best_s": float(kv["best_s"]), "median_s": float(kv["median_s"]), "touch_s": float(kv["touch_s"]),
                            "us_per_leaf": float(kv["median_s"]) / leaves * 1e6, "GB_s": nbytes / float(kv["median_s"]) / 1e9,
                            "roots_match_python": sum(r == e for r, e in zip(roots, pos_py if pos else py_roots, strict=True)) if not pos else sum(r == e.hex() for r, e in zip(roots, pos_py, strict=True)),
                            "roots_match_commitments_json": (sum(r == e for r, e in zip(roots, expected, strict=True)) if (expected and not pos) else None)}
            if not pos:
                results[key]["roots_match_python"] = sum(r == e for r, e in zip(roots, py_hex, strict=True))
    doc = {"schema": "veritor/cperf/c1/v1", "source": source, "provenance": provenance(), "trees": len(trees), "leaves": leaves, "bytes": nbytes,
           "python_packed": {"median_s": statistics.median(py), "us_per_leaf": statistics.median(py) / leaves * 1e6, "roots_match_commitments_json": sum(r == e for r, e in zip(py_hex, expected, strict=True)) if expected else None},
           "python_pos": {"s": py_pos_s, "us_per_leaf": py_pos_s / leaves * 1e6},
           "floors": {"sha256_contiguous_s": sha_s, "sha256_GB_s": nbytes / sha_s / 1e9, "sha_blocks_per_leaf_prod": statistics.median((len(h) + 8 + len(v) + 9 + 63) // 64 for t in trees for h, v in t),
                      "sha_blocks_per_leaf_pos": statistics.median((len(POS_LEAF_TAG) + 8 + len(v) + 9 + 63) // 64 for t in trees for _, v in t),
                      "memcpy_s": copy_s, "memcpy_GB_s": nbytes / copy_s / 1e9,
                      "T_lower_hash_s_prod": None, "note": "T_lower_hash = total SHA blocks (leaves + 2 per node) / (sha256_GB_s / 64 B per block)"},
           "vce": results}
    blocks_prod = sum((len(h) + 8 + len(v) + 9 + 63) // 64 for t in trees for h, v in t) + sum(2 * (len(t) - 1) for t in trees)
    blocks_pos = sum((len(POS_LEAF_TAG) + 8 + len(v) + 9 + 63) // 64 for t in trees for _, v in t) + sum(2 * (len(t) - 1) for t in trees)
    rate_blocks = (nbytes / 64) / sha_s
    doc["floors"]["T_lower_hash_s_prod"] = blocks_prod / rate_blocks
    doc["floors"]["T_lower_hash_s_pos"] = blocks_pos / rate_blocks
    doc["floors"]["T_lower_mem_s"] = copy_s / 2  # one read of the bytes (copy = read + write)
    print("CPERF-C1-JSON " + json.dumps(doc, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
