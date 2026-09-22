#!/usr/bin/env python3
"""CPROF: microbenchmarks of the commitment path's CPU primitives (GUIDANCE §32.11).

Measures, on the machine it runs on (record the CPU next to every number):

1. raw ``hashlib.sha256`` streaming throughput vs message size (the compute floor of the
   current implementation),
2. ``verity_vllm.hashing.leaf_hash`` per-leaf cost vs value size (adds the Python framing:
   tag + canonical id + dtype + shape text, four ``update`` calls),
3. ``node_hash`` / ``lift_hash`` per-call cost (the Merkle fold's inner step),
4. ``MerkleTree.__init__`` per-leaf cost from precomputed 32-byte leaf hashes (fold + id
   parse + sort, the real constructor), and ``ids_digest``,
5. a least-squares fit  t_leaf(b) = alpha_leaf + beta * b  and the fold coefficient
   alpha_node, which the commitment cost model consumes:

       T_tree(L, B) ~= L * alpha_leaf + beta * B + (L - 1) * alpha_merkle + read(B)

   (a tree of L leaves has exactly L-1 internal node/lift hashes, lifts included).

Usage:  python scripts/cprof_hash_micro.py --out docs/data/commitment-prof/hash-micro.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import ssl
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
for p in (str(POC),):
    if p not in sys.path:
        sys.path.insert(0, p)

from verity_vllm import hashing  # noqa: E402
from verity_vllm.commit import MerkleTree  # noqa: E402

LEAF_ID = "p=abcdef012345/r=req-0/t=33/ph=decode/l=12/op=layers.12.mlp.down_proj.gemm/port=0"


def bench(fn, *, min_reps: int = 5, min_seconds: float = 0.05) -> dict:
    """Median wall seconds per call; loops until min_seconds of total work or min_reps."""
    times: list[float] = []
    total = 0.0
    while len(times) < min_reps or total < min_seconds:
        t0 = time.perf_counter_ns()
        fn()
        dt = (time.perf_counter_ns() - t0) / 1e9
        times.append(dt)
        total += dt
        if len(times) >= 10_000:
            break
    return {"median_s": statistics.median(times), "min_s": min(times), "reps": len(times)}


def cpu_model() -> str:
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sizes = [64, 256, 1024, 1152, 2048, 4096, 16384, 65536, 262144, 1 << 20, 8 << 20, 64 << 20]
    doc: dict = {
        "schema": "veritor/cprof/hash-micro/v1",
        "cpu": cpu_model(),
        "nproc": os.cpu_count(),
        "python": sys.version.split()[0],
        "openssl": ssl.OPENSSL_VERSION,
        "platform": platform.platform(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "time.perf_counter_ns around each call, median over reps (>=5 reps, >=50 ms per point)",
    }

    # 1. raw sha256
    raw_rows = []
    for n in sizes:
        buf = secrets.token_bytes(n)
        r = bench(lambda b=buf: hashlib.sha256(b).digest())
        raw_rows.append({"bytes": n, **r, "gb_per_s": n / r["median_s"] / 1e9})
    doc["sha256_raw"] = raw_rows

    # 2. leaf_hash (the real leaf function: tag framing + id + dtype + shape text + value)
    leaf_rows = []
    for n in sizes:
        buf = secrets.token_bytes(n)
        shape = (n // 2,)  # bf16 row of n bytes
        r = bench(lambda b=buf, s=shape: hashing.leaf_hash(LEAF_ID, "bfloat16", s, b))
        leaf_rows.append({"bytes": n, **r, "gb_per_s": n / r["median_s"] / 1e9})
    doc["leaf_hash"] = leaf_rows

    # 3. node/lift
    left, right = secrets.token_bytes(32), secrets.token_bytes(32)
    doc["node_hash"] = bench(lambda: hashing.node_hash(left, right))
    doc["lift_hash"] = bench(lambda: hashing.lift_hash(left))

    # 4. MerkleTree fold from precomputed leaves (parse + sort + fold, the real constructor)
    fold_rows = []
    for n in [4, 64, 364, 1024, 4096, 11616, 65536, 262144]:
        leaves = {f"w=micro/c={i}": secrets.token_bytes(32) for i in range(n)}
        r = bench(lambda lv=leaves: MerkleTree(lv, "weights"), min_reps=3, min_seconds=0.2)
        fold_rows.append({"leaves": n, **r, "per_leaf_us": r["median_s"] / n * 1e6})
        ids = sorted(leaves)
        rid = bench(lambda i=ids: hashing.ids_digest(i), min_reps=3, min_seconds=0.05)
        fold_rows[-1]["ids_digest_s"] = rid["median_s"]
    doc["merkle_init"] = fold_rows

    # 5. fits (exclude the largest sizes from the alpha fit so the intercept is driven by
    #    small leaves; beta from the big-buffer asymptote)
    small = [r for r in leaf_rows if r["bytes"] <= 65536]
    a, b = fit_affine([r["bytes"] for r in small], [r["median_s"] for r in small])
    big = leaf_rows[-1]
    beta_big = big["median_s"] / big["bytes"]
    raw_big = raw_rows[-1]
    fold_big = fold_rows[-1]
    doc["fit"] = {
        "alpha_leaf_us": a * 1e6,
        "beta_us_per_kb_smallfit": b * 1024 * 1e6,
        "beta_us_per_kb_bigbuffer": beta_big * 1024 * 1e6,
        "sha256_asymptote_gb_per_s": raw_big["gb_per_s"],
        "alpha_node_us": doc["node_hash"]["median_s"] * 1e6,
        "alpha_merkle_us_per_leaf": fold_big["per_leaf_us"],
        "note": "T_tree(L,B) ~= L*alpha_leaf + beta*B + (L-1)*alpha_merkle (seconds; alpha in us). "
                "alpha_merkle from the 262144-leaf MerkleTree point (includes id parse + sort). "
                "Fit over leaf sizes <= 64 KiB; beta from the 64 MiB point.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["fit"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
