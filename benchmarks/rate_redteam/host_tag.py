"""How fast a host CPU computes `ro-blake3/v1` tags: the relay through the prover's own host process.

A tag is BLAKE3-keyed(K, u64be j ‖ E_j) over a 2,056-byte block (bandwidth PROTOCOL.md §4). The gate checks the
`blake3` package against the repository's plain-Python reference (`verity/ml/crypto/blake3.py`, loaded by path, so
nothing else of `verity` is imported); the bench then hashes distinct random blocks from a buffer larger than the
caches, in P processes started together, and reports tags/s and the bytes of E tagged per second.

    python host_tag.py gate
    python host_tag.py bench --procs 48 --blocks 16384 --reps 8 --out host-tag.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import blake3

BLOCK = 2056
REF = Path(__file__).resolve().parents[2] / "verity" / "ml" / "crypto" / "blake3.py"


def reference():
    spec = importlib.util.spec_from_file_location("blake3_reference", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tag(key: bytes, j: int, block: bytes | memoryview) -> bytes:
    h = blake3.blake3(key=key)
    h.update(struct.pack(">Q", j))
    h.update(block)
    return h.digest()


def gate(n: int = 24) -> int:
    ref = reference()
    key = os.urandom(32)
    bad = 0
    for j in range(n):
        block = os.urandom(BLOCK)
        if tag(key, j, block) != ref.keyed(key, struct.pack(">Q", j) + block):
            bad += 1
    print(f"gate: {n} tags against the reference, {bad} differ")
    return 1 if bad else 0


def worker(args):
    blocks, reps, barrier, out = args
    key = os.urandom(32)
    buf = memoryview(os.urandom(blocks * BLOCK))
    barrier.wait()
    t0 = time.perf_counter()
    for r in range(reps):
        base = r * blocks
        for i in range(blocks):
            tag(key, base + i, buf[i * BLOCK:(i + 1) * BLOCK])
    out.put((blocks * reps, time.perf_counter() - t0))


def bench(procs: int, blocks: int, reps: int) -> dict:
    ctx = mp.get_context("fork")
    barrier, out = ctx.Barrier(procs), ctx.Queue()
    ps = [ctx.Process(target=worker, args=((blocks, reps, barrier, out),)) for _ in range(procs)]
    for p in ps:
        p.start()
    res = [out.get() for _ in ps]
    for p in ps:
        p.join()
    tags = sum(n for n, _ in res)
    secs = max(s for _, s in res)
    return {"procs": procs, "blocks_per_proc": blocks, "reps": reps, "tags": tags, "seconds": secs,
            "tags_per_s": tags / secs, "e_bytes_per_s": tags * BLOCK / secs,
            "per_proc_tags_per_s": [n / s for n, s in res]}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate")
    b = sub.add_parser("bench")
    b.add_argument("--procs", type=int, nargs="+", default=[1, 12, 24, 48])
    b.add_argument("--blocks", type=int, default=16384)
    b.add_argument("--reps", type=int, default=8)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "gate":
        return gate()
    rows = []
    for p in a.procs:
        row = bench(p, a.blocks, a.reps)
        print(f"procs {p}: {row['tags_per_s']:.4e} tags/s, {row['e_bytes_per_s'] / 1e9:.1f} GB/s of E")
        rows.append(row)
    meta = {"block": BLOCK, "blake3": getattr(blake3, "__version__", "?"), "python": sys.version.split()[0],
            "affinity": sorted(os.sched_getaffinity(0)), "rows": rows}
    Path(a.out).write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
