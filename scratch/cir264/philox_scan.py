"""Seeds whose stream word at one position is within --within of 0 or of 2^32 - 1: the words `tl.rand(seed, pos)` maps
to its smallest uniforms (words 0, 1, 2^32 - 2 and 2^32 - 1 give the floor itself).

Scans seed = 0 .. --seeds - 1 at --pos over every core of the job; each hit is checked with the scalar
`sampling.stream_key`.  Prints one JSON object and writes it to --out.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from verity.definitions.registry import sampling as SA
from verity.definitions.registry.sampling_rows import philox_word0

CHUNK = 1 << 20


def scan(job):
    lo, hi, pos, within = job
    hits = []
    for a in range(lo, hi, CHUNK):
        seed = np.arange(a, min(a + CHUNK, hi), dtype=np.uint64)
        w = philox_word0(pos & 0xFFFFFFFF, pos >> 32, 0, 0, seed & np.uint64(0xFFFFFFFF), seed >> np.uint64(32)).astype(np.int64)
        near = np.flatnonzero((w < within) | (w >= (1 << 32) - within))
        hits += [(int(seed[i]), int(w[i])) for i in near]
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=1002)
    ap.add_argument("--seeds", type=int, default=1 << 32)
    ap.add_argument("--within", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cores = len(os.sched_getaffinity(0))
    step = 1 << 26
    jobs = [(lo, min(lo + step, a.seeds), a.pos, a.within) for lo in range(0, a.seeds, step)]
    with Pool(cores) as pool:
        hits = sorted(h for part in pool.imap_unordered(scan, jobs) for h in part)
    for seed, word in hits:
        assert SA.stream_key(seed, a.pos) == word, (seed, word)
    floor = [(s, w) for s, w in hits if w in (0, 1, (1 << 32) - 2, (1 << 32) - 1)]
    out = {"pos": a.pos, "seeds_scanned": a.seeds, "within": a.within, "cores": cores, "hits": len(hits), "floor_hits": floor,
           "all": hits}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "all"}))
    print("first hits", hits[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
