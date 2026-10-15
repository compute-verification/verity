"""The verifier's cost to check one opened block of a chain step, on the CPU: the block's rows of X_s and X_{s+1} opened
under their HM96 roots (`audit.Hm96Domain`, `merkle.verify_opening`), W_s expanded from the coin (SHA-512 in counter mode,
k n bytes), and X_s[block] W_s recomputed exactly (float64, exact below 2^53) and compared with X_{s+1}[block]'s bytes.
The prover's side (the trees) is built first and not timed. X_s is uniform bytes (its distribution doesn't change the
cost) and X_{s+1} is honest on the opened rows. One JSON line a (shape, block).

    python3 verify.py M K B [B ...]
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

from verity.core.protocols.pouw import audit
from verity.ml.commitments import VerificationLimits, verify_opening

from gen import SALT_KEY, binding, domain

COIN = hashlib.sha512(b"verity/chained-tensor-work/coin/v1").digest()


def expand(coin: bytes, s: int, nbytes: int) -> bytes:
    """W_s's bytes: SHA-512(coin || u64 s || u64 ctr), ctr = 0, 1, ..."""
    h = hashlib.sha512
    pre = coin + struct.pack("<Q", s)
    return b"".join(h(pre + struct.pack("<Q", c)).digest() for c in range((nbytes + 63) // 64))[:nbytes]


def product_bytes(x: np.ndarray, wt: np.ndarray) -> np.ndarray:
    c = x.astype(np.float64) @ wt.astype(np.float64).T
    return c.astype(np.int64).astype("<i4").view(np.int8).reshape(x.shape)


def main(argv: list[str]) -> int:
    m, k, blocks = int(argv[0]), int(argv[1]), [int(b) for b in argv[2:]]
    n, s = k // 4, 3
    rng = np.random.default_rng(7)
    wt = np.frombuffer(expand(COIN, s, n * k), np.int8).reshape(n, k)
    xs = rng.integers(-128, 128, (m, k), dtype=np.int8)
    xs1 = rng.integers(-128, 128, (m, k), dtype=np.int8)
    opened = rng.choice(m, max(blocks), replace=False)
    xs1[opened] = product_bytes(xs[opened], wt)
    t0 = time.perf_counter()
    trees = [audit.commit_rows(binding(j), j, [x[i].tobytes() for i in range(m)], audit.HM96, SALT_KEY, nonce=j)
             for j, x in ((s, xs), (s + 1, xs1))]
    t_commit = time.perf_counter() - t0
    limits = VerificationLimits()
    for b in blocks:
        rows = opened[:b]
        openings = [[t.open(int(i)) for i in rows] for t in trees]
        t0 = time.perf_counter()
        ok = all(verify_opening(domain(j, m), t.commitment, o, audit.ROW_SCHEMA, limits)
                 for j, t, os_ in zip((s, s + 1), trees, openings) for o in os_)
        t1 = time.perf_counter()
        w = np.frombuffer(expand(COIN, s, n * k), np.int8).reshape(n, k)
        t2 = time.perf_counter()
        x = np.stack([np.frombuffer(audit.opened_value(audit.HM96, o.value), np.int8) for o in openings[0]])
        y = np.stack([np.frombuffer(audit.opened_value(audit.HM96, o.value), np.int8) for o in openings[1]])
        ok &= bool((product_bytes(x, w) == y).all())
        t3 = time.perf_counter()
        bad = y.copy()
        bad[0, 0] ^= 1
        rejects = not bool((product_bytes(x, w) == bad).all())
        print(json.dumps({"m": m, "k": k, "n": n, "block_rows": b, "accept": ok, "control_rejected": rejects,
                          "s_openings": round(t1 - t0, 6), "s_expand_w": round(t2 - t1, 6), "s_product": round(t3 - t2, 6),
                          "s_total": round(t3 - t0, 6), "openings": 2 * b, "path_len": len(openings[0][0].path),
                          "macs": b * k * n, "w_bytes": n * k, "s_prover_trees_python": round(t_commit, 3),
                          "cpus": os.cpu_count(), "omp_threads": os.environ.get("OMP_NUM_THREADS")}))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
