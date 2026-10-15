"""The bound receipt binds a block's sketch to the prior tile's noised output z.

The design (remote-bandwidth-exhaustion.md §3, §5) proposes a per-block receipt

    receipt_{j,p} = H(sigma, p, sketch(E_j (*) expand(z_{p-1})), z_{p-1}),

with z mixed in *before* the linear sketch, elementwise, so that a relay must move
min(|z|, |E|) bytes per block per pass. The sketch is a fixed +-1 projection whose
sign matrix S is derived from the window challenge sigma (ra-sketch-tag:
K_S = derive(sigma, ...)), so S is constant across the window's passes.

This script shows, bit-exact over the integers, that for a fixed window the block's
entire contribution to every pass's receipt collapses to a static digest
C = contract(S, E, pattern), of shape (rows x L) where L is the number of distinct
z-values the block sees. Each pass's sketch is then C @ z_p, read from C alone; the
block E_j is never touched after C is built. So:

  * one read of the store builds every block's C (cost |C|, not |E|, per block);
  * every subsequent pass's receipts cost zero store reads;
  * if z_p is equal across audit passes (filler, or a static-activation decode step),
    even C @ z is computed once and the pass index p only relabels the hash.

That defeats the multi-pass residency the receipt is meant to force: a window's
R*T bytes of reads scale with T through the number of passes, and this reads the
store at most once per window regardless of T.

`python3 bound_receipt_attack.py [PASSES]` prints JSON lines: the factorization
check (direct vs C @ z, must be exact), the digest-vs-block byte ratio for a decode
tile geometry, and the per-window read reduction over P passes.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


def sketch_direct(S: np.ndarray, E: np.ndarray, zexp: np.ndarray) -> np.ndarray:
    """The receipt's inner sketch, computed the honest way: project E (*) expand(z)."""
    return S @ (E.astype(np.int64) * zexp.astype(np.int64))


def digest(S: np.ndarray, E: np.ndarray, g: np.ndarray, L: int) -> np.ndarray:
    """C[r, l] = sum_{i : g[i] = l} S[r, i] * E[i]. Static in the window: no z, no pass."""
    rows, n = S.shape
    C = np.zeros((rows, L), dtype=np.int64)
    SE = S.astype(np.int64) * E.astype(np.int64)[None, :]
    for l in range(L):
        C[:, l] = SE[:, g == l].sum(axis=1)
    return C


def receipt(sigma: bytes, p: int, inner: np.ndarray, z: np.ndarray) -> str:
    h = hashlib.blake2b(sigma, digest_size=32)
    h.update(p.to_bytes(8, "little"))
    h.update(inner.astype("<i8").tobytes())
    h.update(z.astype("<i8").tobytes())
    return h.hexdigest()


def main(passes: int) -> None:
    rng = np.random.default_rng(20261012)
    rows = 16  # sketch outputs, k_s = 16

    # Decode weight block: 4 KiB int8, shape k_blk x n_blk, batch m. expand(z) broadcasts
    # the prior tile output (one value per output column) across the block's k rows, so the
    # block sees L = n_blk distinct z-values. |z|/|E| = 4m/k in the design; here L/(k_blk*n_blk).
    for (k_blk, n_blk, m) in [(256, 16, 8), (128, 32, 8), (64, 64, 8), (64, 64, 64)]:
        n = k_blk * n_blk
        L = n_blk
        E = rng.integers(-128, 128, size=n, dtype=np.int64)
        # g maps block coordinate (a, b) -> column b in [0, L); a = i // n_blk, b = i % n_blk.
        g = np.arange(n) % n_blk
        sigma = hashlib.blake2b(f"{k_blk}x{n_blk}m{m}".encode(), digest_size=32).digest()
        # S derived from sigma: fixed for the whole window.
        s_rng = np.random.default_rng(int.from_bytes(sigma[:8], "little"))
        S = s_rng.choice(np.array([-1, 1], dtype=np.int64), size=(rows, n))
        C = digest(S, E, g, L)

        ok = True
        receipts_direct, receipts_cached = [], []
        for p in range(passes):
            z = rng.integers(-(1 << 20), 1 << 20, size=L, dtype=np.int64)  # fresh per pass
            zexp = z[g]
            inner_direct = sketch_direct(S, E, zexp)  # reads E
            inner_cached = C @ z                      # reads only C
            ok &= bool(np.array_equal(inner_direct, inner_cached))
            receipts_direct.append(receipt(sigma, p, inner_direct, z))
            receipts_cached.append(receipt(sigma, p, inner_cached, z))

        digest_elems = rows * L
        # Store reads over P passes: honest re-reads E each pass; the attack reads E once to
        # build C, then reads only C (digest_elems int32) each pass.
        honest_reads = passes * n
        attack_reads = n + (passes - 1) * digest_elems * 4  # bytes; C in int32
        print(json.dumps({
            "k_blk": k_blk, "n_blk": n_blk, "m": m, "block_bytes": n, "L": L,
            "factorization_exact": ok,
            "receipts_match": receipts_direct == receipts_cached,
            "digest_bytes_int32": digest_elems * 4,
            "digest_over_block": round(digest_elems * 4 / n, 4),
            "passes": passes,
            "honest_store_bytes": honest_reads,
            "attack_store_bytes": attack_reads,
            "read_reduction_x": round(honest_reads / attack_reads, 1),
        }))

    # The sharper collapse: when z is identical across audit passes (filler blocks have no
    # matmul; a decode step over a frozen store repeats the same tile output), the inner
    # sketch is computed once and the pass index only relabels the hash.
    n = 4096
    E = rng.integers(-128, 128, size=n, dtype=np.int64)
    sigma = hashlib.blake2b(b"static-z", digest_size=32).digest()
    s_rng = np.random.default_rng(int.from_bytes(sigma[:8], "little"))
    S = s_rng.choice(np.array([-1, 1], dtype=np.int64), size=(rows, n))
    z = rng.integers(-(1 << 20), 1 << 20, size=n, dtype=np.int64)
    inner = S @ (E * z)  # computed exactly once
    rs = [receipt(sigma, p, inner, z) for p in range(passes)]
    print(json.dumps({
        "case": "static_z_across_passes", "passes": passes,
        "inner_computed_times": 1, "distinct_receipts": len(set(rs)),
        "store_reads_over_window_bytes": n, "honest_store_reads_bytes": passes * n,
    }))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
