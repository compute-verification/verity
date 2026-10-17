#!/usr/bin/env python3
"""Classify the words that depth adds to the same-word fold's kernel, from `fold_root_kernel`'s E lines.

For each E line with R >= 2 (a word w' in K_R outside K_1, at keys (a0, a1) and word w) this computes Φ = F_w and
Ψ = F_{w'} on every state and records: whether Ψ commutes with Φ; the order of E = Ψ Φ^-1; whether E is an XOR by a
constant (and which); and the least d with Ψ^d = Φ^d. If Ψ commutes with Φ then Ψ^R = E^R Φ^R, so w' is in K_R exactly
when the order of E divides R.

  python3 fold_root_classify.py --n N FILE.txt [FILE.txt ...] [--max-per-file M] [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_root_sweep as sw


def fold_table(n: int, r1: int, r2: int, a0: int, a1: int, w) -> np.ndarray:
    """F_w on every n-bit state."""
    m = (1 << n) - 1
    y = np.arange(1 << n, dtype=np.uint64)
    for (x, c, a, r) in ((w[0], w[1], a0, r1), (w[2], w[3], a1, r2)):
        y = (((y << np.uint64(r)) | (y >> np.uint64(n - r))) & np.uint64(m)) ^ np.uint64(x)
        y = (y * np.uint64(a) + np.uint64(c)) & np.uint64(m)
    return y.astype(np.int64)


def order(p: np.ndarray, cap: int = 1 << 20) -> int:
    """The order of the permutation p (lcm of its cycle lengths)."""
    seen = np.zeros(len(p), dtype=bool)
    out = 1
    for s in range(len(p)):
        if seen[s]:
            continue
        ln, x = 0, s
        while not seen[x]:
            seen[x] = True
            x = p[x]
            ln += 1
        out = int(np.lcm(out, ln))
        if out > cap:
            return out
    return out


def near(a: int) -> int:
    """The largest k with a ≡ ±1 mod 2^k."""
    v = lambda x: (x & -x).bit_length() - 1 if x else 64
    return max(v(a - 1), v(a + 1))


def classify(n: int, a0: int, a1: int, w, w2, rmax: int = 6) -> dict:
    r1, r2 = sw.ROT[n]
    phi, psi = fold_table(n, r1, r2, a0, a1, w), fold_table(n, r1, r2, a0, a1, w2)
    inv = np.empty_like(phi)
    inv[phi] = np.arange(len(phi))
    e = psi[inv]
    xor = np.unique(e ^ np.arange(len(e)))
    d, p, q = None, phi.copy(), psi.copy()
    for j in range(1, rmax + 1):
        if np.array_equal(p, q):
            d = j
            break
        p, q = phi[p], psi[q]
    return {"commutes": bool(np.array_equal(psi[phi], phi[psi])), "order_E": order(e),
            "E_xor": int(xor[0]) if len(xor) == 1 else None, "period": d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--max-per-file", type=int, default=0, help="classify at most M distinct (job, w') a file (0: all)")
    ap.add_argument("--json")
    a = ap.parse_args()
    total = collections.Counter()
    examples = []
    r1 = sw.ROT[a.n][0]
    for f in a.files:
        seen, degenerate = {}, set()
        with open(f) as fh:
            for line in fh:
                if line.startswith("S "):
                    p = list(map(int, line.split()[1:]))
                    if p[-1]:
                        degenerate.add(tuple(p[:6]))
                if not line.startswith("E "):
                    continue
                p = list(map(int, line.split()[1:]))
                key, rr, w2 = tuple(p[:6]), p[6], tuple(p[7:11])
                if rr >= 2 and (key, w2) not in seen:
                    seen[(key, w2)] = rr
        items = [it for it in seen.items() if it[0][0] not in degenerate]
        if a.max_per_file and len(items) > a.max_per_file:
            items = items[:: max(1, len(items) // a.max_per_file)][: a.max_per_file]
        c = collections.Counter()
        for (key, w2), rr in items:
            r = classify(a.n, key[0], key[1], key[2:6], w2)
            lvl = max(near(key[0]), near(key[1]))
            c["elements"] += 1
            c["commutes"] += r["commutes"]
            c[f"level{lvl}"] += 1
            c[f"level{lvl}_commutes"] += r["commutes"]
            c["E_xor"] += r["E_xor"] is not None
            c[f"order_{r['order_E'] if r['order_E'] <= 64 else 'big'}"] += 1
            c[f"period_{r['period']}"] += 1
            if not r["commutes"] and len(examples) < 20 and lvl < r1 + 2:
                examples.append({"file": Path(f).name, "keys": key[:2], "w": key[2:], "w2": w2, **r})
        c["keys"] = len({k[0][:2] for k, _ in items})
        c["degenerate_jobs_skipped"] = len(degenerate)
        print(f"{Path(f).name}: {dict(sorted(c.items()))}")
        total.update(c)
    print(f"total: {dict(sorted(total.items()))}")
    if examples:
        print("non-commuting examples:", *examples[:5], sep="\n  ")
    if a.json:
        Path(a.json).write_text(json.dumps({"total": total, "non_commuting": examples}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
