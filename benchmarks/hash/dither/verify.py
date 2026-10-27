"""After the lease: kbench's dumped receipt rows against compute accounting's numpy twin (benchmarks/pouw/dc/dgemm_ref.py).

    python verify.py OUT [--rows 3]

dgemm_ref.receipts keys chain (i, tn, c) by the row's index in the X it is given, so for a row i of a large X this applies
its loop to the chosen rows with their own indices (`receipts_rows`), using dgemm_ref's mix32, rotl and duo keys; it first
checks that `receipts_rows` equals dgemm_ref.receipts on a small random case. Inputs are regenerated from kbench's seeds
(X: the first m rows of rng(1000 + k), zero rows to the tile; pg: both operands K-periodic). Writes OUT/verify.json and
exits 0 only if every checked row matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pouw" / "dc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dgemm_ref as ref
from hb_kbench import MAX_M, PER, inputs, w_seed, x_seed


def receipts_rows(X: np.ndarray, rows: list[int], W: np.ndarray, seed: int, bn: int, L: int = 32, nch: int = 16):
    """dgemm_ref.receipts for the rows X (r x k) whose indices in the full X are `rows`: C (r x n) and R (r x tiles_n x nch)."""
    r, k = X.shape
    n = W.shape[0]
    tiles_n = n // bn
    idx = ((np.asarray(rows, dtype=np.int64)[:, None, None] * tiles_n + np.arange(tiles_n)[None, :, None]) * nch
           + np.arange(nch)[None, None, :]).astype(np.uint32)
    with np.errstate(over="ignore"):
        y = ref.mix32(np.uint32(seed) ^ ref.mix32(idx))
        z = ref.mix32(y + np.uint32(0x9E3779B9))
        T = np.zeros((r, n), dtype=np.int64)
        X64 = X.astype(np.int64)
        for s in range(k // L):
            T += X64[:, s * L:(s + 1) * L] @ W[:, s * L:(s + 1) * L].astype(np.int64).T
            Tw = (T & 0xFFFFFFFF).astype(np.uint32).reshape(r, tiles_n, bn // nch, nch)
            for u in range(bn // nch):
                t = u % ref.NPOS
                v = y ^ Tw[:, :, u, :] ^ ref.KEY_XOR[t]
                y = v * ref.KEY_MUL[t] + z
                z = ref.rotl(v, ref.ROT[t]) + y
        return T, y ^ ref.rotl(z, 16)


def self_check() -> bool:
    g = np.random.default_rng(1)
    X = g.integers(-127, 128, (6, 96), dtype=np.int8)
    W = g.integers(-127, 128, (64, 96), dtype=np.int8)
    T, R = ref.receipts(X, W, 7, 32)
    T2, R2 = receipts_rows(X[[1, 4, 5]], [1, 4, 5], W, 7, 32)
    return bool((T[[1, 4, 5]] == T2).all() and (R[[1, 4, 5]] == R2).all())


def periodic(a: np.ndarray) -> np.ndarray:
    return np.tile(a[:, :PER], (1, a.shape[1] // PER))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    doc = json.loads((out / "kbench.json").read_text())
    bn = doc["tile"][1]
    res = {"schema": "dither-hb/verify/v1", "twin": "benchmarks/pouw/dc/dgemm_ref.py", "self_check": self_check(),
           "checks": []}
    xs: dict[int, np.ndarray] = {}
    for s in doc["shapes"]:
        name, m, n, k = s["linear"], s["m"], s["n"], s["k"]
        if k not in xs:
            xs[k] = inputs(x_seed(k), MAX_M, k)
        W = inputs(w_seed(name), n, k)
        for arm, g in s.get("gates", {}).items():
            f = out / f"R-{arm}-{name}-m{m}.npy"
            if not f.exists():
                continue
            Rg = np.load(f)
            gr = g["rows"]
            pick = sorted({0, len(gr) // 2, len(gr) - 1})[:args.rows]
            rows = [gr[p] for p in pick]
            X = np.zeros((len(rows), k), dtype=np.int8)
            for j, i in enumerate(rows):
                if i < m:
                    X[j] = xs[k][i]
            if arm == "pg":
                X, Wa, abn = periodic(X), periodic(W), 512 // g["cfg"]["nb"]
            else:
                Wa, abn = W, bn
            _, R = receipts_rows(X, rows, Wa, 7, abn)
            ok = bool((Rg[pick] == R).all())
            res["checks"].append({"linear": name, "m": m, "arm": arm, "rows": rows, "ok": ok,
                                  "bad_words": int((Rg[pick] != R).sum())})
            print(json.dumps(res["checks"][-1]), flush=True)
    res["ok"] = res["self_check"] and bool(res["checks"]) and all(c["ok"] for c in res["checks"])
    (out / "verify.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"verify_ok": res["ok"], "self_check": res["self_check"], "n": len(res["checks"])}))
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
