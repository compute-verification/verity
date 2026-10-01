"""TT-stride(S)'s liveness census for NCP-FP8 on sm_120 (CPU, bit-exact): the bound accumulators of the chained GEMM over
D-3s's formed operands, on every census family, across salts.

The families are FP8 security's 26 (`pearlc_census.py`'s FAMILIES, ACTIVATION_FAMILIES, OUTLIER_FAMILIES and FREEZE_FAMILIES
at their seeds, run 0, delta 1; CENSUS is the file's path), each quantized per row to E4M3 (scale 448 / max|row|, RNE; an
all-zero row is all-zero codes).  Per family, ROWS x COLS cells (X_q's row i against W's row j), each formed under SALTS salts
by #295's own `D3Family("d3s", "xq", "round11", "f16").side` and SHAKE256 stream (check_form.py's reference, on PR #295's
tree), every atom chained on verity's sm_120 atom (check.py's `chained`), the bound words every S-th accumulator.  Per
family and S:

- frozen atoms: Q_t = Q_(t-1); frozen segments: a bound word equal to the one before it (its S atoms left it unchanged),
  also per third of the chain (D-3s's three stacked k-blocks, `frozen_by_block`) and in the worst cell (`frozen_cell_max`)
- salt-independent words: a bound word equal under every salt; the same for its low 16 bits; constant bits: of a bound
  word's 32, how many are equal under every salt (the mean)
- repeated words: a cell under one salt whose bound words aren't pairwise distinct
- exact words: a bound word equal to the exact prefix sum (each product an integer multiple of 2^-18, summed in int64);
  exact cells: a cell whose bound words are all exact under some salt, so any exact method computes its chain (`exact_cells`,
  and `exact_cell_share` over cells and salts).  `credited` repeats the shares over the other cells only (the exact-chain
  debit), with `live_gap`: the share of their bound words that are frozen or salt-independent.

  uv run python benchmarks/pouw/ncp_sm120/liveness.py OUT [PR295]
      (env: CENSUS=/tmp/pearlc_census.py, K=8192, ROWS=6, COLS=6, SALTS=12, STRIDES="1 4 8", FAMILIES: all 26)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quantize(census, X: np.ndarray) -> np.ndarray:
    amax = np.abs(X).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, 448.0 / np.where(amax > 0, amax, 1.0), 0.0)
    return census.enc(X * scale)


def exact_prefix(a: np.ndarray, b: np.ndarray, atoms: int) -> np.ndarray:
    """Each output's exact sum after each atom, in units of 2^-18 (int64)."""
    mag = np.array([(c & 7) if c >> 3 == 0 else (8 + (c & 7)) << ((c >> 3) - 1) for c in range(128)], dtype=np.int64)
    mag[0x7F] = 0
    val = lambda x: np.where(x & 0x80, -mag[x & 0x7F], mag[x & 0x7F])   # units of 2^-9
    out = np.empty((a.shape[0], atoms), dtype=np.int64)
    acc = np.zeros(a.shape[0], dtype=np.int64)
    for t in range(atoms):
        acc = acc + (val(a[:, 32 * t:32 * t + 32].astype(np.int64)) * val(b[:, 32 * t:32 * t + 32].astype(np.int64))).sum(1)
        out[:, t] = acc
    return out


def shares(W: np.ndarray, E: np.ndarray, cells: np.ndarray) -> dict:
    """W: (salts, cells, bound) FP32 bit patterns; E: their exactness; over the cells selected."""
    W, E = W[:, cells], E[:, cells]
    if not W.shape[1]:
        return {"cells": 0}
    prev = np.concatenate([np.zeros_like(W[:, :, :1]), W[:, :, :-1]], axis=2)
    frozen = W == prev
    indep = (W == W[:1]).all(axis=0)
    const_bits = sum(((W >> np.uint32(b)) & 1 == (W[:1] >> np.uint32(b)) & 1).all(axis=0) for b in range(32))
    repeated = np.array([[len(np.unique(W[s, c])) < W.shape[2] for c in range(W.shape[1])] for s in range(W.shape[0])])
    gap = frozen | indep[None]
    return {"cells": int(W.shape[1]), "bound_words": int(W.shape[2]),
            "frozen_segments": round(float(frozen.mean()), 6),
            "frozen_by_block": [round(float(f.mean()), 6) for f in np.array_split(frozen, 3, axis=2)],
            "frozen_cell_max": round(float(frozen.mean(axis=(0, 2)).max()), 6),
            "salt_independent_words": round(float(indep.mean()), 6),
            "salt_independent_low16": round(float(((W & 0xFFFF) == (W[:1] & 0xFFFF)).all(axis=0).mean()), 6),
            "constant_bits_mean": round(float(const_bits.mean()), 3),
            "repeated_word_cells": round(float(repeated.mean()), 6),
            "exact_words": round(float(E.mean()), 6),
            "live_gap": round(float(gap.mean()), 6)}


def family_census(name: str, census, ck, cf, out: str, pr295: str, k: int, rows: int, cols: int, salts: list[bytes],
                  strides: list[int], context: dict) -> dict:
    XA, XB, _, _ = census.family_rows(name, k, 1.0, rows, cols, 0)
    qa, qb = quantize(census, XA), quantize(census, XB)
    jobs = [("shake", True, k, i, qa[i].tobytes(), s) for s in salts for i in range(rows)]
    jobs += [("shake", False, k, j, qb[j].tobytes(), s) for s in salts for j in range(cols)]
    got = cf.run_reference(out, pr295, context, jobs, f"liveness-{name}")
    formed = np.frombuffer(b"".join(g[0] for g in got), np.uint8).reshape(len(jobs), 3 * k)
    fa = formed[:len(salts) * rows].reshape(len(salts), rows, 3 * k)
    fb = formed[len(salts) * rows:].reshape(len(salts), cols, 3 * k)
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    A = fa[:, ii.ravel()].reshape(-1, 3 * k)
    B = fb[:, jj.ravel()].reshape(-1, 3 * k)
    T = 3 * k // 32
    Q = ck.chained(A, B, T)
    P = exact_prefix(A, B, T)
    exact = Q.view(np.float32).astype(np.float64) * 2.0 ** 18 == P.astype(np.float64)
    n = rows * cols
    Q, exact = Q.reshape(len(salts), n, T), exact.reshape(len(salts), n, T)
    res = {"family": name, "k": k, "cells": n, "salts": len(salts), "atoms": T,
           "zero_rows": [int((qa == 0).all(1).sum()), int((qb == 0).all(1).sum())],
           "frozen_atoms": round(float((Q[:, :, 1:] == Q[:, :, :-1]).mean()), 6), "by_S": {}}
    for S in strides:
        W, E = Q[:, :, S - 1::S], exact[:, :, S - 1::S]
        exact_cell = E.all(axis=2)
        res["by_S"][str(S)] = {"all": shares(W, E, np.arange(n)), "exact_cells": int(exact_cell.any(axis=0).sum()),
                               "exact_cell_share": round(float(exact_cell.mean()), 6),
                               "credited": shares(W, E, np.flatnonzero(~exact_cell.any(axis=0)))}
    return res


def main(argv: list[str]) -> int:
    out = argv[1]
    pr295 = argv[2] if len(argv) > 2 else os.environ.get("PR295", "/tmp/pr295")
    os.makedirs(os.path.join(out, "form"), exist_ok=True)
    path = os.environ.get("CENSUS", "/tmp/pearlc_census.py")
    sys.path.insert(0, os.path.join(HERE, ".."))   # pearlc_census imports run_result from benchmarks/pouw
    census = _load("pearlc_census", path)
    ck = _load("ncp_check", os.path.join(HERE, "check.py"))
    cf = _load("ncp_check_form", os.path.join(HERE, "check_form.py"))
    k, rows, cols = (int(os.environ.get(v, d)) for v, d in (("K", "8192"), ("ROWS", "6"), ("COLS", "6")))
    salts = [hashlib.sha256(b"ncp_sm120/liveness salt %d" % s).digest() for s in range(int(os.environ.get("SALTS", "12")))]
    strides = [int(s) for s in os.environ.get("STRIDES", "1 4 8").split()]
    fams = os.environ.get("FAMILIES", "").split() or [
        *census.FAMILIES, *census.ACTIVATION_FAMILIES, *census.OUTLIER_FAMILIES, *census.FREEZE_FAMILIES]
    context = {"salt": "", "index": 7, "root_a": hashlib.sha256(b"ncp_sm120/form root_a").hexdigest(), "weight": 3}
    with open(path, "rb") as f:
        result = {"census": {"path": path, "sha256": hashlib.sha256(f.read()).hexdigest()}, "k": k, "rows": rows,
                  "cols": cols, "salts": len(salts), "strides": strides, "families": []}
    for name in fams:
        r = family_census(name, census, ck, cf, out, pr295, k, rows, cols, salts, strides, context)
        result["families"].append(r)
        for S in strides:
            a, c = r["by_S"][str(S)]["all"], r["by_S"][str(S)]["credited"]
            print(f"{name:24s} S={S}: exact cells {r['by_S'][str(S)]['exact_cells']:2d}/{r['cells']}, exact words "
                  f"{a['exact_words']:.4f}, frozen seg {a['frozen_segments']:.4f}, salt-indep words "
                  f"{a['salt_independent_words']:.4f}, low16 {a['salt_independent_low16']:.4f}, const bits "
                  f"{a['constant_bits_mean']:5.2f}, repeated {a['repeated_word_cells']:.3f}; credited: "
                  f"{c['cells']} cells, gap {c.get('live_gap', float('nan')):.4f}, by block "
                  f"{c.get('frozen_by_block')}, worst cell {c.get('frozen_cell_max')}", flush=True)
        with open(os.path.join(out, "liveness.json"), "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
