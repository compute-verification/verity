"""HT-REDML: three numerical checks on how design A and B break (numpy, CPU, exact where it says so).

1. **B's ball.** For R uniform on the integers in [−127, 127], the relative Frobenius distance ρ = ‖W − R‖/‖R‖ of the
   cheap matrices a developer could pick inside ‖W − R‖ ≤ ρ‖R‖: R with its low k bits dropped, R magnitude-pruned, R
   2:4-pruned, R with whole input columns zeroed, R minus its top singular directions.
2. **A's adapters cancel columns of R.** With V the indicator of r input columns and U = −R[:, cols], M = R + U·Vᵀ has r
   zero columns, and the committed y = a·Rᵀ + (a·V)·Uᵀ equals a·Mᵀ exactly (int64), computed with (n − r)·m MACs a row.
3. **Committed output precision.** If only b bits of each output are committed (an int8 or bf16 output, or a learned
   output scale that coarsens it), a cheater computes a cheap approximation (a_hi·R_hiᵀ at 4×4 bits, or a·R_hiᵀ at
   8×4) and redoes exactly only the outputs whose ±z·σ error interval straddles a rounding boundary. Cost in bit-products
   = c + f, c the approximation's share and f the share redone.
"""

import argparse
import json

import numpy as np


def ball(n, seed):
    rng = np.random.default_rng(seed)
    R = rng.integers(-127, 128, (n, n)).astype(np.float64)
    nr = np.linalg.norm(R)
    out = {"drop_low_bits": {}, "prune_frac_for_rho": {}, "rho_for_prune_frac": {}}
    for k in range(1, 6):
        W = np.clip(np.round(R / 2 ** k) * 2 ** k, -128, 127)
        out["drop_low_bits"][k] = float(np.linalg.norm(W - R) / nr)
    a = np.sort(np.abs(R).ravel())
    cum = np.cumsum(a ** 2) / (a ** 2).sum()
    for f in (0.1, 0.25, 0.5):
        out["rho_for_prune_frac"][f] = float(np.sqrt(cum[int(f * a.size) - 1]))
    for rho in (0.01, 0.05, 0.1, 0.25):
        out["prune_frac_for_rho"][rho] = float(np.searchsorted(cum, rho ** 2) / a.size)
    g = np.abs(R).reshape(n, n // 4, 4)
    two_smallest = np.sort(g, -1)[..., :2]
    out["rho_2to4"] = float(np.sqrt((two_smallest ** 2).sum()) / nr)
    for f in (0.01, 0.1, 0.25):
        cols = int(f * n)
        out[f"rho_zero_{f:g}_of_columns"] = float(np.linalg.norm(R[:, :cols]) / nr)
    m = min(n, 1024)
    s = np.linalg.svd(R[:m, :m], compute_uv=False)
    tot = (s ** 2).sum()
    out["rho_cancel_top_rank_frac"] = {f: float(np.sqrt((s[:max(1, int(f * m))] ** 2).sum() / tot))
                                       for f in (0.01, 0.1, 0.25)}
    return out


def adapter_cancel(n, m, r, rows, seed):
    rng = np.random.default_rng(seed)
    R = rng.integers(-127, 128, (m, n)).astype(np.int64)
    a = rng.integers(-127, 128, (rows, n)).astype(np.int64)
    cols = rng.choice(n, r, replace=False)
    V = np.zeros((n, r), np.int64)
    V[cols, np.arange(r)] = 1
    U = -R[:, cols]
    y_committed = a @ R.T + (a @ V) @ U.T
    keep = np.setdiff1d(np.arange(n), cols)
    y_cheap = a[:, keep] @ R[:, keep].T
    honest = n * m + r * (n + m)
    return {"exact": bool((y_committed == y_cheap).all()), "zero_columns": int((R + U @ V.T == 0).all(0).sum()),
            "honest_macs_per_row": honest, "cheater_macs_per_row": (n - r) * m,
            "gap_of_credit": r * m / (n * m), "adapter_share_of_credit": r * (n + m) / (n * m)}


def split(v):
    hi = np.floor(v / 16)
    return hi, v - 16 * hi


def precision(K, rows, cols, seed, z, bits_list):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, K))
    q = np.round(x / np.abs(x).max(1, keepdims=True) * 126)
    a = q + rng.integers(-1, 2, q.shape)
    R = rng.integers(-127, 128, (cols, K)).astype(np.float64)
    y = a @ R.T
    a_hi, _ = split(a)
    R_hi, _ = split(R)
    approx = {"hi4x4 (c=1/4)": (256 * a_hi @ R_hi.T, 0.25), "a8xhi4 (c=1/2)": (16 * a @ R_hi.T, 0.5)}
    sy = y.std()
    out = {"K": K, "outputs": int(y.size), "z": z, "sigma_y": float(sy), "strategies": {}}
    for name, (yh, c) in approx.items():
        e = y - yh
        mu, se = e.mean(), e.std()
        rows_out = {}
        for b in bits_list:
            step = 8 * sy / 2 ** b
            lo = np.floor((yh + mu - z * se) / step + 0.5)
            hi = np.floor((yh + mu + z * se) / step + 0.5)
            undetermined = lo != hi
            truth = np.floor(y / step + 0.5)
            wrong = (~undetermined) & (lo != truth)
            f = float(undetermined.mean())
            rows_out[b] = {"redo_frac": f, "cost": c + f, "wrong_among_determined": int(wrong.sum())}
        out["strategies"][name] = {"err_sigma_over_sigma_y": float(se / sy), "by_bits": rows_out}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {"ball": ball(a.n, a.seed),
           "adapter_cancel": [adapter_cancel(n, m, r, 64, a.seed) for n, m, r in ((512, 512, 8), (1024, 4096, 16))],
           "precision": precision(a.K, 256, 1024, a.seed, 6.0, [1, 2, 3, 4, 6, 8, 12, 16])}
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
