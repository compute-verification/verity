"""SSN on the H200's float tensor cores: which committed words the adversary is charged for, and what serving loses.

``python -m benchmarks.pouw.ssn_fp8.study --out DIR [--K 1024 4096] [--salts 16]``.  For each format (E4M3 k32, E5M2
k32, BF16 k16), each design (`design.form`: stock, fixed, pin with the cs or the partial bound, as one block or in
proof blocks of 4 steps) and each data case, it runs every step of a small call bit for bit (`sim`) under ``--salts``
independent coin draws and reports, over all committed words (every step's accumulator word):

* ``free``: words equal under every salt (a salt-independent function of the input: free in M0);
* ``dup``: words whose function over the salts repeats another word's;
* ``pinned`` / ``affine`` (pin, fixed): every word in its row's binade, and word - u * noise the same under every salt
  (the int8 lemma's premise: exact words, affine in the coin bits);
* the served value's error against the exact product of the format's values and against the unquantized data, as a
  share of the output's rms, beside stock's: the T3 proxy (``err``).

The data cases: ``llama`` (heavy-tailed activations with outlier channels, Gaussian weights, one FP8 scale per tensor)
with rows and columns overwritten by the degenerate cases ssn_1l's check uses (zero row, identical rows, a massive row,
zero k-blocks, identical columns, a zero weight block) and three float ones (a large prefix then zeros, alternating
cancellation, uniformly random words).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import design as D
from . import sim

ROWS = ["zero_row", "ident_a", "ident_b", "massive_row", "zero_blocks", "big_then_zero", "cancel", "uniform"]


def data(dtype: str, m: int, n: int, K: int, rng: np.random.Generator):
    """(A words, B words, scale, exact float64 product) of one call: the llama-like base with the degenerate rows/cols."""
    x = rng.standard_t(4, (m, K))
    out = rng.choice(K, max(2, K // 512), replace=False)
    x[:, out] *= 30.0
    w = rng.standard_normal((n, K)) * 0.02 * np.exp(0.3 * rng.standard_normal((n, 1)))
    top = {"e4m3": 448.0, "e5m2": 57344.0, "bf16": None}[dtype]
    sx = np.abs(x).max() / top if top else 1.0
    sw = np.abs(w).max() / top if top else 1.0
    A = D.encode(dtype, x / sx)
    B = D.encode(dtype, w / sw)
    big = D.encode(dtype, np.array([np.abs(x / sx).max()]))[0]
    bigw = D.encode(dtype, np.array([np.abs(w / sw).max()]))[0]
    r = {name: i for i, name in enumerate(ROWS)}
    A[r["zero_row"]] = 0
    A[r["ident_b"]] = A[r["ident_a"]]
    A[r["massive_row"]] = 0
    A[r["massive_row"], out[0]] = big
    A[r["zero_blocks"], K // 4: 3 * K // 4] = 0
    A[r["big_then_zero"]] = 0
    A[r["big_then_zero"], :128] = big
    sign = np.uint16(0x8000) if dtype == "bf16" else np.uint8(0x80)
    A[r["cancel"]] = big
    A[r["cancel"], 1::2] = big ^ sign
    f = 16 if dtype == "bf16" else 8
    A[r["uniform"]] = rng.integers(0, 1 << f, K)
    if dtype == "bf16":  # every exponent of a 2^+-20 span (the full range overflows FP32 in a 1,024-term sum)
        A[r["uniform"]] = (rng.integers(0, 2, K) << 15) | (rng.integers(107, 148, K) << 7) | rng.integers(0, 128, K)
    B[1] = B[0]
    B[2, K // 2:] = 0
    B[3] = 0
    B[3, :128] = bigw
    # keep finite words only (uniform bits can draw NaN / inf codes)
    _, _, _, nonfinite = sim.products(dtype, A, np.zeros_like(A))
    A = np.where(nonfinite, 0, A).astype(A.dtype)
    xa, wb = D.decode(dtype, A) * sx, D.decode(dtype, B) * sw
    return A, B, sx * sw, xa @ wb.T, x @ w.T


def analyse(dtype: str, A, B, scale, exact_q, exact_true, mode: str, block, bound: str, salts: int, seed: int) -> dict:
    words, forms = [], []
    for s in range(salts):
        f = D.form(dtype, A, B, mode, np.random.default_rng([seed, s]), block=block, bound=bound)
        w, bad = D.run(dtype, f)
        if bad.any():
            raise RuntimeError(f"{mode}: {int(bad.sum())} outputs outside the finite model")
        words.append(w)
        forms.append(f)
    W = np.stack(words)  # (salts, m, n, S)
    salts_, m, n, S = W.shape
    flat = W.reshape(salts_, -1).T
    free_mask = (flat == flat[:, :1]).all(1)
    _, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    dup = int((counts[inv.ravel()] > 1).sum() - (counts > 1).sum())
    res = {"words": int(flat.shape[0]), "steps": S, "free": int(free_mask.sum()), "dup": dup}
    rows = {}
    fm = free_mask.reshape(m, n, S)
    for name in ROWS + ["llama"]:
        sel = [ROWS.index(name)] if name in ROWS else list(range(len(ROWS), m))
        rows[name] = int(fm[sel].sum())
    res["free_by_row"] = rows
    res["free_cols_0_3"] = [int(fm[len(ROWS):, c].sum()) for c in range(4)]
    if mode in ("pin", "fixed"):
        v = sim.fp32_value(W)
        N = np.stack([f.noise for f in forms]).astype(np.float64)
        u = forms[0].u[None, :, None, None]
        core = v - u * N
        res["affine_fail"] = int((~(core == core[:1]).all(0)).sum())
        if mode == "pin":
            p = forms[0].p
            e = ((W.astype(np.int64) >> 23) & 0xFF) - 127
            res["pin_fail"] = int((e != p[None, :, None, None]).sum())
            res["unpinnable_rows"] = [ROWS[i] if i < len(ROWS) else "llama" for i in np.nonzero(forms[0].unpinnable)[0]]
            pinned_rows = ~forms[0].unpinnable
            res["affine_fail_pinnable_rows"] = int((~(core == core[:1]).all(0))[pinned_rows].sum())
            res["free_pinnable_rows"] = int(fm[pinned_rows].sum())
            res["p_minus_log2_rms"] = float(np.median(p - np.log2(np.sqrt((exact_q / scale) ** 2).mean(1) + 1e-300)))
    # served values and the T3 proxy over the llama rows and the untouched columns, every salt
    sel = (slice(len(ROWS), None), slice(NCOLS, None))
    rms = np.sqrt((exact_true[sel] ** 2).mean())
    ys = np.stack([D.served(dtype, f, w) * scale for f, w in zip(forms, W)])
    noise = np.stack([D.served_noise(f) * scale for f in forms])
    res["err_vs_format_exact"] = float(np.sqrt(((ys - exact_q)[(slice(None),) + sel] ** 2).mean()) / rms)
    res["err_vs_true"] = float(np.sqrt(((ys - exact_true)[(slice(None),) + sel] ** 2).mean()) / rms)
    res["served_noise_rms"] = float(np.sqrt((noise[(slice(None),) + sel] ** 2).mean()) / rms)
    res["served_noise_mean"] = float(noise[(slice(None),) + sel].mean() / rms)
    return res


NCOLS = 4  # columns 0-3 hold the degenerate weight cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--salts", type=int, default=16)
    ap.add_argument("--formats", nargs="+", default=["e4m3", "e5m2", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    designs = [("stock", None, "-"), ("fixed", None, "-"), ("pin", None, "cs"), ("pin", None, "partial"),
               ("pin", 4, "cs"), ("pin", 4, "partial")]
    results = []
    for dtype in args.formats:
        for K in args.K:
            rng = np.random.default_rng([args.seed, K])
            A, B, scale, eq, et = data(dtype, args.m, args.n, K, rng)
            sel = (slice(len(ROWS), None), slice(NCOLS, None))
            rms = np.sqrt((et[sel] ** 2).mean())
            quant = float(np.sqrt(((eq - et)[sel] ** 2).mean()) / rms)
            for mode, block, bound in designs:
                t0 = time.time()
                r = analyse(dtype, A, B, scale, eq, et, mode, block, bound, args.salts, args.seed)
                r.update({"format": dtype, "K": K, "mode": mode, "block": block, "bound": bound, "quant_err": quant,
                          "seconds": round(time.time() - t0, 1)})
                results.append(r)
                print(json.dumps(r), flush=True)
    (out / "study.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))


if __name__ == "__main__":
    main()
