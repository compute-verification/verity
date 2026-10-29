"""ssn_1l's committed words three ways, checked equal, and each program's price in MH100w (W1 units).

honest   A′·B′ in the served layout (step 1: 28 data lanes and a₁'s four coin bits against the public code C; step t ≥ 2:
         31 data lanes and a_t against a public 1), accumulated k32 step by k32 step from −β_j: one wgmma register a
         word (tc 32), plus forming (one LOP3 a row-step, one funnel shift a row-step not at bit 24).
free_d   the data's partial dots D_t are free (MH100w's F holds every salt-independent function of the committed
         input), so word = D_t − β_j + G[p_i, j] + P_t(i): one IADD3 a word, one data-dependent lookup G[p_i, j] a dot
         product (`other`), one prefix add a row-step.
chain    step 1 on the tensor core from the free D_1 − β_j (the honest's own step-1 lanes: one wgmma register a dot
         product, one LOP3 a row), then word_t = word_{t−1} + d_t + a_t[i]: one IADD3 a word, one bit extraction a
         row-step.

Prints JSON: the check on random and degenerate operands, and per Llama-3.2-1B linear and row count the words, each
program's W1 and the least adversary's W1 over the honest's. Exit 0 iff every check passes.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

TC = INT32 = OTHER = 32


def steps(k: int) -> int:
    return 1 + math.ceil((k - 28) / 31)


def code(n: int) -> np.ndarray:
    c = np.arange(1, n + 1)
    c = np.where(c < 14_280, c, c + 1)
    return np.stack([(c // 13**e) % 13 - 6 for e in range(4)])  # (4, n)


def layout(a: np.ndarray, b: np.ndarray, a1: np.ndarray, at: np.ndarray):
    """A′ (m, S, 32) and B′ (n, S, 32), int64, in ssn_1l's served layout."""
    m, k = a.shape
    n = b.shape[0]
    s = steps(k)
    ap = np.zeros((m, s, 32), np.int64)
    bp = np.zeros((n, s, 32), np.int64)
    ap[:, 0, :28] = a[:, :28]
    bp[:, 0, :28] = b[:, :28]
    ap[:, 0, 28:] = a1
    bp[:, 0, 28:] = code(n).T
    for t in range(1, s):
        lo, hi = 28 + 31 * (t - 1), min(k, 28 + 31 * t)
        ap[:, t, : hi - lo] = a[:, lo:hi]
        bp[:, t, : hi - lo] = b[:, lo:hi]
        ap[:, t, 31] = at[:, t]
        bp[:, t, 31] = 1
    return ap, bp


def beta(n: int, s: int) -> np.ndarray:
    return np.rint((code(n).sum(0) + s - 1) / 2).astype(np.int64)


def wrap(x: np.ndarray) -> np.ndarray:
    return ((x + 2**31) % 2**32 - 2**31).astype(np.int64)


def honest(a, b, a1, at):
    ap, bp = layout(a, b, a1, at)
    per = np.einsum("isk,jsk->ijs", ap, bp)
    return wrap(np.cumsum(per, axis=2) - beta(b.shape[0], ap.shape[1])[None, :, None])


def partial_dots(a, b):
    """D_t(i, j): the data's partial dot through step t (salt-independent: free in MH100w)."""
    k = a.shape[1]
    s = steps(k)
    ends = [min(k, 28 + 31 * t) for t in range(s)]
    full = np.cumsum(np.einsum("ik,jk->ijk", a.astype(np.int64), b.astype(np.int64)), axis=2)
    return full[:, :, [e - 1 for e in ends]]


def free_d(a, b, a1, at):
    n = b.shape[0]
    d = partial_dots(a, b)
    s = d.shape[2]
    pats = (a1 * (1 << np.arange(4))).sum(1)
    g = np.stack([(((p >> np.arange(4)) & 1)[:, None] * code(n)).sum(0) for p in range(16)])  # public, 16 x n
    prefix = np.cumsum(np.where(np.arange(s)[None, :] > 0, at, 0), axis=1)  # P_t(i), at[:, 0] unused
    return wrap(d - beta(n, s)[None, :, None] + g[pats][:, :, None] + prefix[:, None, :])


def chain(a, b, a1, at):
    n = b.shape[0]
    d = partial_dots(a, b)
    s = d.shape[2]
    w = np.empty_like(d)
    w[:, :, 0] = d[:, :, 0] - beta(n, s)[None, :] + a1 @ code(n)  # step 1: the honest's step-1 lanes on the tensor core
    for t in range(1, s):
        w[:, :, t] = w[:, :, t - 1] + (d[:, :, t] - d[:, :, t - 1]) + at[:, t][:, None]
    return wrap(w)


def price(m: int, n: int, k: int) -> dict:
    s = steps(k)
    words = m * n * s
    w = 1 + math.ceil(max(0, s - 29) / 32)
    hon = TC * words + INT32 * m * s + INT32 * m * (s - w)
    fd = INT32 * words + OTHER * m * n + INT32 * m * s
    ch = INT32 * m * n * (s - 1) + TC * m * n + INT32 * m + INT32 * m * s
    least = min(fd, ch)
    return {"m": m, "n": n, "k": k, "S": s, "words": words, "honest_w1": hon, "free_d_w1": fd, "chain_w1": ch,
            "least_adv_over_honest": least / hon, "gap_share_of_stock": (hon - least) / (m * n * k)}


LINEARS = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192)}


def main() -> int:
    rng = np.random.default_rng(20261029)
    out = {"checks": {}, "shapes": []}
    ok = True
    m, n = 6, 300
    for k in (2048, 8192, 60, 59):
        cases = {
            "random": (rng.integers(-127, 128, (m, k)), rng.integers(-127, 128, (n, k))),
            "zero": (np.zeros((m, k), np.int64), rng.integers(-127, 128, (n, k))),
            "massive": (np.full((m, k), 127), np.full((n, k), -127)),
            "identical_rows": (np.tile(rng.integers(-127, 128, (1, k)), (m, 1)), rng.integers(-127, 128, (n, k))),
        }
        for name, (a, b) in cases.items():
            s = steps(k)
            a1 = rng.integers(0, 2, (m, 4))
            at = rng.integers(0, 2, (m, s))
            h = honest(a, b, a1, at)
            good = bool(np.array_equal(h, free_d(a, b, a1, at)) and np.array_equal(h, chain(a, b, a1, at)))
            out["checks"][f"k{k}_{name}"] = good
            ok &= good
    for name, (nn, k) in LINEARS.items():
        for rows in (64, 512, 8192):
            out["shapes"].append({"linear": name, **price(rows, nn, k)})
    tot = {key: sum(r[key] for r in out["shapes"] if r["m"] == 8192) for key in ("words", "honest_w1", "free_d_w1",
                                                                                  "chain_w1")}
    out["llama_t8192"] = tot
    out["ok"] = ok
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
