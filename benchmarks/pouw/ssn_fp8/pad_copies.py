"""Copies a coin-reading prover finds in ssn-1l-tag's committed words when rows are padded (decode, m < 64).

    python pad_copies.py [--salts 16] [--n 48] [--k 2048] [--m 1,8,32] [--json OUT]

After step 1, ssn-1l-tag's noise is one coin bit a row a step, a_t[i], added to every column. A row whose data is zero
in step t (a padding row, or a real row whose 31-entry chunk is all zero) has word(i, ·, t) = word(i, ·, t-1) +
a_t[i]·1, so with probability 1/2 its row-step vector equals its previous one at the same salt: a copy, never a write.
Row tags are in step 1 and don't change that. This counts, over salts, the share of row-step vectors equal to any
earlier vector of the call at the same salt, with the padding rows zero (`zero`) or a public one-hot a step
(`onehot`: padding row r's lane (r mod 31) of every later step is 1, of step 1 lane (r mod 24)), whose later-step
data is a weight column, nonzero.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

TAG_LANES = 4
CODE_LANES = 4
FIRST = 32 - TAG_LANES - CODE_LANES
LATER = 31


def steps(k: int) -> int:
    return 1 + math.ceil(max(0, k - FIRST) / LATER)


def balanced13(z: int, digits: int) -> list[int]:
    out = []
    for _ in range(digits):
        r = ((z + 6) % 13) - 6
        out.append(r)
        z = (z - r) // 13
    return out


def tags(m: int) -> np.ndarray:
    return np.array([balanced13((i // 2 + 1) * (-1) ** i, TAG_LANES) for i in range(m)], dtype=np.int64)


def code(n: int) -> np.ndarray:
    c = np.arange(n) + 1
    return np.array([((c // 13**e) % 13) - 6 for e in range(CODE_LANES)], dtype=np.int64)


def chunks(k: int) -> list[slice]:
    s = steps(k)
    return [slice(0, min(k, FIRST))] + [slice(FIRST + LATER * (t - 1), min(k, FIRST + LATER * t)) for t in range(1, s)]


def words(a: np.ndarray, b: np.ndarray, g: np.ndarray, a1: np.ndarray, at: np.ndarray) -> np.ndarray:
    """(m, n, S) committed words of one call: the running data sum, −β_j, the coin code, the row tags against G, and
    each later step's coin bit."""
    m, k = a.shape
    n = b.shape[0]
    s = steps(k)
    cj = code(n)
    beta = np.rint((cj.sum(0) + s - 1) / 2).astype(np.int64)
    init = -beta[None, :] + a1 @ cj + tags(m) @ g
    out = np.empty((m, n, s), dtype=np.int64)
    acc = init.copy()
    for t, sl in enumerate(chunks(k)):
        acc = acc + a[:, sl] @ b[:, sl].T
        if t:
            acc = acc + at[:, t - 1 : t]
        out[:, :, t] = acc
    return out


def padded(rng: np.random.Generator, m: int, k: int, pad: str, rows: int = 64) -> np.ndarray:
    a = np.zeros((rows, k), dtype=np.int64)
    a[:m] = rng.integers(-127, 128, size=(m, k))
    if pad == "onehot":
        for r in range(m, rows):
            for t, sl in enumerate(chunks(k)):
                width = sl.stop - sl.start
                a[r, sl.start + r % (FIRST if t == 0 else LATER) % width] = 1
    return a


def copy_share(w: np.ndarray, m: int) -> dict:
    """Shares of row-step vectors equal to an earlier vector of the call (any row, any step) at this salt, and of the
    padding rows' later-step vectors equal to their own previous step's."""
    rows, _, s = w.shape
    seen, rep, own = set(), 0, 0
    for t in range(s):
        for i in range(rows):
            key = w[i, :, t].tobytes()
            rep += key in seen
            seen.add(key)
            if t and i >= m:
                own += bool(np.array_equal(w[i, :, t], w[i, :, t - 1]))
    pad_later = (rows - m) * (s - 1)
    return {"repeat": rep / (rows * s), "pad_own_prev": own / pad_later if pad_later else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salts", type=int, default=16)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--m", default="1,8,32")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(7)
    b = rng.integers(-127, 128, size=(args.n, args.k))
    g = rng.integers(-3, 4, size=(TAG_LANES, args.n))
    s = steps(args.k)
    rep = {"n": args.n, "k": args.k, "steps": s, "salts": args.salts, "rows": 64, "cases": []}
    for m in (int(x) for x in args.m.split(",")):
        for pad in ("zero", "onehot"):
            a = padded(np.random.default_rng(100 + m), m, args.k, pad)
            tot = {"repeat": 0.0, "pad_own_prev": 0.0}
            for _ in range(args.salts):
                a1 = rng.integers(0, 2, size=(64, CODE_LANES))
                at = rng.integers(0, 2, size=(64, s - 1))
                c = copy_share(words(a, b, g, a1, at), m)
                for key in tot:
                    tot[key] += c[key] / args.salts
            rep["cases"].append({"m": m, "pad": pad, **tot})
            print(f"m {m:2d} pad {pad:6s}: row-step vectors equal to an earlier one {tot['repeat']:.4f}; "
                  f"padding rows' later steps equal to their previous {tot['pad_own_prev']:.4f}")
    ok = all((c["pad"] == "zero") == (c["pad_own_prev"] > 0.4) for c in rep["cases"]) and all(
        c["repeat"] == 0.0 for c in rep["cases"] if c["pad"] == "onehot")
    rep["ok"] = ok
    print("ok" if ok else "FAIL: zero padding should copy about half its later steps, one-hot padding none")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
