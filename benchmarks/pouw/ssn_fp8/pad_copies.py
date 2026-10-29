"""Copies a coin-reading prover finds in the tagged one-lane words when a row's data chunks are zero (padding at decode,
m < 64, or a real row whose 31-entry chunks quantize to zero), under ssn_1l's later lane and under ssn-1l-tag's.

    python pad_copies.py [--salts 16] [--n 48] [--k 2048] [--m 1,8,32] [--json OUT]

After step 1, each later step adds one coin a row a step to every column: ssn_1l's a_t[i], or ssn-1l-tag's 1 + a_t[i]
(the lift: a public offset the centring subtracts). Under a_t[i], a row whose data is zero in step t has
word(i, ·, t) = word(i, ·, t-1) + a_t[i]·1, so with probability 1/2 its row-step vector equals its previous one at the
same salt: a copy, never a write. Row tags are in step 1 and don't change that. Under the lift a zero chunk moves the
vector by 1 or 2 and never copies it. Data chosen to cancel the lift (a lane of -1 against a weight row of 1) copies
again; that isn't modelled here. This counts, over salts, the share of row-step vectors equal to any earlier vector of
the call at the same salt, for each case:

- `zero`: the padding rows are zero, under a_t.
- `onehot`: padding row r's lane (r mod 31) of every later step is 1, of step 1 lane (r mod 24), under a_t; its
  later-step data is a weight column, nonzero.
- `zero-lift`: zero padding under the lift.
- `sparse` and `sparse-lift`: no padding (m = 64 real rows), each real row's later chunks zero with probability 0.98,
  as the stock int8 recipe's sink-token rows on the down projection, under a_t and under the lift.
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
ROWS = 64
SPARSE = 0.98
CASES = {"zero": (False, 0), "onehot": (False, 0), "zero-lift": (False, 1), "sparse": (True, 0), "sparse-lift": (True, 1)}


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


def words(a: np.ndarray, b: np.ndarray, g: np.ndarray, a1: np.ndarray, at: np.ndarray, lift: int = 0) -> np.ndarray:
    """(m, n, S) committed words of one call: the running data sum, −β_j, the coin code, the row tags against G, and
    each later step's coin lane, lift + a_t[i]."""
    m, k = a.shape
    n = b.shape[0]
    s = steps(k)
    cj = code(n)
    beta = np.rint((cj.sum(0) + (2 * lift + 1) * (s - 1)) / 2).astype(np.int64)
    init = -beta[None, :] + a1 @ cj + tags(m) @ g
    out = np.empty((m, n, s), dtype=np.int64)
    acc = init.copy()
    for t, sl in enumerate(chunks(k)):
        acc = acc + a[:, sl] @ b[:, sl].T
        if t:
            acc = acc + lift + at[:, t - 1 : t]
        out[:, :, t] = acc
    return out


def padded(rng: np.random.Generator, m: int, k: int, case: str) -> np.ndarray:
    if case.startswith("sparse"):
        a = rng.integers(-127, 128, size=(ROWS, k))
        for t, sl in enumerate(chunks(k)):
            if t:
                a[rng.random(ROWS) < SPARSE, sl] = 0
        return a
    a = np.zeros((ROWS, k), dtype=np.int64)
    a[:m] = rng.integers(-127, 128, size=(m, k))
    if case == "onehot":
        for r in range(m, ROWS):
            for t, sl in enumerate(chunks(k)):
                width = sl.stop - sl.start
                a[r, sl.start + r % (FIRST if t == 0 else LATER) % width] = 1
    return a


def copy_share(w: np.ndarray, first: int) -> dict:
    """Shares of row-step vectors equal to an earlier vector of the call (any row, any step) at this salt, and of rows
    `first`.. later-step vectors equal to their own previous step's."""
    rows, _, s = w.shape
    seen, rep, own = set(), 0, 0
    for t in range(s):
        for i in range(rows):
            key = w[i, :, t].tobytes()
            rep += key in seen
            seen.add(key)
            if t and i >= first:
                own += bool(np.array_equal(w[i, :, t], w[i, :, t - 1]))
    later = (rows - first) * (s - 1)
    return {"repeat": rep / (rows * s), "own_prev": own / later if later else 0.0}


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
    rep = {"n": args.n, "k": args.k, "steps": s, "salts": args.salts, "rows": ROWS, "sparse": SPARSE, "cases": []}
    runs = [(m, c) for m in (int(x) for x in args.m.split(",")) for c in CASES if not c.startswith("sparse")]
    runs += [(ROWS, "sparse"), (ROWS, "sparse-lift")]
    for m, case in runs:
        sparse, lift = CASES[case]
        a = padded(np.random.default_rng(100 + m), m, args.k, case)
        tot = {"repeat": 0.0, "own_prev": 0.0}
        for _ in range(args.salts):
            a1 = rng.integers(0, 2, size=(ROWS, CODE_LANES))
            at = rng.integers(0, 2, size=(ROWS, s - 1))
            c = copy_share(words(a, b, g, a1, at, lift), 0 if sparse else m)
            for key in tot:
                tot[key] += c[key] / args.salts
        rep["cases"].append({"m": m, "case": case, **tot})
        print(f"m {m:2d} {case:11s}: row-step vectors equal to an earlier one {tot['repeat']:.4f}; "
              f"{'rows' if sparse else 'padding rows'}' later steps equal to their previous {tot['own_prev']:.4f}")
    want_copies = {"zero": 0.4, "sparse": 0.4 * SPARSE}
    ok = all(c["own_prev"] > want_copies[c["case"]] if c["case"] in want_copies else c["repeat"] == 0.0
             for c in rep["cases"])
    rep["ok"] = ok
    print("ok" if ok else "FAIL: zero chunks should copy about half their steps, and one-hot padding and the lift none")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
