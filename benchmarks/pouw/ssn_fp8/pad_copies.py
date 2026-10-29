"""Copies a coin-reading prover finds in the tagged one-lane words when a row's data chunks are zero (padding at decode,
m < 64, or a real row whose later chunks quantize to zero), under ssn_1l's later lane, ssn-1l-tag's, and ssn-1l-tag's
with `blc4a`'s coin lanes.

    python pad_copies.py [--salts 16] [--n 48] [--k 2048] [--m 1,8,32] [--json OUT]

After step 1, each later step adds one coin a row a step to every column: ssn_1l's a_t[i], or ssn-1l-tag's 1 + a_t[i]
(the lift: a public offset the centring subtracts). Under a_t[i], a row whose data is zero in step t has
word(i, ·, t) = word(i, ·, t-1) + a_t[i]·1, so with probability 1/2 its row-step vector equals its previous one at the
same salt: a copy, never a write. Row tags are in step 1 and don't change that. Under the lift a zero chunk moves the
vector by 1 or 2 and never copies it. Data chosen to cancel the lift (a lane of -1 against a weight row of 1) copies
again; that isn't modelled here. `blc4a` gives up lanes 28-30 of each later step to three odd coins in [-127, 127]
against a constant 1, one a (tile, step, column), on every later step: a committed padding row makes the tile full.

Two counts, over salts:

- `repeat`: the share of row-step vectors equal to any earlier vector of the call (a whole-vector copy), and
  `own_prev`, the padding rows' later-step vectors equal to their own previous one;
- `word` and `word_tile`: the share of the padding rows' words (row, column, step) equal to a word the prover already
  holds, in the same column or anywhere in the tile, at an earlier step or an earlier row of this step. This is model
  P's copy, one word free. Zero padding's step-1 words are a1(i)·C_j + T(i)·G_j, small integers that repeat across
  rows, and its later words move by coins alone, so a prover knows them without a MAC. `word_real` is the real
  rows' in-column share: equal values it can't know before their MACs, so the scale of chance repeats.

Cases (`m` real rows of uniform int8 data, the rest padding, in a 64-row tile):

- `zero`: the padding rows are zero, under a_t.
- `onehot`: padding row r's lane (r mod 31) of every later step is 1, of step 1 lane (r mod 24), under a_t; its
  later-step data is a weight column, nonzero.
- `zero-lift`, `onehot-lift`: the same under the lift.
- `zero-blc4a`, `onehot-blc4a`: under the lift and `blc4a`'s lanes.
- `dense-lift`, `dense-blc4a`: padding rows hold a public uniform int8 pattern, fixed across calls, so a prover can
  keep their products and pays only the word writes, which the honest prover pays too.
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
BLC_LANES = 3
BLC_E = 7
ROWS = 64
SPARSE = 0.98
#: case: (padding pattern or "sparse", lift, blc4a's coin lanes)
CASES = {"zero": ("zero", 0, 0), "onehot": ("onehot", 0, 0), "zero-lift": ("zero", 1, 0),
         "onehot-lift": ("onehot", 1, 0), "dense-lift": ("dense", 1, 0), "zero-blc4a": ("zero", 1, BLC_LANES),
         "onehot-blc4a": ("onehot", 1, BLC_LANES), "dense-blc4a": ("dense", 1, BLC_LANES),
         "sparse": ("sparse", 0, 0), "sparse-lift": ("sparse", 1, 0)}
#: the in-column word copies dense padding stays under, and zero and one-hot padding over
WORD_DENSE = 0.01
WORD_SPARSE_PAD = 0.3


def later(coins: int) -> int:
    return LATER - coins


def steps(k: int, coins: int = 0) -> int:
    return 1 + math.ceil(max(0, k - FIRST) / later(coins))


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


def chunks(k: int, coins: int = 0) -> list[slice]:
    s, w = steps(k, coins), later(coins)
    return [slice(0, min(k, FIRST))] + [slice(FIRST + w * (t - 1), min(k, FIRST + w * t)) for t in range(1, s)]


def words(a: np.ndarray, b: np.ndarray, g: np.ndarray, a1: np.ndarray, at: np.ndarray, lift: int = 0,
          sc: np.ndarray | None = None) -> np.ndarray:
    """(m, n, S) committed words of one call: the running data sum, −β_j, the coin code, the row tags against G, each
    later step's row lane, lift + a_t[i], and `sc` (S - 1, n), the later steps' blc4a coin sums, if given."""
    m, k = a.shape
    n = b.shape[0]
    coins = 0 if sc is None else BLC_LANES
    s = steps(k, coins)
    cj = code(n)
    beta = np.rint((cj.sum(0) + (2 * lift + 1) * (s - 1)) / 2).astype(np.int64)
    init = -beta[None, :] + a1 @ cj + tags(m) @ g
    out = np.empty((m, n, s), dtype=np.int64)
    acc = init.copy()
    for t, sl in enumerate(chunks(k, coins)):
        acc = acc + a[:, sl] @ b[:, sl].T
        if t:
            acc = acc + lift + at[:, t - 1 : t]
            if sc is not None:
                acc = acc + sc[t - 1][None, :]
        out[:, :, t] = acc
    return out


def padded(rng: np.random.Generator, m: int, k: int, case: str, coins: int = 0) -> np.ndarray:
    pad = CASES[case][0]
    if pad == "sparse":
        a = rng.integers(-127, 128, size=(ROWS, k))
        for t, sl in enumerate(chunks(k, coins)):
            if t:
                a[rng.random(ROWS) < SPARSE, sl] = 0
        return a
    a = np.zeros((ROWS, k), dtype=np.int64)
    a[:m] = rng.integers(-127, 128, size=(m, k))
    if pad == "onehot":
        for r in range(m, ROWS):
            for t, sl in enumerate(chunks(k, coins)):
                width = sl.stop - sl.start
                a[r, sl.start + r % (FIRST if t == 0 else later(coins)) % width] = 1
    elif pad == "dense":
        a[m:] = np.random.default_rng(1).integers(-127, 128, size=(ROWS - m, k))
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
    later_steps = (rows - first) * (s - 1)
    return {"repeat": rep / (rows * s), "own_prev": own / later_steps if later_steps else 0.0}


def word_share(w: np.ndarray, first: int) -> dict:
    """Shares of rows `first`..'s words equal to a word held before them, in step-major, then row, order: in their
    column (`word`) or anywhere in the tile (`word_tile`); `word_real`, the same in-column share for the rows before
    `first`, whose values a prover can't know before their MACs."""
    rows, n, s = w.shape
    v = np.ascontiguousarray(w.transpose(2, 0, 1))
    pad = np.zeros((s, rows, n), dtype=bool)
    pad[:, first:, :] = True
    out = {}
    for name, keyed, sel in (("word", lambda x: x.transpose(2, 0, 1), pad), ("word_tile", lambda x: x, pad),
                             ("word_real", lambda x: x.transpose(2, 0, 1), ~pad)):
        vals, mask = keyed(v).reshape(-1), keyed(sel).reshape(-1)
        if name != "word_tile":
            vals = vals + np.repeat(np.arange(n, dtype=np.int64), s * rows) * (1 << 40)
        repeated = np.ones(vals.size, dtype=bool)
        repeated[np.unique(vals, return_index=True)[1]] = False
        out[name] = float(repeated[mask].sum() / mask.sum()) if mask.any() else 0.0
    return out


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
    rep = {"n": args.n, "k": args.k, "salts": args.salts, "rows": ROWS, "sparse": SPARSE, "cases": []}
    runs = [(m, c) for m in (int(x) for x in args.m.split(",")) for c in CASES if CASES[c][0] != "sparse"]
    runs += [(ROWS, "sparse"), (ROWS, "sparse-lift")]
    for m, case in runs:
        pad, lift, coins = CASES[case]
        s = steps(args.k, coins)
        first = 0 if pad == "sparse" else m
        a = padded(np.random.default_rng(100 + m), m, args.k, case, coins)
        tot = {"repeat": 0.0, "own_prev": 0.0, "word": 0.0, "word_tile": 0.0, "word_real": 0.0}
        for _ in range(args.salts):
            a1 = rng.integers(0, 2, size=(ROWS, CODE_LANES))
            at = rng.integers(0, 2, size=(ROWS, s - 1))
            sc = None
            if coins:
                odd = 2 * rng.integers(0, 2**BLC_E, size=(s - 1, coins, args.n)) - (2**BLC_E - 1)
                sc = odd.sum(1)
            w = words(a, b, g, a1, at, lift, sc)
            for key, val in {**copy_share(w, first), **word_share(w, first)}.items():
                tot[key] += val / args.salts
        rep["cases"].append({"m": m, "case": case, "steps": s, **tot})
        print(f"m {m:2d} {case:12s}: vectors equal to an earlier one {tot['repeat']:.4f}; "
              f"{'rows' if pad == 'sparse' else 'padding rows'}' later steps equal to their previous "
              f"{tot['own_prev']:.4f}; words held before, in the column {tot['word']:.5f}, in the tile "
              f"{tot['word_tile']:.5f}; real rows' in the column {tot['word_real']:.5f}")
    want_copies = {"zero": 0.4, "sparse": 0.4 * SPARSE}
    ok = True
    for c in rep["cases"]:
        if c["case"] in want_copies:
            ok &= c["own_prev"] > want_copies[c["case"]]
        else:
            ok &= c["repeat"] == 0.0
        if CASES[c["case"]][0] in ("zero", "onehot"):
            ok &= c["word"] > WORD_SPARSE_PAD
        if CASES[c["case"]][0] == "dense":
            ok &= c["word"] < WORD_DENSE
    rep["ok"] = ok
    print("ok" if ok else "FAIL: zero chunks should copy about half their steps, one-hot padding and the lift no "
          f"vector, zero and one-hot padding more than {WORD_SPARSE_PAD:.0%} of their words in the column, and dense "
          f"padding under {WORD_DENSE:.0%}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
