"""N-ADAPT: what a prover that branches on the coins saves on the lead's committed words, built word for word.

The committed word is word(i, j, t) = −β_j + Σ_{τ ≤ t} (data dot of k32 step τ + coin term of step τ). Designs:

- `tag` (`ssn-1l-tag`): step 1 holds 24 data lanes, T(i)·G[:, j] (G ∈ [−3, 3]^{4×n} drawn after B) and a₁·C[:, j]
  (a₁ ∈ {0, 1}⁴); each later step holds 31 data lanes and (1 + b_t[i])·1.
- `hw` (`ssn-1l-hw`): step 1 holds 28 data lanes and a₁·C[:, j] (a₁ ∈ {0..15}⁴); each later step holds 31 data lanes
  and (1 + 2b_t[i])·σ_t[j], σ_t ∈ {±1}ⁿ public.
- `fix2`: `tag` with each later step's (1 + b_t[i])·w_t[j], w_t ∈ {±1}ⁿ drawn after B.

Chosen data is zero except one lane a later step: B's row there is a pattern p_t fixed before the coins, and every
row's A entry is x_t.

- `walk` (`tag`): p_t = 1, x_t = −1, −2, −1, … A row's step-t vector is its step-1 vector plus W_t·1, where W steps by
  x_t + 1 + b_t, in {0, 1} or {−1, 0}: a lazy walk, so a row holds about √(2S/π) distinct vectors instead of S.
- `cancel`: p_t = 1 (`tag`, `fix2`) or σ_t (`hw`), x_t = −1. The step's coin term is then b_t·1, 2b_t·σ_t or a word
  (1 + b_t)w_t[j] − 1, which is zero where b_t = 0 (and w_t[j] = +1 for `fix2`).
- `zero`: zero data past step 1 (padding tails); `same`: that, with every row's step-1 data equal (identical rows).

The constructive adversary (`adversary`) keys row i's step t on its coin state (W_t for `tag`, the set of steps with
b = 1 for `hw`), computes one vector a key from data and coins alone, and copies it to every row-step with that key at
one pointer a row-step. Its words must equal `honest`'s, which multiplies the served lane layout out. `oracle` counts
distinct row vectors and 4-word tiles: no prover that copies whole row vectors (tiles), at one pointer a copy, writes
fewer. `patch` is a constructive tile prover (copy the nearest earlier tile at one word, rewrite the rest), whose
saving is open to a design whose words carry few coin bits each. `at_scale` runs the walk at Llama's S.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

TILE = 4
TAG_LANES = 4
STEPS = {"K 2,048": 67, "K 8,192": 265}


def code(n: int) -> np.ndarray:
    """An injective column code C[e, j] ∈ [−6, 6]⁴: balanced base-13 digits of j + 1."""
    return balanced_digits(np.arange(1, n + 1), 4).T


def balanced_digits(z: np.ndarray, lanes: int) -> np.ndarray:
    out, r = [], z.astype(np.int64).copy()
    for _ in range(lanes):
        d = ((r + 6) % 13) - 6
        out.append(d)
        r = (r - d) // 13
    assert (r == 0).all()
    return np.stack(out, axis=-1)


def row_tags(m: int) -> np.ndarray:
    i = np.arange(m)
    return balanced_digits(np.where(i % 2 == 0, i // 2 + 1, -(i // 2 + 1)), TAG_LANES)


def public(design: str, n: int, s: int, rng: np.random.Generator) -> dict:
    """What is fixed before the data: the column code and, for `hw`, the signs."""
    pub = {"C": code(n)}
    if design == "hw":
        pub["sigma"] = rng.choice(np.array([-1, 1]), size=(s, n))
    return pub


def coins(design: str, m: int, n: int, s: int, rng: np.random.Generator) -> dict:
    """The coins, drawn after the data: a₁, the step bits, and G and w for the tagged designs."""
    co = {"b": rng.integers(0, 2, size=(m, s))}
    co["a1"] = rng.integers(0, 16 if design == "hw" else 2, size=(m, 4))
    if design in ("tag", "fix2"):
        co["G"] = rng.integers(-3, 4, size=(TAG_LANES, n))
    if design == "fix2":
        co["w"] = rng.choice(np.array([-1, 1]), size=(s, n))
    return co


def first_data(design: str) -> int:
    return 28 if design == "hw" else 24


def step_weights(design: str, pub: dict, co: dict, n: int, s: int) -> np.ndarray:
    """The coin lane's B row at each step t ≥ 2 (row 0 unused)."""
    if design == "hw":
        return pub["sigma"]
    if design == "fix2":
        return co["w"]
    return np.ones((s, n), dtype=np.int64)


def step_lanes(design: str, co: dict) -> np.ndarray:
    return 1 + 2 * co["b"] if design == "hw" else 1 + co["b"]


def data(case: str, design: str, m: int, n: int, s: int, pub: dict, rng: np.random.Generator):
    """A (m × data lanes) and B (data lanes × n), fixed before the coins."""
    d1 = first_data(design)
    k = d1 + 31 * (s - 1)
    a = np.zeros((m, k), dtype=np.int64)
    b = np.zeros((k, n), dtype=np.int64)
    a[:, :d1] = rng.integers(-127, 128, size=(1 if case == "same" else m, d1))
    b[:d1] = rng.integers(-127, 128, size=(d1, n))
    if case in ("zero", "same"):
        return a, b
    for t in range(1, s):
        lane = d1 + 31 * (t - 1)
        b[lane] = pub["sigma"][t] if design == "hw" else 1
        a[:, lane] = -1 if case == "cancel" or t % 2 == 1 else -2
    return a, b


def bias(c: np.ndarray, s: int) -> np.ndarray:
    return np.rint((c.sum(axis=0) + 3 * (s - 1)) / 2).astype(np.int64)


def honest(design: str, a, b, pub: dict, co: dict, s: int) -> np.ndarray:
    """The committed words (m × S × n): the served layout X·Y by k32 steps, from −β."""
    m, n = a.shape[0], b.shape[1]
    d1 = first_data(design)
    xp, yp = [a[:, :d1]], [b[:d1]]
    if design in ("tag", "fix2"):
        xp.append(row_tags(m))
        yp.append(co["G"])
    xp.append(co["a1"])
    yp.append(pub["C"])
    lanes, wts = step_lanes(design, co), step_weights(design, pub, co, n, s)
    for t in range(1, s):
        lo = d1 + 31 * (t - 1)
        xp += [a[:, lo:lo + 31], lanes[:, t:t + 1]]
        yp += [b[lo:lo + 31], wts[t:t + 1]]
    x, y = np.concatenate(xp, axis=1), np.concatenate(yp, axis=0)
    assert x.shape[1] == 32 * s and np.abs(x).max() <= 127 and np.abs(y).max() <= 127
    steps = np.einsum("mtk,tkn->mtn", x.reshape(m, s, 32), y.reshape(s, 32, n))
    return np.cumsum(steps, axis=1) - bias(pub["C"], s)[None, None, :]


def adversary(design: str, case: str, a, b, pub: dict, co: dict, s: int, key_bits: bool = True):
    """Copy-based prover: one vector a coin-state key, computed from data and coins; copies at one pointer each.

    Returns (words, writes, pointers). `key_bits=False` drops the step bits from the key (a mutation)."""
    m, n = a.shape[0], b.shape[1]
    d1 = first_data(design)
    first = a[:, :d1] @ b[:d1] + co["a1"] @ pub["C"] - bias(pub["C"], s)[None, :]
    if design in ("tag", "fix2"):
        first = first + row_tags(m) @ co["G"]
    out = np.empty((m, s, n), dtype=np.int64)
    writes = pointers = 0
    for i in range(m):
        seen: dict = {}
        level = 0
        ones: tuple = ()
        for t in range(s):
            if t > 0:
                lane = d1 + 31 * (t - 1)
                if design == "tag":
                    level += int(a[i, lane]) + 1 + int(co["b"][i, t]) * key_bits
                elif co["b"][i, t] and key_bits:
                    ones = ones + (t,)
            key = level if design == "tag" else ones
            if key in seen:
                out[i, t] = seen[key]
                pointers += 1
                continue
            if design == "tag":
                v = first[i] + level
            else:
                v = first[i] + sum((2 * pub["sigma"][u] for u in ones), np.zeros(n, dtype=np.int64))
            seen[key] = v
            out[i, t] = v
            writes += n
    return out, writes, pointers


def oracle(words: np.ndarray) -> dict:
    m, s, n = words.shape
    vecs = len(np.unique(words.reshape(m * s, n), axis=0))
    tiles = words.reshape(m * s * (n // TILE), TILE)
    distinct_tiles = len(np.unique(tiles, axis=0))
    return {
        "row_steps": m * s,
        "distinct_vectors": vecs,
        "vector_saving": (m * s - vecs) * (n - 1) / (m * s * n),
        "tiles": len(tiles),
        "distinct_tiles": distinct_tiles,
        "tile_saving": (len(tiles) - distinct_tiles) * (TILE - 1) / (m * s * n),
    }


def patch(words: np.ndarray, copy_price: int = 1) -> dict:
    """A tile prover that copies the nearest earlier tile at its column position and rewrites the words that differ.

    Tiles are made in (row, step) order; a tile costs min(TILE, copy_price + its Hamming distance to the nearest
    earlier tile at its position), in words. Cross-position and salt-free sources are left out, so the saving is a
    floor on what this prover gets."""
    m, s, n = words.shape
    tiles = words.reshape(m * s, n // TILE, TILE)
    cost = 0
    hist = np.zeros(TILE + 1, dtype=np.int64)
    for q in range(n // TILE):
        t = tiles[:, q, :]
        d = (t[:, None, :] != t[None, :, :]).sum(axis=-1)
        d[np.triu_indices(len(t))] = TILE
        near = d.min(axis=1)
        hist += np.bincount(near, minlength=TILE + 1)
        cost += np.minimum(TILE, copy_price + near).sum()
    return {"patch_saving": 1 - cost / words.size,
            "nearest_distance_shares": (hist / hist.sum()).round(4).tolist()}


def small(seed: int = 7, m: int = 48, n: int = 64, s: int = 67) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    rows, checks = [], []
    for design, cases in (("tag", ("zero", "same", "cancel", "walk")), ("hw", ("zero", "same", "cancel")),
                          ("fix2", ("zero", "same", "cancel", "walk"))):
        for case in cases:
            pub = public(design, n, s, rng)
            a, b = data(case, design, m, n, s, pub, rng)
            co = coins(design, m, n, s, rng)
            words = honest(design, a, b, pub, co, s)
            row = {"design": design, "case": case, "m": m, "n": n, "S": s, **oracle(words), **patch(words)}
            if design != "fix2" and case in ("cancel", "walk"):
                adv, writes, pointers = adversary(design, case, a, b, pub, co, s)
                ok = bool(np.array_equal(adv, words))
                mut, _, _ = adversary(design, case, a, b, pub, co, s, key_bits=False)
                caught = not np.array_equal(mut, words)
                row |= {"constructive_saving": 1 - (writes + pointers) / words.size, "equal": ok,
                        "mutation_caught": caught}
                checks += [{"name": f"{design}/{case} adversary equals honest", "ok": ok},
                           {"name": f"{design}/{case} mutation (bits dropped from the key) caught", "ok": caught}]
            rows.append(row)
    return rows, checks


def walk_distinct(s: int, rows: int, rng: np.random.Generator, pattern: str) -> float:
    """Mean distinct values of W_1..W_S (W_1 = 0), W stepping by x_t + 1 + b_t."""
    b = rng.integers(0, 2, size=(rows, s - 1))
    x = np.array([-1 if (pattern == "cancel" or t % 2 == 1) else -2 for t in range(1, s)])
    w = np.concatenate([np.zeros((rows, 1), dtype=np.int64), np.cumsum(x + 1 + b, axis=1)], axis=1)
    srt = np.sort(w, axis=1)
    return float((1 + (np.diff(srt, axis=1) != 0).sum(axis=1)).mean())


def at_scale(seed: int = 11, rows: int = 50_000, n: int = 2048) -> dict:
    """The walk and cancel savings at Llama's S (a share of the committed words, one pointer a copied row-step)."""
    rng = np.random.default_rng(seed)
    out = {}
    for label, s in STEPS.items():
        e = {"S": s}
        for pattern in ("cancel", "walk"):
            d = walk_distinct(s, rows, rng, pattern)
            e[pattern] = {"distinct_vectors_a_row": d, "saving": (s - d) * (n - 1) / (s * n)}
        e["walk_estimate_1_plus_sqrt_2S_over_pi"] = 1 + math.sqrt(2 * (s - 1) / math.pi)
        e["hw_cancel_saving"] = (s - 1) / 2 * (n - 1) / (s * n)
        out[label] = e
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", help="write the results here")
    args = p.parse_args(argv)
    rows, checks = small()
    res = {"small": rows, "checks": checks, "at_scale": at_scale()}
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps(res["at_scale"], indent=1))
    passed = sum(c["ok"] for c in checks)
    print(f"checks: {passed} of {len(checks)} pass")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
