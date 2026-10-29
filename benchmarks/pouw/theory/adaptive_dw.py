"""What a prover that branches on the coins saves on the lead's committed words, built word for word.

The committed word is word(i, j, t) = −β_j + Σ_{τ ≤ t} (data dot of k32 step τ + coin term of step τ). Designs:

- `tag` (`ssn-1l-tag`): step 1 holds 24 data lanes, T(i)·G[:, j] (G ∈ [−3, 3]^{4×n} drawn after B) and a₁·C[:, j]
  (a₁ ∈ {0, 1}⁴); each later step holds 31 data lanes and (1 + b_t[i])·1.
- `hw` (`ssn-1l-hw`): step 1 holds 28 data lanes and a₁·C[:, j] (a₁ ∈ {0..15}⁴); each later step holds 31 data lanes
  and (1 + 2b_t[i])·σ_t[j], σ_t ∈ {±1}ⁿ public.
- `fix2`: `tag` with each later step's (1 + b_t[i])·w_t[j], w_t ∈ {±1}ⁿ drawn after B.
- `blc` (`ssn-1l-tag + blc4a-t64-e7`): `tag` with 28 data lanes a later step and three coin lanes, A′ = 1 against
  B′ = s_t[tile, ℓ, j], odd and uniform in ±127 for each tile of 64 rows and column, drawn after A.

Chosen data is zero except one lane a later step: B's row there is a pattern p_t fixed before the coins, and every
row's A entry is x_t.

- `walk` (`tag`): p_t = 1, x_t = −1, −2, −1, … A row's step-t vector is its step-1 vector plus W_t·1, where W steps by
  x_t + 1 + b_t, in {0, 1} or {−1, 0}: a lazy walk, so a row holds about √(2S/π) distinct vectors instead of S.
- `cancel`: p_t = 1 (`tag`, `fix2`) or σ_t (`hw`), x_t = −1. The step's coin term is then b_t·1, 2b_t·σ_t or a word
  (1 + b_t)w_t[j] − 1, which is zero where b_t = 0 (and w_t[j] = +1 for `fix2`).
- `a2r` (`hw`, an alternating per-column cancel): A = 1 and B_t[j] = −σ_t[j](1 + 2e_t[j]), e_t[j] = ẽ_t where σ_t[j] = +1 and 1 − ẽ_t
  elsewhere, ẽ_t alternating. Each column walks lazily, on b_t or 1 − b_t as σ_t[j] says, so entries revisit their
  values but columns don't walk together.
- `zero`: zero data past step 1 (padding tails); `same`: that, with every row's step-1 data equal (identical rows).

The constructive adversary (`adversary`) keys row i's step t on its coin state (W_t for `tag`, the set of steps with
b = 1 for `hw`), computes one vector a key from data and coins alone, and copies it to every row-step with that key at
one pointer a row-step. Its words must equal `honest`'s, which multiplies the served lane layout out. `oracle` counts
distinct row vectors and 4-word tiles: no prover that copies whole row vectors (tiles), at one pointer a copy, writes
fewer. `patch` is a constructive tile prover (copy the nearest earlier tile at one word, rewrite the rest), whose
saving is open to a design whose words carry few coin bits each. `ball` is `adaptive_ball_table_le`'s lower bound at
tile grain (free copies, free decisions), which the free-copy patch cost must exceed; `rows_sweep` follows it as rows
grow. `entry_credit` sets a check that credits each entry's distinct values beside what two write models pay: P (a
word write costs one, moves free) and W (a whole register of consecutive columns costs its lanes). `at_scale` runs the
walk at Llama's S.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np

TILE = 4
TAG_LANES = 4
BLC_TILE, BLC_LANES, BLC_E = 64, 3, 7
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
    if design in ("tag", "fix2", "blc"):
        co["G"] = rng.integers(-3, 4, size=(TAG_LANES, n))
    if design == "fix2":
        co["w"] = rng.choice(np.array([-1, 1]), size=(s, n))
    if design == "blc":
        tiles = -(-m // BLC_TILE)
        co["s"] = 2 * rng.integers(0, 2 ** BLC_E, size=(s, tiles, BLC_LANES, n)) - (2 ** BLC_E - 1)
    return co


def first_data(design: str) -> int:
    return 28 if design == "hw" else 24


def later_data(design: str) -> int:
    return 32 - 1 - BLC_LANES if design == "blc" else 31


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
    d1, dl = first_data(design), later_data(design)
    k = d1 + dl * (s - 1)
    a = np.zeros((m, k), dtype=np.int64)
    b = np.zeros((k, n), dtype=np.int64)
    a[:, :d1] = rng.integers(-127, 128, size=(1 if case == "same" else m, d1))
    b[:d1] = rng.integers(-127, 128, size=(d1, n))
    if case in ("zero", "same"):
        return a, b
    for t in range(1, s):
        lane = d1 + dl * (t - 1)
        if case == "a2r":
            sig, e = pub["sigma"][t], t % 2
            b[lane] = -sig * (1 + 2 * np.where(sig == 1, e, 1 - e))
            a[:, lane] = 1
            continue
        b[lane] = pub["sigma"][t] if design == "hw" else 1
        a[:, lane] = -1 if case == "cancel" or t % 2 == 1 else -2
    return a, b


def bias(c: np.ndarray, s: int) -> np.ndarray:
    return np.rint((c.sum(axis=0) + 3 * (s - 1)) / 2).astype(np.int64)


def honest(design: str, a, b, pub: dict, co: dict, s: int) -> np.ndarray:
    """The committed words (m × S × n): the served layout X·Y by k32 steps, from −β."""
    m, n = a.shape[0], b.shape[1]
    d1, dl = first_data(design), later_data(design)
    xp, yp = [a[:, :d1]], [b[:d1]]
    if design in ("tag", "fix2", "blc"):
        xp.append(row_tags(m))
        yp.append(co["G"])
    xp.append(co["a1"])
    yp.append(pub["C"])
    lanes, wts = step_lanes(design, co), step_weights(design, pub, co, n, s)
    for t in range(1, s):
        lo = d1 + dl * (t - 1)
        xp.append(a[:, lo:lo + dl])
        yp.append(b[lo:lo + dl])
        if design == "blc":
            xp.append(np.ones((m, BLC_LANES), dtype=np.int64))
            yp.append(np.zeros((BLC_LANES, n), dtype=np.int64))
        xp.append(lanes[:, t:t + 1])
        yp.append(wts[t:t + 1])
    x, y = np.concatenate(xp, axis=1), np.concatenate(yp, axis=0)
    assert x.shape[1] == 32 * s and np.abs(x).max() <= 127 and np.abs(y).max() <= 127
    steps = np.einsum("mtk,tkn->mtn", x.reshape(m, s, 32), y.reshape(s, 32, n))
    if design == "blc":
        steps[:, 1:] += co["s"][1:, np.arange(m) // BLC_TILE].sum(axis=2).transpose(1, 0, 2)
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


def forest(words: np.ndarray, copy_price: int = 0) -> dict:
    """A free-decision tile prover that pools positions: it makes each distinct tile once, along a spanning forest of
    tiles sharing words (Kruskal over shared-word buckets, most shared first), copying its parent at `copy_price` and
    rewriting the other words; a component's first tile costs TILE, and every repeated instance one copy. Its cost
    bounds the free-decision optimum from above."""
    tiles = np.unique(words.reshape(-1, TILE), axis=0)
    n_t = len(tiles)
    parent = np.arange(n_t)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cost = TILE * n_t + copy_price * (words.size // TILE - n_t)
    for k in range(TILE - 1, 0, -1):
        save = TILE - min(TILE, copy_price + TILE - k)
        if save <= 0:
            continue
        for cols in itertools.combinations(range(TILE), k):
            _, inv = np.unique(tiles[:, list(cols)], axis=0, return_inverse=True)
            order = np.argsort(inv.reshape(-1), kind="stable")
            g = inv.reshape(-1)[order]
            starts = np.r_[True, g[1:] != g[:-1]]
            rep = order[np.maximum.accumulate(np.where(starts, np.arange(n_t), 0))]
            for x, y in zip(order[~starts].tolist(), rep[~starts].tolist()):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry
                    cost -= save
    return {f"forest_saving_copy_{copy_price}": 1 - cost / words.size}


def ball(words: np.ndarray) -> dict:
    """`adaptive_ball_table_le`'s count at tile grain, with free copies at any address and free decisions.

    Targets are the distinct 4-word tiles, pooled over positions. A tile sharing no 2 of its words with another
    target is at distance ≥ 3 from all of them and gets radius 1, any other radius 0, so a run writes at least
    Σ (r + 1) less 2 a memory slot. The share is that count over the honest words."""
    tiles = np.unique(words.reshape(-1, TILE), axis=0)
    near = np.zeros(len(tiles), dtype=bool)
    for p in range(TILE):
        for q in range(p + 1, TILE):
            _, inv, cnt = np.unique(tiles[:, [p, q]], axis=0, return_inverse=True, return_counts=True)
            near |= cnt[inv.reshape(-1)] > 1
    bound = 2 * len(tiles) - int(near.sum())
    return {"distinct_tiles_pooled": len(tiles), "radius_1_tiles": len(tiles) - int(near.sum()),
            "ball_share": bound / words.size}


def entry_credit(words: np.ndarray, lanes: int = 32) -> dict:
    """A per-entry check beside what each model's lower bound pays, with free decisions; shares of the honest words.

    `credit_b` counts each entry's distinct values over the steps. Model P (word copies free, `run_values_le`) pays
    the distinct word values pooled over the call. Model W (`adaptive_lanes_le`, a register of `lanes` consecutive
    columns of one row) pays `lanes` times the distinct registers pooled, which is short of `credit_b` by at most
    `lanes` a register value repeated across groups (`model_w_excess`)."""
    m, s, n = words.shape
    srt = np.sort(words, axis=1)
    credit_b = int((1 + (np.diff(srt, axis=1) != 0).sum(axis=1)).sum())
    regs = words.reshape(m, s, n // lanes, lanes).transpose(0, 2, 1, 3)
    per_group = sum(len(np.unique(regs[i, g], axis=0)) for i in range(m) for g in range(n // lanes))
    pooled_w = len(np.unique(regs.reshape(-1, lanes), axis=0))
    return {"credit_b": credit_b / words.size, "model_p_pays": len(np.unique(words)) / words.size,
            "model_w_pays": lanes * pooled_w / words.size,
            "model_w_excess": lanes * (per_group - pooled_w) / words.size}


def masked_rows(words: np.ndarray, sources: int, levels: np.ndarray | None = None) -> dict:
    """Masked cross-row copies: each row-step copies, column by column under a predicate, the words it shares with up
    to `sources` earlier row-steps (greedy), at one word a source for its address, and writes the rest. Masks and
    predicates are free, as when predicate-only writes are free; a share of the honest words saved.

    With `levels` (m × S, the row's coin level), sources are earlier rows at the same step and level, whose shared
    columns are where two per-row vectors the prover holds compare equal: a mask from predicates alone."""
    m, s, n = words.shape
    vecs = words.transpose(1, 0, 2).reshape(s * m, n)
    flat = None if levels is None else levels.T.reshape(s * m)
    paid = n
    for r in range(1, s * m):
        cand = np.arange(r) if flat is None else np.arange(r - r % m, r)[flat[r - r % m:r] == flat[r]]
        if cand.size == 0:
            paid += n
            continue
        hit = vecs[cand] == vecs[r]
        left = np.ones(n, dtype=bool)
        cost = 0
        for _ in range(sources):
            gain = (hit & left).sum(axis=1)
            best = int(gain.argmax())
            if gain[best] <= 1:
                break
            left &= ~hit[best]
            cost += 1
        paid += int(left.sum()) + cost
    return {f"masked_saving_{sources}{'' if levels is None else '_level'}": 1 - paid / words.size}


def row_levels(design: str, a, co: dict, s: int) -> np.ndarray:
    """Each row's coin level W_t: the cumulative row-lane term and chosen lane, which every column shares."""
    d1, dl = first_data(design), later_data(design)
    lanes = step_lanes(design, co)
    inc = np.zeros((a.shape[0], s), dtype=np.int64)
    for t in range(1, s):
        inc[:, t] = a[:, d1 + dl * (t - 1)] + lanes[:, t]
    return np.cumsum(inc, axis=1)


def rows_sweep(seed: int = 13, ms: tuple[int, ...] = (48, 512, 4096), n: int = 64, s: int = 67) -> list[dict]:
    """`fix2`'s ball share and pooled distinct tiles as one call's rows grow, for identical rows and zero tails."""
    rng = np.random.default_rng(seed)
    out = []
    for case in ("same", "zero"):
        for m in ms:
            pub = public("fix2", n, s, rng)
            a, b = data(case, "fix2", m, n, s, pub, rng)
            words = honest("fix2", a, b, pub, coins("fix2", m, n, s, rng), s)
            out.append({"design": "fix2", "case": case, "m": m, "n": n, "S": s,
                        "tile_instances": words.size // TILE, **ball(words)})
    return out


def small(seed: int = 7, m: int = 48, n: int = 64, s: int = 67) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    rows, checks = [], []
    for design, cases in (("tag", ("zero", "same", "cancel", "walk")), ("hw", ("zero", "same", "cancel", "a2r")),
                          ("fix2", ("zero", "same", "cancel", "walk")), ("blc", ("zero", "same", "cancel", "walk"))):
        for case in cases:
            pub = public(design, n, s, rng)
            a, b = data(case, design, m, n, s, pub, rng)
            co = coins(design, m, n, s, rng)
            words = honest(design, a, b, pub, co, s)
            free = {"patch_saving_free_copy": patch(words, copy_price=0)["patch_saving"]}
            row = {"design": design, "case": case, "m": m, "n": n, "S": s, **oracle(words), **patch(words), **free,
                   **forest(words, 0), **forest(words, 1), **ball(words), **entry_credit(words)}
            checks.append({"name": f"{design}/{case} ball bound ≤ free-copy patch and forest costs",
                           "ok": row["ball_share"] <= 1 - max(row["patch_saving_free_copy"],
                                                              row["forest_saving_copy_0"])})
            checks.append({"name": f"{design}/{case} check (b) ≤ model W's payment + excess (adaptive_lanes_le)",
                           "ok": row["credit_b"] <= row["model_w_pays"] + row["model_w_excess"] + 1e-12})
            if design in ("tag", "blc"):
                for k in (1, 4, 16):
                    row |= masked_rows(words, k)
                row |= masked_rows(words, 16, row_levels(design, a, co, s))
                checks.append({"name": f"{design}/{case} masked copies pay at least the pooled values (run_values_le)",
                               "ok": 1 - row["masked_saving_16"] >= row["model_p_pays"] - 1e-12})
            if design in ("tag", "hw") and case in ("cancel", "walk"):
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
    res = {"small": rows, "checks": checks, "at_scale": at_scale(), "rows_sweep": rows_sweep()}
    for r in rows + res["rows_sweep"]:
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
