"""What a prover that branches on the coins saves on the lead's committed words, built word for word.

The committed word is word(i, j, t) = −β_j + Σ_{τ ≤ t} (data dot of k32 step τ + coin term of step τ). Designs:

- `tag` (`ssn-1l-tag`): step 1 holds 24 data lanes, T(i)·G[:, j] (G ∈ [−3, 3]^{4×n} drawn after B) and a₁·C[:, j]
  (a₁ ∈ {0, 1}⁴); each later step holds 31 data lanes and (1 + b_t[i])·1.
- `hw` (`ssn-1l-hw`): step 1 holds 28 data lanes and a₁·C[:, j] (a₁ ∈ {0..15}⁴); each later step holds 31 data lanes
  and (1 + 2b_t[i])·σ_t[j], σ_t ∈ {±1}ⁿ public.
- `fix2`: `tag` with each later step's (1 + b_t[i])·w_t[j], w_t ∈ {±1}ⁿ drawn after B.
- `blc` (`ssn-1l-tag + blc4a-t64-e7`): `tag` with 28 data lanes a later step and three coin lanes, A′ = 1 against
  B′ = s_t[tile, ℓ, j], odd and uniform in ±127 for each tile of 64 rows and column, drawn after A.
- `rot` (`blc4a-t64-e7-rot`): `blc` with A′ = (127, 1, 1) on even rows and (1, 127, 1) on odd rows. `lane_bounds`
  gives either design's exact chance of a kept word and tile, and `kept_sweep` counts them.

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
word write costs one, moves free) and W (a whole register of consecutive columns costs its lanes). `masked_rows` is a
cross-row prover under free predicates, which that check doesn't see. `class_shared` bounds what a class of outcomes
(the coins a program's uniform branches pin, `CLASSES`) shares across entries, `Priced.shared`'s residual: the
(entry, value) pairs another entry takes at every outcome drawn in the class. `at_scale` runs the walk at Llama's S.
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
#: The coin-lane designs, and `rot`'s row weights A′ on the three lanes, by row parity.
BLC = ("blc", "rot")
ROT = np.array([[127, 1, 1], [1, 127, 1]])
STEPS = {"K 2,048": 67, "K 8,192": 265}


def lane_weights(design: str, m: int) -> np.ndarray:
    """A′, each row's weights on its tile's three coin lanes: 1 for `blc`, `ROT` by row parity for `rot`."""
    return ROT[np.arange(m) % 2] if design == "rot" else np.ones((m, BLC_LANES), dtype=np.int64)


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
    if design in ("tag", "fix2", *BLC):
        co["G"] = rng.integers(-3, 4, size=(TAG_LANES, n))
    if design == "fix2":
        co["w"] = rng.choice(np.array([-1, 1]), size=(s, n))
    if design in BLC:
        tiles = -(-m // BLC_TILE)
        co["s"] = 2 * rng.integers(0, 2 ** BLC_E, size=(s, tiles, BLC_LANES, n)) - (2 ** BLC_E - 1)
    return co


def first_data(design: str) -> int:
    return 28 if design == "hw" else 24


def later_data(design: str) -> int:
    return 32 - 1 - BLC_LANES if design in BLC else 31


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
    if design in ("tag", "fix2", *BLC):
        xp.append(row_tags(m))
        yp.append(co["G"])
    xp.append(co["a1"])
    yp.append(pub["C"])
    lanes, wts = step_lanes(design, co), step_weights(design, pub, co, n, s)
    for t in range(1, s):
        lo = d1 + dl * (t - 1)
        xp.append(a[:, lo:lo + dl])
        yp.append(b[lo:lo + dl])
        if design in BLC:
            xp.append(lane_weights(design, m))
            yp.append(np.zeros((BLC_LANES, n), dtype=np.int64))
        xp.append(lanes[:, t:t + 1])
        yp.append(wts[t:t + 1])
    x, y = np.concatenate(xp, axis=1), np.concatenate(yp, axis=0)
    assert x.shape[1] == 32 * s and np.abs(x).max() <= 127 and np.abs(y).max() <= 127
    steps = np.einsum("mtk,tkn->mtn", x.reshape(m, s, 32), y.reshape(s, 32, n))
    if design in BLC:
        steps[:, 1:] += np.einsum("ml,tmln->mtn", lane_weights(design, m), co["s"][1:, np.arange(m) // BLC_TILE])
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

    With `levels` (m × S, the row's coin level and tile), sources are earlier rows at the same step and key, and the
    mask is where the two rows' step-1 words, which the prover holds, compare equal: predicates alone. The words it
    copies equal the honest ones when that mask is the words' own (`mask_mismatch` counts where it isn't)."""
    m, s, n = words.shape
    vecs = words.transpose(1, 0, 2).reshape(s * m, n)
    flat = None if levels is None else levels.T.reshape(s * m)
    paid, mismatch = n, 0
    for r in range(1, s * m):
        cand = np.arange(r) if flat is None else np.arange(r - r % m, r)[flat[r - r % m:r] == flat[r]]
        if cand.size == 0:
            paid += n
            continue
        hit = vecs[cand] == vecs[r]
        if flat is not None:
            held = words[cand % m, 0] == words[r % m, 0]
            mismatch += int((held != hit).sum())
            hit = held
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
    if flat is None:
        return {f"masked_saving_{sources}": 1 - paid / words.size}
    return {f"masked_saving_{sources}_level": 1 - paid / words.size, "mask_mismatch": mismatch}


def row_levels(design: str, a, co: dict, s: int) -> np.ndarray:
    """Each row's key: its coin level W_t (the cumulative row-lane term and chosen lane, which every column shares) and,
    for `blc`, its tile, whose coin lanes it shares."""
    m = a.shape[0]
    d1, dl = first_data(design), later_data(design)
    lanes = step_lanes(design, co)
    inc = np.zeros((m, s), dtype=np.int64)
    for t in range(1, s):
        inc[:, t] = a[:, d1 + dl * (t - 1)] + lanes[:, t]
    tiles = -(-m // BLC_TILE) if design in BLC else 1
    group = np.arange(m) // BLC_TILE % tiles
    if design == "rot":
        tiles, group = 2 * tiles, 2 * group + np.arange(m) % 2
    return np.cumsum(inc, axis=1) * tiles + group[:, None]


#: MMA output tiles (rows, columns): a warp's 128 words at the finest wgmma, and a warpgroup's m64n8.
TILES = ((16, 8), (64, 8))
#: An off-SM copy's least box, 16 bytes of one row-step: its excess is reported, not held to zero, since (i) is stated at
#: tile grain and a box can repeat where a tile doesn't.
BOX = (1, 4)


def tile_excess(words: np.ndarray, rows: int, cols: int) -> float:
    """Model W at tile grain (`adaptive_lanes_le` with the tile's words as the lanes): what the per-group distinct
    tiles exceed the pooled ones by, times the tile's words, as a share of the words. Zero means no two groups share a
    whole tile value, `adaptive_registers_le`'s hypothesis at tile grain."""
    m, s, n = words.shape
    t = words.reshape(m // rows, rows, s, n // cols, cols).transpose(0, 3, 2, 1, 4).reshape(-1, s, rows * cols)
    per_group = sum(len(np.unique(g, axis=0)) for g in t)
    pooled = len(np.unique(t.reshape(-1, rows * cols), axis=0))
    return rows * cols * (per_group - pooled) / words.size


def kept(words: np.ndarray, rows: int = 1, cols: int = 1) -> float:
    """The share of a tile's step transitions (at 1 × 1, a word's) that leave the whole tile as it was, which a prover
    need not write."""
    m, s, n = words.shape
    t = words.reshape(m // rows, rows, s, n // cols, cols).transpose(0, 3, 2, 1, 4).reshape(-1, s, rows * cols)
    return float((t[:, 1:] == t[:, :-1]).all(axis=2).mean())


def lane_bounds(design: str) -> dict:
    """Exact bounds from the coin law, for any data fixed before the coins. A word is kept only if its row's weighted
    lane sum (odd) hits the one target whose parity the row bit picks: at most half the law's largest point mass. A 16 × 8
    tile is kept only if one column's lanes, which pin all 16 row bits, are joined by 7 more columns' lanes hitting every
    row class's targets: at most 2⁻¹⁶ times the classes' joint largest point mass to the 7th."""
    v = 2 * np.arange(2 ** BLC_E) - (2 ** BLC_E - 1)
    lanes = np.stack(np.meshgrid(v, v, v, indexing="ij"), axis=-1).reshape(-1, BLC_LANES)
    sums = lanes @ np.unique(lane_weights(design, 2), axis=0).T
    word = max(np.unique(c, return_counts=True)[1].max() for c in sums.T) / len(lanes)
    joint = np.unique(sums, axis=0, return_counts=True)[1].max() / len(lanes)
    return {"design": design, "point_mass": float(word), "joint_point_mass": float(joint), "kept_word_bound": float(word / 2),
            "kept_word_zero_data": float((sums == -1).mean() / 2), "kept_tile_16x8_bound": float(2.0 ** -16 * joint ** 7)}


def kept_sweep(seed: int = 41, m: int = BLC_TILE, n: int = 2048, s: int = 265) -> list[dict]:
    """Kept words and tiles on zero data at one coin tile and Llama's width, where `rot`'s rarer lane hits still occur:
    one seed draws the same coins for `blc` and `rot`."""
    out = []
    for design in BLC:
        rng = np.random.default_rng(seed)
        pub = public(design, n, s, rng)
        a, b = data("zero", design, m, n, s, pub, rng)
        words = honest(design, a, b, pub, coins(design, m, n, s, rng), s)
        out.append({"design": design, "case": "zero", "m": m, "n": n, "S": s, "kept_word": kept(words),
                    "kept_tile_16x8": kept(words, 16, 8), "tile_excess_16x8": tile_excess(words, 16, 8)})
    return out


#: Coins a class pins, as a layout's warp-uniform view sees them: a warp over one row's columns sees the row coins, a
#: warp over one column's rows the column coins. `point` is one outcome alone, a class that pins every coin.
CLASSES = {"point": None, "rows": ("b", "a1"), "columns": ("G", "s"), "all_but_G": ("b", "a1", "s")}


def refine(key: np.ndarray, words: np.ndarray) -> np.ndarray:
    """Split the entry-steps' classes of equal values (`key`, entry-major) by one more outcome's words (m × S × n)."""
    return np.unique(np.stack([key, words.transpose(0, 2, 1).ravel()]), axis=1, return_inverse=True)[1].ravel()


def shared_count(key: np.ndarray, entry: np.ndarray) -> int:
    """The (entry, value) pairs whose value, as a function on the outcomes `key` has seen, another entry also takes at
    some step: an upper bound on what a class containing those outcomes shares across entries (`Priced.shared`), since
    agreement throughout the class implies agreement on them."""
    pairs = np.unique(np.stack([key, entry]), axis=1)
    return int((np.bincount(pairs[0])[pairs[0]] >= 2).sum())


def class_shared(design: str, case: str, m: int, n: int, s: int, seed: int, k: int = 8, k_max: int = 64) -> dict:
    """Each class's shared pairs, as a share of the words, beside check (b)'s credit at the first outcome. Outcomes are
    drawn in the class (the coins it pins kept, the rest afresh), at least k and then until no pair is left or k_max:
    a pair two entries hold equal at every draw is only an upper bound, which more draws can only shrink."""
    rng = np.random.default_rng(seed)
    pub = public(design, n, s, rng)
    a, b = data(case, design, m, n, s, pub, rng)
    co = coins(design, m, n, s, rng)
    words = honest(design, a, b, pub, co, s)
    out = {"design": design, "case": case, "m": m, "n": n, "S": s, **entry_credit(words)}
    out |= {f"tile_excess_{r}x{c}": tile_excess(words, r, c) for r, c in TILES if m % r == 0 and n % c == 0}
    out["box_excess"] = tile_excess(words, *BOX)
    out |= {"kept_word": kept(words), "kept_tile_16x8": kept(words, 16, 8)}
    entry = np.repeat(np.arange(m * n), s)
    first = refine(np.zeros(entry.size, dtype=np.int64), words)
    for name, keep in CLASSES.items():
        key, drawn, shared = first, 0, shared_count(first, entry)
        while keep is not None and (drawn < k or shared) and drawn < k_max:
            fresh = coins(design, m, n, s, rng)
            key = refine(key, honest(design, a, b, pub, {c: co[c] if c in keep else fresh[c] for c in co}, s))
            drawn += 1
            shared = shared_count(key, entry) if drawn >= k else shared
        out[f"shared_{name}"], out[f"draws_{name}"] = shared / words.size, drawn
    return out


def class_sweep(seed: int = 31, design: str = "blc") -> list[dict]:
    """A coin-lane design's class residuals: the four cases at the small shape and `same` at a tile and two at both of
    Llama's S. One seed draws the same data and coins for `blc` and `rot`, which differ only in A′."""
    shapes = [(design, case, 48, 64, 67) for case in ("zero", "same", "cancel", "walk")]
    shapes += [(design, "same", m, 64, s) for s in STEPS.values() for m in (BLC_TILE, 2 * BLC_TILE)]
    return [class_shared(*shape, seed=seed + i) for i, shape in enumerate(shapes)]


def level_sweep(seed: int = 21, n: int = 64) -> list[dict]:
    """`blc` with identical rows: the same-level masked copies at a whole tile and two, at both of Llama's S."""
    rng = np.random.default_rng(seed)
    out = []
    for s in STEPS.values():
        for m in (BLC_TILE, 2 * BLC_TILE):
            pub = public("blc", n, s, rng)
            a, b = data("same", "blc", m, n, s, pub, rng)
            co = coins("blc", m, n, s, rng)
            words = honest("blc", a, b, pub, co, s)
            out.append({"design": "blc", "case": "same", "m": m, "n": n, "S": s,
                        **masked_rows(words, 16, row_levels("blc", a, co, s)), **entry_credit(words)})
    return out


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
                checks += [{"name": f"{design}/{case} masked copies pay at least the pooled values (run_values_le)",
                            "ok": 1 - row["masked_saving_16"] >= row["model_p_pays"] - 1e-12},
                           {"name": f"{design}/{case} same-level masks from held step-1 words copy honest words",
                            "ok": row["mask_mismatch"] == 0}]
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
    res = {"small": rows, "checks": checks, "at_scale": at_scale(), "rows_sweep": rows_sweep(),
           "level_sweep": level_sweep(), "class_sweep": class_sweep(), "class_sweep_rot": class_sweep(design="rot"),
           "lane_bounds": [lane_bounds(d) for d in BLC], "kept_sweep": kept_sweep()}
    checks += [{"name": f"blc/same m {r['m']} S {r['S']} same-level masks from held step-1 words copy honest words",
                "ok": r["mask_mismatch"] == 0} for r in res["level_sweep"]]
    blc = res["lane_bounds"][0]
    checks.append({"name": "blc's lane law keeps a word with at most 3/1024 and a 16 × 8 tile with 2⁻¹⁶(3/512)⁷",
                   "ok": blc["kept_word_bound"] == 3 / 1024 and blc["kept_tile_16x8_bound"] == 2.0 ** -16 * (3 / 512) ** 7})
    checks.append({"name": "blc's kept words at one tile and n 2,048 are within 10% of the law's 3/1024",
                   "ok": abs(res["kept_sweep"][0]["kept_word"] / blc["kept_word_zero_data"] - 1) < 0.1})
    for r in res["class_sweep"] + res["class_sweep_rot"] + res["kept_sweep"]:
        tag = f"{r['design']}/{r['case']} m {r['m']} S {r['S']}"
        checks.append({"name": f"{tag} no 16 × 8 tile is kept from one step to the next", "ok": r["kept_tile_16x8"] == 0})
    checks += [{"name": f"{r['design']}/zero m {r['m']} n {r['n']} S {r['S']} no two groups share a 16 × 8 tile",
                "ok": r["tile_excess_16x8"] == 0} for r in res["kept_sweep"]]
    for r in res["class_sweep"] + res["class_sweep_rot"]:
        tag = f"{r['design']}/{r['case']} m {r['m']} S {r['S']}"
        checks += [{"name": f"{tag} a class pinning only {name} coins shares nothing (credit_le_cost_separated)",
                    "ok": r[f"shared_{name}"] == 0} for name in CLASSES if name != "point"]
        checks.append({"name": f"{tag} no two groups share a whole register (lanes_le_cost_separated)",
                       "ok": r["model_w_excess"] == 0})
        checks += [{"name": f"{tag} no two groups share a whole {key[12:]} tile (adaptive_registers_le, tile grain)",
                    "ok": v == 0} for key, v in r.items() if key.startswith("tile_excess_")]
    for r in rows + res["rows_sweep"] + res["level_sweep"] + res["class_sweep"] + res["class_sweep_rot"] + res["kept_sweep"]:
        print({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps(res["at_scale"], indent=1))
    print(json.dumps(res["lane_bounds"], indent=1))
    passed = sum(c["ok"] for c in checks)
    print(f"checks: {passed} of {len(checks)} pass")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
