"""N-RED: attacks on ssn_1l (served sparse noise) by an adaptive prover, each built word for word and priced.

The prover reads a call's coins before it computes: the noise is an operand, so it has to. Its program may then branch
on them. `DistinctWritesMH100On` (H100W.lean) bounds one straight-line program on a grid of coins with the data fixed,
and SW-LEAN's `ssnTargetG` / `ssn1lBit_charged` instantiate it; neither says anything about a program chosen per draw.
On a branch that has read row i's S + 3 bits, every word of row i is a constant, which M0 counts free.

The attacks below use only what real hardware has: one pointer a row a step (an integer op, 32) and bulk copies
(TMA or cp.async.bulk: no register written, `Op100.move`). Rows whose step-t words are equal as whole vectors share one
computed vector:

- A1 identical rows (a batch of m identical requests; layer 0's repeated tokens): row i's step-t vector is fixed by
  (a₁[i], c_t[i] = Σ_{t'≤t} a_t'[i]), so a call needs one vector per occupied pair, not one per row;
- A1b shared prefixes: rows that agree up to their last k32 block, so pairwise distinct (a row dedupe passes them);
- A2 zero tails: a zero k32 data block with a_t = 0 repeats the row's previous vector;
- A3 identical columns (a prover-chosen model): within a row, word − row scalar = p·C[:, j] − β_j takes few values.
  This needs one copy a word by a public pattern (a per-word gather), so it is the model's per-word copy, not a bulk
  one;
- A4 the final word alone: even if every intermediate word were made distinct, the served output of identical rows
  carries only the served noise, and T3 keeps that small;
- A5 M0 plus device memory (`advice_table`): a replayed input's every possible vector, tabled ahead, salt-free;
- A6 padding (`padding`): decode's zero padding rows are identical rows.

The fix checked here: L = 4 row-tag lanes in step 1 carrying the balanced base-13 digits of the row index, T(i),
against G ∈ [−3, 3]^{4×n} drawn from coins after B is committed (`tag`), and later lanes 1 + a_t ∈ {1, 2}
(`lane1`). Rows then differ by T(i)·G, which no coin pattern or data difference can cancel unless G is known when B
is chosen (the `public_g` attack shows it can be then).

Every attack is built by `adversary`, which keys a row's step-t vector on (data prefix without trailing zeros, tag,
a₁, lane sum), computes one vector a key from those alone, and copies it; the check is that its words equal the honest
words of `honest` exactly. `oracle` counts the distinct row vectors (and 4-word tiles) of the honest words: no
copy-based adversary writes fewer. Prices: 32 a word written (an MMA k32 step's accumulator, or an add), 32 a row a
step for the pointer, 0 for a bulk copy. Savings are shares of the honest words' cost (α·stock) and of stock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns  # noqa: E402

TAG_LANES = 4
G_MAX = 3
TILE = 4
SHAPES = {"llama": ns.LLAMA, "v4_dense": ns.V4_DENSE}
MS = (64, 512, 8192)


def code(n: int) -> np.ndarray:
    return ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]


def balanced_digits(z: np.ndarray, lanes: int) -> np.ndarray:
    out, r = [], z.astype(np.int64).copy()
    for _ in range(lanes):
        d = ((r + 6) % 13) - 6
        out.append(d)
        r = (r - d) // 13
    assert (r == 0).all(), "too many rows for the tag"
    return np.stack(out, axis=-1)


def row_tags(m: int) -> np.ndarray:
    """T(i): balanced base-13 digits of z_i = (⌊i/2⌋ + 1)·(−1)^i, injective in i for m ≤ 28,560."""
    i = np.arange(m)
    z = np.where(i % 2 == 0, i // 2 + 1, -(i // 2 + 1))
    return balanced_digits(z, TAG_LANES)


def bias(c: np.ndarray, s: int, lane1: bool) -> np.ndarray:
    return np.rint((c.sum(axis=0) + (s - 1) * (3 if lane1 else 1)) / 2).astype(np.int64)


def data_k(s: int, tag: bool) -> int:
    return ns.DATA_PER_STEP - (TAG_LANES if tag else 0) + 31 * (s - 1)


def lane_sums(h: np.ndarray, s: int, lane1: bool) -> tuple[np.ndarray, np.ndarray]:
    a1, al = ns.ssn_1l_bits(h, s)
    lanes = al + (1 if lane1 else 0)
    return a1, np.concatenate([np.zeros((h.shape[0], 1), dtype=np.int64), np.cumsum(lanes, axis=1)], axis=1)


def honest(a, b, h, s, g=None, lane1=False) -> np.ndarray:
    """ssn_1l's committed words (m × n × S) through the served layout: X·Y by k32 steps from −β."""
    m, n = a.shape[0], b.shape[1]
    d1 = ns.DATA_PER_STEP - (0 if g is None else TAG_LANES)
    assert a.shape[1] == d1 + 31 * (s - 1)
    c = code(n)
    a1, al = ns.ssn_1l_bits(h, s)
    lanes = al + (1 if lane1 else 0)
    xp, yp = [a[:, :d1]], [b[:d1]]
    if g is not None:
        xp.append(row_tags(m))
        yp.append(g)
    xp.append(a1)
    yp.append(c)
    for t in range(1, s):
        lo = d1 + 31 * (t - 1)
        xp += [a[:, lo:lo + 31], lanes[:, t - 1:t]]
        yp += [b[lo:lo + 31], np.ones((1, n), dtype=np.int64)]
    x, y = np.concatenate(xp, axis=1), np.concatenate(yp, axis=0)
    assert x.shape[1] == 32 * s and np.abs(x).max() <= 127 and np.abs(y).max() <= 127
    init = -np.broadcast_to(bias(c, s, lane1), (1, m, n))
    return ns._words(x[None], y[None], init)[0]


def adversary(a, b, h, s, g=None, lane1=False, key_lanes=True) -> tuple[np.ndarray, int]:
    """The adaptive prover's words and its written-word count: one vector a key, copied to every (row, step) with it.

    `key_lanes=False` drops the lane sum from the key (a mutation the equality check must catch)."""
    m, n = a.shape[0], b.shape[1]
    d1 = ns.DATA_PER_STEP - (0 if g is None else TAG_LANES)
    c, beta = code(n), bias(code(n), s, lane1)
    tags = row_tags(m) if g is not None else np.zeros((m, 0), dtype=np.int64)
    a1, cs = lane_sums(h, s, lane1)
    out = np.empty((m, n, s), dtype=np.int64)
    table, writes = {}, 0
    for t in range(s):
        kt = d1 + 31 * t
        for i in range(m):
            pre = a[i, :kt]
            nz = np.flatnonzero(pre)
            pre = pre[:nz[-1] + 1] if nz.size else pre[:0]
            key = (pre.tobytes(), tags[i].tobytes(), a1[i].tobytes(), int(cs[i, t]) if key_lanes else 0)
            v = table.get(key)
            if v is None:
                v = pre @ b[:len(pre)] + a1[i] @ c + cs[i, t] - beta
                if g is not None:
                    v = v + tags[i] @ g
                table[key] = v
                writes += n
            out[i, :, t] = v
    return out, writes


def adversary_columns(a, b, h, s) -> tuple[np.ndarray, int]:
    """A3 on identical columns: a row's step-t words are one scalar plus the public f_p(j) = p·C[:, j] − β_j, so it
    writes one word per distinct f_p value and places them by p's public pattern."""
    m, n = a.shape[0], b.shape[1]
    assert (b == b[:, :1]).all()
    c, beta = code(n), bias(code(n), s, False)
    a1, cs = lane_sums(h, s, False)
    out = np.empty((m, n, s), dtype=np.int64)
    writes = 0
    for i in range(m):
        uniq, inv = np.unique(a1[i] @ c - beta, return_inverse=True)
        for t in range(s):
            kt = ns.DATA_PER_STEP + 31 * t
            scalar = a[i, :kt] @ b[:kt, 0] + cs[i, t]
            out[i, :, t] = (scalar + uniq)[inv]
            writes += len(uniq)
    return out, writes


def oracle(words: np.ndarray) -> dict:
    """Distinct row vectors over all (row, step), and distinct TILE-word tiles: no copy-based adversary does better."""
    m, n, s = words.shape
    vec = np.unique(words.transpose(0, 2, 1).reshape(m * s, n), axis=0).shape[0]
    q = n // TILE
    tiles = words[:, :q * TILE].transpose(0, 2, 1).reshape(m * s * q, TILE)
    til = np.unique(tiles, axis=0).shape[0]
    val = sum(len(np.unique(words[i, :, t])) for i in range(m) for t in range(s))
    return {"vectors": vec, "vector_words": vec * n, "tiles": til, "tile_words": til * TILE,
            "row_values": val, "words": m * n * s}


def saving(words: int, writes: int, pointers: int) -> float:
    return 1 - (writes + pointers) / words


def small_case(case: str, m: int, n: int, s: int, rng: np.random.Generator, g=None, lane1=False):
    tag = g is not None
    k = data_k(s, tag)
    row = np.clip(np.rint(rng.normal(0, 30, (1, k))), -127, 127).astype(np.int64)
    a = np.clip(np.rint(rng.normal(0, 30, (m, k))), -127, 127).astype(np.int64)
    b = np.clip(np.rint(rng.normal(0, 25, (k, n))), -127, 127).astype(np.int64)
    if case == "identical_rows":
        a = np.repeat(row, m, axis=0)
    elif case == "prefix_rows":
        a[:, :k - 31] = row[:, :k - 31]
    elif case == "zero_tails":
        a[:, ns.DATA_PER_STEP - (TAG_LANES if tag else 0):] = 0
    elif case == "identical_cols":
        b = np.repeat(b[:, :1], n, axis=1)
    elif case == "public_g":
        a = np.repeat(row, m, axis=0)
        a[:, :TAG_LANES] = -row_tags(m)
        b[:TAG_LANES] = g
    elif case != "natural":
        raise ValueError(case)
    h = rng.integers(0, 2 ** 32, (m, ns.ssn_1l_words(s)), dtype=np.int64)
    return a, b, h


def small_checks(seed: int = 7) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    m, n, s = 384, 32, 8
    res, detail = [], {}

    def ok(name, cond, d):
        res.append({"check": name, "pass": bool(cond), "detail": d})

    g = rng.integers(-G_MAX, G_MAX + 1, (TAG_LANES, n), dtype=np.int64)
    for variant, gg, lane1 in (("ssn_1l", None, False), ("fix", g, True)):
        for case in ("natural", "identical_rows", "prefix_rows", "zero_tails") + (("public_g",) if gg is not None else ()):
            a, b, h = small_case(case, m, n, s, rng, gg, lane1)
            w = honest(a, b, h, s, gg, lane1)
            adv, writes = adversary(a, b, h, s, gg, lane1)
            orc = oracle(w)
            r = {"words": w.size, "adversary_writes": writes, "pointers": m * s,
                 "saving": saving(w.size, writes, m * s), "oracle": orc,
                 "oracle_vector_saving": saving(w.size, orc["vector_words"], m * s),
                 "oracle_tile_saving": 1 - orc["tile_words"] / w.size,
                 "served_err_std": float((w[:, :, -1] - a @ b).std())}
            detail[f"{variant}/{case}"] = r
            ok(f"{variant} {case}: the adaptive prover's words equal the honest words", np.array_equal(adv, w),
               {"writes": writes, "words": w.size})
    d = detail
    ok("ssn_1l natural rows: no saving (every key is its own row)", d["ssn_1l/natural"]["adversary_writes"] == m * n * s,
       d["ssn_1l/natural"]["saving"])
    ok("A1 ssn_1l identical rows: the adaptive prover saves > 70% of the words at m 384, n 32, S 8",
       d["ssn_1l/identical_rows"]["saving"] > 0.7, d["ssn_1l/identical_rows"]["saving"])
    ok("A1b ssn_1l prefix rows (pairwise distinct): saves > 60%", d["ssn_1l/prefix_rows"]["saving"] > 0.6,
       d["ssn_1l/prefix_rows"]["saving"])
    ok("A2 ssn_1l zero tails (distinct rows): saves > 30%", d["ssn_1l/zero_tails"]["saving"] > 0.3,
       d["ssn_1l/zero_tails"]["saving"])
    for case in ("identical_rows", "prefix_rows", "zero_tails"):
        o = d[f"fix/{case}"]["oracle"]
        ok(f"fix {case}: every row vector distinct (oracle) and the adaptive prover saves nothing",
           o["vectors"] == m * s and d[f"fix/{case}"]["adversary_writes"] == m * n * s, o)
    ok("fix identical rows: 4-word tiles almost all distinct (< 2% repeat)",
       d["fix/identical_rows"]["oracle_tile_saving"] < 0.02, d["fix/identical_rows"]["oracle_tile_saving"])
    ok("public_g: a G known when B is chosen lets chosen data cancel the tags (oracle saves > 75%)",
       d["fix/public_g"]["oracle_vector_saving"] > 0.75, d["fix/public_g"]["oracle_vector_saving"])
    a, b, h = small_case("identical_rows", m, n, s, rng)
    adv, _ = adversary(a, b, h, s, key_lanes=False)
    ok("mutation: dropping the lane sum from the key breaks equality (the check can fail)",
       not np.array_equal(adv, honest(a, b, h, s)), "key without c_t")
    a, b, h = small_case("identical_cols", m, n, s, rng)
    w = honest(a, b, h, s)
    adv, writes = adversary_columns(a, b, h, s)
    detail["ssn_1l/identical_cols"] = {"words": w.size, "per_word_gather_writes": writes,
                                       "saving": saving(w.size, writes, m * s), "oracle": oracle(w)}
    ok("A3 identical columns: per-row value sharing reproduces the honest words", np.array_equal(adv, w), writes)
    a, b, h = small_case("natural", m, n, s, rng, g, True)
    detail["fix/served_err_std_natural"] = float((honest(a, b, h, s, g, True)[:, :, -1] - a @ b).std())
    return res, detail


def keys_identical(a1: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Occupied (a₁, c_t) pairs a step for identical rows: the vectors a call needs."""
    p = a1 @ (1 << np.arange(4))
    combo = p[:, None] * 100_000 + cs
    return np.array([len(np.unique(combo[:, t])) for t in range(cs.shape[1])])


def column_values(n: int, s: int) -> np.ndarray:
    c, beta = code(n), bias(code(n), s, False)
    pats = np.array([[(p >> e) & 1 for e in range(4)] for p in range(16)])
    return np.array([len(np.unique(pt @ c - beta)) for pt in pats])


def at_scale(seed: int = 11) -> dict:
    """Each attack's adversary writes at the priced shapes, from coin words drawn and read by the served extraction."""
    rng = np.random.default_rng(seed)
    out = {}
    for fam, shapes in SHAPES.items():
        for m in MS:
            acc = {"stock": 0.0, "honest_words": 0.0}
            per = {}
            for lin, (n, k) in shapes.items():
                s = ns.steps1l(k)
                h = rng.integers(0, 2 ** 32, (m, ns.ssn_1l_words(s)), dtype=np.int64)
                a1, cs = lane_sums(h, s, False)
                keys = keys_identical(a1, cs)
                words = m * n * s
                pts = m * s
                w_ident = n * keys.sum()
                w_prefix = n * (keys[:-1].sum() + m)
                w_tail = n * int((1 + cs[:, -1]).sum())
                uv = column_values(n, s)
                p = a1 @ (1 << np.arange(4))
                w_cols = s * int(uv[p].sum())
                final = (1 - keys[-1] / m) / s
                per[lin] = {"n": n, "K": k, "S": s, "alpha": ns.H32 * s / k, "words": words,
                            "keys_first_mid_last": [int(keys[0]), int(keys[s // 2]), int(keys[-1])],
                            "A1_identical_rows": saving(words, w_ident, pts),
                            "A1b_prefix_rows": saving(words, w_prefix, pts),
                            "A2_zero_tails": saving(words, w_tail, pts),
                            "A3_identical_cols_per_word": saving(words, w_cols, pts),
                            "A3_values_per_row_min_mean_max": [int(uv.min()), float(uv.mean()), int(uv.max())],
                            "A4_final_word_only": final}
                acc["stock"] += m * n * k
                acc["honest_words"] += words
                for key in ("A1_identical_rows", "A1b_prefix_rows", "A2_zero_tails", "A3_identical_cols_per_word",
                            "A4_final_word_only"):
                    acc[key] = acc.get(key, 0.0) + per[lin][key] * words * ns.H32
            tot = {key: acc[key] / acc["stock"] for key in acc if key.startswith("A")}
            out[f"{fam}_m{m}"] = {"per_linear": per, "gap_share_of_stock": tot,
                                  "alpha": ns.H32 * acc["honest_words"] / acc["stock"]}
    return out


def t3_value_collapse(seed: int = 5) -> dict:
    """A4 at the model's per-word copy: identical rows' final words carry only the served noise. Distinct values a
    column among m rows for rounded Gaussian noise of the served std, and the entropy bound p_coll ≥ 2^−H with
    H ≤ ½·log₂(2πe(σ² + 1/12))."""
    rng = np.random.default_rng(seed)
    out = {}
    for sigma in (6.0, 9.2, 30.0):
        hmax = 0.5 * math.log2(2 * math.pi * math.e * (sigma ** 2 + 1 / 12))
        row = {"H_max_bits": hmax, "p_coll_min": 2 ** -hmax}
        for m in MS:
            v = np.rint(rng.normal(0, sigma, (64, m)))
            d = np.mean([len(np.unique(x)) for x in v])
            row[f"distinct_of_m{m}"] = float(d)
            row[f"collapse_m{m}"] = 1 - d / m
        out[f"sigma_{sigma}"] = row
    return out


def fix_noise(seed: int = 3) -> dict:
    """The fix's served noise std in accumulator units: the tag T(i)·G_j (G uniform in [−G_MAX, G_MAX]) beside
    ssn_1l's own (lane1 serves 1 + a_t, whose variance is a_t's)."""
    out = {}
    for m in MS:
        t = row_tags(m).astype(float)
        var_g = ((2 * G_MAX + 1) ** 2 - 1) / 12
        tag_var = float((t ** 2).sum(axis=1).mean() * var_g)
        for k in (2048, 8192):
            base = ns.ssn_noise_std(k)["ssn_1l_std_units"]
            out[f"m{m}_K{k}"] = {"tag_std": math.sqrt(tag_var), "ssn_1l_std": base,
                                 "total_std": math.sqrt(base ** 2 + tag_var)}
    return out


def fix_costs() -> dict:
    """The fix's price: S at Llama's K (the tag lanes take step 1's slack), and G's coin bytes a call or a window."""
    out = {}
    for lin, (n, k) in ns.LLAMA.items():
        s0, s1 = ns.steps1l(k), 1 + math.ceil((k - (ns.DATA_PER_STEP - TAG_LANES)) / 31)
        g_bytes = TAG_LANES * n * math.log2(2 * G_MAX + 1) / 8
        out[lin] = {"S": s0, "S_with_tag": s1,
                    "alpha_change": ns.H32 * (s1 - s0) / k,
                    "g_coin_share_per_call_uncredited": {f"m{m}": g_bytes * ns.COIN_PRICED_BYTE / (m * n * k) for m in MS},
                    "g_forming_share_per_call": {f"m{m}": ns.H32 * TAG_LANES * n / 4 / (m * n * k) for m in MS}}
    return out


def tiles_at_scale(seed: int = 19, n: int = 64, s: int = 8) -> dict:
    """The fix's tile repeats at m up to 8,192, words built directly (no X·Y): identical rows share D_t(j), identical
    columns share D_t(i). The share of (row, step, TILE-column tile) slots whose tile equals an earlier one."""
    rng = np.random.default_rng(seed)
    c = code(n)
    out = {}
    for m in MS:
        h = rng.integers(0, 2 ** 32, (m, ns.ssn_1l_words(s)), dtype=np.int64)
        g = rng.integers(-G_MAX, G_MAX + 1, (TAG_LANES, n), dtype=np.int64)
        for case in ("identical_rows", "identical_cols"):
            if case == "identical_rows":
                d = np.broadcast_to(rng.integers(-5000, 5000, (1, n, s)), (m, n, s))
            else:
                d = np.broadcast_to(rng.integers(-5000, 5000, (m, 1, s)), (m, n, s))
            for variant, lane1, tag in (("ssn_1l", False, False), ("fix", True, True)):
                a1, cs = lane_sums(h, s, lane1)
                w = d + (a1 @ c)[:, :, None] + cs[:, None, :] - bias(c, s, lane1)[None, :, None]
                if tag:
                    w = w + (row_tags(m) @ g)[:, :, None]
                tl = w.transpose(0, 2, 1).reshape(m * s * (n // TILE), TILE)
                out[f"{variant}/{case}/m{m}"] = 1 - np.unique(tl, axis=0).shape[0] / tl.shape[0]
    return out


HBM_BYTES = 141e9


def advice_table() -> dict:
    """A5, M0 plus device memory: for a call whose input repeats (a replayed batch), a salt-independent table of every
    (a₁, c_t) vector of every row and step, Σ_t 16·t = 8·S·(S + 1) vectors a row, turns each word into a bulk copy.
    The share of one forward's words such tables cover in one H200's HBM (cheapest per word first), and that share's
    saving of stock; free in M0, so the bound needs the atomic call and PeriodInventory."""
    out = {}
    for m in MS:
        lins = []
        for lin, (n, k) in ns.LLAMA.items():
            s = ns.steps1l(k)
            words = m * n * s * ns.LLAMA_LAYERS
            lins.append((8 * s * (s + 1) / s, words, n * k * m * ns.LLAMA_LAYERS))
        lins.sort()
        room, covered, saved = HBM_BYTES, 0.0, 0.0
        for per_word, words, stock in lins:
            take = min(words, room / (4 * per_word))
            covered += take
            saved += take * ns.H32
            room -= take * 4 * per_word
        tot_words = sum(w for _, w, _ in lins)
        tot_stock = sum(st for _, _, st in lins)
        out[f"m{m}"] = {"table_bytes_full": 4 * sum(p * w for p, w, _ in lins), "covered_word_share": covered / tot_words,
                        "gap_share_of_stock": saved / tot_stock}
    return out


def padding(seed: int = 13) -> dict:
    """A6, decode below 64 rows: the kernel's tile pads r real rows to 64 with zero rows, which are identical rows (A1).
    The padded words' saving as a share of the padded call's honest words, and of the r real rows' stock."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in (1, 8, 32):
        acc_words = acc_saved = acc_stock = 0.0
        for n, k in ns.LLAMA.values():
            s = ns.steps1l(k)
            pad = 64 - r
            h = rng.integers(0, 2 ** 32, (pad, ns.ssn_1l_words(s)), dtype=np.int64)
            a1, cs = lane_sums(h, s, False)
            keys = keys_identical(a1, cs)
            acc_saved += ns.H32 * (pad * n * s - n * keys.sum() - pad * s)
            acc_words += 64 * n * s
            acc_stock += r * n * k
        out[f"r{r}"] = {"share_of_padded_call_words": acc_saved / (ns.H32 * acc_words),
                        "share_of_real_rows_stock": acc_saved / acc_stock}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the report here")
    args = ap.parse_args(argv)
    checks, small = small_checks()
    scale = at_scale()
    worst = scale["llama_m8192"]
    checks.append({"check": "A1 at Llama m 8,192: gap > 1% of stock (T1 fails on identical rows)",
                   "pass": worst["gap_share_of_stock"]["A1_identical_rows"] > 0.01,
                   "detail": worst["gap_share_of_stock"]})
    tiles = tiles_at_scale()
    checks.append({"check": "fix: < 1% of 4-word tiles repeat at m up to 8,192 (identical rows and identical columns)",
                   "pass": max(v for k, v in tiles.items() if k.startswith("fix/")) < 0.01, "detail": tiles})
    checks.append({"check": "ssn_1l: > 40% of 4-word tiles repeat on identical rows or columns (tile-level A1 and A3)",
                   "pass": min(v for k, v in tiles.items() if k.startswith("ssn_1l/")) > 0.4, "detail": tiles})
    rep = {"checks": checks, "small": small, "at_scale": scale, "t3_value_collapse": t3_value_collapse(),
           "fix_noise": fix_noise(), "fix_costs": fix_costs(), "advice_table": advice_table(), "padding": padding(),
           "tiles_at_scale": tiles,
           "periods_A1_llama_m8192": ns.periods(worst["gap_share_of_stock"]["A1_identical_rows"], worst["alpha"]),
           "params": {"tag_lanes": TAG_LANES, "g_max": G_MAX, "tile_words": TILE, "ms": MS, "H32": ns.H32}}
    text = json.dumps(rep, indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']}")
    for key in ("llama_m64", "llama_m512", "llama_m8192", "v4_dense_m8192"):
        g = scale[key]["gap_share_of_stock"]
        print(key, " ".join(f"{k} {100 * v:.2f}%" for k, v in g.items()))
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
