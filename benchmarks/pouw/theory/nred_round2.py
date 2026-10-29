"""N-RED round 2: attacks on the leads after interim 1, each built word for word or counted on the real messages.

Designs (the committed word is D_t(i, j) + N_t(i, j) − β_j, D_t the data's partial product through k32 step t):

- `ssn_1l` (SW-NOISE): N_1 = a₁·C_j, a₁ ∈ {0,1}⁴; each later step adds b_t·1.
- `fix` (N-RED interim 1, `ssn_1l-tag`): N_1 = T(i)·G_j + a₁·C_j; each later step adds (1 + b_t)·1.
- `hw` (N-PREM's `ssn_1l_hw`): N_1 = a₁·C_j, a₁ ∈ {0..15}⁴; each later step adds (1 + 2b_t)·σ_t[j], σ_t public signs.
- `fix2` (this file): `fix` with each later step adding (1 + b_t)·w_t[j], w_t ∈ {±1}ⁿ drawn from coins after B is
  committed (once a window).
- `rc_magic` (N-EXACT): N_1 = r_1(i) + c_j (a row bit and a fresh column bit); each later step adds r_t(i)·1. Both
  are removed in the epilogue, so the served output is exact.

Attacks:

- A2c, cancelled offsets (chosen B): a public, coin-independent part of a step's noise, e.g. `fix`'s 1·1 or `hw`'s
  σ_t, is cancelled by one data lane (A = 1, B = −pattern), which leaves b_t·(·), zero when b_t = 0: the row's step-t
  vector then equals its step t − 1 vector, a copy. `fix2`'s w_t is drawn after B, so no B carries it.
- A7, birthday at large m (identical rows): `hw` separates identical rows by 16 code bits and the step bits, so two
  rows coincide at step t when their a₁ and their bits through t agree; the share grows with m. `fix`/`fix2`'s tags are
  injective only up to 13^L − 1 rows with L lanes.
- A1/A2/A3 on `rc_magic`: identical rows coincide whenever their row-bit prefix sums agree (at most t + 1 keys a step),
  a zero tail repeats when r_t = 0, and identical columns leave two values a row-step.
- H2, A's row digests carry no position (frame_v3 PROTOCOL: "an x row and a W column with equal bits have one
  digest"): identical rows of A share every row-seg compression within one call, under L0, L1 and L2 alike, and no
  noise design touches A. Counted on the real messages by an instrumented SHA-512 that must match hashlib.
- A5 corrected (`a5_step_greedy`): advice tables each row's earliest steps of every linear, the best split of one
  H200's HBM by exhaustive search; interim 1 tabled whole linears. It needs each linear's input known before the
  noise; `nred_requant.py` measures how often the served noise changes it.

Every constructive adversary computes one vector a key from data and coins and copies it; its words must equal the
honest words. `na.oracle` counts the distinct row vectors (and tiles): no copy-based adversary writes fewer.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns  # noqa: E402
import nred_attacks as na  # noqa: E402

DESIGNS = ("ssn_1l", "fix", "hw", "fix2", "rc_magic")
BIG_MS = (8192, 32768, 131072, 1 << 20)
H2_BLOCK = 128


def first_lanes(design: str) -> int:
    return {"ssn_1l": 28, "hw": 28, "fix": 24, "fix2": 24, "rc_magic": 30}[design]


def rc_steps(k: int) -> int:
    return 1 + -(-(k - 30) // 31)


def steps(design: str, k: int) -> int:
    return rc_steps(k) if design == "rc_magic" else ns.steps1l(k)


def public(design: str, n: int, s: int, m: int, rng: np.random.Generator) -> dict:
    """The design's weights: C, G (window coins), σ (public signs), w (window coins), tags."""
    return {"c": na.code(n), "g": rng.integers(-na.G_MAX, na.G_MAX + 1, (na.TAG_LANES, n)),
            "sig": rng.choice([-1, 1], (s - 1, n)), "w": rng.choice([-1, 1], (s - 1, n)),
            "tags": na.row_tags(m) if design in ("fix", "fix2") else np.zeros((m, 0), dtype=np.int64)}


def coins(design: str, m: int, n: int, s: int, rng: np.random.Generator, h=None) -> dict:
    """A call's coins. `h` (coin words) reads ssn_1l's and fix's bits through the served extraction."""
    if h is not None and design in ("ssn_1l", "fix", "fix2"):
        a1, b = ns.ssn_1l_bits(h, s)
    else:
        a1 = rng.integers(0, 16 if design == "hw" else 2, (m, 4))
        b = rng.integers(0, 2, (m, s - 1))
    return {"a1": a1.astype(np.int64), "b": b.astype(np.int64), "r": rng.integers(0, 2, (m, s)),
            "cbit": rng.integers(0, 2, n)}


def noise(design: str, pub: dict, co: dict) -> tuple[np.ndarray, np.ndarray]:
    """N (m × n × S) and β (n)."""
    c, a1, b = pub["c"], co["a1"], co["b"]
    m, s1 = b.shape
    n = c.shape[1]
    if design == "rc_magic":
        r = np.cumsum(co["r"], axis=1)
        return r[:, None, :] + co["cbit"][None, :, None], np.zeros(n, dtype=np.int64)
    n1 = a1 @ c + (pub["tags"] @ pub["g"] if design in ("fix", "fix2") else 0)
    if design == "ssn_1l":
        inc = b[:, :, None] * np.ones((1, 1, n), dtype=np.int64)
        mean = c.sum(0) / 2 + s1 / 2
    elif design == "fix":
        inc = (1 + b)[:, :, None] * np.ones((1, 1, n), dtype=np.int64)
        mean = c.sum(0) / 2 + 1.5 * s1
    elif design == "hw":
        inc = (1 + 2 * b)[:, :, None] * pub["sig"][None]
        mean = 7.5 * c.sum(0) + 2 * pub["sig"].sum(0)
    elif design == "fix2":
        inc = (1 + b)[:, :, None] * pub["w"][None]
        mean = c.sum(0) / 2 + 1.5 * pub["w"].sum(0)
    else:
        raise ValueError(design)
    cum = np.concatenate([np.zeros((m, 1, n), dtype=np.int64), np.cumsum(inc, axis=1)], axis=1)
    return n1[:, :, None] + cum.transpose(0, 2, 1), np.rint(mean).astype(np.int64)


def final_noise(design: str, pub: dict, co: dict) -> np.ndarray:
    """The served noise N_S − β (m × n), without the m × n × S intermediate."""
    c, a1, b = pub["c"], co["a1"], co["b"]
    n1 = a1 @ c + (pub["tags"] @ pub["g"] if design in ("fix", "fix2") else 0)
    s1 = b.shape[1]
    if design == "ssn_1l":
        return n1 + b.sum(1)[:, None] - np.rint(c.sum(0) / 2 + s1 / 2).astype(np.int64)
    if design == "fix":
        return n1 + (1 + b).sum(1)[:, None] - np.rint(c.sum(0) / 2 + 1.5 * s1).astype(np.int64)
    if design == "hw":
        inc = _exact_matmul(1 + 2 * b, pub["sig"])
        return n1 + inc - np.rint(7.5 * c.sum(0) + 2 * pub["sig"].sum(0)).astype(np.int64)
    if design == "fix2":
        return n1 + _exact_matmul(1 + b, pub["w"]) - np.rint(c.sum(0) / 2 + 1.5 * pub["w"].sum(0)).astype(np.int64)
    raise ValueError(design)


def _exact_matmul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x @ y through float64 BLAS; exact while every partial sum stays below 2^53."""
    assert np.abs(x).max(initial=0) * np.abs(y).max(initial=0) * x.shape[1] < 2 ** 53
    return np.rint(x.astype(np.float64) @ y.astype(np.float64)).astype(np.int64)


def partials(a: np.ndarray, b: np.ndarray, d1: int, s: int) -> np.ndarray:
    out, acc, lo = [], np.zeros((a.shape[0], b.shape[1]), dtype=np.int64), 0
    for t in range(s):
        hi = d1 + 31 * t
        acc = acc + a[:, lo:hi] @ b[lo:hi]
        out.append(acc)
        lo = hi
    return np.stack(out, axis=2)


def words(design, a, b, pub, co) -> np.ndarray:
    s = co["b"].shape[1] + 1
    nz, beta = noise(design, pub, co)
    return partials(a, b, first_lanes(design), s) + nz - beta[None, :, None]


def case_data(case: str, design: str, m: int, n: int, s: int, pub: dict, rng: np.random.Generator):
    d1 = first_lanes(design)
    k = d1 + 31 * (s - 1)
    row = np.clip(np.rint(rng.normal(0, 30, (1, k))), -127, 127).astype(np.int64)
    a = np.clip(np.rint(rng.normal(0, 30, (m, k))), -127, 127).astype(np.int64)
    b = np.clip(np.rint(rng.normal(0, 25, (k, n))), -127, 127).astype(np.int64)
    if case == "identical_rows":
        a = np.repeat(row, m, axis=0)
    elif case == "zero_tails":
        a[:, d1:] = 0
    elif case in ("cancel", "cancel_identical"):
        if case == "cancel_identical":
            a = np.repeat(row, m, axis=0)
        a[:, d1:] = 0
        pat = {"fix": np.ones((s - 1, n)), "hw": pub["sig"], "fix2": pub["sig"], "ssn_1l": np.ones((s - 1, n)),
               "rc_magic": np.ones((s - 1, n))}[design]
        for t in range(1, s):
            lane = d1 + 31 * (t - 1)
            a[:, lane] = 1
            b[lane] = -pat[t - 1]
    elif case == "identical_cols":
        b = np.repeat(b[:, :1], n, axis=1)
    elif case != "natural":
        raise ValueError(case)
    return a, b


def last_one(b: np.ndarray) -> np.ndarray:
    """For each row and step t (1-based steps 1..S), the last step ≤ t whose bit is 1 (1 if none)."""
    m, s1 = b.shape
    idx = np.where(b == 1, np.arange(2, s1 + 2)[None], 0)
    return np.concatenate([np.ones((m, 1), dtype=np.int64), np.maximum.accumulate(np.maximum(idx, 1), axis=1)], 1)


def adversary_keys(design: str, case: str, a: np.ndarray, co: dict) -> np.ndarray:
    """The adaptive prover's key for every (row, step), from data and coins alone; one vector a key."""
    m, s = a.shape[0], co["b"].shape[1] + 1
    d1 = first_lanes(design)
    rows = np.unique(a, axis=0, return_inverse=True)[1].reshape(m)
    step = np.arange(s)[None]
    if design == "rc_magic" and case == "identical_rows":
        key = (rows[:, None] * s + step) * (s + 1) + np.cumsum(co["r"], axis=1)
    elif design == "rc_magic" and case == "zero_tails":
        key = np.arange(m)[:, None] * (s + 1) + np.cumsum(co["r"], axis=1)
    elif case == "cancel" and design in ("fix", "hw", "fix2"):
        key = np.arange(m)[:, None] * (s + 2) + last_one(co["b"])
    else:
        key = np.arange(m * s).reshape(m, s)
    assert d1 > 0
    return key


def build(design, a, b, pub, co, key) -> tuple[np.ndarray, int]:
    """One vector a key, computed from the key's first (row, step) by the honest formula, copied to the others."""
    w = words(design, a, b, pub, co)
    m, n, s = w.shape
    out = np.empty_like(w)
    first = {}
    for i in range(m):
        for t in range(s):
            src = first.setdefault(int(key[i, t]), (i, t))
            out[i, :, t] = w[src[0], :, src[1]]
    return out, len(first) * n


def small(seed: int = 23) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    m, n, s = 384, 32, 8
    res, det = [], {}

    def ok(name, cond, d):
        res.append({"check": name, "pass": bool(cond), "detail": d})

    for design in ("ssn_1l", "fix"):
        pub = public(design, n, s, m, rng)
        a, b = case_data("natural", design, m, n, s, pub, rng)
        h = rng.integers(0, 2 ** 32, (m, ns.ssn_1l_words(s)), dtype=np.int64)
        co = coins(design, m, n, s, rng, h)
        ref = na.honest(a, b, h, s, pub["g"] if design == "fix" else None, design == "fix")
        ok(f"{design}: the formula's words equal interim 1's served-layout words (X·Y by k32 steps)",
           np.array_equal(words(design, a, b, pub, co), ref), {"m": m, "n": n, "S": s})
    for design in DESIGNS:
        pub = public(design, n, s, m, rng)
        for case in ("natural", "identical_rows", "zero_tails", "cancel", "cancel_identical", "identical_cols"):
            a, b = case_data(case, design, m, n, s, pub, rng)
            co = coins(design, m, n, s, rng)
            w = words(design, a, b, pub, co)
            orc = na.oracle(w)
            rep = int(sum((w[:, :, t] == w[:, :, t - 1]).all(axis=1).sum() for t in range(1, s)))
            r = {"oracle_vector_saving": na.saving(w.size, orc["vector_words"], m * s),
                 "oracle_tile_saving": 1 - orc["tile_words"] / w.size, "repeat_previous_step": rep / (m * (s - 1)),
                 "row_values_share": orc["row_values"] / w.size}
            key = adversary_keys(design, case, a, co)
            adv, writes = build(design, a, b, pub, co, key)
            r["adversary_equal"] = bool(np.array_equal(adv, w))
            r["adversary_saving"] = na.saving(w.size, writes, m * s) if r["adversary_equal"] else 0.0
            if design == "rc_magic":
                r["exact_after_epilogue"] = bool((w[:, :, -1] - co["r"].sum(1)[:, None] - co["cbit"][None]
                                                  == a @ b).all())
            det[f"{design}/{case}"] = r
    d = det
    for design in ("fix", "hw"):
        ok(f"A2c {design}: a chosen B lane of −(public step weight) makes the adaptive prover's copies save > 35%, "
           "words equal", d[f"{design}/cancel"]["adversary_equal"] and d[f"{design}/cancel"]["adversary_saving"] > 0.35,
           d[f"{design}/cancel"])
    ok("fix2: the same cancellation finds no repeat (oracle: every row vector distinct), and the copying prover's "
       "words differ from honest (it fails)",
       d["fix2/cancel"]["oracle_vector_saving"] <= 0 and not d["fix2/cancel"]["adversary_equal"], d["fix2/cancel"])
    for case in ("identical_rows", "zero_tails", "cancel_identical"):
        ok(f"fix2 {case}: every row vector distinct", d[f"fix2/{case}"]["oracle_vector_saving"] <= 0, d[f"fix2/{case}"])
    ok("fix2: < 2% of 4-word tiles repeat on identical rows and on identical columns",
       max(d["fix2/identical_rows"]["oracle_tile_saving"], d["fix2/identical_cols"]["oracle_tile_saving"]) < 0.02,
       {c: d[f"fix2/{c}"]["oracle_tile_saving"] for c in ("identical_rows", "identical_cols")})
    ok("A1 rc_magic identical rows: the adaptive prover saves > 80%, words equal",
       d["rc_magic/identical_rows"]["adversary_equal"] and d["rc_magic/identical_rows"]["adversary_saving"] > 0.8,
       d["rc_magic/identical_rows"])
    ok("A2 rc_magic zero tails: saves > 30%, words equal",
       d["rc_magic/zero_tails"]["adversary_equal"] and d["rc_magic/zero_tails"]["adversary_saving"] > 0.3,
       d["rc_magic/zero_tails"])
    ok("A3 rc_magic identical columns: ≤ 2 values a row-step (row_values share ≤ 2/n)",
       d["rc_magic/identical_cols"]["row_values_share"] <= 2 / n + 1e-12, d["rc_magic/identical_cols"])
    ok("rc_magic stays exact after the epilogue on every case",
       all(d[f"rc_magic/{c}"]["exact_after_epilogue"] for c in ("natural", "identical_rows", "zero_tails")), "")
    pub = public("hw", n, s, m, rng)
    a, b = case_data("cancel", "hw", m, n, s, pub, rng)
    co = coins("hw", m, n, s, rng)
    bad = np.arange(m)[:, None] * (s + 1) + np.arange(s)[None] * 0
    adv, _ = build("hw", a, b, pub, co, bad)
    ok("mutation: keying hw's cancelled rows on the row alone breaks equality",
       not np.array_equal(adv, words("hw", a, b, pub, co)), "")
    return res, det


def birthday(seed: int = 29) -> dict:
    """A7 on identical rows: distinct keys a step among m rows, `hw` keyed (a₁ ∈ 16 bits, step bits through t), `fix2`
    keyed (tag mod 13^L − 1, a₁ ∈ 4 bits, step bits through t). Gap as a share of stock over Llama's linears."""
    rng = np.random.default_rng(seed)
    out = {}
    for label, nb, lanes in (("hw", 16, None), ("fix2_L4", 4, 4), ("fix2_L5", 4, 5), ("fix2_L6", 4, 6)):
        for m in BIG_MS:
            saved = stock = 0.0
            for lin, (n, k) in ns.LLAMA.items():
                s = ns.steps1l(k)
                key = rng.integers(0, 1 << nb, m).astype(np.int64)
                if lanes is not None:
                    key = key + (np.arange(m) % (13 ** lanes - 1)) * (1 << nb)
                ident = np.unique(key, return_inverse=True)[1].reshape(m)
                last = int(ident.max()) + 1
                writes = last
                for t in range(1, s):
                    if last == m:
                        writes += (s - t) * m
                        break
                    bit = rng.integers(0, 2, m)
                    ident = np.unique(ident * 2 + bit, return_inverse=True)[1].reshape(m)
                    last = int(ident.max()) + 1
                    writes += last
                saved += max(ns.H32 * (n * (m * s - writes) - m * s), 0.0)
                stock += m * n * k
            out[f"{label}/m{m}"] = saved / stock
    return out


def rc_at_scale(seed: int = 31) -> dict:
    """A1 (identical rows), A2 (zero tails), A3 (identical columns, per-word) and A6 (decode padding, r real rows of
    64) on rc_magic, as a share of stock over each family's linears."""
    rng = np.random.default_rng(seed)
    out = {}
    for fam, shapes in na.SHAPES.items():
        for m in na.MS:
            acc = {"A1": 0.0, "A2": 0.0, "A3": 0.0, "stock": 0.0, "words": 0.0}
            for n, k in shapes.values():
                s = rc_steps(k)
                r = rng.integers(0, 2, (m, s))
                cum = np.cumsum(r, axis=1)
                keys = sum(len(np.unique(cum[:, t])) for t in range(s))
                tail = int(m + r[:, 1:].sum())
                w, pts = m * n * s, m * s
                acc["A1"] += ns.H32 * (w - n * keys - pts)
                acc["A2"] += ns.H32 * (w - n * tail - pts)
                acc["A3"] += ns.H32 * (w - 2 * m * s - pts)
                acc["stock"] += m * n * k
                acc["words"] += w
            out[f"{fam}_m{m}"] = {key: acc[key] / acc["stock"] for key in ("A1", "A2", "A3")}
            out[f"{fam}_m{m}"]["alpha"] = ns.H32 * acc["words"] / acc["stock"]
    pad = {}
    for real in (1, 8, 32):
        sv = tot = 0.0
        for n, k in ns.LLAMA.values():
            s = rc_steps(k)
            p = 64 - real
            cum = np.cumsum(rng.integers(0, 2, (p, s)), axis=1)
            keys = sum(len(np.unique(cum[:, t])) for t in range(s))
            sv += ns.H32 * (p * n * s - n * keys - p * s)
            tot += ns.H32 * 64 * n * s
        pad[f"r{real}"] = sv / tot
    out["A6_padding_share_of_padded_words"] = pad
    return out


def a2c_at_scale(seed: int = 37) -> dict:
    """A2c on fix and hw: with the public step weight cancelled, a row's step t repeats step t − 1 whenever b_t = 0."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in na.MS:
        sv = st = 0.0
        for n, k in ns.LLAMA.values():
            s = ns.steps1l(k)
            b = rng.integers(0, 2, (m, s - 1))
            writes = n * (m + int(b.sum()))
            sv += ns.H32 * (m * n * s - writes - m * s)
            st += m * n * k
        out[f"m{m}"] = sv / st
    return out


# ---- H2: SHA-512 compressions of A's row digests, counted on the real messages ----

_K512 = [int(x, 16) for x in """
428a2f98d728ae22 7137449123ef65cd b5c0fbcfec4d3b2f e9b5dba58189dbbc 3956c25bf348b538 59f111f1b605d019 923f82a4af194f9b
ab1c5ed5da6d8118 d807aa98a3030242 12835b0145706fbe 243185be4ee4b28c 550c7dc3d5ffb4e2 72be5d74f27b896f 80deb1fe3b1696b1
9bdc06a725c71235 c19bf174cf692694 e49b69c19ef14ad2 efbe4786384f25e3 0fc19dc68b8cd5b5 240ca1cc77ac9c65 2de92c6f592b0275
4a7484aa6ea6e483 5cb0a9dcbd41fbd4 76f988da831153b5 983e5152ee66dfab a831c66d2db43210 b00327c898fb213f bf597fc7beef0ee4
c6e00bf33da88fc2 d5a79147930aa725 06ca6351e003826f 142929670a0e6e70 27b70a8546d22ffc 2e1b21385c26c926 4d2c6dfc5ac42aed
53380d139d95b3df 650a73548baf63de 766a0abb3c77b2a8 81c2c92e47edaee6 92722c851482353b a2bfe8a14cf10364 a81a664bbc423001
c24b8b70d0f89791 c76c51a30654be30 d192e819d6ef5218 d69906245565a910 f40e35855771202a 106aa07032bbd1b8 19a4c116b8d2d0c8
1e376c085141ab53 2748774cdf8eeb99 34b0bcb5e19b48a8 391c0cb3c5c95a63 4ed8aa4ae3418acb 5b9cca4f7763e373 682e6ff3d6b2b8a3
748f82ee5defb2fc 78a5636f43172f60 84c87814a1f0ab72 8cc702081a6439ec 90befffa23631e28 a4506cebde82bde9 bef9a3f7b2c67915
c67178f2e372532b ca273eceea26619c d186b8c721c0c207 eada7dd6cde0eb1e f57d4f7fee6ed178 06f067aa72176fba 0a637dc5a2c898a6
113f9804bef90dae 1b710b35131c471b 28db77f523047d84 32caab7b40c72493 3c9ebe0a15c9bebc 431d67c49c100d4c 4cc5d4becb3e42b6
597f299cfc657e2a 5fcb6fab3ad6faec 6c44198c4a475817""".split()]
_IV512 = (0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
          0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179)
_M64 = (1 << 64) - 1


def _rotr(x: int, r: int) -> int:
    return ((x >> r) | (x << (64 - r))) & _M64


def _compress(state: tuple, block: bytes) -> tuple:
    w = list(struct.unpack(">16Q", block))
    for t in range(16, 80):
        s0 = _rotr(w[t - 15], 1) ^ _rotr(w[t - 15], 8) ^ (w[t - 15] >> 7)
        s1 = _rotr(w[t - 2], 19) ^ _rotr(w[t - 2], 61) ^ (w[t - 2] >> 6)
        w.append((w[t - 16] + s0 + w[t - 7] + s1) & _M64)
    a, b, c, d, e, f, g, h = state
    for t in range(80):
        t1 = (h + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + ((e & f) ^ (~e & g)) + _K512[t] + w[t]) & _M64
        t2 = ((_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))) & _M64
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & _M64, c, b, a, (t1 + t2) & _M64
    return tuple((x + y) & _M64 for x, y in zip(state, (a, b, c, d, e, f, g, h)))


def sha512_counted(msg: bytes, seen: set, tally: list, const_first: bool) -> bytes:
    """SHA-512 of msg, recording each compression's input (state, block) in `seen` and counting it in tally[0]. A
    constant first block (`const_first`) is a midstate every program computes once a call, so it is not counted."""
    pad = msg + b"\x80" + b"\0" * ((111 - len(msg)) % 128) + (8 * len(msg)).to_bytes(16, "big")
    st = _IV512
    for o in range(0, len(pad), H2_BLOCK):
        blk = pad[o:o + H2_BLOCK]
        if not (const_first and o == 0):
            seen.add((st, blk))
            tally[0] += 1
        st = _compress(st, blk)
    return b"".join(x.to_bytes(8, "big") for x in st)


def row_seg_counted(row: bytes, n_bits: int, seen: set, tally: list, tag: bytes = b"") -> bytes:
    """rowleaf.sha512_row_seg_digest, with every compression counted. `tag` (the fix) goes into each segment's first
    block, so the segment prefix is no longer a constant midstate."""
    from verity.ml.commitments import rowleaf
    segs = rowleaf.sha512_row_seg_segments(n_bits)
    s = len(segs)
    ds = []
    for j, mb in enumerate(segs):
        pre = rowleaf.sha512_row_seg_segment_prefix(j, s, mb)
        if tag:
            pre = (pre.rstrip(b"\0") + b"\0" + tag).ljust(128, b"\0")
        seg = row[j * rowleaf.ROW_SEG_BYTES:j * rowleaf.ROW_SEG_BYTES + (mb + 7) // 8]
        ds.append(sha512_counted(pre + seg, seen, tally, const_first=not tag))
    return sha512_counted(rowleaf.sha512_row_seg_prefix(n_bits) + b"".join(ds), seen, tally, const_first=True)


def h2(seed: int = 41) -> dict:
    """The share of A's row-digest compressions a call needs once identical inputs are computed once (ShaSpeed and
    hash-compose force distinct compressions only). Tree leaves (2 compressions, position-bound) and nodes (3) are
    counted as distinct. Rows are int8 (K bytes), row-seg (Daniel's ruling, card 637f70c6)."""
    from verity.ml.commitments import rowleaf
    rng = np.random.default_rng(seed)
    out, checks = {}, []
    for k in (2048, 8192):
        for m in (64, 256):
            for case in ("identical_rows", "prefix_rows", "natural"):
                base = rng.integers(-127, 128, k, dtype=np.int64)
                rows = np.repeat(base[None], m, axis=0)
                if case == "prefix_rows":
                    rows[:, -1024:] = rng.integers(-127, 128, (m, 1024))
                elif case == "natural":
                    rows = rng.integers(-127, 128, (m, k))
                for fixed in (False, True):
                    seen, tally = set(), [0]
                    for i in range(m):
                        rb = (rows[i] & 0xFF).astype(np.uint8).tobytes()
                        tag = (i.to_bytes(8, "big") if fixed else b"")
                        dig = row_seg_counted(rb, 8 * k, seen, tally, tag)
                        if not fixed and i < 2:
                            checks.append(dig == rowleaf.sha512_row_seg_digest(rb, 8 * k))
                    tree = 2 * m + 3 * (m - 1)
                    tot, dist = tally[0] + tree, len(seen) + tree
                    out[f"K{k}/m{m}/{case}/{'fix' if fixed else 'row_seg'}"] = {
                        "compressions": tot, "distinct": dist, "saved_share": 1 - dist / tot}
    for k in (2048, 8192):
        a, f = out[f"K{k}/m64/natural/row_seg"], out[f"K{k}/m64/natural/fix"]
        out[f"K{k}/fix_cost"] = f["compressions"] / a["compressions"] - 1
    return {"counts": out, "digest_matches_rowleaf": all(checks), "n_checked": len(checks)}


def fix2_costs() -> dict:
    """fix2's extra coins and forming: w_t is n·(S − 1) signs a linear a window, B′'s step lanes re-formed once a window."""
    out = {}
    for lin, (n, k) in ns.LLAMA.items():
        s = ns.steps1l(k)
        out[lin] = {"sign_bits_a_window": n * (s - 1) * ns.LLAMA_LAYERS,
                    "forming_entries_a_window": n * (s - 1) * ns.LLAMA_LAYERS}
    return out


def final_noise_matches(seed: int = 47) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for design in ("ssn_1l", "fix", "hw", "fix2"):
        m, n, s = 96, 40, 9
        pub = public(design, n, s, m, rng)
        co = coins(design, m, n, s, rng)
        nz, beta = noise(design, pub, co)
        out[design] = bool(np.array_equal(final_noise(design, pub, co), nz[:, :, -1] - beta[None]))
    return out


def a5_vectors(design: str, t: int) -> float:
    """A row's possible step-t vectors once the window's public draws are known: 16 a₁ codes times ssn_1l's t offset
    sums (b_t·1 adds one vector), or fix2's 2^(t−1) sign patterns (each step's w_t differs). With w drawn per call,
    after any advice, a fix2 vector past step 1 can't be tabled."""
    if design == "ssn_1l":
        return 16 * t
    if design == "fix2":
        return 16 * 2 ** (t - 1)
    return 16 if t == 1 else float("inf")


def a5_step_greedy(designs=("ssn_1l", "fix2", "fix2_call_w")) -> dict:
    """A5 corrected: advice tables each row's earliest steps (every possible vector of steps 1..τ_ℓ of linear ℓ) in one
    H200's HBM, so those words are copies; the row then resumes the k32 chain, a 32-a-word load of its step-τ vector,
    unless τ_ℓ = S. The best (τ_ℓ) by exhaustive search, its saving as a share of stock. Interim 1's `advice_table`
    tabled whole linears and understated it."""
    out = {}
    lins = [(n, k, ns.steps1l(k)) for n, k in ns.LLAMA.values()]
    for design in designs:
        for m in na.MS:
            per = []
            for n, k, s in lins:
                unit = n * m * ns.LLAMA_LAYERS
                cost, opts = 0.0, [(0.0, 0.0, 0)]
                for t in range(1, s + 1):
                    cost += a5_vectors(design, t) * 4 * unit
                    if cost > na.HBM_BYTES:
                        break
                    opts.append((cost, ns.H32 * unit * (t if t == s else t - 1), t))
                per.append(opts)
            best = (0.0, (0,) * len(lins))
            last_costs = [o[0] for o in per[3]]
            for o0 in per[0]:
                for o1 in per[1]:
                    if o0[0] + o1[0] > na.HBM_BYTES:
                        break
                    for o2 in per[2]:
                        room = na.HBM_BYTES - o0[0] - o1[0] - o2[0]
                        if room < 0:
                            break
                        o3 = per[3][bisect.bisect_right(last_costs, room) - 1]
                        gain = o0[1] + o1[1] + o2[1] + o3[1]
                        if gain > best[0]:
                            best = (gain, (o0[2], o1[2], o2[2], o3[2]))
            stock = sum(n * k * m * ns.LLAMA_LAYERS for n, k, _ in lins)
            out[f"{design}/m{m}"] = {"gap_share_of_stock": best[0] / stock,
                                     "steps_tabled": dict(zip(ns.LLAMA, best[1]))}
    out["fix2_call_w_coin_bits_a_call"] = sum(n * (s - 1) for n, _, s in lins) * ns.LLAMA_LAYERS
    out["fix2_call_w_sha512_compressions_a_call"] = out["fix2_call_w_coin_bits_a_call"] / 512
    return out


def served_std(seed: int = 43) -> dict:
    """The served noise's std (final word minus A·Bᵀ, de-biased) for each design at Llama's K, m 512."""
    rng = np.random.default_rng(seed)
    out = {}
    for design in DESIGNS:
        for k in (2048, 8192):
            n, m = 256, 512
            s = steps(design, k)
            pub = public(design, n, s, m, rng)
            co = coins(design, m, n, s, rng)
            nz, beta = noise(design, pub, co)
            out[f"{design}/K{k}"] = float((nz[:, :, -1] - beta[None]).std())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the report here")
    args = ap.parse_args(argv)
    checks, det = small()
    bd = birthday()
    rc = rc_at_scale()
    a2c = a2c_at_scale()
    hh = h2()
    checks.append({"check": "H2: the instrumented SHA-512 reproduces rowleaf.sha512_row_seg_digest",
                   "pass": hh["digest_matches_rowleaf"] and hh["n_checked"] > 0, "detail": hh["n_checked"]})
    c = hh["counts"]
    checks.append({"check": "H2: identical rows share > 75% of A's commitment compressions at m 64, K 2,048 and 8,192",
                   "pass": min(c[f"K{k}/m64/identical_rows/row_seg"]["saved_share"] for k in (2048, 8192)) > 0.75,
                   "detail": {k: c[f"K{k}/m64/identical_rows/row_seg"] for k in (2048, 8192)}})
    checks.append({"check": "H2 fix: a position in every segment's first block leaves no shared compression",
                   "pass": max(c[f"K{k}/m{m}/{cs}/fix"]["saved_share"] for k in (2048, 8192) for m in (64, 256)
                               for cs in ("identical_rows", "prefix_rows", "natural")) <= 0, "detail": ""})
    checks.append({"check": "A7: hw on identical rows passes 1% of stock by m 2^20",
                   "pass": bd[f"hw/m{1 << 20}"] > 0.01, "detail": {k: v for k, v in bd.items() if k.startswith("hw")}})
    checks.append({"check": "A7: fix2 with 6 tag lanes saves nothing up to m 2^20",
                   "pass": max(v for k, v in bd.items() if k.startswith("fix2_L6")) <= 0, "detail": ""})
    checks.append({"check": "A1 rc_magic: > 1% of stock at Llama m 64",
                   "pass": rc["llama_m64"]["A1"] > 0.01, "detail": rc["llama_m64"]})
    checks.append({"check": "A2c fix/hw: > 30% of stock at every m", "pass": min(a2c.values()) > 0.3, "detail": a2c})
    fn_ok = final_noise_matches()
    checks.append({"check": "final_noise equals the full model's last step minus β, every design but rc_magic",
                   "pass": all(fn_ok.values()), "detail": fn_ok})
    a5 = a5_step_greedy()
    checks.append({"check": "A5 step-greedy: ssn_1l and fix2 > 1% of stock at m 64; fix2 with per-call w saves 0",
                   "pass": min(a5[f"{d}/m64"]["gap_share_of_stock"] for d in ("ssn_1l", "fix2")) > 0.01
                   and max(a5[f"fix2_call_w/m{m}"]["gap_share_of_stock"] for m in na.MS) == 0,
                   "detail": {k: v["gap_share_of_stock"] for k, v in a5.items() if isinstance(v, dict)}})
    rep = {"checks": checks, "small": det, "birthday": bd, "rc_magic_at_scale": rc, "a2c_at_scale": a2c, "h2": hh,
           "a5_step_greedy": a5, "fix2_costs": fix2_costs(), "served_std": served_std(),
           "periods": {"A2c_m8192": ns.periods(a2c["m8192"], 1.044),
                       "rc_A1_llama_m8192": ns.periods(rc["llama_m8192"]["A1"], rc["llama_m8192"]["alpha"]),
                       "ssn_1l_A5_m64": ns.periods(a5["ssn_1l/m64"]["gap_share_of_stock"], 1.044),
                       "fix2_A5_m64": ns.periods(a5["fix2/m64"]["gap_share_of_stock"], 1.044),
                       "hw_A7_m131072": ns.periods(bd["hw/m131072"], 1.044),
                       "hw_A7_m1048576": ns.periods(bd[f"hw/m{1 << 20}"], 1.044)}}
    text = json.dumps(rep, indent=1, default=float)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    for ch in checks:
        print(("PASS " if ch["pass"] else "FAIL ") + ch["check"])
    print(json.dumps({"birthday": bd, "a2c": a2c, "rc_magic": rc, "served_std": rep["served_std"],
                      "h2": {k: v for k, v in c.items() if "/m64/" in k or "fix_cost" in k}}, indent=1, default=float))
    return 0 if all(ch["pass"] for ch in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
