"""NH over the accumulator words against the zero-data decomposition, exact, priced in W1 register writes.

Two layouts of NH, each one 32-bit S a block, mod 2³²:

- `nh2` (P pairs, T steps): S(i, g) = Σ_t Σ_p w_t(i, 2p)·w_t(i, 2p + 1) over a group g of 2P adjacent columns of row i and
  a block of T steps, G = 2PT words an S, one IMAD a pair;
- `nh` (T steps): S(i, j) = Σ_k w_s(i, j)·w_{s+1}(i, j) over the pairs (s, s + 1) of a block of T steps of one
  accumulator, G = T.

The words are a separable zero-data family: w_t(i, j) = d0(i, j) + R_t(i) + C_t(c(i), j), c(i) a row class. `counter`
instances are `hash_counter.Draw`'s (the `lead:*` and `sep` coins, zero or random data, with the split read off its
lanes as its `pack` does); `noise` instances are `commit_space.lead_parts` (X + Y_t(i) + Z_t(par i, tile i, j)), whose
classes are a 64-row tile's two parities, so column work is shared by 32 rows. `chosen` adds data the prover picks:
r data lanes whose row coefficients take two patterns, which only doubles the classes. Random data is not separable,
and honest is the only program run on it.

The decomposition (`attack_nh2`, `attack_nh`) expands each product: the d0 terms by IMAD, row-only and column-only sums
once a row or a column class (amortized over a CALL_M × CALL_N call), and the cross term Σ_t R_t(i)·K_t(c, g) as s8
IMMA chains over balanced byte digits, one chain a digit weight x + y ≤ 3 (`imma`), which is exact mod 2³². Every
attack's S is checked bit for bit against honest's.

Prices (W1, register writes an S): honest pays an MMA a word and an IMAD a pair (nh2 3PT, nh 1.5T); stock is the data
lanes' share of the words' MMAs. `gap_claimed` = (honest − cheapest)/stock refutes a premise that honest's sketch is
the cheapest program (PairWrites) above 1%; `gap_share` = (stock − cheapest)/stock is credit the attack is paid and
didn't spend. The final word is not charged (M0 knows it). The S's seal is common to every program and left out.

`mirror` measures the exact ranks of NH's edge and middle-block sums on mirror-π noise (`commit_space._edge_noise`),
priced as that module prices the quadratic sketch's: an edge of t0 words costs rank·B/32 (B byte products a rank term),
a middle block of L words from the honest word N before it costs its linear words and IMADs plus rank·B/32.

Run (from this directory, with `hash_counter.py` and `commit_space.py`, `noise_space.py`, `noise_sweep.py` in DIR):
`uv run --no-project --with numpy --with-editable ../exact_kit python -B nh_attack.py OUT.json --lib=DIR [--mutations]`.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from exact_kit.checks import Checks

M32 = (1 << 32) - 1
CALL_M = CALL_N = 4096
T1 = 0.01
SKETCH_BYTES = 4
NOISE_STOCK = 28 / 32


def u32(x):
    return np.asarray(x).astype(np.int64).astype(np.uint64) & np.uint64(M32)


def digits(v, n=4):
    """Balanced signed byte digits of v mod 2³² (as int32), n of them: v ≡ Σ_x d_x·256^x mod 2³², d_x in [−128, 127]."""
    s = u32(v).astype(np.int64)
    s = np.where(s >= 1 << 31, s - (1 << 32), s)
    out = []
    for _ in range(n):
        d = ((s + 128) % 256) - 128
        out.append(d)
        s = (s - d) // 256
    return out


def imma(left, right):
    """Σ_k left[r, k]·right[c, k] mod 2³² from s8 digit products, one IMMA chain (depth 32) a digit weight ≤ 3. Returns
    the (rows, cols) result and its price an output entry: Σ_w ⌈n_w·K/32⌉ chains and a shift-add a weight past the first."""
    K = left.shape[1]
    dl, dr = digits(left), digits(right)
    used_l = [x for x in range(4) if np.any(dl[x])]
    used_r = [y for y in range(4) if np.any(dr[y])]
    out = np.zeros((left.shape[0], right.shape[0]), np.int64)
    weights = {}
    for x in used_l:
        for y in used_r:
            if x + y <= 3:
                weights[x + y] = weights.get(x + y, 0) + 1
                out += (dl[x] @ dr[y].T) << (8 * (x + y))
    cost = sum(math.ceil(n * K / 32) for n in weights.values()) + max(0, len(weights) - 1)
    return u32(out), cost


def honest_nh2(W, P, T):
    m, n, S = W.shape
    blocks = []
    for t0 in range(0, S - T + 1, T):
        Wb = W[:, :, t0:t0 + T]
        prod = (Wb[:, 0::2, :] * Wb[:, 1::2, :]) & np.uint64(M32)
        blocks.append(prod.reshape(m, n // (2 * P), P, T).sum(axis=(2, 3)) & np.uint64(M32))
    return np.stack(blocks, -1)


def honest_nh(W, T):
    blocks = []
    for t0 in range(0, W.shape[2] - T + 1, T):
        Wb = W[:, :, t0:t0 + T]
        blocks.append(((Wb[:, :, 0::2] * Wb[:, :, 1::2]) & np.uint64(M32)).sum(axis=2) & np.uint64(M32))
    return np.stack(blocks, -1)


def attack_nh2(parts, P, T, d0_free=False, lowrank=False):
    """nh2 from the split, block by block. Per S: T·Σ d0a·d0b, ΣR·Σd0, Σ (d0a·ΣC_b + d0b·ΣC_a), P·ΣR², the cross
    term Σ_t R_t·K_t(c, g) with K_t the group's column sum, and Q(c, g) = Σ_t Σ_p C_a·C_b once a class. `d0_free` takes
    d0's own sums as precomputed (a salt-independent d0); `lowrank` evaluates the d0 terms from d0 = λ·ρᵀ
    (`parts["d0_factors"]`) as IMMA chains of inner dimension k², k and k in place of 4P + 2 IMADs."""
    d0, R, C, cls, share = parts["d0"], parts["R"], parts["C"], parts["cls"], parts["rows_a_class"]
    m, n = d0.shape
    ng, S = n // (2 * P), R.shape[1]
    d0a, d0b = d0[:, 0::2], d0[:, 1::2]
    if lowrank:
        lam, rho = parts["d0_factors"]
        k = lam.shape[1]
        lam2 = (lam[:, :, None] * lam[:, None, :]).reshape(m, k * k)
        rho2 = (rho[0::2][:, :, None] * rho[1::2][:, None, :]).reshape(ng, P, k * k).sum(1)
        rhog = rho.reshape(ng, 2 * P, k).sum(1)
    out, cross_cost, d0_cost = [], 0, 0
    for t0 in range(0, S - T + 1, T):
        Rb, Cb = R[:, t0:t0 + T], C[:, :, t0:t0 + T]
        sR, sR2 = Rb.sum(1) & np.uint64(M32), (Rb * Rb).sum(1) & np.uint64(M32)
        sC = Cb.sum(2) & np.uint64(M32)
        K = Cb.reshape(len(C), ng, 2 * P, T).sum(2) & np.uint64(M32)
        Q = ((Cb[:, 0::2, :] * Cb[:, 1::2, :]) & np.uint64(M32)).reshape(len(C), ng, P, T).sum((2, 3)) & np.uint64(M32)
        s = (np.uint64(P) * sR2[:, None] + Q[cls]) & np.uint64(M32)
        if lowrank:
            x1, c1 = imma(lam2, (np.uint64(T) * rho2) & np.uint64(M32))
            x2, c2 = imma((sR[:, None] * lam) & np.uint64(M32), rhog)
            s += x1 + x2
            c3 = 0
            for c in range(len(C)):
                rows = np.flatnonzero(cls == c)
                right = ((rho[0::2] * sC[c][1::2, None]) + (rho[1::2] * sC[c][0::2, None])).reshape(ng, P, k).sum(1)
                x3, cc = imma(lam[rows], right & np.uint64(M32))
                c3 = max(c3, cc)
                s[rows] += x3
            d0_cost = max(d0_cost, (1 if d0_free else c1 + c2) + c3 + 2)
        else:
            s += (np.uint64(T) * (d0a * d0b).reshape(m, ng, P).sum(2)) & np.uint64(M32)
            s += sR[:, None] * (d0.reshape(m, ng, 2 * P).sum(2) & np.uint64(M32))
            s += ((d0a * sC[cls][:, 1::2]) + (d0b * sC[cls][:, 0::2])).reshape(m, ng, P).sum(2)
            d0_cost = 2 * P + 1 if d0_free else 4 * P + 2
        s &= np.uint64(M32)
        for c in range(len(C)):
            rows = np.flatnonzero(cls == c)
            x, cost = imma(Rb[rows], K[c])
            cross_cost = max(cross_cost, cost)
            s[rows] = (s[rows] + x) & np.uint64(M32)
        out.append(s)
    fixed = d0_cost + 1 + 3
    d_r = len([x for x in range(4) if np.any(digits(R)[x])])
    row = T * (1.5 + d_r) * 2 * P / CALL_N
    col = (3 * P * T + 4 * T) / share
    if lowrank:
        k = parts["d0_factors"][0].shape[1]
        row += (k * k + k) * 2 * P / CALL_N
        col += 2 * k * P / share + (k * k + 2 * k) * P / CALL_M
    return np.stack(out, -1), {"fixed": fixed, "cross": cross_cost, "row": row, "col": col,
                               "total": fixed + cross_cost + row + col}


def attack_nh(parts, T, d0_free=False):
    """nh from the split: (T/2)·d0², d0·(ΣR + ΣC), Σ R_s·R_{s+1} once a row, Σ C_s·C_{s+1} once a class, and the cross
    term Σ_t R'_t·C_t with R' the row prefix with each pair's two steps swapped."""
    d0, R, C, cls, share = parts["d0"], parts["R"], parts["C"], parts["cls"], parts["rows_a_class"]
    S = R.shape[1]
    out, cross_cost = [], 0
    for t0 in range(0, S - T + 1, T):
        Rb, Cb = R[:, t0:t0 + T], C[:, :, t0:t0 + T]
        Rs = Rb.reshape(len(Rb), T // 2, 2)[:, :, ::-1].reshape(len(Rb), T)
        RR = (Rb[:, 0::2] * Rb[:, 1::2]).sum(1) & np.uint64(M32)
        CC = (Cb[:, :, 0::2] * Cb[:, :, 1::2]).sum(2) & np.uint64(M32)
        s = (np.uint64(T // 2) * d0 * d0) & np.uint64(M32)
        s += d0 * ((Rb.sum(1)[:, None] + Cb.sum(2)[cls]) & np.uint64(M32))
        s += RR[:, None] + CC[cls]
        s &= np.uint64(M32)
        for c in range(len(C)):
            rows = np.flatnonzero(cls == c)
            x, cost = imma(Rs[rows], Cb[c])
            cross_cost = max(cross_cost, cost)
            s[rows] = (s[rows] + x) & np.uint64(M32)
        out.append(s)
    fixed = (1 if d0_free else 2) + 2 + 3
    d_r = len([x for x in range(4) if np.any(digits(R)[x])])
    row = T * (1 + d_r) / CALL_N
    col = 5 * T / share
    return np.stack(out, -1), {"fixed": fixed, "cross": cross_cost, "row": row, "col": col,
                               "total": fixed + cross_cost + row + col}


def counter_parts(hc, coins, data, S, m=64, n=32, q=5, seed=0, chosen=0):
    """`hash_counter.Draw`'s words and their split, read off its lanes as `pack` does: lanes whose column coin is
    constant are row terms, the rest are column terms a row pattern. `chosen` r > 0 adds r data lanes with two row
    patterns against random column data."""
    inst = hc.Inst(m, n, S, q, data, "add", seed, coins)
    d = hc.Draw(inst, hc.Rule("all", "32", "hash"))
    U, V, A, B = d.U.copy(), d.V.copy(), d.A, d.B
    W = d.w.copy()
    if chosen:
        g = np.random.default_rng(seed + 100)
        pat = g.integers(-128, 128, (2, chosen))
        Ad = np.broadcast_to(pat[(np.arange(m) // 2) % 2][:, None, :], (m, S, chosen))
        Bd = g.integers(-128, 128, (n, S, chosen))
        U, V = np.concatenate([U, Ad], 2), np.concatenate([V, Bd], 2)
        W = (W + np.cumsum(np.einsum("itl,jtl->ijt", Ad, Bd), axis=2)) & M32
    rowlane = np.all(V == V[:1], axis=(0, 1))
    pats, cls = np.unique(U[:, :, ~rowlane].reshape(m, -1), axis=0, return_inverse=True)
    cls = cls.reshape(-1)
    R = np.cumsum(np.einsum("itl,tl->it", U[:, :, rowlane], V[0][:, rowlane]), axis=1)
    C = np.stack([np.cumsum(np.einsum("tl,jtl->jt", p.reshape(S, -1), V[:, :, ~rowlane]), axis=1) for p in pats])
    split = (d.w0[:, :, None] + R[:, None, :] + C[cls]) & M32
    separable = data == "zero" and bool(np.array_equal(split, W & M32))
    return {"W": u32(W), "d0": u32(d.w0), "R": u32(R), "C": u32(C), "cls": cls, "separable": separable,
            "rows_a_class": CALL_M / len(pats), "stock_frac": (32 - q) / 32,
            "label": f"counter {coins} {data}" + (f" chosen r{chosen}" if chosen else "")}


def noise_parts(cs, S, m=128, n=32, seed=0, data_sigma=0.0):
    """`commit_space.lead_parts`' words, split as X + Y_t(i) + Z_t(class i, j), the class a 64-row tile's parity."""
    p = cs.lead_parts(m, n, S, np.random.default_rng(seed), data_sigma)
    tile = cs.nsp.BLC_TILE
    key = (np.arange(m) % 2) * 1000 + np.arange(m) // tile
    keys, first, cls = np.unique(key, return_index=True, return_inverse=True)
    C = p["z"][:, first, :].transpose(1, 2, 0)
    separable = bool(np.array_equal(C[cls], p["z"].transpose(1, 2, 0))) and not data_sigma
    g = np.random.default_rng(seed)
    tag = cs._balanced13(np.where(np.arange(m) % 2 == 0, np.arange(m) // 2 + 1, -(np.arange(m) // 2 + 1)))
    code = cs._balanced13(np.arange(n))
    gc, a1 = g.integers(-3, 4, size=(4, n)), g.integers(0, 2, size=(4, m))
    beta = (code.sum(0) + 3 * (S - 1) + 1) // 2
    lam = np.hstack([tag.T, a1.T, np.ones((m, 1), np.int64)])
    rho = np.hstack([gc.T, code.T, -beta[:, None]])
    factors = (u32(lam), u32(rho)) if np.array_equal(lam @ rho.T, p["x"]) else None
    return {"W": u32(p["words"].transpose(1, 2, 0)), "d0": u32(p["x"]), "R": u32(p["y"].T), "C": u32(C),
            "cls": cls.reshape(-1), "separable": separable, "rows_a_class": tile / 2, "stock_frac": NOISE_STOCK,
            "d0_factors": factors, "label": "noise lead" + (" random" if data_sigma else " zero")}


def gaps(honest, cheapest, stock):
    return {"gap_claimed": round(max(0.0, honest - cheapest) / stock, 6),
            "gap_share": round(max(0.0, stock - cheapest) / stock, 6)}


def sweep(parts):
    rows = []
    S = parts["R"].shape[1]
    for P in (1, 4, 8, 16):
        if parts["d0"].shape[1] % (2 * P):
            continue
        for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if T > S:
                continue
            ref = honest_nh2(parts["W"], P, T)
            honest, stock = 3 * P * T, parts["stock_frac"] * 2 * P * T
            row = {"inst": parts["label"], "layout": "nh2", "P": P, "T": T, "G": 2 * P * T, "honest": honest,
                   "stock": stock}
            if parts["separable"]:
                variants = [("", False, False), ("_d0free", True, False)]
                if parts.get("d0_factors"):
                    variants += [("_lowrank", False, True), ("_lowrank_d0free", True, True)]
                for tag, free, low in variants:
                    got, cost = attack_nh2(parts, P, T, free, low)
                    row["exact" + tag] = bool(np.array_equal(got, ref))
                    row["attack" + tag] = {k: round(v, 3) for k, v in cost.items()}
                cheapest = min(honest, *(row[f"attack{t}"]["total"] for t, free, _ in variants if not free))
                if_free = min(honest, *(row[f"attack{t}"]["total"] for t, _, _ in variants))
            else:
                cheapest = if_free = honest
            row.update(gaps(honest, cheapest, stock))
            row["if_d0_free"] = gaps(honest, if_free, stock)
            rows.append(row)
    for T in (2, 4, 8, 16, 32, 64, 128, 256):
        if T > S:
            continue
        ref = honest_nh(parts["W"], T)
        honest, stock = 1.5 * T, parts["stock_frac"] * T
        row = {"inst": parts["label"], "layout": "nh", "T": T, "G": T, "honest": honest, "stock": stock}
        if parts["separable"]:
            for free in (False, True):
                got, cost = attack_nh(parts, T, free)
                row["exact" + ("_d0free" if free else "")] = bool(np.array_equal(got, ref))
                row["attack" + ("_d0free" if free else "")] = {k: round(v, 3) for k, v in cost.items()}
            cheapest = min(honest, row["attack"]["total"])
            if_free = min(cheapest, row["attack_d0free"]["total"])
        else:
            cheapest = if_free = honest
        row.update(gaps(honest, cheapest, stock))
        row["if_d0_free"] = gaps(honest, if_free, stock)
        rows.append(row)
    return rows


def frontier(rows):
    """For each (instance, layout, P): the first G whose attack is paid credit it didn't spend (gap_share > 1%), and the
    largest G up to which the credit holds and up to which honest's sketch is the cheapest program (PairWrites), with d0
    paid and with d0 taken as salt-independent."""
    groups = {}
    for r in rows:
        key = f"{r['inst']} {r['layout']}" + (f" P{r['P']}" if r.get("P") else "")
        groups.setdefault(key, []).append(r)

    def holds_to(rs, ok):
        best = 0
        for r in sorted(rs, key=lambda r: r["G"]):
            if not ok(r):
                break
            best = r["G"]
        return best

    out = {}
    for key, rs in groups.items():
        out[key] = {
            "first_G_overcredited": min([r["G"] for r in rs if r["gap_share"] > T1], default=None),
            "credit_holds_to_G": holds_to(rs, lambda r: r["gap_share"] <= T1),
            "pairwrites_holds_to_G": holds_to(rs, lambda r: r["gap_claimed"] <= T1),
            "credit_holds_to_G_if_d0_free": holds_to(rs, lambda r: r["if_d0_free"]["gap_share"] <= T1),
            "pairwrites_holds_to_G_if_d0_free": holds_to(rs, lambda r: r["if_d0_free"]["gap_claimed"] <= T1),
        }
    return out


def rank(x):
    return int(np.linalg.matrix_rank(x.astype(np.float64)))


def mirror_ranks(cs, lanes, r0, P=8, t0_max=8, size=360, seed=3):
    """Exact ranks of NH's edge sums on mirror-π noise at zero data: nh's over (i, j) for the first and last t0 words
    (t0 even), and nh2's over (i, g) for a group of P pairs. The last word is left out, as for the quadratic sketch."""
    rng = np.random.default_rng(seed)
    s = 2 * t0_max + 2
    w = np.cumsum(cs._edge_noise(size, size, s, lanes, r0, rng), axis=0).astype(np.int64)
    out = {"lanes": lanes, "r0": r0, "size": size, "nh": {"prefix": {}, "suffix": {}}, "nh2": {"prefix": {}}}
    for t0 in range(2, t0_max + 1, 2):
        pre = w[:t0]
        suf = w[s - 1 - t0:s - 1]
        out["nh"]["prefix"][t0] = rank((pre[0::2] * pre[1::2]).sum(0))
        out["nh"]["suffix"][t0] = rank((suf[0::2] * suf[1::2]).sum(0))
    groups = size
    w2 = np.cumsum(cs._edge_noise(size, 2 * P * groups, 2 * (t0_max // 2) + 2, lanes, r0, rng), axis=0)
    for t0 in range(1, t0_max // 2 + 1):
        pre = w2[:t0].astype(np.int64)
        prod = (pre[:, :, 0::2] * pre[:, :, 1::2]).sum(0).reshape(size, groups, P).sum(2)
        out["nh2"]["prefix"][t0] = rank(prod)
    return out


def mirror_saves(edge, block, B=SKETCH_BYTES, P=8):
    """Each attack's best saving, eq a word or an output, at B byte products a rank term. nh: a middle block of L words
    from the honest word N before it is (L/2)·N² + N·ΣΔ + Σ Δ_s·Δ_{s+1}: ΣΔ, two IMADs and the rank term, 3 eq, against
    honest's 1.5L; an edge saves 1.5·t0 − rank·B/32 (with r0 > 0, the wide block separates like N at one more word, 4).
    nh2: a middle block of P pairs and L steps pays ΣΔ a cell and three IMADs a pair, 5P, against 3PL."""
    def best(ranks, size, fixed, honest_a_word, per_word):
        vals = [(honest_a_word * t - fixed - r * B / 32) / (t if per_word else 1)
                for t, r in ranks.items() if r < size]
        return max([0.0, *vals])
    nh = {side: max(best(edge["nh"][side], edge["size"], 0, 1.5, False),
                    best(block["nh"][side], block["size"], 4, 1.5, False) if edge["r0"] else 0.0)
          for side in ("prefix", "suffix")}
    nh["block_a_word"] = best(block["nh"]["prefix"], block["size"], 3, 1.5, True)
    nh2_block = max([0.0] + [(3 * P * L - 5 * P - r * B / 32) / (2 * P * L)
                             for L, r in block["nh2"]["prefix"].items() if r < block["size"]])
    nh2_edge = max([0.0] + [(3 * P * t - r * B / 32) / (2 * P * t)
                            for t, r in edge["nh2"]["prefix"].items() if r < edge["size"]])
    return {"nh": nh, "nh2": {"block_a_word": nh2_block, "edge_a_word": nh2_edge}}


def mirror_score(cs, saves, lanes, r0):
    """nh on mirror-π scored as `commit_space.score` scores the quadratic sketch: one S an output of s steps, s from
    Llama's K by MAC share, h over the lead's from 0.5 IMAD a word and the S's seal, T1 the family's base plus the edges'
    and blocks' saving over stock. Returns h at the lead's two readings."""
    shares = cs._llama_k_shares()
    h_ratio, gap = 0.0, 0.0
    for k, sh in shares.items():
        s = cs.nsp.blc_steps(k)
        h_ratio += sh * (0.5 + cs.SHA_WORD / s) / cs.SHA_WORD
        parts = (saves["nh"]["prefix"] + saves["nh"]["suffix"]) / s + saves["nh"]["block_a_word"]
        gap += sh * min(1.5, parts) / cs.STOCK_EQ
    quad = sum(sh * (1 + cs.SHA_WORD / cs.nsp.blc_steps(k)) / cs.SHA_WORD for k, sh in shares.items())
    return {"lanes": lanes, "r0": r0, "h_over_lead": round(h_ratio, 6), "h": [round(h_ratio * h, 3) for h in cs.H_LEAD],
            "quad_h": [round(quad * h, 3) for h in cs.H_LEAD], "t1_sketch": round(gap, 6),
            "t1": round(gap + 0.0062, 6), "steps": {k: cs.nsp.blc_steps(k) for k in shares}}


def main() -> int:
    checks = Checks()
    opts = dict(a[2:].split("=", 1) for a in checks.args if a.startswith("--") and "=" in a)
    args = [a for a in checks.args if not a.startswith("--")]
    out_path = args[0] if args else "nh_attack.json"
    sys.path.insert(0, os.path.abspath(opts.get("lib", ".")))
    import commit_space as cs
    import hash_counter as hc

    S = int(opts.get("steps", 256))
    insts = [counter_parts(hc, c, "zero", S) for c in ("lead:rot", "lead:e7", "lead:w127", "lead:e4", "sep")]
    insts += [counter_parts(hc, "lead:rot", "zero", S, chosen=r) for r in (1, 4)]
    insts += [counter_parts(hc, "lead:rot", "random", S), noise_parts(cs, S), noise_parts(cs, S, data_sigma=25.0)]
    for p in insts:
        want = "random" not in p["label"]
        checks.check(f"{p['label']}: the split reproduces the words", p["separable"] == want,
                     f"separable {p['separable']}, classes {len(p['C'])}")
    rows = [r for p in insts for r in sweep(p)]
    checks.check("noise lead: X is λ·ρᵀ at rank 9 (tag, a₁ and bias against g, code and β)",
                 insts[-2]["d0_factors"] is not None)
    attacked = [r for r in rows if "exact" in r]
    checks.check("every decomposition matches honest's S bit for bit",
                 all(v for r in attacked for k, v in r.items() if k.startswith("exact")),
                 f"{sum(k.startswith('exact') for r in attacked for k in r)} (instance, layout, G, variant)")
    rand = [r for r in rows if "random" in r["inst"]]
    checks.check("random data: no decomposition runs, honest is cheapest", all(r["gap_claimed"] == 0 for r in rand))
    lead = [r for r in rows if r["inst"] in ("counter lead:rot zero", "noise lead zero")]
    summary = frontier(rows)
    checks.check("on the lead's zero-data words NH at G 1,024 credits work no program spent (gap_share > 1%)",
                 all(r["gap_share"] > T1 for r in lead if r["G"] == 1024),
                 {k: v["first_G_overcredited"] for k, v in summary.items() if k.startswith(("counter lead:rot zero",
                                                                                            "noise lead zero"))})

    mirror = {}
    if "--no-mirror" not in checks.args:
        for lanes in (3, 4):
            block = mirror_ranks(cs, lanes, 0)
            for r0 in (0, 4):
                edge = block if r0 == 0 else mirror_ranks(cs, lanes, r0)
                sv = mirror_saves(edge, block)
                mirror[f"l{lanes}_r{r0}"] = {"edge": edge, "saves": sv, "score": mirror_score(cs, sv, lanes, r0)}
                print(f"mirror-pi l{lanes} r0={r0}: nh {sv['nh']}  nh2 {sv['nh2']}  {mirror[f'l{lanes}_r{r0}']['score']}",
                      flush=True)
        checks.check("mirror-π: nh2's group sum decomposes (a middle block saves > 1% of stock a word)",
                     all(v["saves"]["nh2"]["block_a_word"] / NOISE_STOCK > T1 for v in mirror.values()),
                     {k: round(v["saves"]["nh2"]["block_a_word"] / NOISE_STOCK, 4) for k, v in mirror.items()})

    if checks.mutations:
        p = counter_parts(hc, "lead:rot", "zero", 64)
        ref = honest_nh2(p["W"], 8, 16)
        bad = dict(p, C=p["C"].copy())
        bad["C"][1] = bad["C"][0]
        checks.mutation("nh2 decomposition with both row classes given class 0's columns",
                        not np.array_equal(attack_nh2(bad, 8, 16)[0], ref))
        bad = dict(p, d0=p["d0"] ^ np.uint64(1))
        checks.mutation("nh decomposition with every d0 off in bit 0",
                        not np.array_equal(attack_nh(bad, 16)[0], honest_nh(p["W"], 16)))
        q = noise_parts(cs, 64)
        lam, rho = q["d0_factors"]
        rho = rho.copy()
        rho[:, -1] += np.uint64(1)
        checks.mutation("nh2 low-rank d0 terms with β off by one",
                        not np.array_equal(attack_nh2(dict(q, d0_factors=(lam, rho)), 8, 16, lowrank=True)[0],
                                           honest_nh2(q["W"], 8, 16)))
        x, _ = imma(p["R"][:, :16], p["C"][0][:, :16])
        full = (p["R"][:, None, :16] * p["C"][0][None, :, :16]).sum(2) & np.uint64(M32)
        checks.check("imma's digit chains equal the dot product mod 2³²", np.array_equal(x, full))
        big = u32(np.random.default_rng(9).integers(0, 1 << 32, (8, 40)))
        y, _ = imma(big, big)
        checks.check("imma on uniform 32-bit operands", np.array_equal(y, (big[:, None, :] * big[None, :, :]).sum(2)
                                                                     & np.uint64(M32)))
    json.dump({"schema": "nh-attack/v2", "steps": S, "rows": rows, "frontier": summary, "mirror": mirror},
              open(out_path, "w"), indent=1, default=str)
    for r in rows:
        a = min([v["total"] for k, v in r.items() if k.startswith("attack") and "d0free" not in k], default=None)
        print(f"{r['inst']:28s} {r['layout']:3s} P{r.get('P', '-'):>2} T{r['T']:>4} G{r['G']:>5} honest {r['honest']:8.1f} "
              f"stock {r['stock']:8.1f} attack {a if a is not None else '-':>8} claimed {r['gap_claimed']:.3f} "
              f"share {r['gap_share']:.3f}")
    for k, v in summary.items():
        print(f"frontier {k:36s} {v}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
