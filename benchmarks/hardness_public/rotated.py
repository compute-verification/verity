"""HT-PUBLIC's split search (`natural.py`, `tiles.py`) and HT-PROV's low-rank + clipped-residual projections
(`benchmarks/hardness_prov/routes.py`) on design U's codes: each linear's input rotated by a random-sign Hadamard matrix
(`hardness_useful/rotation.py`) and each row clipped so the matrix's Strassen level-1 forms overflow int8 at 2%
(HT-USEFUL). Prices are HT-PREMISE's (`pouw/theory/hardness_premise/natural.py`): W1 per MAC of E·Wᵀ, E on a = 4 values,
s8 1, b1 0.125, nvf4 0.5, a register write 32, the fast reading; the honest leaf is 1 + 32/K. A correction price in
--alus is the writes one corrected entry costs per noise row (HT-PROTO-U's floors 1.48 and 1.78, and 0.84).

The codes come from an exported file (--codes, `hardness_useful/qat.py`'s NAME.q) or from Hugging Face weights rotated
and clipped here (--hf, `hardness_useful/price.py`'s np_calibrate on rotation.rotate(W)).

A split is q = P + S, P a cheap part and S = rint(q − P) its residual, which takes natural.py's cheapest route whose
root isn't the plain leaf (the depth-4 Strassen/Winograd DP with its s8 leaves and gather, ALU, b1 and FP4
corrections, or the plane routes) or a 2:4 pass with the rest on the ALU. The plain leaf on S is excluded as HT-USEFUL's
`alt` excludes it on q: S overflows int8 wherever q's leaf doesn't, so P plus S's leaf costs at least the honest leaf
plus P, the honest kernel with a detour, which tends to it as P → 0.

- `none`: P = 0, HT-USEFUL's `cheapest`, the reference;
- `svd`: P the real rank-r truncated SVD, priced r/n + r/K (`lr1`, HT-PUBLIC's: one int8 pass a factor) or
  r/n + 3r/K (`lr3`, HT-PROV's: the s32 intermediate in three int8 passes);
- `isvd`: the exact one: P = A·Bᵀ with A (n×r) and B (K×r) int8, B = rint(β·V) with β_j = √(σ_j·√(K/n)) (which
  balances the two roundings' errors) and A the rounded least-squares fit to q given B, so S = q − A·Bᵀ exactly. E·Bᵀ
  is an s8 pass (r/n); its output T, with E − 2 on [−2, 1], needs d int8 digits for 2·Σ|B_j| (d ≥ 2 once that exceeds
  127), each an s8 pass against Aᵀ (d·r/K), and each of their d accumulators joins the output in one write (a shift
  and an add, as natural.py's 256·(A·hi) correction);
- `rpca`: P a rank-r fit refit twice with the entries whose residual exceeds a masked to P;
- `tiles`: HT-PUBLIC's per-tile P = 0 routes, each tile its cheapest of b1 planes (H200) or FP4 digits (sm_120) on
  [−a, a] with the entries beyond a on the ALU, plus the tile's output writes (generous: the noise rows' loads are not
  charged). Tiles that keep the s8 leaf are the honest kernel's, so the family's price is the mix with s8 tiles when
  that is below the leaf (a saving), else every tile off s8.

`exact` is the best of none, isvd and tiles; `generous` adds svd and rpca at lr1, whose real P makes no exact route
(E·rint(q − P)ᵀ misses E·(q − P − rint(q − P))ᵀ), so they bound their family from the cheater's side. Two projections,
which are not routes either:

- `prov`: HT-PROV's `routes.residual` on each svd residual at the correction price (an entry beyond a costs 32·w), its
  dense range a < 127 (a = 127 is the plain leaf), plus lr3, in its units (dense 1, no output write), compared with
  the leaf as (p + 32/K)/(1 + 32/K);
- `floor`: the Eckart–Young no-outlier projection. A residual within ±a has RMS ≤ a + 0.5, so P needs rank ≥ r_min(a),
  the least r whose SVD tail RMS is at most a + 0.5; the projection prices r_min(a) at lr1 plus the cheapest route for
  a residual within ±a with no correction: the depth-d tree on a form that never overflows, 2^d·a ≤ 127 (Strassen's
  bound; Winograd's forms can exceed it), or b1 or FP4 planes that hold ±a. r = 0 counts only if q itself is within
  ±a. The RMS condition bites only where a is well under q's RMS, so the summary's floor takes a ≤ FLOOR_MAX; above
  it the measured residuals' share beyond ±a decides.

On a crop (n 256, K 512), the trees chosen for q and for each exact winner's S are executed exactly against E·Sᵀ, and
for isvd the whole split against E·qᵀ.

The defences of design U against the split (HT-DEFENCE) are priced here too:
- `--target` a list: each matrix at each clip target, one output directory per target;
- `--rot`: the input-side rotation, design U's (`hadamard`), a second independent one after it (`hadamard2`), a dense
  Haar-random orthogonal matrix (`haar`, every rotation's limit) or design U's with a random-sign Hadamard on the
  output side as well (`both`);
- `--live-noise f`: live full-rank weight noise F uniform on [−f, f], served (x + E)·(q + F)ᵀ: q + F's own routes, and
  the split of q committed before F with its residual S + F;
- `--kcap`: K cut into natural.py's equal chunks of at most kcap, each its own product with its output committed, so
  the leaf is 1 + 32·chunks/K and each chunk takes its own route (none or its own split);
- `--orders`: every residual, and q, priced in HT-USEFUL's row and column orders too (`price.permute`), the least kept;
- `--fast`: the exact routes only (none, isvd, tiles), without the svd, rpca, prov and floor bounds.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pouw" / "theory" / "hardness_premise"))
import natural as nat  # noqa: E402  (HT-PREMISE's, which shadows this directory's natural.py)

sys.path.insert(0, str(HERE.parent / "hardness_useful"))
import price  # noqa: E402
import rotation  # noqa: E402

sys.path.insert(0, str(HERE.parent / "hardness_prov"))
import routes  # noqa: E402

CHIPS = ("h200", "sm120")
ALUS = "0.84=8:0.84 1.48=8:1.48 1.78=8:1.78"
A_NOISE = 4
DEPTH = 4
RPCA = ((32, 31), (32, 15), ("m/8", 31), ("m/8", 15))
FLOOR_A = (1, 2, 3, 4, 7, 15, 31, 63)
FLOOR_MAX = 15
TILES = ((8, 256), (16, 64), (128, 128), (1, None))
TILE_A = (1, 2, 3, 4, 7, 15, 27, 31, 63)


def keys(alus):
    return [f"{chip}@{aname}" for aname in alus for chip in CHIPS]


def residual_routes(S: np.ndarray, alus: dict, orders=("none",)) -> dict:
    """W1 per MAC of the cheapest route for E·Sᵀ whose root isn't the plain leaf, per chip@price: natural.py's tree,
    its plane routes, or 2:4, in each of `orders` (`price.permute`; the attacker reorders n and K freely)."""
    n, K = S.shape
    out = {}
    for order in orders:
        Sp = price.permute(S, order)
        C = Sp.T
        st, ps = nat.forms(C, 0, DEPTH), nat.plane_stats(C)
        nz24 = sparse24_stats(Sp)
        for aname, alu in alus.items():
            w = min(alu.values())
            for chip in CHIPS:
                N = nat.Natural(chip, "fast", A_NOISE, alu)
                N.node(st, (0, A_NOISE - 1), 0)
                cands = {"alt": (N.root_alt[0] + 32 * n) / (n * K)}
                cands.update({k: (v + 32 * n) / (n * K) for k, v in N.planes(ps, K, n).items()})
                cands["2:4"] = sparse24_cost(nz24, chip, w, K)
                how = min(cands, key=cands.get)
                k = f"{chip}@{aname}"
                if k not in out or cands[how] < out[k]["cost"]:
                    out[k] = {"cost": cands[how], "how": how if order == "none" else f"{order}:{how}"}
        del Sp, C, st
    return out


def sparse24_stats(S: np.ndarray) -> dict | None:
    n, K = S.shape
    if K % 4:
        return None
    ab = np.abs(S)
    out = {}
    for a in (1, 3, 7, 127):
        small_nz = (ab <= a) & (S != 0)
        cnt = small_nz.reshape(n, K // 4, 4).sum(axis=2)
        out[a] = (float(np.maximum(cnt - 2, 0).sum()) / S.size, float((ab > a).mean()))
    return out


def sparse24_cost(st: dict | None, chip: str, w: float, K: int) -> float:
    """2:4 on S's entries within ±a (s8 at half price; on sm_120 FP4 at a quarter for a ≤ 3), the groups' excess and the
    entries beyond a on the ALU at 32·w each, one output write."""
    if st is None:
        return math.inf
    best = math.inf
    for a, (excess, beyond) in st.items():
        mma = 0.25 if chip == "sm120" and a <= 3 else 0.5
        best = min(best, mma + 32 * w * (excess + beyond) + 32 / K)
    return best


def lr1(r, n, K):
    return r / n + r / K


def integer_split(q: np.ndarray, s: np.ndarray, Vt: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(S, A, B, d): q = S + A·Bᵀ exactly, A and B int8, d the int8 digits of T = (E − 2)·B."""
    n, K = q.shape
    beta = np.sqrt(np.maximum(s[:r].astype(np.float64), 1e-9) * math.sqrt(K / n))
    B = np.clip(np.rint(Vt[:r].T.astype(np.float64) * beta), -127, 127)
    At = np.linalg.lstsq(B, q.T.astype(np.float64), rcond=None)[0]
    A = np.clip(np.rint(At.T), -127, 127).astype(np.int64)
    B = B.astype(np.int64)
    S = (q.astype(np.int64) - A @ B.T).astype(np.int32)
    width = 2 * int(np.abs(B).sum(axis=0).max())
    d = 1 if width <= 127 else 2 if width <= 32639 else 3
    return S, A, B, d


def lrx(r, n, K, d):
    """E·B and its d digits' products with Aᵀ in s8, T's r·d digit writes and the d accumulators' writes."""
    return r / n + d * r / K + 32 * d * r / (n * K) + 32 * d / K


def crop_exact_split(S: np.ndarray, A: np.ndarray, B: np.ndarray, alus: dict) -> bool:
    """On a crop, E·qᵀ = run(tree of S) + (E·B)·Aᵀ for each chip and price's chosen trees."""
    n, K = min(256, S.shape[0]), min(512, S.shape[1])
    Sc, Ac, Bc = np.ascontiguousarray(S[:n, :K]), A[:n], B[:K]
    qc = Sc.astype(np.int64) + Ac @ Bc.T
    E = np.random.default_rng(0).integers(0, A_NOISE, (32, K), dtype=np.int64)
    ok = True
    for alu in alus.values():
        for chip in CHIPS:
            N = nat.Natural(chip, "fast", A_NOISE, alu)
            N.node(nat.forms(Sc.T, 0, DEPTH), (0, A_NOISE - 1), 0)
            tree = N.root_alt[1]
            if tree is not None:
                ok &= bool(np.array_equal(nat.run(tree, E, Sc.T.astype(np.int64)) + (E @ Bc) @ Ac.T, E @ qc.T))
    return bool(ok)


def rsvd(M: np.ndarray, r: int, seed: int = 0, iters: int = 2) -> np.ndarray:
    """A rank-r approximation of M by randomized SVD (Halko et al., r + 10 columns, `iters` power iterations)."""
    rng = np.random.default_rng(seed)
    Y = M @ rng.standard_normal((M.shape[1], r + 10)).astype(M.dtype)
    for _ in range(iters):
        Y, _ = np.linalg.qr(Y)
        Y = M @ (M.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ M
    u, s, vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ (u[:, :r] * s[:r])) @ vt[:r]


def residual_stats(S: np.ndarray) -> dict:
    ab = np.abs(S)
    return {"sigma": float(S.astype(np.float64).std()), "absmax": int(ab.max()),
            "gt15": float((ab > 15).mean()), "gt31": float((ab > 31).mean()), "gt63": float((ab > 63).mean()),
            "gt127": float((ab > 127).mean())}


@functools.cache
def zero_tree_costs(n: int, K: int) -> dict:
    """The cheapest no-correction tree at each depth d ≤ DEPTH (its forms never overflow), W1 per MAC with the root
    write: chip- and price-independent, since without corrections every leaf is an s8 MMA."""
    out = {0: 1 + 32 / K}
    for d in range(1, DEPTH + 1):
        st = nat.forms(np.zeros((K, n), dtype=np.int16), 0, d)
        N = nat.Natural("h200", "fast", A_NOISE, 1.0)
        cost, _t, _a, _b = N.node(st, (0, A_NOISE - 1), 0)
        out[d] = min(out[d - 1], (cost + 32 * n) / (n * K))
    return out


def plane_nocorr(n: int, K: int) -> dict:
    """Per chip, the plane routes' cost with no correction (ps all None), by route name."""
    ps = {f"b1_s{s}": None for s in range(1, 8)} | {"nvf4_d1": None, "nvf4_d2": None}
    out = {}
    for chip in CHIPS:
        N = nat.Natural(chip, "fast", A_NOISE, 1.0)
        out[chip] = {k: (v + 32 * n) / (n * K) for k, v in N.planes(ps, K, n).items()}
    return out


def floor_route(a: int, chip: str, trees: dict, planes: dict) -> float:
    d = max((d for d in trees if (1 << d) * a <= 127), default=0)
    best = trees[d]
    s = math.ceil(math.log2(2 * a + 1))
    if f"b1_s{s}" in planes[chip]:
        best = min(best, planes[chip][f"b1_s{s}"])
    if a <= 3 and "nvf4_d1" in planes[chip]:
        best = min(best, planes[chip]["nvf4_d1"])
    elif a <= 27 and "nvf4_d2" in planes[chip]:
        best = min(best, planes[chip]["nvf4_d2"])
    return best


def prov_projection(S: np.ndarray, chip: str, w: float) -> tuple[float, int]:
    """routes.residual with an entry beyond the dense range corrected at 32·w W1 instead of IDP4A's 8."""
    ab = np.minimum(np.abs(S), 255).astype(np.int64).ravel()
    gt = 1.0 - np.cumsum(np.bincount(ab, minlength=256)) / ab.size
    alu = 32 * w
    best, arg = math.inf, None
    for a in routes.ALPHA[:-1]:
        c = routes.small(chip, a) + gt[a] * alu
        if c < best:
            best, arg = c, a
    return float(best), int(arg)


def tile_routes(q: np.ndarray, alus: dict) -> dict:
    n, K = q.shape
    out = {}
    pE = 2
    for tr, tk in TILES:
        tk = tk or K
        nt, kt = n // tr, K // tk
        ab = np.abs(q[: nt * tr, : kt * tk].astype(np.int32)).reshape(nt, tr, kt, tk)
        beyond = {a: (ab > a).mean(axis=(1, 3)) for a in TILE_A}
        for aname, alu in alus.items():
            w = min(alu.values())
            for chip in CHIPS:
                best = np.full((nt, kt), math.inf)
                for a in TILE_A:
                    dense = []
                    if chip == "h200":
                        s = math.ceil(math.log2(2 * a + 1))
                        dense.append(0.125 * pE * s + 32 * (pE * s + 1) / tk)
                    else:
                        d = 1 if a <= 3 else 2 if a <= 27 else None
                        if d:
                            dense.append(0.5 * d + 32 * (3 * d - 1) / tk)
                    if dense:
                        best = np.minimum(best, min(dense) + 32 * w * beyond[a])
                mix = float(np.minimum(best, 1.0).mean()) + 32 / K
                cost = mix if mix < 1 + 32 / K else float(best.mean()) + 32 / K
                out[f"{tr}x{tk if tk != K else 'K'}/{chip}@{aname}"] = cost
    return out


def crop_exact(S: np.ndarray, alus: dict) -> bool:
    n, K = min(256, S.shape[0]), min(512, S.shape[1])
    Sc = np.ascontiguousarray(S[:n, :K])
    ok = True
    for aname, alu in alus.items():
        for chip in CHIPS:
            N = nat.Natural(chip, "fast", A_NOISE, alu)
            _c, tree, _ac, _at = N.node(nat.forms(Sc.T, 0, DEPTH), (0, A_NOISE - 1), 0)
            ok &= price.run_tree(tree, Sc, A_NOISE, 32)
            if N.root_alt[1] is not None:
                ok &= price.run_tree(N.root_alt[1], Sc, A_NOISE, 32)
    return bool(ok)


def block_hadamard(x: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """x·D·H/√B as `rotation.rotate`, with the signs sg."""
    n, k = x.shape
    m, a, b = rotation.factors(k)
    return (rotation.hadamard(a) @ (x * sg).reshape(n, m, a, b) @ rotation.hadamard(b)).reshape(n, k)


def rotate_variant(w: np.ndarray, how: str) -> np.ndarray:
    """W rotated on its input side as `how` (the module docstring's --rot); every variant is orthogonal."""
    n, k = w.shape
    if how == "hadamard":
        return rotation.rotate(w)
    if how == "hadamard2":
        x = rotation.rotate(w)[:, np.random.default_rng(k + 1).permutation(k)]
        return block_hadamard(x, np.random.default_rng(k + 2).integers(0, 2, k) * 2.0 - 1)
    if how == "haar":
        Q, R = np.linalg.qr(np.random.default_rng(k + 3).standard_normal((k, k)))
        return w.astype(np.float64) @ (Q * np.sign(np.diag(R)))
    assert how == "both", how
    return rotation.rotate(rotation.rotate(w).T).T


def load_codes(src) -> tuple[np.ndarray, float | None]:
    """The codes q of a job; with deflate R > 0, of W·R less its top-R SVD part (served apart as its own product)."""
    if src[0] == "codes":
        from safetensors import safe_open

        with safe_open(src[1], "numpy") as f:
            return f.get_tensor(src[2]).astype(np.int16), None
    _kind, url, base, entry, target, deflate, rot = src
    w = rotate_variant(nat.load_tensor(url, base, entry), rot)
    if deflate:
        U, s, Vt = np.linalg.svd(w, full_matrices=False)
        w = w - (U[:, :deflate] * s[:deflate]) @ Vt[:deflate]
    c, q = price.np_calibrate(w, target)
    return q, c


def chunking(K: int, kcap: int) -> int:
    """natural.price's chunk count: ⌈K/kcap⌉, raised until the chunks are equal and split to DEPTH."""
    chunks = math.ceil(K / kcap)
    while chunks > 1 and (K % chunks or (K // chunks) % 2**DEPTH):
        chunks += 1
    return chunks


def work(job) -> dict:
    name, src, alus, ranks, opts = job
    t0 = time.time()
    q, c = load_codes(src)
    n, K = q.shape
    if opts.get("kcap") and chunking(K, opts["kcap"]) > 1:
        return work_chunked(name, src, q, c, alus, ranks, opts, t0)
    fast, orders = opts.get("fast", False), opts.get("orders", ("none",))
    m = min(n, K)
    leaf = 1 + 32 / K
    rec = {"tensor": name, "shape": [n, K], "macs_per_token": int(q.size), "c": c, "leaf": leaf,
           "target": src[4] if src[0] == "hf" else None, "rot": src[6] if src[0] == "hf" else None,
           "sigma_q": float(q.astype(np.float64).std()), "overflow1": nat.overflow_profile(q, 1)["strassen"][0],
           "saturated": float((np.abs(q) == 127).mean())}
    rec["none"] = residual_routes(q, alus, orders)
    noise = opts.get("live_noise", 0)
    if noise:
        qn = np.clip(q + np.random.default_rng(7).integers(-noise, noise + 1, q.shape), -127, 127).astype(np.int16)
        rec["noise"] = {"f": noise, "none": residual_routes(qn, alus, orders), "isvd": {}}
    if opts.get("nosplit"):
        rec |= {"svd": {}, "prov": {}, "isvd": {}, "rpca": {}, "floor": {}, "tiles": tile_routes(q, alus)}
        rec["best"] = {k: score(rec, k) for k in keys(alus)}
        rec["exact_crop"] = {"none": crop_exact(q, alus)}
        rec["seconds"] = time.time() - t0
        return rec
    qf = q.astype(np.float32)
    U, s, Vt = np.linalg.svd(qf, full_matrices=False)
    s2 = s.astype(np.float64) ** 2
    tail = np.concatenate([np.cumsum(s2[::-1])[::-1], [0.0]])
    rec["energy_top"] = {str(r): float(1 - tail[r] / tail[0]) for r in (1, 8, 32, 128, m // 8, m // 4, m // 2)}
    tail_rms = np.sqrt(tail / (n * K))
    rec["tail_rms"] = {str(r): float(tail_rms[r]) for r in (0, 1, 8, 32, 128, m // 8, m // 4, m // 2)}
    budget = 1.05 * leaf
    rec["svd"], rec["prov"], rec["isvd"] = {}, {}, {}
    for r in [r for r in ranks if 0 < r < m] if ranks else nat_ranks(m):
        if lr1(r, n, K) >= budget:
            continue
        if not fast:
            L = (U[:, :r] * s[:r]) @ Vt[:r]
            S = np.rint(qf - L).astype(np.int32)
            del L
            rr = residual_routes(S, alus, orders)
            rec["svd"][str(r)] = {"lr1": lr1(r, n, K), "lr3": routes.lowrank(r, n, K), **residual_stats(S),
                                  "routes": rr}
            rec["prov"][str(r)] = {}
            for aname, alu in alus.items():
                w = min(alu.values())
                for chip in CHIPS:
                    p, a = prov_projection(S, chip, w)
                    tot = routes.lowrank(r, n, K) + p
                    rec["prov"][str(r)][f"{chip}@{aname}"] = {"p": tot, "a": a, "over_leaf": (tot + 32 / K) / leaf}
            del S
        S, A, B, d = integer_split(q, s, Vt, r)
        if lrx(r, n, K, d) < budget:
            rec["isvd"][str(r)] = {"d": d, "Amax": int(np.abs(A).max()), "Bmax": int(np.abs(B).max()),
                                   **residual_stats(S), "routes": residual_routes(S, alus, orders)}
            if noise:
                rec["noise"]["isvd"][str(r)] = {"d": d, "routes": residual_routes(S + (qn - q), alus, orders)}
        del S, A, B
    rec["rpca"] = {}
    for r_spec, a in () if fast else RPCA:
        r = m // 8 if r_spec == "m/8" else r_spec
        if r >= m or lr1(r, n, K) >= budget:
            continue
        L = (U[:, :r] * s[:r]) @ Vt[:r]
        for it in range(2):
            out = np.abs(np.rint(qf - L)) > a
            L = rsvd(np.where(out, L, qf), r, seed=it)
        S = np.rint(qf - L).astype(np.int32)
        del L
        rec["rpca"][f"{r}/{a}"] = {"lr1": lr1(r, n, K), **residual_stats(S), "routes": residual_routes(S, alus)}
        del S
    del U
    rec["floor"] = {}
    if not fast:
        trees, planes = zero_tree_costs(n, K), plane_nocorr(n, K)
        rec["zero_trees"] = {str(d): v for d, v in trees.items()}
        qmax = int(np.abs(q).max())
        for a in FLOOR_A:
            ok = np.nonzero(tail_rms <= a + 0.5)[0]
            rmin = max(int(ok[0]) if ok.size else m, int(qmax > a))
            rec["floor"][str(a)] = {"r_min": rmin, **{chip: lr1(rmin, n, K) + floor_route(a, chip, trees, planes)
                                                      for chip in CHIPS}}
    rec["tiles"] = tile_routes(q, alus)
    rec["best"] = {k: score(rec, k) for k in keys(alus)}
    if noise:
        rec["noise"]["best"] = {k: noise_score(rec, k) for k in keys(alus)}
    checked = {"none": crop_exact(q, alus)}
    for k, b in rec["best"].items():
        tag = b["route"]
        if tag.startswith("isvd/") and tag not in checked:
            S, A, B, _d = integer_split(q, s, Vt, int(tag.split("/")[1]))
            checked[tag] = crop_exact(S, alus) and crop_exact_split(S, A, B, alus)
        if tag.startswith("isvd/") and b["over_leaf"] < 1.05 and f"full/{tag}/{k}" not in rec:
            S, A, B, _d = integer_split(q, s, Vt, int(tag.split("/")[1]))
            rec[f"full/{tag}/{k}"] = full_exact_split(S, A, B, k, alus)
    rec["exact_crop"] = checked
    rec["seconds"] = time.time() - t0
    return rec


def score(rec: dict, k: str) -> dict:
    """The best exact and generous routes for chip@price k, over the leaf, from the record's parts."""
    n, K = rec["shape"]
    leaf, chip = rec["leaf"], k.split("@")[0]
    exact = {"none": rec["none"][k]["cost"]}
    exact |= {f"isvd/{r}": lrx(int(r), n, K, v["d"]) + v["routes"][k]["cost"] for r, v in rec["isvd"].items()}
    t = {kk.split("/")[0]: v for kk, v in rec["tiles"].items() if kk.endswith(f"/{k}")}
    exact |= {f"tiles/{s}": v for s, v in t.items()}
    gen = dict(exact)
    gen |= {f"svd/{r}": v["lr1"] + v["routes"][k]["cost"] for r, v in rec["svd"].items()}
    gen |= {f"rpca/{r}": v["lr1"] + v["routes"][k]["cost"] for r, v in rec["rpca"].items()}
    lr3 = dict(exact)
    lr3 |= {f"svd/{r}": v["lr3"] + v["routes"][k]["cost"] for r, v in rec["svd"].items()}
    route, groute = min(exact, key=exact.get), min(gen, key=gen.get)
    prov = min((v[k]["over_leaf"] for v in rec["prov"].values()), default=math.inf)
    floor = min((v[chip] for a, v in rec["floor"].items() if int(a) <= FLOOR_MAX), default=math.inf) / leaf
    return {"cost": exact[route], "over_leaf": exact[route] / leaf, "route": route,
            "none_over_leaf": exact["none"] / leaf, "generous_over_leaf": gen[groute] / leaf,
            "generous_route": groute, "lr3_over_leaf": min(lr3.values()) / leaf, "prov_over_leaf": prov,
            "floor_over_leaf": floor}


def noise_score(rec: dict, k: str) -> dict:
    """With live weight noise F: the honest leaf against q + F's own route and the split of q (committed before F)
    with its residual S + F."""
    n, K = rec["shape"]
    nz = rec["noise"]
    cands = {"none": nz["none"][k]["cost"]}
    cands |= {f"isvd/{r}": lrx(int(r), n, K, v["d"]) + v["routes"][k]["cost"] for r, v in nz["isvd"].items()}
    route = min(cands, key=cands.get)
    return {"over_leaf": cands[route] / rec["leaf"], "route": route, "none_over_leaf": cands["none"] / rec["leaf"]}


def work_chunked(name, src, q, c, alus, ranks, opts, t0) -> dict:
    """K in natural.price's chunks, each its own product with its own output words: per chunk, its cheapest of its
    none route and its own exact splits; the leaf is 1 + 32·chunks/K."""
    n, K = q.shape
    chunks = chunking(K, opts["kcap"])
    kc = K // chunks
    leaf = 1 + 32 * chunks / K
    orders = opts.get("orders", ("none",))
    rec = {"tensor": name, "shape": [n, K], "macs_per_token": int(q.size), "c": c, "leaf": leaf, "chunked": True,
           "chunks": chunks, "K_chunk": kc, "target": src[4] if src[0] == "hf" else None,
           "rot": src[6] if src[0] == "hf" else None, "sigma_q": float(q.astype(np.float64).std()),
           "overflow1": nat.overflow_profile(q, 1)["strassen"][0], "per_chunk": []}
    checked = {}
    for i in range(chunks):
        qc = np.ascontiguousarray(q[:, i * kc:(i + 1) * kc])
        part = {"none": residual_routes(qc, alus, orders), "isvd": {}}
        _U, s, Vt = np.linalg.svd(qc.astype(np.float32), full_matrices=False)
        del _U
        s2 = s.astype(np.float64) ** 2
        part["energy_top32"] = float(s2[:32].sum() / s2.sum())
        for r in [r for r in ranks if 0 < r < min(n, kc)] if ranks else nat_ranks(min(n, kc)):
            S, A, B, d = integer_split(qc, s, Vt, r)
            if lrx(r, n, kc, d) < 1.05 * leaf:
                part["isvd"][str(r)] = {"d": d, **residual_stats(S), "routes": residual_routes(S, alus, orders)}
            if i == 0 and f"isvd/{r}" not in checked:
                checked[f"isvd/{r}"] = crop_exact(S, alus) and crop_exact_split(S, A, B, alus)
            del S, A, B
        if i == 0:
            checked["none"] = crop_exact(qc, alus)
        rec["per_chunk"].append(part)
    rec["best"] = {}
    for k in keys(alus):
        none = exact = 0.0
        routes_k = []
        for part in rec["per_chunk"]:
            cands = {"none": part["none"][k]["cost"]}
            cands |= {f"isvd/{r}": lrx(int(r), n, kc, v["d"]) + v["routes"][k]["cost"] for r, v in part["isvd"].items()}
            route = min(cands, key=cands.get)
            none += cands["none"] / chunks
            exact += cands[route] / chunks
            routes_k.append(route)
        rec["best"][k] = {"cost": exact, "over_leaf": exact / leaf, "route": "+".join(routes_k),
                          "none_over_leaf": none / leaf, "generous_over_leaf": exact / leaf,
                          "generous_route": "+".join(routes_k), "lr3_over_leaf": exact / leaf,
                          "prov_over_leaf": math.inf, "floor_over_leaf": math.inf}
    rec["exact_crop"] = checked
    rec["seconds"] = time.time() - t0
    return rec


def full_exact_split(S: np.ndarray, A: np.ndarray, B: np.ndarray, k: str, alus: dict) -> bool:
    """The served-size alt tree of S at chip@price k plus (E·B)·Aᵀ, against E·qᵀ for 2^DEPTH noise rows."""
    chip, aname = k.split("@")
    N = nat.Natural(chip, "fast", A_NOISE, alus[aname])
    N.node(nat.forms(S.T, 0, DEPTH), (0, A_NOISE - 1), 0)
    E = np.random.default_rng(1).integers(0, A_NOISE, (2**DEPTH, S.shape[1]), dtype=np.int64)
    C = S.T.astype(np.int64)
    return bool(np.array_equal(nat.run(N.root_alt[1], E, C) + (E @ B) @ A.T, E @ (C + B @ A.T)))


def nat_ranks(m: int) -> list[int]:
    return sorted({r for r in (1, 8, 32, 128, m // 16, m // 8, m // 4, m // 2) if 0 < r < m})


def code_jobs(path: Path, layers):
    from safetensors import safe_open

    with safe_open(str(path), "numpy") as f:
        shapes = {k[: -len(".q")] + ".weight": f.get_slice(k).get_shape() for k in f.keys() if k.endswith(".q")}
    return [(t, ("codes", str(path), t[: -len(".weight")] + ".q", shapes[t])) for t in sorted(shapes)
            if keep(t, layers)]


def load_shape(src) -> list[int]:
    return src[3] if src[0] == "codes" else src[3]["shape"]


def hf_jobs(repo: str, target: float, layers, deflate: int = 0, rot: str = "hadamard"):
    return [(t, (src[0], src[1], src[2], src[3], target, deflate, rot)) for t, src in price.hf_jobs(repo, target, True)
            if keep(t, layers)]


def keep(t: str, layers, match=None) -> bool:
    return ((layers is None or any(t.startswith(f"model.layers.{i}.") for i in layers))
            and (match is None or any(s in t for s in match)))


def summarize(path: Path) -> dict:
    """Per chip@price, each column MAC-weighted, its least, and the matrices under 1.05 of the leaf; every record
    re-scored by `score`."""
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    for r in recs:
        if not r.get("chunked"):
            r["best"] = {k: score(r, k) for k in r["best"]}
    macs = sum(r["macs_per_token"] for r in recs)
    out = {"matrices": len(recs), "macs_per_token": macs, "by_price": {},
           "exact": all(all(r["exact_crop"].values()) for r in recs),
           "full_checks": {f"{r['tensor']}|{k[5:]}": v for r in recs for k, v in r.items() if k.startswith("full/")},
           "strassen1": sorted(r["overflow1"] for r in recs)[len(recs) // 2] if recs else None}
    cols = {"none": "none_over_leaf", "exact": "over_leaf", "generous": "generous_over_leaf",
            "lr3": "lr3_over_leaf", "prov": "prov_over_leaf", "floor": "floor_over_leaf"}
    for k in recs[0]["best"] if recs else []:
        b = {c: {"macw": 0.0, "min": math.inf, "argmin": None, "route": None, "below_1.05": [], "below_1": 0}
             for c in cols}
        b["exact_split_wins"], b["by_proj"] = 0, {}
        for r in recs:
            x = r["best"][k]
            wgt = r["macs_per_token"] / macs
            for c, f in cols.items():
                v = x[f]
                bc = b[c]
                bc["macw"] += wgt * v
                route = x["route"] if c == "exact" else x["generous_route"] if c == "generous" else None
                if v < bc["min"]:
                    bc["min"], bc["argmin"], bc["route"] = v, r["tensor"], route
                if v < 1.05:
                    bc["below_1.05"].append([r["tensor"], round(v, 4)] + ([route] if route else []))
                bc["below_1"] += v * r["leaf"] < 1
                pb = b["by_proj"].setdefault(r["tensor"].split(".")[-2], {})
                pb[c] = min(pb.get(c, math.inf), v)
            b["exact_split_wins"] += x["route"] != "none"
        out["by_price"][k] = b
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", type=Path)
    ap.add_argument("--hf", help="a Hugging Face repo, rotated and clipped here at --target")
    ap.add_argument("--target", default="0.02", help="comma-separated clip targets; several write out/tTARGET/ each")
    ap.add_argument("--layers", default=None, help="comma-separated layer indices (default every layer)")
    ap.add_argument("--match", default=None, help="comma-separated substrings; keep the tensors holding any")
    ap.add_argument("--deflate", type=int, default=0,
                    help="with --hf, clip W·R less its top-R SVD part, served apart (a candidate hardening)")
    ap.add_argument("--rot", default="hadamard", choices=("hadamard", "hadamard2", "haar", "both"),
                    help="with --hf, the rotation (module docstring)")
    ap.add_argument("--live-noise", type=int, default=0, help="price live weight noise uniform on [-f, f]")
    ap.add_argument("--kcap", type=int, default=0, help="K in chunks of at most this, each committed (0 = served K)")
    ap.add_argument("--orders", default="none", help="comma-separated orders of n and K (`price.permute`)")
    ap.add_argument("--fast", action="store_true", help="exact routes only (none, isvd, tiles)")
    ap.add_argument("--no-split", action="store_true", help="no SVD and no split: the none and tile routes only")
    ap.add_argument("--ranks", default=None, help="comma-separated split ranks (default nat_ranks)")
    ap.add_argument("--alus", default=ALUS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summarize", action="store_true", help="only rewrite summary.json from matrices.jsonl")
    args = ap.parse_args(argv)
    if args.summarize:
        (args.out / "summary.json").write_text(json.dumps(summarize(args.out / "matrices.jsonl"), indent=1))
        return 0
    alus = price.parse_alus(args.alus)
    for chip in CHIPS:
        ours, theirs = nat.check_counts(chip, "fast", 2, A_NOISE)
        assert abs(ours - theirs) < 1e-6 * theirs, (chip, ours, theirs)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    targets = [float(t) for t in args.target.split(",")]
    jobs = code_jobs(args.codes, layers) if args.codes else [
        j for t in targets for j in hf_jobs(args.hf, t, layers, args.deflate, args.rot)]
    ranks = [int(x) for x in args.ranks.split(",")] if args.ranks else None
    opts = {"fast": args.fast, "orders": tuple(args.orders.split(",")), "kcap": args.kcap,
            "live_noise": args.live_noise, "nosplit": args.no_split}
    if args.match:
        jobs = [j for j in jobs if keep(j[0], None, args.match.split(","))]
    jobs.sort(key=lambda j: (-math.prod(load_shape(j[1])), j[0]))
    outs = {t: args.out / f"t{t:g}" if len(targets) > 1 else args.out for t in targets}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    fhs = {t: open(d / "matrices.jsonl", "w") for t, d in outs.items()}
    ok = True
    print(f"{len(jobs)} jobs, {args.workers} workers, OMP {os.environ.get('OMP_NUM_THREADS')}, {opts}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, (name, src, alus, ranks, opts)) for name, src in jobs]
        for fut in concurrent.futures.as_completed(futs):
            rec = fut.result()
            ok &= all(rec["exact_crop"].values())
            fh = fhs[rec["target"] if rec["target"] is not None else targets[0]]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            sm = {k: f"{v['over_leaf']:.3f}({v['route']}) none {v['none_over_leaf']:.3f}"
                  for k, v in rec["best"].items() if k.endswith(("@0.84", "@1.78"))}
            if "noise" in rec:
                sm["noise@0.84"] = {k: round(v["over_leaf"], 4) for k, v in rec["noise"]["best"].items()
                                    if k.endswith("@0.84")}
            print(rec["tensor"], f"t={rec['target']}", rec["shape"], f"ov1={rec['overflow1']:.4f}", json.dumps(sm),
                  f"exact={all(rec['exact_crop'].values())}", f"{rec['seconds']:.0f}s", flush=True)
    for t, d in outs.items():
        fhs[t].close()
        summary = summarize(d / "matrices.jsonl")
        (d / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"target {t}:", json.dumps({k: {c: {f: v[c][f] for f in ("macw", "min", "argmin", "route")} | {
            "n_below_1.05": len(v[c]["below_1.05"])} for c in ("none", "exact")} for k, v in summary["by_price"].items()},
            indent=1), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
