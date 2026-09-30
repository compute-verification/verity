"""Pearl-C4 noise designs compared on the operands the chain reads (CPU): what rank-1 FP32-FMA noise with clean-path block
scales (`pearl-c-fp4` v3, bc-a8466279's v-c) gives up against v1's rank-32 noise with noisy scales.

For each design and census family, over S independent salts, per formed element (value = E2M1 code x its block scale):

  salt_dead     the value is the same at every draw (P3's `saltDead`, on values)
  debit_2of4    the replayed debit's 2:4 term (`theory-pearl-c4-domain.md` §2, item 1) as a share of the chain's credit on
                one operand: 0.5 per MAC over each 4-group with 2-3 salt-dead values, 1.0 if all 4 are
  delta         the share of values that differ from their modal value (the best salt-free guess): the base split's Δ
  basesplit     the base split's cost per dense MAC over 16 x 64 fragments of A (GPU 7's statistic, with the mode as
                the base): 0 if Δ is empty, 0.5 if Δ fits one 2:4 pass, else 1
  weak_cols     the share of (salt, k position) pairs whose Δ density over the rows is below a quarter of the mean:
                rank-1's weak columns, which the salt reveals to the adversary
  saturated     the share of values whose noisy input exceeds 6 x its block scale and saturates
  err_ratio     the useful product's error after the exact peel, (A′ − N_A)(B̃ − N_B)ᵀ against A·Bᵀ, over plain NVFP4's
                (the clean quantizer, no noise)

Designs (δ = 1/4 row-relative noise unless named; the noise std is δ·ρ·α₀ per element on average):

  v1            rank-32 real-unit noise (uniform E2M1 lines, as `fp4-design.md` §2) and block scales from the noisy amax
  v3-gauss      rank-1 e_i·f_t with Gaussian e, f; clean-path block scales (v-c as written)
  v3-sign       rank-1 with fixed magnitude: e_i = ±1, f_t = ±1 (noise exactly ±δρα₀); clean scales
  v3-sign-d2    the same at δ = 1/2
  v3-sign-jN    v3-sign, each row's clean block scales multiplied by 2^u_i, u_i uniform in [0, 1/N) from the salt
                (one FMUL per block of 16)
  v3-sign-h     v3-sign with the noise's headroom in the clean block scale: (amax_b + δρα₀)/6, so no value saturates
                (one FADD per block)
  v3-sign-hjN   both: (amax_b + δρα₀)·2^u_i/6, one FFMA per block
  v3-sign-hjNm  v3-sign-hjN with a salted per-row noise magnitude, |e_i| = 1 + w_i, w_i uniform in [0, 1/4) (free: a
                per-row constant), and the headroom at the largest magnitude; it gives identical rows distinct codes
  v3-sign-nm    signs-only rank-1 with the per-row magnitude, and block scales from the noisy amax as v1 (one FMNMX per
                element, uncredited)

    python benchmarks/pouw/fp4_v3_census.py --out v3-census.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp4_emulation import E2M1_HALVES, e2m1_rn, family, ue4m3_dyadic, ue4m3_rn  # noqa: E402

DESIGNS = ("v1", "v3-gauss", "v3-sign", "v3-sign-d2", "v3-sign-j4", "v3-sign-j2", "v3-sign-h", "v3-sign-hj4", "v3-sign-hj4m", "v3-sign-nm")


def value_of(codes: np.ndarray) -> np.ndarray:
    v = E2M1_HALVES[codes & 7] / 2.0
    return np.where(codes & 8, -v, v)


def scale_val(code: np.ndarray) -> np.ndarray:
    m, e = ue4m3_dyadic(code)
    return m * np.exp2(e)


def row_stats(x: np.ndarray, delta: float):
    s = np.abs(x).max(axis=1, keepdims=True)
    rho = np.sqrt((x[:, ::8] ** 2).mean(axis=1, keepdims=True))
    alpha = 2688.0 / np.maximum(s + 4 * delta * rho, 2.0 ** -32)
    return alpha, delta * rho * alpha                      # alpha0, and the noise std in scaled units


def form(x: np.ndarray, design: str, rng: np.random.Generator):
    """One salt's formed operand: (values, noise added, saturated mask), all [R, k]."""

    delta = 0.5 if design.endswith("-d2") else 0.25
    alpha, sig = row_stats(x, delta)
    ax = alpha * x
    r, k = x.shape
    if design == "v1":
        codes_e = rng.integers(0, 16, (r, 32))
        codes_f = rng.integers(0, 16, (k, 32))
        noise = sig * (value_of(codes_e) @ value_of(codes_f).T) / np.sqrt(2346.125)
    elif design == "v3-gauss":
        noise = sig * rng.standard_normal((r, 1)) * rng.standard_normal((1, k))
    else:
        mag = 1.0 + rng.uniform(0, 0.25, (r, 1)) if design.endswith("m") else 1.0
        noise = sig * mag * rng.choice([-1.0, 1.0], (r, 1)) * rng.choice([-1.0, 1.0], (1, k))
    d = ax + noise
    src = np.abs(d) if design in ("v1", "v3-sign-nm") else np.abs(ax)   # the noisy amax, or the clean quantizer's
    amax = src.reshape(r, k // 16, 16).max(axis=2)
    if "-h" in design:
        amax = amax + sig * (1.25 if design.endswith("m") else 1.0)   # the noise's headroom: its largest magnitude
    amax = amax / 6.0
    if "j" in design.split("-")[-1]:
        j = int(design.split("-")[-1].strip("hjm"))
        amax = amax * np.exp2(rng.uniform(0, 1.0 / j, (r, 1)))
    sc = scale_val(np.maximum(ue4m3_rn(amax), 1))
    scb = np.repeat(sc, 16, axis=1)
    q = d / scb
    codes = e2m1_rn(q)
    return value_of(codes) * scb, noise, np.abs(q) > 6.0


def plain(x: np.ndarray):
    alpha, _ = row_stats(x, 0.25)
    ax = alpha * x
    r, k = x.shape
    sc = scale_val(np.maximum(ue4m3_rn(np.abs(ax).reshape(r, k // 16, 16).max(axis=2) / 6.0), 1))
    scb = np.repeat(sc, 16, axis=1)
    return value_of(e2m1_rn(ax / scb)) * scb / alpha


def census(fam: str, design: str, rows: int, k: int, salts: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    xa = family(fam, "A", rows, k, np.random.default_rng(seed + 1))
    xb = family("gaussian", "B", rows, k, np.random.default_rng(seed + 2))
    vals = np.empty((salts, rows, k), np.float32)
    sat = 0.0
    for s in range(salts):
        v, _, m = form(xa, design, rng)
        vals[s] = v
        sat += m.mean()
    dead = (vals == vals[0]).all(axis=0)
    g = dead.reshape(rows, k // 4, 4).sum(axis=2)
    debit = (0.5 * ((g >= 2) & (g < 4)) + 1.0 * (g == 4)).mean()
    srt = np.sort(vals, axis=0)                                           # modal value per element
    runs = np.concatenate([np.ones((1, rows, k), bool), srt[1:] != srt[:-1]], axis=0)
    idx = np.cumsum(runs, axis=0)
    best = np.zeros((rows, k), np.int64)
    mode = srt[0].copy()
    for s in range(salts):                                                # longest run of equal values
        cnt = (idx == idx[s]).sum(axis=0)
        upd = cnt > best
        best = np.where(upd, cnt, best)
        mode = np.where(upd, srt[s], mode)
    diff = vals != mode
    delta = diff.mean()
    fr = diff[:, : rows // 16 * 16].reshape(salts, rows // 16, 16, k // 64, 64)
    nz4 = fr.reshape(salts, rows // 16, 16, k // 64, 16, 4).sum(axis=5)
    empty = ~fr.any(axis=(2, 4))
    one_pass = (nz4 <= 2).all(axis=(2, 4))
    basesplit = np.where(empty, 0.0, np.where(one_pass, 0.5, 1.0)).mean()
    col = diff.mean(axis=1)                                               # [salts, k]: per salt, per position
    weak = float((col < 0.25 * col.mean()).mean())
    # quality: two salts, the exact peel (subtract the noise the protocol removes), against plain NVFP4
    exact = xa @ xb.T
    ref = np.linalg.norm(plain(xa) @ plain(xb).T - exact)
    errs = []
    for _ in range(2):
        va, na, _ = form(xa, design, rng)
        vb, nb, _ = form(xb, design, rng)
        aa, _ = row_stats(xa, 0.5 if design.endswith("-d2") else 0.25)
        ab, _ = row_stats(xb, 0.5 if design.endswith("-d2") else 0.25)
        u = ((va - na) / aa) @ ((vb - nb) / ab).T
        errs.append(np.linalg.norm(u - exact) / ref)
    dup = np.repeat(xa[:1], 64, axis=0)                                   # identical rows: are their formed rows equal?
    same = np.mean([(lambda v: (v[1:] == v[0]).all(axis=1).mean())(form(dup, design, rng)[0]) for _ in range(4)])
    return {"salt_dead": float(dead.mean()), "debit_2of4": float(debit), "delta": float(delta), "dup_rows_equal": float(same),
            "basesplit": float(basesplit), "weak_cols": weak, "saturated": float(sat / salts),
            "err_ratio": float(np.mean(errs))}


def crafted(design: str, k: int, salts: int, seed: int) -> dict:
    """The adversary's best row template for salt-dead 4-groups: per block, a anchors at ±V (code 6) placed in pairs in
    the same 4-groups, the rest at ±c·V/6 for a code level c; the worst debit over the grid, with D-SS's dead-block screen
    (clean amax_b² > 144 ρ² marks a block dead; a row with more than 1 in 64 dead blocks is rejected)."""

    worst = {"debit_2of4": 0.0}
    rng = np.random.default_rng(seed)
    for a in (2, 4, 8, 12):
        for c in (0, 1, 2, 3, 4):
            blk = np.full(16, c / 6.0)
            blk[:a] = 1.0
            x = np.tile(blk, (64, k // 16)) * rng.choice([-1.0, 1.0], (64, k))
            s = np.abs(x).max(axis=1, keepdims=True)
            rho = np.sqrt((x[:, ::8] ** 2).mean(axis=1, keepdims=True))
            amax = np.abs(x).reshape(64, k // 16, 16).max(axis=2)
            if ((amax ** 2 > 144 * rho ** 2).mean(axis=1) > 1 / 64).all() or (s == 0).all():
                continue                                                  # rejected by D-SS
            vals = np.stack([form(x, design, rng)[0] for _ in range(salts)])
            dead = (vals == vals[0]).all(axis=0)
            g = dead.reshape(64, k // 4, 4).sum(axis=2)
            debit = float((0.5 * ((g >= 2) & (g < 4)) + 1.0 * (g == 4)).mean())
            if debit >= worst["debit_2of4"]:
                worst = {"debit_2of4": debit, "salt_dead": float(dead.mean()), "anchors": a, "level": c}
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default="gaussian,t4,massive,massive3000,spread")
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--salts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = {"rows": args.rows, "k": args.k, "salts": args.salts, "seed": args.seed, "families": {}, "crafted": {}}
    for d in args.designs.split(","):
        for f in args.families.split(","):
            r = census(f, d, args.rows, args.k, args.salts, args.seed)
            res["families"].setdefault(f, {})[d] = r
            print(f"{f:12s} {d:11s} " + " ".join(f"{a}={b:.4f}" for a, b in r.items()), flush=True)
        res["crafted"][d] = crafted(d, args.k, min(args.salts, 32), args.seed)
        print(f"{'crafted':12s} {d:11s} {res['crafted'][d]}", flush=True)
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
