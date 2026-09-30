"""The verifier's per-tile cost with and without F1′ (κ inside the expectation, the exact noise law), on CPU, and cheaper
evaluations that stay sound (`docs/pouw/pearl-c4-fix.md` §7; the conditions are the assessor's 15:18Z line in
`internal/pouw/red-team/ratings.md`).

The replay without F1′ is a vectorised exact replay of one drawn 64 x 64 NVFP4 tile at depth k: both sides' forming (the
noise atom's sum, the noisy block scale, the cast), the chain (`fp4_emulation.atoms` + `native_chain`, the pinned atom
bit for bit), the fused F columns A′·F_B and B̃·F_A (32 each), the clean-up's product, today's debit terms (the witness,
2:4 units, screened blocks, D-24 windows, forming elements) and a hash of the words.

F1′ per side (64 rows), on the exact law:

    census   the census as run: the per-row pmf by a 16,384-point FFT, P(code) over all 15 cells, the modal byte from 24
             draws of the law (768 MACs per element)
    product  the same, the modal byte by the product formula over every byte the row's noise support reaches (exact)
    fast     sound bounds that only raise the debit: the pmf's FFT sized to each row's support; p_t from two lookups,
             1 − max(P(own cell), P(below), P(above)) ≤ p_t (S_w only grows as any p_t falls); and a block counted off
             its modal byte only when its realised byte is not a local maximum of the byte law (4 product-formula
             evaluations per block), which counts no more blocks than the modal-byte rule does, so κ can only fall

Each option then runs the same κ-inside window DPs, R1's per-row sum and F2's byte histogram.

    python benchmarks/pouw/fp4_f1_verifier.py cost --k 8192 16384
    python benchmarks/pouw/fp4_f1_verifier.py check          # fast against product on random rows (sound and tight)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp4_f1_pricing as F  # noqa: E402
from fp4_emulation import atoms, e2m1_rn, native_chain  # noqa: E402
from fp4_v4_coverage import T, bf16  # noqa: E402

UE4M3_ONE = 0x38
row_law, p_le, below, p_lower, byte_cdf, modal_product, modal_local = (F.row_law, F.p_leL, F.belowL, F.p_lower, F.byte_cdf,
                                                                    F.modal_product, F.modal_local)


# ---- one tile: forming, the replay, and F1′ ---------------------------------------------------------------------------

def form_side(X, rng):
    alpha, ae, dnf, rho2 = F.stats_fmt(X, "nv")
    R, k = X.shape
    en, fn = rng.integers(0, 16, (R, 32)), rng.integers(0, 16, (k, 32))
    eh, fh = F.H[en], F.H[fn]
    v = alpha[:, None] * X
    D = v + (ae / 64)[:, None] * (eh.astype(np.float64) @ fh.T.astype(np.float64))
    byt = F.scale_byte(np.abs(D).reshape(R, k // 16, 16).max(2), "nv")
    codes = e2m1_rn(D / np.repeat(F.scale_value(byt, "nv"), 16, axis=1))
    return dict(X=X, v=v, alpha=alpha, ae=ae, delta=ae / 64, rho2=rho2, eh=eh, fn=fn, codes=codes, bytes=byt)


def replay(XA, XB, rng):
    """The tile's replay without F1′; returns (timings, formed sides)."""
    t = {}
    s = time.perf_counter()
    a, b = form_side(XA, rng), form_side(XB, rng)
    t["forming"] = time.perf_counter() - s
    s = time.perf_counter()
    _, mant, x, part = atoms(a["codes"], a["bytes"], b["codes"], b["bytes"])
    c, same = native_chain(mant, x, part, np.zeros((len(XA), len(XB)), np.uint32), record=True)
    t["chain"] = time.perf_counter() - s
    s = time.perf_counter()
    k = XA.shape[1]
    for side, other in ((a, b), (b, a)):
        fcols = other["fn"].T.copy()
        _, mant, x, part = atoms(side["codes"], side["bytes"], fcols, np.full((32, k // 16), UE4M3_ONE))
        native_chain(mant, x, part, np.zeros((len(side["codes"]), 32), np.uint32))
    t["fused"] = time.perf_counter() - s
    s = time.perf_counter()
    pa = rng.standard_normal((len(XA), 64)).astype(np.float32)
    pb = rng.standard_normal((len(XB), 64)).astype(np.float32)
    u = (pa @ pb.T).view(np.uint32) & 0xFFFF0000
    t["clean_up"] = time.perf_counter() - s
    s = time.perf_counter()
    for side in (a, b):
        sd = F.side_fmt(side["X"], "nv")
        h = side["codes"] & 7
        (np.count_nonzero(h.reshape(len(h), -1, 4), axis=2) <= 2).reshape(len(h), -1, 32).all(2)
        side["sd"] = sd
    t["debit_today"] = time.perf_counter() - s
    s = time.perf_counter()
    hashlib.blake2b(c.tobytes() + u.tobytes()).digest()
    t["hash"] = time.perf_counter() - s
    return t, a, b, same


def f1_side(side, rng, option):
    """F1′ (κ inside) per row window for one formed side, by `option`; returns (timings, per-window F1′, per-row F1′
    per word in FP4 units, the number of blocks the fast option could not certify)."""
    t = {}
    X, v, delta, eh = side["X"], side["v"], side["delta"], side["eh"]
    R, k = X.shape
    sd = side["sd"]
    zero = np.repeat(sd["screened"], 16, axis=1) | sd["dead"]
    uncertified = 0
    s = time.perf_counter()
    cdf, L = row_law(eh) if option == "fast" else (F.row_cdf(eh), F.LMAX)
    t["pmf"] = time.perf_counter() - s
    s = time.perf_counter()
    if option == "census":
        mod = F.modal_of([F.draw_bytes(v, eh, delta, rng) for _ in range(24)])
    elif option == "product":
        mod = modal_product(v, delta, cdf, L)[0]
    else:
        mod = modal_local(v, delta, cdf, L, side["bytes"])
        bad = mod < 0
        uncertified = int(bad.sum())
    t["modal_byte"] = time.perf_counter() - s
    s = time.perf_counter()
    if option == "fast":
        # an uncertified block is charged on the safe side: no κ relief and p = 0
        off = (mod != side["bytes"]) & ~bad
        sm = np.repeat(F.scale_value(np.where(bad, side["bytes"], mod), "nv"), 16, axis=1)
        p = np.where(np.repeat(bad, 16, axis=1), 0.0, p_lower(v, sm, delta, cdf, L))
    else:
        off = mod != side["bytes"]
        sm = np.repeat(F.scale_value(mod, "nv"), 16, axis=1)
        p = np.concatenate([F.p_exact(v[r0:r0 + 16], sm[r0:r0 + 16], delta[r0:r0 + 16], cdf[r0:r0 + 16])[0]
                            for r0 in range(0, R, 16)])
    p = np.where(zero, 0.0, p)
    t["p"] = time.perf_counter() - s
    s = time.perf_counter()
    _, d_in, _ = F.f1_windows(p, off.astype(int))
    rows = 128 * d_in.sum(1)
    t["dp"] = time.perf_counter() - s
    s = time.perf_counter()
    np.bincount(side["bytes"].ravel()).max()
    t["f2_r1"] = time.perf_counter() - s
    return t, d_in, rows, uncertified


def tile_inputs(k, rng):
    """A drawn tile's rows: activations with heavy-tailed channels (a few loud channels, as real inputs have) and
    Gaussian weights."""
    scale = np.exp(0.8 * rng.standard_normal(k))
    XA = bf16(rng.standard_normal((T, k)) * scale[None, :])
    XB = bf16(rng.standard_normal((T, k)) * 0.02)
    return XA, XB


def cost(args):
    out = []
    for k in args.k:
        rng = np.random.default_rng(7)
        XA, XB = tile_inputs(k, rng)
        best = None
        for _ in range(args.repeats):
            t, a, b, _ = replay(XA, XB, np.random.default_rng(1))
            best = t if best is None else {key: min(best[key], t[key]) for key in t}
        rec = {"k": k, "replay": best, "replay_total": sum(best.values())}
        for option in args.options:
            tb = None
            for _ in range(args.repeats):
                ts = [f1_side(side, np.random.default_rng(2), option)[0] for side in (a, b)]
                t = {key: ts[0][key] + ts[1][key] for key in ts[0]}
                tb = t if tb is None else {key: min(tb[key], t[key]) for key in t}
            rec[option] = tb
            rec[option + "_total"] = sum(tb.values())
            rec[option + "_ratio"] = rec[option + "_total"] / rec["replay_total"]
        out.append(rec)
        print(json.dumps(rec), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


def check(args):
    """fast against product on random tiles: fast must charge at least as much per window (sound), and not much more."""
    worst_under, over, rows, unc, blocks = 0.0, [], [], 0, 0
    for seed in range(args.seeds):
        rng = np.random.default_rng(100 + seed)
        XA, XB = tile_inputs(args.k, rng)
        _, a, b, _ = replay(XA, XB, rng)
        for side in (a, b):
            _, d_fast, r_fast, u = f1_side(side, np.random.default_rng(seed), "fast")
            _, d_prod, r_prod, _ = f1_side(side, np.random.default_rng(seed), "product")
            worst_under = max(worst_under, float((d_prod - d_fast).max()))
            over.append(float(d_fast.sum() - d_prod.sum()) / max(float(d_prod.sum()), 1e-300))
            rows.append((float(r_fast.max()), float(r_prod.max())))
            unc, blocks = unc + u, blocks + side["bytes"].size
    rec = {"k": args.k, "seeds": args.seeds, "max_window_undercharge": worst_under,
           "fast_over_product_relative": [min(over), max(over)], "row_max_fast_vs_product": rows,
           "uncertified_blocks": unc, "blocks": blocks}
    print(json.dumps(rec), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost")
    c.add_argument("--k", type=int, nargs="+", default=[8192])
    c.add_argument("--options", nargs="+", default=["census", "product", "fast"])
    c.add_argument("--repeats", type=int, default=2)
    c.add_argument("--out", default=None)
    k = sub.add_parser("check")
    k.add_argument("--k", type=int, default=4096)
    k.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)
    {"cost": cost, "check": check}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
