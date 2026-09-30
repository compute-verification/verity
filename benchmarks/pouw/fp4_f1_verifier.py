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
from fp4_v4_coverage import T, bf16, uval  # noqa: E402

UE4M3_ONE = 0x38
_PHI_CACHE: dict[int, np.ndarray] = {}


# ---- the exact law, sized per row -------------------------------------------------------------------------------------

def _phi(size):
    if size not in _PHI_CACHE:
        w = 2 * np.pi * np.arange(size // 2 + 1) / size
        _PHI_CACHE[size] = np.stack([(2 + 2 * sum(np.cos(w * v * h) for h in F._MAGS)) / 16 for v in F._MAGS])
    return _PHI_CACHE[size]


def row_law(eh):
    """(cdf, L): P(n ≤ m) for m in [−L, L] per row, L the rows' largest support 12·Σ|e_j|, each row's FFT sized to its own
    support (no aliasing inside it, zero outside)."""
    Lr = 12 * np.abs(eh).sum(1)
    L = int(Lr.max())
    counts = np.stack([(np.abs(eh) == v).sum(1) for v in F._MAGS], 1)
    sizes = np.maximum(64, 1 << np.ceil(np.log2(2 * Lr + 1)).astype(int))
    n = np.arange(-L, L + 1)
    cdf = np.empty((len(eh), 2 * L + 1))
    for size in np.unique(sizes):
        idx = np.nonzero(sizes == size)[0]
        phi = np.ones((len(idx), size // 2 + 1))
        ph = _phi(int(size))
        for i in range(len(F._MAGS)):
            phi *= np.power(ph[i][None, :], counts[idx, i:i + 1])
        pmf = np.fft.irfft(phi, int(size), axis=1)[:, n % size]
        pmf = np.where(np.abs(n)[None, :] <= Lr[idx, None], np.maximum(pmf, 0), 0)
        c = np.cumsum(pmf, 1)
        cdf[idx] = c / c[:, -1:]
    return cdf, L


def p_le(cdf, L, m):
    idx = m + L
    base = (np.arange(len(cdf)) * (2 * L + 1)).reshape((-1,) + (1,) * (m.ndim - 1))
    out = cdf.ravel()[base + np.clip(idx, 0, 2 * L)]
    return np.where(idx < 0, 0.0, out)


def below(u, delta, tie_down, L):
    z = u / delta
    r = np.rint(z)
    tie = np.abs(z - r) < 1e-9 * np.maximum(1.0, np.abs(z))
    m = np.where(tie, r - np.where(tie_down, 0, 1), np.ceil(z) - 1)
    return np.clip(m, -L - 1, L + 1).astype(np.int64)


def p_lower(v, s, delta, cdf, L, j=None):
    """A lower bound on p_t from 2j CDF lookups: the j edges on either side of the element's own cell resolve the cells
    between them exactly, and the mass beyond them on each side is taken as one cell, so
    1 − max(those cells, the two tails) ≤ 1 − max_c P(code = c) = p_t. j = 7 is exact."""
    j = P_LOOKUPS // 2 if j is None else j
    c0 = np.searchsorted(F.EDGES, v / s, side="right")
    d = delta[:, None]
    Fs = []
    for i in range(-j, j):
        e = c0 + i
        ec = np.clip(e, 0, 13)
        f = p_le(cdf, L, below(F.EDGES[ec] * s - v, d, F.TIE_DOWN[ec], L))
        Fs.append(np.where(e < 0, 0.0, np.where(e > 13, 1.0, f)))
    best = np.maximum(Fs[0], 1 - Fs[-1])
    for lo, hi in zip(Fs[:-1], Fs[1:]):
        best = np.maximum(best, hi - lo)
    return np.clip(1 - best, 0, 1)


P_LOOKUPS = 4


# ---- the byte law: P(byte ≤ b) by the product formula ----------------------------------------------------------------

_U4 = uval(np.arange(127))
_MID = np.concatenate([(_U4[:-1] + _U4[1:]) / 2, [np.inf]])      # byte ≤ b ⇔ amax/6 < _MID[b] (b = 126: always)


def byte_cdf(vb, rows, delta, cdf, L, b):
    """P(byte ≤ b) for NVFP4 blocks: vb (N, 16) the blocks' values α·x, rows (N,) each block's row (its law and δ),
    b (N, C) candidate bytes; the floor at byte 1 makes P(byte ≤ 0) = 0."""
    U = 6 * _MID[np.clip(b, 0, 126)]
    d = delta[rows][:, None, None]
    Ub = np.where(np.isfinite(U), U, 0.0)[..., None]
    hi = np.clip(np.ceil((Ub - vb[:, None, :]) / d) - 1, -L - 1, L + 1).astype(np.int64)
    lo = np.clip(np.floor((-Ub - vb[:, None, :]) / d), -L - 1, L + 1).astype(np.int64)
    base = (rows * (2 * L + 1))[:, None, None]
    flat = cdf.ravel()
    ph = np.where(hi + L < 0, 0.0, flat[base + np.clip(hi + L, 0, 2 * L)])
    pl = np.where(lo + L < 0, 0.0, flat[base + np.clip(lo + L, 0, 2 * L)])
    g = np.prod(np.clip(ph - pl, 0, 1), axis=2)
    g = np.where(np.isfinite(U), g, 1.0)
    return np.where(b <= 0, 0.0, g)


def _blocks(v):
    R, k = v.shape
    return v.reshape(R * (k // 16), 16), np.repeat(np.arange(R), k // 16)


def modal_product(v, delta, cdf, L):
    """The exact modal byte: P(byte = b) over every byte the rows' noise support reaches."""
    R, k = v.shape
    vb, rows = _blocks(v)
    amax = np.abs(vb).max(1)
    reach = delta[rows] * L
    lo = F.scale_byte(np.maximum(amax - reach, 0), "nv")
    hi = F.scale_byte(amax + reach, "nv")
    C = int((hi - lo).max()) + 2
    cand = lo[:, None] - 1 + np.arange(C)[None, :]
    out = np.empty(len(vb), np.int64)
    for a in range(0, len(vb), 4096):
        sl = slice(a, a + 4096)
        pb = np.diff(byte_cdf(vb[sl], rows[sl], delta, cdf, L, cand[sl]), axis=1)
        out[sl] = cand[sl, 1:][np.arange(len(pb)), pb.argmax(1)]
    return out.reshape(R, k // 16), C


def modal_local(v, delta, cdf, L, rea, widths=(1, 4, 16)):
    """The modal byte by a certified local search around the realised byte r: with P(b) for b in r − w … r + w and the
    mass outside that window, the argmax inside is the modal byte whenever it is at least the outside mass (no byte
    outside can beat it). Blocks not certified at one width try the next, on those blocks only; any still uncertified
    are returned as -1, for the caller to charge on the safe side. Costs 2·(2w + 2)·16 CDF lookups per block."""
    R, k = v.shape
    vb, rows = _blocks(v)
    r = rea.ravel()
    modal = np.full(len(r), -1, np.int64)
    todo = np.arange(len(r))
    for w in widths:
        if len(todo) == 0:
            break
        cand = r[todo][:, None] + np.arange(-w - 1, w + 1)[None, :]
        g = byte_cdf(vb[todo], rows[todo], delta, cdf, L, cand)
        pb = np.diff(g, axis=1)
        ok = pb.max(1) >= 1 - (g[:, -1] - g[:, 0])
        modal[todo[ok]] = r[todo[ok]] + pb[ok].argmax(1) - w
        todo = todo[~ok]
    return modal.reshape(R, k // 16)


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
