#!/usr/bin/env python3
"""Llama-3.1-70B's NVFP4 coverage census on the GPU: bc-f5bf55c8's version 4 (`fp4_f1_pricing.linear` on the certified
law, the rule of `internal/pouw/rtx-pro/fp4-coverage-70b/job.md`: R1, B̃ at F1_B_OVERFIT, term 3 per element, #556's
`volunteer` rows, the 1/400 tile reject) ported to torch, over every 64-row weight band of every linear and one or more
salts.

The port reads the census's own sources (`--src`: the files its fill job embeds, sha256-checked against SOURCES) for
every constant and table, and leaves on the CPU what must be the census's own code or bits: the draws (the same numpy
generator, in `linear`'s order and chunks), each row's ρ² (the census's expression, so α, a_E and D-NF are its bits),
`volunteer_rows`, and the summary. Per element everything runs on the device in float64: the witness, the realised
bytes, each row's noise law, the certified modal byte, p_t's lower bound, F1′'s window DPs, and each tile's sums.

It is not bit for bit: a row's law is an FFT of `pow`s, whose last bits differ between numpy and the device, and the
device sums in another order, so continuous figures agree to about 1e-12 relative. `gate` compares the port with
`linear` itself on the same draws: every count (tiles, rows, blocks, the rows `volunteer` gives up) must be equal and
every other figure within RTOL.

    fp4_cov70b_gpu.py ref  --src S --out O --layers 0,79      # CPU: `linear` itself, the gate's side
    fp4_cov70b_gpu.py gate --src S --out O --layers 0,79      # GPU: the port on the same draws
    fp4_cov70b_gpu.py run  --src S --out O --layers 0-19 --salts 0-3 --budget-min 6.5
    fp4_cov70b_gpu.py summary --src S --out O --salts 0-3

(--acts and --weights default to node 2's capture and checkpoint.) `ref` and `gate` do each layer's seven linears at
the census's 8 weight bands and its k_proj and v_proj at every band (the run's path: several draw chunks). `run` does
every band, one file per (salt, layer, linear); salt 0 is the census's seed [20260930, L, i, 0], salt s > 0 appends s.
Each exits 99 once --budget-min is spent with work left, and 0 when done, after copying --then (a fill job script) into
the fill queue once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCES = {"fp4_f1_pricing.py": "75809dde753cd21c40cc6ef6ea1d945767b38cd9e57cf80baef5b62c997a7ecc",
           "fp4_v4_coverage.py": "4a9c92aa4af1acb0dde5312998d3ce8780342da23ca0b88d32875b7cae3b222a",
           "fp4_emulation.py": "2cefa854fbdf713b4d39001806d827e2c01539a2f92a19d6c096a4feb38e4a00",
           "cov70b.py": "510088f9ca08a8a10ff5093238dbbbeb589a656d2814fc7065ac669ba98e7c02"}
SEED, CENSUS_COLS, TOKENS = 20260930, 8, 256
MODEL = "unsloth/Meta-Llama-3.1-70B"
RTOL = 1e-9
FILL_QUEUE = Path(os.environ.get("FILL_DIR", "/workspace/pouw/fill")) / "queue"
F = E = None


def load(src: str, check: bool = True):
    """The census's modules from `src`, each file's sha256 checked."""
    global F, E
    root = Path(src).resolve()
    for name, sha in SOURCES.items():
        got = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if check and got != sha:
            raise SystemExit(f"{root / name}: sha256 {got}, not the census's {sha}")
    sys.path.insert(0, str(root))
    import fp4_emulation
    import fp4_f1_pricing
    F, E = fp4_f1_pricing, fp4_emulation
    assert F.DELTA["nv"] == 0.25 and not F.ZERO_SCREENED and F.P_LOOKUPS == 4 and F.XM == 4
    return F


def seed(layer: int, i: int, salt: int) -> list[int]:
    return [SEED, layer, i, 0] if salt == 0 else [SEED, layer, i, 0, salt]


# ---- the census's element-wise rules on the device ------------------------------------------------------------------

class Port:
    """`fp4_f1_pricing`'s NVFP4 certified evaluation on a torch device, in float64."""

    def __init__(self, dev: str, batch_elems: int = 1 << 24):
        import torch
        torch.set_default_dtype(torch.float64)
        self.t, self.dev, self.batch = torch, torch.device(dev), batch_elems
        f = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        self.f = f
        self.edges, self.tie_down = f(F.EDGES), f(F.TIE_DOWN, torch.bool)
        self.mid6 = f(6 * F._MID)
        self.halves = f(E.E2M1_HALVES)
        self.uvals = f(F.uval(np.arange(128)))
        self.vx = f(F.VX)
        combos = np.array(F.COMBOS)
        self.js = [f(combos[:, i], torch.long) for i in range(4)]
        self.exc = [[i for i, x in enumerate(F.EXC) if x == v] for v in range(F.XM + 1)]
        self.phis: dict[int, object] = {}

    # formats
    def ue4m3_rn(self, v):
        t = self.t
        sub = v < 2.0 ** -6
        ms = t.round(v / 2.0 ** -9)
        out_sub = t.where(ms >= 8, 8.0, ms)
        fr = t.frexp(t.where(sub, 1.0, v))
        e = fr.exponent.long() - 1
        m = t.round((2 * fr.mantissa - 1.0) * 8)
        e = t.where(m == 8, e + 1, e)
        m = t.where(m == 8, 0.0, m).long()
        ef = e + 7
        sat = (ef > 15) | ((ef == 15) & (m > 6))
        ef, m = t.where(sat, 15, ef), t.where(sat, 6, m)
        return t.where(sub, out_sub.long(), ef * 8 + m)

    def uval(self, code):
        return self.uvals[code]

    def scale_byte(self, amax):
        return self.t.clamp(self.ue4m3_rn(amax / 6), min=1)

    def scale_value(self, byte):
        return self.t.clamp(self.uval(byte), min=2.0 ** -9)

    def e2m1_rn(self, v):
        mag = v.abs()
        code = self.t.zeros(v.shape, dtype=self.t.long, device=self.dev)
        for mid, up in E._E2M1_MIDS:
            code += (mag >= mid) if up else (mag > mid)
        return self.t.where((v < 0) & (code > 0), code | 8, code)

    def e2v(self, code):
        mag = self.halves[code & 7] / 2
        return self.t.where((code & 8) != 0, -mag, mag)

    # one side's rule: statistics, the witness, D-SS's screen
    def side(self, X, xs8: np.ndarray) -> dict:
        """`side_fmt(X, "nv")`: X (R, k) on the device, xs8 its columns ::8 as a float64 array (ρ² is the census's own
        numpy expression, so α, a_E and D-NF are its bits)."""
        t = self.t
        R, k = X.shape
        rho2 = (xs8 ** 2).mean(1)
        s = X.abs().amax(1).cpu().numpy()
        sig = np.sqrt(rho2) / 4
        alpha = 2688 / np.maximum(s + 4 * sig, 2.0 ** -32)
        ae = F.uval(F.ue4m3_rn(sig * alpha * F.KB1))
        dnf = ae >= 2.0 ** -6
        al, aet = self.f(alpha), self.f(ae)
        dead = t.empty((R, k), dtype=t.bool, device=self.dev)
        step = max(1, self.batch // (4 * k))
        for a in range(0, R, step):
            dead[a:a + step] = self.witness(X[a:a + step], al[a:a + step], aet[a:a + step])
        amax = X.abs().reshape(R, -1, 16).amax(2)
        scr = (amax * amax) > self.f(90 * rho2)[:, None]
        d4 = dead.reshape(R, -1, 4).sum(2)
        units = t.where(d4 == 4, 4, t.where(d4 >= 2, 2, 0)).sum(1).double()
        return dict(dnf=dnf, units=units, scr=scr, sdw=dead.sum(1).double(), dead=dead, alpha=al, ae=aet)

    def witness(self, X, alpha, ae):
        t = self.t
        R, k = X.shape
        C = (alpha[:, None] * X).reshape(R, k // 16, 16)
        A = C.abs()
        top = A.amax(2, keepdim=True)
        second = A.topk(2, dim=2).values[:, :, 1:2]
        rest = t.where(A == top, second, top)
        N = (72.0 * ae)[:, None, None]
        vals = []
        for d in (C + N, C, C - N):
            r = 1 / t.clamp(self.uval(self.ue4m3_rn(t.maximum(d.abs(), rest) / 6)), min=2.0 ** -9)
            vals.append(self.e2v(self.e2m1_rn(d * r)))
        return ((vals[0] == vals[2]) & (vals[0] == vals[1])).reshape(R, k)

    # the exact law, the certified evaluation
    def phi(self, size: int):
        if size not in self.phis:
            self.phis[size] = self.f(F._phi(size))
        return self.phis[size]

    def row_law(self, eh: np.ndarray):
        t = self.t
        Lr = 12 * np.abs(eh).sum(1)
        L = int(Lr.max())
        counts = np.stack([(np.abs(eh) == v).sum(1) for v in F._MAGS], 1)
        sizes = np.maximum(64, 1 << np.ceil(np.log2(2 * Lr + 1)).astype(int))
        n = t.arange(-L, L + 1, device=self.dev)
        Lt = self.f(Lr)
        cdf = t.empty((len(eh), 2 * L + 1), device=self.dev)
        for size in np.unique(sizes):
            idx = self.f(np.nonzero(sizes == size)[0], t.long)
            ph = self.phi(int(size))
            cnt = self.f(counts[idx.cpu().numpy()])
            phi = t.ones((len(idx), int(size) // 2 + 1), device=self.dev)
            for i in range(len(F._MAGS)):
                phi *= ph[i][None, :] ** cnt[:, i:i + 1]
            pmf = t.fft.irfft(phi, int(size), dim=1)[:, n % int(size)]
            pmf = t.where(n.abs()[None, :] <= Lt[idx, None], t.clamp(pmf, min=0), 0.0)
            c = t.cumsum(pmf, 1)
            cdf[idx] = c / c[:, -1:]
        return cdf, L

    def p_le(self, cdf, L, m):
        idx = m + L
        base = (self.t.arange(len(cdf), device=self.dev) * (2 * L + 1)).reshape((-1,) + (1,) * (m.dim() - 1))
        out = cdf.reshape(-1)[base + idx.clamp(0, 2 * L)]
        return self.t.where(idx < 0, 0.0, out)

    def below(self, u, delta, tie_down, L):
        t = self.t
        z = u / delta
        r = t.round(z)
        tie = (z - r).abs() < 1e-9 * t.clamp(z.abs(), min=1.0)
        m = t.where(tie, r - t.where(tie_down, 0.0, 1.0), t.ceil(z) - 1)
        return t.clamp(m, -L - 1, L + 1).long()

    def p_lower(self, v, s, delta, cdf, L, j=2):
        t = self.t
        c0 = t.searchsorted(self.edges, (v / s).contiguous(), right=True)
        d = delta[:, None]
        Fs = []
        for i in range(-j, j):
            e = c0 + i
            ec = e.clamp(0, 13)
            f = self.p_le(cdf, L, self.below(self.edges[ec] * s - v, d, self.tie_down[ec], L))
            Fs.append(t.where(e < 0, 0.0, t.where(e > 13, 1.0, f)))
        best = t.maximum(Fs[0], 1 - Fs[-1])
        for lo, hi in zip(Fs[:-1], Fs[1:]):
            best = t.maximum(best, hi - lo)
        return t.clamp(1 - best, 0, 1)

    def byte_cdf(self, vb, rows, delta, cdf, L, b):
        t = self.t
        U = self.mid6[b.clamp(0, 126)]
        d = delta[rows][:, None, None]
        fin = t.isfinite(U)
        Ub = t.where(fin, U, 0.0)[..., None]
        hi = t.clamp(t.ceil((Ub - vb[:, None, :]) / d) - 1, -L - 1, L + 1).long()
        lo = t.clamp(t.floor((-Ub - vb[:, None, :]) / d), -L - 1, L + 1).long()
        base = (rows * (2 * L + 1))[:, None, None]
        flat = cdf.reshape(-1)
        ph = t.where(hi + L < 0, 0.0, flat[base + (hi + L).clamp(0, 2 * L)])
        pl = t.where(lo + L < 0, 0.0, flat[base + (lo + L).clamp(0, 2 * L)])
        x = t.clamp(ph - pl, 0, 1)
        g = x[..., 0]
        for j in range(1, x.shape[-1]):
            g = g * x[..., j]
        g = t.where(fin, g, 1.0)
        return t.where(b <= 0, 0.0, g)

    def modal_local(self, v, delta, cdf, L, rea, widths=(1, 4, 16)):
        t = self.t
        R, k = v.shape
        vb = v.reshape(R * (k // 16), 16)
        rows = t.arange(R, device=self.dev).repeat_interleave(k // 16)
        r = rea.reshape(-1)
        modal = t.full((len(r),), -1, dtype=t.long, device=self.dev)
        todo = t.arange(len(r), device=self.dev)
        for w in widths:
            if len(todo) == 0:
                break
            cand = r[todo][:, None] + t.arange(-w - 1, w + 1, device=self.dev)[None, :]
            step = max(1, self.batch // (16 * cand.shape[1]))
            ok_all, arg_all = [], []
            for a in range(0, len(todo), step):
                sl = todo[a:a + step]
                g = self.byte_cdf(vb[sl], rows[sl], delta, cdf, L, cand[a:a + step])
                pb = g[:, 1:] - g[:, :-1]
                ok_all.append(pb.max(1).values >= 1 - (g[:, -1] - g[:, 0]))
                arg_all.append(pb.argmax(1))
            ok, arg = t.cat(ok_all), t.cat(arg_all)
            modal[todo[ok]] = r[todo[ok]] + arg[ok] - w
            todo = todo[~ok]
        return modal.reshape(R, k // 16)

    def law_side(self, X, sd, eh: np.ndarray, draws) -> dict:
        """`law_side(X, rng, "certified", sd=sd)` given its draws: eh (R, 32) and, per chunk of rows (a, b), the F of
        `draw_bytes`."""
        t = self.t
        R, k = X.shape
        delta = sd["ae"] / 64
        v = sd["alpha"][:, None] * X
        ehd = self.f(eh)
        rea = t.empty((R, k // 16), dtype=t.long, device=self.dev)
        for a, b, Fm in draws:
            D = v[a:b] + delta[a:b, None] * (ehd[a:b] @ self.f(Fm).T)
            rea[a:b] = self.scale_byte(D.abs().reshape(b - a, k // 16, 16).amax(2))
        cdf, L = self.row_law(eh)
        mod = self.modal_local(v, delta, cdf, L, rea)
        bad = mod < 0
        s_ = self.scale_value(t.where(bad, rea, mod)).repeat_interleave(16, dim=1)
        p = t.where(bad.repeat_interleave(16, dim=1), 0.0, self.p_lower(v, s_, delta, cdf, L))
        mod = t.where(bad, rea, mod)
        p = t.where(sd["dead"], 0.0, p)
        return dict(p=p, mod=mod, rea=rea, off=(mod != rea), uncertified=int(bad.sum()), blocks=int(rea.numel()))

    # F1′'s window DPs
    def conv_trunc(self, a, b):
        m = F.XM
        out = self.t.zeros_like(a)
        for i in range(m + 1):
            for j in range(m + 1 - i):
                out[..., i + j] += a[..., i] * b[..., j]
        out[..., m] += a.sum(-1) * b.sum(-1) - out.sum(-1)
        return out

    def block_excess(self, p):
        t = self.t
        s = t.sort(p, dim=1).values
        pa, pb = s[:, 0::2], s[:, 1::2]
        pj = t.stack([(1 - pa) * (1 - pb), pa * (1 - pb) + (1 - pa) * pb, pa * pb], -1)
        nch = p.shape[1] // 8
        d = None
        for c0 in range(nch):
            ch = pj[:, c0::nch]
            js = self.js
            terms = ch[:, 0][:, js[0]] * ch[:, 1][:, js[1]] * ch[:, 2][:, js[2]] * ch[:, 3][:, js[3]]
            dc = t.zeros((p.shape[0], F.XM + 1), device=self.dev)
            for x, idx in enumerate(self.exc):
                for i in idx:
                    dc[:, x] += terms[:, i]
            d = dc if d is None else self.conv_trunc(d, dc)
        return d

    def changed_dist(self, p):
        m = F.XM
        d = self.t.zeros((p.shape[0], m + 1), device=self.dev)
        d[:, 0] = 1
        for j in range(p.shape[1]):
            q = p[:, j:j + 1]
            nd = d * (1 - q)
            nd[:, 1:] += d[:, :-1] * q
            nd[:, m] += d[:, m] * q[:, 0]
            d = nd
        return d

    def f1_windows(self, p, off):
        t = self.t
        R, k = p.shape
        W, nb = k // 128, 8
        kap = t.clamp(F.C * off[:, :W * nb].reshape(-1, nb).sum(1).double(), max=0.125)
        pw = p[:, :W * 128].reshape(-1, 128)
        be = self.block_excess(pw.reshape(-1, 16)).reshape(len(pw), nb, F.XM + 1)
        dx = be[:, 0]
        for j in range(1, nb):
            dx = self.conv_trunc(dx, be[:, j])
        d0, d1 = self.changed_dist(pw[:, :64]), self.changed_dist(pw[:, 64:128])
        vx = self.vx
        s = (dx * vx).sum(1) + (d0 * vx).sum(1) + (d1 * vx).sum(1)
        in24 = (dx * t.clamp(vx[None, :] - kap[:, None], min=0)).sum(1)
        pair = vx[:, None] + vx[None, :]
        inskip = t.einsum("wa,wb,wab->w", d0, d1, t.clamp(pair[None] - kap[:, None, None], min=0))
        return (t.clamp(t.clamp(s, max=1.0) - kap, min=0).reshape(R, W), t.clamp(in24 + inskip, max=1.0).reshape(R, W),
                s.reshape(R, W))

    def windows(self, p, off):
        """f1_windows in row batches."""
        R, k = p.shape
        step = max(1, self.batch // k)
        parts = [self.f1_windows(p[a:a + step], off[a:a + step]) for a in range(0, R, step)]
        return tuple(self.t.cat(z) for z in zip(*parts))


# ---- one linear ------------------------------------------------------------------------------------------------------

def _hist_max(K, rea, mask=None):
    """Per 64-row band of rea (R, k/16): the largest count of one byte over the band's (masked) rows."""
    t = K.t
    nb = rea.shape[0] // F.T
    b = rea.reshape(nb, -1)
    if mask is not None:
        b = t.where(mask.reshape(nb, F.T, 1), rea.reshape(nb, F.T, -1), 256).reshape(nb, -1)
    off = (t.arange(nb, device=K.dev) * 257)[:, None]
    h = t.bincount((b + off).reshape(-1), minlength=257 * nb).reshape(nb, 257)[:, :256]
    return h.max(1).values.cpu().numpy()


def linear(K: Port, X, W, rng, max_cols: int, chunk_elems: int = 4_000_000) -> dict:
    """`F.linear(X, W, rng, "certified", max_cols, 0, "nv", False, chunk_elems)` on K's device: the same dict. X: (m, k)
    float activations; W: (n, k) BF16 weights as a CPU torch tensor."""
    t = K.t
    T, B, CAP, FS = F.T, 16, F.CAP, F.FS
    X = F.bf16(X)
    m, k_full = X.shape
    n = W.shape[0]
    m -= m % T
    ntiles = n // T
    every = max_cols <= 0 or max_cols >= ntiles
    cols = list(range(ntiles)) if every else sorted(rng.choice(ntiles, max_cols, replace=False))
    k = X.shape[1] - X.shape[1] % B
    cL = F.c_L(m, k, n, B)
    Xa = X[:m, :k]
    a = K.side(K.f(Xa), np.ascontiguousarray(Xa[:, ::8]))
    eh = F.H[rng.integers(0, 16, (m, 32))]
    la = K.law_side(K.f(Xa), a, eh, [(0, m, F.H[rng.integers(0, 16, (k, 32))].astype(np.float64))])
    da, dia, sa = K.windows(la["p"], la["off"])
    ua, uia = (128 * da.sum(1)).cpu().numpy(), (128 * dia.sum(1)).cpu().numpy()
    fa_row = da.mean(1).cpu().numpy()
    cw = k + (32 + FS) * k / n + 256
    ok_all = a["dnf"]
    exa = uia > CAP * cw
    na = m // T
    # A's bands: sums over the ok (credited, D-NF) rows and the ok & ~R1 rows
    okA = ok_all[:m].reshape(na, T)
    okrA = okA & ~exa.reshape(na, T)
    units_a, sdw_a = a["units"].cpu().numpy(), a["sdw"].cpu().numpy()
    nI, nr = okA.sum(1), okrA.sum(1)
    SU_a = np.where(okA, units_a.reshape(na, T), 0).sum(1)
    Sua = np.where(okA, ua.reshape(na, T), 0).sum(1)
    Suia = np.where(okA, uia.reshape(na, T), 0).sum(1)
    Ssdw = np.where(okA, sdw_a.reshape(na, T), 0).sum(1)
    Suia_r = np.where(okrA, uia.reshape(na, T), 0).sum(1)
    Ssdw_r = np.where(okrA, sdw_a.reshape(na, T), 0).sum(1)
    okA_t, okrA_t = K.f(okA, t.bool), K.f(okrA, t.bool)
    dead_a, scr_a = a["dead"], a["scr"]
    Dok = (dead_a.reshape(na, T, k) & okA_t[:, :, None]).sum(1).double()
    Dokr = (dead_a.reshape(na, T, k) & okrA_t[:, :, None]).sum(1).double()
    Sok = (scr_a.reshape(na, T, -1) & okA_t[:, :, None]).sum(1).double()
    Sokr = (scr_a.reshape(na, T, -1) & okrA_t[:, :, None]).sum(1).double()
    fa_cnt = _hist_max(K, la["rea"], okA_t)
    fa_band = fa_row.reshape(na, T).mean(1)
    # B, in `linear`'s draw chunks, grouped into device batches
    per = max(1, int(chunk_elems // (T * k)))
    chunks = [cols[c0:c0 + per] for c0 in range(0, len(cols), per)]
    group = max(1, K.batch // (per * T * k))
    nbt = len(cols)
    dead_b = t.empty((nbt * T, k), dtype=t.bool, device=K.dev)
    vec = {key: np.empty(nbt * T) for key in ("units", "ub", "uib", "uibn", "sdw", "db")}
    exb, exbn = np.empty(nbt * T, bool), np.empty(nbt * T, bool)
    band = {key: [] for key in ("Dall", "Dn", "Dc", "Sall", "Sc")}
    fb_cnt = np.empty(nbt, np.int64)
    wstat = dict(sb=[], off=0.0, blocks=0, unc=0)
    widx = t.as_tensor(np.array(cols, np.int64))
    for g0 in range(0, len(chunks), group):
        gch = chunks[g0:g0 + group]
        c_lo = sum(len(c) for c in chunks[:g0])
        nbands = sum(len(c) for c in gch)
        rows = (widx[c_lo:c_lo + nbands][:, None] * T + t.arange(T)[None, :]).reshape(-1)
        Wr = W[rows]
        Xb = Wr.to(K.dev).double()[:, :k]
        b = K.side(Xb, np.ascontiguousarray(Wr[:, :k:8].float().numpy(), np.float64))
        ehs, draws, at = [], [], 0
        for cc in gch:
            ehs.append(F.H[rng.integers(0, 16, (len(cc) * T, 32))])
            draws.append((at, at + len(cc) * T, F.H[rng.integers(0, 16, (k, 32))].astype(np.float64)))
            at += len(cc) * T
        lb = K.law_side(Xb, b, np.concatenate(ehs), draws)
        db, dib, sb = K.windows(lb["p"], lb["off"])
        ub, uib = 128 * db.sum(1), 128 * dib.sum(1)
        uibn = 128 * t.clamp(F.F1_B_OVERFIT * dib, max=1.0).sum(1)
        eb, ebn = uib > CAP * cw, uibn > CAP * cw
        r0, r1 = c_lo * T, (c_lo + nbands) * T
        dead_b[r0:r1] = b["dead"]
        for key, val in (("units", b["units"]), ("ub", ub), ("uib", uib), ("uibn", uibn), ("sdw", b["sdw"]),
                         ("db", db.mean(1))):
            vec[key][r0:r1] = val.cpu().numpy()
        exb[r0:r1], exbn[r0:r1] = eb.cpu().numpy(), ebn.cpu().numpy()
        dB = b["dead"].reshape(nbands, T, k)
        sB = b["scr"].reshape(nbands, T, -1)
        okcn, okc = (~ebn).reshape(nbands, T, 1), (~eb).reshape(nbands, T, 1)
        band["Dall"].append(dB.sum(1).double())
        band["Dn"].append((dB & okcn).sum(1).double())
        band["Dc"].append((dB & okc).sum(1).double())
        band["Sall"].append(sB.sum(1).double())
        band["Sc"].append((sB & okc).sum(1).double())
        fb_cnt[c_lo:c_lo + nbands] = _hist_max(K, lb["rea"])
        for a0, a1, _ in draws:
            wstat["sb"].append(float(sb[a0:a1].mean()))
        wstat["off"] += float(lb["off"].sum())
        wstat["blocks"] += lb["blocks"]
        wstat["unc"] += lb["uncertified"]
    band = {key: t.cat(v) for key, v in band.items()}
    dot = lambda x, y: (x @ y.T).cpu().numpy()
    t3e_n, P_n = dot(Dok, band["Dall"]), dot(Dokr, band["Dn"])
    Pc_n, t3_n, t3c_n = dot(Dokr, band["Dc"]), dot(Sok, band["Sall"]), dot(Sokr, band["Sc"])
    ncb = nbt
    rs = lambda v: v.reshape(ncb, T)
    SU_b, Sub, Suib = rs(vec["units"]).sum(1), rs(vec["ub"]).sum(1), rs(vec["uib"]).sum(1)
    okcnB, okcB = ~rs(exbn), ~rs(exb)
    nc_n, nc_c = okcnB.sum(1), okcB.sum(1)
    Suibn_n = np.where(okcnB, rs(vec["uibn"]), 0).sum(1)
    Suib_c = np.where(okcB, rs(vec["uib"]), 0).sum(1)
    db_band = rs(vec["db"]).mean(1)
    fa = fa_cnt / np.maximum(nI * (k // B), 1)
    fb = fb_cnt / (T * (k // B))
    # tiles, in `linear`'s order: per draw chunk, per A band (with a credited row), per weight band
    order = [(r, ci) for c0 in range(0, nbt, per) for r in range(na) if nI[r] > 0
             for ci in range(c0, min(c0 + per, nbt))]
    rows_o, rn, r1, r1e, vol_tiles, guard = [], [], [], [], [], []
    dA = dead_a
    for r, ci in order:
        cred = nI[r] * T * cw
        t1 = (T * SU_a[r] + nI[r] * SU_b[ci]) / cred
        f1 = (T * Sua[r] + nI[r] * Sub[ci]) / cred
        f1i = (T * Suia[r] + nI[r] * Suib[ci]) / cred
        t3 = B * t3_n[r, ci] / cred
        t3e = float(t3e_n[r, ci]) / cred
        t4 = T * FS / n * Ssdw[r] / cred
        sv = max(0.0, 1 - cL - 4.0 * (2 - fa[r] - fb[ci]))
        sv0 = max(0.0, 1 - 0.0 - 4.0 * (2 - fa[r] - fb[ci]))
        f2 = sv * nI[r] * T * k / cred
        rows_o.append((cred, t1 + t3 + t4, f1 + t3 + t4 + f2, f1i + t3 + t4 + f2, f1i + t3e + t4 + f2, f1, f1i, t3, t3e,
                       t4, f2, sv0, fa[r] + fb[ci], fa_band[r] + db_band[ci]))
        nrr, nc = int(nr[r]), int(nc_n[ci])
        if nrr == 0 or nc == 0:
            rn.append((cred, 0.0, 0.0))
        else:
            crn = nrr * nc * cw
            debn = ((nc * Suia_r[r] + nrr * Suibn_n[ci] + P_n[r, ci] + nc * FS / n * Ssdw_r[r]) / crn + sv * k / cw)
            rn.append((cred, crn if debn <= CAP else 0.0, debn))
            if debn > F.VOL_STORE * CAP:
                I, J = slice(r * T, (r + 1) * T), slice(ci * T, (ci + 1) * T)
                pairs = (dA[I].double() @ dead_b[J].double().T).cpu().numpy()
                vol_tiles.append(F.VolTile(r * T + np.arange(T), cols[ci] * T + np.arange(T), okrA[r], okcnB[ci],
                                           uia[I], vec["uibn"][J], pairs, sdw_a[I], sv))
            else:
                guard.append((r, cols[ci], nrr, nc, debn))
        nc = int(nc_c[ci])
        if nrr == 0 or nc == 0:
            r1.append((cred, 0.0, 0.0))
            r1e.append((cred, 0.0, 0.0))
            continue
        cr = nrr * nc * cw
        rest = ((nc * Suia_r[r] + nrr * Suib_c[ci]) / cr + nc * FS / n * Ssdw_r[r] / cr + sv * nrr * nc * k / cr)
        deb = rest + B * t3c_n[r, ci] / cr
        debe = rest + float(Pc_n[r, ci]) / cr
        r1.append((cred, cr if deb <= CAP else 0.0, deb))
        r1e.append((cred, cr if debe <= CAP else 0.0, debe))
    keys = ("cred", "base", "f1", "f1_in", "f1_in_t3e", "t_f1", "t_f1_in", "t3", "t3e", "t4", "f2", "f2_cl0", "fAB",
            "f1_chain")
    arr = np.array(rows_o, np.float64)
    o = {key: arr[:, j] for j, key in enumerate(keys)}
    exb_n = int(exb.sum())
    wst = dict(off=wstat["off"], blocks=wstat["blocks"], unc=wstat["unc"], exb=exb_n, nb=len(exb), sb=wstat["sb"],
               fb=vec["db"])
    s = dict(k=k, n=n, cols=cols, ntiles=ntiles, m=m, k_full=k_full, ok_all=ok_all, cL=cL, fa_row=fa_row,
             dia_row=dia.mean(1).cpu().numpy(), sa=float(sa.mean()), la_off=float(la["off"].double().mean()),
             la_unc=la["uncertified"], la_blocks=la["blocks"], exa=exa, cw=cw)
    return _finish(o, s, wst, rn, r1, r1e, vol_tiles, guard)


def _finish(o, s, wstat, rn, r1, r1e, vol_tiles, guard) -> dict:
    """The tail of `F.linear`, as it is there, on the port's per-tile figures."""
    T, CAP = F.T, F.CAP
    cred = o["cred"]
    tot = cred.sum()
    at = lambda d: dict((str(f), int((d > f * CAP).sum())) for f in (0.5, 0.8, 0.9, 1.0))  # noqa: E731
    fa_row, fb_row = s["fa_row"], wstat["fb"]
    m, n, k = s["m"], s["n"], s["k"]
    out = dict(k=k, n=n, tiles=len(cred), weight_tiles=len(s["cols"]), weight_tiles_total=s["ntiles"],
               macs=float(m * n * s["k_full"]), dnf_fail=float(1 - s["ok_all"].mean()), c_L=s["cL"], side_share=0.0,
               law="certified", f2_max=float(o["f2"].max()), f2_cl0_max=float(o["f2_cl0"].max()),
               fAB_max=float(o["fAB"].max()), fAB_median=float(np.median(o["fAB"])),
               tiles_f1_chain_over=int((o["f1_chain"] > CAP).sum()),
               f1_row_a=dict(mean=float(fa_row.mean()), p99=float(np.quantile(fa_row, 0.99)), max=float(fa_row.max())),
               f1_row_w=dict(mean=float(fb_row.mean()), max=float(fb_row.max())),
               f1_in_row_a_max=float(s["dia_row"].max()),
               S_w_mean=dict(a=s["sa"], w=float(np.mean(wstat["sb"]))),
               modal_byte_share=dict(a=float(1 - s["la_off"]), w=float(1 - wstat["off"] / wstat["blocks"])),
               uncertified=dict(a=s["la_unc"], a_blocks=s["la_blocks"], w=wstat["unc"], w_blocks=wstat["blocks"]),
               worst_row=int(np.argmax(fa_row)))
    for v in F.VARIANTS:
        d = o[v]
        iw = int(np.argmax(d))
        out[v] = dict(over=float(cred[d > CAP].sum() / tot), over_half=float(cred[d > CAP / 2].sum() / tot),
                      max=float(d.max()), mean=float((cred * d).sum() / tot), tiles_over=int((d > CAP).sum()),
                      tiles_at=at(d),
                      worst_terms={t: float(o[t][iw]) for t in ("t_f1", "t_f1_in", "t3", "t3e", "t4", "f2")})
    q = np.array(r1)
    out["r1"] = dict(uncredited=float(1 - q[:, 1].sum() / q[:, 0].sum()),
                     tiles_rejected=int(((q[:, 1] == 0) & (q[:, 2] > CAP)).sum()),
                     rows_excluded_a=float(s["exa"][:m].mean()),
                     rows_excluded_b=float(wstat["exb"] / max(wstat["nb"], 1)),
                     max_debit=float(q[:, 2].max()), tiles_at=at(q[:, 2]))
    qn = np.array(rn)
    va, vb, left = F.volunteer_rows(vol_tiles, m, n, s["cw"], k)
    over_guard = 0
    for r, c, nr, nc, d in guard:
        ka = sum(1 for g in va if r * T <= g < (r + 1) * T)
        kb = sum(1 for g in vb if c * T <= g < (c + 1) * T)
        if ka >= nr or kb >= nc or d * nr / (nr - ka) * nc / (nc - kb) > CAP:
            over_guard += 1
    out["norm"] = dict(uncredited=float(1 - qn[:, 1].sum() / qn[:, 0].sum()),
                       tiles_rejected=int(((qn[:, 1] == 0) & (qn[:, 2] > CAP)).sum()), max_debit=float(qn[:, 2].max()),
                       tiles_at=at(qn[:, 2]),
                       volunteer_a=len(va), volunteer_b=len(vb), volunteer_tiles_left=left,
                       volunteer_guard_failed=over_guard, volunteer_cap_a=-(-m // F.VOLUNTARY_PER),
                       volunteer_cap_b=-(-n // F.VOLUNTARY_PER))
    qe = np.array(r1e)
    out["r1_t3e"] = dict(uncredited=float(1 - qe[:, 1].sum() / qe[:, 0].sum()),
                         tiles_rejected=int(((qe[:, 1] == 0) & (qe[:, 2] > CAP)).sum()),
                         max_debit=float(qe[:, 2].max()), tiles_at=at(qe[:, 2]))
    out["vol_tiles"] = len(vol_tiles)
    return out


def compare(ref, got, path="") -> tuple[list[str], int, int, float]:
    """(mismatches, counts compared, figures compared, the largest relative difference of a figure) between two of
    `linear`'s dicts; a count (int, bool, str) must be equal, a figure (float) within RTOL."""
    bad, nd, nc, worst = [], 0, 0, 0.0
    if isinstance(ref, dict):
        for key in ref:
            if key == "seconds":
                continue
            if key not in got:
                bad.append(f"{path}/{key}: missing")
                continue
            b, d, c, w = compare(ref[key], got[key], f"{path}/{key}")
            bad, nd, nc, worst = bad + b, nd + d, nc + c, max(worst, w)
        return bad, nd, nc, worst
    if isinstance(ref, float) and not isinstance(got, bool):
        rel = abs(ref - got) / max(abs(ref), 1e-300) if ref != got else 0.0
        if rel > RTOL and abs(ref - got) > 1e-18:
            bad.append(f"{path}: {ref!r} vs {got!r}")
        return bad, 0, 1, rel if abs(ref - got) > 1e-18 else 0.0
    if ref != got:
        bad.append(f"{path}: {ref!r} vs {got!r}")
    return bad, 1, 0, 0.0


# ---- the census's inputs ---------------------------------------------------------------------------------------------

class Inputs:
    """Layer L's captured activations (`cov70b.py capture`) and its linears' BF16 weights (the pinned checkpoint)."""

    def __init__(self, acts: str, weights: str):
        self.acts, self.root = Path(acts), Path(weights)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self._layer, self._npz = None, None

    def names(self, layer: int) -> list[str]:
        return sorted(self.npz(layer).files)

    def npz(self, layer: int):
        if self._layer != layer:
            self._npz, self._layer = np.load(self.acts / f"layer-{layer:02d}.npz"), layer
        return self._npz

    def x(self, layer: int, nm: str) -> np.ndarray:
        import torch
        return torch.from_numpy(self.npz(layer)[nm]).view(torch.bfloat16).float().numpy()

    def w(self, nm: str):
        from safetensors import safe_open
        with safe_open(str(self.root / self.index[nm + ".weight"]), "pt") as f:
            return f.get_tensor(nm + ".weight")


def ranges(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def gate_cases(inp: Inputs, layers) -> list[tuple[int, int, str, int]]:
    """(layer, i, name, max_cols): each linear at the census's 8 weight bands; k_proj and v_proj at every band too."""
    cases = []
    for L in layers:
        for i, nm in enumerate(inp.names(L)):
            cases.append((L, i, nm, CENSUS_COLS))
            if nm.endswith(("k_proj", "v_proj")):
                cases.append((L, i, nm, 0))
    return cases


def case_name(L, i, nm, mc, salt=0) -> str:
    return f"s{salt:02d}--L{L:02d}--{i}-{nm.rsplit('.', 1)[-1]}--mc{mc}"


def queue_next(out: Path, job: str) -> None:
    """Copy the fill job `job` into the fill queue, once."""
    if not job:
        return
    try:
        fd = os.open(out / f"queued-{Path(job).name}", os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return
    os.close(fd)
    tmp = FILL_QUEUE / f".{Path(job).name}.tmp"
    tmp.write_bytes(Path(job).read_bytes())
    tmp.chmod(0o755)
    tmp.replace(FILL_QUEUE / Path(job).name)


def _ref_task(job):
    src, acts, weights, L, i, nm, mc = job
    load(src)
    inp = Inputs(acts, weights)
    t0 = time.time()
    r = F.linear(inp.x(L, nm), inp.w(nm), np.random.default_rng(seed(L, i, 0)), "certified", mc, 0, "nv", False)
    r["seconds"] = round(time.time() - t0, 1)
    return case_name(L, i, nm, mc), r


def cmd_ref(args) -> int:
    from multiprocessing import get_context
    inp = Inputs(args.acts, args.weights)
    d = Path(args.out) / "ref"
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(args.src, args.acts, args.weights, L, i, nm, mc) for L, i, nm, mc in gate_cases(inp, args.layers)
            if not (d / f"{case_name(L, i, nm, mc)}.json").exists()]
    jobs.sort(key=lambda j: -EST_S[j[5].rsplit(".", 1)[-1]] * (2 if j[6] == 0 else 1))
    with get_context("spawn").Pool(min(args.workers, max(len(jobs), 1))) as pool:
        for name, r in pool.imap_unordered(_ref_task, jobs):
            (d / f"{name}.tmp").write_text(json.dumps(r))
            (d / f"{name}.tmp").rename(d / f"{name}.json")
            print(f"ref {name}: {r['seconds']} s", flush=True)
    queue_next(Path(args.out), args.then)
    return 0


def _run_case(K, inp, L, i, nm, mc, salt):
    x, w = inp.x(L, nm), inp.w(nm)
    K.t.cuda.synchronize() if K.dev.type == "cuda" else None
    t0 = time.time()
    r = linear(K, x, w, np.random.default_rng(seed(L, i, salt)), mc)
    K.t.cuda.synchronize() if K.dev.type == "cuda" else None
    r["seconds"] = round(time.time() - t0, 2)
    return r


def cmd_gate(args) -> int:
    inp = Inputs(args.acts, args.weights)
    K = Port(args.device)
    out = Path(args.out)
    d = out / "gate"
    d.mkdir(parents=True, exist_ok=True)
    rep = {}
    for L, i, nm, mc in gate_cases(inp, args.layers):
        name = case_name(L, i, nm, mc)
        ref = json.loads((out / "ref" / f"{name}.json").read_text())
        got = json.loads(json.dumps(_run_case(K, inp, L, i, nm, mc, 0)))
        (d / f"{name}.json").write_text(json.dumps(got))
        bad, nd, nc, worst = compare(ref, got)
        rep[name] = dict(mismatches=bad, counts=nd, figures=nc, max_rel=worst, seconds=got["seconds"],
                         ref_seconds=ref.get("seconds"), vol_tiles=got["vol_tiles"],
                         volunteer=[got["norm"]["volunteer_a"], got["norm"]["volunteer_b"]],
                         tiles_rejected=got["norm"]["tiles_rejected"])
        print(f"gate {name}: {len(bad)} mismatches over {nd} counts and {nc} figures, max rel {worst:.2e}, "
              f"{got['seconds']} s (CPU {ref.get('seconds')} s)", flush=True)
        for b in bad[:10]:
            print("   ", b, flush=True)
    summ = dict(cases=len(rep), mismatches=sum(len(v["mismatches"]) for v in rep.values()),
                counts=sum(v["counts"] for v in rep.values()), figures=sum(v["figures"] for v in rep.values()),
                max_rel=max(v["max_rel"] for v in rep.values()),
                kinds=sorted({n.split("--")[2].split("-", 1)[1] for n in rep}), rtol=RTOL, device=str(K.dev),
                cases_detail=rep)
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    head = ("cases", "mismatches", "counts", "figures", "max_rel", "kinds")
    print(json.dumps({key: summ[key] for key in head}), flush=True)
    if summ["mismatches"]:
        return 1
    for job in args.then.split(",") if args.then else []:
        queue_next(out, job)
    return 0


EST_S = {"k_proj": 8, "v_proj": 8, "q_proj": 40, "o_proj": 40, "gate_proj": 150, "up_proj": 150, "down_proj": 150}


def cmd_run(args) -> int:
    inp = Inputs(args.acts, args.weights)
    K = Port(args.device)
    t0 = time.monotonic()
    todo = []
    for salt in args.salts:
        for L in args.layers:
            d = Path(args.out) / f"s{salt:02d}"
            for i, nm in enumerate(inp.names(L)):
                if not (d / f"{case_name(L, i, nm, 0, salt)}.json").exists():
                    todo.append((salt, L, i, nm))
    seen: dict[str, float] = {}
    for f in Path(args.out).glob("s[0-9][0-9]/*--mc0.json"):
        kind = f.stem.split("--")[2].split("-", 1)[1]
        seen[kind] = max(seen.get(kind, 0.0), json.loads(f.read_text())["seconds"] + 5)
    left, done = len(todo), 0
    for salt, L, i, nm in todo:
        kind = nm.rsplit(".", 1)[-1]
        est = 1.3 * seen.get(kind, EST_S.get(kind, 150))
        if done and time.monotonic() - t0 + est > 60 * args.budget_min:
            break
        t1 = time.monotonic()
        r = _run_case(K, inp, L, i, nm, 0, salt)
        seen[kind] = max(seen.get(kind, 0.0), time.monotonic() - t1)
        d = Path(args.out) / f"s{salt:02d}"
        d.mkdir(parents=True, exist_ok=True)
        name = case_name(L, i, nm, 0, salt)
        (d / f"{name}.tmp").write_text(json.dumps(r))
        (d / f"{name}.tmp").rename(d / f"{name}.json")
        left, done = left - 1, done + 1
        n = r["norm"]
        print(f"{name}: {r['tiles']} tiles, {r['seconds']} s | rejected {n['tiles_rejected']}, worst credited "
              f"{n['max_debit'] / F.CAP:.3f} of the cap, volunteer {n['volunteer_a']}+{n['volunteer_b']} rows, "
              f"uncertified {r['uncertified']['a']}+{r['uncertified']['w']}", flush=True)
    if left:
        print(f"{left} linears remain", flush=True)
        return 99
    queue_next(Path(args.out), args.then)
    return 0


def cmd_summary(args) -> int:
    out = Path(args.out)
    per = {}
    for salt in args.salts:
        files = sorted((out / f"s{salt:02d}").glob("s*--L*--mc0.json"))
        res = {}
        for f in files:
            _, lay, lin, _ = f.stem.split("--")
            res[f"{lay}.{lin}"] = json.loads(f.read_text())
        if not res:
            continue
        a = SimpleNamespace(model=MODEL, law="certified", max_cols=0, layer_step=1, split=False)
        s = F.summarize(a, res, TOKENS, time.time(), "nv")
        s["salt"], s["linears_expected"] = salt, 560
        s["gpu_seconds"] = sum(r["seconds"] for r in res.values())
        per[salt] = s
        n = s["norm"]
        print(f"salt {salt}: {s['linears']} linears, {s['tiles']} tiles | THE RULE: uncredited "
              f"{100 * n['uncredited']:.4f}% | tiles rejected {n['tiles_rejected']} | worst credited tile "
              f"{n['max_tile_debit'] / F.CAP:.3f} of the cap ({n['worst_linear']}) | over 0.5/0.8/0.9/1.0 of the cap "
              f"{'/'.join(str(v) for v in n['tiles_at_fraction_of_cap'].values())} | volunteer gives up "
              f"{n['volunteer_a']} A + {n['volunteer_b']} B rows in {n['volunteer_linears']} linears, "
              f"{n['volunteer_tiles_left']} tiles left over | uncertified blocks A {s['uncertified_blocks']['a']} of "
              f"{s['uncertified_blocks']['a_blocks']}, W {s['uncertified_blocks']['w']} of "
              f"{s['uncertified_blocks']['w_blocks']} | {s['gpu_seconds'] / 3600:.2f} GPU-h", flush=True)
    keys = ("uncredited", "tiles_rejected", "max_tile_debit", "volunteer_a", "volunteer_b", "volunteer_tiles_left")
    across = {key: [per[s]["norm"][key] for s in per] for key in keys}
    spread = {key: dict(min=min(v), max=max(v), mean=float(np.mean(v))) for key, v in across.items() if v}
    summ = dict(salts=sorted(per), per_salt=per, across_salts=spread)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["ref", "gate", "run", "summary"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--acts", default="/workspace/pouw/fill-out/fp4-coverage-70b/acts")
    ap.add_argument("--weights", default="/workspace/hf/hub/models--unsloth--Meta-Llama-3.1-70B/snapshots/"
                                         "1b7306651142d0cc65d993076a250a6a82cf046c")
    ap.add_argument("--layers", default="0,79")
    ap.add_argument("--salts", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget-min", type=float, default=6.5)
    ap.add_argument("--then", default="")
    ap.add_argument("--no-check", action="store_true", help="skip the sources' sha256 check (tests)")
    args = ap.parse_args(argv)
    load(args.src, not args.no_check)
    args.layers, args.salts = ranges(args.layers), ranges(args.salts)
    return {"ref": cmd_ref, "gate": cmd_gate, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
