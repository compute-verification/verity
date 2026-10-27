"""Quantizers (bf16, FP8 E4M3 per-tensor, MXFP8, NVFP4) as indices into sorted grids, and the entropy estimators."""
import lzma
import torch

NB = 16  # position-in-cell context bins


def _grid(vals):
    g = torch.unique(vals[torch.isfinite(vals)].float())
    mids = (g[1:] + g[:-1]) / 2
    lo = torch.cat([g[:1] - (g[1] - g[0]) / 2, mids])
    hi = torch.cat([mids, g[-1:] + (g[-1] - g[-2]) / 2])
    return {"g": g, "mids": mids, "lo": lo, "hi": hi}


G_BF16 = _grid(torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
_e4 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
G_E4M3 = _grid(_e4)
G_E4M3_POS = _grid(_e4[_e4 >= 0])
_e2 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
G_E2M1 = _grid(torch.cat([-_e2, _e2]))


def q(y, G):
    """nearest grid index (saturating) and position-in-cell bin"""
    idx = torch.searchsorted(G["mids"], y.contiguous())
    pos = (y - G["lo"][idx]) / (G["hi"][idx] - G["lo"][idx])
    b = (pos.clamp(0, 0.9999) * NB).long()
    return idx, b


FORMATS = {
    "bf16": dict(kind="plain"),
    "fp8t": dict(kind="tensor"),  # E4M3, per-tensor scale from the registered tensor (delayed scaling)
    "mxfp8": dict(kind="block", B=32, elem=G_E4M3, emax=8, scale="e8m0"),
    "nvfp4": dict(kind="block", B=16, elem=G_E2M1, emax=None, scale="e4m3"),
}
BITS = {"bf16": 16, "fp8t": 8, "mxfp8": 8, "nvfp4": 4}


def block_scales(X, f, ref_amax):
    """per-block scale code (int64) and value, X shaped (..., nb, B)"""
    amax = X.abs().amax(-1)
    if f["scale"] == "e8m0":
        code = torch.floor(torch.log2(amax.clamp_min(1e-30))).long() - f["emax"]
        return code, torch.exp2(code.float())
    g = ref_amax / (6 * 448)
    idx, _ = q(amax / (6 * g), G_E4M3_POS)
    return idx, G_E4M3_POS["g"][idx] * g


def encode(X, fmt, ref_amax, scale=None):
    """dict: code (per value), deq, ctx bin, per-block scode (None for unblocked).
    `scale` = (scode, sval) forces the block scales (to predict elements under the fresh scales)."""
    f = FORMATS[fmt]
    if f["kind"] == "plain":
        idx, b = q(X, G_BF16)
        return dict(code=idx, ctx=b, deq=G_BF16["g"][idx], scode=None)
    if f["kind"] == "tensor":
        s = ref_amax / 448
        idx, b = q(X / s, G_E4M3)
        return dict(code=idx, ctx=b, deq=G_E4M3["g"][idx] * s, scode=None)
    B = f["B"]
    Xb = X.reshape(*X.shape[:-1], X.shape[-1] // B, B)
    scode, sval = scale if scale is not None else block_scales(Xb, f, ref_amax)
    y = torch.where(sval[..., None] > 0, Xb / sval[..., None].clamp_min(1e-38), torch.zeros_like(Xb))
    idx, b = q(y, f["elem"])
    deq = f["elem"]["g"][idx] * sval[..., None]
    return dict(code=idx.reshape(X.shape), ctx=b.reshape(X.shape), deq=deq.reshape(X.shape),
                scode=scode, sval=sval)


def H(x):
    _, c = torch.unique(x.reshape(-1), return_counts=True)
    p = c.double() / c.sum()
    return float(-(p * torch.log2(p)).sum())


def Hcond(a, b):
    """H(a|b), plug-in"""
    a = a.reshape(-1).long(); b = b.reshape(-1).long()
    a = a - a.min(); b = b - b.min()
    return H(a * (int(b.max()) + 1) + b) - H(b)


def lzma_bits(t):
    t = t.reshape(-1)
    lo, hi = int(t.min()), int(t.max())
    dt = torch.int8 if -128 <= lo and hi <= 127 else torch.int16
    return 8 * len(lzma.compress(t.to(dt).numpy().tobytes(), preset=6))


PER_BINADE = {"bf16": 128, "fp8t": 8, "mxfp8": 8, "nvfp4": 1}


def ctx_of(P, fmt):
    """context for the residual: the predicted code's binade and its position in the cell"""
    return (P["code"] // PER_BINADE[fmt]) * NB + P["ctx"]


def fresh_bits(Xhat, E1, fmt, ref_amax):
    """two-part code for the fresh encoding E1 given a registered/predicted float Xhat: (block scales of E1 given Xhat's
    scales) + (element codes of E1 given Xhat quantized under E1's scales, conditioned on Xhat's position in its cell).
    Returns total bits (entropy) and the residual streams."""
    if E1["scode"] is None:
        P = encode(Xhat, fmt, ref_amax)
        d = E1["code"] - P["code"]
        return Hcond(d, ctx_of(P, fmt)) * d.numel(), d, None
    f = FORMATS[fmt]
    B = f["B"]
    Xb = Xhat.reshape(*Xhat.shape[:-1], Xhat.shape[-1] // B, B)
    sc_hat, _ = block_scales(Xb, f, ref_amax)
    ds = E1["scode"] - sc_hat
    P = encode(Xhat, fmt, ref_amax, scale=(E1["scode"], E1["sval"]))
    d = E1["code"] - P["code"]
    return Hcond(d, ctx_of(P, fmt)) * d.numel() + H(ds) * ds.numel(), d, ds


def combined_code(E, fmt):
    if E["scode"] is None:
        return E["code"]
    B = FORMATS[fmt]["B"]
    sc = E["scode"].repeat_interleave(B, dim=-1).reshape(E["code"].shape)
    sc = sc - sc.min()
    return sc * 1000 + E["code"]


def analyze_pair(X0, X1, fmts, mdl_ranks=(), want_lzma=True, want_svd=True):
    """X0 (registered), X1 (fresh): float32 tensors (N, d). Per-format metrics in bits per value."""
    N = X0.numel()
    ref_amax = X0.abs().max()
    out = {}
    dX = X1 - X0
    if want_svd:
        U, S, Vh = torch.linalg.svd(dX, full_matrices=False)
        e = (S.double() ** 2).cumsum(0)
        e = e / e[-1] if e[-1] > 0 else e
        out["r90"] = int((e < 0.90).sum()) + 1 if e[-1] > 0 else 0
        out["r99"] = int((e < 0.99).sum()) + 1 if e[-1] > 0 else 0
        out["rel_dnorm"] = float(dX.norm() / X0.norm())
    for fmt in fmts:
        E0 = encode(X0, fmt, ref_amax)
        E1 = encode(X1, fmt, ref_amax)
        m = {}
        m["frac"] = float((E0["deq"] != E1["deq"]).float().mean())
        c0, c1 = combined_code(E0, fmt), combined_code(E1, fmt)
        m["H_cond"] = Hcond(c1, c0)
        bits0, d, ds = fresh_bits(X0, E1, fmt, ref_amax)
        m["H_ctx"] = bits0 / N
        m["H_delta"] = H(d) + (H(ds) * ds.numel() / N if ds is not None else 0)
        if ds is not None:
            m["frac_scale"] = float((ds != 0).float().mean())
        if want_lzma:
            lb = min(lzma_bits(d), lzma_bits(d.T.contiguous()))
            if ds is not None:
                lb += lzma_bits(ds)
            m["lzma"] = lb / N
            raw = lzma_bits(E1["code"]) + (lzma_bits(E1["scode"]) if ds is not None else 0)
            m["lzma_raw"] = raw / N
        if mdl_ranks and want_svd:
            T, D = X0.shape
            best = (bits0, 0)
            for r in mdl_ranks:
                if r >= min(T, D):
                    break
                A = (U[:, :r] * S[:r]).bfloat16().float()
                Bm = Vh[:r].bfloat16().float()
                bits, _, _ = fresh_bits(X0 + A @ Bm, E1, fmt, ref_amax)
                bits += r * (T + D) * 16
                if bits < best[0]:
                    best = (bits, r)
            m["mdl"] = best[0] / N
            m["mdl_r"] = best[1]
            m["naive_lowrank"] = out["r99"] * (T + D) * BITS[fmt] / N
        out[fmt] = m
    return out
