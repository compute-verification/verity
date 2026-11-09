"""Certified sensitivity of the logits to one in-range fault, beside its true worst case.

A fault sets one coordinate of one site, at a prompt's last position t, to any value in its range; everything else is
honest, and every read is clamped to `clamp.py`'s boxes (`lm.Boxes`). The range is the readers' box (`box`); at a
linear's input, one FP8 code at its token's own group scale (`code`: what a block writing one 16-bit value can put
there); or the values the --prompts' honest activations took at that coordinate, without headroom (`seen`: narrower
than any clamp that would pass honest serving). The coordinates are --coords random ones and the --widest boxes'. The
sites are the final norm's output and, for each depth k of --depths, the residual stream into layer L - k (`resid`)
and that layer's MLP hidden units (`mlph`).

At each of the --reach positions from t (t's own logits, then the next positions', which read t through attention):

  true  over --grid values spanning the range, through the clamped model: the largest move of any logit (delta), of
        any token's log-probability at temperature 1 (eps), and whether the greedy token changes. Any certificate must
        allow at least these.
  cert  interval bound propagation (IBP) from the range to the logits through the same layers, with the range split
        into P equal parts for each P of --pieces (a part's bound holds for every value in it, and a fault's bound is
        its worst part's): bounds on the same three that hold for every value in the range. Under --iv-clamp each
        site's interval is also cut to its box, as the readers' clamps cut its values.

eps bounds every served token's log-likelihood ratio at temperature 1 (at T it scales about as 1/T). Over the reach a
fault's zCDP charge is log2(e) * sum_n eps_n^2 / 2 bits (`rho_bits`), its pure-DP charge log2(e) * sum_n eps_n
(`pure_bits`), and its KL charge (`kl_bits`) log2(e) times the sum of the tokens' KL divergences (true) or of
osc_n^2 / 8, Hoeffding's bound on them (cert). With --reach-true N the true worst case is also taken over N positions
from t, the later ones following the honest greedy tokens (`true_long`).

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/certify.py --model gpt2 --boxes boxes.pt --out "$OUT"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import E4M3_MAX, GROUP, GROUP_SITES, LM, MODELS, Boxes, Cache, Fault, ue8m0_scale
from prompts import PROMPTS

LOG2E = 1 / math.log(2)
INF = float("inf")
Iv = tuple[torch.Tensor, torch.Tensor]


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _valley(f) -> tuple[float, float]:
    """The minimum of a function that falls to it and rises after (GELU, SiLU): where it is, and its value."""
    xs = torch.linspace(-4.0, 0.0, 400_001, dtype=torch.float64)
    ys = f(xs)
    i = int(ys.argmin())
    grid = torch.linspace(-40.0, 40.0, 80_001, dtype=torch.float64)
    g = f(grid)
    left, right = g[grid <= xs[i]], g[grid >= xs[i]]
    if not (bool((left[1:] <= left[:-1] + 1e-12).all()) and bool((right[1:] >= right[:-1] - 1e-12).all())):
        raise ValueError("the activation is not unimodal around its minimum; its interval bound would be unsound")
    return float(xs[i]), float(ys[i])


VALLEYS = {"gelu": (_gelu, *_valley(_gelu)), "silu": (F.silu, *_valley(F.silu))}


def act(x: Iv, name: str) -> Iv:
    """The exact range of a scalar activation over each interval: it falls to its valley and rises after it."""
    f, xv, yv = VALLEYS[name]
    lo, hi = x
    flo = torch.where(lo == -INF, torch.zeros_like(lo), f(lo))
    fhi = torch.where(hi == INF, torch.full_like(hi, INF), f(hi))
    inside = (lo <= xv) & (hi >= xv)
    return torch.where(inside, torch.full_like(lo, yv), torch.minimum(flo, fhi)), torch.maximum(flo, fhi)


def mul(a: Iv, b: Iv) -> Iv:
    p = torch.stack([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]).nan_to_num(nan=0.0)
    return p.amin(0), p.amax(0)


def add(a: Iv, b: Iv) -> Iv:
    return a[0] + b[0], a[1] + b[1]


def excl_sum(a: torch.Tensor) -> torch.Tensor:
    """sum over k != i of a[..., k], without subtracting (exact for infinities and dominant terms)."""
    z = torch.zeros_like(a[..., :1])
    pre = torch.cat([z, a[..., :-1]], -1).cumsum(-1)
    suf = torch.cat([a[..., 1:], z], -1).flip(-1).cumsum(-1).flip(-1)
    return pre + suf


def norm(x: Iv, w: torch.Tensor, b: torch.Tensor | None, eps: float, center: bool) -> Iv:
    """LayerNorm (center) or RMSNorm over intervals. Output i is f(d_i, S) = d_i / sqrt((d_i^2 + S) / n + eps), with
    d_i the centred (or raw) input and S the other coordinates' sum of squares. f rises in d_i and falls in S where
    d_i > 0 (rises where d_i < 0), so its range over the enclosures of d_i and S is at their corners; |f| <= sqrt(n)."""
    lo, hi = x[0].double(), x[1].double()
    n = lo.shape[-1]
    if center:
        c, r = (lo + hi) / 2, (hi - lo) / 2
        dc = c - c.mean(-1, keepdim=True)
        dr = r * (1 - 2 / n) + r.sum(-1, keepdim=True) / n
        lo, hi = dc - dr, dc + dr
    zero_in = (lo <= 0) & (hi >= 0)
    sq_lo = torch.where(zero_in, torch.zeros_like(lo), torch.minimum(lo * lo, hi * hi))
    sq_hi = torch.maximum(lo * lo, hi * hi)
    s_lo, s_hi = excl_sum(sq_lo), excl_sum(sq_hi)

    def f(d, s):
        return d / torch.sqrt((d * d + s) / n + eps)

    lim = math.sqrt(n)
    fmax = f(hi, torch.where(hi >= 0, s_lo, s_hi)).nan_to_num(nan=lim).clamp(-lim, lim)
    fmin = f(lo, torch.where(lo <= 0, s_lo, s_hi)).nan_to_num(nan=-lim).clamp(-lim, lim)
    wd = w.double()
    ylo = torch.where(wd >= 0, fmin * wd, fmax * wd)
    yhi = torch.where(wd >= 0, fmax * wd, fmin * wd)
    if b is not None:
        ylo, yhi = ylo + b.double(), yhi + b.double()
    return ylo.to(x[0].dtype), yhi.to(x[0].dtype)


def rope(x: Iv, inv_freq: torch.Tensor, pos0: int) -> Iv:
    """lm.LM._rope over intervals: a rotation, so the radius goes through |cos| and |sin|."""
    lo, hi = x
    t = lo.shape[2]
    freqs = (pos0 + torch.arange(t, dtype=torch.float32))[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(lo.dtype), emb.sin().to(lo.dtype)
    c, r = (lo + hi) / 2, (hi - lo) / 2
    h = c.shape[-1] // 2
    yc = c * cos + torch.cat([-c[..., h:], c[..., :h]], -1) * sin
    yr = r * cos.abs() + torch.cat([r[..., h:], r[..., :h]], -1) * sin.abs()
    return yc - yr, yc + yr


def simplex(w_lo: torch.Tensor, w_hi: torch.Tensor, vals: torch.Tensor, upper: bool) -> torch.Tensor:
    """max (upper) or min over w with w_lo <= w <= w_hi and sum(w) = 1 of sum_m w_m vals_m, per coordinate.
    w: (..., M); vals: (..., M, e). Fill the mass left over w_lo into the best values first."""
    base = (w_lo[..., None] * vals).nan_to_num(nan=0.0).sum(-2)
    rem = (1 - w_lo.sum(-1)).clamp_min(0)[..., None, None]
    cap = (w_hi - w_lo).clamp_min(0)[..., None].expand_as(vals)
    order = vals.argsort(dim=-2, descending=upper)
    sv, sc = vals.gather(-2, order), cap.gather(-2, order)
    before = sc.cumsum(-2) - sc
    alloc = torch.minimum((rem - before).clamp_min(0), sc)
    return base + (alloc * sv).nan_to_num(nan=0.0).sum(-2)


def attend(q: Iv, k: Iv, v: Iv, kpre: torch.Tensor, vpre: torch.Tensor) -> Iv:
    """Causal attention of the R suffix queries over the honest prefix (kpre, vpre: (1, KVH, t, e) points) and the
    suffix's own keys and values (intervals), heads (K, H, R, e). Scores are interval dot products; each weight's bound
    is its score at one end against every other at the other end; the output is bounded over the weights' box
    intersected with the simplex."""
    hq, hk = q[0].shape[1], k[0].shape[1]
    rep = hq // hk
    kpre, vpre = kpre.repeat_interleave(rep, 1), vpre.repeat_interleave(rep, 1)
    k = tuple(z.repeat_interleave(rep, 1) for z in k)
    v = tuple(z.repeat_interleave(rep, 1) for z in v)
    e = q[0].shape[-1]
    sc = e ** -0.5
    qc, qr = (q[0] + q[1]) / 2, (q[1] - q[0]) / 2
    spc = qc @ kpre.transpose(-1, -2)
    spr = qr @ kpre.abs().transpose(-1, -2)
    a, b = q[0][:, :, :, None, :], q[1][:, :, :, None, :]
    c, d = k[0][:, :, None, :, :], k[1][:, :, None, :, :]
    prods = torch.stack([a * c, a * d, b * c, b * d]).nan_to_num(nan=0.0)
    s_lo = torch.cat([spc - spr, prods.amin(0).sum(-1)], -1).double() * sc
    s_hi = torch.cat([spc + spr, prods.amax(0).sum(-1)], -1).double() * sc
    K, H, R, M = s_lo.shape
    t = M - R
    allowed = torch.ones(R, M, dtype=torch.bool)
    allowed[:, t:] = torch.tril(torch.ones(R, R, dtype=torch.bool))
    s_lo = s_lo.masked_fill(~allowed, -INF).nan_to_num(nan=-INF)
    s_hi = s_hi.masked_fill(~allowed, -INF).nan_to_num(nan=INF)
    m = s_hi.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    eu, el = (s_hi - m).exp(), (s_lo - m).exp()
    w_lo = el / (el + excl_sum(eu)).clamp_min(1e-300)
    w_hi = eu / (eu + excl_sum(el)).clamp_min(1e-300)
    w_hi = torch.where(torch.isnan(w_hi) | (eu == INF), torch.ones_like(w_hi), w_hi).clamp(0, 1)
    w_lo = w_lo.nan_to_num(nan=0.0).clamp(0, 1)
    w_lo, w_hi = w_lo.masked_fill(~allowed, 0.0).float(), w_hi.masked_fill(~allowed, 0.0).float()
    vp = vpre.expand(K, H, t, e)[:, :, None].expand(K, H, R, t, e)
    vl = torch.cat([vp, v[0][:, :, None].expand(K, H, R, R, e)], 3)
    vh = torch.cat([vp, v[1][:, :, None].expand(K, H, R, R, e)], 3)
    return simplex(w_lo, w_hi, vl, upper=False), simplex(w_lo, w_hi, vh, upper=True)


class Ctx:
    """The interval forward's state: the model, the boxes when intervals are cut to them, the suffix positions'
    classes, the honest prefix's keys and values, and |W| for each weight read."""

    absw: dict[str, torch.Tensor] = {}

    def __init__(self, lm: LM, boxes: Boxes | None, cls: torch.Tensor, pre: Cache, pos0: int):
        self.lm, self.boxes, self.cls, self.pre, self.pos0 = lm, boxes, cls, pre, pos0

    def lin(self, x: Iv, name: str, bias: str | None = None, wt: bool = False) -> Iv:
        W = self.lm.E if name == "E" else self.lm.W[name]
        if name not in self.absw:
            self.absw[name] = W.abs()
        A = self.absw[name]
        c, r = (x[0] + x[1]) / 2, (x[1] - x[0]) / 2
        yc = c @ (W.T if wt else W)
        if bias is not None:
            yc = yc + self.lm.W[bias]
        yr = r @ (A.T if wt else A)
        return (yc - yr).nan_to_num(nan=-INF), (yc + yr).nan_to_num(nan=INF)

    def clamp(self, site: str, layer: int, x: Iv) -> Iv:
        """The readers' clamp over intervals: a group site's group is scaled by a power of two in [fmin, 1], then every
        value is within its box."""
        if self.boxes is None or (site, layer) not in self.boxes.hi:
            return x
        hb = self.boxes.hi[(site, layer)][self.cls].to(x[0].dtype)
        lo, hi = x
        if site in GROUP_SITES:
            K, R, w = lo.shape
            gl, gh = lo.reshape(K, R, w // GROUP, GROUP), hi.reshape(K, R, w // GROUP, GROUP)
            amax = torch.maximum(gl.abs(), gh.abs()).amax(-1)
            smax = hb.reshape(R, w // GROUP, GROUP)[..., 0] / E4M3_MAX
            s = ue8m0_scale(amax.float()).to(lo.dtype)
            fmin = torch.where(s > smax, smax / s, torch.ones_like(s)).nan_to_num(nan=0.0)[..., None]
            cands = torch.stack([gl, gh, gl * fmin, gh * fmin]).nan_to_num(nan=0.0)
            lo, hi = cands.amin(0).reshape(K, R, w), cands.amax(0).reshape(K, R, w)
        return torch.maximum(torch.minimum(lo, hb), -hb), torch.maximum(torch.minimum(hi, hb), -hb)


def heads(x: Iv, n: int, e: int) -> Iv:
    K, R, _ = x[0].shape
    return tuple(z.reshape(K, R, n, e).transpose(1, 2) for z in x)


def unheads(x: Iv) -> Iv:
    K, n, R, e = x[0].shape
    return tuple(z.transpose(1, 2).reshape(K, R, n * e) for z in x)


def iv_attention_part(ctx: Ctx, l: int, x: Iv) -> Iv:
    """Layer l up to its residual stream after attention (`mid`), over intervals, for the suffix positions."""
    lm, s = ctx.lm, ctx.lm.spec
    W = lm.W
    if s.arch == "gpt2":
        p = f"h.{l}."
        h = ctx.clamp("ln1", l, norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"], s.eps, True))
        qkv = ctx.lin(h, p + "attn.c_attn.weight", p + "attn.c_attn.bias")
        q, k, v = (tuple(z[..., i * s.d:(i + 1) * s.d] for z in qkv) for i in range(3))
        q, k, v = heads(q, s.heads, s.head_dim), heads(k, s.heads, s.head_dim), heads(v, s.heads, s.head_dim)
    else:
        p = f"layers.{l}."
        h = ctx.clamp("ln1", l, norm(x, W[p + "input_layernorm.weight"], None, s.eps, False))
        q = heads(ctx.lin(h, p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias", wt=True), s.heads, s.head_dim)
        k = heads(ctx.lin(h, p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias", wt=True), s.kv_heads,
                  s.head_dim)
        v = heads(ctx.lin(h, p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias", wt=True), s.kv_heads,
                  s.head_dim)
        q, k = rope(q, lm.inv_freq, ctx.pos0), rope(k, lm.inv_freq, ctx.pos0)
    k = heads(ctx.clamp("k", l, unheads(k)), k[0].shape[1], s.head_dim)
    v = heads(ctx.clamp("v", l, unheads(v)), v[0].shape[1], s.head_dim)
    kpre, vpre = ctx.pre.kv[l]
    a = ctx.clamp("attnh", l, unheads(attend(q, k, v, kpre, vpre)))
    if s.arch == "gpt2":
        a = ctx.lin(a, p + "attn.c_proj.weight", p + "attn.c_proj.bias")
    else:
        a = ctx.lin(a, p + "self_attn.o_proj.weight", wt=True)
    return ctx.clamp("mid", l, add(x, ctx.clamp("attn", l, a)))


def iv_mlp_hidden(ctx: Ctx, l: int, x: Iv) -> Iv:
    s, W = ctx.lm.spec, ctx.lm.W
    if s.arch == "gpt2":
        p = f"h.{l}."
        h = ctx.clamp("ln2", l, norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"], s.eps, True))
        m = act(ctx.lin(h, p + "mlp.c_fc.weight", p + "mlp.c_fc.bias"), "gelu")
    else:
        p = f"layers.{l}."
        h = ctx.clamp("ln2", l, norm(x, W[p + "post_attention_layernorm.weight"], None, s.eps, False))
        m = mul(act(ctx.lin(h, p + "mlp.gate_proj.weight", wt=True), "silu"),
                ctx.lin(h, p + "mlp.up_proj.weight", wt=True))
    return ctx.clamp("mlph", l, m)


def iv_mlp_out(ctx: Ctx, l: int, x: Iv, m: Iv) -> Iv:
    if ctx.lm.spec.arch == "gpt2":
        o = ctx.lin(m, f"h.{l}.mlp.c_proj.weight", f"h.{l}.mlp.c_proj.bias")
    else:
        o = ctx.lin(m, f"layers.{l}.mlp.down_proj.weight", wt=True)
    return ctx.clamp("resid", l, add(x, ctx.clamp("mlp", l, o)))


def iv_head(ctx: Ctx, x: Iv | None, h: Iv | None = None) -> Iv:
    """Logits' intervals from the residual stream after the last layer (x), or from the final norm's output (h)."""
    s, W = ctx.lm.spec, ctx.lm.W
    if h is None:
        if s.arch == "gpt2":
            h = norm(x, W["ln_f.weight"], W["ln_f.bias"], s.eps, True)
        else:
            h = norm(x, W["norm.weight"], None, s.eps, False)
        h = ctx.clamp("final", s.layers, h)
    return ctx.lin(h, "E", wt=True)


def iv_forward(ctx: Ctx, site: str, layer: int, entry: dict, coord: int, parts: torch.Tensor) -> Iv:
    """Logits' intervals (P, R, V) for the fault at (site, layer, coord) at the first suffix position, its value in
    each of the P parts [parts[i], parts[i + 1]]; `entry` holds the honest activations the forward starts from."""
    L = ctx.lm.spec.layers
    P = len(parts) - 1

    def faulted(z: torch.Tensor) -> Iv:
        lo, hi = z.expand(P, -1, -1).clone(), z.expand(P, -1, -1).clone()
        lo[:, 0, coord], hi[:, 0, coord] = parts[:-1].to(z.dtype), parts[1:].to(z.dtype)
        return lo, hi

    if site == "final":
        return iv_head(ctx, None, faulted(entry["final"]))
    if site == "mlph":
        mid = entry["mid"].expand(P, -1, -1)
        x = iv_mlp_out(ctx, layer, (mid, mid), faulted(entry["mlph"]))
        first = layer + 1
    else:
        x, first = faulted(entry["resid_in"]), layer + 1
    for l in range(first, L):
        x = iv_attention_part(ctx, l, x)
        x = iv_mlp_out(ctx, l, x, iv_mlp_hidden(ctx, l, x))
    return iv_head(ctx, x)


def ranges(lm: LM, site: str, layer: int, x: torch.Tensor, cls: int, kinds: list[str],
           seen: dict[tuple[str, int], Iv]) -> dict[str, Iv]:
    """Per kind, each coordinate's range [lo, hi] at a position of class cls, x its honest activation there as read.
    box: the clamp's. code: one FP8 code at plus or minus 448 times the token's own group scale. seen: the values the
    prompts' honest activations took there, without headroom: narrower than any deployable clamp."""
    hb = lm.boxes.hi[(site, layer)][cls].float()
    out = {}
    for kind in kinds:
        if kind == "box":
            out[kind] = (-hb, hb)
        elif kind == "code" and site in GROUP_SITES:
            own = torch.minimum(E4M3_MAX * ue8m0_scale(x.reshape(-1, GROUP).abs().amax(-1).float()).repeat_interleave(
                GROUP), hb)
            out[kind] = (-own, own)
        elif kind == "seen" and (site, layer) in seen:
            lo, hi = seen[(site, layer)]
            out[kind] = (torch.minimum(lo, x), torch.maximum(hi, x))
    return out


def metrics(zh: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> dict[str, torch.Tensor]:
    """From the honest logits zh (R, V) and the faulted logits' bounds (P, R, V): per part and position, the largest
    logit move (delta), the log-likelihood ratio bound at temperature 1 (eps), the logit spread (osc), and whether
    the greedy token is certified."""
    zh = zh.double()
    a, b = lo.double() - zh, hi.double() - zh
    lp = torch.log_softmax(zh, -1)
    up = b.amax(-1) - torch.logsumexp(lp + a, -1)
    dn = torch.logsumexp(lp + b, -1) - a.amin(-1)
    top = zh.argmax(-1)
    others = hi.double().clone()
    others.scatter_(-1, top[None, :, None].expand(others.shape[0], -1, 1), -INF)
    greedy = lo.double().gather(-1, top[None, :, None].expand(lo.shape[0], -1, 1))[..., 0] > others.amax(-1)
    return {"delta": torch.maximum(a.abs(), b.abs()).amax(-1), "eps": torch.maximum(up, dn).clamp_min(0),
            "osc": b.amax(-1) - a.amin(-1), "greedy": greedy}


def selftest(n: int = 300, seed: int = 1) -> None:
    """Each interval primitive against its function at random points inside random intervals."""
    g = torch.Generator().manual_seed(seed)

    def box(*shape: int, scale: float = 3.0) -> Iv:
        c = torch.randn(*shape, generator=g) * scale
        r = torch.rand(*shape, generator=g) * scale * (torch.rand(*shape, generator=g) < 0.5)
        return c - r, c + r

    def sample(iv: Iv) -> torch.Tensor:
        return iv[0] + torch.rand(n, *iv[0].shape, generator=g) * (iv[1] - iv[0])

    def inside(y: torch.Tensor, iv: Iv, what: str) -> None:
        tol = 1e-4 * (1 + y.abs())
        if bool(((y < iv[0] - tol) | (y > iv[1] + tol)).any()):
            raise AssertionError(f"selftest: {what} leaves its interval bound")

    x, y = box(64), box(64)
    xs, ys = sample(x), sample(y)
    for name, (f, _, _) in VALLEYS.items():
        inside(f(xs), act(x, name), name)
    inside(xs * ys, mul(x, y), "mul")
    w, b = torch.randn(64, generator=g), torch.randn(64, generator=g)
    xn = tuple(z.clone() for z in box(3, 64))
    xn[0][:, 5] -= 50
    xn[1][:, 5] += 50
    xns = sample(xn)
    inside(F.layer_norm(xns, (64,), w, b, 1e-5), norm(xn, w, b, 1e-5, True), "layer_norm")
    inside(w * (xns * torch.rsqrt(xns.pow(2).mean(-1, keepdim=True) + 1e-6)), norm(xn, w, None, 1e-6, False),
           "rms_norm")
    inv = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
    xr = box(2, 3, 4, 8)
    xrs = sample(xr)
    freqs = (5 + torch.arange(4, dtype=torch.float32))[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], -1)
    inside(xrs * emb.cos() + torch.cat([-xrs[..., 4:], xrs[..., :4]], -1) * emb.sin(), rope(xr, inv, 5), "rope")
    K, H, KVH, R, t, e = 2, 4, 2, 3, 5, 8
    q, k, v = box(K, H, R, e, scale=1.0), box(K, KVH, R, e, scale=1.0), box(K, KVH, R, e, scale=1.0)
    kpre, vpre = torch.randn(1, KVH, t, e, generator=g), torch.randn(1, KVH, t, e, generator=g)
    bound = attend(q, k, v, kpre, vpre)
    qs, ks, vs = sample(q), sample(k), sample(v)
    kk = torch.cat([kpre.expand(n * K, -1, -1, -1), ks.reshape(n * K, KVH, R, e)], 2).repeat_interleave(H // KVH, 1)
    vv = torch.cat([vpre.expand(n * K, -1, -1, -1), vs.reshape(n * K, KVH, R, e)], 2).repeat_interleave(H // KVH, 1)
    mask = torch.ones(R, t + R, dtype=torch.bool)
    mask[:, t:] = torch.tril(torch.ones(R, R, dtype=torch.bool))
    out = F.scaled_dot_product_attention(qs.reshape(n * K, H, R, e), kk, vv, attn_mask=mask)
    inside(out.reshape(n, K, H, R, e), bound, "attention")


def main() -> int:
    selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depths", default="1,2,4,8", help="k: the residual stream into layer L - k, its MLP hidden units")
    ap.add_argument("--sites", default="final,resid,mlph")
    ap.add_argument("--kinds", default="box,code,seen")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--coords", type=int, default=4, help="random coordinates per site and prompt")
    ap.add_argument("--widest", type=int, default=2, help="and the coordinates with the widest boxes")
    ap.add_argument("--reach", type=int, default=4, help="positions from the fault's whose logits are bounded")
    ap.add_argument("--reach-true", type=int, default=0, help="positions over which the true worst case is also taken")
    ap.add_argument("--grid", type=int, default=65, help="fault values for the true worst case")
    ap.add_argument("--pieces", default="1,16", help="parts the range is split into for IBP, one forward each")
    ap.add_argument("--iv-clamp", default="on", choices=("on", "off", "both"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    lm = LM(a.model)
    lm.boxes = Boxes.load(a.boxes)
    tok = lm.tokenizer()
    L = lm.spec.layers
    depths = [int(k) for k in a.depths.split(",")]
    sites = a.sites.split(",")
    kinds = a.kinds.split(",")
    pieces = [int(p) for p in a.pieces.split(",")]
    clamps = {"on": [True], "off": [False], "both": [True, False]}[a.iv_clamp]
    gen = torch.Generator().manual_seed(a.seed)
    shrink = 1 - 2.0 ** -7
    plan = [("final", L)] if "final" in sites else []
    for k in depths:
        if "resid" in sites:
            plan.append(("resid", L - k - 1))
        if "mlph" in sites:
            plan.append(("mlph", L - k))
    seen: dict[tuple[str, int], Iv] = {}
    if "seen" in kinds:
        for text in PROMPTS[:a.prompts]:
            taps = {key: None for key in plan}
            lm.forward(torch.tensor([tok.encode(text).ids]), Cache(), taps=taps)
            for key, x in taps.items():
                x = x[0, 1:].float()
                lo, hi = x.amin(0), x.amax(0)
                seen[key] = (torch.minimum(seen[key][0], lo), torch.maximum(seen[key][1], hi)) if key in seen else (
                    lo, hi)
    recs = []
    t_start = time.time()
    with open(a.out / "faults.jsonl", "w") as fh:
        for pi, text in enumerate(PROMPTS[:a.prompts]):
            ids = torch.tensor([tok.encode(text).ids])
            t = ids.shape[1] - 1
            pre = Cache()
            lm.forward(ids[:, :t], pre)
            c = Cache(dict(pre.kv), pre.sink)
            suffix = [ids[0, t]]
            for _ in range(max(a.reach, a.reach_true) - 1):
                suffix.append(lm.forward(suffix[-1].view(1, 1), c)[0, -1].argmax())
            sid = torch.stack(suffix).view(1, -1)
            R, R_all = a.reach, sid.shape[1]
            keys = {("resid", l) for s, l in plan if s in ("resid", "mlph")} | {("resid", l - 1) for s, l in plan
                                                                                if s == "mlph"}
            keys |= {("mid", l) for s, l in plan if s == "mlph"} | {(s, l) for s, l in plan if s in ("mlph", "final")}
            keys |= {("resid", L - 1)}
            taps = {key: None for key in keys}
            zh_all = lm.forward(sid, Cache(dict(pre.kv), pre.sink), taps=taps)[0].float()
            zh = zh_all[:R]
            cls_all, _ = lm.boxes.classes(t, R_all, sid, pre.sink)
            cls = cls_all[0, :R]
            margin = zh.topk(2, -1).values
            for site, layer in plan:
                x_site = taps[(site, layer)][0, 0].float()
                rng = ranges(lm, site, layer, x_site, int(cls[0]), kinds, seen)
                if not rng:
                    continue
                hb = lm.boxes.hi[(site, layer)][int(cls[0])].float()
                cs = torch.randperm(x_site.numel(), generator=gen)[:a.coords].tolist()
                picks = dict.fromkeys(cs, "random")
                for c_ in hb.argsort(descending=True)[:a.widest].tolist():
                    picks.setdefault(c_, "widest")
                entry = {name: taps[key][0, :R] if key in taps else None
                         for name, key in (("final", ("final", L)), ("mid", ("mid", layer)), ("mlph", ("mlph", layer)),
                                           ("resid_in", ("resid", layer)))}
                x_in = taps[("resid", layer - 1)] if site == "mlph" else None
                for coord, pick in picks.items():
                    for kind, (lo_v, hi_v) in rng.items():
                        lo_c, hi_c = float(lo_v[coord]), float(hi_v[coord])
                        rec = {"prompt": pi, "t": t, "class": int(cls[0]), "site": site, "layer": layer,
                               "depth": L - layer if site != "resid" else L - layer - 1, "coord": coord,
                               "pick": pick, "kind": kind, "lo": lo_c, "hi": hi_c, "B": max(-lo_c, hi_c),
                               "honest": float(x_site[coord]), "margin": (margin[:, 0] - margin[:, 1]).tolist()}
                        if site == "final":
                            rec["depth"] = 0
                        mid_c = (lo_c + hi_c) / 2
                        vals = (mid_c + (torch.linspace(lo_c, hi_c, a.grid) - mid_c) * shrink).to(torch.bfloat16)
                        vals = vals.float()[(vals.float() >= lo_c) & (vals.float() <= hi_c)]
                        if not len(vals):
                            vals = x_site[coord].view(1)
                        G = len(vals)
                        if site == "final":
                            f = Fault.of(t, [("final", L, coord, float(v)) for v in vals])
                            zg_all = lm.forward(sid.expand(G, R_all), Cache(dict(pre.kv), pre.sink).expand(G),
                                                fault=f, x_in=taps[("resid", L - 1)].expand(G, -1, -1),
                                                first_layer=L, pos0=t)
                        elif site == "mlph":
                            f = Fault.of(t, [("mlph", layer, coord, float(v)) for v in vals])
                            zg_all = lm.forward(sid.expand(G, R_all), Cache(dict(pre.kv), pre.sink).expand(G),
                                                fault=f, x_in=x_in.expand(G, -1, -1), first_layer=layer, pos0=t)
                        else:
                            xg = taps[("resid", layer)].expand(G, -1, -1).clone()
                            xg[:, 0, coord] = vals
                            zg_all = lm.forward(sid.expand(G, R_all), Cache(dict(pre.kv), pre.sink).expand(G),
                                                x_in=xg, first_layer=layer + 1, pos0=t)
                        zg_all = zg_all.float()
                        zg = zg_all[:, :R]
                        dz = (zg - zh).double()
                        lp_h = torch.log_softmax(zh_all.double(), -1)
                        lp_g = torch.log_softmax(zg_all.double(), -1)
                        lpd_all = (lp_g - lp_h).abs().amax(-1)
                        kl_all = (lp_g.exp() * (lp_g - lp_h)).sum(-1).clamp_min(0)
                        del lp_g
                        lpd, kl = lpd_all[:, :R], kl_all[:, :R]
                        rec["true"] = {
                            "delta": dz.abs().amax(-1).amax(0).tolist(),
                            "eps": lpd.amax(0).tolist(),
                            "osc": (dz.amax(-1) - dz.amin(-1)).amax(0).tolist(),
                            "flip": (zg.argmax(-1) != zh.argmax(-1)).any(0).tolist(),
                            "rho_bits": float((lpd.pow(2).sum(-1) / 2).amax()) * LOG2E,
                            "pure_bits": float(lpd.sum(-1).amax()) * LOG2E,
                            "kl_bits": float(kl.sum(-1).amax()) * LOG2E,
                        }
                        if R_all > R:
                            rec["true_long"] = {
                                "eps": lpd_all.amax(0).tolist(),
                                "kl": kl_all.amax(0).tolist(),
                                "flip": (zg_all.argmax(-1) != zh_all.argmax(-1)).any(0).tolist(),
                                "rho_bits": float((lpd_all.pow(2).sum(-1) / 2).amax()) * LOG2E,
                                "pure_bits": float(lpd_all.sum(-1).amax()) * LOG2E,
                                "kl_bits": float(kl_all.sum(-1).amax()) * LOG2E,
                            }
                        rec["cert"] = {}
                        for clamp_on in clamps:
                            ctx = Ctx(lm, lm.boxes if clamp_on else None, cls, pre, t)
                            for P in pieces:
                                parts = torch.linspace(lo_c, hi_c, P + 1)
                                t0 = time.time()
                                lo, hi = iv_forward(ctx, site, layer, entry, coord, parts)
                                m = metrics(zh, lo, hi)
                                idx = torch.bucketize(vals, parts[1:-1].contiguous())
                                viol = torch.maximum(lo[idx] - zg, zg - hi[idx])
                                tol = 1e-4 * (1 + zg.abs())
                                eps_parts = m["eps"]
                                rec["cert"][f"{'clamp' if clamp_on else 'raw'}:{P}"] = {
                                    "delta": m["delta"].amax(0).tolist(),
                                    "eps": eps_parts.amax(0).tolist(),
                                    "osc": m["osc"].amax(0).tolist(),
                                    "greedy": m["greedy"].all(0).tolist(),
                                    "rho_bits": float((eps_parts.pow(2).sum(-1) / 2).amax()) * LOG2E,
                                    "pure_bits": float(eps_parts.sum(-1).amax()) * LOG2E,
                                    "kl_bits": float((m["osc"].pow(2).sum(-1) / 8).amax()) * LOG2E,
                                    "width": float((hi - lo).amax()),
                                    "violations": int((viol > tol).sum()),
                                    "worst_violation": float(viol.amax()),
                                    "seconds": time.time() - t0,
                                }
                        recs.append(rec)
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
            print(f"prompt {pi}: {len(recs)} faults, {time.time() - t_start:.0f} s", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summarize(recs, a, lm), indent=1))
    return 0


def summarize(recs: list[dict], a: argparse.Namespace, lm: LM) -> dict:
    """Per (site, depth, kind, pick): the true and certified figures at the fault's own position and over the reach,
    as medians and maxima over the faults."""
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault((r["site"], r["depth"], r["kind"], r["pick"]), []).append(r)

    def q(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")

    out = []
    for (site, depth, kind, pick), rs in sorted(groups.items()):
        g = {"site": site, "depth": depth, "kind": kind, "pick": pick, "n": len(rs),
             "range_width": {"median": q([r["hi"] - r["lo"] for r in rs], 0.5)}}
        for key in ("delta", "eps"):
            own = [r["true"][key][0] for r in rs]
            later = [max(r["true"][key][1:], default=0.0) for r in rs]
            g[f"true_{key}_t"] = {"median": q(own, 0.5), "p90": q(own, 0.9), "max": max(own)}
            g[f"true_{key}_later"] = {"median": q(later, 0.5), "max": max(later)}
        g["true_flip_t"] = sum(r["true"]["flip"][0] for r in rs) / len(rs)
        for key in ("rho_bits", "pure_bits", "kl_bits"):
            g[f"true_{key}"] = {"median": q([r["true"][key] for r in rs], 0.5), "max": max(r["true"][key] for r in rs)}
        if "true_long" in rs[0]:
            g["true_long"] = {key: {"median": q([r["true_long"][key] for r in rs], 0.5),
                                    "max": max(r["true_long"][key] for r in rs)}
                              for key in ("rho_bits", "pure_bits", "kl_bits")}
            n = len(rs[0]["true_long"]["eps"])
            g["true_long"]["eps_by_position"] = {"median": [q([r["true_long"]["eps"][i] for r in rs], 0.5)
                                                            for i in range(n)],
                                                 "max": [max(r["true_long"]["eps"][i] for r in rs) for i in range(n)]}
            g["true_long"]["flips_later"] = q([sum(r["true_long"]["flip"][1:]) for r in rs], 0.5)
        g["cert"] = {}
        for name in rs[0]["cert"]:
            cs = [r["cert"][name] for r in rs]
            g["cert"][name] = {
                "delta_t": {"median": q([c["delta"][0] for c in cs], 0.5), "max": max(c["delta"][0] for c in cs)},
                "eps_t": {"median": q([c["eps"][0] for c in cs], 0.5), "max": max(c["eps"][0] for c in cs)},
                "eps_later_max": max(max(c["eps"][1:], default=0.0) for c in cs),
                "greedy_t": sum(c["greedy"][0] for c in cs) / len(cs),
                "greedy_all": sum(all(c["greedy"]) for c in cs) / len(cs),
                "rho_bits": {"median": q([c["rho_bits"] for c in cs], 0.5), "max": max(c["rho_bits"] for c in cs)},
                "kl_bits": {"median": q([c["kl_bits"] for c in cs], 0.5), "max": max(c["kl_bits"] for c in cs)},
                "inflation_t": {"median": q([c["delta"][0] / max(r["true"]["delta"][0], 1e-12)
                                             for c, r in zip(cs, rs)], 0.5)},
                "violations": sum(c["violations"] for c in cs),
            }
        out.append(g)
    return {"model": a.model, "revision": MODELS[a.model]["revision"], "boxes": str(a.boxes), "layers": lm.spec.layers,
            "vocab": lm.spec.vocab, "prompts": a.prompts, "reach": a.reach, "reach_true": a.reach_true,
            "grid": a.grid, "pieces": a.pieces,
            "iv_clamp": a.iv_clamp, "faults": len(recs), "groups": out}


if __name__ == "__main__":
    sys.exit(main())
