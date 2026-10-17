"""The built-in steps. Linears (from tonight's `linears_layerwise.py` / `exact_grid.py`): `smooth` (SmoothQuant, its
foldable variant under `kv_shared`), `rotate` (one orthogonal Q per k on x and W), `int8` (activations one scale per
token, rounded or dithered; weights one scale per output row or per group of inputs, capped at `weight_top`, rounded or
by GPTQ). Attention (from `attention.py`): `rotate` (one random Hadamard R of the head dimension on Q, K, V and R^T on
the output), `center_keys` (K minus its per-channel mean), `int8` (one limb: Q, K per (head, token), P per query row or
per key block, `p_sink` putting key 0 alone, V per channel; K and V rounded or dithered) and `peel` (a low-rank noise
E the server adds and peels exactly, on Q or on P, with P served as P - offset + E).

A new intervention is one more function here (or in any module `run.py --plugin` imports) under `@linear_step` or
`@attention_step`; `spec.py` says what it receives."""
from __future__ import annotations

import math

import torch

from spec import attention_step, linear_step

GROUP = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv", "self_attn.o_proj": "o",
         "mlp.gate_proj": "gu", "mlp.up_proj": "gu", "mlp.down_proj": "down"}
#: the linear whose input stands for its group's (calibration reads its input)
FIRST = {"qkv": "self_attn.q_proj", "o": "self_attn.o_proj", "gu": "mlp.gate_proj", "down": "mlp.down_proj"}
PEEL_L = 32


# ================================================================ linears

def sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def paley28() -> torch.Tensor:
    """Paley's construction I over GF(27) = GF(3)[x] / (x^3 + 2x + 1): a Hadamard matrix of order 28 (entries +-1)."""
    els = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]

    def mul(u, v):
        p = [0] * 5
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                p[i + j] += ui * vj
        for d in (4, 3):  # x^3 = x + 2 (mod 3)
            c, p[d] = p[d], 0
            p[d - 3] += 2 * c
            p[d - 2] += c
        return tuple(x % 3 for x in p[:3])

    squares = {mul(e, e) for e in els if e != (0, 0, 0)}
    chi = lambda e: 0 if e == (0, 0, 0) else (1 if e in squares else -1)  # noqa: E731
    q = torch.tensor([[chi(tuple((a - b) % 3 for a, b in zip(u, v))) for v in els] for u in els], dtype=torch.float64)
    s = torch.zeros(28, 28, dtype=torch.float64)
    s[0, 1:], s[1:, 0], s[1:, 1:] = 1, -1, q
    return torch.eye(28, dtype=torch.float64) + s


class Rotation:
    """x -> x Q for Q = diag(signs) (H_a x R_b): kind `q` is exact_grid.rotation(k) (a = k / odd Sylvester, R a random
    odd x odd orthogonal, both drawn from the generator seeded k, R first), kind `h` the flat one (R = H_28 / sqrt(28)
    when 7 divides k; the same signs). Applied through the Kronecker factors in FP32."""

    def __init__(self, k: int, kind: str, device):
        self.kind, self.k = kind, k
        g = torch.Generator().manual_seed(k)
        odd = k
        while odd % 2 == 0:
            odd //= 2
        r = torch.linalg.qr(torch.randn(odd, odd, generator=g, dtype=torch.float64))[0] if odd > 1 else torch.ones(1, 1, dtype=torch.float64)
        if kind == "h" and odd > 1:
            if not (odd == 7 and k % 28 == 0):
                raise ValueError(f"the flat rotation needs k = 7 * 2^j, j >= 2 (k = {k})")
            r = paley28() / math.sqrt(28)
        elif kind not in ("q", "h"):
            raise KeyError(kind)
        self.signs = (torch.randint(0, 2, (k,), generator=g).double() * 2 - 1).float().to(device)
        self.r = r.float().to(device)
        self.h = sylvester(k // r.shape[0]).float().to(device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        a, b = self.h.shape[0], self.r.shape[0]
        t = (x * self.signs).view(m, a, b)
        if b > 1:
            t = t @ self.r
        t = (t.transpose(1, 2).reshape(m * b, a) @ self.h).view(m, b, a).transpose(1, 2)
        return t.reshape(m, k)

    def dense(self) -> torch.Tensor:
        return self.apply(torch.eye(self.k, device=self.signs.device))


def quant_rows(m: torch.Tensor, top: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = m.abs().amax(dim=1).clamp(min=1e-30) / top
    return torch.round(m / s[:, None]).clamp(-top, top), s


def act_quant(x: torch.Tensor, mode: str, bits: torch.Tensor | None, top: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer rows and per-row scales: rounded to [-top, top], or the dither floor(v) + b with v = x / s on
    [-top + 1, top - 1] (|X| <= top)."""
    if mode == "round":
        return quant_rows(x, top)
    t = top - 1
    s = x.abs().amax(dim=1).clamp(min=1e-30) / t
    return torch.floor((x / s[:, None]).clamp(-t, t)) + bits, s


def gptq_round(w: torch.Tensor, h: torch.Tensor, top: int, block: int = 128, damp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ (Frantar et al.) of w (n x k, FP32) against the input second moment h (k x k), each row's scale fixed at
    absmax / top: integers in [-top, top] and the scales. Damping 1%, retried at 10x and 100x."""
    w = w.clone()
    n, k = w.shape
    h = h.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    s = w.abs().amax(dim=1).clamp(min=1e-30) / top
    h = (h + h.T) / 2
    mean_diag = torch.mean(torch.diag(h))
    for d in (damp, 10 * damp, 100 * damp):
        try:
            hd = h + d * mean_diag * torch.eye(k, device=w.device)
            hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hd)), upper=True)
            break
        except torch.linalg.LinAlgError:
            if d == 100 * damp:
                raise
    del h, hd
    q = torch.zeros_like(w)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1, err1, hb = w[:, i1:i2].clone(), torch.zeros(n, i2 - i1, device=w.device), hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            qi = torch.round(col / s).clamp(-top, top)
            q[:, i1 + i] = qi
            e = (col - qi * s) / hb[i, i]
            w1[:, i:] -= e[:, None] * hb[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q, s


def smooth_factors(mm, alpha: float, kv_shared: bool) -> torch.Tensor:
    """m = amax_x^alpha / amax_W^(1 - alpha) per input channel of the linear's group (amax_x over the calibration
    windows' inputs to the group, amax_W over its linears' raw weights); `kv_shared` (o_proj) one m per (KV head,
    channel), shared by the group's query heads, so 1 / m folds into v_proj's rows."""
    g = GROUP[mm.name]
    ax = mm.run.calib["amax"][g]
    lay = mm.run.layer
    aw = torch.stack([lay.weight(n).float().abs().amax(dim=0) for n in GROUP if GROUP[n] == g]).amax(dim=0)
    if kv_shared and g == "o":
        rep, d = lay.cfg.rep, lay.cfg.head_dim
        ax, aw = (a.view(-1, rep, d).amax(1, keepdim=True).expand(-1, rep, d).reshape(-1) for a in (ax, aw))
    return (ax.clamp(min=1e-5) ** alpha / aw.clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)


def _check_smooth(p):
    if not 0 <= p["alpha"] <= 1:
        raise ValueError("smooth.alpha is in [0, 1]")


@linear_step("smooth", phase="real", calib=True, check=_check_smooth)
def smooth(mm, nxt, *, alpha: float, kv_shared: bool = False):
    """SmoothQuant before the rotation: x / m and W m per input channel, m from the calibration windows' channel maxima
    (1 / m folds into the RMSNorm weights before q/k/v and gate/up, and into up_proj's rows before down_proj).
    `kv_shared` shares o_proj's m across each KV group's query heads, so it folds into v_proj's rows: the foldable
    variant. Reads the raw weights, so it comes first."""
    m = mm.cached("m", lambda: smooth_factors(mm, alpha, kv_shared))
    return nxt(mm.replace(x=mm.x / m[None, :], w=lambda w: w * m[None, :]))


@linear_step("rotate", phase="real")
def rotate_linear(mm, nxt, *, kind: str = "q"):
    """x Q and W Q for one orthogonal Q per k: `q` signs (H_{k/odd} x R_odd), R a random odd x odd orthogonal, seeded
    by k (exact_grid's); `h` the flat Q = signs (H_{k/28} x H_28) / sqrt(k) (Paley's H_28)."""
    k = mm.x.shape[1]
    r = mm.shared(("linear", kind, k, str(mm.device)), lambda: Rotation(k, kind, mm.device))
    return nxt(mm.replace(x=r.apply(mm.x), w=r.apply))


def _check_int8_linear(p):
    if p["act"] not in ("round", "dither"):
        raise ValueError("int8.act is round or dither")
    if not 1 <= p["weight_top"] <= 127 or not 2 <= p["act_top"] <= 127:
        raise ValueError("int8: tops within 127")
    if p["gptq"] and p["group"]:
        raise ValueError("int8: gptq runs on one scale per row")


@linear_step("int8", phase="quantize", seeded=lambda p: p["act"] == "dither", calib=lambda p: p["gptq"],
             check=_check_int8_linear)
def int8_linear(mm, nxt, *, act: str = "round", act_top: int = 127, weight_top: int = 127, group: int = 0,
                gptq: bool = False):
    """The int8 grid: activations one scale per token (`round` to [-act_top, act_top], or `dither`: floor(x / s) + b,
    b a fresh coin bit, s = absmax / (act_top - 1)), weights one scale per output row (or per (row, `group` inputs),
    each group's s32 read and scaled on its own) on [-weight_top, weight_top], rounded or by GPTQ (blocks of 128, on
    the calibration windows' served inputs). The integer product is exact (s8 x s8 -> s32), scaled after."""
    hkey = (mm.name, tuple(s.key for s in mm.prefix[:-1]))
    if mm.run.hessian is not None:
        if gptq:
            mm.run.hessian(hkey, mm.x)
        return None

    def weights():
        w = mm.weight()
        if gptq:
            q, s = gptq_round(w, mm.run.calib["H"][hkey], weight_top)
            return q.to(torch.int8), s
        if group:
            n, k = w.shape
            blocks = w.view(n, k // group, group)
            s = blocks.abs().amax(dim=2).clamp(min=1e-30) / weight_top
            return torch.round(blocks / s[:, :, None]).clamp(-weight_top, weight_top).view(n, k).to(torch.int8), s
        q, s = quant_rows(w, weight_top)
        return q.to(torch.int8), s

    wq, sw = mm.cached("int8", weights)
    bits = mm.coins(mm.x.shape) if act == "dither" else None
    xq, sa = act_quant(mm.x, act, bits, act_top)
    if group:
        k = xq.shape[1]
        y = torch.zeros(xq.shape[0], wq.shape[0], device=xq.device)
        for j in range(k // group):
            a, b = j * group, (j + 1) * group
            y += nxt(mm.integer(xq[:, a:b].contiguous(), wq[:, a:b].contiguous(), keys=(a, b))).float() * sw[None, :, j]
        return y * sa[:, None]
    return nxt(mm.integer(xq, wq)).float() * sa[:, None] * sw[None, :]


# ================================================================ attention

def hadamard_rotation(d: int, seed: int, device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=g).double() * 2 - 1
    return (signs[:, None] * h / math.sqrt(d)).float().to(device)


def quant(x: torch.Tensor, top: int, dims) -> tuple[torch.Tensor, torch.Tensor]:
    """Integers |q| <= top and scales s (absmax over `dims` / top) with x ~ q s."""
    s = x.abs().amax(dim=dims, keepdim=True).clamp(min=1e-30) / top
    return torch.round(x / s).clamp(-top, top), s


def floor_quant(x: torch.Tensor, dims) -> tuple[torch.Tensor, torch.Tensor]:
    """floor(x / s) in [-128, 126] and s, the smallest scale (to 2^-20) that keeps x / s in [-128, 127): with a fresh bit
    added, the served X stays within int8."""
    s = torch.maximum(x.amax(dim=dims, keepdim=True) / 127, (-x).amax(dim=dims, keepdim=True) / 128)
    s = s.clamp(min=1e-30) * (1 + 2 ** -20)
    return torch.floor(x / s), s


def causal_peel(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """E V for E_ik = u_{i, k mod L} [k <= i], u (s x L) and integer-valued v (... x s x d), in float64 (exact): the prefix
    scan over blocks of L rows, s L d MACs."""
    s, L = u.shape
    pad = -s % L
    uu = torch.nn.functional.pad(u.double(), (0, 0, 0, pad)).view(-1, L, L)
    vv = torch.nn.functional.pad(v.double(), (0, 0, 0, pad))
    vv = vv.view(*v.shape[:-2], -1, L, v.shape[-1])
    tri = torch.ones(L, L, dtype=torch.float64, device=u.device).tril()
    out = uu @ (vv.cumsum(-3) - vv) + (uu * tri) @ vv
    return out.reshape(*v.shape[:-2], -1, v.shape[-1])[..., :s, :]


def fold_peel(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """E K^T for E_ic = u_{i, c mod L}, u (s x L) and integer-valued k (... x t x d): u (W^T K^T), s t L MACs; exact in
    FP32 while L max|u| (d / L) max|k| <= 2^24."""
    L = u.shape[1]
    folded = k.reshape(*k.shape[:-1], k.shape[-1] // L, L).sum(-2)
    return u @ folded.transpose(-1, -2)


def p_edges(s: int, block: int, sink: bool) -> list[int]:
    """P's key blocks, as edges: one block, blocks of `block` keys, and key 0 alone under `sink`."""
    edges = set(range(0, s, block)) if block else {0}
    if sink and s > 1:
        edges.add(1)
    return sorted(edges) + [s]


@attention_step("rotate", phase="real", seeded=True)
def rotate_attention(mm, nxt):
    """Q R, K R and V R for one random Hadamard R of the head dimension (signs times Sylvester's H, seeded by the arm's
    seed and the layer), and the output times R^T: S and O unchanged before any edit."""
    d = mm.extra["d"]
    R = mm.shared(("attention", d, mm.seed, mm.layer, str(mm.device)),
                  lambda: hadamard_rotation(d, 7919 * (mm.seed + 1) + mm.layer, mm.device))
    if mm.kind == "qk":
        return nxt(mm.replace(x=mm.x @ R, w=lambda k: k @ R))
    return nxt(mm.replace(w=lambda v: v @ R)) @ R.T


@attention_step("center_keys", phase="real", on=("qk",))
def center_keys(mm, nxt):
    """K - mu, mu its per-channel mean over the window: q mu is the same for every key of q's row, so softmax(S) is
    unchanged, and a part of K constant over tokens (Qwen2.5's k bias) no longer sets K's scale."""
    return nxt(mm.replace(w=lambda k: k - k.mean(dim=0, keepdim=True)))


def _check_int8_attention(p):
    for x in ("k", "v"):
        if p[x] not in ("round", "dither"):
            raise ValueError(f"int8.{x} is round or dither")
    if p["v_scale"] not in ("channel", "head"):
        raise ValueError("int8.v_scale is channel or head")
    if p["p_block"] and (p["p_block"] % PEEL_L or p["p_block"] > 2 ** 16):
        raise ValueError(f"int8.p_block is a multiple of {PEEL_L} keys, at most 2^16")
    if not all(1 <= p[t] <= 127 for t in ("q_top", "k_top", "p_top", "v_top")):
        raise ValueError("int8: tops within 127")


@attention_step("int8", phase="quantize", seeded=lambda p: "dither" in (p["k"], p["v"]), check=_check_int8_attention)
def int8_attention(mm, nxt, *, q_top: int = 127, k: str = "round", k_top: int = 127, p_top: int = 127,
                   p_block: int = 0, p_sink: bool = False, v: str = "round", v_top: int = 127, v_scale: str = "channel"):
    """Both attention matmuls on one int8 limb. Q, K: one scale per (head, token), Q rounded to [-q_top, q_top], K
    rounded or dithered (`dither`: floor(x / s) + b, s the smallest scale with x / s in [-128, 127), b a fresh bit for
    every query head). P: after the softmax, one scale per query row, or per (row, block of `p_block` keys), `p_sink`
    putting key 0 in a block of its own, rounded to [0, p_top], each block's product read and scaled on its own. V: one
    scale per (KV head, channel) (`v_scale = "head"`: per KV head), rounded or dithered. Products exact in FP32 chunks."""
    rep = mm.extra["rep"]
    if mm.kind == "qk":
        qi, sq = quant(mm.x, q_top, -1)

        def keys():
            kg = mm.weight()
            kq, sk = floor_quant(kg, -1) if k == "dither" else quant(kg, k_top, -1)
            kq = kq.expand(rep, *kq.shape)
            if k == "dither":
                kq = kq + mm.coins(kq.shape)
            return kq, sk
        ki, sk = mm.cached("k", keys)
        return nxt(mm.integer(qi, ki)).float() * (sq * sk.T[None])
    p = mm.x
    s = p.shape[-1]

    def values():
        vg = mm.weight()
        dims = 0 if v_scale == "channel" else (0, 1)
        vq, sv = floor_quant(vg, dims) if v == "dither" else quant(vg, v_top, dims)
        vq = vq.expand(rep, *vq.shape)
        if v == "dither":
            vq = vq + mm.coins(vq.shape)
        return vq, sv
    vi, sv = mm.cached("v", values)
    edges = p_edges(s, p_block, p_sink)
    num = None
    for a, b in zip(edges, edges[1:]):
        sb = p[..., a:b].amax(-1, keepdim=True).clamp(min=1e-30) / p_top
        pi = torch.round(p[..., a:b] / sb).clamp(0, p_top)
        part = sb.double() * nxt(mm.integer(pi, vi[..., a:b, :], keys=(a, b), v_full=vi))
        num = part if num is None else num + part
    return (num * sv.double()).float()


def draw_nonzero(width: int, shape, gen: torch.Generator, device) -> torch.Tensor:
    """Uniform on {+-1, ..., +-width}."""
    table = torch.tensor([v for v in range(-width, width + 1) if v], dtype=torch.float32, device=device)
    return table[torch.randint(0, len(table), shape, generator=gen, device=device)]


def _check_peel(p):
    if p["operand"] not in ("q", "p"):
        raise ValueError("peel.operand is q or p")
    if not 1 <= p["width"] <= 127 or not 0 <= p["offset"] <= 128:
        raise ValueError("peel: width in 1..127, offset in 0..128")
    if p["offset"] and p["operand"] != "p":
        raise ValueError("peel.offset serves P - offset")


@attention_step("peel", phase="integer", check=_check_peel)
def peel(mm, nxt, *, operand: str, width: int, offset: int = 0):
    """Low-rank noise the server adds and peels exactly: E_ik = U_{i, k mod 32}, U (tokens x 32) uniform on
    {+-1..+-width}, one draw per layer shared by its heads; on P only where key <= query, P served as P - offset + E
    (within int8), the offset added back as offset sum_{k <= i} V_k. The served products, peeled ((Q + E) K^T - U (W^T
    K^T), the causal prefix scan for P), are the direct ones exactly: checked on the check window for every layer, KV
    group and P block. The quantizer's top must leave room (Q: q_top + width <= 127)."""
    acc = nxt(mm)
    if not mm.checking or (operand == "q") != (mm.kind == "qk"):
        return acc
    r0, r1 = mm.extra["rows"]
    s = mm.extra["s"]
    U = mm.run.peel_u.get(operand)
    if U is None:
        U = mm.run.peel_u[operand] = draw_nonzero(width, (s, PEEL_L), mm.run.ugen(mm.device), mm.device)
    if mm.kind == "qk":
        d = mm.x.shape[-1]
        ki = mm.weight()
        served = mm.x + U[r0:r1][:, torch.arange(d, device=mm.device) % PEEL_L]
        got = mm.int_product(served, ki) - fold_peel(U[r0:r1], ki).double()
    else:
        a, b = mm.extra["keys"]
        causal = mm.extra["causal"][:, a:b].float()
        keys = torch.arange(a, b, device=mm.device)
        served = mm.x - offset * causal + U[r0:r1][:, keys % PEEL_L] * causal
        vi = mm.extra["v_full"]
        mask = torch.zeros(s, 1, device=mm.device)
        mask[a:b] = 1
        got = mm.int_product(served, mm.weight()) - causal_peel(U, vi * mask)[:, r0:r1]
        if offset:
            got = got + offset * (vi * mask).double().cumsum(-2)[:, r0:r1]
    mm.gate("peel", bool(torch.equal(got, acc)))
    return acc
