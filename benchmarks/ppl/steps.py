"""The built-in steps. Linears (from tonight's `linears_layerwise.py` / `exact_grid.py`): `smooth` (SmoothQuant, its
foldable variant under `kv_shared`), `rotate` (one orthogonal Q per k on x and W), `int8` (activations one scale per
token, rounded or dithered; weights one scale per output row or per group of inputs, capped at `weight_top`, rounded or
by GPTQ). Attention (from `attention.py`): `rotate` (one random Hadamard R of the head dimension on Q, K, V and R^T on
the output), `center_keys` (K minus its per-channel mean), `int8` (one limb: Q, K per (head, token), P per query row or
per key block, `p_sink` putting key 0 alone, V per channel; K and V rounded or dithered) and `peel` (a low-rank noise
E the server adds and peels exactly, on Q or on P, with P served as P - offset + E). Low-bit grids (`GRIDS`): `grid`
on linears and on attention puts both operands of every matmul on FP4's or FP8's integers, each rounded or dithered.

A new intervention is one more function here (or in any module `run.py --plugin` imports) under `@linear_step` or
`@attention_step`; `spec.py` says what it receives."""
from __future__ import annotations

import math
import zlib

import torch

import gates as G
from matmul import Softmaxed
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


def dither_shift(dither_bits: int) -> int:
    """m = 2^(b - 1) - 1 for a b-bit dither: the served floor(v) - m + R, R uniform on {0, ..., 2^b - 1} (mean m + 1/2),
    has E = floor(v) + 1/2 given v at every b, the 1-bit dither's (m = 0), and only R's variance (4^b - 1) / 12 grows.
    It reaches m below floor(v) and m + 1 above it."""
    return 2 ** (dither_bits - 1) - 1


def act_quant(x: torch.Tensor, mode: str, bits: torch.Tensor | None, top: int,
              coined: torch.Tensor | None = None, dither_bits: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer rows and per-row scales: rounded to [-top, top], or the dither floor(v) - m + R with v = x / s on
    [-t, t], t = top - 2^(b - 1), R the coins `bits` on {0, ..., 2^b - 1} and m = `dither_shift(b)` (b = 1: floor(v)
    plus a bit on [-top + 1, top - 1]), so -top < X <= top. `coined` (a mask over k): the dither only there, round(v) on
    the same s elsewhere."""
    if mode == "round":
        return quant_rows(x, top)
    m = dither_shift(dither_bits)
    t = top - 1 - m
    s = x.abs().amax(dim=1).clamp(min=1e-30) / t
    v = (x / s[:, None]).clamp(-t, t)
    q = torch.floor(v) - m + bits
    return (q if coined is None else torch.where(coined, q, torch.round(v))), s


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
    for x in ("act", "weight"):
        if p[x] not in ("round", "dither"):
            raise ValueError(f"int8.{x} is round or dither")
    if not 1 <= p["weight_top"] <= 127 or not 2 <= p["act_top"] <= 127:
        raise ValueError("int8: tops within 127")
    if p["gptq"] and p["group"]:
        raise ValueError("int8: gptq runs on one scale per row")
    if p["weight"] == "dither" and (p["gptq"] or p["group"] or p["weight_top"] < 2):
        raise ValueError("int8: a dithered weight has one scale per row, no gptq and weight_top >= 2")
    per = p["density"] * WORD_K
    if not 0 < p["density"] <= 1 or abs(per - round(per)) > 1e-9:
        raise ValueError(f"int8.density is in (0, 1], a multiple of 1/{WORD_K}")
    if p["density"] < 1 and p["act"] != "dither":
        raise ValueError("int8.density thins the activations' dither: act = dither")
    if p["density"] < 1 and p["weight"] == "dither":
        raise ValueError("int8.density thins the activations' dither only: a dithered weight takes density = 1 "
                         "(the grid step thins both operands)")
    _check_dither_bits(p["dither_bits"], "int8.dither_bits",
                       [p[f"{x}_top"] for x in ("act", "weight") if p[x] == "dither"], 1)


def _check_dither_bits(b: int, name: str, tops: list, floor_room: int) -> None:
    if not 1 <= b <= 6:
        raise ValueError(f"{name} is in 1..6")
    if b > 1 and not tops:
        raise ValueError(f"{name} widens a dither: dither an operand")
    if any(t - floor_room - dither_shift(b) < 1 for t in tops):
        raise ValueError(f"{name} = {b} leaves no room under a dithered operand's top (top >= {2 ** (b - 1) + floor_room})")


@linear_step("int8", phase="quantize", seeded=lambda p: "dither" in (p["act"], p["weight"]), calib=lambda p: p["gptq"],
             check=_check_int8_linear)
def int8_linear(mm, nxt, *, act: str = "round", act_top: int = 127, weight: str = "round", weight_top: int = 127,
                group: int = 0, gptq: bool = False, density: float = 1.0, dither_bits: int = 1):
    """The int8 grid: activations one scale per token (`round` to [-act_top, act_top], or `dither`: floor(x / s) + b,
    b a fresh coin bit, s = absmax / (act_top - 1)), weights one scale per output row (or per (row, `group` inputs),
    each group's s32 read and scaled on its own) on [-weight_top, weight_top], rounded, by GPTQ (blocks of 128, on
    the calibration windows' served inputs) or dithered like the activations (fresh bits every call, drawn after the
    activations'). The integer product is exact (s8 x s8 -> s32), scaled after. `density` < 1: the activations'
    dither only on a coin-chosen density * 64 of every k64 word's positions (`coin_positions`), the rest rounded on the
    same scale; a weight-side coin (the `peel` step's U) sits on every position. `dither_bits` = b widens every
    dithered operand's coin to R uniform on {0, ..., 2^b - 1}, served as floor(x / s) - (2^(b - 1) - 1) + R with
    s = absmax / (top - 2^(b - 1)) (`act_quant`): the same mean given x as the bit's, within (-top, top]."""
    hkey = (mm.name, tuple(s.key for s in mm.prefix[:-1]))
    if mm.run.hessian is not None:
        if gptq:
            mm.run.hessian(hkey, mm.x)
        return None

    m = dither_shift(dither_bits)

    def weights():
        w = mm.weight()
        if weight == "dither":
            t = weight_top - 1 - m
            s = w.abs().amax(dim=1).clamp(min=1e-30) / t
            return (torch.floor((w / s[:, None]).clamp(-t, t)) - m).to(torch.int8), s
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
    levels = 2 ** dither_bits
    bits = mm.coins(mm.x.shape, levels) if act == "dither" else None
    k = mm.x.shape[1]
    coined = mm.cached("pos", lambda: coin_positions(mm, k, density)) if density < 1 else None
    xq, sa = act_quant(mm.x, act, bits, act_top, coined, dither_bits)
    if weight == "dither":
        wq = (wq.float() + mm.coins(wq.shape, levels)).to(torch.int8)
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


def floor_quant(x: torch.Tensor, dims, top: int = 127, dither_bits: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """floor(x / s) - m in [-top - 1, top - 2m - 1] and s, the smallest scale (to 2^-20) that keeps x / s in
    [-top - 1 + m, top - m), m = `dither_shift(dither_bits)`: with fresh coins R on {0, ..., 2^b - 1} added, the served
    X stays within [-top - 1, top] (int8 at top = 127). b = 1: floor(x / s) in [-top - 1, top - 1], plus a bit."""
    m = dither_shift(dither_bits)
    s = torch.maximum(x.amax(dim=dims, keepdim=True) / (top - m), (-x).amax(dim=dims, keepdim=True) / (top + 1 - m))
    s = s.clamp(min=1e-30) * (1 + 2 ** -20)
    return torch.floor(x / s) - m, s


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


K_SCALES = ("token", "head", "calib")
SOFTMAXES = ("float", "lut")
LUT_TOP = 127
#: the sink's weight against the max over keys 1..pos is at most LUT_TOP 2^LUT_SINK_OCTAVES (within int32)
LUT_SINK_OCTAVES = 23


def lut_table(res: int, size: int, device=None) -> torch.Tensor:
    """LUT[t] = rint(127 2^(-t / res)) for t in 0..size-1 (fattn.cu's at res 16, size 256), as int64."""
    return torch.round(LUT_TOP * torch.exp2(-torch.arange(size, dtype=torch.float64) / res)).long().to(device)


def lut_up(res: int, device=None) -> torch.Tensor:
    """H[r] = rint(127 2^(r / res)) for r in 0..res-1: the sink's weight above the max is H[u mod res] 2^(u div res)."""
    return torch.round(LUT_TOP * torch.exp2(torch.arange(res, dtype=torch.float64) / res)).long().to(device)


def key_quant(kg: torch.Tensor, mode: str, top: int, bits: int, scale: str, sink: bool,
              cal: tuple | None) -> tuple[torch.Tensor, torch.Tensor]:
    """K's integers and its scales, one per key (s x 1): per token (`floor_quant` or `quant`, as before), or one scale
    for the KV head, from the window's keys (`head`) or the calibration windows' (`calib`, `cal` = (max, max of -K),
    the window's keys clamped into it). `sink`: key 0 on its own per-token scale, the head's scale from keys 1..s-1."""
    if scale == "token":
        return floor_quant(kg, -1, top, bits) if mode == "dither" else quant(kg, top, -1)
    rest = kg[1:] if sink else kg
    hi, lo = (rest.amax(), (-rest).amax()) if scale == "head" else cal
    m = dither_shift(bits)

    def on(x, hi, lo):
        if mode == "dither":
            s = torch.maximum(hi / (top - m), lo / (top + 1 - m)).clamp(min=1e-30) * (1 + 2 ** -20)
            return torch.floor(x / s).clamp(-top - 1 + m, top - m - 1) - m, s
        s = torch.maximum(hi, lo).clamp(min=1e-30) / top
        return torch.round(x / s).clamp(-top, top), s

    q, s = on(kg, hi, lo)
    sk = s.reshape(1, 1).expand(kg.shape[0], 1).clone()
    if sink:
        q0, s0 = on(kg[:1], kg[0].amax(), (-kg[0]).amax())
        q, sk = torch.cat([q0, q[1:]]), torch.cat([s0.reshape(1, 1), sk[1:]])
    return q, sk


def calibrated_keys(mm, sink: bool) -> tuple:
    """(max K, max -K) for this KV head over the calibration windows' keys (the bf16 stream's, after RoPE), each window
    through this chain's maps of K (the rotation, centred keys), key 0 left out under `sink`."""
    def go():
        cal = mm.run.calib["keys"][:, mm.extra["group"]]
        hi = lo = None
        for w in range(cal.shape[0]):
            k = cal[w].float()
            for f in mm.maps:
                k = f(k)
            k = k[1:] if sink else k
            hi = k.amax() if hi is None else torch.maximum(hi, k.amax())
            lo = (-k).amax() if lo is None else torch.maximum(lo, (-k).amax())
        return hi, lo
    key = ("kcal", mm.layer, mm.extra["group"], mm.seed, tuple(s.key for s in mm.prefix[:-1]), sink, str(mm.device))
    return mm.shared(key, go)


def lut_softmax(mm, S: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor, *, res: int, size: int, sink: bool):
    """fattn.cu's integer softmax (fattn_ref.py) on the exact S = Q K^T (rep x rows x s): per query row the fixed-point
    A = rint(s_q s_k log2(e) res / sqrt(d) 2^32) (key 0's own A_0 from its own s_k when K's key 0 has one), the logits
    lam = S A (key 0: S A_0), m their max over keys 0..pos (1..pos under `sink`), t = floor((m - lam) / 2^32),
    e = LUT[min(t, size - 1)] (`lut_table`), L = sum of e over keys 0..pos, s_q = float32(1) / float32(L) and z_q =
    float32(e(0)) / float32(L). Under `sink`, key 0's t may be negative, u = -t (at most res LUT_SINK_OCTAVES): its
    weight is H[u mod res] 2^(u div res) (`lut_up`). Returns P = e s_q (z_q at key 0) for P V, which reads e, s_q and
    z_q from `extra["lut"]`. Gated on every call: A within uint32 and m - S within 32 bits."""
    causal = mm.extra["causal"]
    c = math.log2(math.e) * res * mm.extra["d"] ** -0.5 * 2.0 ** 32
    A = torch.round(sq.double() * sk[-1, 0].double() * c).long()
    A0 = torch.round(sq.double() * sk[0, 0].double() * c).long()
    Si = S.long()
    lam = Si * A
    lam[..., 0] = Si[..., 0] * A0[..., 0]
    floor = torch.iinfo(torch.int64).min // 4
    lam = lam.masked_fill(~causal, floor)
    if sink:
        m = lam[..., 1:].amax(-1, keepdim=True) if lam.shape[-1] > 1 else lam[..., :1]
        m = torch.where(m == floor, lam[..., :1], m)
    else:
        m = lam.amax(-1, keepdim=True)
    span = (m - lam).masked_fill(~causal, 0)
    a_max = int(torch.maximum(A, A0).max())
    d = span[..., 1:] // A.clamp(min=1)
    d_max = int(d.max()) if d.numel() else 0
    del d
    mm.gate("lut", a_max < 2 ** 32 and d_max < 2 ** 32, {"A_max": a_max, "m_minus_S_max": d_max})
    t = span >> 32
    del span, lam
    lut = lut_table(res, size, S.device)
    e = lut[t.clamp(0, size - 1)]
    if sink:
        u = (-t[..., 0]).clamp(0, res * LUT_SINK_OCTAVES)
        e[..., 0] = torch.where(t[..., 0] >= 0, e[..., 0], lut_up(res, S.device)[u % res] << (u // res))
    e = e * causal
    L = e.sum(-1, keepdim=True)
    Lf = L.float().double()
    s_q = (1.0 / Lf).float()
    z_q = (e[..., :1].float().double() / Lf).float()
    p = e.float() * s_q
    p[..., :1] = z_q
    return Softmaxed(p, {"lut": (e, s_q, z_q)})


def _check_int8_attention(p):
    for x in ("q", "k", "p", "v"):
        if p[x] not in ("round", "dither"):
            raise ValueError(f"int8.{x} is round or dither")
    if p["v_scale"] not in ("channel", "head"):
        raise ValueError("int8.v_scale is channel or head")
    if p["k_scale"] not in K_SCALES:
        raise ValueError(f"int8.k_scale is one of {K_SCALES}")
    if p["k_sink"] and p["k_scale"] == "token":
        raise ValueError("int8.k_sink gives key 0 its own scale beside the KV head's: k_scale = head or calib")
    if p["softmax"] not in SOFTMAXES:
        raise ValueError(f"int8.softmax is one of {SOFTMAXES}")
    if p["softmax"] == "lut":
        if p["k_scale"] == "token":
            raise ValueError("int8.softmax = lut puts every key's S in one unit: k_scale = head or calib")
        if p["p"] != "round" or p["p_block"] or p["p_top"] != LUT_TOP:
            raise ValueError("int8.softmax = lut serves P = e on [0, 127]: p = round, p_block = 0, p_top = 127")
        if not 1 <= p["lut_res"] <= 64 or not 2 <= p["lut_size"] <= 4096:
            raise ValueError("int8.lut_res in 1..64, lut_size in 2..4096")
        if int(lut_table(p["lut_res"], p["lut_size"])[-1]) != 0:
            raise ValueError("int8: the LUT's last entry (where t saturates) must be 0: raise lut_size")
    elif p["lut_sink"]:
        raise ValueError("int8.lut_sink is the integer softmax's: softmax = lut")
    if p["p_block"] and (p["p_block"] % PEEL_L or p["p_block"] > 2 ** 16):
        raise ValueError(f"int8.p_block is a multiple of {PEEL_L} keys, at most 2^16")
    if not all(1 <= p[t] <= 127 for t in ("q_top", "k_top", "p_top", "v_top")):
        raise ValueError("int8: tops within 127")
    _check_dither_bits(p["qk_bits"], "int8.qk_bits", [p[f"{x}_top"] for x in ("q", "k") if p[x] == "dither"], 0)


@attention_step("int8", phase="quantize", seeded=lambda p: "dither" in (p["q"], p["k"], p["p"], p["v"]),
                calib=lambda p: p["k_scale"] == "calib", check=_check_int8_attention)
def int8_attention(mm, nxt, *, q: str = "round", q_top: int = 127, k: str = "round", k_top: int = 127,
                   k_scale: str = "token", k_sink: bool = False, p: str = "round", p_top: int = 127, p_block: int = 0,
                   p_sink: bool = False, v: str = "round", v_top: int = 127, v_scale: str = "channel", qk_bits: int = 1,
                   softmax: str = "float", lut_res: int = 16, lut_size: int = 256, lut_sink: bool = False):
    """Both attention matmuls on one int8 limb. Q, K: one scale per (head, token), Q rounded to [-q_top, q_top] or
    dithered, K rounded or dithered (`dither`: floor(x / s) + b, s the smallest scale with x / s in [-top - 1, top), b
    a fresh bit (for K, one for every query head), so the served operand is within [-top - 1, top]). `k_scale`: K's
    scale per token, or one per (layer, KV head) from the window's keys (`head`) or the calibration windows' (`calib`,
    the window's keys clamped into it); `k_sink` gives key 0 its own per-token scale beside it (`key_quant`).
    `softmax = "lut"`: fattn.cu's integer softmax (`lut_softmax`, LUT[t] = rint(127 2^(-t / lut_res)), `lut_size`
    entries; `lut_sink`: key 0 out of the max) in place of the float one, P = e on keys 1..pos and key 0 as z_q V(0),
    the block of keys 1..pos scaled by s_q (p_sink's two blocks, whatever `p_sink` says). P: after the
    softmax, one scale per query row, or per (row, block of `p_block` keys), `p_sink` putting key 0 in a block of its
    own, rounded to [0, p_top], or (`dither`) rounded to [0, p_top - 1] plus a fresh bit where key <= query (so a
    dithered P serves no future key), the bits' mean, half the causal sum of V, taken off the product in the epilogue:
    P is non-negative and mostly under one step, so a bit's +1/2 would not average out over the keys. Each block's
    product read and scaled on its own. V: one scale per (KV head, channel) (`v_scale = "head"`: per KV head), rounded
    or dithered. Products exact in FP32 chunks. `qk_bits` = b widens Q's and K's coins to R uniform on
    {0, ..., 2^b - 1}, served as floor(x / s) - (2^(b - 1) - 1) + R (`floor_quant`), still within [-top - 1, top]; V
    and P keep their bit."""
    rep = mm.extra["rep"]
    if mm.kind == "qk":
        levels = 2 ** qk_bits
        if q == "dither":
            qi, sq = floor_quant(mm.x, -1, q_top, qk_bits)
            qi = qi + mm.coins(qi.shape, levels)
        else:
            qi, sq = quant(mm.x, q_top, -1)

        def keys():
            kg = mm.weight()
            cal = calibrated_keys(mm, k_sink) if k_scale == "calib" else None
            kq, sk = key_quant(kg, k, k_top, qk_bits, k_scale, k_sink, cal)
            kq = kq.expand(rep, *kq.shape)
            if k == "dither":
                kq = kq + mm.coins(kq.shape, levels)
            return kq, sk
        ki, sk = mm.cached("k", keys)
        prod = nxt(mm.integer(qi, ki))
        if softmax == "lut":
            return lut_softmax(mm, prod, sq, sk, res=lut_res, size=lut_size, sink=lut_sink)
        return prod.float() * (sq * sk.T[None])
    pr = mm.x
    s = pr.shape[-1]

    def values():
        vg = mm.weight()
        dims = 0 if v_scale == "channel" else (0, 1)
        vq, sv = floor_quant(vg, dims, v_top) if v == "dither" else quant(vg, v_top, dims)
        vq = vq.expand(rep, *vq.shape)
        if v == "dither":
            vq = vq + mm.coins(vq.shape)
        return vq, sv
    vi, sv = mm.cached("v", values)
    lut = mm.extra.get("lut")
    if softmax == "lut" and lut is None:
        raise ValueError(f"{mm.where}: softmax = lut on P V without the integer softmax's e (int8 on Q K^T)")
    edges = p_edges(s, 0, True) if lut is not None else p_edges(s, p_block, p_sink)
    num = None
    for a, b in zip(edges, edges[1:]):
        blk = pr[..., a:b]
        top = p_top - 1 if p == "dither" else p_top
        if lut is not None:
            e, s_q, z_q = lut
            pi, sb = (torch.ones_like(blk), z_q) if a == 0 else (e[..., a:b].float(), s_q)
        else:
            sb = blk.amax(-1, keepdim=True).clamp(min=1e-30) / top
            pi = torch.round(blk / sb).clamp(0, top)
        if p == "dither":
            causal = mm.extra["causal"][:, a:b]
            pi = pi + mm.coins(pi.shape) * causal
        prod = nxt(mm.integer(pi, vi[..., a:b, :], keys=(a, b), v_full=vi))
        if p == "dither":
            prod = prod - 0.5 * (causal.double() @ vi[..., a:b, :].double())
        part = sb.double() * prod
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


# ================================================================ low-bit grids

#: each grid's levels as integers: E2M1's contiguous region (+-2 in steps of 0.5) and its every value, both times 2,
#: and the integers E4M3 holds exactly. Every product of two of them is exact (k max^2 < 2^24).
GRIDS = {"fp4c": tuple(range(-4, 5)), "fp4": (-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12),
         "int5": tuple(range(-16, 17))}
GRID_MODES = ("round", "dither")
#: linears also take `sr`, unbiased stochastic rounding, whose coin is a uniform draw rather than one bit
LINEAR_GRID_MODES = GRID_MODES + ("sr",)
#: block scales a linear's `grid` may choose from, per block of k: the UE4M3 odd integers (an odd scale keeps every
#: coin's bilinear coefficient odd) or every integer UE4M3 holds up to 15
SCALES = {"odd": tuple(range(1, 16, 2)), "int": tuple(range(1, 16))}
#: a committed word's length in k (one FP4 m16n8k64 firing); `density` picks its coined positions per word
WORD_K = 64


def bracket(v: torch.Tensor, grid: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The grid's levels lo <= v < hi around each v in [min, max) (hi = max at v = max)."""
    levels = GRIDS[grid]
    if levels == tuple(range(levels[0], levels[-1] + 1)):
        lo = torch.floor(v).clamp(levels[0], levels[-1] - 1)
        return lo, lo + 1
    table = torch.tensor(levels, dtype=v.dtype, device=v.device)
    i = (torch.searchsorted(table, v.contiguous(), right=True) - 1).clamp(0, len(levels) - 2)
    return table[i], table[i + 1]


def to_grid(v: torch.Tensor, grid: str, mode: str, bits: torch.Tensor | None,
            coined: torch.Tensor | None = None) -> torch.Tensor:
    """v (within the grid's range) onto the grid's levels: `round` the nearest, `dither` the level below or above by the
    fair bit, `sr` the level above with probability (v - lo) / (hi - lo) (`bits` uniform in [0, 1)). `coined` (a mask
    broadcast over v) keeps the coin only where it is set and rounds the rest."""
    lo, hi = bracket(v, grid)
    nearest = torch.where(v - lo <= hi - v, lo, hi)
    if mode == "round":
        return nearest
    if mode == "sr":
        bits = (bits < (v - lo) / (hi - lo)).float()
    out = lo + bits * (hi - lo)
    return out if coined is None else torch.where(coined, out, nearest)


def on_grid(x: torch.Tensor, grid: str, mode: str, bits: torch.Tensor | None, dims,
            coined: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Integers on the grid and scales s (one per slice over `dims`, kept) with x ~ q s: `round` the nearest level,
    s = absmax / top; `dither` the level below v = x / s or the one above it by the fair bit b, s = absmax / top
    (1 + 2^-20) so that |v| < top (on a uniform grid, floor(v) + b); `sr` the level above with probability
    (v - lo) / (hi - lo), `bits` holding uniform draws in [0, 1), the same s (unbiased: E q = v). `coined`: the coin
    only at those positions, the rest rounded on the same s."""
    top = GRIDS[grid][-1]
    a = x.abs().amax(dim=dims, keepdim=True).clamp(min=1e-30)
    s = a / top if mode == "round" else a * (1 + 2 ** -20) / top
    return to_grid(x / s, grid, mode, bits, coined), s


def on_blocks(x: torch.Tensor, grid: str, mode: str, bits: torch.Tensor | None, block: int, scales: str,
              coined: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4-style block scaling of the rows of x (m x k): one float scale s per row, s = absmax / (top * max scale)
    (1 + 2^-20), and per block of `block` k-positions the smallest integer scale sigma in SCALES[scales] with the
    block's |x / s| <= top * sigma; each element x / (s sigma) on the grid (`to_grid`). Returns the integers q sigma
    (the FP4 code times its block scale: the integer the block-scaled MMA multiplies) and s."""
    m, k = x.shape
    top, table = GRIDS[grid][-1], torch.tensor(SCALES[scales], dtype=x.dtype, device=x.device)
    s = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) * (1 + 2 ** -20) / (top * table[-1])
    v = (x / s).view(m, k // block, block)
    need = (v.abs().amax(-1) / top).contiguous()
    sigma = table[torch.searchsorted(table, need).clamp(max=len(table) - 1)][..., None]
    b = None if bits is None else bits.view(m, k // block, block)
    c = None if coined is None else coined.view(k // block, block)
    return (to_grid(v / sigma, grid, mode, b, c) * sigma).view(m, k), s


def coin_positions(mm, k: int, density: float) -> torch.Tensor:
    """The coined k-positions: round(density * WORD_K) of every word of WORD_K, uniformly, drawn per (seed, layer,
    linear) like the weight's coins (after commitment: nothing the server commits depends on them)."""
    g = torch.Generator(device=mm.device)
    g.manual_seed(zlib.crc32(f"pos/{mm.seed}/{mm.layer}/{mm.name}".encode()))
    words = -(-k // WORD_K)
    ranks = torch.rand(words, WORD_K, generator=g, device=mm.device).argsort(-1).argsort(-1)
    mask = ranks < round(density * WORD_K)
    r = k - (words - 1) * WORD_K
    if r < WORD_K:  # a short last word: round(density * r) of its r positions
        mask[-1] = False
        mask[-1, :r] = ranks[-1, :r].argsort().argsort() < round(density * r)
    return mask.view(-1)[:k]


def weight_bits(mm, shape, mode: str = "dither") -> torch.Tensor:
    """The weight's coins: one draw per (seed, layer, linear), so every window (and shard) serves one weight; fair bits
    (`dither`) or uniforms in [0, 1) (`sr`)."""
    g = torch.Generator(device=mm.device)
    g.manual_seed(zlib.crc32(f"w/{mm.seed}/{mm.layer}/{mm.name}".encode()))
    if mode == "sr":
        return torch.rand(tuple(shape), generator=g, device=mm.device)
    return torch.randint(0, 2, tuple(shape), generator=g, device=mm.device).float()


def act_coins(mm, mode: str):
    """The activations' coins: the linear's fair bits (`dither`), or uniforms (`sr`) from its own stream per (seed,
    layer, linear, window)."""
    if mode == "dither":
        return mm.coins(mm.x.shape)
    if mode != "sr":
        return None
    g = torch.Generator(device=mm.device)
    g.manual_seed(zlib.crc32(f"u/{mm.seed}/{mm.layer}/{mm.name}/{mm.window}".encode()))
    return torch.rand(tuple(mm.x.shape), generator=g, device=mm.device)


def _check_grid(p, operands, modes=GRID_MODES):
    if p["grid"] not in GRIDS:
        raise ValueError(f"grid.grid is one of {sorted(GRIDS)}")
    for o in operands:
        if p[o] not in modes:
            raise ValueError(f"grid.{o} is one of {modes}")


def _check_grid_linear(p):
    _check_grid(p, ("act", "weight"), LINEAR_GRID_MODES)
    if p["block"] not in (0, 16):
        raise ValueError("grid.block is 0 (one scale per row) or 16 (NVFP4's blocks)")
    if p["scales"] not in SCALES:
        raise ValueError(f"grid.scales is one of {sorted(SCALES)}")
    per = p["density"] * WORD_K
    if not 0 < p["density"] <= 1 or abs(per - round(per)) > 1e-9:
        raise ValueError(f"grid.density is in (0, 1], a multiple of 1/{WORD_K}")


@linear_step("grid", phase="quantize", seeded=lambda p: {p["act"], p["weight"]} != {"round"}, check=_check_grid_linear)
def grid_linear(mm, nxt, *, grid: str, act: str = "dither", weight: str = "dither", block: int = 0,
                scales: str = "odd", density: float = 1.0):
    """Both operands on a low-bit grid (`GRIDS`: fp4c, fp4, int5): activations one scale per token, weights one per
    output row, both applied after the exact integer product; each `round`ed, `dither`ed or `sr` (`on_grid`), the
    weight's coins drawn once per seed and layer (`weight_bits`). `block = 16`: NVFP4's block scales over k on both
    operands, integers from SCALES[`scales`] chosen per block from the data (`on_blocks`), multiplied into the integer
    operands, the per-row and per-token float scales after accumulation. `density` < 1: only a coin-chosen
    density * 64 of every k64 word's positions carry the coin, on both operands (`coin_positions`); the rest round."""
    k = mm.x.shape[1]
    if block and k % block:
        raise ValueError(f"{mm.where}: k = {k} is not a multiple of the block")
    coined = mm.cached("pos", lambda: coin_positions(mm, k, density)) if density < 1 else None
    top = GRIDS[grid][-1] * (SCALES[scales][-1] if block else 1)

    def quantize(x, mode, bits):
        return on_blocks(x, grid, mode, bits, block, scales, coined) if block else on_grid(x, grid, mode, bits, 1, coined)

    def weights():
        w = mm.weight()
        q, s = quantize(w, weight, weight_bits(mm, w.shape, weight) if weight != "round" else None)
        return q.to(torch.int8) if top <= 127 else q, s[:, 0]

    wq, sw = mm.cached("grid", weights)
    xq, sa = quantize(mm.x, act, act_coins(mm, act))
    wide = {"top": top} if top > 127 else {}
    return nxt(mm.integer(xq, wq, **wide)).float() * sa * sw[None, :]


def _check_linear_peel(p):
    if p["alphabet"] not in ("bit", "pm"):
        raise ValueError("peel.alphabet is bit ({0, 1}) or pm (+-1..+-width)")
    if not 1 <= p["width"] <= 64 or not 1 <= p["top"] <= 127:
        raise ValueError("peel: width in 1..64, top in 1..127")


@linear_step("peel", phase="integer", check=_check_linear_peel)
def peel_linear(mm, nxt, *, alphabet: str = "bit", width: int = 1, top: int = 127):
    """The weight-side coin of the int8 design: a low-rank U (output rows x PEEL_L, period PEEL_L in k) served as
    W + U and peeled exactly, X (W + U)^T - fold(X) U^T with fold(X)_l = sum over k = l mod PEEL_L of X_k (L / k of the
    GEMM). `alphabet`: `bit`, U in {0, 1} (one plane at weight 1), or `pm`, U uniform on +-1..+-`width` (the adopted
    design's, -width + planes 1, 2, 4, ..., width + 1). The quantizer's weights must leave room: |W + U| <= `top`
    (16: E4M3's exact integers). On the check window, every layer and linear: the served W + U within +-top and the
    peeled product equal to the direct one. The product the chain returns is the direct one."""
    acc = nxt(mm)
    if not mm.checking:
        return acc
    w = mm.weight().float()
    n, k = w.shape
    g = torch.Generator(device=mm.device)
    g.manual_seed(zlib.crc32(f"peel/{mm.seed}/{mm.layer}/{mm.name}".encode()))
    if alphabet == "bit":
        U = torch.randint(0, 2, (n, PEEL_L), generator=g, device=mm.device).float()
    else:
        U = draw_nonzero(width, (n, PEEL_L), g, mm.device)
    served = w + U[:, torch.arange(k, device=mm.device) % PEEL_L]
    fits = G.within(top, served)
    fold = mm.x.double().view(mm.x.shape[0], k // PEEL_L, PEEL_L).sum(1)
    got = mm.int_product(mm.x, served.clamp(-127, 127).to(torch.int8)).double() - fold @ U.double().T
    mm.gate("peel", bool(fits and torch.equal(got, acc.double())), {"fits": fits})
    return acc


@attention_step("grid", phase="quantize", seeded=lambda p: "dither" in (p["q"], p["k"], p["p"], p["v"]),
                check=lambda p: _check_grid(p, ("q", "k", "p", "v")))
def grid_attention(mm, nxt, *, grid: str, q: str = "dither", k: str = "dither", p: str = "dither", v: str = "dither",
                   p_sink: bool = False):
    """Both attention matmuls on a low-bit grid (`GRIDS`), each operand `round`ed or `dither`ed (`on_grid`): Q and K one
    scale per (head, token), P one per query row (`p_sink`: key 0 in a block of its own, read and scaled on its own),
    V one per (KV head, channel). K's and V's bits are fresh for every query head; P's are drawn only where key <= query
    (the mask is structure), so a dithered P serves no future key."""
    rep = mm.extra["rep"]
    if mm.kind == "qk":
        qi, sq = on_grid(mm.x, grid, q, mm.coins(mm.x.shape) if q == "dither" else None, -1)

        def keys():
            kg = mm.weight().expand(rep, -1, -1)
            return on_grid(kg, grid, k, mm.coins(kg.shape) if k == "dither" else None, -1)
        ki, sk = mm.cached("k", keys)
        return nxt(mm.integer(qi, ki)).float() * (sq * sk.transpose(-1, -2))
    pr = mm.x
    s = pr.shape[-1]

    def values():
        vg = mm.weight().expand(rep, -1, -1)
        return on_grid(vg, grid, v, mm.coins(vg.shape) if v == "dither" else None, -2)
    vi, sv = mm.cached("v", values)
    edges = p_edges(s, 0, p_sink)
    num = None
    for a, b in zip(edges, edges[1:]):
        blk = pr[..., a:b]
        pi, sb = on_grid(blk, grid, p, mm.coins(blk.shape) if p == "dither" else None, -1)
        if p == "dither":
            pi = pi * mm.extra["causal"][:, a:b]
        part = sb.double() * nxt(mm.integer(pi, vi[..., a:b, :], keys=(a, b), v_full=vi))
        num = part if num is None else num + part
    return (num * sv.double()).float()
