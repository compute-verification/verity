"""Every linear of a Llama on the exact int8 grid, evaluated layer by layer so a 70B fits one GPU and every method shares
one pass over the weights: each layer's weights are read once from the safetensors, rotated once, quantized once per
weight setting, and every method's hidden states (all windows) go through it side by side.  Attention stays the model's
bf16 SDPA (causal), so a method's cost is its linears' alone (`attention.py`'s `bf16-lin8r`, `bf16-lin8rd` settings).

The product is `exact_grid.py`'s: x and W multiplied by one orthogonal Q per k, activations one scale per token, weights
one per output channel (unless `wgG`), the integer product exact (torch._int_mm, s8 x s8 -> s32) and scaled after
accumulation.  Method names are '-'-joined tokens:

    bf16        no edit
    rT / dT     activations rounded to [-127, 127] (`r`) or dithered (`d`: floor(x / s) + b, b a fresh coin bit, s =
                absmax / 126, so |X| <= 127), weights rounded to [-T, T]: `r127` is `int8_rot@-` (lin8r), `d63`
                `int8_rot@a:dither:wtop63` (lin8rd), `r63` lin8r63, `d127` lin8rd127; `d111` leaves room for U in +-16
    hq          the flat rotation: Q = signs (H_{k/28} x H_28) / sqrt(k) with Paley's H_28 for k = 7 2^j (every entry
                +-1 / sqrt(k)), instead of exact_grid's H_{k/7} x a random 7 x 7 orthogonal; `nr` no rotation
    wgG         weights one scale per (output channel, G inputs), each group's s32 read and scaled on its own
    gptq        weights rounded by GPTQ (blocks of 128, 1% damping) on the calibration windows' rotated inputs (smoothed
                first under smA)
    smA         SmoothQuant before the rotation, alpha = A / 100: x / m and W m per input channel, m from the calibration
                windows' channel maxima (1 / m folds into the RMSNorm weights before q/k/v and gate/up, and into up_proj's
                rows before down_proj); `smkv` shares o_proj's m across each KV group's query heads, so it folds into
                v_proj's rows (V is quantized per channel, so its integers are unchanged)
    ocN         the N input channels with the largest calibration maxima served on their own (one int8 product of depth
                N, its own per-token and per-row scales), the rest rotated as usual: two s32 reads per output
    t0          the row at position 0 (BOS) on two activation limbs (X = 127 X_hi + X_lo + b), every linear
    xtK         the K rows per window with the largest crest factor (absmax / rms before the rotation) on two limbs
    a2=L.L..    every row of the listed layers' linears on two activation limbs; `w2=L.L..` their weights on two limbs
                (W = 64 W_hi + W_lo, W_lo in 0..63 so W_lo + U fits int8 for U up to 64)
    t0x, lx=L.L..   diagnostics, outside the protocol: position 0 exact, or the listed layers exact
    sN          the dither's seed
    at          attention by `attention.py` (branch cursor/d4-attention-ea24, with its `exact_grid.py`, found in
                --attn-dir) under its method --attn (e.g. the adopted `int8-rot-eqp=w64-poff64-psink-kc-kv=dith-s0`),
                one window at a time: the end to end.  Its peels are checked on window 0 of every layer

    python linears_layerwise.py SNAPSHOT EVAL_IDS.npy OUT_DIR [--calib CALIB_IDS.npy] [--batch B] [--diag D]
        [--attn METHOD --attn-dir DIR] METHOD...
    python linears_layerwise.py --selftest [ATTN_DIR]
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers.models.llama import modeling_llama as L

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
GROUP = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv", "self_attn.o_proj": "o",
         "mlp.gate_proj": "gu", "mlp.up_proj": "gu", "mlp.down_proj": "down"}
#: the last linear reading each input group (its rotated input is dropped after it)
LAST = {"self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"}
A2_BASE, W2_BASE = 127, 64
#: attention.py, once --attn-dir is on the path, and the method its `at` methods' attention runs
ATTN: dict = {"mod": None, "method": None}


def load_attention(attn_dir: str, method: str) -> None:
    sys.path.insert(0, attn_dir)
    import attention
    attention.parse(method)
    ATTN.update(mod=attention, method=method)


# ---------------------------------------------------------------- rotations

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
    when 7 divides k; the same signs), kind `n` the identity.  Applied through the Kronecker factors in FP32."""

    def __init__(self, k: int, kind: str, device):
        self.kind, self.k = kind, k
        if kind == "n":
            return
        g = torch.Generator().manual_seed(k)
        odd = k
        while odd % 2 == 0:
            odd //= 2
        r = torch.linalg.qr(torch.randn(odd, odd, generator=g, dtype=torch.float64))[0] if odd > 1 else torch.ones(1, 1, dtype=torch.float64)
        if kind == "h" and odd > 1:
            assert odd == 7 and k % 28 == 0, k
            r = paley28() / math.sqrt(28)
        elif kind not in ("q", "h"):
            raise KeyError(kind)
        self.signs = (torch.randint(0, 2, (k,), generator=g).double() * 2 - 1).float().to(device)
        self.r = r.float().to(device)
        self.h = sylvester(k // r.shape[0]).float().to(device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "n":
            return x
        m, k = x.shape
        a, b = self.h.shape[0], self.r.shape[0]
        t = (x * self.signs).view(m, a, b)
        if b > 1:
            t = t @ self.r
        t = (t.transpose(1, 2).reshape(m * b, a) @ self.h).view(m, b, a).transpose(1, 2)
        return t.reshape(m, k)

    def dense(self) -> torch.Tensor:
        return self.apply(torch.eye(self.k, device=self.signs.device if self.kind != "n" else "cpu"))


# ---------------------------------------------------------------- methods

def parse(method: str) -> dict:
    spec = {"method": method, "a": None, "top": 127, "rot": "q", "wg": 0, "gptq": False, "sm": 0, "oc": 0, "t0": False,
            "xt": 0, "a2": set(), "w2": set(), "t0x": False, "lx": set(), "seed": 0, "at": False, "smkv": False}
    toks = method.split("-")
    if toks[0] == "bf16":
        if toks[1:] not in ([], ["at"]):
            raise ValueError(method)
        spec["at"] = len(toks) > 1
        return spec
    m0 = re.fullmatch(r"([rd])(\d+)", toks[0])
    if not m0:
        raise ValueError(f"{method}: first token rT or dT")
    spec["a"], spec["top"] = m0.group(1), int(m0.group(2))
    if not 1 <= spec["top"] <= 127:
        raise ValueError(method)
    for t in toks[1:]:
        if t in ("hq", "nr"):
            spec["rot"] = "h" if t == "hq" else "n"
        elif t in ("gptq", "t0", "t0x", "at", "smkv"):
            spec[t] = True
        elif re.fullmatch(r"(wg|sm|oc|xt|s)\d+", t):
            key = re.match(r"[a-z]+", t).group(0)
            spec[{"s": "seed"}.get(key, key)] = int(t[len(key):])
        elif re.fullmatch(r"(a2|w2|lx)=\d+(\.\d+)*", t):
            spec[t[:2]] = {int(v) for v in t[3:].split(".")}
        else:
            raise ValueError(f"{method}: unknown token {t!r}")
    if spec["gptq"] and (spec["wg"] or spec["oc"]):
        raise ValueError(f"{method}: gptq runs on one rotated weight matrix (smoothed or not)")
    if spec["smkv"] and not spec["sm"]:
        raise ValueError(f"{method}: smkv qualifies smA")
    return spec


def wkey(spec: dict) -> tuple:
    """What a method's weight integers depend on (methods with the same key share them)."""
    return (spec["rot"], spec["top"], spec["wg"], spec["gptq"], spec["sm"], spec["smkv"], spec["oc"], tuple(sorted(spec["w2"])))


def hkey(spec: dict) -> tuple:
    """What a GPTQ method's Hessian depends on (its served inputs)."""
    return (spec["rot"], spec["sm"], spec["smkv"])


def needs_calib(spec: dict) -> bool:
    return bool(spec["gptq"] or spec["sm"] or spec["oc"])


# ---------------------------------------------------------------- integer products

def int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a b^T for integer-valued float a (m x k) and int8 b (n x k), |a| <= 127: exact, as int64 (rows padded to 32 for
    torch._int_mm)."""
    m = a.shape[0]
    pad = max(0, 32 - m)
    ai = a.to(torch.int8)
    if pad:
        ai = torch.nn.functional.pad(ai, (0, 0, 0, pad))
    return torch._int_mm(ai, b.T)[:m].long()


def wlimbs_product(a: torch.Tensor, wl: list) -> torch.Tensor:
    return sum(w * int_mm(a, b) for b, w in wl)


def quant_rows(m: torch.Tensor, top: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = m.abs().amax(dim=1).clamp(min=1e-30) / top
    return torch.round(m / s[:, None]).clamp(-top, top), s


def act_quant(x: torch.Tensor, mode: str, bits: torch.Tensor | None, top: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer rows and per-row scales: rounded to [-top, top], or the dither floor(v) + b with v = x / s on [-top + 1,
    top - 1] (top - 1 = 126 on one limb: |X| <= 127)."""
    if mode == "r":
        return quant_rows(x, top)
    t = top - 1
    s = x.abs().amax(dim=1).clamp(min=1e-30) / t
    return torch.floor((x / s[:, None]).clamp(-t, t)) + bits, s


def two_limb_rows(x: torch.Tensor, mode: str, bits: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows on two activation limbs, X = 127 X_hi + X_lo (+ b on X_lo for the dither): (X_hi, X_lo, scale), both limbs
    within int8 (X_hi in -128..127, X_lo + b in 0..127)."""
    top = A2_BASE * A2_BASE
    if mode == "r":
        q, s = quant_rows(x, top)
        hi = torch.div(q, A2_BASE, rounding_mode="floor")
        return hi, q - A2_BASE * hi, s
    t = top - 1
    s = x.abs().amax(dim=1).clamp(min=1e-30) / t
    f = torch.floor((x / s[:, None]).clamp(-t, t))
    hi = torch.div(f, A2_BASE, rounding_mode="floor")
    return hi, f - A2_BASE * hi + bits, s


def gptq(w: torch.Tensor, h: torch.Tensor, top: int, block: int = 128, damp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ (Frantar et al.) of w (n x k, FP32) against the input second moment h (k x k), each row's scale fixed at
    absmax / top: integers in [-top, top] and the scales."""
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


# ---------------------------------------------------------------- one layer's weights

class LayerWeights:
    """A layer's weights, rotated once per rotation kind, and each method's weight integers (by `wkey`)."""

    def __init__(self, layer, li: int, rots: dict, calib: dict | None):
        self.layer, self.li, self.rots, self.calib = layer, li, rots, calib or {}
        self.w = {name: layer.get_submodule(name).weight for name in LINEARS}
        self.cache: dict = {}
        self.rotated: dict = {}

    def rot(self, kind: str, k: int) -> Rotation:
        key = (kind, k)
        if key not in self.rots:
            self.rots[key] = Rotation(k, kind, self.layer.input_layernorm.weight.device)
        return self.rots[key]

    def smooth(self, spec: dict, name: str) -> torch.Tensor | None:
        if not spec["sm"]:
            return None
        g = GROUP[name]
        ax = self.calib["amax"][g]
        aw = torch.stack([self.w[n].float().abs().amax(dim=0) for n in LINEARS if GROUP[n] == g]).amax(dim=0)
        if spec["smkv"] and g == "o":
            # one scale per (KV head, channel), shared by the group's query heads, so 1 / m folds into v_proj's rows
            at = self.layer.self_attn
            rep, d = at.num_key_value_groups, at.head_dim
            ax, aw = (a.view(-1, rep, d).amax(1, keepdim=True).expand(-1, rep, d).reshape(-1) for a in (ax, aw))
        al = spec["sm"] / 100
        return (ax.clamp(min=1e-5) ** al / aw.clamp(min=1e-5) ** (1 - al)).clamp(min=1e-5)

    def hessian(self, spec: dict, name: str) -> torch.Tensor:
        """The calibration windows' second moment of the served (smoothed, rotated) inputs."""
        return self.calib["H"][(hkey(spec), GROUP[name])]

    def served_input(self, spec: dict, name: str, x: torch.Tensor) -> torch.Tensor:
        sm = self.smooth(spec, name)
        return self.rot(spec["rot"], x.shape[1]).apply(x / sm[None, :] if sm is not None else x)

    def outliers(self, spec: dict, name: str) -> torch.Tensor | None:
        if not spec["oc"]:
            return None
        return torch.topk(self.calib["amax"][GROUP[name]], spec["oc"]).indices.sort().values

    def get(self, spec: dict, name: str) -> dict:
        key = (name, wkey(spec))
        if key in self.cache:
            return self.cache[key]
        w = self.w[name].float()
        k = w.shape[1]
        sm, oc = self.smooth(spec, name), self.outliers(spec, name)
        if sm is not None:
            w = w * sm[None, :]
        out: dict = {"sm": sm, "oc": oc}
        if oc is not None:
            w1 = w[:, oc]
            out["w1"] = quant_rows(w1, spec["top"])
            w = w.clone()
            w[:, oc] = 0
        rk = (spec["rot"], name)
        if sm is None and oc is None and rk in self.rotated:
            wr = self.rotated[rk]
        else:
            wr = self.rot(spec["rot"], k).apply(w)
            if sm is None and oc is None:
                self.rotated[rk] = wr
        top = spec["top"]
        if spec["gptq"]:
            q, s = gptq(wr, self.hessian(spec, name), top)
            out["wl"], out["s"] = [(q.to(torch.int8), 1)], s
        elif spec["wg"]:
            g = spec["wg"]
            n = wr.shape[0]
            blocks = wr.view(n, k // g, g)
            s = blocks.abs().amax(dim=2).clamp(min=1e-30) / top
            q = torch.round(blocks / s[:, :, None]).clamp(-top, top).view(n, k)
            out["wl"], out["s"], out["g"] = [(q.to(torch.int8), 1)], s, g
        else:
            q, s = quant_rows(wr, top)
            out["wl"], out["s"] = [(q.to(torch.int8), 1)], s
        if self.li in spec["w2"]:
            q2, s2 = quant_rows(wr, W2_BASE * 127)
            hi = torch.div(q2, W2_BASE, rounding_mode="floor")
            out["wl2"], out["s2"] = [(hi.to(torch.int8), W2_BASE), ((q2 - W2_BASE * hi).to(torch.int8), 1)], s2
        self.cache[key] = out
        return out


# ---------------------------------------------------------------- the hooked product

STATE: dict = {"spec": None}


def bits_for(shape, tag: str, device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(zlib.crc32(tag.encode()))
    return torch.randint(0, 2, shape, generator=g, device=device).float()


def product(spec: dict, name: str, x: torch.Tensor, lw: LayerWeights, ctx: dict) -> torch.Tensor:
    """The method's y = x W^T (FP32) for x (m x k, FP32) at layer lw.li; ctx carries the windows' positions and the
    rotated-input cache."""
    li = lw.li
    if li in spec["lx"]:
        return x @ lw.w[name].float().T
    wd = lw.get(spec, name)
    m, k = x.shape
    tag = f"{spec['seed']}/{li}/{name}/{ctx['w0']}"
    ckey = (GROUP[name], spec["rot"], spec["sm"], spec["smkv"], spec["oc"])
    if ctx.get("xr_key") == ckey:
        xr, x1 = ctx["xr"], ctx["x1"]
    else:
        xs = x / wd["sm"][None, :] if wd["sm"] is not None else x
        x1 = None
        if wd["oc"] is not None:
            x1 = xs[:, wd["oc"]]
            xs = xs.clone()
            xs[:, wd["oc"]] = 0
        xr = lw.rot(spec["rot"], k).apply(xs)
        ctx.update(xr_key=ckey, xr=xr, x1=x1, crest=(x.abs().amax(1) / x.pow(2).mean(1).sqrt().clamp(min=1e-30)))
    bits = bits_for((m, k), tag, x.device) if spec["a"] == "d" else None
    two_layer = li in spec["a2"]
    wl, sw = wd["wl"], wd["s"]
    if two_layer:
        rows = torch.arange(m, device=x.device)
    else:
        sel = []
        if spec["t0"]:
            sel.append(ctx["pos0"])
        if spec["xt"]:
            T = ctx["T"]
            cr = ctx["crest"].view(-1, T)
            top = torch.topk(cr, spec["xt"], dim=1).indices + torch.arange(cr.shape[0], device=x.device)[:, None] * T
            sel.append(top.flatten())
        rows = torch.unique(torch.cat(sel)) if sel else None
    if "wl2" in wd or two_layer:
        # every row two limbs (and the weights too, under w2): X W^T from the limbs, exact
        wl_, sw_ = (wd["wl2"], wd["s2"]) if "wl2" in wd else (wl, sw)
        if two_layer:
            hi, lo, sa = two_limb_rows(xr, spec["a"], bits)
            acc = A2_BASE * wlimbs_product(hi, wl_) + wlimbs_product(lo, wl_)
        else:
            q, sa = act_quant(xr, spec["a"], bits, 127)
            acc = wlimbs_product(q, wl_)
        y = acc.float() * sa[:, None] * sw_[None, :]
        rows = None
    elif "g" in wd:
        q, sa = act_quant(xr, spec["a"], bits, 127)
        g = wd["g"]
        y = torch.zeros(m, wl[0][0].shape[0], device=x.device)
        for j in range(k // g):
            y += int_mm(q[:, j * g:(j + 1) * g].contiguous(), wl[0][0][:, j * g:(j + 1) * g].contiguous()).float() * wd["s"][None, :, j]
        y *= sa[:, None]
    else:
        q, sa = act_quant(xr, spec["a"], bits, 127)
        y = wlimbs_product(q, wl).float() * sa[:, None] * sw[None, :]
    if rows is not None and len(rows):
        assert "g" not in wd, "two-limb rows with grouped weights"
        hi, lo, sa2 = two_limb_rows(xr[rows], spec["a"], bits[rows] if bits is not None else None)
        acc = A2_BASE * wlimbs_product(hi, wl) + wlimbs_product(lo, wl)
        y[rows] = acc.float() * sa2[:, None] * sw[None, :]
    if x1 is not None:
        b1 = bits_for(tuple(x1.shape), tag + "/oc", x.device) if spec["a"] == "d" else None
        q1, s1 = act_quant(x1, spec["a"], b1, 127)
        (w1q, w1s) = wd["w1"]
        if rows is not None and len(rows):
            q1r, s1r = quant_rows(x1[rows], A2_BASE * A2_BASE)
            y1 = (q1 @ w1q.T) * s1[:, None] * w1s[None, :]
            y1[rows] = (q1r.double() @ w1q.double().T).float() * s1r[:, None] * w1s[None, :]
        else:
            y1 = (q1.double() @ w1q.double().T).float() * s1[:, None] * w1s[None, :]
        y = y + y1
    if spec["t0x"]:
        p0 = ctx["pos0"]
        y[p0] = x[p0] @ lw.w[name].float().T
    return y


def rel_rows(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (y - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-30)


def make_hook(name: str, lw: LayerWeights):
    def fn(mod, inp, out):
        spec, ctx = STATE["spec"], STATE["ctx"]
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
        if ctx.get("collect") is not None:
            ctx["collect"](name, x)
        if ctx.get("diag") is not None:
            ctx["diag"](name, x, lw)
        if spec is None or spec["a"] is None:
            res = None
        else:
            res = product(spec, name, x, lw, ctx).to(out.dtype).reshape(out.shape)
        if name in LAST:
            ctx.pop("xr_key", None)
            ctx.pop("xr", None)
            ctx.pop("x1", None)
        return res
    return fn


# ---------------------------------------------------------------- loading

class Weights:
    """Tensors by name from a snapshot's safetensors, read lazily (a background thread prefetches the next layer)."""

    def __init__(self, snapshot: str):
        self.snap = Path(snapshot)
        idx = self.snap / "model.safetensors.index.json"
        if idx.exists():
            self.where = json.loads(idx.read_text())["weight_map"]
        else:
            with safe_open(str(self.snap / "model.safetensors"), "pt") as f:
                self.where = {k: "model.safetensors" for k in f.keys()}
        self.pending: dict = {}

    def read(self, names: list[str]) -> dict:
        out, by_file = {}, {}
        for n in names:
            by_file.setdefault(self.where[n], []).append(n)
        for fn, ns in by_file.items():
            with safe_open(str(self.snap / fn), "pt") as f:
                for n in ns:
                    out[n] = f.get_tensor(n)
        return out

    def layer_names(self, li: int) -> list[str]:
        p = f"model.layers.{li}."
        return [n for n in self.where if n.startswith(p)]

    def prefetch(self, li: int):
        def run():
            self.pending[li] = self.read(self.layer_names(li))
        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.pending[("thread", li)] = t

    def layer(self, li: int) -> dict:
        t = self.pending.pop(("thread", li), None)
        if t is not None:
            t.join()
            return self.pending.pop(li)
        return self.read(self.layer_names(li))


def build_layer(config, li: int, tensors: dict, device) -> torch.nn.Module:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            layer = L.LlamaDecoderLayer(config, li)
    finally:
        torch.set_default_dtype(prev)
    layer = layer.to_empty(device=device)
    p = f"model.layers.{li}."
    layer.load_state_dict({n[len(p):]: t.to(torch.bfloat16) for n, t in tensors.items()}, strict=True)
    return layer.eval()


# ---------------------------------------------------------------- the evaluation

def evaluate(snapshot: str, ids: np.ndarray, methods: list[str], out: Path, calib_ids: np.ndarray | None = None,
             batch: int = 4, diag_windows: int = 0, device="cuda", log=print) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = transformers.AutoConfig.from_pretrained(snapshot)
    config._attn_implementation = "sdpa"
    specs = {m: parse(m) for m in methods}
    if any(needs_calib(s) for s in specs.values()) and calib_ids is None:
        raise ValueError("gptq, sm and oc need --calib")
    A = ATTN["mod"]
    if any(s["at"] for s in specs.values()) and A is None:
        raise ValueError("`at` needs --attn and --attn-dir")
    aspec = A.parse(ATTN["method"]) if A is not None else None
    attn_rec: dict = {m: {"peel": [], "err": [0.0, 0.0]} for m in methods if specs[m]["at"]}
    W = Weights(snapshot)
    n_layers = config.num_hidden_layers
    W.prefetch(0)
    top = W.read(["model.embed_tokens.weight", "model.norm.weight"])
    emb = top["model.embed_tokens.weight"].to(device)
    N, T = ids.shape
    ids_t = torch.from_numpy(ids).to(device)
    rotary = L.LlamaRotaryEmbedding(config, device=device)
    pos = torch.arange(T, device=device)[None]
    streams = {m: emb[ids_t].clone() for m in ["bf16"] + [m for m in methods if m != "bf16"]}
    cal = emb[torch.from_numpy(calib_ids).to(device)].clone() if calib_ids is not None else None
    del emb
    hkeys = sorted({hkey(s) for s in specs.values() if s["gptq"]})
    rots: dict = {}
    diag: list = []
    diverge: dict = {m: [] for m in methods if m != "bf16"}
    resid: list = []
    t_start = time.time()
    for li in range(n_layers):
        t0 = time.time()
        layer = build_layer(config, li, W.layer(li), device)
        if li + 1 < n_layers:
            W.prefetch(li + 1)
        lw = LayerWeights(layer, li, rots, None)
        hooks = [layer.get_submodule(n).register_forward_hook(make_hook(n, lw)) for n in LINEARS]

        def run(h: torch.Tensor, spec, w0: int, extra=None) -> torch.Tensor:
            if spec is not None and spec["at"]:
                return torch.cat([run_at(h[j:j + 1], spec, w0 + j) for j in range(h.shape[0])])
            B = h.shape[0]
            ctx = {"w0": w0, "T": T, "pos0": torch.arange(B, device=device) * T}
            if extra:
                ctx.update(extra)
            STATE.update(spec=spec, ctx=ctx)
            cos, sin = rotary(h, pos)
            with torch.no_grad():
                return layer(h, attention_mask=None, position_embeddings=(cos, sin))

        def run_at(h: torch.Tensor, spec, w: int) -> torch.Tensor:
            # attention.py's edited attention replaces the eager path, one window at a time (its window seeds the noise)
            config._attn_implementation = "eager"
            A.STATE.update(spec=aspec, calls=0, err=[0.0, 0.0], record={}, peel=[], window=w)
            try:
                y = run(h, {**spec, "at": False}, w)
            finally:
                config._attn_implementation = "sdpa"
                A.STATE["spec"] = None
            assert A.STATE["calls"] == 1, A.STATE["calls"]
            rec = attn_rec[spec["method"]]
            rec["peel"].extend(A.STATE["peel"])
            rec["err"] = [a + b for a, b in zip(rec["err"], A.STATE["err"])]
            return y

        try:
            # calibration on the bf16 stream: every input group's channel maxima, then (GPTQ) the second moments of its
            # served inputs, a Gram matrix of the smoothed, rotated rows (positive semidefinite as computed)
            if cal is not None:
                amax: dict = {}
                H: dict = {}
                first = ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")

                def collect(name, x):
                    if name in first:
                        g, a = GROUP[name], x.abs().amax(0)
                        amax[g] = torch.maximum(amax[g], a) if g in amax else a

                def collect_h(name, x):
                    if name in first:
                        for hk in hkeys:
                            xr = lw.served_input({"rot": hk[0], "sm": hk[1], "smkv": hk[2]}, name, x)
                            h = xr.T @ xr
                            k2 = (hk, GROUP[name])
                            H[k2] = H[k2] + h if k2 in H else h
                            del xr, h

                calib_in = cal.clone() if hkeys else None
                for i in range(0, len(cal), batch):
                    cal[i:i + batch] = run(cal[i:i + batch], None, i, {"collect": collect})
                lw.calib = {"amax": amax}
                if hkeys:
                    for i in range(0, len(cal), batch):
                        run(calib_in[i:i + batch], None, i, {"collect": collect_h})
                    del calib_in
                ntok = cal.shape[0] * cal.shape[1]
                lw.calib["H"] = {k: v * (2 / ntok) for k, v in H.items()}
            # the bf16 stream, with diagnostics on its first diag_windows windows
            ref_out = torch.empty_like(streams["bf16"])
            dlayer: dict = {}
            for i in range(0, N, batch):
                extra = None
                if i < diag_windows:
                    extra = {"diag": make_diag(dlayer, specs, ids, i, T, lw)}
                ref_out[i:i + batch] = run(streams["bf16"][i:i + batch], None, i, extra)
            if dlayer:
                diag.append({"layer": li, **{n: finish_diag(v) for n, v in dlayer.items()}})
                hb = ref_out[:diag_windows].float()
                mx = hb.abs().amax(-1)
                flat = mx.flatten()
                topi = torch.topk(flat, 4).indices
                resid.append({"layer": li, "median_token_max": float(flat.median()),
                              "top": [{"w": int(j // T), "pos": int(j % T), "id": int(ids[j // T, j % T]),
                                       "max": float(flat[j]), "channel": int(hb.view(-1, hb.shape[-1])[j].abs().argmax())}
                                      for j in topi.tolist()]})
            for m in methods:
                if m == "bf16":
                    continue
                h = streams[m]
                for i in range(0, N, batch):
                    h[i:i + batch] = run(h[i:i + batch], specs[m], i)
                diverge[m].append(divergence(h, ref_out, li))
            streams["bf16"] = ref_out
        finally:
            for hk in hooks:
                hk.remove()
            STATE.update(spec=None, ctx=None)
        del lw, layer
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None
        log(json.dumps({"layer": li, "seconds": round(time.time() - t0, 1),
                        "div": {m: round(d[-1]["rel"], 5) for m, d in diverge.items()}}))
    norm_w = top["model.norm.weight"].to(device)
    head = W.read(["lm_head.weight"] if "lm_head.weight" in W.where else ["model.embed_tokens.weight"])
    head_w = next(iter(head.values())).to(device)
    norm = L.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(norm_w)
    results = {}
    nll_all = {}
    for m in ["bf16"] + [m for m in methods if m != "bf16"]:
        nlls = []
        with torch.no_grad():
            for w in range(N):
                logits = (norm(streams[m][w]) @ head_w.T).float()
                nlls.append(torch.nn.functional.cross_entropy(logits[:-1], ids_t[w, 1:], reduction="none"))
        nll = torch.stack(nlls)
        nll_all[m] = nll.cpu().numpy()
        results[m] = {"method": m, "ppl": float(torch.exp(nll.mean())),
                      "ppl_windows": [float(torch.exp(x.mean())) for x in nll], "nll": float(nll.mean())}
    base = nll_all["bf16"]
    for m in results:
        if m == "bf16":
            continue
        d = nll_all[m] - base
        flat = np.argsort(d.flatten())[::-1][:12]
        results[m]["worst_tokens"] = [{"w": int(j // (T - 1)), "pos": int(j % (T - 1)) + 1,
                                       "id": int(ids[j // (T - 1), j % (T - 1) + 1]), "dnll": float(d.flatten()[j])}
                                      for j in flat]
        results[m]["divergence"] = diverge[m]
        if m in attn_rec:
            rec = attn_rec[m]
            results[m].update(attn=ATTN["method"], attn_peel_exact=all(rec["peel"]), attn_peel_checks=len(rec["peel"]),
                              attn_rel_err_w0=math.sqrt(rec["err"][0] / rec["err"][1]) if rec["err"][1] else None)
    np.savez_compressed(out / "nll.npz", **{m.replace("=", "_"): v for m, v in nll_all.items()})
    results["_meta"] = {"windows": int(N), "T": int(T), "batch": batch, "diag_windows": diag_windows,
                        "seconds": round(time.time() - t_start, 1), "diag": diag, "resid": resid}
    return results


DIAG_CONFIGS = ("r127", "r63", "d127", "d63", "d63-hq", "r127-hq")


def make_diag(dlayer: dict, specs: dict, ids: np.ndarray, w0: int, T: int, lw: LayerWeights):
    """Per linear on the bf16 stream: the input's per-token statistics before and after the rotation, and each of
    DIAG_CONFIGS' per-token relative error against the exact product."""
    cfgs = {c: parse(c) for c in DIAG_CONFIGS}

    def fn(name, x, lw_):
        ref = x @ lw_.w[name].float().T
        rec = dlayer.setdefault(name, {"err": {c: [] for c in cfgs}, "absmax": [], "crest": [], "crest_rot": [],
                                       "chan": None, "where": []})
        a = x.abs()
        am = a.amax(1)
        rec["absmax"].append(am.cpu())
        rec["crest"].append((am / x.pow(2).mean(1).sqrt().clamp(min=1e-30)).cpu())
        xr = lw_.rot("q", x.shape[1]).apply(x)
        rec["crest_rot"].append((xr.abs().amax(1) / xr.pow(2).mean(1).sqrt().clamp(min=1e-30)).cpu())
        ch = a.amax(0).cpu()
        rec["chan"] = ch if rec["chan"] is None else torch.maximum(rec["chan"], ch)
        rec["w0"] = rec.get("w0", []) + [w0]
        for c, s in cfgs.items():
            ctx = {"w0": w0, "T": T, "pos0": torch.arange(x.shape[0] // T, device=x.device) * T}
            en = (product(s, name, x, lw_, ctx) - ref).norm(dim=1)
            rn = ref.norm(dim=1)
            rec["err"][c].append(torch.stack([en / rn.clamp(min=1e-30), en, rn]).cpu())
        rec["ids"] = ids
        rec["T"] = T
    return fn


def finish_diag(rec: dict) -> dict:
    T = rec["T"]
    am, cr, crr = (torch.cat(rec[k]) for k in ("absmax", "crest", "crest_rot"))
    out = {"absmax_max": float(am.max()), "absmax_median": float(am.median()), "crest_max": float(cr.max()),
           "crest_median": float(cr.median()), "crest_rot_max": float(crr.max()), "crest_rot_median": float(crr.median())}
    j = int(am.argmax())
    out["absmax_at"] = {"w": j // T, "pos": j % T, "id": int(rec["ids"][j // T, j % T])}
    top = torch.topk(rec["chan"], 6)
    out["top_channels"] = [[int(i), float(v)] for v, i in zip(top.values, top.indices)]
    pos = torch.arange(len(am)) % T
    for c, parts in rec["err"].items():
        e = torch.cat(parts, dim=1)
        rel, en, rn = e[0], e[1], e[2]
        k = int(rel.argmax())
        rest = pos != 0
        out[c] = {"fro": float(en.pow(2).sum().sqrt() / rn.pow(2).sum().sqrt()),
                  "fro_rest": float(en[rest].pow(2).sum().sqrt() / rn[rest].pow(2).sum().sqrt()),
                  "median": float(rel.median()), "max": float(rel.max()),
                  "max_at": {"w": k // T, "pos": k % T, "id": int(rec["ids"][k // T, k % T])},
                  "pos0": float(rel[pos == 0].mean())}
    return out


def divergence(h: torch.Tensor, ref: torch.Tensor, li: int) -> dict:
    T = h.shape[1]
    d = (h.float() - ref.float()).norm(dim=-1)
    r = ref.float().norm(dim=-1).clamp(min=1e-30)
    rel = d / r
    flat = rel.flatten()
    k = int(flat.argmax())
    return {"layer": li, "rel": float(d.pow(2).sum().sqrt() / r.pow(2).sum().sqrt()),
            "rel_rest": float(d[:, 1:].pow(2).sum().sqrt() / r[:, 1:].pow(2).sum().sqrt()),
            "pos0": float(rel[:, 0].mean()), "median": float(flat.median()), "max": float(flat.max()),
            "max_at": [k // T, k % T]}


# ---------------------------------------------------------------- selftest

def selftest(attn_dir: str | None = None) -> dict:
    torch.manual_seed(0)
    # whole-model divergences are compared; one thread fixes the CPU kernels' reduction order (attention.py does the same)
    torch.set_num_threads(1)
    out: dict = {}
    if attn_dir:
        sys.path.insert(0, attn_dir)
    h28 = paley28()
    assert torch.equal(h28 @ h28.T, 28 * torch.eye(28, dtype=torch.float64)) and set(h28.unique().tolist()) == {-1.0, 1.0}
    for k in (56, 64, 448):
        for kind in ("q", "h"):
            r = Rotation(k, kind, "cpu")
            qd = r.dense().double()
            assert torch.allclose(qd @ qd.T, torch.eye(k, dtype=torch.float64), atol=1e-5), (k, kind)
            if kind == "h":
                assert torch.allclose(qd.abs(), torch.full_like(qd, 1 / math.sqrt(k)), atol=1e-6), k
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import exact_grid
            ref = exact_grid.rotation(k, "cpu")
            assert torch.allclose(Rotation(k, "q", "cpu").dense().double(), ref, atol=1e-5), k
            out["matches_exact_grid_rotation"] = True
        except ImportError:
            out["matches_exact_grid_rotation"] = "exact_grid not importable"
    # a tiny Llama saved as safetensors, through the layer-wise path against the model's own forward
    import tempfile
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=448, num_hidden_layers=3, num_attention_heads=8,
                        num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    model = L.LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        # a massive activation: one residual channel large at position 0, as Llama's BOS
        model.model.embed_tokens.weight[7, 3] = 60.0
    ids = torch.randint(0, 512, (4, 64))
    ids[:, 0] = 7
    tmp = Path(tempfile.mkdtemp())
    model.save_pretrained(tmp, safe_serialization=True)
    with torch.no_grad():
        logits = model(ids).logits.float()
    want = float(torch.exp(torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512), ids[:, 1:].reshape(-1))))
    calib = torch.randint(0, 512, (4, 64))
    calib[:, 0] = 7
    methods = ["bf16", "r127", "r63", "d63", "d63-s1", "d63-hq", "d63-t0", "d63-t0x", "d63-xt4", "d111", "d63-wg32",
               "d63-gptq", "r63-gptq", "d63-sm50", "d63-sm50-smkv", "d63-sm50-gptq", "d63-oc8", "d63-a2=0.1.2", "d63-a2=0.1.2-w2=0.1.2",
               "r127-lx=0.1.2", "d63-oc8-t0"]
    res = evaluate(str(tmp), ids.numpy(), methods, tmp, calib_ids=calib.numpy(), batch=2, diag_windows=2, device="cpu",
                   log=lambda s: None)
    ppl = {m: res[m]["ppl"] for m in methods}
    out["ppl"] = ppl
    out["want_bf16"] = want
    assert abs(ppl["bf16"] / want - 1) < 5e-3, (ppl["bf16"], want)
    assert abs(ppl["r127-lx=0.1.2"] / ppl["bf16"] - 1) < 5e-3, ppl
    assert ppl["d63-s1"] != ppl["d63"]
    assert abs(ppl["d63-a2=0.1.2-w2=0.1.2"] / ppl["bf16"] - 1) < 2e-3, ppl
    div = {m: res[m]["divergence"][-1]["rel"] for m in methods if m != "bf16"}
    out["div"] = div
    assert div["r127"] < div["r63"] and div["r127"] < div["d63"], div
    assert div["d63-a2=0.1.2"] < div["d63"], div
    out["xt4_layers"] = [round(x["rel"], 5) for x in res["d63-xt4"]["divergence"]]
    out["d63_layers"] = [round(x["rel"], 5) for x in res["d63"]["divergence"]]
    assert div["d63-a2=0.1.2-w2=0.1.2"] < 0.75 * div["d63-a2=0.1.2"] and div["r127-lx=0.1.2"] < div["r127"], div
    assert res["d63-t0x"]["divergence"][-1]["pos0"] < 1e-2 and res["d63-t0"]["divergence"][-1]["pos0"] < res["d63"]["divergence"][-1]["pos0"]
    assert math.isfinite(ppl["d63-sm50-smkv"]) and ppl["d63-sm50-smkv"] != ppl["d63-sm50"], ppl
    assert math.isfinite(ppl["d63-sm50-gptq"]) and ppl["d63-sm50-gptq"] != ppl["d63-sm50"], ppl
    lw0 = LayerWeights(model.model.layers[0], 0, {}, {"amax": {"o": torch.rand(cfg.hidden_size) * 10}})
    m_kv = lw0.smooth(parse("d63-sm50-smkv"), "self_attn.o_proj").view(cfg.num_key_value_heads, -1, cfg.head_dim)
    assert torch.equal(m_kv, m_kv[:, :1].expand_as(m_kv)) and not torch.equal(m_kv, lw0.smooth(parse("d63-sm50"), "self_attn.o_proj").view_as(m_kv))
    d = res["_meta"]["diag"][0]
    assert set(DIAG_CONFIGS) <= set(d["self_attn.q_proj"]) and d["self_attn.q_proj"]["r127"]["fro"] < d["self_attn.q_proj"]["d63"]["fro"]
    out["diag_l0_q"] = d["self_attn.q_proj"]
    if attn_dir:
        # attention.py's unedited FP32 attention through the `at` path is the model's, and the adopted design's peels
        # are exact on window 0 of every layer
        load_attention(attn_dir, "ref")
        ra = evaluate(str(tmp), ids.numpy(), ["bf16", "bf16-at", "r127-lx=0.1.2-at"], tmp, batch=2, device="cpu",
                      log=lambda s: None)
        out["at_ref_ppl"] = {m: ra[m]["ppl"] for m in ("bf16", "bf16-at", "r127-lx=0.1.2-at")}
        assert all(abs(v / ra["bf16"]["ppl"] - 1) < 5e-3 for v in out["at_ref_ppl"].values()), out["at_ref_ppl"]
        adopted = "int8-rot-eqp=w64-poff64-psink-kc-kv=dith-s1"
        load_attention(attn_dir, adopted)
        ra = evaluate(str(tmp), ids.numpy(), ["bf16", "bf16-at", "d63-at", "d63-sm50-at"], tmp, calib_ids=calib.numpy(),
                      batch=2, device="cpu", log=lambda s: None)
        out["at_adopted"] = {m: {k: ra[m].get(k) for k in ("ppl", "attn_peel_exact", "attn_peel_checks", "attn_rel_err_w0")}
                             for m in ("bf16-at", "d63-at", "d63-sm50-at")}
        for m, r in out["at_adopted"].items():
            assert r["attn_peel_exact"] and r["attn_peel_checks"] == cfg.num_hidden_layers * cfg.num_key_value_heads * 3, (m, r)
        assert ra["d63-at"]["ppl"] != ra["bf16-at"]["ppl"] and math.isfinite(ra["d63-sm50-at"]["ppl"]), out["at_adopted"]
        ATTN.update(mod=None, method=None)
    # the product, per method, against an independent float64 computation on one linear
    x = torch.randn(64, 448) * torch.rand(448)
    x[0, 5] = 300.0
    wmat = torch.randn(256, 448)
    lin = torch.nn.Linear(448, 256, bias=False)
    with torch.no_grad():
        lin.weight.copy_(wmat)
    fake = type("F", (), {})()
    fake.input_layernorm = type("N", (), {"weight": torch.ones(1)})()
    fake.get_submodule = lambda n: lin
    lw = LayerWeights(fake, 0, {}, {"amax": {"down": x.abs().amax(0)}, "H": {}})
    rot = Rotation(448, "q", "cpu")
    ctx = {"w0": 0, "T": 64, "pos0": torch.tensor([0])}
    y = product(parse("r63"), "mlp.down_proj", x, lw, dict(ctx))
    xq, sx = quant_rows(rot.apply(x), 127)
    wq, sw = quant_rows(rot.apply(wmat), 63)
    assert torch.allclose(y.double(), (xq.double() @ wq.double().T) * sx.double()[:, None] * sw.double()[None, :], rtol=1e-5, atol=1e-4)
    ref = x.double() @ wmat.double().T
    err = {m: float((product(parse(m), "mlp.down_proj", x, lw, dict(ctx)).double() - ref).norm() / ref.norm())
           for m in ("r127", "r63", "d63", "d63-t0", "d63-oc8", "d63-a2=0", "d63-a2=0-w2=0", "d63-wg32")}
    out["one_linear_err"] = err
    assert err["d63-a2=0"] < err["d63"] and err["d63-a2=0-w2=0"] < err["d63-a2=0"] / 10 and err["d63-oc8"] < err["d63"], err
    e0 = {m: float(rel_rows(product(parse(m), "mlp.down_proj", x, lw, dict(ctx)).double(), ref)[0]) for m in ("d63", "d63-t0", "d63-oc8")}
    assert e0["d63-t0"] < e0["d63"] and e0["d63-oc8"] < e0["d63"], e0
    out["row0_err"] = e0
    hi, lo, _ = two_limb_rows(torch.randn(50, 64) * 10, "d", bits_for((50, 64), "t", "cpu"))
    assert float(hi.min()) >= -128 and float(hi.max()) <= 127 and float(lo.min()) >= 0 and float(lo.max()) <= 127
    q, _ = act_quant(torch.randn(50, 64), "d", bits_for((50, 64), "t", "cpu"), 127)
    assert float(q.abs().max()) <= 127
    return out


def main(argv: list[str]) -> int:
    if argv[0] == "--selftest":
        print(json.dumps(selftest(argv[1] if len(argv) > 1 else None), indent=1, default=str))
        return 0
    snapshot, ids_path, out = argv[0], argv[1], Path(argv[2])
    rest = argv[3:]
    calib, batch, diag, attn, attn_dir = None, 4, 0, None, None
    while rest and rest[0].startswith("--"):
        flag, val, rest = rest[0], rest[1], rest[2:]
        if flag == "--calib":
            calib = np.load(val)
        elif flag == "--batch":
            batch = int(val)
        elif flag == "--diag":
            diag = int(val)
        elif flag == "--attn":
            attn = val
        elif flag == "--attn-dir":
            attn_dir = val
        else:
            raise SystemExit(f"unknown flag {flag}")
    if (attn is None) != (attn_dir is None):
        raise SystemExit("--attn and --attn-dir go together")
    if attn:
        load_attention(attn_dir, attn)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.load(ids_path)
    for m in rest:
        parse(m)
    res = evaluate(snapshot, ids, rest, out, calib_ids=calib, batch=batch, diag_windows=diag,
                   log=lambda s: print(s, file=sys.stderr, flush=True))
    meta = res.pop("_meta")
    (out / "diag.json").write_text(json.dumps(meta))
    with open(out / "results.jsonl", "w") as f:
        for m, r in res.items():
            f.write(json.dumps(r) + "\n")
    bf = res["bf16"]["ppl"]
    base = res.get("r127", {}).get("ppl")
    lines = ["| method | perplexity | Δ% vs bf16 | Δ% vs r127 (plain int8) | windows min-max |", "|---|---:|---:|---:|---|"]
    for m, r in res.items():
        db = 100 * (r["ppl"] / bf - 1)
        di = f"{100 * (r['ppl'] / base - 1):+.2f}" if base else "-"
        lines.append(f"| {m} | {r['ppl']:.4f} | {db:+.2f} | {di} | {min(r['ppl_windows']):.3f}-{max(r['ppl_windows']):.3f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"seconds": meta["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
