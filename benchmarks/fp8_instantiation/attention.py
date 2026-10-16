"""The error side of the security-against-error frontier for attention: Llama-3.1-8B's WikiText-2 perplexity with every
linear in bf16 and both attention matmuls of every head and layer, S = Q K^T and O = P V, served from edited operands.

An edit puts each operand on an exact integer grid, adds fresh unpeeled noise in units of the grid's LSB, and serves the
exact integer product times the operands' scales.  The scales factor out of each k-sum, so the integer product is what
the tensor core returns (IMMA s8 x s8 -> s32 for int8; QMMA e4m3 with FP32 accumulation for E4M3's integers, whose every
partial sum stays below 2^24 here):

    Q, K       one scale per (head, token): the row of head_dim entries the k-sum of S runs over
    P          after the softmax, one scale per query row (`prow`), or the fixed 1/M (`pfix`); `prow2` is P per row on two
               int8 limbs, P = B P_hi + P_lo with B = 128 - max|n|, its noise on P_lo (P V costs two int8 matmuls);
               `vchan2` is V per channel on two signed limbs the same way, and `qk2` puts Q and K on two limbs each
    V          one scale per KV head (`vhead`) or per (KV head, channel) (`vchan`); the k-sum of O runs over tokens

Noise is drawn per instance: a fresh draw for every entry of every query head's Q, K, P and V in every layer (K and V are
shared by GQA's four query heads; each head's product gets its own noise).  P's noise lands only on the causal support
(key <= query), so no query reads a later token.  An operand that carries noise is quantized to the top M = top - max|n|,
so its noised integers stay on the grid (|x| <= 127 for int8, <= 16 for E4M3's integers).  `rot` multiplies Q, K and V by
one random Hadamard R of the head dimension (random signs times Sylvester's H_128) before quantizing and the output by R^T
after, which preserves S = (Q R)(K R)^T and O = P V exactly before the edit.

Method names are '-'-joined tokens:

    grid       `int8` (both matmuls), `e4m3` (both on E4M3's integers <= 16), `e4m3qk` (Q K^T on E4M3's integers, P V on int8)
    P, V, QK   `prow`/`pfix`/`prow2`, `vhead`/`vchan`/`vchan2`, `qk2`
    noise      an alphabet for all four operands: `none`, `pm1` {+-1}, `pm13` {+-1, +-3}, `pm57` {+-5, +-7}, `u1` ... `u16`
               uniform {-k..k}, `w8`/`w64` {+-1..+-k}, `pm1245` {+-1, +-2, +-4, +-5}, `dE` sparse (+-1 each with
               probability 2^-E / 2, else 0); `qk`/`pv` limits it to that matmul's operands; `X=ALPHABET` sets the
               operands X (any of `q`, `k`, `p`, `v`) alone.  `dith` (K, V): the dither X = floor(x / s) + b, b a fresh
               bit and s the smallest scale with x / s in [-128, 127), so X stays within int8
               (note:20261016T0525Z-draft-fullfloor-s-r6)
    peeled     `eX=ALPHABET` (X in `q`, `p`, `qp`): low-rank noise the server sees and the model doesn't
               (note:20261016T0001Z-draft-rs-d2-attention-lowrank-peel).  E_ik = U_{i, k mod L}, L = 32, with U (tokens x L)
               one draw per layer shared by its heads; P's E only where key <= query.  The served products are
               (P + E_P)(V + F_V) and (Q + E_Q)(K + F_K)^T, and the exact peels U (W^T (V + F_V)) (the causal prefix scan)
               and U (W^T (K + F_K))^T are subtracted, so the model gets P (V + F_V) and Q (K + F_K)^T exactly; the
               peeled operand's top drops by max|E|.  One limb only (`prow`/`pfix`, no `qk2`).  `poffN`: P keeps its
               full grid and is served as P - N + E_P on the causal support (within int8's [-128, 127]); the public
               offset is peeled with E, N sum_{k <= i} (V + F_V)_k
    P's base   one-limb int8 P only: `pnorm` divides O's integer rows by P's integer row sums (instead of multiplying by
               P's scale); `pblkB` gives P one scale per (query row, block of B keys), each block's product read and
               scaled on its own; `psink` puts key 0 alone in a block; `ptopT` sets P's top (with `poffN` to serve
               P - N within int8, the offset added back as N sum_{k <= i} V_k, peeled or not)
    K's base   `kc` serves K - mu, mu its per-channel mean over the window (any mu leaves softmax(Q K^T) unchanged, since
               q mu is the same for every key of q's row), so a part of K constant over tokens (Qwen2.5's k bias) no
               longer sets K's scale
    linears    `lin8r`: every linear on exact_grid.py's `int8_rot@-` (rotated, int8 per row, s8 x s8 -> s32); `lin8rd`
               on `int8_rot@a:dither:wtop63:sN` (the activations served as floor(x / s) + b, the weights on [-63, 63])
    `rot`, `sN` (noise seed, also the linears' under `lin8rd`)

`bf16` is the model's own eager attention (Q K^T rounded to bf16 before the scale), `sdpa` its bf16 attention through
torch's fused causal kernel (scores in FP32, as served), and `ref` this file's attention in FP32 with no edit (`ref-rot`
checks the rotation).  IDS is one window (T,) or several (N, T) (`windows.py`); perplexity is over every token of every window, and
window w draws its own noise (window 0's is the single-window runs').  On window 0 every peel is checked on every layer:
Q K^T's per KV group, and P V's per KV group and key block.

The models are Llama's and Qwen2's (eager attention patched in both; Qwen2.5 has no BOS, so its key 0 is an ordinary
token).  Past 8K the s x s work runs on chunks of query rows (ROWS_WHOLE), and so do bf16's eager attention and the loss;
the rows are independent, so the chunks change no bit of the edited methods (the selftest).  With several GPUs visible
the model is split over them by layers (Llama-3.1-70B on two).

    python attention.py MODEL_SNAPSHOT IDS.npy OUT_DIR --ctx N [--prefix IDS2.npy] METHOD...
    python attention.py --selftest
    python attention.py --table RESULTS.jsonl...
    python attention.py --peel-table RESULTS.jsonl...
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.llama import modeling_llama
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2 import modeling_qwen2

import exact_grid

FP32_EXACT = 2 ** 24
S32_EXACT = 2 ** 31 - 1
#: grid -> (top integer, the accumulator's exact range)
GRIDS = {"int8": (127, S32_EXACT), "e4m3": (16, FP32_EXACT)}
#: a method's grid token -> (Q K^T's grid, P V's grid)
GRID_PAIRS = {"int8": ("int8", "int8"), "e4m3": ("e4m3", "e4m3"), "e4m3qk": ("e4m3", "int8")}
ALPHABETS = {"none": [0], "pm1": [-1, 1], "pm13": [-3, -1, 1, 3], "pm57": [-7, -5, 5, 7], "pm1245": [-5, -4, -2, -1, 1, 2, 4, 5],
             **{f"w{k}": [v for v in range(-k, k + 1) if v] for k in (8, 64)},
             **{f"u{k}": list(range(-k, k + 1)) for k in (1, 2, 4, 8, 16, 32, 64)}}
OPERANDS = "qkpv"
PEEL_L = 32
LINS = {"int8_rot": "int8_rot@-", "int8_rot_dith63": "int8_rot@a:dither:wtop63:s{seed}"}
#: P's blocks: each block's s32 read is exact while B max|P| max|V| = B 128 128 < 2^31 (exact_mm chunks it in FP32)
PBLK_MAX = 2 ** 16
#: K or V served as floor(x / s) + b, b a fresh bit (note:20261016T0525Z-draft-fullfloor-s-r6)
DITHER = "dith"

#: the models whose eager attention is patched, and their own
EAGER = (modeling_llama, modeling_qwen2)
ORIG = {m.__name__: m.eager_attention_forward for m in EAGER}
#: a KV group's s x s work runs on all query rows at once while a (rep, s, s) float64 tensor is within ROWS_WHOLE bytes,
#: else on chunks of ROWS_BYTES (32K); the rows are independent, so the chunks give the same products (the selftest)
ROWS_WHOLE, ROWS_BYTES = 2 ** 32, 2 ** 31
#: bf16 attention is the model's own eager while its (heads, s, s) FP32 weights are within EAGER_BYTES, else row chunks
EAGER_BYTES = 2 ** 34
#: tokens per chunk of the head and loss beyond 8192 (the logits of 32K tokens would not fit beside the model)
LOSS_ROWS = 4096
STATE: dict = {"spec": None}


def is_alphabet(name: str) -> bool:
    return name in ALPHABETS or name == DITHER or (name[:1] == "d" and name[1:].isdigit())


def alphabet_info(name: str) -> dict:
    """Entropy (bits per entry), RMS and max magnitude (LSBs) of one entry's noise; for the dither, of its error against
    x / s (MSE 1/3)."""
    if name == DITHER:
        return {"bits": 1.0, "rms": math.sqrt(1 / 3), "max": 1}
    if name in ALPHABETS:
        a = ALPHABETS[name]
        return {"bits": math.log2(len(a)) if len(a) > 1 else 0.0, "rms": math.sqrt(sum(v * v for v in a) / len(a)),
                "max": max(abs(v) for v in a)}
    rho = 2.0 ** -int(name[1:])
    return {"bits": -(1 - rho) * math.log2(1 - rho) - rho * math.log2(rho / 2), "rms": math.sqrt(rho), "max": 1}


def parse(method: str) -> dict:
    spec = {"method": method, "grid": None, "p": "prow", "v": "vchan", "qk2": False, "rot": False, "seed": 0,
            "lin": "bf16", "e": {"q": "none", "p": "none"}, "poff": 0, "pnorm": False, "pblk": 0, "psink": False,
            "ptop": 0, "kc": False}
    default, where, over = "none", "both", {}
    for t in method.split("-"):
        ops, _, a = t.partition("=")
        if a and is_alphabet(a) and ops[:1] == "e" and ops[1:] and set(ops[1:]) <= set("qp"):
            spec["e"].update({x: a for x in ops[1:]})
            continue
        if a and is_alphabet(a) and ops and set(ops) <= set(OPERANDS):
            over.update({x: a for x in ops})
            continue
        if t in GRID_PAIRS or t in ("bf16", "ref", "sdpa"):
            spec["grid"] = t
        elif t in ("prow", "pfix", "prow2"):
            spec["p"] = t
        elif t in ("vhead", "vchan", "vchan2"):
            spec["v"] = t
        elif is_alphabet(t):
            default = t
        elif t == "rot":
            spec["rot"] = True
        elif t == "qk2":
            spec["qk2"] = True
        elif t in ("qk", "pv", "both"):
            where = t
        elif t == "lin8r":
            spec["lin"] = "int8_rot"
        elif t == "lin8rd":
            spec["lin"] = "int8_rot_dith63"
        elif t[:4] == "poff" and t[4:].isdigit():
            spec["poff"] = int(t[4:])
        elif t in ("pnorm", "psink", "kc"):
            spec[t] = True
        elif t[:4] in ("pblk", "ptop") and t[4:].isdigit():
            spec[t[:4]] = int(t[4:])
        elif t[0] == "s" and t[1:].isdigit():
            spec["seed"] = int(t[1:])
        else:
            raise ValueError(f"{method}: unknown token {t!r}")
    if spec["grid"] is None:
        raise ValueError(f"{method}: no grid")
    on = {"both": OPERANDS, "qk": "qk", "pv": "pv"}[where]
    spec["n"] = {x: over.get(x, default if x in on else "none") for x in OPERANDS}
    if spec["e"] != {"q": "none", "p": "none"} and (spec["grid"] not in GRID_PAIRS or spec["qk2"] or spec["p"] == "prow2"):
        raise ValueError(f"{method}: peeled noise needs a one-limb grid")
    pbase = spec["poff"] or spec["pnorm"] or spec["pblk"] or spec["psink"] or spec["ptop"]
    if pbase and (spec["grid"] not in GRID_PAIRS or GRID_PAIRS[spec["grid"]][1] != "int8" or spec["p"] != "prow"
                  or spec["n"]["p"] != "none"):
        raise ValueError(f"{method}: P's offset, top, blocks and row-sum norm need P on one int8 limb (prow), unserved")
    if spec["kc"] and spec["grid"] not in GRID_PAIRS:
        raise ValueError(f"{method}: K's centering is an edit of the served K")
    if spec["pblk"] and (spec["pblk"] % PEEL_L or spec["pblk"] > PBLK_MAX):
        raise ValueError(f"{method}: P's blocks are a multiple of {PEEL_L} keys, at most {PBLK_MAX}")
    dithered = [x for x in OPERANDS if spec["n"][x] == DITHER]
    if dithered and (set(dithered) - set("kv") or spec["grid"] != "int8" or spec["qk2"] or spec["v"] == "vchan2"):
        raise ValueError(f"{method}: the dither serves K or V on one int8 limb")
    return spec


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


def draw(alphabet: str, shape, gen: torch.Generator, device) -> torch.Tensor:
    if alphabet in ALPHABETS:
        table = torch.tensor(ALPHABETS[alphabet], dtype=torch.float32, device=device)
        return table[torch.randint(0, len(table), shape, generator=gen, device=device)]
    rho = 2.0 ** -int(alphabet[1:])
    u = torch.rand(shape, generator=gen, device=device)
    return (u < rho / 2).float() * -1 + ((u >= rho / 2) & (u < rho)).float()


def exact_mm(a: torch.Tensor, b: torch.Tensor, term: int) -> torch.Tensor:
    """a @ b for integer-valued FP32 a, b with |a_ik b_kj| <= term: FP32 GEMMs over k-chunks whose every partial sum is
    <= 2^24 (so exact in any order), summed in float64."""
    c = max(1, FP32_EXACT // term)
    k = a.shape[-1]
    if k <= c:
        return (a @ b).double()
    out = None
    for i in range(0, k, c):
        part = (a[..., i:i + c] @ b[..., i:i + c, :]).double()
        out = part if out is None else out + part
    return out


def floor_div(q: torch.Tensor, base: int) -> torch.Tensor:
    """floor(q / base) for integer-valued q, in integer arithmetic (a float quotient can round across an integer)."""
    return torch.div(q.to(torch.int32), base, rounding_mode="floor").to(q.dtype)


def edit_p(p: torch.Tensor, mode: str, alphabet: str, top: int, causal: torch.Tensor, gen, device):
    """P's int8 limbs with their weights, [(limb, weight)], and its scale; noise (only where key <= query) on the last
    limb.  `prow2` splits P = B P_hi + P_lo with B = top + 1, P_hi in 0..127 and P_lo in 0..B-1, so P's grid has top
    128 B - 1 and P_lo plus noise stays within int8."""
    if mode == "pfix":
        sp = torch.full_like(p[..., :1], 1.0 / top)
    else:
        sp = p.amax(dim=-1, keepdim=True).clamp(min=1e-30) / (128 * (top + 1) - 1 if mode == "prow2" else top)
    if mode == "prow2":
        pi = torch.round(p / sp).clamp(0, 128 * (top + 1) - 1)
        hi = floor_div(pi, top + 1)
        limbs = [(hi, top + 1), (pi - (top + 1) * hi, 1)]
    else:
        limbs = [(torch.round(p / sp).clamp(0, top), 1)]
    if alphabet != "none":
        last = limbs[-1][0] + draw(alphabet, p.shape, gen, device) * causal
        limbs[-1] = (last, 1)
    return limbs, sp


def signed_limbs(x: torch.Tensor, top: int, dims, two: bool) -> tuple[list, torch.Tensor]:
    """x's int8 limbs [(limb, weight)] and its scale (absmax over `dims`): one limb |q| <= top, or two, x ~ (B q_hi + q_lo) s
    with B = top + 1, q_hi in -128..127 and q_lo in 0..B-1 (so q_lo plus noise up to 127 - top stays within int8)."""
    if not two:
        q, s = quant(x, top, dims)
        return [(q, 1)], s
    base = top + 1
    q, s = quant(x, 128 * base - 1, dims)
    hi = floor_div(q, base)
    return [(hi, base), (q - base * hi, 1)], s


def in_int8(x: torch.Tensor) -> bool:
    return -128 <= float(x.min()) and float(x.max()) <= 127


def floor_quant(x: torch.Tensor, dims) -> tuple[torch.Tensor, torch.Tensor]:
    """floor(x / s) in [-128, 126] and s, the smallest scale (to 2^-20) that keeps x / s in [-128, 127): with a fresh bit
    added, the served X stays within int8."""
    s = torch.maximum(x.amax(dim=dims, keepdim=True) / 127, (-x).amax(dim=dims, keepdim=True) / 128)
    s = s.clamp(min=1e-30) * (1 + 2 ** -20)
    q = torch.floor(x / s)
    assert float(q.max()) <= 126 and float(q.min()) >= -128
    return q, s


def add_noise(limbs: list, alphabet: str, gen, device) -> list:
    """Noise on the last (lowest) limb."""
    if alphabet == "none":
        return limbs
    last, w = limbs[-1]
    if alphabet == DITHER:
        return limbs[:-1] + [(last + torch.randint(0, 2, last.shape, generator=gen, device=device).float(), w)]
    return limbs[:-1] + [(last + draw(alphabet, last.shape, gen, device), w)]


def limb_product(al: list, bl: list, term: int) -> torch.Tensor:
    """The exact integer (sum_a w_a A_a)(sum_b u_b B_b), one exact int8 product per pair of limbs, combined in float64."""
    return sum(w * u * exact_mm(a, b, term) for a, w in al for b, u in bl)


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


def p_edges(s: int, spec: dict) -> list[int]:
    """The key blocks of P's scales, as edges: one block, blocks of `pblk` keys, and key 0 alone under `psink`."""
    edges = set(range(0, s, spec["pblk"])) if spec["pblk"] else {0}
    if spec["psink"] and s > 1:
        edges.add(1)
    return sorted(edges) + [s]


def blocked_pv(p, vi, top: int, vmax: int, edges: list[int], term: int, poff: int, up, ep, causal,
               check_range: bool, check_peel: bool, norm: bool, r0: int = 0, checks: dict | None = None) -> torch.Tensor:
    """O's integer rows times P's scales for P on one int8 limb with one scale per (query row, key block): block b serves
    P_b = round(p_b / s_b) in 0..top (s_b its row's max over the block, over top), its product P_b V_b read exactly and
    scaled by s_b, summed over blocks; `norm` divides by sum_b s_b rowsum(P_b) instead.  p, ep and causal hold the query
    rows r0.. (all of them unless chunked); check_range: every served block P_b - poff + E_b is within int8; check_peel:
    each one's product, peeled and offset back, is P_b V_b (one verdict per block in `checks`, AND over row chunks)."""
    rep, s, d = vi.shape
    rows = p.shape[-2]
    num = torch.zeros(rep, rows, d, dtype=torch.float64, device=vi.device)
    den = torch.zeros(rep, rows, 1, dtype=torch.float64, device=vi.device)
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        sb = p[..., a:b].amax(-1, keepdim=True).clamp(min=1e-30) / top
        pi = torch.round(p[..., a:b] / sb).clamp(0, top)
        vb = vi[..., a:b, :]
        ob = exact_mm(pi, vb, top * vmax)
        if check_range or check_peel:
            pt = pi - poff * causal[:, a:b].float() if poff else pi
            pt = pt + ep[:, a:b] if ep is not None else pt
            assert in_int8(pt)
            if check_peel and (poff or up is not None):
                mask = torch.zeros(s, 1, device=vi.device)
                mask[a:b] = 1
                served = exact_mm(pt, vb, term)
                if up is not None:
                    served = served - causal_peel(up, vi * mask)[:, r0:r0 + rows]
                if poff:
                    served = served + poff * (vi * mask).double().cumsum(-2)[:, r0:r0 + rows]
                ok = bool(torch.equal(served, ob))
                if checks is None:
                    STATE["peel"].append(ok)
                else:
                    checks[i] = checks.get(i, True) and ok
        num += sb.double() * ob
        den += sb.double() * pi.double().sum(-1, keepdim=True)
    return num / den if norm else num


def row_chunk(rep: int, s: int) -> int:
    """Query rows per chunk of a KV group's s x s work: all of them while one (rep, s, s) float64 tensor is within
    ROWS_WHOLE bytes (every 2K and 8K run so far), else ROWS_BYTES' worth; STATE["rows"] overrides (the selftest)."""
    if STATE.get("rows"):
        return STATE["rows"]
    if rep * s * s * 8 <= ROWS_WHOLE:
        return s
    return max(1, ROWS_BYTES // (rep * s * 8))


def chunked_eager(module, query, key, value, attention_mask, scaling):
    """The model's own eager attention (repeat_kv, bf16 Q K^T times the scale plus the mask, FP32 softmax cast to bf16,
    bf16 P V) over chunks of query rows, for contexts whose (heads, s, s) weights don't fit at once."""
    assert attention_mask is not None, "eager's causal mask"
    rep = module.num_key_value_groups
    k = key[:, :, None].expand(-1, -1, rep, -1, -1).reshape(key.shape[0], -1, *key.shape[2:])
    v = value[:, :, None].expand(-1, -1, rep, -1, -1).reshape(value.shape[0], -1, *value.shape[2:])
    b, H, s, d = query.shape
    rows = STATE.get("rows") or max(1, ROWS_BYTES // (H * s * 4))
    out = torch.empty(b, s, H, d, dtype=query.dtype, device=query.device)
    for r0 in range(0, s, rows):
        w = torch.matmul(query[:, :, r0:r0 + rows], k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, r0:r0 + rows, :s]
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        out[:, r0:r0 + rows] = torch.matmul(w, v).transpose(1, 2)
        del w
    return out, None


def edited_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    spec = STATE["spec"]
    b, H, s, d = query.shape
    assert kwargs.get("sliding_window") is None or kwargs["sliding_window"] >= s, "full causal attention only"
    if spec is not None and spec["grid"] == "sdpa":
        assert b == 1, "one unpadded sequence"
        return sdpa_attention_forward(module, query, key, value, None, dropout=dropout, scaling=scaling, is_causal=True)
    if spec is None or spec["grid"] == "bf16":
        if H * s * s * 4 <= EAGER_BYTES and not STATE.get("rows"):
            return ORIG[type(module).__module__](module, query, key, value, attention_mask, scaling=scaling,
                                                 dropout=dropout, **kwargs)
        return chunked_eager(module, query, key, value, attention_mask, scaling)
    STATE["calls"] += 1
    G = key.shape[1]
    rep = H // G
    assert b == 1, "one unpadded sequence"
    dev, layer = query.device, module.layer_idx
    quantized = spec["grid"] in GRID_PAIRS
    first = STATE["window"] == 0
    noise_seed = (spec["seed"] + 1) * 1_000_003 + layer + STATE["window"] * 7_919_003
    gen = torch.Generator(device=dev).manual_seed(noise_seed)
    R = hadamard_rotation(d, 7919 * (spec["seed"] + 1) + layer, dev) if spec["rot"] else None
    rows = row_chunk(rep, s)
    if rows < s and quantized and spec["n"]["p"] != "none":
        raise ValueError(f"{spec['method']}: P's served noise is drawn over whole rows (no row chunks)")
    keys = torch.arange(s, device=dev)
    uq = up = None
    if quantized:
        n, e = spec["n"], spec["e"]
        gqk, gpv = (GRIDS[g] for g in GRID_PAIRS[spec["grid"]])
        nmax = {x: alphabet_info(n[x])["max"] for x in OPERANDS}
        emax = {x: alphabet_info(e.get(x, "none"))["max"] for x in OPERANDS}
        poff = spec["poff"]
        m = {x: (gqk if x in "qk" else gpv)[0] - nmax[x] - (0 if x == "p" and poff else emax[x]) for x in OPERANDS}
        if spec["ptop"]:
            m["p"] = spec["ptop"]
        if min(m.values()) < 1:
            raise ValueError(f"{spec['method']}: its noise does not fit the grid ({m})")
        two = {"q": spec["qk2"], "k": spec["qk2"], "p": spec["p"] == "prow2", "v": spec["v"] == "vchan2"}
        lmax = {x: (128 if x != "p" else 127) if two[x] else m[x] + nmax[x] + emax[x] for x in OPERANDS}
        lmax.update({x: 128 for x in OPERANDS if n[x] == DITHER})
        if poff or spec["ptop"]:
            lo, hi = -poff - nmax["p"] - emax["p"], m["p"] - poff + nmax["p"] + emax["p"]
            if lo < -128 or hi > 127:
                raise ValueError(f"{spec['method']}: P - {poff} + E spans [{lo}, {hi}], outside int8")
            lmax["p"] = max(-lo, hi)
        edges = p_edges(s, spec)
        STATE["record"] = {"tops": m, "acc_qk": gqk[1], "acc_pv": gpv[1], "bound_qk": d * lmax["q"] * lmax["k"],
                           "bound_pv": s * lmax["p"] * lmax["v"],
                           "int8_products": {"qk": (1 + two["q"]) * (1 + two["k"]), "pv": (1 + two["p"]) * (1 + two["v"])},
                           "p_blocks": len(edges) - 1, "row_chunk": rows}
        # U has its own stream, so the served noise is the same draw with or without a peel.
        ugen = torch.Generator(device=dev).manual_seed(noise_seed + 104_729 * 10 ** 9)
        if e["q"] != "none":
            assert d % PEEL_L == 0
            uq = draw(e["q"], (s, PEEL_L), ugen, dev)
            eq = uq[:, torch.arange(d, device=dev) % PEEL_L]
        if e["p"] != "none":
            up = draw(e["p"], (s, PEEL_L), ugen, dev)
        if uq is not None or up is not None:
            STATE["record"]["peel"] = {"L": PEEL_L, "qk_macs_over_product": PEEL_L / d if uq is not None else 0,
                                       "pv_macs_over_causal_product": 2 * PEEL_L / s if up is not None else 0}
    out = torch.empty(b, s, H, d, dtype=query.dtype, device=dev)
    for g in range(G):
        qg = query[0, g * rep:(g + 1) * rep].float()
        kg, vg = key[0, g].float(), value[0, g].float()
        ref = torch.empty(rep, s, d, device=dev) if quantized and first else None
        q0, k0, v0 = qg, kg, vg
        if R is not None:
            qg, kg, vg = qg @ R, kg @ R, vg @ R
        if quantized:
            if spec["kc"]:
                kg = kg - kg.mean(dim=0, keepdim=True)
            ql, sq = signed_limbs(qg, m["q"], -1, two["q"])
            if n["k"] == DITHER:
                kq, sk = floor_quant(kg, -1)
                kl = [(kq, 1)]
            else:
                kl, sk = signed_limbs(kg, m["k"], -1, two["k"])
            kl = [(x.expand(rep, s, d), w) for x, w in kl]
            ql, kl = add_noise(ql, n["q"], gen, dev), add_noise(kl, n["k"], gen, dev)
            vdims = (0, 1) if spec["v"] == "vhead" else 0
            if n["v"] == DITHER:
                vq, sv = floor_quant(vg, vdims)
                vl0 = [(vq, 1)]
            else:
                vl0, sv = signed_limbs(vg, m["v"], vdims, two["v"])
            # V's noise is drawn after P's (the generator's order with one chunk), so at the first chunk
            vl = None
            qk_ok = pv_ok = True
            blk_ok: dict = {}
        o = torch.empty(rep, s, d, device=dev)
        for r0 in range(0, s, rows):
            r1 = min(s, r0 + rows)
            causal = keys[None, :] <= torch.arange(r0, r1, device=dev)[:, None]
            if ref is not None:
                sc = (q0[:, r0:r1] @ k0.T) * scaling
                ref[:, r0:r1] = torch.softmax(sc.masked_fill(~causal, float("-inf")), dim=-1) @ v0
                del sc
            if quantized:
                if uq is not None:
                    [(qi, _)], [(ki, _)] = ql, kl
                    term = lmax["q"] * lmax["k"]
                    qe = qi[:, r0:r1] + eq[r0:r1]
                    served = exact_mm(qe, ki.transpose(1, 2), term)
                    sint = served - fold_peel(uq[r0:r1], ki).double()
                    del served
                    if first:
                        assert float(qe.abs().max()) <= 127 and in_int8(ki)
                        qk_ok = qk_ok and bool(torch.equal(sint, exact_mm(qi[:, r0:r1], ki.transpose(1, 2), term)))
                    sint = sint.float()
                else:
                    sint = limb_product([(x[:, r0:r1], w) for x, w in ql], [(x.transpose(1, 2), w) for x, w in kl],
                                        lmax["q"] * lmax["k"]).float()
                scores = sint * (sq[:, r0:r1] * sk.T[None]) * scaling
                del sint
            else:
                scores = (qg[:, r0:r1] @ kg.T) * scaling
            p = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            del scores
            if quantized:
                pl, sp = edit_p(p, spec["p"], n["p"], m["p"], causal, gen, dev)
                if vl is None:
                    vl = add_noise([(x.expand(rep, s, d), w) for x, w in vl0], n["v"], gen, dev)
                ep = up[r0:r1][:, keys % PEEL_L] * causal if up is not None else None
                if len(edges) > 2:
                    [(vi, _)] = vl
                    o[:, r0:r1] = (blocked_pv(p, vi, m["p"], lmax["v"], edges, lmax["p"] * lmax["v"], poff, up, ep, causal,
                                              first, first, spec["pnorm"], r0, blk_ok) * sv.double()).float()
                else:
                    if up is not None or poff:
                        [(pi, _)], [(vi, _)] = pl, vl
                        term = lmax["p"] * lmax["v"]
                        pt = pi - poff * causal.float() if poff else pi
                        pt = pt + ep if up is not None else pt
                        oint = exact_mm(pt, vi, term)
                        if up is not None:
                            oint = oint - causal_peel(up, vi)[:, r0:r1]
                        if poff:
                            oint = oint + poff * vi.double().cumsum(-2)[:, r0:r1]
                        if first:
                            assert in_int8(pt) and in_int8(vi)
                            pv_ok = pv_ok and bool(torch.equal(oint, exact_mm(pi, vi, m["p"] * lmax["v"])))
                        del pt
                    else:
                        oint = limb_product(pl, vl, lmax["p"] * lmax["v"])
                    if spec["pnorm"]:
                        [(pi, _)] = pl
                        o[:, r0:r1] = (oint / pi.double().sum(-1, keepdim=True) * sv.double()).float()
                    else:
                        o[:, r0:r1] = (oint * sp.double() * sv.double()).float()
                    del oint
                del pl, ep
            else:
                o[:, r0:r1] = p @ vg
            del p, causal
        if quantized and first:
            if uq is not None:
                STATE["peel"].append(qk_ok)
            if len(edges) <= 2 and (up is not None or poff):
                STATE["peel"].append(pv_ok)
            STATE["peel"].extend(blk_ok[i] for i in sorted(blk_ok))
        if R is not None:
            o = o @ R.T
        if ref is not None:
            STATE["err"][0] += float(((o - ref) ** 2).sum())
            STATE["err"][1] += float((ref ** 2).sum())
        out[0, :, g * rep:(g + 1) * rep] = o.transpose(0, 1).to(query.dtype)
    return out, None


for _m in EAGER:
    _m.eager_attention_forward = edited_attention


def evaluate(model, ids: torch.Tensor, method: str) -> dict:
    """Perplexity over every token of the windows ids (N x T, or one window T); the attention error and the peel's
    exactness (each peeled product against the direct one) on window 0."""
    spec = parse(method)
    ids = ids if ids.dim() == 2 else ids[None]
    STATE.update(spec=spec, calls=0, err=[0.0, 0.0], record={}, peel=[], window=0)
    log: list = []
    lin = LINS[spec["lin"]].format(seed=spec["seed"]) if spec["lin"] != "bf16" else None
    hooks = exact_grid.attach(model, lin, log) if lin else []
    t0 = time.time()
    nlls = []
    try:
        with torch.no_grad():
            for w, row in enumerate(ids):
                STATE["window"] = w
                if len(row) <= 8192:
                    logits = model(row[None], use_cache=False).logits[0]
                    nlls.append(torch.nn.functional.cross_entropy(logits[:-1].float(), row[1:].to(logits.device),
                                                                  reduction="none"))
                    del logits
                else:
                    h = model.model(row[None], use_cache=False).last_hidden_state[0]
                    tgt = row[1:].to(h.device)
                    nlls.append(torch.cat([torch.nn.functional.cross_entropy(
                        model.lm_head(h[i:min(i + LOSS_ROWS, len(tgt))]).float(), tgt[i:i + LOSS_ROWS], reduction="none")
                        for i in range(0, len(tgt), LOSS_ROWS)]))
                    del h
    finally:
        STATE["spec"] = None
        for h in hooks:
            h.remove()
    if spec["grid"] not in ("bf16", "sdpa"):
        assert STATE["calls"] == model.config.num_hidden_layers * len(ids), STATE["calls"]
    nll = torch.cat(nlls)
    rec = STATE["record"]
    res = {"method": method, **{k: v for k, v in spec.items() if k != "method"}, "ctx": int(ids.shape[1]),
           "windows": len(ids), "nll": float(nll.mean()), "ppl": float(torch.exp(nll.mean())),
           "ppl_windows": [float(torch.exp(x.mean())) for x in nlls], "seconds": round(time.time() - t0, 1),
           "noise_info": {x: alphabet_info(spec["n"][x]) for x in OPERANDS},
           "attn_rel_err": math.sqrt(STATE["err"][0] / STATE["err"][1]) if STATE["err"][1] else None, **rec}
    if rec:
        res["exact"] = rec["bound_qk"] <= rec["acc_qk"] and rec["bound_pv"] <= rec["acc_pv"]
    if STATE["peel"]:
        res["peel_exact"], res["peel_checks"] = all(STATE["peel"]), len(STATE["peel"])
    if log:
        res["lin_exact_all"] = all(r["exact"] for r in log)
        res["lin_max_bound_over_acc"] = max(r["bound"] / r["acc"] for r in log)
        res["lin_mean_rel_err"] = float(np.mean([r["rel_err"] for r in log]))
    return res


def load(snapshot: str):
    """The model on one GPU, or split by layers over every visible GPU (Llama-3.1-70B on two), each holding at most
    D4A_GPU_GIB of weights so the rest stays free for the edits' work."""
    n = torch.cuda.device_count()
    kw = {"device_map": "auto", "max_memory": {i: f"{os.environ.get('D4A_GPU_GIB', '68')}GiB" for i in range(n)}} if n > 1 else {}
    model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, attn_implementation="eager", **kw)
    return (model if n > 1 else model.cuda()).eval()


def selftest() -> dict:
    """Exact chunked products, the alphabets, causal P noise, the rotation's invariance, and every method on a tiny Llama."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        # Whole-model perplexities are compared exactly; one thread fixes the CPU GEMMs' reduction order across runs.
        torch.set_num_threads(1)
    out: dict = {"device": dev}
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-127, 128, (3, 64, 3000), generator=g).float().to(dev)
    bm = torch.randint(-127, 128, (3, 3000, 32), generator=g).float().to(dev)
    assert torch.equal(exact_mm(a, bm, 127 * 127), a.double() @ bm.double())
    gen = torch.Generator(device=dev).manual_seed(1)
    for name, vals in ALPHABETS.items():
        x = draw(name, (200_000,), gen, dev)
        assert set(x.unique().tolist()) == set(float(v) for v in vals), name
    x = draw("d4", (1_000_000,), gen, dev)
    assert set(x.unique().tolist()) == {-1.0, 0.0, 1.0} and abs(float((x != 0).float().mean()) - 1 / 16) < 2e-3
    assert abs(float(x.mean())) < 2e-3
    assert abs(alphabet_info("u16")["bits"] - math.log2(33)) < 1e-12 and alphabet_info("pm13")["rms"] == math.sqrt(5)
    assert abs(alphabet_info("d1")["bits"] - 1.5) < 1e-12 and alphabet_info("none")["bits"] == 0.0
    assert parse("int8-u4-qk")["n"] == {"q": "u4", "k": "u4", "p": "none", "v": "none"}
    assert parse("int8-qk=u4-p=d8-v=pm1")["n"] == {"q": "u4", "k": "u4", "p": "d8", "v": "pm1"}
    pe = parse("int8-rot-lin8r-eqp=w8-kv=pm1-s2")
    assert pe["n"] == {"q": "none", "k": "pm1", "p": "none", "v": "pm1"} and pe["e"] == {"q": "w8", "p": "w8"}
    assert pe["lin"] == "int8_rot" and pe["seed"] == 2 and alphabet_info("w8")["bits"] == 4.0
    for bad in ("int8-prow2-ep=w8", "int8-qk2-eq=w8", "bf16-eq=w8", "int8-ek=w8"):
        try:
            parse(bad)
            raise AssertionError(f"{bad} must not parse")
        except ValueError:
            pass
    for s_, d_ in ((70, 32), (64, 64), (5, 32)):
        u = draw("w8", (s_, PEEL_L), gen, dev)
        vv = torch.randint(-127, 128, (3, s_, d_), generator=g).float().to(dev)
        mask = torch.ones(s_, s_, dtype=torch.bool, device=dev).tril()
        dense = (u[:, torch.arange(s_, device=dev) % PEEL_L] * mask).double() @ vv.double()
        assert torch.equal(causal_peel(u, vv), dense), (s_, d_)
        kk = torch.randint(-127, 128, (3, s_ + 3, 128), generator=g).float().to(dev)
        uq = draw("w8", (s_, PEEL_L), gen, dev)
        dense = uq[:, torch.arange(128, device=dev) % PEEL_L].double() @ kk.double().transpose(1, 2)
        assert torch.equal(fold_peel(uq, kk).double(), dense), s_
    causal = torch.ones(50, 50, dtype=torch.bool, device=dev).tril()
    p = torch.softmax(torch.randn(2, 50, 50, device=dev).masked_fill(~causal, float("-inf")), -1)
    [(pi, _)], _ = edit_p(p, "prow", "u16", 111, causal, gen, dev)
    assert bool((pi.triu(1) == 0).all()) and bool((pi.tril() != torch.round(p / (p.amax(-1, keepdim=True) / 111))).any())
    [(hi, base), (lo, _)], sp = edit_p(p, "prow2", "none", 111, causal, gen, dev)
    assert base == 112 and float(hi.max()) == 127 and 0 <= float(lo.min()) and float(lo.max()) <= 111
    assert torch.allclose((base * hi + lo) * sp, p, atol=float(sp.max()) / 2 + 1e-12)
    [(hi, base), (lo, _)], _ = edit_p(p, "prow2", "u16", 111, causal, gen, dev)
    assert float(lo.min()) >= -16 and float(lo.max()) <= 127 and bool((lo.triu(1) == 0).all())
    v = torch.randn(300, 32, device=dev)
    [(hi, base), (lo, _)], sv = signed_limbs(v, 111, 0, True)
    assert base == 112 and float(hi.min()) >= -128 and float(hi.max()) <= 127 and 0 <= float(lo.min()) <= float(lo.max()) <= 111
    assert float((base * hi + lo).abs().max()) == 128 * 112 - 1 and torch.allclose((base * hi + lo) * sv, v, atol=float(sv.max()))
    noisy = add_noise([(hi, base), (lo, 1)], "u16", gen, dev)
    assert torch.equal(noisy[0][0], hi) and -16 <= float(noisy[1][0].min()) and float(noisy[1][0].max()) <= 127
    L = transformers.models.llama.modeling_llama
    torch.manual_seed(0)
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8,
                        num_key_value_heads=2, head_dim=32, max_position_embeddings=1024, tie_word_embeddings=False,
                        attn_implementation="eager")
    model = L.LlamaForCausalLM(cfg).to(torch.bfloat16).to(dev).eval()
    ids = torch.randint(0, 512, (1, 96), device=dev)
    runs = {m: evaluate(model, ids, m) for m in ("bf16", "sdpa", "ref", "ref-rot", "int8-none", "int8-pfix-vhead-none",
                                                 "int8-u16", "int8-rot-u16", "e4m3-none", "e4m3-pm57-rot", "int8-u4-qk",
                                                 "int8-u4-pv-s2", "e4m3qk-qk=u4-v=u2-p=d6-rot", "int8-p=d2",
                                                 "int8-prow2-none", "int8-prow2-p=u16-rot", "int8-prow2-vchan2-qk2-none",
                                                 "int8-prow2-vchan2-qk2-u16-rot", "int8-vchan2-v=u8")}
    out["tiny"] = {m: {k: r[k] for k in ("ppl", "attn_rel_err", "exact") if k in r} for m, r in runs.items()}
    assert abs(runs["ref"]["ppl"] / runs["bf16"]["ppl"] - 1) < 1e-2, out
    assert abs(runs["sdpa"]["ppl"] / runs["ref"]["ppl"] - 1) < 1e-2, out
    assert abs(runs["ref-rot"]["ppl"] / runs["ref"]["ppl"] - 1) < 1e-3, out
    assert runs["int8-none"]["attn_rel_err"] < runs["int8-u16"]["attn_rel_err"], out
    assert runs["int8-none"]["attn_rel_err"] < runs["e4m3-none"]["attn_rel_err"], out
    assert runs["e4m3qk-qk=u4-v=u2-p=d6-rot"]["tops"] == {"q": 12, "k": 12, "p": 126, "v": 125}
    assert runs["int8-prow2-none"]["attn_rel_err"] <= runs["int8-none"]["attn_rel_err"], out
    assert runs["int8-prow2-p=u16-rot"]["exact"] and runs["int8-prow2-p=u16-rot"]["int8_products"] == {"qk": 1, "pv": 2}
    full = runs["int8-prow2-vchan2-qk2-u16-rot"]
    assert full["exact"] and full["int8_products"] == {"qk": 4, "pv": 4}, full
    assert full["attn_rel_err"] < runs["int8-u16"]["attn_rel_err"] / 10, out
    assert runs["int8-prow2-vchan2-qk2-none"]["attn_rel_err"] < runs["int8-none"]["attn_rel_err"] / 10, out
    try:
        evaluate(model, ids, "e4m3-u16")
        raise AssertionError("e4m3-u16 must not fit")
    except ValueError:
        pass
    peel = {m: evaluate(model, ids, m) for m in ("bf16-lin8r", "int8-rot-lin8r", "int8-rot-eqp=w8", "int8-rot-eqp=w8-kv=pm1",
                                                 "int8-rot-eqp=u8-kv=pm1", "int8-rot-eq=w8-k=pm1", "int8-rot-ep=w8-v=pm1")}
    out["tiny_peel"] = {m: {k: r.get(k) for k in ("ppl", "attn_rel_err", "peel_exact", "peel_checks", "lin_exact_all")}
                        for m, r in peel.items()}
    assert peel["bf16-lin8r"]["lin_exact_all"] and abs(peel["bf16-lin8r"]["ppl"] / runs["bf16"]["ppl"] - 1) < 5e-2, out
    assert peel["int8-rot-eqp=w8"]["tops"] == {"q": 119, "k": 127, "p": 119, "v": 127}, out
    assert peel["int8-rot-eqp=w8-kv=pm1"]["tops"] == {"q": 119, "k": 126, "p": 119, "v": 126}, out
    for m in ("int8-rot-eqp=w8", "int8-rot-eqp=w8-kv=pm1", "int8-rot-eqp=u8-kv=pm1", "int8-rot-eq=w8-k=pm1",
              "int8-rot-ep=w8-v=pm1"):
        assert peel[m]["peel_exact"] and peel[m]["exact"], (m, out)
    assert peel["int8-rot-eqp=w8-kv=pm1"]["peel_checks"] == 2 * 2 * 2 and peel["int8-rot-eq=w8-k=pm1"]["peel_checks"] == 4
    # The CPU's bf16 GEMMs can round with the buffers' alignment, which earlier tests' allocations move, so on the CPU
    # equal perplexities agree to 1e-5; on the GPU `exactly` is bit equality
    same = lambda a, b: abs(a / b - 1) < (1e-5 if dev == "cpu" else 1e-6)  # noqa: E731

    def exactly(a, b) -> bool:
        if dev != "cpu":
            return a == b
        return all(same(x, y) for x, y in zip(a, b)) if isinstance(a, list) else same(a, b)
    w8, u8 = peel["int8-rot-eqp=w8-kv=pm1"]["ppl"], peel["int8-rot-eqp=u8-kv=pm1"]["ppl"]
    assert same(w8, u8), out
    ids2 = torch.cat([ids, torch.randint(0, 512, (1, 96), device=dev)])
    two_w = evaluate(model, ids2, "int8-rot-lin8r-eqp=w8-v=pm1")
    one_w = evaluate(model, ids2[0], "int8-rot-lin8r-eqp=w8-v=pm1")
    assert two_w["windows"] == 2 and same(two_w["ppl_windows"][0], one_w["ppl"]) and two_w["peel_exact"], (two_w, one_w)
    assert not same(two_w["ppl_windows"][1], evaluate(model, ids2[1], "int8-rot-lin8r-eqp=w8-v=pm1")["ppl"])
    out["tiny_windows"] = two_w["ppl_windows"]
    assert alphabet_info("pm1245")["bits"] == 3.0 and alphabet_info("pm1245")["max"] == 5 and alphabet_info("w64")["bits"] == 7.0
    wide = {m: evaluate(model, ids, m) for m in ("int8-rot-eqp=w64-poff64", "int8-rot-eq=w64", "int8-rot-eqp=w8-kv=pm1245",
                                                 "int8-rot-eqp=w64-poff64-kv=pm1245")}
    out["tiny_wide"] = {m: {k: r.get(k) for k in ("ppl", "attn_rel_err", "peel_exact", "peel_checks")} for m, r in wide.items()}
    tops = {m: r["tops"] for m, r in wide.items()}
    assert tops["int8-rot-eqp=w64-poff64"] == {"q": 63, "k": 127, "p": 127, "v": 127}, tops
    assert tops["int8-rot-eqp=w8-kv=pm1245"] == {"q": 119, "k": 122, "p": 119, "v": 122}, tops
    assert tops["int8-rot-eqp=w64-poff64-kv=pm1245"] == {"q": 63, "k": 122, "p": 127, "v": 122}, tops
    for m, r in wide.items():
        assert r["exact"] and (r["peel_exact"] and r["peel_checks"] >= 4), (m, out["tiny_wide"])
    assert wide["int8-rot-eqp=w64-poff64-kv=pm1245"]["bound_pv"] == 96 * 128 * 127
    off, plain = wide["int8-rot-eqp=w64-poff64"]["ppl"], wide["int8-rot-eq=w64"]["ppl"]
    assert same(off, plain), out["tiny_wide"]
    for bad, where in (("int8-rot-ep=w64-poff32", evaluate), ("int8-prow2-poff64", parse), ("int8-ptop255", evaluate),
                       ("int8-rot-eqp=w64-ptop255-poff128", evaluate), ("int8-pblk48", parse), ("int8-p=pm1-pnorm", parse),
                       ("e4m3-psink", parse), ("int8-pfix-pblk32", parse), ("int8-pblk131072", parse)):
        try:
            where(model, ids, bad) if where is evaluate else where(bad)
            raise AssertionError(f"{bad} must not fit")
        except ValueError:
            pass
    out["bit_identical"] = {"w8_vs_u8": w8 == u8, "window0": two_w["ppl_windows"][0] == one_w["ppl"],
                            "poff64_vs_unpeeled_p": off == plain}
    xs = torch.randn(4, 3000, 64, generator=g).to(dev) * torch.rand(1, 1, 64, generator=g).to(dev)
    xs[0, :, 0] = xs[0, :, 0].abs()
    fq, fs = floor_quant(xs, 1)
    y = xs / fs
    assert float(fq.min()) >= -128 and float(fq.max()) <= 126 and bool(((y - fq >= 0) & (y - fq < 1)).all())
    assert bool(((y.amax(1) > 126.99) | (y.amin(1) < -127.99)).all())
    served = fq + torch.randint(0, 2, fq.shape, generator=torch.Generator(device=dev).manual_seed(5), device=dev).float()
    err = served - y
    assert abs(float(err.mean())) < 5e-3 and abs(float((err ** 2).mean()) - 1 / 3) < 5e-3, (float(err.mean()), float((err ** 2).mean()))
    for bad in ("int8-q=dith", "int8-dith", "int8-qk2-k=dith", "e4m3-v=dith"):
        try:
            parse(bad)
            raise AssertionError(f"{bad} must not parse")
        except ValueError:
            pass
    dith = {m: evaluate(model, ids, m) for m in ("int8-rot-eqp=w64-poff64-kv=dith", "int8-rot-eqp=w64-poff64-v=dith",
                                                 "int8-rot-eqp=w64-poff64-k=dith", "int8-rot-eqp=w64-poff64-kv=pm1",
                                                 "int8-rot-kv=dith")}
    out["tiny_dither"] = {m: {k: r.get(k) for k in ("ppl", "attn_rel_err", "peel_exact", "peel_checks")} for m, r in dith.items()}
    kv = dith["int8-rot-eqp=w64-poff64-kv=dith"]
    assert kv["tops"] == {"q": 63, "k": 126, "p": 127, "v": 126} and kv["bound_pv"] == 96 * 128 * 128, kv
    for m, r in dith.items():
        assert r["exact"] and (r.get("peel_exact", True) and (r.get("peel_checks", 0) >= 4 or "eqp" not in m)), (m, out["tiny_dither"])
    assert kv["attn_rel_err"] < dith["int8-rot-eqp=w64-poff64-kv=pm1"]["attn_rel_err"], out["tiny_dither"]
    causal = torch.ones(50, 50, dtype=torch.bool, device=dev).tril()
    p = torch.softmax((3 * torch.randn(2, 50, 50, device=dev)).masked_fill(~causal, float("-inf")), -1)
    vv = torch.randint(-127, 128, (2, 50, 16), generator=g).float().to(dev)
    [(pi, _)], sp = edit_p(p, "prow", "none", 127, causal, gen, dev)
    one = blocked_pv(p, vv, 127, 127, [0, 50], 127 * 127, 0, None, None, causal, True, True, False)
    assert torch.equal(one, exact_mm(pi, vv, 127 * 127) * sp.double())
    ones = torch.ones(2, 50, 4, device=dev)
    for edges in ([0, 50], [0, 1, 50], [0, 32, 50], [0, 1, 32, 50]):
        assert torch.equal(blocked_pv(p, ones, 127, 1, edges, 127, 0, None, None, causal, True, True, True),
                           torch.ones(2, 50, 4, dtype=torch.float64, device=dev)), edges
    eye = torch.eye(50, device=dev).expand(2, 50, 50)
    err = {str(e_): float((blocked_pv(p, eye, 127, 1, e_, 127, 0, None, None, causal, False, False, False) - p.double()).abs().max())
           for e_ in ([0, 50], [0, 32, 50])}
    assert err["[0, 32, 50]"] <= err["[0, 50]"], err
    assert p_edges(96, parse("int8-pblk32-psink")) == [0, 1, 32, 64, 96] and p_edges(96, parse("int8-psink")) == [0, 1, 96]
    assert p_edges(5, parse("int8-pblk32")) == [0, 5] and parse("int8-rot-lin8rd-s2")["lin"] == "int8_rot_dith63"
    pb = {m: evaluate(model, ids, m) for m in (
        "int8-rot", "int8-rot-pnorm", "int8-rot-psink", "int8-rot-pblk32", "int8-rot-pblk32-psink-pnorm",
        "int8-rot-ptop255-poff128", "int8-rot-eqp=w64-poff64-pblk32-kv=dith", "int8-rot-eq=w64-pblk32-kv=dith",
        "int8-rot-eqp=w64-poff64-pnorm-kv=dith", "int8-rot-eq=w64-pnorm-kv=dith",
        "int8-rot-eqp=w64-poff64-psink-pnorm-kv=dith", "int8-rot-eq=w64-psink-pnorm-kv=dith",
        "bf16-lin8rd", "bf16-lin8rd-s1", "bf16-lin8r", "int8-rot-lin8rd-eqp=w64-poff64-kv=dith-s1")}
    out["tiny_pbase"] = {m: {k: r.get(k) for k in ("ppl", "attn_rel_err", "peel_exact", "peel_checks", "p_blocks", "lin_exact_all")}
                         for m, r in pb.items()}
    for m, r in pb.items():
        assert r.get("exact", True) and r.get("peel_exact", True) and r.get("lin_exact_all", True), (m, out["tiny_pbase"])
    assert [pb[m]["p_blocks"] for m in ("int8-rot", "int8-rot-psink", "int8-rot-pblk32", "int8-rot-pblk32-psink-pnorm")] == [1, 2, 3, 4]
    assert pb["int8-rot-pblk32"]["attn_rel_err"] < pb["int8-rot"]["attn_rel_err"], out["tiny_pbase"]
    top = pb["int8-rot-ptop255-poff128"]
    assert top["tops"]["p"] == 255 and top["bound_pv"] == 96 * 128 * 127 and top["peel_checks"] == 4, top
    assert top["attn_rel_err"] < pb["int8-rot"]["attn_rel_err"], out["tiny_pbase"]
    # Q K^T's peel per (layer, KV group), and each of P's 3 blocks' on every layer too
    assert pb["int8-rot-eqp=w64-poff64-pblk32-kv=dith"]["peel_checks"] == 2 * 2 + 2 * 2 * 3, out["tiny_pbase"]
    for a_, b_ in (("eqp=w64-poff64-pblk32", "eq=w64-pblk32"), ("eqp=w64-poff64-pnorm", "eq=w64-pnorm"),
                   ("eqp=w64-poff64-psink-pnorm", "eq=w64-psink-pnorm")):
        assert exactly(pb[f"int8-rot-{a_}-kv=dith"]["ppl"], pb[f"int8-rot-{b_}-kv=dith"]["ppl"]), (a_, out["tiny_pbase"])
    assert len({pb[m]["ppl"] for m in ("bf16-lin8rd", "bf16-lin8rd-s1", "bf16-lin8r")}) == 3, out["tiny_pbase"]
    assert pb["int8-rot-lin8rd-eqp=w64-poff64-kv=dith-s1"]["lin_exact_all"], out["tiny_pbase"]
    chunk_methods = ("bf16", "int8-rot-lin8r", "int8-rot-psink", "int8-rot-eq=w8-k=pm1", "int8-rot-eqp=w64-poff64-kv=dith",
                     "int8-rot-eqp=w64-poff64-psink-kv=dith", "int8-rot-eqp=w64-poff64-pblk32-kv=dith",
                     "int8-rot-ptop255-poff128", "int8-prow2-vchan2-qk2-none-rot", "int8-rot-pnorm",
                     "int8-rot-lin8rd-eqp=w64-poff64-psink-kv=dith-s1")
    whole = {m: evaluate(model, ids2, m) for m in chunk_methods}
    STATE["rows"] = 20
    try:
        chunked = {m: evaluate(model, ids2, m) for m in chunk_methods}
        try:
            evaluate(model, ids, "int8-p=u4")
            raise AssertionError("P's served noise must not run in row chunks")
        except ValueError:
            pass
    finally:
        STATE["rows"] = 0
    out["tiny_chunks"] = {m: [whole[m]["ppl"], chunked[m]["ppl"], chunked[m].get("peel_checks")] for m in chunk_methods}
    assert abs(chunked["bf16"]["ppl"] / whole["bf16"]["ppl"] - 1) < 1e-3, out["tiny_chunks"]
    for m in chunk_methods[1:]:
        w_, c_ = whole[m], chunked[m]
        assert c_["row_chunk"] == 20 and w_["row_chunk"] == 96, (m, c_["row_chunk"], w_["row_chunk"])
        assert exactly(c_["ppl_windows"], w_["ppl_windows"]) and c_.get("peel_checks") == w_.get("peel_checks"), (m, out["tiny_chunks"])
        assert c_.get("peel_exact", True) and c_["exact"], (m, out["tiny_chunks"])
    assert chunked["int8-rot-eqp=w64-poff64-psink-kv=dith"]["peel_checks"] == 2 * 2 * 3
    exact_grid.WCACHE = False
    try:
        nocache = {m: evaluate(model, ids, m)["ppl"] for m in ("bf16-lin8rd-s1", "int8-rot-lin8r")}
    finally:
        exact_grid.WCACHE = True
    cached = {m: evaluate(model, ids, m)["ppl"] for m in nocache}
    assert all(exactly(nocache[m], cached[m]) for m in nocache), (nocache, cached)
    out["bit_identical"].update(chunks=all(chunked[m]["ppl_windows"] == whole[m]["ppl_windows"] for m in chunk_methods[1:]),
                                wcache_off=nocache == cached)
    Q2 = modeling_qwen2
    torch.manual_seed(1)
    qcfg = Q2.Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8,
                          num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=False,
                          use_sliding_window=False, attn_implementation="eager")
    qm = Q2.Qwen2ForCausalLM(qcfg).to(torch.bfloat16).to(dev).eval()
    with torch.no_grad():
        for mod in qm.modules():
            if isinstance(mod, torch.nn.Linear) and mod.bias is not None:
                mod.bias.normal_(0, 1)
        plain = qm(ids, use_cache=False).logits.float()
        hooks = exact_grid.attach(qm, "int8_rot@-", [])
        try:
            hooked = qm(ids, use_cache=False).logits.float()
        finally:
            for h in hooks:
                h.remove()
    lin_err = float(torch.linalg.norm(hooked - plain) / torch.linalg.norm(plain))
    qw = {m: evaluate(qm, ids, m) for m in ("bf16", "sdpa", "ref", "int8-rot-lin8r", "int8-rot-k=dith", "int8-rot-kc-k=dith",
                                            "int8-rot-lin8rd-eqp=w64-poff64-psink-kv=dith-s1",
                                            "int8-rot-lin8rd-eqp=w64-poff64-psink-kc-kv=dith-s1")}
    out["tiny_qwen2"] = {"lin_rel_err": lin_err, **{m: {k: r.get(k) for k in ("ppl", "attn_rel_err", "peel_exact", "peel_checks")}
                                                    for m, r in qw.items()}}
    assert lin_err < 0.05, out["tiny_qwen2"]
    assert abs(qw["ref"]["ppl"] / qw["bf16"]["ppl"] - 1) < 1e-2, out["tiny_qwen2"]
    assert abs(qw["sdpa"]["ppl"] / qw["ref"]["ppl"] - 1) < 1e-2, out["tiny_qwen2"]
    e2e = qw["int8-rot-lin8rd-eqp=w64-poff64-psink-kv=dith-s1"]
    assert e2e["exact"] and e2e["peel_exact"] and e2e["peel_checks"] == 2 * 2 * 3 and e2e["lin_exact_all"], out["tiny_qwen2"]
    assert qw["int8-rot-lin8r"]["attn_rel_err"] < 0.1, out["tiny_qwen2"]
    assert qw["int8-rot-kc-k=dith"]["attn_rel_err"] < qw["int8-rot-k=dith"]["attn_rel_err"], out["tiny_qwen2"]
    kc = qw["int8-rot-lin8rd-eqp=w64-poff64-psink-kc-kv=dith-s1"]
    assert kc["exact"] and kc["peel_exact"] and kc["peel_checks"] == 2 * 2 * 3, out["tiny_qwen2"]
    return out


def operand_noise(r: dict) -> dict:
    """Per-operand alphabets of a record (records from before per-operand noise carry `noise` and `where`)."""
    if "n" in r:
        return r["n"]
    on = {"both": OPERANDS, "qk": "qk", "pv": "pv"}[r["where"]]
    return {x: r["noise"] if x in on else "none" for x in OPERANDS}


def _key(r: dict, clean: bool = False, bare: bool = False) -> tuple:
    """`clean`: the same edit without served noise; `bare`: without peeled noise too."""
    n = {x: "none" for x in OPERANDS} if clean or bare else operand_noise(r)
    e = {"q": "none", "p": "none"} if bare else r.get("e", {"q": "none", "p": "none"})
    return (r["grid"], r["p"], r["v"], r.get("qk2", False), r["rot"], n["q"], n["k"], n["p"], n["v"],
            r.get("lin", "bf16"), e["q"], e["p"], 0 if bare else r.get("poff", 0), r["ctx"], "" if bare else p_base(r))


def p_base(r: dict) -> str:
    """P's base options of a record: blocks, sink, norm, top (and an offset not tied to a peel)."""
    parts = [f"blk{r['pblk']}" if r.get("pblk") else "", "sink" if r.get("psink") else "",
             "norm" if r.get("pnorm") else "", f"top{r['ptop']}" if r.get("ptop") else "",
             f"off{r['poff']}" if r.get("poff") and r.get("e", {}).get("p", "none") == "none" else ""]
    return " ".join(x for x in parts if x)


def noise_label(n: dict) -> str:
    if len(set(n.values())) == 1:
        return n["q"]
    parts = []
    for ops in ("qk", "p", "v") if n["q"] == n["k"] else ("q", "k", "p", "v"):
        if n[ops[0]] != "none":
            parts.append(f"{ops.upper()}:{n[ops[0]]}")
    return ", ".join(parts)


def rows(results: list[dict]) -> list[dict]:
    """One row per (edit, context): perplexity (mean over seeds, with the range), and its change against the same edit
    without noise (mean over the seeds it ran at) and against bf16 at the same context."""
    groups: dict = {}
    seen = set()
    for r in results:
        if (_key(r), r["seed"]) not in seen:
            seen.add((_key(r), r["seed"]))
            groups.setdefault(_key(r), []).append(r)
    mean = lambda rs: sum(x["ppl"] for x in rs) / len(rs)  # noqa: E731
    bf16 = {(k[13], k[9]): mean(rs) for k, rs in groups.items() if k[0] == "bf16"}
    out = []
    for key, rs in groups.items():
        r, n = rs[0], operand_noise(rs[0])
        ppls = [x["ppl"] for x in rs]
        ppl = sum(ppls) / len(ppls)
        base, bare = groups.get(_key(r, clean=True)), groups.get(_key(r, bare=True))
        e = r.get("e", {"q": "none", "p": "none"})
        info = {x: alphabet_info(n[x]) for x in OPERANDS}
        lin = r.get("lin", "bf16")
        out.append({"ctx": r["ctx"], "grid": r["grid"], "p": r["p"], "v": r["v"], "qk2": r.get("qk2", False), "rot": r["rot"], "n": n,
                    "products": r.get("int8_products"), "lin": lin, "e": e, "poff": r.get("poff", 0), "tops": r.get("tops"),
                    "pbase": p_base(r),
                    "windows": r.get("windows", 1), "peel_exact": [x.get("peel_exact") for x in rs],
                    "label": noise_label(n), "info": info, "seeds": len(rs), "ppl": ppl, "min": min(ppls), "max": max(ppls),
                    "d_grid": 100 * (ppl / mean(base) - 1) if base and any(v != "none" for v in n.values()) else None,
                    "d_bare": 100 * (ppl / mean(bare) - 1) if bare and key != _key(r, bare=True) else None,
                    "d_bf16": 100 * (ppl / bf16[r["ctx"], "bf16"] - 1) if (r["ctx"], "bf16") in bf16 else None,
                    "d_lin": 100 * (ppl / bf16[r["ctx"], lin] - 1) if lin != "bf16" and (r["ctx"], lin) in bf16 else None,
                    "err": r.get("attn_rel_err")})
    out.sort(key=lambda x: (x["ctx"], x["grid"] not in GRID_PAIRS, x["grid"], x["lin"], x["p"], x["v"], x["qk2"], x["rot"],
                            tuple(x["e"].values()) != ("none", "none"),
                            tuple(x["info"][o]["bits"] for o in "qpv"), tuple(x["info"][o]["rms"] for o in "qpv")))
    return out


def peel_table(results: list[dict]) -> str:
    """The peel measurement: perplexity (mean over seeds, with the range) against bf16, against bf16 attention with the
    same linears, against the same grid with no noise (`bare`: the one-limb baseline at top 127), and against the same
    edit without served noise (`clean`: the shrunk grid, its peeled noise kept)."""
    lines = ["| ctx | linears | attention | tops Q, K, P, V | peeled (U) | served (F) | windows x seeds | perplexity | "
             "Δ% vs bf16 | Δ% vs bf16 attn | Δ% vs bare grid | Δ% vs clean | peel exact | attn rel. err |",
             "|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|"]
    fmt = lambda d: f"{d:+.2f}" if d is not None else "-"  # noqa: E731
    served_order = lambda x: ["qkpv", "kv", "v", "k"].index(  # noqa: E731
        "".join(o for o in OPERANDS if x["n"][o] != "none") or "qkpv") if x["label"] != "none" else 0
    order = lambda x: (x["ctx"], x["grid"] in GRID_PAIRS, x["lin"] != "bf16",  # noqa: E731
                       max(alphabet_info(a)["max"] for a in x["e"].values()), x["poff"],
                       max(x["info"][o]["max"] for o in OPERANDS), served_order(x))
    for x in sorted(rows(results), key=order):
        q = x["grid"] in GRID_PAIRS
        attn = (f"{x['grid']}{'-rot' if x['rot'] else ''} {x['p']}/{x['v']}" if q else x["grid"])
        attn += f" P {x['pbase']}" if x["pbase"] else ""
        tops = ", ".join(str(x["tops"][o]) for o in OPERANDS) if x["tops"] else "-"
        peeled = ", ".join(f"{o.upper()}: {a}" for o, a in x["e"].items() if a != "none") or "-"
        peeled += f"; P offset -{x['poff']}" if x["poff"] else ""
        served = x["label"] if any(v != "none" for v in x["n"].values()) else "-"
        pe = x["peel_exact"]
        exact = "-" if all(v is None for v in pe) else "yes" if all(pe) else "NO"
        spread = f" ({x['min']:.3f}-{x['max']:.3f})" if x["seeds"] > 1 else ""
        err = f"{x['err']:.4f}" if x["err"] is not None else "-"
        lines.append(f"| {x['ctx']} | {x['lin']} | {attn} | {tops} | {peeled} | {served} | {x['windows']} x {x['seeds']} | "
                     f"{x['ppl']:.3f}{spread} | {fmt(x['d_bf16'])} | {fmt(x['d_lin'])} | {fmt(x['d_bare'])} | "
                     f"{fmt(x['d_grid'])} | {exact} | {err} |")
    return "\n".join(lines)


def table(results: list[dict]) -> str:
    """Marks: ** within 1%, * within 10% (of the grid baseline; the bf16 column has its own)."""
    lines = ["| ctx | grid | P | V | rot | noise (operand: alphabet) | rms / max, LSB (Q,K; P; V) | bits/entry (Q,K; P; V) "
             "| seeds | perplexity | Δ% vs grid | Δ% vs bf16 | mark (grid / bf16) | attn rel. err |",
             "|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|"]
    for x in rows(results):
        i = x["info"]
        q = x["grid"] in GRID_PAIRS
        mag = "; ".join(f"{i[o]['rms']:.2f} / {i[o]['max']}" for o in "qpv")
        bits = "; ".join(f"{i[o]['bits']:.3f}" for o in "qpv")
        mark = lambda d: "" if d is None else ("**" if d < 1 else "*" if d < 10 else "-")  # noqa: E731
        spread = f" ({x['min']:.3f}-{x['max']:.3f})" if x["seeds"] > 1 else ""
        dg = f"{x['d_grid']:+.2f}" if x["d_grid"] is not None else "-"
        db = f"{x['d_bf16']:+.2f}" if x["d_bf16"] is not None else "-"
        err = f"{x['err']:.4f}" if x["err"] is not None else "-"
        lines.append(f"| {x['ctx']} | {x['grid']}{' qk2' if x['qk2'] else ''} | {x['p'] if q else '-'} | {x['v'] if q else '-'} | "
                     f"{'yes' if x['rot'] else 'no'} | {x['label']} | {mag} | {bits} | {x['seeds']} | {x['ppl']:.3f}{spread} | "
                     f"{dg} | {db} | {mark(x['d_grid'])} / {mark(x['d_bf16'])} | {err} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if argv[0] == "--selftest":
        print(json.dumps(selftest(), indent=1))
        return 0
    if argv[0] in ("--table", "--peel-table"):
        results = [json.loads(line) for p in argv[1:] for line in Path(p).read_text().splitlines() if line.startswith("{")]
        print((table if argv[0] == "--table" else peel_table)(results))
        return 0
    snapshot, ids_path, out = argv[0], argv[1], Path(argv[2])
    rest = argv[3:]
    assert rest[0] == "--ctx"
    ctx, rest = int(rest[1]), rest[2:]
    prefix = None
    if rest and rest[0] == "--prefix":
        prefix, rest = np.load(rest[1]), rest[2:]
    out.mkdir(parents=True, exist_ok=True)
    ids = np.load(ids_path)
    ids = ids[None] if ids.ndim == 1 else ids
    assert ids.shape[1] >= ctx, (ids.shape, ctx)
    ids = ids[:, :ctx]
    if prefix is not None:
        info = {"ctx": ctx, "windows": len(ids), "prefix_len": len(prefix),
                "prefix_matches": bool(np.array_equal(ids[0, :len(prefix)], prefix[:ctx]))}
        (out / f"ids_{ctx}.json").write_text(json.dumps(info))
        print(json.dumps(info), file=sys.stderr, flush=True)
    ids_t = torch.from_numpy(ids).cuda()
    model = load(snapshot)
    for method in rest:
        res = evaluate(model, ids_t, method)
        print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
