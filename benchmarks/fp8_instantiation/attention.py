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
               uniform {-k..k}, `w8` {+-1..+-8}, `dE` sparse (+-1 each with probability 2^-E / 2, else 0); `qk`/`pv`
               limits it to that matmul's operands; `X=ALPHABET` sets the operands X (any of `q`, `k`, `p`, `v`) alone
    peeled     `eX=ALPHABET` (X in `q`, `p`, `qp`): low-rank noise the server sees and the model doesn't
               (note:20261016T0001Z-draft-rs-d2-attention-lowrank-peel).  E_ik = U_{i, k mod L}, L = 32, with U (tokens x L)
               one draw per layer shared by its heads; P's E only where key <= query.  The served products are
               (P + E_P)(V + F_V) and (Q + E_Q)(K + F_K)^T, and the exact peels U (W^T (V + F_V)) (the causal prefix scan)
               and U (W^T (K + F_K))^T are subtracted, so the model gets P (V + F_V) and Q (K + F_K)^T exactly; the
               peeled operand's top drops by max|E|.  One limb only (`prow`/`pfix`, no `qk2`)
    linears    `lin8r`: every linear on exact_grid.py's `int8_rot@-` (rotated, int8 per row, s8 x s8 -> s32)
    `rot`, `sN` (noise seed)

`bf16` is the model's own eager attention and `ref` this file's attention in FP32 with no edit (`ref-rot` checks the
rotation).  IDS is one window (T,) or several (N, T) (`windows.py`); perplexity is over every token of every window, and
window w draws its own noise (window 0's is the single-window runs').

    python attention.py MODEL_SNAPSHOT IDS.npy OUT_DIR --ctx N [--prefix IDS2.npy] METHOD...
    python attention.py --selftest
    python attention.py --table RESULTS.jsonl...
    python attention.py --peel-table RESULTS.jsonl...
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.llama import modeling_llama

import exact_grid

FP32_EXACT = 2 ** 24
S32_EXACT = 2 ** 31 - 1
#: grid -> (top integer, the accumulator's exact range)
GRIDS = {"int8": (127, S32_EXACT), "e4m3": (16, FP32_EXACT)}
#: a method's grid token -> (Q K^T's grid, P V's grid)
GRID_PAIRS = {"int8": ("int8", "int8"), "e4m3": ("e4m3", "e4m3"), "e4m3qk": ("e4m3", "int8")}
ALPHABETS = {"none": [0], "pm1": [-1, 1], "pm13": [-3, -1, 1, 3], "pm57": [-7, -5, 5, 7],
             "w8": [v for v in range(-8, 9) if v],
             **{f"u{k}": list(range(-k, k + 1)) for k in (1, 2, 4, 8, 16, 32, 64)}}
OPERANDS = "qkpv"
PEEL_L = 32
LINS = {"int8_rot": "int8_rot@-"}

ORIG = modeling_llama.eager_attention_forward
STATE: dict = {"spec": None}


def is_alphabet(name: str) -> bool:
    return name in ALPHABETS or (name[:1] == "d" and name[1:].isdigit())


def alphabet_info(name: str) -> dict:
    """Entropy (bits per entry), RMS and max magnitude (LSBs) of one entry's noise."""
    if name in ALPHABETS:
        a = ALPHABETS[name]
        return {"bits": math.log2(len(a)) if len(a) > 1 else 0.0, "rms": math.sqrt(sum(v * v for v in a) / len(a)),
                "max": max(abs(v) for v in a)}
    rho = 2.0 ** -int(name[1:])
    return {"bits": -(1 - rho) * math.log2(1 - rho) - rho * math.log2(rho / 2), "rms": math.sqrt(rho), "max": 1}


def parse(method: str) -> dict:
    spec = {"method": method, "grid": None, "p": "prow", "v": "vchan", "qk2": False, "rot": False, "seed": 0,
            "lin": "bf16", "e": {"q": "none", "p": "none"}}
    default, where, over = "none", "both", {}
    for t in method.split("-"):
        ops, _, a = t.partition("=")
        if a and is_alphabet(a) and ops[:1] == "e" and ops[1:] and set(ops[1:]) <= set("qp"):
            spec["e"].update({x: a for x in ops[1:]})
            continue
        if a and is_alphabet(a) and ops and set(ops) <= set(OPERANDS):
            over.update({x: a for x in ops})
            continue
        if t in GRID_PAIRS or t in ("bf16", "ref"):
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


def add_noise(limbs: list, alphabet: str, gen, device) -> list:
    """Noise on the last (lowest) limb."""
    if alphabet == "none":
        return limbs
    last, w = limbs[-1]
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


def edited_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    spec = STATE["spec"]
    if spec is None or spec["grid"] == "bf16":
        return ORIG(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    STATE["calls"] += 1
    b, H, s, d = query.shape
    G = key.shape[1]
    rep = H // G
    assert b == 1, "one unpadded sequence"
    dev, layer = query.device, module.layer_idx
    quantized = spec["grid"] in GRID_PAIRS
    first = STATE["window"] == 0
    noise_seed = (spec["seed"] + 1) * 1_000_003 + layer + STATE["window"] * 7_919_003
    gen = torch.Generator(device=dev).manual_seed(noise_seed)
    R = hadamard_rotation(d, 7919 * (spec["seed"] + 1) + layer, dev) if spec["rot"] else None
    causal = torch.ones(s, s, dtype=torch.bool, device=dev).tril()
    uq = up = None
    if quantized:
        n, e = spec["n"], spec["e"]
        gqk, gpv = (GRIDS[g] for g in GRID_PAIRS[spec["grid"]])
        nmax = {x: alphabet_info(n[x])["max"] for x in OPERANDS}
        emax = {x: alphabet_info(e.get(x, "none"))["max"] for x in OPERANDS}
        m = {x: (gqk if x in "qk" else gpv)[0] - nmax[x] - emax[x] for x in OPERANDS}
        if min(m.values()) < 1:
            raise ValueError(f"{spec['method']}: its noise does not fit the grid ({m})")
        two = {"q": spec["qk2"], "k": spec["qk2"], "p": spec["p"] == "prow2", "v": spec["v"] == "vchan2"}
        lmax = {x: (128 if x != "p" else 127) if two[x] else m[x] + nmax[x] + emax[x] for x in OPERANDS}
        STATE["record"] = {"tops": m, "acc_qk": gqk[1], "acc_pv": gpv[1], "bound_qk": d * lmax["q"] * lmax["k"],
                           "bound_pv": s * lmax["p"] * lmax["v"],
                           "int8_products": {"qk": (1 + two["q"]) * (1 + two["k"]), "pv": (1 + two["p"]) * (1 + two["v"])}}
        # U has its own stream, so the served noise is the same draw with or without a peel.
        ugen = torch.Generator(device=dev).manual_seed(noise_seed + 104_729 * 10 ** 9)
        if e["q"] != "none":
            assert d % PEEL_L == 0
            uq = draw(e["q"], (s, PEEL_L), ugen, dev)
            eq = uq[:, torch.arange(d, device=dev) % PEEL_L]
        if e["p"] != "none":
            up = draw(e["p"], (s, PEEL_L), ugen, dev)
            ep = up[:, torch.arange(s, device=dev) % PEEL_L] * causal
        if uq is not None or up is not None:
            STATE["record"]["peel"] = {"L": PEEL_L, "qk_macs_over_product": PEEL_L / d if uq is not None else 0,
                                       "pv_macs_over_causal_product": 2 * PEEL_L / s if up is not None else 0}
    out = torch.empty(b, s, H, d, dtype=query.dtype, device=dev)
    for g in range(G):
        qg = query[0, g * rep:(g + 1) * rep].float()
        kg, vg = key[0, g].float(), value[0, g].float()
        ref = None
        if quantized and first:
            sc = (qg @ kg.T) * scaling
            ref = torch.softmax(sc.masked_fill(~causal, float("-inf")), dim=-1) @ vg
        if R is not None:
            qg, kg, vg = qg @ R, kg @ R, vg @ R
        if quantized:
            ql, sq = signed_limbs(qg, m["q"], -1, two["q"])
            kl, sk = signed_limbs(kg, m["k"], -1, two["k"])
            kl = [(x.expand(rep, s, d), w) for x, w in kl]
            ql, kl = add_noise(ql, n["q"], gen, dev), add_noise(kl, n["k"], gen, dev)
            if uq is not None:
                [(qi, _)], [(ki, _)] = ql, kl
                served = exact_mm(qi + eq, ki.transpose(1, 2), 127 * 127)
                sint = served - fold_peel(uq, ki).double()
                if first:
                    assert float((qi + eq).abs().max()) <= 127 and float(ki.abs().max()) <= 127
                    STATE["peel"].append(bool(torch.equal(sint, exact_mm(qi, ki.transpose(1, 2), 127 * 127))))
                sint = sint.float()
            else:
                sint = limb_product(ql, [(x.transpose(1, 2), w) for x, w in kl], lmax["q"] * lmax["k"]).float()
            scores = sint * (sq * sk.T[None]) * scaling
        else:
            scores = (qg @ kg.T) * scaling
        p = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        if quantized:
            pl, sp = edit_p(p, spec["p"], n["p"], m["p"], causal, gen, dev)
            vl, sv = signed_limbs(vg, m["v"], (0, 1) if spec["v"] == "vhead" else 0, two["v"])
            vl = add_noise([(x.expand(rep, s, d), w) for x, w in vl], n["v"], gen, dev)
            if up is not None:
                [(pi, _)], [(vi, _)] = pl, vl
                served = exact_mm(pi + ep, vi, 127 * 127)
                oint = served - causal_peel(up, vi)
                if first:
                    assert float((pi + ep).abs().max()) <= 127 and float(vi.abs().max()) <= 127
                    STATE["peel"].append(bool(torch.equal(oint, exact_mm(pi, vi, 127 * 127))))
            else:
                oint = limb_product(pl, vl, lmax["p"] * lmax["v"])
            o = (oint * sp.double() * sv.double()).float()
        else:
            o = p @ vg
        if R is not None:
            o = o @ R.T
        if ref is not None:
            STATE["err"][0] += float(((o - ref) ** 2).sum())
            STATE["err"][1] += float((ref ** 2).sum())
        out[0, :, g * rep:(g + 1) * rep] = o.transpose(0, 1).to(query.dtype)
    return out, None


modeling_llama.eager_attention_forward = edited_attention


def evaluate(model, ids: torch.Tensor, method: str) -> dict:
    """Perplexity over every token of the windows ids (N x T, or one window T); the attention error and the peel's
    exactness (each peeled product against the direct one) on window 0."""
    spec = parse(method)
    ids = ids if ids.dim() == 2 else ids[None]
    STATE.update(spec=spec, calls=0, err=[0.0, 0.0], record={}, peel=[], window=0)
    log: list = []
    hooks = exact_grid.attach(model, LINS[spec["lin"]], log) if spec["lin"] != "bf16" else []
    t0 = time.time()
    nlls = []
    try:
        with torch.no_grad():
            for w, row in enumerate(ids):
                STATE["window"] = w
                logits = model(row[None], use_cache=False).logits[0]
                nlls.append(torch.nn.functional.cross_entropy(logits[:-1].float(), row[1:], reduction="none"))
                del logits
    finally:
        STATE["spec"] = None
        for h in hooks:
            h.remove()
    if spec["grid"] != "bf16":
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
    model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, attn_implementation="eager")
    return model.cuda().eval()


def selftest() -> dict:
    """Exact chunked products, the alphabets, causal P noise, the rotation's invariance, and every method on a tiny Llama."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
    runs = {m: evaluate(model, ids, m) for m in ("bf16", "ref", "ref-rot", "int8-none", "int8-pfix-vhead-none",
                                                 "int8-u16", "int8-rot-u16", "e4m3-none", "e4m3-pm57-rot", "int8-u4-qk",
                                                 "int8-u4-pv-s2", "e4m3qk-qk=u4-v=u2-p=d6-rot", "int8-p=d2",
                                                 "int8-prow2-none", "int8-prow2-p=u16-rot", "int8-prow2-vchan2-qk2-none",
                                                 "int8-prow2-vchan2-qk2-u16-rot", "int8-vchan2-v=u8")}
    out["tiny"] = {m: {k: r[k] for k in ("ppl", "attn_rel_err", "exact") if k in r} for m, r in runs.items()}
    assert abs(runs["ref"]["ppl"] / runs["bf16"]["ppl"] - 1) < 1e-2, out
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
    same = lambda a, b: abs(a / b - 1) < 1e-6  # noqa: E731
    w8, u8 = peel["int8-rot-eqp=w8-kv=pm1"]["ppl"], peel["int8-rot-eqp=u8-kv=pm1"]["ppl"]
    assert same(w8, u8), out
    ids2 = torch.cat([ids, torch.randint(0, 512, (1, 96), device=dev)])
    two_w = evaluate(model, ids2, "int8-rot-lin8r-eqp=w8-v=pm1")
    one_w = evaluate(model, ids2[0], "int8-rot-lin8r-eqp=w8-v=pm1")
    assert two_w["windows"] == 2 and same(two_w["ppl_windows"][0], one_w["ppl"]) and two_w["peel_exact"], (two_w, one_w)
    assert not same(two_w["ppl_windows"][1], evaluate(model, ids2[1], "int8-rot-lin8r-eqp=w8-v=pm1")["ppl"])
    out["tiny_windows"] = two_w["ppl_windows"]
    out["bit_identical"] = {"w8_vs_u8": w8 == u8, "window0": two_w["ppl_windows"][0] == one_w["ppl"]}
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
            r.get("lin", "bf16"), e["q"], e["p"], r["ctx"])


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
    bf16 = {(r["ctx"], r.get("lin", "bf16")): r["ppl"] for r in results if r["grid"] == "bf16"}
    mean = lambda rs: sum(x["ppl"] for x in rs) / len(rs)  # noqa: E731
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
                    "products": r.get("int8_products"), "lin": lin, "e": e, "tops": r.get("tops"),
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
    order = lambda x: (x["ctx"], x["grid"] in GRID_PAIRS, x["lin"] != "bf16", x["e"] != {"q": "none", "p": "none"},  # noqa: E731
                       served_order(x))
    for x in sorted(rows(results), key=order):
        q = x["grid"] in GRID_PAIRS
        attn = (f"{x['grid']}{'-rot' if x['rot'] else ''} {x['p']}/{x['v']}" if q else x["grid"])
        tops = ", ".join(str(x["tops"][o]) for o in OPERANDS) if x["tops"] else "-"
        peeled = ", ".join(f"{o.upper()}: {a}" for o, a in x["e"].items() if a != "none") or "-"
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
