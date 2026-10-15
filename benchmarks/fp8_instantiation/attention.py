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
               uniform {-k..k}, `dE` sparse (+-1 each with probability 2^-E / 2, else 0); `qk`/`pv` limits it to that
               matmul's operands; `X=ALPHABET` sets operand X (`q`, `k`, `p`, `v`, `qk`, `pv`) alone
    `rot`, `sN` (noise seed)

`bf16` is the model's own eager attention and `ref` this file's attention in FP32 with no edit (`ref-rot` checks the
rotation).

    python attention.py MODEL_SNAPSHOT IDS.npy OUT_DIR --ctx N [--prefix IDS2.npy] METHOD...
    python attention.py --selftest
    python attention.py --table RESULTS.jsonl...
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

FP32_EXACT = 2 ** 24
S32_EXACT = 2 ** 31 - 1
#: grid -> (top integer, the accumulator's exact range)
GRIDS = {"int8": (127, S32_EXACT), "e4m3": (16, FP32_EXACT)}
#: a method's grid token -> (Q K^T's grid, P V's grid)
GRID_PAIRS = {"int8": ("int8", "int8"), "e4m3": ("e4m3", "e4m3"), "e4m3qk": ("e4m3", "int8")}
ALPHABETS = {"none": [0], "pm1": [-1, 1], "pm13": [-3, -1, 1, 3], "pm57": [-7, -5, 5, 7],
             **{f"u{k}": list(range(-k, k + 1)) for k in (1, 2, 4, 8, 16)}}
OPERANDS = "qkpv"

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
    spec = {"method": method, "grid": None, "p": "prow", "v": "vchan", "qk2": False, "rot": False, "seed": 0}
    default, where, over = "none", "both", {}
    for t in method.split("-"):
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
        elif "=" in t and t.split("=")[0] in ("q", "k", "p", "v", "qk", "pv") and is_alphabet(t.split("=")[1]):
            ops, a = t.split("=")
            over.update({x: a for x in ops})
        elif t[0] == "s" and t[1:].isdigit():
            spec["seed"] = int(t[1:])
        else:
            raise ValueError(f"{method}: unknown token {t!r}")
    if spec["grid"] is None:
        raise ValueError(f"{method}: no grid")
    on = {"both": OPERANDS, "qk": "qk", "pv": "pv"}[where]
    spec["n"] = {x: over.get(x, default if x in on else "none") for x in OPERANDS}
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
    gen = torch.Generator(device=dev).manual_seed((spec["seed"] + 1) * 1_000_003 + layer)
    R = hadamard_rotation(d, 7919 * (spec["seed"] + 1) + layer, dev) if spec["rot"] else None
    causal = torch.ones(s, s, dtype=torch.bool, device=dev).tril()
    if quantized:
        n = spec["n"]
        gqk, gpv = (GRIDS[g] for g in GRID_PAIRS[spec["grid"]])
        nmax = {x: alphabet_info(n[x])["max"] for x in OPERANDS}
        m = {x: (gqk if x in "qk" else gpv)[0] - nmax[x] for x in OPERANDS}
        if min(m.values()) < 1:
            raise ValueError(f"{spec['method']}: its noise does not fit the grid ({m})")
        two = {"q": spec["qk2"], "k": spec["qk2"], "p": spec["p"] == "prow2", "v": spec["v"] == "vchan2"}
        lmax = {x: (128 if x != "p" else 127) if two[x] else m[x] + nmax[x] for x in OPERANDS}
        STATE["record"] = {"tops": m, "acc_qk": gqk[1], "acc_pv": gpv[1], "bound_qk": d * lmax["q"] * lmax["k"],
                           "bound_pv": s * lmax["p"] * lmax["v"],
                           "int8_products": {"qk": (1 + two["q"]) * (1 + two["k"]), "pv": (1 + two["p"]) * (1 + two["v"])}}
    out = torch.empty(b, s, H, d, dtype=query.dtype, device=dev)
    for g in range(G):
        qg = query[0, g * rep:(g + 1) * rep].float()
        kg, vg = key[0, g].float(), value[0, g].float()
        ref = None
        if quantized:
            sc = (qg @ kg.T) * scaling
            ref = torch.softmax(sc.masked_fill(~causal, float("-inf")), dim=-1) @ vg
        if R is not None:
            qg, kg, vg = qg @ R, kg @ R, vg @ R
        if quantized:
            ql, sq = signed_limbs(qg, m["q"], -1, two["q"])
            kl, sk = signed_limbs(kg, m["k"], -1, two["k"])
            kl = [(x.expand(rep, s, d), w) for x, w in kl]
            ql, kl = add_noise(ql, n["q"], gen, dev), add_noise(kl, n["k"], gen, dev)
            sint = limb_product(ql, [(x.transpose(1, 2), w) for x, w in kl], lmax["q"] * lmax["k"]).float()
            scores = sint * (sq * sk.T[None]) * scaling
        else:
            scores = (qg @ kg.T) * scaling
        p = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        if quantized:
            pl, sp = edit_p(p, spec["p"], n["p"], m["p"], causal, gen, dev)
            vl, sv = signed_limbs(vg, m["v"], (0, 1) if spec["v"] == "vhead" else 0, two["v"])
            vl = add_noise([(x.expand(rep, s, d), w) for x, w in vl], n["v"], gen, dev)
            o = (limb_product(pl, vl, lmax["p"] * lmax["v"]) * sp.double() * sv.double()).float()
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
    spec = parse(method)
    STATE.update(spec=spec, calls=0, err=[0.0, 0.0], record={})
    t0 = time.time()
    try:
        with torch.no_grad():
            logits = model(ids, use_cache=False).logits[0]
    finally:
        STATE["spec"] = None
    if spec["grid"] != "bf16":
        assert STATE["calls"] == model.config.num_hidden_layers, STATE["calls"]
    nll = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[0, 1:], reduction="none")
    del logits
    rec = STATE["record"]
    res = {"method": method, **{k: v for k, v in spec.items() if k != "method"}, "ctx": int(ids.shape[1]),
           "nll": float(nll.mean()), "ppl": float(torch.exp(nll.mean())), "seconds": round(time.time() - t0, 1),
           "noise_info": {x: alphabet_info(spec["n"][x]) for x in OPERANDS},
           "attn_rel_err": math.sqrt(STATE["err"][0] / STATE["err"][1]) if STATE["err"][1] else None, **rec}
    if rec:
        res["exact"] = rec["bound_qk"] <= rec["acc_qk"] and rec["bound_pv"] <= rec["acc_pv"]
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
    return out


def operand_noise(r: dict) -> dict:
    """Per-operand alphabets of a record (records from before per-operand noise carry `noise` and `where`)."""
    if "n" in r:
        return r["n"]
    on = {"both": OPERANDS, "qk": "qk", "pv": "pv"}[r["where"]]
    return {x: r["noise"] if x in on else "none" for x in OPERANDS}


def _key(r: dict, clean: bool = False) -> tuple:
    n = {x: "none" for x in OPERANDS} if clean else operand_noise(r)
    return (r["grid"], r["p"], r["v"], r.get("qk2", False), r["rot"], n["q"], n["k"], n["p"], n["v"], r["ctx"])


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
    without noise (seed 0) and against bf16 at the same context."""
    groups: dict = {}
    seen = set()
    for r in results:
        if (_key(r), r["seed"]) not in seen:
            seen.add((_key(r), r["seed"]))
            groups.setdefault(_key(r), []).append(r)
    bf16 = {r["ctx"]: r["ppl"] for r in results if r["grid"] == "bf16"}
    out = []
    for key, rs in groups.items():
        r, n = rs[0], operand_noise(rs[0])
        ppls = [x["ppl"] for x in rs]
        ppl = sum(ppls) / len(ppls)
        base = groups.get(_key(r, clean=True))
        info = {x: alphabet_info(n[x]) for x in OPERANDS}
        out.append({"ctx": r["ctx"], "grid": r["grid"], "p": r["p"], "v": r["v"], "qk2": r.get("qk2", False), "rot": r["rot"], "n": n,
                    "products": r.get("int8_products"),
                    "label": noise_label(n), "info": info, "seeds": len(rs), "ppl": ppl, "min": min(ppls), "max": max(ppls),
                    "d_grid": 100 * (ppl / base[0]["ppl"] - 1) if base and any(v != "none" for v in n.values()) else None,
                    "d_bf16": 100 * (ppl / bf16[r["ctx"]] - 1) if r["ctx"] in bf16 else None,
                    "err": r.get("attn_rel_err")})
    out.sort(key=lambda x: (x["ctx"], x["grid"] not in GRID_PAIRS, x["grid"], x["p"], x["v"], x["qk2"], x["rot"],
                            tuple(x["info"][o]["bits"] for o in "qpv"), tuple(x["info"][o]["rms"] for o in "qpv")))
    return out


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
    if argv[0] == "--table":
        print(table([json.loads(line) for p in argv[1:] for line in Path(p).read_text().splitlines() if line.startswith("{")]))
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
    assert len(ids) >= ctx, (len(ids), ctx)
    ids = ids[:ctx]
    if prefix is not None:
        info = {"ctx": ctx, "prefix_len": len(prefix), "prefix_matches": bool(np.array_equal(ids[:len(prefix)], prefix))}
        (out / f"ids_{ctx}.json").write_text(json.dumps(info))
        print(json.dumps(info), file=sys.stderr, flush=True)
    ids_t = torch.from_numpy(ids)[None].cuda()
    model = load(snapshot)
    for method in rest:
        res = evaluate(model, ids_t, method)
        print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
