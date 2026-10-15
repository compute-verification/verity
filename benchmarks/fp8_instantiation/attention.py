"""The error side of the security-against-error frontier for attention: Llama-3.1-8B's WikiText-2 perplexity with every
linear in bf16 and both attention matmuls of every head and layer, S = Q K^T and O = P V, served from edited operands.

An edit puts each operand on an exact integer grid, adds fresh unpeeled noise in units of the grid's LSB, and serves the
exact integer product times the operands' scales.  The scales factor out of each k-sum, so the integer product is what
the tensor core returns (IMMA s8 x s8 -> s32 for int8; QMMA e4m3 with FP32 accumulation for E4M3's integers, whose every
partial sum stays below 2^24 here):

    Q, K       one scale per (head, token): the row of head_dim entries the k-sum of S runs over
    P          after the softmax, one scale per query row (`prow`), or the fixed 1/M (`pfix`)
    V          one scale per KV head (`vhead`) or per (KV head, channel) (`vchan`); the k-sum of O runs over tokens

Noise is drawn per instance: a fresh draw for every entry of every query head's Q, K, P and V in every layer (K and V are
shared by GQA's four query heads; each head's product gets its own noise).  P's noise lands only on the causal support
(key <= query), so no query reads a later token.  An operand that carries noise is quantized to the top M = top - max|n|,
so its noised integers stay on the grid (|x| <= 127 for int8, <= 16 for E4M3's integers).  `rot` multiplies Q, K and V by
one random Hadamard R of the head dimension (random signs times Sylvester's H_128) before quantizing and the output by R^T
after, which preserves S = (Q R)(K R)^T and O = P V exactly before the edit.

Method names are '-'-joined tokens: a grid (`int8`, `e4m3`), `prow`/`pfix`, `vhead`/`vchan`, an alphabet (`none`,
`pm1` {+-1}, `pm13` {+-1, +-3}, `pm57` {+-5, +-7}, `u1` ... `u16` uniform {-k..k}), optional `rot`, optional `qk`/`pv`
(noise on that matmul's operands only), optional `sN` (noise seed).  `bf16` is the model's own eager attention and `ref`
this file's attention in FP32 with no edit (`ref-rot` checks the rotation).

    python attention.py MODEL_SNAPSHOT IDS.npy OUT_DIR --ctx N METHOD...
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
ALPHABETS = {"none": [0], "pm1": [-1, 1], "pm13": [-3, -1, 1, 3], "pm57": [-7, -5, 5, 7],
             **{f"u{k}": list(range(-k, k + 1)) for k in (1, 2, 4, 8, 16)}}
#: WikiText-2 (raw) test split at the revision cached on node 2 (`benchmarks/pouw/pearl_c_vllm/e2e.py`)
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet")

ORIG = modeling_llama.eager_attention_forward
STATE: dict = {"spec": None}


def entropy_bits(alphabet: str) -> float:
    return math.log2(len(ALPHABETS[alphabet]))


def rms(alphabet: str) -> float:
    a = ALPHABETS[alphabet]
    return math.sqrt(sum(v * v for v in a) / len(a))


def parse(method: str) -> dict:
    spec = {"method": method, "grid": None, "p": "prow", "v": "vchan", "noise": "none", "rot": False, "where": "both",
            "seed": 0}
    for t in method.split("-"):
        if t in GRIDS or t in ("bf16", "ref"):
            spec["grid"] = t
        elif t in ("prow", "pfix"):
            spec["p"] = t
        elif t in ("vhead", "vchan"):
            spec["v"] = t
        elif t in ALPHABETS:
            spec["noise"] = t
        elif t == "rot":
            spec["rot"] = True
        elif t in ("qk", "pv", "both"):
            spec["where"] = t
        elif t[0] == "s" and t[1:].isdigit():
            spec["seed"] = int(t[1:])
        else:
            raise ValueError(f"{method}: unknown token {t!r}")
    if spec["grid"] is None:
        raise ValueError(f"{method}: no grid")
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
    table = torch.tensor(ALPHABETS[alphabet], dtype=torch.float32, device=device)
    return table[torch.randint(0, len(table), shape, generator=gen, device=device)]


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


def edit_p(p: torch.Tensor, spec: dict, top: int, causal: torch.Tensor, noisy: bool, gen, device):
    """P's integers (noise only where key <= query) and its scale."""
    if spec["p"] == "pfix":
        sp = torch.full_like(p[..., :1], 1.0 / top)
    else:
        sp = p.amax(dim=-1, keepdim=True).clamp(min=1e-30) / top
    pi = torch.round(p / sp).clamp(0, top)
    if noisy:
        pi = pi + draw(spec["noise"], pi.shape, gen, device) * causal
    return pi, sp


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
    quantized = spec["grid"] in GRIDS
    gen = torch.Generator(device=dev).manual_seed((spec["seed"] + 1) * 1_000_003 + layer)
    R = hadamard_rotation(d, 7919 * (spec["seed"] + 1) + layer, dev) if spec["rot"] else None
    causal = torch.ones(s, s, dtype=torch.bool, device=dev).tril()
    if quantized:
        top, acc = GRIDS[spec["grid"]]
        nmax = max(abs(v) for v in ALPHABETS[spec["noise"]])
        nqk = spec["noise"] != "none" and spec["where"] in ("both", "qk")
        npv = spec["noise"] != "none" and spec["where"] in ("both", "pv")
        m_qk, m_pv = top - (nmax if nqk else 0), top - (nmax if npv else 0)
        if min(m_qk, m_pv) < 1:
            raise ValueError(f"{spec['method']}: noise {spec['noise']} does not fit the {spec['grid']} grid")
        STATE["record"] = {"top_qk": m_qk, "top_pv": m_pv, "acc": acc,
                           "bound_qk": d * (m_qk + nmax * nqk) ** 2, "bound_pv": s * (m_pv + nmax * npv) ** 2}
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
            qi, sq = quant(qg, m_qk, -1)
            ki, sk = quant(kg, m_qk, -1)
            ki = ki.expand(rep, s, d)
            if nqk:
                qi = qi + draw(spec["noise"], qi.shape, gen, dev)
                ki = ki + draw(spec["noise"], ki.shape, gen, dev)
            scores = (qi @ ki.transpose(1, 2)) * (sq * sk.T[None]) * scaling
        else:
            scores = (qg @ kg.T) * scaling
        p = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        if quantized:
            pi, sp = edit_p(p, spec, m_pv, causal, npv, gen, dev)
            vi, sv = quant(vg, m_pv, (0, 1) if spec["v"] == "vhead" else 0)
            vi = vi.expand(rep, s, d)
            if npv:
                vi = vi + draw(spec["noise"], vi.shape, gen, dev)
            o = (exact_mm(pi, vi, (m_pv + nmax * npv) ** 2) * sp.double() * sv.double()).float()
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
           "entropy_bits": entropy_bits(spec["noise"]) if spec["noise"] != "none" else 0.0,
           "noise_rms_lsb": rms(spec["noise"]), "noise_max_lsb": max(abs(v) for v in ALPHABETS[spec["noise"]]),
           "attn_rel_err": math.sqrt(STATE["err"][0] / STATE["err"][1]) if STATE["err"][1] else None, **rec}
    if rec:
        res["exact"] = max(rec["bound_qk"], rec["bound_pv"]) <= rec["acc"]
    return res


def load(snapshot: str):
    model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, attn_implementation="eager")
    return model.cuda().eval()


def wikitext_ids(snapshot: str, n: int) -> np.ndarray:
    """BOS and the first n - 1 tokens of WikiText-2's test split joined with blank lines (window 0 at length n)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, rev, name = WIKITEXT
    text = "\n\n".join(pq.read_table(hf_hub_download(repo, name, repo_type="dataset", revision=rev)).column("text").to_pylist())
    tok = transformers.AutoTokenizer.from_pretrained(snapshot)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return np.array([tok.bos_token_id] + ids[:n - 1], dtype=np.int64)


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
    assert abs(entropy_bits("u16") - math.log2(33)) < 1e-12 and rms("pm13") == math.sqrt(5)
    causal = torch.ones(50, 50, dtype=torch.bool, device=dev).tril()
    p = torch.softmax(torch.randn(2, 50, 50, device=dev).masked_fill(~causal, float("-inf")), -1)
    pi, _ = edit_p(p, parse("int8-u16"), 111, causal, True, gen, dev)
    assert bool((pi.triu(1) == 0).all()) and bool((pi.tril() != torch.round(p / (p.amax(-1, keepdim=True) / 111))).any())
    L = transformers.models.llama.modeling_llama
    torch.manual_seed(0)
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8,
                        num_key_value_heads=2, head_dim=32, max_position_embeddings=1024, tie_word_embeddings=False,
                        attn_implementation="eager")
    model = L.LlamaForCausalLM(cfg).to(torch.bfloat16).to(dev).eval()
    ids = torch.randint(0, 512, (1, 96), device=dev)
    runs = {m: evaluate(model, ids, m) for m in ("bf16", "ref", "ref-rot", "int8-none", "int8-pfix-vhead-none",
                                                 "int8-u16", "int8-rot-u16", "e4m3-none", "e4m3-pm57-rot", "int8-u4-qk",
                                                 "int8-u4-pv-s2")}
    out["tiny"] = {m: {k: r[k] for k in ("ppl", "attn_rel_err", "exact")  if k in r} for m, r in runs.items()}
    assert abs(runs["ref"]["ppl"] / runs["bf16"]["ppl"] - 1) < 1e-2, out
    assert abs(runs["ref-rot"]["ppl"] / runs["ref"]["ppl"] - 1) < 1e-3, out
    assert runs["int8-none"]["attn_rel_err"] < runs["int8-u16"]["attn_rel_err"], out
    assert runs["int8-none"]["attn_rel_err"] < runs["e4m3-none"]["attn_rel_err"], out
    try:
        evaluate(model, ids, "e4m3-u16")
        raise AssertionError("e4m3-u16 must not fit")
    except ValueError:
        pass
    return out


def _key(r: dict, **over) -> tuple:
    q = {**r, **over}
    return (q["grid"], q["p"], q["v"], q["noise"], q["rot"], q["where"], q["ctx"])


def table(results: list[dict]) -> str:
    """One row per (edit, alphabet, context): perplexity (mean over seeds, with the range), and its change against the
    same edit without noise (seed 0) and against bf16 at the same context.  Marks against bf16: ** < 1%, * < 10%."""
    groups: dict = {}
    for r in results:
        groups.setdefault(_key(r), []).append(r)
    bf16 = {r["ctx"]: r["ppl"] for r in results if r["grid"] == "bf16"}
    rows = ["| ctx | grid | P | V | rot | noise on | alphabet | rms / max (LSB) | entropy (bits/entry) | seeds | perplexity "
            "| Δ% vs grid | Δ% vs bf16 | mark | attn rel. err |", "|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for key in sorted(groups, key=lambda k: (k[6], k[0] not in GRIDS, k[0], k[4], k[1], k[2], k[5], groups[k][0]["entropy_bits"],
                                             groups[k][0]["noise_rms_lsb"])):
        rs = groups[key]
        r = rs[0]
        ppls = [x["ppl"] for x in rs]
        ppl = sum(ppls) / len(ppls)
        base = groups.get(_key(r, noise="none", where="both"))
        dg = f"{100 * (ppl / base[0]['ppl'] - 1):+.2f}" if base and r["noise"] != "none" else "-"
        db = 100 * (ppl / bf16[r["ctx"]] - 1) if r["ctx"] in bf16 else None
        mark = "" if db is None else ("**" if db < 1 else "*" if db < 10 else "")
        spread = f" ({min(ppls):.3f}-{max(ppls):.3f})" if len(ppls) > 1 else ""
        err = f"{r['attn_rel_err']:.4f}" if r.get("attn_rel_err") is not None else "-"
        quant = r["grid"] in GRIDS
        rows.append(f"| {r['ctx']} | {r['grid']} | {r['p'] if quant else '-'} | {r['v'] if quant else '-'} | "
                    f"{'yes' if r['rot'] else 'no'} | {r['where'] if r['noise'] != 'none' else '-'} | {r['noise']} | "
                    f"{r['noise_rms_lsb']:.2f} / {r['noise_max_lsb']} | {r['entropy_bits']:.3f} | {len(rs)} | "
                    f"{ppl:.3f}{spread} | {dg} | {'-' if db is None else f'{db:+.2f}'} | {mark} | {err} |")
    return "\n".join(rows)


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
    assert argv[3] == "--ctx"
    ctx, methods = int(argv[4]), argv[5:]
    out.mkdir(parents=True, exist_ok=True)
    ids = np.load(ids_path)
    if ctx != len(ids):
        long = wikitext_ids(snapshot, ctx)
        info = {"ctx": ctx, "prefix_matches_ids": bool(np.array_equal(long[:len(ids)], ids)), "ids_len": len(ids)}
        (out / f"ids_{ctx}.json").write_text(json.dumps(info))
        print(json.dumps(info), file=sys.stderr, flush=True)
        ids = long
    ids_t = torch.from_numpy(ids)[None].cuda()
    model = load(snapshot)
    for method in methods:
        res = evaluate(model, ids_t, method)
        print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
