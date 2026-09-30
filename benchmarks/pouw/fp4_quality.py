#!/usr/bin/env python3
"""Quality of FP4 on Qwen2.5-7B: WikiText-2 perplexity of BF16, FP8, NVFP4 and MXFP4 (W4A4 and W4A16), and a stand-in for
the FP4 PoUW output, streamed one decoder layer at a time (the model does not fit a small VM's RAM; on a GPU the same code
runs with ``--device cuda``).

Every transformer linear is quantized; ``lm_head`` stays in BF16 in every variant (as the FP8 lane's pearlc_split7b).
Quantization is exact emulation: values are rounded onto the format's grid and computed in FP32.

- ``fp8``: per-row E4M3, alpha = 448/max|row|, activations per token and weights per output channel (the FP8 lane's plain FP8).
- ``nvf4``: NVIDIA's two-level NVFP4 -- a per-tensor FP32 scale g = amax/(6*448) (activations: per call), per 16-block along
  k a UE4M3 scale e4m3(amax_b/(6g)), elements RNE onto E2M1 saturating at 6.
- ``mxf4``: OCP MX v1.0 MXFP4 -- per 32-block along k a UE8M0 scale 2^(floor(log2 amax_b) - 2), elements RNE onto E2M1
  saturating at 6.
- ``*_w4a16``: the weights only.
- ``pouw_nvf4`` / ``pouw_mxf4`` (stand-in until the FP4 PoUW design fixes its noise): the useful output of a Pearl-C-style
  scheme, where both operands are quantized with noise added and the clean-up removes the noise exactly, so the effective
  operand is deq(Q(x + n)) - n.  n has RMS sigma_c code units of each block's clean scale (amax_b/6), fresh per call for
  activations and fixed per run for weights.

    python benchmarks/pouw/fp4_quality.py --eval-windows 4 --seqlen 512                  # CPU, the FP8 lane's subset
    python benchmarks/pouw/fp4_quality.py --device cuda --eval-windows 0 --seqlen 2048   # GPU: the whole test split
    python benchmarks/pouw/fp4_quality.py ... --dump-acts DIR --dump-layers 0,3,13,27   # real operands for fp4_census.py real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_result  # noqa: E402

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
           "mlp.down_proj")
VARIANTS = ("bf16", "fp8", "nvf4", "mxf4", "nvf4_w4a16", "mxf4_w4a16", "pouw_nvf4", "pouw_mxf4")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MID = (E2M1[1:] + E2M1[:-1]) / 2


def seed_of(*parts: object) -> int:
    h = hashlib.sha256("|".join(["verity/pouw/fp4/quality/v1", *map(str, parts)]).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def e2m1(v: torch.Tensor) -> torch.Tensor:
    """RNE onto E2M1 values, saturating at +-6; ties go to the even mantissa (0, 1, 2, 4 have even codes)."""
    a = v.abs().clamp(max=6.0)
    mid, grid = E2M1_MID.to(v.device), E2M1.to(v.device)
    i = torch.bucketize(a, mid)                                   # mid[i-1] < a <= mid[i] -> i: a tie lands on the lower code
    tie = (i < 7) & (a == mid[i.clamp(max=6)])
    i = torch.where(tie & (i % 2 == 1), i + 1, i)                 # a tie keeps the even code
    return grid[i].copysign(v) * (v != 0)


def e4m3(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


def q_fp8(x: torch.Tensor) -> torch.Tensor:
    alpha = 448.0 / x.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return e4m3(x * alpha) / alpha


def q_nvf4(x: torch.Tensor) -> torch.Tensor:
    r, k = x.shape
    g = (x.abs().amax() / (6.0 * 448.0)).clamp_min(1e-30)
    xb = (x / g).reshape(r, k // 16, 16)
    s = e4m3(xb.abs().amax(-1, keepdim=True) / 6.0)
    q = torch.where(s > 0, e2m1(xb / torch.where(s > 0, s, torch.ones_like(s))), torch.zeros_like(xb))
    return (q * s).reshape(r, k) * g


def q_mxf4(x: torch.Tensor) -> torch.Tensor:
    r, k = x.shape
    xb = x.reshape(r, k // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 2
    s = torch.exp2(e.clamp(-127, 127))
    return (e2m1(xb / s) * s * (amax > 0)).reshape(r, k)


QUANT = {"fp8": q_fp8, "nvf4": q_nvf4, "mxf4": q_mxf4}


def block_noise(x: torch.Tensor, fmt: str, sigma_c: float, gen: torch.Generator) -> torch.Tensor:
    """RMS sigma_c code units of each block's clean scale amax_b/6 (the stand-in noise)."""
    r, k = x.shape
    blk = 16 if fmt == "nvf4" else 32
    xb = x.reshape(r, k // blk, blk)
    s = xb.abs().amax(-1, keepdim=True) / 6.0
    z = torch.randn(xb.shape, generator=gen, device="cpu").to(x.device)
    return (sigma_c * s * z).reshape(r, k)


def pouw_operand(x: torch.Tensor, fmt: str, sigma_c: float, gen: torch.Generator) -> torch.Tensor:
    n = block_noise(x, fmt, sigma_c, gen)
    return QUANT[fmt](x + n) - n


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2 as Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=4, help="0: every window of the test split")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--sigma-c", type=float, default=0.5, help="stand-in PoUW noise, RMS in code units of the block scale")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--window-batch", type=int, default=8, help="windows per layer call (memory)")
    ap.add_argument("--dump-acts", default="", help="directory for real operands (fp4_census.py real)")
    ap.add_argument("--dump-layers", default="0,1,3,7,13,20,27")
    ap.add_argument("--dump-rows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="fp4_quality.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    variants = tuple(v for v in args.variants.split(",") if v)
    assert variants[0] == "bf16", "bf16 is the reference and runs first"
    path = Path(snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"]))
    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "sdpa"
    idx = path / "model.safetensors.index.json"
    wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def tensor(name: str) -> torch.Tensor:
        f = path / wmap[name] if wmap else path / "model.safetensors"
        with safe_open(f, framework="pt") as h:
            return h.get_tensor(name)

    names = list(wmap) if wmap else list(safe_open(path / "model.safetensors", framework="pt").keys())
    tok = AutoTokenizer.from_pretrained(path)
    test = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids[0]
    T = args.seqlen
    nwin = test.numel() // T if args.eval_windows == 0 else args.eval_windows
    ev = torch.stack([test[w * T:(w + 1) * T] for w in range(nwin)])
    emb = tensor("model.embed_tokens.weight")
    hs = {v: emb[ev].to(torch.float32) for v in variants}               # kept on the CPU, moved per window batch
    del emb
    rot = Q.Qwen2RotaryEmbedding(config=cfg).to(dev)
    pos = torch.arange(T, device=dev)[None]
    dump_layers = {int(x) for x in args.dump_layers.split(",") if x} if args.dump_acts else set()
    if args.dump_acts:
        Path(args.dump_acts).mkdir(parents=True, exist_ok=True)
    state = {"mode": "bf16", "call": 0, "capture": None}
    t0 = time.time()
    for i in range(cfg.num_hidden_layers):
        layer = Q.Qwen2DecoderLayer(cfg, i)
        pre = f"model.layers.{i}."
        layer.load_state_dict({kk[len(pre):]: tensor(kk).to(torch.float32) for kk in names if kk.startswith(pre)})
        layer.eval().to(dev)
        mods = {name: layer.get_submodule(name) for name in LINEARS}
        cache: dict = {}

        def weight(name: str, mod, mode: str) -> torch.Tensor:
            if (name, mode) not in cache:
                w = mod.weight
                if mode in ("fp8", "nvf4", "mxf4", "nvf4_w4a16", "mxf4_w4a16"):
                    w = QUANT[mode.split("_")[0]](w)
                elif mode.startswith("pouw_"):
                    w = pouw_operand(w, mode[5:], args.sigma_c, torch.Generator().manual_seed(seed_of(args.seed, i, name, "B")))
                cache[(name, mode)] = w
            return cache[(name, mode)]

        def make_fwd(name, mod):
            def fwd(x):
                mode = state["mode"]
                shape = x.shape
                x2 = x.reshape(-1, shape[-1])
                if state["capture"] is not None:
                    state["capture"].setdefault(name, []).append(x2[: args.dump_rows].detach().float().cpu().numpy())
                if mode in ("fp8", "nvf4", "mxf4"):
                    x2 = QUANT[mode](x2)
                elif mode.startswith("pouw_"):
                    gen = torch.Generator().manual_seed(seed_of(args.seed, i, name, "A", state["call"]))
                    x2 = pouw_operand(x2, mode[5:], args.sigma_c, gen)
                y = x2 @ weight(name, mod, mode).T
                if mod.bias is not None:
                    y = y + mod.bias
                return y.reshape(*shape[:-1], -1)
            return fwd

        for name, mod in mods.items():
            mod.forward = make_fwd(name, mod)
        for v in variants:
            state["mode"] = v
            outs = []
            for b0 in range(0, nwin, args.window_batch):
                h = hs[v][b0:b0 + args.window_batch].to(dev)
                state["call"] = b0
                state["capture"] = {} if (v == "bf16" and i in dump_layers and b0 == 0) else None
                pe = rot(h, pos.expand(h.shape[0], T))
                with torch.no_grad():
                    o = layer(h, position_ids=pos.expand(h.shape[0], T), position_embeddings=pe)
                outs.append((o[0] if isinstance(o, tuple) else o).float().cpu())
                if state["capture"]:
                    rows = {n: np.concatenate(x)[: args.dump_rows] for n, x in state["capture"].items()}
                    np.savez(Path(args.dump_acts) / f"layer{i:02d}.npz",
                             **{f"{n}/x": rows[n].astype(np.float32) for n in rows},
                             **{f"{n}/w": mods[n].weight[: args.dump_rows].detach().float().cpu().numpy() for n in rows})
                    state["capture"] = None
            hs[v] = torch.cat(outs)
            for key in [kk for kk in cache if kk[1] == v]:
                del cache[key]
        del layer, cache
        print(json.dumps({"layer": i, "seconds": round(time.time() - t0, 1)}), flush=True)
    norm_w = tensor("model.norm.weight").to(torch.float32).to(dev)
    head = (tensor("lm_head.weight") if "lm_head.weight" in names else tensor("model.embed_tokens.weight")).to(torch.float32).to(dev)
    nll = {}
    for v in variants:
        per = []
        for w in range(nwin):
            h = hs[v][w].to(dev)
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w
            per.append(F.cross_entropy((h @ head.T)[:-1], ev[w, 1:].to(dev), reduction="none").cpu())
        nll[v] = torch.cat(per).numpy().astype(np.float64)
    res = {}
    for v in variants:
        d = nll[v] - nll["bf16"]
        blocks = d.reshape(nwin, -1).mean(1)                            # windows as blocks: tokens within one are correlated
        se = float(blocks.std(ddof=1) / math.sqrt(nwin)) if nwin > 1 else float("nan")
        res[v] = {"ppl": float(math.exp(nll[v].mean())), "vs_bf16_pct": float(100 * (math.exp(d.mean()) - 1)),
                  "vs_bf16_pct_se": float(100 * se), "vs_bf16_pct_upper95": float(100 * (math.exp(d.mean() + 1.645 * se) - 1)),
                  "tokens": int(len(d))}
    pairs = [("pouw_nvf4", "nvf4"), ("pouw_mxf4", "mxf4"), ("mxf4", "nvf4")]
    for a, b in pairs:
        if a in nll and b in nll:
            d = nll[a] - nll[b]
            blocks = d.reshape(nwin, -1).mean(1)
            res[f"{a}_vs_{b}"] = {"pct": float(100 * (math.exp(d.mean()) - 1)),
                                  "pct_se": float(100 * blocks.std(ddof=1) / math.sqrt(nwin)) if nwin > 1 else float("nan")}
    doc = {"tool": "pouw_fp4_quality", "model": args.model, "dataset": "wikitext-2-raw-v1:test", "seqlen": T, "windows": nwin,
           "device": args.device, "sigma_c": args.sigma_c, "lm_head": "BF16 in every variant", "variants": list(variants),
           "quality": res, "dump_acts": args.dump_acts or None, "seconds": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    run_result.write(out, all(math.isfinite(res[v]["ppl"]) for v in variants), "fp4-quality",
                     "; ".join(f"{v} {res[v]['vs_bf16_pct']:+.2f}%" for v in variants[1:]) + f"; ppl bf16 {res['bf16']['ppl']:.4f}",
                     [("ppl_bf16", res["bf16"]["ppl"], "perplexity")] + [(f"{v}_vs_bf16", res[v]["vs_bf16_pct"], "percent") for v in variants[1:]])
    print(json.dumps({"quality": res}))


if __name__ == "__main__":
    main()
