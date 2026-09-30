#!/usr/bin/env python3
"""The channel split on Qwen2.5-7B, on CPU, streamed one decoder layer at a time (the model does not fit in RAM).

1. **Calibration.** A fixed slice of WikiText-2 *train* (the committed calibration data) runs through the BF16 model. For
   every linear layer, the split set S (the input channels that go to the BF16 side GEMM) is chosen from it by
   `pearlc_headline.split_channels`, before any evaluation data is seen. S is then fixed per layer.
2. **Quality.** Evaluation windows of WikiText-2 *test* run through four variants of every transformer linear layer, with
   the same tokens: BF16 (the reference), plain FP8 (per-row α = 448/max), FP8 with Pearl's headroom α (α = 448/(s + 4ρ),
   ρ over every 8th position: `Costs.adopted`'s quantizer), and FP8 with α plus the fixed split (S in BF16, the rest in FP8
   with α computed on the kept channels). The lm_head stays in BF16 in every variant. Perplexity and paired per-token NLL
   differences against BF16.
3. **Admission with the fixed S.** For every linear layer, on the evaluation activations (BF16 stream) and the weights:
   the rows that fail Π's rules before and after the fixed split, and the share of 512-token units (every row of the
   window and every weight row) that are admissible.

    python benchmarks/pouw/pearlc_split7b.py --eval-windows 4 --seqlen 512
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlc_headline as H  # noqa: E402
import run_result  # noqa: E402

Q_MAX = 448.0
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
           "mlp.down_proj")


def e4m3(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(-Q_MAX, Q_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def q_plain(x: torch.Tensor) -> torch.Tensor:
    alpha = Q_MAX / x.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return e4m3(x * alpha) / alpha


def q_headroom(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    s = x.abs().amax(1, keepdim=True)
    rho = x[:, ::8].pow(2).mean(1, keepdim=True).sqrt()
    alpha = Q_MAX / (s + 4 * delta * rho).clamp_min(1e-30)
    return e4m3(x * alpha) / alpha


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2 as Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=4)
    ap.add_argument("--calib-tokens", type=int, default=512)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--budget-share", type=float, default=0.01, help="most input channels S may take, as a share of k")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="pearlc_split7b.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    path = Path(snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"]))
    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "sdpa"
    wmap = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name: str) -> torch.Tensor:
        with safe_open(path / wmap[name], framework="pt") as f:
            return f.get_tensor(name)

    tok = AutoTokenizer.from_pretrained(path)
    test = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids[0]
    train = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:4000]), return_tensors="pt").input_ids[0]
    T = args.seqlen
    ev = torch.stack([test[w * T:(w + 1) * T] for w in range(args.eval_windows)])
    ca = train[:args.calib_tokens][None]
    emb = tensor("model.embed_tokens.weight")
    variants = ("bf16", "fp8", "fp8a", "fp8as")
    hs = {v: emb[ev].to(torch.float32) for v in variants}
    hc = emb[ca].to(torch.float32)
    del emb
    rot = Q.Qwen2RotaryEmbedding(config=cfg)
    pos_e, pos_c = torch.arange(T)[None].expand(args.eval_windows, T), torch.arange(ca.shape[1])[None]
    pe_e, pe_c = rot(hs["bf16"], pos_e), rot(hc, pos_c)
    report, t0 = [], time.time()
    state = {"mode": "bf16", "capture": None}
    for i in range(cfg.num_hidden_layers):
        layer = Q.Qwen2DecoderLayer(cfg, i)
        pre = f"model.layers.{i}."
        layer.load_state_dict({kk[len(pre):]: tensor(kk).to(torch.float32) for kk in wmap if kk.startswith(pre)})
        layer.eval()
        mods = {name: layer.get_submodule(name) for name in LINEARS}
        cache: dict = {}

        def make_fwd(name, mod):
            def fwd(x):
                mode = state["mode"]
                shape = x.shape
                x2 = x.reshape(-1, shape[-1])
                if state["capture"] is not None:
                    state["capture"].setdefault(name, []).append(x2.detach().clone())
                if mode == "bf16":
                    y = x2 @ mod.weight.T
                else:
                    y = q_dense(name, mod, x2, mode)
                if mod.bias is not None:
                    y = y + mod.bias
                return y.reshape(*shape[:-1], -1)
            return fwd

        def q_dense(name, mod, x2, mode):
            if mode == "fp8":
                if (name, mode) not in cache:
                    cache[(name, mode)] = q_plain(mod.weight)
                return q_plain(x2) @ cache[(name, mode)].T
            if mode == "fp8a":
                if (name, mode) not in cache:
                    cache[(name, mode)] = q_headroom(mod.weight, args.delta)
                return q_headroom(x2, args.delta) @ cache[(name, mode)].T
            S = split[name]
            if not len(S):
                return q_dense(name, mod, x2, "fp8a")
            keep = keeps[name]
            if (name, mode) not in cache:
                cache[(name, mode)] = (q_headroom(mod.weight[:, keep], args.delta), mod.weight[:, S])
            wq, ws = cache[(name, mode)]
            return q_headroom(x2[:, keep], args.delta) @ wq.T + bf16(x2[:, S]) @ ws.T

        for name, mod in mods.items():
            mod.forward = make_fwd(name, mod)
        # calibration: the BF16 model on the committed slice; S per linear from it
        state.update(mode="bf16", capture={})
        with torch.no_grad():
            hc = layer(hc, position_ids=pos_c, position_embeddings=pe_c)
        calib = {n: torch.cat(v).numpy().astype(np.float32) for n, v in state["capture"].items()}
        split, keeps = {}, {}
        for name, mod in mods.items():
            W = mod.weight.detach().numpy().astype(np.float32)
            k = W.shape[1]
            S, _ = H.split_channels(calib[name], W, args.delta, max(1, int(args.budget_share * k)))
            split[name] = torch.tensor(sorted(S), dtype=torch.long)
            keeps[name] = torch.tensor(np.setdiff1d(np.arange(k), S), dtype=torch.long)
        # evaluation: four variants; the BF16 stream's inputs are kept for the admission test
        evals = None
        for v in variants:
            state.update(mode=v, capture={} if v == "bf16" else None)
            with torch.no_grad():
                hs[v] = layer(hs[v], position_ids=pos_e, position_embeddings=pe_e)
            if v == "bf16":
                evals = {n: torch.cat(x).numpy().astype(np.float32) for n, x in state["capture"].items()}
            for key in [kk for kk in cache if kk[1] == v]:
                del cache[key]
        state["capture"] = None
        for name, mod in mods.items():
            W = mod.weight.detach().numpy().astype(np.float32)
            X = evals[name]
            k = W.shape[1]
            S = split[name].numpy()
            keep = keeps[name].numpy()
            fA, lA, _, _ = H.rules(X, args.delta)
            fB, lB, _, _ = H.rules(W, args.delta)
            fA2, lA2, _, _ = H.rules(X[:, keep], args.delta)
            fB2, lB2, _, _ = H.rules(W[:, keep], args.delta)
            okA, okA2 = fA & lA, fA2 & lA2
            units = okA.reshape(args.eval_windows, T).all(1) & (fB & lB).all()
            units2 = okA2.reshape(args.eval_windows, T).all(1) & (fB2 & lB2).all()
            report.append({"layer": i, "linear": name, "k": k, "n": W.shape[0], "split_channels": int(len(S)),
                           "calib_rows_fail": int((~np.logical_and(*H.rules(calib[name], args.delta)[:2])).sum()),
                           "eval_rows": int(len(X)), "eval_rows_fail": int((~okA).sum()), "eval_rows_fail_split": int((~okA2).sum()),
                           "weight_rows_fail": int((~(fB & lB)).sum()), "weight_rows_fail_split": int((~(fB2 & lB2)).sum()),
                           "units_admissible": int(units.sum()), "units_admissible_split": int(units2.sum()), "units": args.eval_windows})
        del layer, calib, evals
        print(json.dumps({"layer": i, "seconds": round(time.time() - t0, 1),
                          "split": {n: int(len(split[n])) for n in LINEARS},
                          "units_split": {r["linear"]: f"{r['units_admissible_split']}/{r['units']}" for r in report if r["layer"] == i}}), flush=True)
    norm_w = tensor("model.norm.weight").to(torch.float32)
    head = tensor("lm_head.weight").to(torch.float32)
    nll = {}
    for v in variants:
        h = hs[v]
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w
        per = []
        for w in range(args.eval_windows):
            logits = h[w] @ head.T
            per.append(F.cross_entropy(logits[:-1], ev[w, 1:], reduction="none"))
        nll[v] = torch.cat(per).numpy().astype(np.float64)
    res = {}
    for v in variants:
        d = nll[v] - nll["bf16"]
        n = len(d)
        # windows as blocks for the standard error, since tokens within a window are correlated
        blocks = d.reshape(args.eval_windows, -1).mean(1)
        se = float(blocks.std(ddof=1) / math.sqrt(args.eval_windows)) if args.eval_windows > 1 else float("nan")
        res[v] = {"ppl": float(math.exp(nll[v].mean())), "vs_bf16_pct": float(100 * (math.exp(d.mean()) - 1)),
                  "vs_bf16_pct_upper95": float(100 * (math.exp(d.mean() + 1.645 * se) - 1)), "tokens": n}
    for a, b in (("fp8a", "fp8"), ("fp8as", "fp8a"), ("fp8as", "fp8")):
        d = nll[a] - nll[b]
        blocks = d.reshape(args.eval_windows, -1).mean(1)
        se = float(blocks.std(ddof=1) / math.sqrt(args.eval_windows))
        res[f"{a}_vs_{b}"] = {"pct": float(100 * (math.exp(d.mean()) - 1)), "pct_se": float(100 * se)}
    down = [r for r in report if r["linear"] == "mlp.down_proj"]
    doc = {"tool": "pouw_pearlc_split7b", "model": args.model, "seqlen": T, "eval_windows": args.eval_windows,
           "calib": f"wikitext-2-raw-v1:train, first {args.calib_tokens} tokens of the first 4000 lines", "delta": args.delta,
           "budget_share": args.budget_share, "lm_head": "BF16 in every variant", "quality": res, "linears": report,
           "summary": {"split_channels_max": max(r["split_channels"] for r in report),
                       "split_channels_share_max": max(r["split_channels"] / r["k"] for r in report),
                       "linears_with_split": sum(r["split_channels"] > 0 for r in report),
                       "units_admissible": sum(r["units_admissible"] for r in report),
                       "units_admissible_split": sum(r["units_admissible_split"] for r in report),
                       "units": sum(r["units"] for r in report),
                       "down_units_admissible_split": sum(r["units_admissible_split"] for r in down),
                       "down_units": sum(r["units"] for r in down)},
           "seconds": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    q = res
    run_result.write(out, all(math.isfinite(q[v]["ppl"]) for v in variants), "pearlc-split7b",
                     f"ppl bf16 {q['bf16']['ppl']:.4f}, fp8 {q['fp8']['vs_bf16_pct']:+.2f}%, fp8+alpha {q['fp8a']['vs_bf16_pct']:+.2f}%, "
                     f"fp8+alpha+split {q['fp8as']['vs_bf16_pct']:+.2f}%; units admissible with the split "
                     f"{doc['summary']['units_admissible_split']}/{doc['summary']['units']}",
                     [("ppl_bf16", q["bf16"]["ppl"], "perplexity"), ("fp8_vs_bf16", q["fp8"]["vs_bf16_pct"], "percent"),
                      ("fp8a_vs_bf16", q["fp8a"]["vs_bf16_pct"], "percent"), ("fp8as_vs_bf16", q["fp8as"]["vs_bf16_pct"], "percent"),
                      ("units_admissible_split", doc["summary"]["units_admissible_split"], "count")])
    print(json.dumps({"quality": res, "summary": doc["summary"]}))


if __name__ == "__main__":
    main()
