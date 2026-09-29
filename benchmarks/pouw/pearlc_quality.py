#!/usr/bin/env python3
"""Quality of Pearl-C's useful output U on Qwen2.5-0.5B, WikiText-2 test, 2,048-token windows (CPU screen).

Variants, every linear layer and lm_head patched:
  bf16    the model as loaded;
  fp8     plain FP8: per-row E4M3 with alpha = 448/max|row| on activations (per token) and weights (per output channel);
  pearlc  Pearl-C's U: each row x is noised, y = x + bf16(delta*rms(x)*Z) with Z a rank-32 noise row of unit RMS
          (lines of entries +-(1..128), fresh per call for activations, fixed per run for weights), alpha = 448/max|y|,
          A' = RNE_E4M3(alpha*y); the clean-up removes the noise exactly, so the operand is A'/alpha - delta*rms(x)*Z.
          U = A_eff . W_eff^T in FP32.

What the emulation leaves out: the 14-bit truncating accumulation of the ticket chain and the BF16 rounding of the peel
columns (both below 2^-8 relative, against E4M3's 2^-4), and the promotion adds. The quantization of the noisy operands,
the effect that matters for quality, is exact.

    python benchmarks/pouw/pearlc_quality.py --variant pearlc --max-windows 54 --output pearlc.json

Under `research run` it writes result.json (the perplexity as the measurement).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_result  # noqa: E402

Q_MAX = 448.0
R = 32


def seed_of(*parts: object) -> int:
    h = hashlib.sha256("|".join(["verity/pouw/pearl-c/quality/v1", *map(str, parts)]).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def e4m3(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(-Q_MAX, Q_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def noise_rows(rows: int, k: int, gen: torch.Generator) -> torch.Tensor:
    def lines(n: int) -> torch.Tensor:
        mag = torch.randint(1, 129, (n, R), generator=gen).to(torch.float32)
        sgn = torch.randint(0, 2, (n, R), generator=gen).to(torch.float32) * 2 - 1
        return mag * sgn
    z = lines(rows) @ lines(k).T
    return z / z.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-30)


def plain_fp8(x: torch.Tensor) -> torch.Tensor:
    alpha = Q_MAX / x.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return e4m3(x * alpha) / alpha


def pearlc_operand(x: torch.Tensor, delta: float, gen: torch.Generator) -> torch.Tensor:
    rms = x.pow(2).mean(1, keepdim=True).sqrt()
    n = (delta * rms * noise_rows(x.shape[0], x.shape[1], gen))
    n_add = n.to(torch.bfloat16).to(torch.float32)
    y = x + n_add
    alpha = Q_MAX / y.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return e4m3(alpha * y) / alpha - n


def patch(model: torch.nn.Module, variant: str, delta: float, run_seed: int) -> int:
    count = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        count += 1
        mod._pc_name, mod._pc_calls = name, 0

        def fwd(x: torch.Tensor, mod=mod) -> torch.Tensor:
            shape = x.shape
            x2 = x.reshape(-1, shape[-1]).to(torch.float32)
            w = mod.weight.to(torch.float32)
            if variant == "fp8":
                a, wq = plain_fp8(x2), plain_fp8(w)
            else:
                ga = torch.Generator().manual_seed(seed_of(run_seed, mod._pc_name, "A", mod._pc_calls))
                gw = torch.Generator().manual_seed(seed_of(run_seed, mod._pc_name, "B"))
                a, wq = pearlc_operand(x2, delta, ga), pearlc_operand(w, delta, gw)
            mod._pc_calls += 1
            y = a @ wq.T
            if mod.bias is not None:
                y = y + mod.bias.to(torch.float32)
            return y.to(x.dtype).reshape(*shape[:-1], -1)

        mod.forward = fwd
    return count


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--variant", choices=("bf16", "fp8", "pearlc"), required=True)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    patched = 0 if args.variant == "bf16" else patch(model, args.variant, args.delta, args.seed)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    nwin = ids.numel() // args.seqlen
    if args.max_windows:
        nwin = min(nwin, args.max_windows)
    nll_sum, ntok, per = 0.0, 0, []
    t0 = time.time()
    with torch.no_grad():
        for w in range(nwin):
            chunk = ids[w * args.seqlen:(w + 1) * args.seqlen].unsqueeze(0)
            logits = model(chunk).logits.float()
            loss = F.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum").item()
            nll_sum += loss
            ntok += chunk.numel() - 1
            per.append(loss / (chunk.numel() - 1))
            print(json.dumps({"variant": args.variant, "window": w + 1, "windows": nwin, "nll": per[-1],
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    res = {"model": args.model, "dataset": "wikitext-2-raw-v1:test", "variant": args.variant, "delta": args.delta,
           "seqlen": args.seqlen, "windows": nwin, "tokens": ntok, "patched_linears": patched,
           "ppl": math.exp(nll_sum / ntok), "window_nll": per, "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in res.items() if k != "window_nll"}))
    out = Path(args.output or f"pearlc_quality_{args.variant}.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    run_result.write(out, math.isfinite(res["ppl"]), "pearlc-quality", f"{args.variant}: ppl {res['ppl']:.4f} over {nwin} windows",
                     [("ppl", res["ppl"], "perplexity"), ("windows", nwin, "count")])


if __name__ == "__main__":
    main()
