"""Train an `lf` start and register its weights at 50% and 100% of the run as starts `<size>50` and `<size>100`.

The recipe is learning_frontier's dense reference at its best measured rate (s: 3e-3, m: 1.5e-3): AdamW (0.9, 0.95; weight
decay 0.1 on matrices), clipping at 1.0, 64 sequences of 1024 tokens per step, 8 tokens per non-embedding parameter, linear
warmup over 5% and cosine decay to 10%, bf16 autocast, initialization seed 0, data order seed 0 (the corpus's windows
permuted by numpy default_rng(0)). Each saved start is the fp32 parameters at that step; nothing of the optimizer is kept.

    python starts.py --size m [--lr 1.5e-3] [--max-steps N]
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

import prepare
from model import CTX, SIZES, LM

LR = {"s": 3e-3, "m": 1.5e-3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", choices=sorted(LR) + ["tiny"], required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tokens-per-param", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=0, help="a smoke test: stop and register after this many steps")
    ap.add_argument("--name-suffix", default="", help="appended to the registered names (a smoke test's)")
    ap.add_argument("--ckpt", type=Path, default=None, help="resume state here; a chunk past --chunk-minutes saves and exits 3")
    ap.add_argument("--chunk-minutes", type=float, default=26.0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    size = SIZES[a.size]
    lr_peak = a.lr or LR.get(a.size, 3e-3)
    torch.manual_seed(0)
    model = LM(size).cuda()
    n_mat = sum(p.numel() for n, p in model.named_parameters() if p.ndim == 2 and n != "embed")
    n_norm = sum(p.numel() for p in model.parameters() if p.ndim == 1)
    n_steps = max(1, round(a.tokens_per_param * (n_mat + n_norm) / (a.batch * CTX)))
    stop = min(n_steps, a.max_steps) if a.max_steps else n_steps
    marks = {max(1, round(n_steps / 2)): "50", n_steps: "100"}
    if a.max_steps:
        marks = {max(1, stop // 2): "50", stop: "100"}
    train = np.memmap(prepare.LF_CORPUS / "train.bin", dtype=np.uint16, mode="r")
    order = np.random.default_rng(0).permutation((len(train) - 1) // CTX)[: n_steps * a.batch]
    decay = [p for p in model.parameters() if p.ndim >= 2]
    keep = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": keep, "weight_decay": 0.0}],
                            lr=lr_peak, betas=(0.9, 0.95), eps=1e-8, fused=True)
    fwd = torch.compile(model)
    warm = max(1, round(0.05 * n_steps))
    t0 = time.perf_counter()
    ckpt = a.ckpt / f"{a.size}{a.name_suffix}.pt" if a.ckpt else None
    first, spent = 0, 0.0
    if ckpt and ckpt.exists():
        s = torch.load(ckpt, map_location="cuda", weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        first, spent = s["step"], s["seconds"]
        print(f"starts: resumed at step {first}", flush=True)
    for step in range(first, stop):
        if ckpt and (time.perf_counter() - t0) / 60 > a.chunk_minutes:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "seconds": spent + time.perf_counter() - t0}, ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").rename(ckpt)
            print(f"starts: chunk ended at step {step}; state saved", flush=True)
            return 3
        lr = lr_peak * (step + 1) / warm if step < warm else lr_peak * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, n_steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = np.sort(order[step * a.batch: (step + 1) * a.batch])
        rows = torch.from_numpy(np.stack([train[i * CTX: i * CTX + CTX + 1] for i in idx]).astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(rows[:, :-1])
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), rows[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        if done % 200 == 0:
            print(json.dumps({"step": done, "of": n_steps, "loss": float(loss), "s": round(time.perf_counter() - t0, 1)}),
                  flush=True)
        if done in marks:
            name = f"{a.size}{marks[done]}{a.name_suffix}"
            theta = {n: p.detach().float().cpu().contiguous() for n, p in model.named_parameters()}
            rec = prepare.register_start(name, theta, size.config(), "gpt2", {
                "recipe": "learning_frontier dense", "lr": lr_peak, "step": done, "of_steps": n_steps,
                "tokens": done * a.batch * CTX, "batch": a.batch, "tokens_per_param": a.tokens_per_param,
                "init_seed": 0, "data_seed": 0, "train_loss": float(loss), "seconds": spent + time.perf_counter() - t0})
            print(json.dumps({"registered": name, "digest": rec["digest"]}), flush=True)
    if ckpt and ckpt.exists():
        ckpt.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
