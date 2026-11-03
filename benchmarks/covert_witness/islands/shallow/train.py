"""Island `shallow` (only the top 2 blocks and the final norm may differ, any size): tune the top blocks with AdamW, the rest
frozen, so the backward stops at the lowest tuned block. The head is the tied embedding, which the input lookup also reads (layer
0), so it stays frozen. E is the tuned tensors in full (`prepare.encode_full`).

This file is the island's one editable file; prepare.py, evaluate.py, guard.py and candidate.sh are fixed (README.md, the program).

    python guard.py islands/shallow/train.py --start m100 --out SUB [--blocks 2] [--lr 3e-4] [--batch 64]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    L = rec["config"]["n_layer"]
    final = "norm" if rec["config"]["family"] == "lf" else "model.norm.weight"
    pat = re.compile(r"(?:blocks|model\.layers)\.(\d+)\.")
    tuned = sorted(n for n in params if n == final or ((m := pat.match(n)) and int(m.group(1)) >= L - a.blocks))
    for n, p in params.items():
        p.requires_grad_(n in tuned)
        if n in tuned:
            p.data = p.data.float()
    opt = torch.optim.AdamW([params[n] for n in tuned], lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd, fused=prepare.DEVICE.startswith("cuda"))
    trainable = {"full": tuned}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net)
    steps, snaps = [], []
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)}
    print(json.dumps({"steps": n_steps, "tuned": len(tuned), "flops_per_step": per_step}), flush=True)
    t0 = time.perf_counter()
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for g in opt.param_groups:
            g["lr"] = a.lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in tuned], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((prepare.encode_full(tensors(params, theta, tuned)), len(steps)))
    delta = tensors(params, theta, tuned)
    model = dict(theta) | delta
    prepare.submit(Path(a.out), a.start, prepare.encode_full(delta), model, trainable, steps, changed=tuned,
                   notes={"method": "top-block tuning", "args": vars(a), "steps": n_steps}, snapshots=snaps)
    return 0


def tensors(params: dict, theta: dict, names: list[str]) -> dict:
    return {n: params[n].detach().to(theta[n].dtype).cpu() for n in names}


if __name__ == "__main__":
    raise SystemExit(main())
