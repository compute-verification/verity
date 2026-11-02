"""Island `reference` (no cap on the update): continue the start with Muon on the blocks' matrices and AdamW on the embedding
and the norms, every tensor trained, within the fixed training-FLOP budget. E is every tensor in full (`prepare.encode_full`).

This file is the island's one editable file; prepare.py, evaluate.py, guard.py and candidate.sh are fixed (README.md, the program).

    python guard.py islands/reference/train.py --start m100 --out SUB [--muon-lr 0.02] [--adam-lr 3e-4] [--batch 64]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import prepare

KLASS = "reference"


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


def ns_flops(shape: tuple[int, int], steps: int = 5) -> int:
    m, n = min(shape), max(shape)
    return steps * (4 * m * m * n + 2 * m ** 3)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, wd: float = 0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum, "wd": wd})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.data = p.data.float()
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    mats = [n for n, p in params.items() if p.ndim == 2 and n != embed]
    rest = [n for n in params if n not in mats]
    muon = Muon([params[n] for n in mats], lr=a.muon_lr)
    adam = torch.optim.AdamW([params[n] for n in rest], lr=a.adam_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=prepare.DEVICE.startswith("cuda"))
    trainable = {"full": sorted(params)}
    tokens = a.batch * prepare.EVAL_CTX
    extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net)
    steps, snaps = [], []
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)}
    print(json.dumps({"steps": n_steps, "flops_per_step": per_step, "tokens_per_step": tokens}), flush=True)
    t0 = time.perf_counter()
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for o, lr in ((muon, a.muon_lr), (adam, a.adam_lr)):
            for g in o.param_groups:
                g["lr"] = lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        muon.step()
        adam.step()
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        steps += [{"op": "train", "tokens": tokens}, {"op": "extra", "flops": extra, "what": "muon newton-schulz"}]
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((prepare.encode_full(model_of(params, theta)), len(steps)))
    model = model_of(params, theta)
    prepare.submit(prepare.Path(a.out), a.start, KLASS, prepare.encode_full(model), model, trainable, steps,
                   changed=sorted(params), notes={"optimizer": "muon+adamw", "args": vars(a), "steps": n_steps},
                   snapshots=snaps)
    return 0


def model_of(params: dict, theta: dict) -> dict:
    return {n: p.detach().to(theta[n].dtype).cpu() for n, p in params.items()}


if __name__ == "__main__":
    raise SystemExit(main())
