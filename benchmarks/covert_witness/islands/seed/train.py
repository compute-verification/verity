"""Island `seed` (E is a seed log, at most 100,000 bytes): MeZO, zeroth-order SGD. Each step draws z from a CUDA generator
seeded `seed + step` over the perturbed tensors, measures the loss at θ_t + εz and θ_t − εz on one batch (two forward passes,
no backward), and steps θ_{t+1} = θ_t − c_t z with c_t = lr_t · (L₊ − L₋) / 2ε. E is the coefficients (`prepare.encode_seedlog`,
f32), so the decoder replays every step from the seeds; the update is applied with `prepare.seedlog_step`, the decoder's own.

This file is the island's one editable file; prepare.py, evaluate.py, guard.py and candidate.sh are fixed (program.md).

    python guard.py islands/seed/train.py --start m100 --out SUB [--lr 1e-6] [--eps 1e-3] [--batch 32] [--perturb blocks]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

KLASS = "seed"
COEFF = "f32"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--perturb", default="blocks", help="`blocks` (every block tensor and the final norm) or `all`")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    names = sorted(params) if a.perturb == "all" else sorted(n for n in params if n != embed)
    for p in params.values():
        p.requires_grad_(False)
    masters = [params[n].detach().float().clone() for n in names]
    live = [params[n] for n in names]
    trainable = {"full": names}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": tokens}] * 2)["flops"]
    cap = prepare.CLASSES[KLASS]["max_e_bytes"]
    header = len(prepare.encode_seedlog(names, a.seed, [], COEFF))
    n_steps = min(int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step), (cap - header) // 4)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net)
    gen = torch.Generator(device=prepare.DEVICE)
    steps, coeffs, snaps = [], [], []
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)}
    print(json.dumps({"steps": n_steps, "perturbed": len(names), "flops_per_step": per_step}), flush=True)

    @torch.no_grad()
    def loss_at(sign: float, seed: int, x: torch.Tensor) -> float:
        gen.manual_seed(seed)
        for w, m in zip(live, masters):
            z = torch.randn(m.shape, generator=gen, device=prepare.DEVICE, dtype=torch.float32)
            w.copy_((m + sign * a.eps * z).to(w.dtype))
        with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
            logits = fwd(x[:, :-1])
        return float(F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)))

    t0 = time.perf_counter()
    ema = None
    for step in range(n_steps):
        x = data.next(a.batch)
        lp, lm = loss_at(1.0, a.seed + step, x), loss_at(-1.0, a.seed + step, x)
        c = prepare.quantize_coefficient(a.lr * (1 - step / n_steps) * (lp - lm) / (2 * a.eps), COEFF)
        prepare.seedlog_step(masters, a.seed + step, c, gen)
        coeffs.append(c)
        steps += [{"op": "forward", "tokens": tokens}] * 2
        ema = (lp + lm) / 2 if ema is None else 0.99 * ema + 0.01 * (lp + lm) / 2
        if (step + 1) % 500 == 0:
            print(json.dumps({"step": step + 1, "loss_ema": ema, "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((prepare.encode_seedlog(names, a.seed, coeffs, COEFF), len(steps)))
    blob = prepare.encode_seedlog(names, a.seed, coeffs, COEFF)
    model = dict(theta) | {n: m.to(theta[n].dtype).cpu() for n, m in zip(names, masters)}
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=names,
                   notes={"method": "mezo", "args": vars(a), "steps": n_steps}, snapshots=snaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
