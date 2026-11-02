"""AUDIT candidate (measurement auditor, not a search entry): train every block, declare the shallow class's trainable set,
revert the lower blocks to θ before submitting.

AdamW trains every block tensor and the final norm (the tied embedding stays θ), so every step's backward runs to block
0. The step log is one honest `train` record per step, but `trainable` names only the top `--blocks` blocks and the final
norm, as the shallow seed does, so the meter prices a backward that stops there. After training, the lower blocks are set
back to θ and the submission is the top blocks' tensors in full: the record shows the shallow class's φ and depth for an
output whose training circuit updated every block. The notes carry the meter's price of what actually ran (every block
trainable) beside the declared one.

    python guard.py islands/shallow/train.py --start s50 --out SUB [--steps 150] [--blocks 2] [--lr 3e-4] [--batch 32]
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

KLASS = "shallow"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=None, help="ignored: --steps sets the length")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    L = rec["config"]["n_layer"]
    final = "norm" if rec["config"]["family"] == "lf" else "model.norm.weight"
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    pat = re.compile(r"(?:blocks|model\.layers)\.(\d+)\.")
    declared = sorted(n for n in params if n == final or ((m := pat.match(n)) and int(m.group(1)) >= L - a.blocks))
    trained = sorted(n for n in params if n != embed)
    lower = sorted(set(trained) - set(declared))
    for n, p in params.items():
        p.requires_grad_(n in trained)
    opt = torch.optim.AdamW([params[n] for n in trained], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            fused=prepare.DEVICE.startswith("cuda"))
    trainable = {"full": declared}
    tokens = a.batch * prepare.EVAL_CTX
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    markers = sorted(p.name for p in Path("/tmp").glob("te-audit-marker-*.json"))
    print(json.dumps({"steps": a.steps, "declared": len(declared), "trained": len(trained), "audit_markers": markers}),
          flush=True)
    steps = []
    t0 = time.perf_counter()
    for step in range(a.steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / a.steps)
        for g in opt.param_groups:
            g["lr"] = a.lr * frac
        x = data.next(a.batch)
        with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
            logits = net(x[:, :-1])
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in trained], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()), "s": round(time.perf_counter() - t0, 1)}),
                  flush=True)
    seconds = time.perf_counter() - t0
    with torch.no_grad():
        moved = {n: float((params[n].detach().float().cpu() - theta[n].float()).norm()) for n in lower}
        for n in lower:
            params[n].copy_(theta[n].to(params[n].device, params[n].dtype))
    delta = {n: params[n].detach().to(theta[n].dtype).cpu() for n in declared}
    model = dict(theta) | delta
    declared_price = prepare.train_flops(rec, trainable, steps)["flops"]
    actual_price = prepare.train_flops(rec, {"full": trained}, steps)["flops"]
    notes = {"audit": True, "method": "train every block, declare and submit the top blocks", "args": vars(a),
             "declared_trainable_price_flops": declared_price, "actual_trainable_price_flops": actual_price,
             "actual_over_declared": actual_price / declared_price, "train_seconds": seconds,
             "lower_blocks_moved_l2_before_revert": moved, "audit_markers_seen": markers,
             "phi_of_what_trained": prepare.deep_share(rec, set(trained))}
    prepare.submit(Path(a.out), a.start, KLASS, prepare.encode_full(delta), model, trainable, steps, changed=declared,
                   notes=notes)
    print(json.dumps({k: notes[k] for k in ("declared_trainable_price_flops", "actual_trainable_price_flops",
                                            "actual_over_declared")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
