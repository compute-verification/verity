"""Island `e100kb` (E sized to 100,000 bytes, `--e-bytes`): LoRA on the blocks' output projections (o and down),
sized so the factors (and the norm gains, when trained) fit that size in bf16; E is the factors (`prepare.encode_lora`),
and the evaluated model merges them.

The base stays frozen. Each target weight W gets A (r x d_in, N(0, 1/d_in)) and B (d_out x r, zero) used as
W + s · B @ A with s = sqrt(d_in / r), trained by AdamW within the fixed training-FLOP budget. The submitted model is
`prepare.lora_merge` of the bf16-rounded factors, exactly what the decoder rebuilds.

This file is the island's one editable file; prepare.py, evaluate.py, guard.py and candidate.sh are fixed (README.md, the program).

    python guard.py islands/e100kb/train.py --start m100 --out SUB [--targets out] [--rank 0 (largest that fits)] [--norms]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

#: the island's update size, its own choice (nothing caps it); `--e-bytes` sets another
E_BYTES = 100_000
DEFAULT_TARGETS = "out"
DEFAULT_NORMS = True
LF_MATS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
HF_MATS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def targets_of(rec: dict, params: dict, spec: str) -> list[str]:
    lf = rec["config"]["family"] == "lf"
    want = (set(LF_MATS if lf else HF_MATS) if spec == "all" else {"wo", "w_down", "o_proj", "down_proj"} if spec == "out"
            else set(spec.split(",")))
    out = []
    for n, p in params.items():
        if p.ndim != 2 or n in ("embed", "model.embed_tokens.weight"):
            continue
        leaf = n.split(".")[-2] if n.endswith(".weight") else n.split(".")[-1]
        if leaf in want:
            out.append(n)
    return sorted(out)


def e_bytes(theta, targets, rank, norms) -> int:
    fac = {n: (torch.zeros(rank, theta[n].shape[1]), torch.zeros(theta[n].shape[0], rank), 1.0) for n in targets}
    return len(prepare.encode_lora(fac, "bf16", {n: theta[n] for n in norms}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help="`all`, `out` (each block's two output projections), or leaf names")
    ap.add_argument("--rank", type=int, default=0, help="0: the largest that fits --e-bytes")
    ap.add_argument("--e-bytes", type=int, default=E_BYTES, help="the size E is sized to")
    ap.add_argument("--norms", action=argparse.BooleanOptionalAction, default=DEFAULT_NORMS)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--norm-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    want = a.e_bytes
    targets = targets_of(rec, params, a.targets)
    norms = sorted(n for n, p in params.items() if p.ndim == 1) if a.norms else []
    rank = a.rank
    if not rank:
        rank = 1
        while e_bytes(theta, targets, rank + 1, norms) <= want - 512:
            rank += 1
    size = e_bytes(theta, targets, rank, norms)
    if size > want:
        raise SystemExit(f"e100kb: rank {rank} on {len(targets)} weights{' and the norms' if norms else ''} is {size} bytes, "
                         f"over --e-bytes {want}")
    for p in params.values():
        p.requires_grad_(False)
    for n in norms:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    g = torch.Generator(device=prepare.DEVICE).manual_seed(a.seed)
    factors = {}
    for n in targets:
        w = params[n]
        A = torch.nn.Parameter(torch.randn(rank, w.shape[1], device=prepare.DEVICE, generator=g) / math.sqrt(w.shape[1]))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], rank, device=prepare.DEVICE))
        s = math.sqrt(w.shape[1] / rank)
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))
    opt = torch.optim.AdamW([{"params": [p for A, B, _ in factors.values() for p in (A, B)], "lr": a.lr, "weight_decay": 0.0},
                             {"params": [params[n] for n in norms], "lr": a.norm_lr, "weight_decay": 0.0}],
                            betas=(0.9, 0.95), fused=prepare.DEVICE.startswith("cuda"))
    trainable = {"full": norms, "adapters": {n: rank for n in targets}}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net)
    steps, snaps = [], []
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)}
    print(json.dumps({"steps": n_steps, "rank": rank, "targets": len(targets), "norms": len(norms), "E_bytes": size,
                      "flops_per_step": per_step}), flush=True)
    t0 = time.perf_counter()
    lrs = [grp["lr"] for grp in opt.param_groups]
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for grp, lr in zip(opt.param_groups, lrs):
            grp["lr"] = lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((encode(factors, params, norms), len(steps)))
    blob = encode(factors, params, norms)
    model = prepare.decode(theta, blob)
    prepare.submit(Path(a.out), a.start, blob, model, trainable, steps, changed=targets + norms,
                   notes={"method": "lora", "rank": rank, "args": vars(a), "steps": n_steps}, snapshots=snaps)
    return 0


def encode(factors: dict, params: dict, norms: list[str]) -> bytes:
    return prepare.encode_lora({n: (A, B, s) for n, (A, B, s) in factors.items()}, "bf16", {n: params[n] for n in norms})


if __name__ == "__main__":
    raise SystemExit(main())
