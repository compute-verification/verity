"""Route `subset`: train the parameters `trainable` names (comma-separated patterns over the start's parameter names,
fnmatch, `blocks.5.*,norm`), the rest frozen, with AdamW at `lr`, warmed up over `warmup` steps and decaying linearly
to 0, the gradient clipped to norm `clip` (0: unclipped), over micro-batches of `micro` windows (0: the whole batch).
E is the trained tensors in full.
"""
from __future__ import annotations

import fnmatch

import torch
import torch.nn.functional as F

from training_slowdown.codecs import Identity, Plan
from training_slowdown.model import LM

VERSION = 1
DIALS = {"trainable": "*", "lr": 3e-4, "weight_decay": 0.0, "warmup": 0, "clip": 1.0, "micro": 0}


def trained(model: torch.nn.Module, dials: dict) -> list[str]:
    patterns = [p.strip() for p in dials["trainable"].split(",") if p.strip()]
    names = [n for n, _ in model.named_parameters() if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if not names:
        raise SystemExit(f"subset: trainable {dials['trainable']!r} names no parameter of the start")
    return names


def build(shape: dict, dials: dict):
    model = LM.of(shape)
    names = set(trained(model, dials))
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    params = [p for n, p in model.named_parameters() if n in names]
    return model, [torch.optim.AdamW(params, lr=dials["lr"], betas=(0.9, 0.95), weight_decay=dials["weight_decay"])]


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    frac = (min(1.0, (t + 1) / dials["warmup"]) if dials["warmup"] else 1.0) * (1 - t / n)
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = dials["lr"] * frac
    micro = dials["micro"] or len(rows)
    for i in range(0, len(rows), micro):
        x = rows[i: i + micro]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        (loss * (len(x) / len(rows))).backward()
    if dials["clip"]:
        torch.nn.utils.clip_grad_norm_([p for o in optimizers for g in o.param_groups for p in g["params"]],
                                       dials["clip"])
    for o in optimizers:
        o.step()
        o.zero_grad(set_to_none=True)


def plan(dials: dict) -> Plan:
    return Plan([(p.strip(), Identity()) for p in dials["trainable"].split(",") if p.strip()])
