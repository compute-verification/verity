"""Route `subset`: train the parameters `trainable` names (comma-separated patterns over the start's parameter names,
fnmatch, `blocks.5.*,norm`), the rest frozen.

- `opt`: the optimizer. Four train every one of them, the embedding included, under `weight_decay`: `adamw` (betas
  0.9, 0.95), `sgd` (torch's, Nesterov momentum 0.9, its decay added to the gradient), `lion` and `adafactor`
  (`_optimizers`; Adafactor's rate is at most 1/sqrt(t) and relative to each weight's RMS). Two train the matrices
  and leave the gains to AdamW, undecayed, with no rule for the embedding: `muon` (Nesterov momentum 0.95, the
  update orthogonalized by five Newton-Schulz steps in bf16) and `shampoo` (the gradient between the inverse fourth
  roots of its two Kronecker factors), both in `_optimizers`, `weight_decay` decoupled.
- `lr`: every optimizer's rate, times the schedule: a ramp over the first `warmup` steps (a share of the steps when
  below 1; 0: none), then 1 until the last `decay` share of the steps, where it falls linearly to 0 (1: from the first
  step).
- `clip`: the trained gradients' norm is clipped to it (0: unclipped).
- `micro_batches`: the batch is cut into this many parts, each with its own forward and backward, their gradients
  summed. A count, not a size: the step's graph is then the same at every batch.
- `store`: what E holds of each trained tensor. `full` is the tensor; `qB` is its delta from θ in B-bit codes
  (`codecs.ScaledQuant`), and a checkpoint's output is then the model those codes rebuild.
"""
from __future__ import annotations

import fnmatch

import torch
import torch.nn.functional as F

from training_slowdown.codecs import Identity, Plan, ScaledQuant
from training_slowdown.model import LM

from ._optimizers import Adafactor, Lion, Muon, Shampoo

VERSION = 2
DIALS = {"trainable": "*", "opt": "adamw", "lr": 3e-4, "weight_decay": 0.0, "warmup": 0.0, "decay": 1.0, "clip": 1.0,
         "micro_batches": 1, "store": "full"}
#: `opt`'s values that train every trained tensor, the embedding included
WHOLE = {"adamw": lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95), weight_decay=wd),
         "sgd": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd),
         "lion": lambda p, lr, wd: Lion(p, lr=lr, weight_decay=wd),
         "adafactor": lambda p, lr, wd: Adafactor(p, lr=lr, weight_decay=wd)}
#: and those that train the matrices, the gains then under AdamW, undecayed: neither has a rule for the embedding
MATRICES = {"muon": Muon, "shampoo": Shampoo}


def codec(dials: dict):
    store = dials["store"]
    if store == "full":
        return Identity()
    if store[:1] == "q" and store[1:].isdigit() and 2 <= int(store[1:]) <= 8:
        return ScaledQuant(int(store[1:]))
    raise ValueError(f"subset: store {store!r} is neither full nor q2 to q8")


def plan(dials: dict) -> Plan:
    if dials["opt"] not in WHOLE and dials["opt"] not in MATRICES:
        raise ValueError(f"subset: opt {dials['opt']!r} is none of {', '.join([*WHOLE, *MATRICES])}")
    if not 0 < dials["decay"] <= 1 or dials["warmup"] < 0 or dials["micro_batches"] < 1:
        raise ValueError("subset: decay is a share of the steps in (0, 1], warmup is not negative and micro_batches "
                         "is at least 1")
    return Plan([(p.strip(), codec(dials)) for p in dials["trainable"].split(",") if p.strip()])


def trained(model: torch.nn.Module, dials: dict) -> list[str]:
    patterns = [p.strip() for p in dials["trainable"].split(",") if p.strip()]
    names = [n for n, _ in model.named_parameters() if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if not names:
        raise SystemExit(f"subset: trainable {dials['trainable']!r} names no parameter of the start")
    return names


def build(shape: dict, dials: dict):
    plan(dials)
    model = LM.of(shape)
    names = set(trained(model, dials))
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    params = [p for n, p in model.named_parameters() if n in names]
    lr, wd = dials["lr"], dials["weight_decay"]
    if dials["opt"] in WHOLE:
        return model, [WHOLE[dials["opt"]](params, lr, wd)]
    if "embed" in names:
        raise SystemExit(f"subset: opt {dials['opt']} has no rule for the embedding; leave it out of trainable")
    matrices, gains = [p for p in params if p.ndim == 2], [p for p in params if p.ndim != 2]
    return model, ([MATRICES[dials["opt"]](matrices, lr=lr, weight_decay=wd)] if matrices else []) + (
        [torch.optim.AdamW(gains, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)] if gains else [])


def schedule(dials: dict, t: int, n: int) -> float:
    """The rate's factor at step `t` of `n`."""
    warm = dials["warmup"] * n if dials["warmup"] < 1 else dials["warmup"]
    return (min(1.0, (t + 1) / warm) if warm > 0 else 1.0) * min(1.0, (1 - t / n) / dials["decay"])


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = dials["lr"] * schedule(dials, t, n)
    for x in rows.chunk(dials["micro_batches"]):
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
