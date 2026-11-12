"""Route `subset`: train the parameters `trainable` names (comma-separated patterns over the start's parameter names,
fnmatch, `blocks.5.*,norm`), the rest frozen.

- `opt`: the optimizer. Four train every one of them, the embedding included, under `weight_decay`: `adamw` (betas
  0.9, 0.95), `sgd` (torch's, Nesterov momentum 0.9, its decay added to the gradient), `lion` and `adafactor`
  (`_optimizers`; Adafactor's rate is at most 1/sqrt(t) and relative to each weight's RMS). Two train the matrices,
  `weight_decay` decoupled, beside AdamW (betas 0.9, 0.95), undecayed, on the gains: `muon` (Nesterov momentum 0.95,
  the update orthogonalized by five Newton-Schulz steps in bf16) and `shampoo` (the gradient between the inverse
  fourth roots of its two Kronecker factors), both in `_optimizers`.
- `embed`: under `muon` or `shampoo`, the embedding's optimizer: `adamw`, the AdamW beside it, with the gains; `opt`,
  the optimizer itself, the embedding one of its matrices.
- `lr`: the rate, times the schedule. `gain_lr` is the gains' own rate and `embed_lr` the embedding's (0: `lr`),
  under whichever optimizer trains them.
- `fused`: AdamW, the model's own or the one beside `muon` or `shampoo`, steps in torch's fused kernel, one op a
  group over every tensor of the group; false leaves the step to torch's default, which is never that kernel. Its
  groups are then one a kind of tensor (the embedding, another matrix, a gain), so the kernel runs once a kind
  whatever the rates; without it a group is a rate. The same update either way, in another order of rounding.
- `schedule`: a ramp over the first `warmup` steps (a share of the steps when below 1; 0: none), times `linear`, 1
  until the last `decay` share of the steps, where it falls linearly to 0 (1: from the first step); `cosine`, half a
  cosine from 1 at the first step to 0 after the last; or `constant`, 1.
- `clip`: the trained gradients' norm is clipped to it (0: unclipped).
- `micro_batches`: the batch is cut into this many parts, each with its own forward and backward, their gradients
  summed. A count, not a size: the step's graph is then the same at every batch.
- `store`: what E holds of each trained tensor. `full` is the tensor; `qB` is its delta from θ in B-bit codes
  (`codecs.ScaledQuant`), and a checkpoint's output is then the model those codes rebuild.
"""
from __future__ import annotations

import fnmatch
import math

import torch
import torch.nn.functional as F

from training_slowdown.codecs import Identity, Plan, ScaledQuant
from training_slowdown.model import LM

from ._optimizers import Adafactor, Lion, Muon, Shampoo

VERSION = 2
DIALS = {"trainable": "*", "opt": "adamw", "lr": 3e-4, "weight_decay": 0.0, "warmup": 0.0, "decay": 1.0, "clip": 1.0,
         "micro_batches": 1, "store": "full", "gain_lr": 0.0, "embed_lr": 0.0, "embed": "adamw", "schedule": "linear",
         "fused": False}
#: the dials whose default is the step without them (the route contract): a config that sets none digests without them
LATER = ("gain_lr", "embed_lr", "embed", "schedule", "fused")
SCHEDULES = ("linear", "cosine", "constant")


def adamw(groups: list[dict], dials: dict, weight_decay: float) -> torch.optim.AdamW:
    """AdamW over `groups`, in torch's fused kernel where `fused` says so. Torch takes a stated false as a choice
    too, and then steps a tensor at a time where its default would not, so a false dial states nothing."""
    return torch.optim.AdamW(groups, lr=dials["lr"], betas=(0.9, 0.95), weight_decay=weight_decay,
                             fused=True if dials["fused"] else None)


#: `opt`'s values that train every trained tensor, the embedding included
WHOLE = {"adamw": adamw,
         "sgd": lambda p, d, wd: torch.optim.SGD(p, lr=d["lr"], momentum=0.9, nesterov=True, weight_decay=wd),
         "lion": lambda p, d, wd: Lion(p, lr=d["lr"], weight_decay=wd),
         "adafactor": lambda p, d, wd: Adafactor(p, lr=d["lr"], weight_decay=wd)}
#: and those that train the matrices, beside AdamW, undecayed, on the gains (and the embedding, unless `embed` is `opt`)
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
    if dials["schedule"] not in SCHEDULES or dials["embed"] not in ("adamw", "opt"):
        raise ValueError(f"subset: schedule is one of {', '.join(SCHEDULES)} and embed is adamw or opt")
    if dials["gain_lr"] < 0 or dials["embed_lr"] < 0:
        raise ValueError("subset: gain_lr and embed_lr are rates, or 0 for lr")
    if dials["fused"] and dials["opt"] != "adamw" and dials["opt"] not in MATRICES:
        raise ValueError(f"subset: fused is AdamW's kernel, and opt {dials['opt']!r} steps no AdamW")
    return Plan([(p.strip(), codec(dials)) for p in dials["trainable"].split(",") if p.strip()])


def trained(model: torch.nn.Module, dials: dict) -> list[str]:
    patterns = [p.strip() for p in dials["trainable"].split(",") if p.strip()]
    names = [n for n, _ in model.named_parameters() if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if not names:
        raise SystemExit(f"subset: trainable {dials['trainable']!r} names no parameter of the start")
    return names


def groups(named: list, dials: dict, kinds: bool = False) -> list[dict]:
    """One optimizer's tensors as its groups, each in the model's order: the embedding at `embed_lr`, another matrix
    at `lr`, a gain at `gain_lr`. One group a rate, or with `kinds` one a kind of those three, two kinds at one rate
    then two groups. A group states its rate twice: `lr`, which `step` scales, and `initial_lr`, where torch's
    schedulers keep a group's own rate, which nothing writes after this."""
    out: dict = {}
    for n, p in named:
        kind = "embed_lr" if n == "embed" else "lr" if p.ndim == 2 else "gain_lr"
        rate = dials[kind] or dials["lr"]
        out.setdefault(kind if kinds else rate, (rate, []))[1].append(p)
    return [{"params": params, "lr": rate, "initial_lr": rate} for rate, params in out.values()]


def build(shape: dict, dials: dict):
    plan(dials)
    model = LM.of(shape)
    names = set(trained(model, dials))
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    named = [(n, p) for n, p in model.named_parameters() if n in names]
    lr, wd = dials["lr"], dials["weight_decay"]
    if dials["opt"] in WHOLE:
        return model, [WHOLE[dials["opt"]](groups(named, dials, dials["fused"]), dials, wd)]
    under_opt = {n for n, p in named if p.ndim == 2 and (n != "embed" or dials["embed"] == "opt")}
    matrices, beside = [x for x in named if x[0] in under_opt], [x for x in named if x[0] not in under_opt]
    return model, ([MATRICES[dials["opt"]](groups(matrices, dials), lr=lr, weight_decay=wd)] if matrices else []) + (
        [adamw(groups(beside, dials, dials["fused"]), dials, 0.0)] if beside else [])


def schedule(dials: dict, t: int, n: int) -> float:
    """The rate's factor at step `t` of `n`. A route that takes this step with no `schedule` dial has the linear one."""
    warm = dials["warmup"] * n if dials["warmup"] < 1 else dials["warmup"]
    ramp = min(1.0, (t + 1) / warm) if warm > 0 else 1.0
    kind = dials.get("schedule", "linear")
    if kind == "cosine":
        return ramp * 0.5 * (1 + math.cos(math.pi * t / n))
    return ramp * (1.0 if kind == "constant" else min(1.0, (1 - t / n) / dials["decay"]))


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    """One training step. Each group's rate is its own (`groups`), times the schedule. A group that states none is at
    `lr`: a route's one group built without `groups`, or the group of an optimizer's state saved before a group had
    a rate of its own, whose `lr` is the last step's scaled rate and never the group's."""
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = g.get("initial_lr", dials["lr"]) * schedule(dials, t, n)
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
