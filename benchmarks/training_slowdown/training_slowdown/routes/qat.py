"""Route `qat`: quantization-aware training. Every weight of the start is frozen but the matrices `targets` names,
and each of those is, in every forward, its start plus the sparse ternary rounding of a latent
(`codecs.SparseTernary`): W = θ + Q(V), V zero at first, so the first step's model is θ. W's gradient is taken as
V's, straight through the rounding, and Muon steps V.

- `targets`: comma-separated patterns over the start's parameter names (fnmatch), naming matrices of the blocks
  (`blocks.[45].w*.weight` is the top two blocks' seven of a six-block start; the default is every block's);
- `keep`: the share of a group's entries the rounding leaves nonzero; `group`: the entries one scale covers (0: a
  matrix has one scale);
- `lr`, `warmup`, `decay`, `clip`, `micro_batches`: `subset`'s, the optimizer Muon over the latents; `cooldown`: how
  the rate falls over the last `decay` share of the steps, `linear`, or `sqrt` for 1 - sqrt of the fall's progress.

A step decodes each matrix from the codes the step before left (`SparseTernary.apply` on θ's own copy of the
matrix; the first step's codes are zero, so it reads θ), runs its forwards and backwards on that, steps the latents,
and rounds them into the next step's codes. So one step hands the next each matrix's codes, two bool planes and its
scales, which is all the next forward reads of the training; the latents and Muon's momentum, which it hands on
too, only the update rule reads.

The latents are parameters θ has none of, and θ's copies and the codes are buffers no state dict holds: θ's load
leaves them as built, but for each copy, which takes θ's matrix as loaded, in θ's own dtype. A kept state's load,
which holds the latents, leaves each copy the tensor θ's load gave, the one the counter holds, for the harness to
write the kept copy into in place (`Harness.resume`). The output hands over
the last codes in place of the matrices (`output`): E is the library's packing of them, and the model a checkpoint
scores is what the library decodes from E on θ, the weights the next step would train on.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from training_slowdown.codecs import Plan, SparseTernary
from training_slowdown.model import LM, Linear

from . import lora
from ._optimizers import Muon

VERSION = 1
DIALS = {"targets": "blocks.*.weight", "keep": 0.5, "group": 0, "lr": 1.5e-3, "warmup": 0.0, "decay": 1.0,
         "cooldown": "linear", "clip": 1.0, "micro_batches": 1}
LATENT = ".v"
#: `cooldown`'s values: the rate's factor at the share `left` of the fall still to go
COOLDOWNS = {"linear": lambda left: left, "sqrt": lambda left: 1.0 - math.sqrt(1.0 - left)}


def plan(dials: dict) -> Plan:
    if not 0 < dials["decay"] <= 1 or dials["warmup"] < 0 or dials["micro_batches"] < 1:
        raise ValueError("qat: decay is a share of the steps in (0, 1], warmup is not negative and micro_batches is "
                         "at least 1")
    if dials["cooldown"] not in COOLDOWNS:
        raise ValueError(f"qat: cooldown {dials['cooldown']!r} is none of {', '.join(COOLDOWNS)}")
    return Plan([(p, SparseTernary(dials["keep"], dials["group"])) for p in lora.patterns(dials)])


def coded(model: torch.nn.Module, dials: dict) -> dict[str, Linear]:
    """Each coded matrix's layer, by the matrix's name."""
    return lora.adapted(model, dials)


def start_as_loaded(layer: Linear, state: dict, prefix: str, *_) -> None:
    if (w := state.get(prefix + "weight")) is not None and prefix + LATENT[1:] not in state:
        layer.start = w.detach().clone()


def latents_are_new(model, incompatible) -> None:
    incompatible.missing_keys[:] = [k for k in incompatible.missing_keys if not k.endswith(LATENT)]


def build(shape: dict, dials: dict):
    plan(dials)
    model = LM.of(shape).requires_grad_(False)
    layers = coded(model, dials)
    if not layers:
        raise SystemExit(f"qat: targets {dials['targets']!r} names no matrix of the start")
    for name, layer in layers.items():
        w = layer.weight.requires_grad_(True)
        if w.numel() % (dials["group"] or w.numel()):
            raise SystemExit(f"qat: groups of {dials['group']} don't divide {name}'s {w.numel()} entries")
        layer.v = torch.nn.Parameter(torch.zeros_like(w))
        for buffer, value in (("start", w.detach().clone()),
                              ("scales", torch.zeros(w.numel() // (dials["group"] or w.numel()), dtype=torch.float16)),
                              ("plus", torch.zeros(w.shape, dtype=torch.bool)),
                              ("minus", torch.zeros(w.shape, dtype=torch.bool))):
            layer.register_buffer(buffer, value, persistent=False)
        layer.register_load_state_dict_pre_hook(start_as_loaded)
    model.register_load_state_dict_post_hook(latents_are_new)
    return model, [Muon([layer.v for layer in layers.values()], lr=dials["lr"])]


def schedule(dials: dict, t: int, n: int) -> float:
    """The rate's factor at step `t` of `n`."""
    warm = dials["warmup"] * n if dials["warmup"] < 1 else dials["warmup"]
    left = min(1.0, (1 - t / n) / dials["decay"])
    return (min(1.0, (t + 1) / warm) if warm > 0 else 1.0) * COOLDOWNS[dials["cooldown"]](left)


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    layers = list(coded(model, dials).values())
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = dials["lr"] * schedule(dials, t, n)
    with torch.no_grad():
        for layer in layers:
            layer.weight.copy_(SparseTernary.apply(layer.start, layer.scales, layer.plus, layer.minus))
    for x in rows.chunk(dials["micro_batches"]):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        (loss * (len(x) / len(rows))).backward()
    for layer in layers:
        layer.v.grad, layer.weight.grad = layer.weight.grad, None
    if dials["clip"]:
        torch.nn.utils.clip_grad_norm_([layer.v for layer in layers], dials["clip"])
    for o in optimizers:
        o.step()
        o.zero_grad(set_to_none=True)
    with torch.no_grad():
        for layer in layers:
            codes = SparseTernary.round(layer.v, dials["keep"], dials["group"])
            for held, new in zip((layer.scales, layer.plus, layer.minus), codes):
                held.copy_(new)


def output(model, dials: dict) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    state = {k: v for k, v in model.state_dict().items() if not k.endswith(LATENT)}
    return state | {name: (layer.scales, layer.plus, layer.minus) for name, layer in coded(model, dials).items()}
