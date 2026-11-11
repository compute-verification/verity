"""Route `ema`: `subset`'s step, and the output model an exponential average of the trained weights.

- every dial of `subset`, with its meaning there;
- `ema_decay`: at each step the average keeps this share of itself and takes the rest from the weights the step
  wrote. It starts as the weights after the first step (`torch.optim.swa_utils.AveragedModel` under
  `get_ema_multi_avg_fn`).

The average of each trained tensor is a buffer of the model, outside its state: the route carries it from step to
step, the step still reads the weights themselves, and a checkpoint's output takes each average in its tensor's
place (`output`), which computes nothing. E is the averages, encoded as `subset`'s `store` says.
"""
from __future__ import annotations

import torch

from . import subset

VERSION = 1
DIALS = subset.DIALS | {"ema_decay": 0.99}
LATER = subset.LATER


def plan(dials: dict):
    if not 0 <= dials["ema_decay"] < 1:
        raise ValueError("ema: ema_decay is a share in [0, 1)")
    return subset.plan(dials)


def averages(model: torch.nn.Module, dials: dict) -> dict[str, torch.Tensor]:
    """Each trained tensor's average, by the tensor's name."""
    return {n: model.get_buffer(f"average_{i}") for i, n in enumerate(subset.trained(model, dials))}


def build(shape: dict, dials: dict):
    plan(dials)
    model, optimizers = subset.build(shape, dials)
    for i, n in enumerate(subset.trained(model, dials)):
        model.register_buffer(f"average_{i}", torch.zeros_like(model.get_parameter(n)), persistent=False)
    return model, optimizers


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    subset.step(model, rows, optimizers, dials, t, n)
    with torch.no_grad():
        for name, average in averages(model, dials).items():
            if t == 0:
                average.copy_(model.get_parameter(name))
            else:
                average.lerp_(model.get_parameter(name), 1 - dials["ema_decay"])


def output(model, dials: dict) -> dict[str, torch.Tensor]:
    return model.state_dict() | averages(model, dials)
