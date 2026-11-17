"""Route `ema`: `subset`'s step, and the output model an exponential average of the trained weights.

- every dial of `subset`, with its meaning there;
- `ema_decay`: at each step the average keeps this share of itself and takes the rest from the weights the step
  wrote. It starts as the weights after the first step (`torch.optim.swa_utils.AveragedModel` under
  `get_ema_multi_avg_fn`);
- `ema_horizon` (0: `ema_decay` at every step): the share of the steps so far the average reaches back over. At step
  t, the first being 0, the average keeps 1 - 1/max(1, `ema_horizon`·(t + 1)) of itself: nothing until that share
  of the steps is a whole step, so until then it is the weights themselves, and more of itself the longer the run.
  A config writes this or `ema_decay` (`EITHER`): over a horizon the route reads no `ema_decay`.

The average of each trained tensor is a buffer of the model, outside its state: the route carries it from step to
step, the step still reads the weights themselves, and a checkpoint's output takes each average in its tensor's
place (`output`), which computes nothing. E is the averages, encoded as `subset`'s `store` says.
"""
from __future__ import annotations

import torch

from . import subset

VERSION = 1
DIALS = subset.DIALS | {"ema_decay": 0.99, "ema_horizon": 0.0}
LATER = (*subset.LATER, "ema_horizon")
EITHER = (("ema_decay", "ema_horizon"),)


def plan(dials: dict):
    if not 0 <= dials["ema_decay"] < 1:
        raise ValueError("ema: ema_decay is a share in [0, 1)")
    if not 0 <= dials["ema_horizon"] <= 1:
        raise ValueError("ema: ema_horizon is a share of the steps so far, in [0, 1]")
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


def keeps(dials: dict, t: int) -> float:
    """The share of itself the average keeps at step `t`."""
    if dials["ema_horizon"]:
        return 1 - 1 / max(1.0, dials["ema_horizon"] * (t + 1))
    return dials["ema_decay"]


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    subset.step(model, rows, optimizers, dials, t, n)
    with torch.no_grad():
        for name, average in averages(model, dials).items():
            if t == 0 and not dials["ema_horizon"]:
                average.copy_(model.get_parameter(name))
            else:
                average.lerp_(model.get_parameter(name), 1 - keeps(dials, t))


def output(model, dials: dict) -> dict[str, torch.Tensor]:
    return model.state_dict() | averages(model, dials)
