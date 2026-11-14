"""The optimizers a route names that torch has none of, or none a step with it can be captured from: each a published
rule in tensor ops, with no value read into Python, so a step under it is one fixed graph (`training_slowdown.capture`).
Not a route: a route imports it as a sibling (`from ._optimizers import Lion`).

All state is the optimizer's own, made at its first step, and every class takes its `defaults` as keyword arguments,
which is how the capture makes its copy.
"""
from __future__ import annotations

import torch

from ._codes import round_bf16, round_int8


def newton_schulz(g: torch.Tensor) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, the update orthogonalized by five Newton-Schulz steps in bf16, `weight_decay` decoupled.

    With `low_state` each parameter is a bf16 matrix and its state is low-width: the momentum is kept as int8 codes at
    one bf16 scale a row, decoded for the step and coded again stochastically (`_codes.round_int8`), and the
    parameter's update is rounded to bf16 stochastically (`_codes.round_bf16`). For each parameter in turn the step
    takes the buffer's random values, then the parameter's, from torch's global generator."""

    def __init__(self, params, lr: float, momentum: float = 0.95, weight_decay: float = 0.0, low_state: bool = False):
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
                         | ({"low_state": True} if low_state else {}))

    def load_state_dict(self, state_dict: dict) -> None:
        """torch's load, which casts each state tensor of a float parameter to the parameter's dtype, with the
        momentum's codes kept int8."""
        super().load_state_dict(state_dict)
        for st in self.state.values():
            if "codes" in st:
                st["codes"] = st["codes"].to(torch.int8)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            low = g.get("low_state", False)
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if low:
                    if "codes" not in st:
                        st["codes"] = torch.zeros_like(p, dtype=torch.int8)
                        st["scales"] = torch.zeros(p.shape[0], 1, dtype=torch.bfloat16, device=p.device)
                    buf = (st["codes"].float() * st["scales"].float()).mul_(g["momentum"]).add_(p.grad.float())
                    for held, new in zip((st["codes"], st["scales"]), round_int8(buf)):
                        held.copy_(new)
                else:
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p)
                    buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                alpha = -g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if low:
                    p.copy_(round_bf16(p.float().add_(u.float(), alpha=alpha)))
                else:
                    p.add_(u, alpha=alpha)


class Lion(torch.optim.Optimizer):
    """The sign of an average of the gradient and its momentum, the momentum under a second, slower average;
    `weight_decay` decoupled."""

    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            now, later = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.lerp(p.grad, st["m"], now).sign(), alpha=-g["lr"])
                st["m"].lerp_(p.grad, 1 - later)


class Adafactor(torch.optim.Optimizer):
    """`torch.optim.Adafactor`'s rule, its arguments and its defaults. torch's own takes the weight's norm and the
    update's as Python numbers, which no capture follows; here both stay tensors. A matrix keeps the row and column
    means of its squared gradients, a vector the squares themselves; the step is relative to the weight's RMS."""

    def __init__(self, params, lr: float = 1e-2, beta2_decay: float = -0.8, eps: tuple = (None, 1e-3), d: float = 1.0,
                 weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "beta2_decay": beta2_decay, "eps": eps, "d": d,
                                  "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    if p.ndim > 1:
                        st["row"] = p.new_zeros(*p.shape[:-1], 1)
                        st["col"] = p.new_zeros(*p.shape[:-2], 1, p.shape[-1])
                    else:
                        st["var"] = torch.zeros_like(p)
                st["step"] += 1
                floor = torch.finfo(p.dtype).eps if g["eps"][0] is None else g["eps"][0]
                mix, rate = st["step"] ** g["beta2_decay"], min(g["lr"], st["step"] ** -0.5)
                scale = (p.norm() / p.numel() ** 0.5).clamp(min=g["eps"][1]) * rate
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                if p.ndim > 1:
                    st["row"].lerp_(p.grad.square().mean(-1, keepdim=True), mix)
                    st["col"].lerp_(p.grad.square().mean(-2, keepdim=True), mix)
                    var = (st["row"] @ st["col"]) / st["row"].mean(-2, keepdim=True).clamp(min=floor)
                else:
                    var = st["var"].lerp_(p.grad.square(), mix)
                u = var.clamp(min=floor * floor).rsqrt() * p.grad
                p.sub_(u * (scale / (u.norm() / (u.numel() ** 0.5 * g["d"])).clamp(min=1.0)))


def inverse_fourth_root(a: torch.Tensor, floor: float) -> torch.Tensor:
    """A^(-1/4) of a symmetric matrix whose eigenvalues are at least `floor`."""
    values, vectors = torch.linalg.eigh(a)
    return (vectors * values.clamp(min=floor).pow(-0.25)) @ vectors.T


class Shampoo(torch.optim.Optimizer):
    """A matrix's gradient G between the inverse fourth roots of its two Kronecker factors, L = eps I + the sum of
    G Gᵀ and R = eps I + the sum of Gᵀ G, both roots taken at every step; `weight_decay` decoupled."""

    def __init__(self, params, lr: float, eps: float = 1e-4, weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["left"], st["right"] = (g["eps"] * torch.eye(n, dtype=p.dtype, device=p.device) for n in p.shape)
                st["left"].add_(p.grad @ p.grad.T)
                st["right"].add_(p.grad.T @ p.grad)
                u = inverse_fourth_root(st["left"], g["eps"]) @ p.grad @ inverse_fourth_root(st["right"], g["eps"])
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u, alpha=-g["lr"])
