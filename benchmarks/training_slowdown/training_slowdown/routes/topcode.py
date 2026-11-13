"""Route `topcode`: the start grown by `grow` new top blocks and trained whole while the final norm's input crosses as
a code. New block i of n, at L + i above the start's L blocks, starts as a copy of the start's block ⌊iL/n⌋ with its
`wo` and `w_down` zero, so the grown model computes the start's function at step 0.

The code (`Code`): the top block's output is divided by its root mean square a token (which the final norm maps to
the same output), rotated by R, a fixed orthogonal matrix, and each token's coordinates rounded to nearest on
2^forward_bits levels over their largest magnitude; the codes cross as a uint8 plane, `forward_bits` bytes for every
eight coordinates (`codecs.wire.pack_codes`), beside the token's step in bf16, and are rotated back. The gradient
crosses back rotated by R, rounded stochastically onto 2^backward_bits − 1 steps whose ends are its largest magnitude
a token (rounded up a bf16 unit), as floats, and rotated back. R is a DCT-II's rows with each coordinate's sign drawn
from a generator seeded 1000 plus the grown depth, from the dials and the shape alone; `build` makes it once, a buffer
outside the state dict, so a resumed job keeps the one it began with. The stochastic rounding draws from torch's global
generator, which a job keeps across a resume.

- `grow`: the new top blocks, at least 1;
- `forward_bits`, `backward_bits`: the code's widths each way;
- `lr`: Muon's rate over every matrix of the blocks (Nesterov momentum 0.95, the update orthogonalized by five
  Newton-Schulz steps in bf16), `weight_decay` its decoupled decay;
- `norm_lr`: the gains' rate, the final norm's included, under one undecayed AdamW (betas 0.9, 0.95) in torch's
  fused kernel; the embedding is frozen;
- `warmup`, `decay`: `subset`'s linear schedule, the rate rising over `warmup` steps and falling to 0 over the last
  `decay` share of them;
- `micro_batches`: `subset`'s; the trained gradients' norm is clipped to 1.

The route reads no weight of the start: `origin` states which of the start's tensors each tensor of its model starts
as, from the dials alone, and the harness builds that model's start from θ (the route contract). E is every tensor
of the output that differs from that start, whole: the grown model, scored without the code.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from training_slowdown.codecs import Identity, Plan
from training_slowdown.codecs.wire import pack_codes, unpack_codes
from training_slowdown.model import LM

from ._optimizers import Muon
from .subset import schedule

VERSION = 1
DIALS = {"grow": 0, "forward_bits": 4, "backward_bits": 4, "lr": 3e-3, "norm_lr": 3e-3, "weight_decay": 0.0,
         "warmup": 40, "decay": 0.25, "micro_batches": 1}
#: a new block's tensors that start at zero: its two output projections, so the block adds nothing to the stream
ZEROED = ("wo.weight", "w_down.weight")
#: a block's tensors, by their names under `blocks.{j}.`
TENSORS = ("attn_norm", "mlp_norm", "wq.weight", "wk.weight", "wv.weight", "wo.weight", "w_gate.weight",
           "w_up.weight", "w_down.weight")
#: the widths the forward's code takes, each a whole number of bytes for every eight coordinates
FORWARD_BITS = range(4, 9)
#: and the backward's
BACKWARD_BITS = range(1, 17)
#: R's seed is this plus the grown model's depth
SEED = 1000


def plan(dials: dict) -> Plan:
    if dials["grow"] < 1:
        raise ValueError("topcode: grow is the new top blocks, at least 1: write it")
    if dials["forward_bits"] not in FORWARD_BITS or dials["backward_bits"] not in BACKWARD_BITS:
        raise ValueError(f"topcode: forward_bits is {FORWARD_BITS.start} to {FORWARD_BITS.stop - 1} and backward_bits "
                         f"{BACKWARD_BITS.start} to {BACKWARD_BITS.stop - 1}")
    if dials["lr"] <= 0 or dials["norm_lr"] < 0 or dials["weight_decay"] < 0:
        raise ValueError("topcode: lr is a rate, norm_lr a rate or 0, and weight_decay not negative")
    if dials["warmup"] < 0 or not 0 < dials["decay"] <= 1 or dials["micro_batches"] < 1:
        raise ValueError("topcode: warmup is not negative, decay a share of the steps in (0, 1] and micro_batches "
                         "at least 1")
    return Plan([("*", Identity())])


def sources(shape: dict, dials: dict) -> list[int]:
    """The start's block each new block copies, from the lowest new block up."""
    n, blocks = dials["grow"], shape["n_layer"]
    return [i * blocks // n for i in range(n)]


def origin(shape: dict, dials: dict) -> dict[str, str | None]:
    """Each new block's tensors: the start's tensor it starts as, or None for zeros. The start's blocks keep their
    names, so they map to themselves and aren't listed."""
    blocks = shape["n_layer"]
    return {f"blocks.{blocks + i}.{t}": None if t in ZEROED else f"blocks.{src}.{t}"
            for i, src in enumerate(sources(shape, dials)) for t in TENSORS}


def rotation(d: int, seed: int) -> torch.Tensor:
    """R, d × d and orthogonal: row k the DCT-II's k-th basis vector, column i's sign drawn from `seed`."""
    i = torch.arange(d, dtype=torch.float32)
    r = torch.stack([torch.cos(math.pi * (i + 0.5) * float(k) / d) for k in range(d)]) * math.sqrt(2.0 / d)
    r[0] = r[0] / math.sqrt(2.0)
    signs = torch.randint(0, 2, (d,), generator=torch.Generator().manual_seed(seed)).float() * 2 - 1
    return r * signs


def nearest(z: torch.Tensor, bits: int) -> torch.Tensor:
    """Each token's coordinates rounded to nearest on 2^bits levels, midpoints of steps of 2/2^bits of their largest
    magnitude (in bf16): the codes cross as bytes (`pack_codes`), and come back as the levels' values."""
    step = (z.abs().amax(-1, keepdim=True) * (2.0 / 2 ** bits)).to(torch.bfloat16).float().clamp_min(1e-30)
    half = 2 ** (bits - 1)
    codes = (torch.floor(z / step).clamp(-half, half - 1) + half).long()
    return (unpack_codes(pack_codes(codes, bits), bits).float() - half + 0.5) * step


def stochastic(g: torch.Tensor, bits: int) -> torch.Tensor:
    """Each token's coordinates rounded stochastically onto 2^bits − 1 steps from −top to top, top their largest
    magnitude rounded up a bf16 unit, the draw torch's global generator's."""
    top = (g.abs().amax(-1, keepdim=True) * (1 + 2 ** -7)).to(torch.bfloat16).float().clamp_min(1e-30)
    step = 2 * top / (2 ** bits - 1)
    return torch.floor((g + top) / step + torch.rand_like(g)).clamp(0, 2 ** bits - 1) * step - top


class Code(torch.autograd.Function):
    """The final norm's input as it crosses: `nearest` in R's basis forward, `stochastic` in it backward, in float32
    under any autocast."""

    @staticmethod
    def forward(ctx, x, r, bits, grad_bits):
        ctx.save_for_backward(r)
        ctx.grad_bits = grad_bits
        with torch.autocast(x.device.type, enabled=False):
            return (nearest(x.float() @ r.T, bits) @ r).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (r,) = ctx.saved_tensors
        with torch.autocast(g.device.type, enabled=False):
            return (stochastic(g.float() @ r.T, ctx.grad_bits) @ r).to(g.dtype), None, None, None


def unit(x: torch.Tensor) -> torch.Tensor:
    """Each token's stream divided by its root mean square, in float32."""
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-12)).to(x.dtype)


def build(shape: dict, dials: dict):
    plan(dials)
    depth = shape["n_layer"] + dials["grow"]
    model = LM.of(shape | {"n_layer": depth})
    model.register_buffer("rotation", rotation(shape["d_model"], SEED + depth), persistent=False)
    model.embed.requires_grad_(False)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gains = [{"params": [p for _, p in named if p.ndim == 1], "lr": dials["norm_lr"], "initial_lr": dials["norm_lr"]}]
    return model, [Muon([p for _, p in named if p.ndim == 2], lr=dials["lr"], weight_decay=dials["weight_decay"]),
                   torch.optim.AdamW(gains, betas=(0.9, 0.95), weight_decay=0.0, fused=True)]


def step(model, rows, optimizers, dials: dict, t: int, n: int) -> None:
    """`subset`'s step, the top block's output crossing as the code before the final norm and the tied head."""
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = g.get("initial_lr", dials["lr"]) * schedule(dials, t, n)
    for x in rows.chunk(dials["micro_batches"]):
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            h = Code.apply(unit(model.hidden(x[:, :-1])), model.rotation, dials["forward_bits"],
                           dials["backward_bits"])
            logits = model.top(h, len(model.blocks))
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        (loss * (len(x) / len(rows))).backward()
    torch.nn.utils.clip_grad_norm_([p for o in optimizers for g in o.param_groups for p in g["params"]], 1.0)
    for o in optimizers:
        o.step()
        o.zero_grad(set_to_none=True)
