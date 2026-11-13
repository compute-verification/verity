"""Route `topcode`: the start grown by `grow` new top blocks and trained whole while the final norm's input crosses as
a code. New block i of n, at L + i above the start's L blocks, starts as a copy of the start's block ⌊iL/n⌋ with its
`wo` and `w_down` zero, so the grown model computes the start's function at step 0.

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

from training_slowdown.codecs import Identity, Plan

VERSION = 1
DIALS = {"grow": 0, "forward_bits": 4, "backward_bits": 4, "lr": 3e-3, "norm_lr": 3e-3, "weight_decay": 0.0,
         "warmup": 40, "decay": 0.25, "micro_batches": 1}
#: a new block's tensors that start at zero: its two output projections, so the block adds nothing to the stream
ZEROED = ("wo.weight", "w_down.weight")
#: a block's tensors, by their names under `blocks.{j}.`
TENSORS = ("attn_norm", "mlp_norm", "wq.weight", "wk.weight", "wv.weight", "wo.weight", "w_gate.weight",
           "w_up.weight", "w_down.weight")
#: the widths the forward's code takes: each token's coordinates on 2^bits levels over its largest magnitude, packed
#: eight codes to `bits` bytes
FORWARD_BITS = range(4, 9)
#: and the backward's, on the stochastic grid whose end levels are the gradient's largest magnitude
BACKWARD_BITS = range(1, 17)


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
