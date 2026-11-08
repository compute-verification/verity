"""A training step's random ops (`rules.training`): each one's values are an input of the step, committed with its generator's
seed.

torch's CUDA samples are curand's Philox4x32-10, keyed by the generator's seed and advanced by its offset, read in the launch's
grid-stride order through torch's uniform, normal (Box-Muller) and integer transforms. No Definition computes that stream exactly,
so a random op is not computed: what the run drew is a field of the Program's `run` input, and beside it the generator's state
it claims to have sampled from, so the record commits each op's values with its seed. Nothing checks the values against the seed
until a Definition of the GPU's generator exists.

    Random  rand / randn / randint / randperm, their generator forms, rand_like / randn_like / randint_like, normal with
            number arguments, and the functional normal_, uniform_, exponential_ and bernoulli_ (with a number p)
"""

from __future__ import annotations

import torch

from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.rules.training_emit import _TrainRule, aten
from verity_vllm.program.frontend.rules.training_views import _info
from verity_vllm.program.frontend.value_types import type_of_tensor

#: each random op reads of its tensor operand, if it has one, only the shape
_DRAWS = {aten.rand.default, aten.rand.generator, aten.randn.default, aten.randn.generator, aten.randint.default,
          aten.randint.generator, aten.randint.low, aten.randint.low_generator, aten.randperm.default, aten.randperm.generator,
          aten.rand_like.default, aten.randn_like.default, aten.randint_like.default, aten.randint_like.low_dtype,
          aten.normal.float_float, aten.normal_functional.default, aten.uniform.default, aten.exponential.default,
          aten.bernoulli.p}
#: the generator's state before a random op: its seed and its Philox offset
SEED = Array(2, Value(64))


class Random(_TrainRule):
    """A random op's values as an input of the step: the field ``draw<k>:<node>`` of the `run` input, and the generator's seed
    and Philox offset before it the field ``draw<k>.seed:<node>``, k counting the step's random ops in graph order. No gates."""
    info = _info("Random", "aten rand/randn/randint/randperm (and generator forms), rand_like/randn_like/randint_like, "
                 "normal.float_float, normal_functional, uniform, exponential, bernoulli.p", "the op reads no operand's values",
                 "values and the generator's (seed, offset): fields of the run input, no gates")

    def targets(self):
        return _DRAWS

    def run_inputs(self, ep, mode):
        return any(n.op == "call_function" and n.target in _DRAWS for n in ep.graph.nodes)

    def applies(self, node, ctx):
        return True if isinstance(node.meta.get("val"), torch.Tensor) else "the op's result is not one tensor"

    def emit(self, node, ctx):
        k = ctx.state.get("draws", 0)
        ctx.state["draws"] = k + 1
        meta = node.meta["val"]
        values = ctx.run_value(f"draw{k}:{node.name}", type_of_tensor(meta), node)
        ctx.run_value(f"draw{k}.seed:{node.name}", SEED, node)
        ctx.define(node, values)
        ctx.record(self, node, {"draw": k, "dtype": str(meta.dtype), "shape": list(meta.shape)}, emitted="run input")


__all__ = ["Random", "SEED"]
