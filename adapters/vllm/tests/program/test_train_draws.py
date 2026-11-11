"""A training step's random ops, which no rule lowers until a Definition of torch's CUDA generator computes what it draws: the
training rules, with and without a compute capability, refuse each form at its own node, and a survey (`rules.training_survey`)
makes each one a hole, in graph order, and derives the rest."""
from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_quant import BOTH
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training_survey import survey

aten = torch.ops.aten
X = torch.randn(4, 16, generator=torch.Generator().manual_seed(1))
#: the ATen ops that read torch's generator
RANDOM = {"randn", "rand", "randint", "randperm", "randn_like", "rand_like", "randint_like", "normal", "normal_functional",
          "uniform", "exponential", "bernoulli"}
FORMS = {
    "randn-gen": lambda x: aten.randn.generator([4, 16], generator=None),
    "rand-gen": lambda x: aten.rand.generator([4, 16], generator=None),
    "randint-low-gen": lambda x: aten.randint.low_generator(0, 7, [4, 16], generator=None),
    "randint-gen": lambda x: aten.randint.generator(7, [4, 16], generator=None),
    "randperm-gen": lambda x: aten.randperm.generator(16, generator=None),
    "randn_like": lambda x: torch.randn_like(x),
    "randint_like": lambda x: torch.randint_like(x, 3, 9),
    "normal": lambda x: torch.normal(0.0, 2.0, (4, 16)),
    "uniform_": lambda x: torch.empty(4, 16).uniform_(-1.0, 1.0),
    "exponential_": lambda x: torch.empty(4, 16).exponential_(),
    "bernoulli_": lambda x: torch.empty(4, 16).bernoulli_(0.3),
    "randn-bf16": lambda x: torch.randn(4, 16, dtype=torch.bfloat16),
    "bernoulli-of-a-value": lambda x: torch.bernoulli(torch.sigmoid(x)),
    "normal-of-a-value": lambda x: torch.normal(x, 1.0),
}


def _graph(f) -> torch.fx.GraphModule:
    return make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(X)


def _traced(x):
    return (x + torch.randn(x.shape), x * torch.rand_like(x), torch.empty(4, 16).normal_(0.0, 0.5) - x,
            x[0] * torch.randint(0, 7, (16,)), x[0] * torch.randperm(16))


@BOTH
@pytest.mark.parametrize("form", FORMS.values(), ids=FORMS.keys())
def test_each_random_op_is_refused(form, rules) -> None:
    """Each form, used in an f32 product, is refused at its own node."""
    gm = _graph(lambda x: (x * form(x),))
    op = next(n for n in gm.graph.nodes if n.op == "call_function" and n.target.__name__.split(".")[0] in RANDOM)
    rep = derive_graph(gm, (X,), mode="reference", rules=rules, name="T", input_names=["x0"])
    assert rep.program is None and isinstance(rep.unsupported, Unsupported), rep.unsupported
    assert rep.unsupported.node == f"{op.name}: {op.target}", rep.unsupported


@BOTH
def test_a_survey_makes_each_random_op_a_hole(rules) -> None:
    """Five random ops in one step: five holes in graph order, and the Program of the rest marked with them."""
    holes, program = survey(_graph(_traced), (X,), mode="reference", rules=rules, name="T", input_names=["x0"])
    assert [h["node"] for h in holes] == ["randn", "rand_like", "normal_functional", "randint", "randperm"], holes
    assert not isinstance(program, Unsupported) and program.annotations["survey_holes"] == holes, program
