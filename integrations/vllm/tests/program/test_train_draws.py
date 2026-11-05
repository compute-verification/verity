"""The training rules' random draws (`rules.training_draws`): each draw a field of the Program's `run` input beside the generator's
seed and offset, in graph order; the Program run on what torch CPU drew against torch's arithmetic on the same draws, with and
without a compute capability; each draw form; a graph without draws has no `run` input; and the draws the rules leave refused."""
from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_quant import BOTH, GPU, _same, _words
from verity.ml.definitions.numerics.serve3_reference import materialize
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity_vllm.program.frontend.rules.training_draws import SEED

aten = torch.ops.aten
X = torch.randn(4, 16, generator=torch.Generator().manual_seed(1))


def _derive(f, *xs, rules=GPU):
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(*xs)
    return derive_graph(gm, xs, mode="reference", rules=rules, name="T", input_names=[f"x{i}" for i in range(len(xs))])


def _step(x, draws):
    a, b, c, i, p = draws
    return x + a, x * b, c - x, x[0] * i, x[0] * p


def _drawn(x, g):
    """What `_traced` draws, from the generator g."""
    return (torch.randn(x.shape, generator=g), torch.rand(x.shape, generator=g),
            torch.empty(4, 16).normal_(0.0, 0.5, generator=g), torch.randint(0, 7, (16,), generator=g),
            torch.randperm(16, generator=g))


def _traced(x):
    return _step(x, (torch.randn(x.shape), torch.rand_like(x), torch.empty(4, 16).normal_(0.0, 0.5), torch.randint(0, 7, (16,)),
                     torch.randperm(16)))


@BOTH
def test_a_draw_is_an_input_committed_with_its_seed(rules) -> None:
    """Five draws, each a field of `run` in graph order with the generator's (seed, offset) after it; the Program on torch's draws
    and arbitrary seed words is torch's arithmetic on the same draws."""
    rep = _derive(_traced, X, rules=rules)
    assert rep.program is not None, rep.unsupported
    run = rep.provenance["run_inputs"]
    names = ["randn", "rand_like", "normal_functional", "randint", "randperm"]
    assert [f["name"] for f in run["fields"]] == [f"draw{k}{s}:{n}" for k, n in enumerate(names) for s in ("", ".seed")]
    assert [f["leaves"] for f in run["fields"]] == [64, 2, 64, 2, 64, 2, 16, 2, 16, 2] and run["fields"][1]["type"] == repr(SEED)
    draws = _drawn(X, torch.Generator().manual_seed(2))
    words = [w for k, d in enumerate(draws) for w in _words(d) + [0x5EED0000 + k, 4 * k]]
    tab = materialize(rep.program, [_words(X), words])
    _same([tab[g] for g in rep.program.circuit.root_outputs()], [w for t in _step(X, draws) for w in _words(t)])


@pytest.mark.parametrize("draw", [
    lambda x: aten.randn.generator([4, 16], generator=None),
    lambda x: aten.rand.generator([4, 16], generator=None),
    lambda x: aten.randint.low_generator(0, 7, [4, 16], generator=None),
    lambda x: aten.randint.generator(7, [4, 16], generator=None),
    lambda x: aten.randperm.generator(16, generator=None),
    lambda x: torch.randn_like(x),
    lambda x: torch.randint_like(x, 3, 9),
    lambda x: torch.normal(0.0, 2.0, (4, 16)),
    lambda x: torch.empty(4, 16).uniform_(-1.0, 1.0),
    lambda x: torch.empty(4, 16).exponential_(),
    lambda x: torch.empty(4, 16).bernoulli_(0.3),
    lambda x: torch.randn(4, 16, dtype=torch.bfloat16),
], ids=["randn-gen", "rand-gen", "randint-low-gen", "randint-gen", "randperm-gen", "randn_like", "randint_like", "normal",
        "uniform_", "exponential_", "bernoulli_", "randn-bf16"])
def test_each_draw_form(draw) -> None:
    """Each form, used in an f32 product, is one field of its dtype and shape, and its seed."""
    rep = _derive(lambda x: (x * draw(x),), X)
    assert rep.program is not None, rep.unsupported
    fields = rep.provenance["run_inputs"]["fields"]
    assert [f["leaves"] for f in fields] == [draw(X).numel(), 2] and fields[1]["name"].startswith("draw0.seed:"), fields


def test_a_step_without_draws_has_no_run_input() -> None:
    rep = _derive(lambda x: (x * 2.0,), X)
    assert rep.program is not None and "run_inputs" not in rep.provenance


@pytest.mark.parametrize("f, rules", [
    (lambda x: (x + torch.randn(x.shape),), DEFAULT_RULES),
    (lambda x: (x + torch.bernoulli(torch.sigmoid(x)),), GPU),
    (lambda x: (x + torch.normal(x, 1.0),), GPU),
], ids=["default-rules", "bernoulli-of-a-value", "normal-of-a-value"])
def test_draws_left_refused(f, rules) -> None:
    """The default rules (serving) keep refusing a draw, and the training rules a draw whose distribution reads a value."""
    u = _derive(f, X, rules=rules).unsupported
    assert isinstance(u, Unsupported) and u.reason == "side_input" and "random" in u.hint, u
