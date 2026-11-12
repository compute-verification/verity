"""`verity_vllm.program.frontend.rules.reference.ElementwiseRefRule._scalar_weight_value`: the value of a 0-d weight operand,
which the rule records beside the Program and the Program never reads. It is the source module's where the module holds the
field and torch hands its one number back, and None otherwise."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verity_vllm.program.frontend.rules.reference import ElementwiseRefRule


def _source(**buffers: torch.Tensor) -> torch.nn.Module:
    module = torch.nn.Module()
    for name, t in buffers.items():
        module.register_buffer(name, t)
    return module


def _value(module, name: str = "scale", aliases: tuple[str, ...] = ()):
    return ElementwiseRefRule()._scalar_weight_value(SimpleNamespace(module=module), SimpleNamespace(name=name, aliases=aliases))


def test_the_value_is_the_source_modules_under_the_weights_name_or_an_alias() -> None:
    module = _source(scale=torch.tensor(0.5))
    assert _value(module) == 0.5
    assert _value(module, "embed_scale", ("scale",)) == 0.5


@pytest.mark.parametrize("module", [None, _source(scale=torch.ones(2)), _source(scale=torch.empty((), device="meta")), _source()],
                         ids=["no source module", "more than one number", "a meta tensor", "no such field"])
def test_the_value_is_none_where_the_source_cannot_give_it(module) -> None:
    assert _value(module) is None
