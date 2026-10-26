"""`value_check._cls` names a module by vLLM's layer class: the first tapped class in its MRO, else ``Linear`` for any vLLM linear
layer, else its own class name; and exactly the pure-forward groups re-execute.  Pinned to the literal groups the check has used,
so the tables living in `engine.profiles.layer_classes` change no `by_class` key of a recorded `value_check.json`."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verity_vllm.check import value_check as vc
from verity_vllm.engine.profiles import layer_classes as LC


def _module(*mro: str) -> object:
    """An instance whose class MRO is ``mro`` (most derived first), as vLLM's layer classes subclass one another."""
    cls = object
    for name in reversed(mro):
        cls = type(name, (cls,), {})
    return cls()


@pytest.mark.parametrize(("mro", "group"), [
    (("RMSNorm",), "RMSNorm"),
    (("GemmaRMSNorm", "CustomOp"), "GemmaRMSNorm"),
    (("SiluAndMul",), "SiluAndMul"),
    (("GeluAndMul",), "GeluAndMul"),
    (("Llama3RotaryEmbedding", "RotaryEmbedding"), "RotaryEmbedding"),
    (("Attention",), "Attention"),
    (("ParallelLMHead", "VocabParallelEmbedding"), "VocabParallelEmbedding"),
    (("LogitsProcessor",), "LogitsProcessor"),
    (("QKVParallelLinear", "ColumnParallelLinear", "LinearBase"), "Linear"),
    (("MergedColumnParallelLinear", "ColumnParallelLinear", "LinearBase"), "Linear"),
    (("RowParallelLinear", "LinearBase"), "Linear"),
    (("ReplicatedLinear", "LinearBase"), "Linear"),
    (("RMSNorm", "LinearBase"), "RMSNorm"),
    (("Linear",), "Linear"),
    (("FusedMoE",), "FusedMoE"),
])
def test_group_of_each_layer_class(mro: tuple[str, ...], group: str) -> None:
    assert vc._cls(_module(*mro)) == group


def test_reexec_groups_and_sampler() -> None:
    assert LC.REEXEC == ("Linear", "RMSNorm", "GemmaRMSNorm", "SiluAndMul", "GeluAndMul", "RotaryEmbedding")
    assert LC.SAMPLER == "Sampler"
