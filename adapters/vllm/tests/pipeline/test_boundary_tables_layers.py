"""`boundary_tables`' layers: `layers_of`, the history tables' layers from a serve's configuration (a count or the layers
themselves), and `mla_layers`, the model's layers a table can be of (each MLA layer the engine bound a bf16 cache of latent
rows to)."""
from __future__ import annotations

import re
from typing import Any

import pytest
import torch

from verity_vllm.pipeline import boundary_tables as BT

NAME = "model.layers.0.self_attn.attn"
#: blocks, block size, words: a bound cache's shape
SHAPE = (2, 4, BT.WORDS)


def test_a_count_is_the_first_that_many_layers():
    assert BT.layers_of(3) == (0, 1, 2)
    assert BT.layers_of(0) == ()


def test_the_layers_themselves_are_kept_in_their_order():
    assert BT.layers_of([3, 61]) == (3, 61)
    assert BT.layers_of((61, 3)) == (61, 3)
    assert BT.layers_of([]) == ()


@pytest.mark.parametrize("flag", [True, False])
def test_a_flag_is_neither_a_count_nor_the_layers(flag):
    with pytest.raises(TypeError, match="a count or the layers themselves, and a bool is neither"):
        BT.layers_of(flag)


def layer(name: Any = NAME, cache: Any = None, rank: int = BT.WORDS - 64, rope: int | None = 64,
          dtype: str | None = None) -> torch.nn.Module:
    """A module as the engine leaves an MLA layer: its layer name, its KV cache (bf16 of `SHAPE` unless given), its latent's
    rank and, unless None, its rope dimension and its cache's configured dtype."""
    m = torch.nn.Module()
    m.layer_name, m.kv_lora_rank = name, rank
    m.kv_cache = torch.zeros(SHAPE, dtype=torch.bfloat16) if cache is None else cache
    for k, v in (("qk_rope_head_dim", rope), ("kv_cache_dtype", dtype)):
        if v is not None:
            setattr(m, k, v)
    return m


def model(*layers: torch.nn.Module) -> torch.nn.Module:
    return torch.nn.Sequential(*layers)


@pytest.mark.parametrize("attrs", [{}, {"dtype": "auto"}, {"dtype": "bfloat16"}, {"rank": BT.WORDS, "rope": None}])
def test_an_mla_layer_is_one_whose_bf16_cache_holds_the_latents_words_a_row(attrs):
    m = layer(**attrs)
    got = BT.mla_layers(model(m))
    assert list(got) == [0] and got[0][0] == NAME and got[0][1] is m.kv_cache


@pytest.mark.parametrize("attrs, cache", [
    ({"cache": torch.zeros(SHAPE, dtype=torch.int16)}, "(auto, torch.int16, (2, 4, 576))"),
    ({"rank": BT.WORDS}, "(auto, torch.bfloat16, (2, 4, 576))"),
    ({"cache": torch.zeros(2, 4, BT.WORDS + 1, dtype=torch.bfloat16)}, "(auto, torch.bfloat16, (2, 4, 577))"),
    ({"cache": torch.zeros(2, 2, 4, BT.WORDS, dtype=torch.bfloat16)}, "(auto, torch.bfloat16, (2, 2, 4, 576))"),
])
def test_a_cache_is_refused_for_its_words_type_its_latents_width_its_rows_length_or_its_dimensions_alone(attrs, cache):
    why = (f"MLA layer {NAME}: its cache {cache} doesn't hold DsaLatentRow_v1's 576 bf16 words a row; serve with "
           f"--kv-cache-dtype auto")
    with pytest.raises(ValueError, match=f"^{re.escape(why)}$"):
        BT.mla_layers(model(layer(**attrs)))


def test_two_mla_layers_of_one_index_are_refused_by_both_names():
    a, b = "model.layers.2.self_attn.attn", "model.layers.2.cross_attn.attn"
    with pytest.raises(ValueError, match=f"^{re.escape(f'MLA layers {a} and {b} share layer 2')}$"):
        BT.mla_layers(model(layer(a), layer(b)))


@pytest.mark.parametrize("attrs", [{"cache": [torch.zeros(0)]}, {"cache": torch.zeros(0, dtype=torch.bfloat16)},
                                   {"name": None}])
def test_a_module_with_no_tensor_for_a_cache_an_empty_one_or_no_layer_name_is_no_layer(attrs):
    bound = layer("model.layers.1.self_attn.attn")
    got = BT.mla_layers(model(layer(**attrs), bound))
    assert list(got) == [1] and got[1][1] is bound.kv_cache
    with pytest.raises(ValueError, match="^the model has no MLA layer with a bound KV cache$"):
        BT.mla_layers(model(layer(**attrs)))


def test_nested_modules_of_one_layer_name_are_one_layer_the_outermost():
    outer = layer()
    outer.impl = layer()
    got = BT.mla_layers(model(outer))
    assert list(got) == [0] and got[0][1] is outer.kv_cache


def test_the_step_hook_refuses_latents_at_other_layers_than_it_reads():
    m = model(layer(), layer("model.layers.3.self_attn.attn"))
    with pytest.raises(ValueError, match=r"^the model's MLA layers \[0, 3\] are not the tables' \[0\]$"):
        BT.StepHook(m, BT.Latents((0,)))
    assert BT.StepHook(m, BT.Latents((3,)), layers=[3]).latents.layers == (3,)
