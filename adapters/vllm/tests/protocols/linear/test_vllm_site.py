"""`verity_vllm.protocols.linear.vllm_site` on stand-in layers, without vLLM: the sites are the linear layers in module order, one
patch per quant method goes through `engine.hooks` and comes off with them, a site's call reaches the stack and any other
layer vLLM's own apply, and nothing is set on the layers."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from verity_vllm.engine import hooks
from verity_vllm.protocols.linear import vllm_site
from verity_vllm.protocols.linear.api import Arm, PassSpec, Shard, Stack


class Weight:
    def __init__(self, n, k, dtype="torch.bfloat16"):
        self.shape, self.dtype = (n, k), dtype


class Apply:
    def apply(self, layer, x, bias=None):
        return ("stock", layer.name, x, bias)


class Linear:
    def __init__(self, name, n, k, quant_method, bias=None):
        self.name, self.weight, self.quant_method, self.bias = name, Weight(n, k), quant_method, bias


class Other:
    pass


class Model:
    def __init__(self, layers):
        self.layers = layers

    def named_modules(self):
        yield "", self
        for i, m in enumerate(self.layers):
            yield f"model.layers.{i}", m


@dataclass
class Executor:
    name: str = "pouw:pearl-c-sm120-v1-h3"
    keeps_weight_state: bool = True
    accepts: frozenset = frozenset({"bf16"})
    controls: frozenset = frozenset()
    schedules: frozenset = frozenset({"serial"})
    calls: list = field(default_factory=list)

    def refusals(self, sites):
        return []

    def begin(self, spec):
        self.spec = spec
        return self

    def call(self, site, x, weight, bias):
        self.calls.append((site.weight_id, site.qualname, x))
        return ("pearl", site.weight_id)

    def end(self):
        return {"calls": len(self.calls)}


@pytest.fixture
def model():
    shared = Apply()
    return Model([Linear("qkv", 6144, 4096, Apply()), Other(), Linear("o", 4096, 4096, shared, bias="b"),
                  Linear("down", 4096, 14336, shared)])


def test_the_sites_are_the_linear_layers_in_module_order(model):
    sites = vllm_site.discover(model, types=(Linear,))
    assert [(s.weight_id, s.ordinal, s.qualname, s.n, s.k, s.weight_format, s.has_bias) for s, _ in sites] == [
        (0, 0, "model.layers.0", 6144, 4096, "bf16", False), (1, 1, "model.layers.2", 4096, 4096, "bf16", True),
        (2, 2, "model.layers.3", 4096, 14336, "bf16", False)]
    ranked = vllm_site.discover(model, Shard(rank=1, world=2, axis="n"), types=(Linear,))
    assert [s.weight_id for s, _ in ranked] == [1, 3, 5] and all(s.shard.rank == 1 for s, _ in ranked)


def test_attach_patches_each_quant_method_once_through_hooks_and_sets_nothing_on_the_layers(model):
    sites = vllm_site.discover(model, types=(Linear,))
    before = [dict(vars(layer)) for _, layer in sites]
    ex = Executor()
    stack = Stack([s for s, _ in sites], ex)
    h = hooks.Hooks(tag="pouw:test")
    vllm_site.attach(stack, sites, h)
    try:
        assert len(h.entries) == 2 and [dict(vars(layer)) for _, layer in sites] == before
        layers = [layer for _, layer in sites]
        assert layers[0].quant_method.apply(layers[0], "x") == ("stock", "qkv", "x", None)       # inactive: vLLM's own
        stack.activate(Arm("pearlc", "bf16", ex.name))
        with stack.pass_(PassSpec(index=3)):
            assert layers[1].quant_method.apply(layers[1], "x1", "b") == ("pearl", 1)
            assert layers[2].quant_method.apply(layers[2], "x2") == ("pearl", 2)
        stranger = Linear("lm_head", 128, 4096, layers[1].quant_method)
        assert stranger.quant_method.apply(stranger, "x") == ("stock", "lm_head", "x", None)   # not a site: vLLM's own
        assert ex.calls == [(1, "model.layers.2", "x1"), (2, "model.layers.3", "x2")] and stack.passes == [{"calls": 2}]
    finally:
        h.uninstall()
    assert hooks.live("pouw:test") == () and all("apply" not in vars(layer.quant_method) for _, layer in sites)
