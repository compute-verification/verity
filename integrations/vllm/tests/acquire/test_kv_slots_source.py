"""The KV-slots source (`acquire.sources.kv_slots_source`) on CPU tensors with a recording committer and a stand-in forward
context: it hooks the modules the engine bound a KV cache to (the outermost of a name), commits a copy of each layer's slot mapping
as `<layer>/slot_mapping` only during a committed step, one copy per distinct tensor of the step, names a layer the context gives
no slots, and stops at `uninstall`.  In GLM-5.3's tree its layers are the MLA layers, not DSA's indexer caches.  A TP rank attaches
it through its taps (`taps.for_ranks`) under tp-commit `--memory-map`."""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import kv_slots_source as S  # noqa: E402


class Recorder:
    def __init__(self) -> None:
        self._cur = None
        self.errors: list[str] = []
        self.acquired: list[tuple[str, object]] = []
        self.sources: list = []
        self.name = "rec"

    def _acquire(self, name, t, job):
        self.acquired.append((name, t))

    def attach_source(self, src):
        self.sources.append(src)


class Attn(torch.nn.Module):
    def __init__(self, name: str, blocks: int) -> None:
        super().__init__()
        self.layer_name, self.kv_cache = name, torch.zeros(blocks, 2, 4, 1, 8)

    def forward(self, x):
        return x


def _model():
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
    for i, layer in enumerate(m.layers):
        layer.attn = Attn(f"model.layers.{i}.attn", 0 if i == 2 else 6)       # layer 2's cache is unbound
    m.layers[0].attn.inner = Attn("model.layers.0.attn", 6)
    return m


@pytest.fixture
def context(monkeypatch):
    ctx = types.SimpleNamespace(slot_mapping={})
    monkeypatch.setitem(sys.modules, "vllm.forward_context", types.SimpleNamespace(get_forward_context=lambda: ctx))
    return ctx


def test_the_source_commits_each_layers_slots_during_a_step(context):
    m, com = _model(), Recorder()
    src = S.attach(com, m)
    assert list(src.layers) == ["model.layers.0.attn", "model.layers.1.attn"] and src.layers["model.layers.0.attn"] is m.layers[0].attn
    assert com.sources == [src] and src.extra_step_bytes({}, m) == {"per_token_bytes": 16, "per_seq_bytes": 0, "tensors_per_step": 2}
    assert src.own_names == {"model.layers.0.attn/slot_mapping", "model.layers.1.attn/slot_mapping"}
    src.install()
    slots = torch.tensor([17, 18, 40], dtype=torch.int32)
    context.slot_mapping = {"model.layers.0.attn": slots, "model.layers.1.attn": slots}
    x = torch.zeros(3)
    m.layers[0].attn(x)                                               # no committed step
    com._cur = types.SimpleNamespace(step=4)
    src.on_step_begin()
    for layer in m.layers[:2]:
        layer.attn(x)
    assert [n for n, _ in com.acquired] == ["model.layers.0.attn/slot_mapping", "model.layers.1.attn/slot_mapping"]
    (_, a), (_, b) = com.acquired
    assert a is b and a.dtype == torch.int64 and a.tolist() == [17, 18, 40] and a.data_ptr() != slots.data_ptr()
    context.slot_mapping = {"model.layers.0.attn": slots}
    m.layers[1].attn(x)
    assert len(com.acquired) == 2 and com.errors == ["kv slots: model.layers.1.attn ran in step 4 with no slot mapping in the forward context (dict)"]
    src.uninstall()
    m.layers[0].attn(x)
    assert len(com.acquired) == 2 and src.stats == {"layers": 2, "tensors": 2, "bytes": 48, "errors": 1}


def test_a_model_without_a_kv_cache_is_refused():
    with pytest.raises(ValueError, match="no module with a bound KV cache"):
        S.attach(Recorder(), torch.nn.Linear(2, 2))


class IndexerCache(torch.nn.Module):
    """vLLM's `DeepseekV32IndexerCache`: its bound cache (bytes, 132 a slot) under its `prefix`, no `layer_name`."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix, self.kv_cache = prefix, torch.zeros(6, 4, 132, dtype=torch.uint8)


def test_glms_kv_layers_are_its_mla_layers_and_not_the_indexer_caches(context):
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    for i, layer in enumerate(m.layers):
        layer.self_attn = torch.nn.Module()
        layer.self_attn.attn = Attn(f"model.layers.{i}.self_attn.attn", 6)
        layer.self_attn.attn.kv_cache = torch.zeros(6, 4, 576, dtype=torch.bfloat16)        # MLA's, bound squeezed of its one head
        layer.self_attn.indexer = torch.nn.Module()
        layer.self_attn.indexer.k_cache = IndexerCache(f"model.layers.{i}.self_attn.indexer.k_cache")
    com = Recorder()
    src = S.attach(com, m)
    assert list(src.layers) == ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    assert [tuple(t.shape) for t in src.caches.values()] == [(6, 4, 576)] * 2
    assert S.caches_of(com) is src.caches and S.caches_of(Recorder()) is None


def test_a_tp_rank_attaches_it_through_its_taps_under_memory_map():
    from verity_vllm.acquire.sources import taps as T

    assert "kv_slots" not in T.for_ranks(T.TapSources()).keywords
    com = Recorder()
    got = T.for_ranks(T.TapSources(), kv_slots=True)(com, _model(), None, 1)
    assert got["kv_slots_source"] == {"layers": 2} and S.caches_of(com) is com.sources[-1].caches
    refused = T.for_ranks(T.TapSources(), kv_slots=True)(Recorder(), torch.nn.Linear(2, 2), None, 0)["kv_slots_source"]
    assert "no module with a bound KV cache" in refused["error"], "a rank without KV caches names it; its map then fails, no verdict"
