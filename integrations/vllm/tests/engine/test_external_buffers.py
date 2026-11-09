"""`Capture._install_external_buffers` records the runner's KV-cache layout in its external-buffers `Note`, and None for each part
the runner doesn't carry or that doesn't read as an integer: a runner without `kv_cache_config`, a config without groups, a block
size or block count that isn't a number.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_vllm.observe import capture as C


class _Writer:
    def __init__(self):
        self.events = []

    def write(self, ev):
        self.events.append(ev)


def _note(runner) -> dict:
    cap = C.Capture.__new__(C.Capture)
    cap.runner, cap.model = runner, SimpleNamespace(named_modules=lambda: [])
    cap.storage = SimpleNamespace(events_pending=lambda: [], use_weakrefs=False)
    cap.writer, cap.events_written = _Writer(), 0
    cap._install_external_buffers()
    (note,) = cap.writer.events
    return note.data


def _group(layers, block_size):
    return SimpleNamespace(layer_names=layers, kv_cache_spec=type("FullAttentionSpec", (), {"block_size": block_size})())


def test_a_kv_cache_config_is_recorded():
    cfg = SimpleNamespace(num_blocks=128, kv_cache_groups=[_group(["model.layers.0.self_attn.attn"], 16)])
    assert _note(SimpleNamespace(kv_cache_config=cfg)) == {
        "buffers": {}, "kv_num_blocks": 128, "block_size": 16,
        "kv_cache_groups": [{"layers": ["model.layers.0.self_attn.attn"], "block_size": 16, "spec": "FullAttentionSpec"}],
        "storage_weakref_frees": False}


@pytest.mark.parametrize("runner, want", [
    (SimpleNamespace(), (None, None, None)),
    (SimpleNamespace(kv_cache_config=SimpleNamespace(num_blocks=None, kv_cache_groups=[])), (None, None, [])),
    (SimpleNamespace(kv_cache_config=SimpleNamespace(num_blocks="n/a", kv_cache_groups=None)), (None, None, None)),
    (SimpleNamespace(kv_cache_config=SimpleNamespace(num_blocks=64, kv_cache_groups=[_group(["l0"], None)])), (64, None, None)),
])
def test_what_the_runner_does_not_carry_is_recorded_as_none(runner, want):
    d = _note(runner)
    assert (d["kv_num_blocks"], d["block_size"], d["kv_cache_groups"]) == want
