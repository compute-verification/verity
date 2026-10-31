"""The int8 row's KV cache in the store (`store.int8_kv`): a token-layer one block of its layer's region, each layer's first
codec word, and the session's bf16 pool taking the cache."""
from __future__ import annotations

import pytest
import torch

from tests.protocols.poub.test_registry import CODEC, RECEIPT
from verity_vllm.protocols.poub.binder import PagedBacking, storage_bytes
from verity_vllm.protocols.poub.int8_kv import BLOCK, kv_refusal, layer_keys, store_bases, token_layer_layout
from verity_vllm.protocols.poub.registry import Store

PAGES, SLOTS, G, D = 3, 16, 8, 128


def _cache(seed: int) -> torch.Tensor:
    """A layer as the int8 backend's cache: [pages][slots][KV heads][K‖V] in memory, vLLM's view [pages, KV heads, slots, 2D]."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(PAGES, SLOTS, G, 2 * D, generator=g).to(torch.bfloat16).transpose(1, 2)


def test_a_token_layer_is_one_block_of_its_layers_region():
    caches = [_cache(1), _cache(2)]
    lay = token_layer_layout(caches[0], PAGES, SLOTS, D)
    assert (lay.planes, lay.slot_bytes, lay.slot_major, lay.pages, lay.slots) == (1, BLOCK, True, PAGES, SLOTS)
    backings = [PagedBacking(storage_bytes(t), t.storage_offset() * t.element_size(), lay) for t in caches]
    s = Store()
    s.register_kv_pool("kv", backings, lay, block_bytes=BLOCK, codec=CODEC, receipt=RECEIPT, encode=False)
    regions = {r.name: r for r in s.layout().regions}
    bases = store_bases(s.layout())
    assert sorted(bases) == [0, 1]
    for li, (t, b) in enumerate(zip(caches, backings)):
        r = regions[f"kv/{li}"]
        assert bases[li] == r.first_word and r.blocks == PAGES * SLOTS
        for j in (0, 1, SLOTS + 3, PAGES * SLOTS - 1):
            page, slot = divmod(j, SLOTS)
            assert b.read(j * BLOCK, BLOCK) == t.transpose(1, 2)[page, slot].contiguous().view(torch.uint8).numpy().tobytes()
    assert store_bases(s.layout(), word_bytes=128) == {li: regions[f"kv/{li}"].first * 32 for li in (0, 1)}
    keys = layer_keys(bases, plain=lambda li: f"flags{li}")(1)
    assert (keys.store_base, keys.bj, keys.plain) == (bases[1], None, "flags1")


def test_other_layouts_and_block_sizes_are_refused():
    with pytest.raises(ValueError, match="strides"):
        token_layer_layout(torch.empty(PAGES, G, SLOTS, 2 * D, dtype=torch.bfloat16), PAGES, SLOTS, D)
    with pytest.raises(ValueError, match="block a token-layer"):
        token_layer_layout(_cache(0), PAGES, SLOTS, D, block_bytes=2 * BLOCK)
    s = Store()
    with pytest.raises(ValueError, match="no KV pool"):
        store_bases(s.layout())


def test_the_sessions_bf16_pool_takes_the_int8_rows_cache():
    """The int8 backend's cache is bf16 (it quantizes each window as it reads it), under the row's kv_cache_dtype auto."""
    caches = [_cache(0)]
    assert kv_refusal({"kv_cache_dtype": "auto"}, caches) is None
    assert kv_refusal({}, caches) is None
    assert "kv_cache_dtype=fp8" in kv_refusal({"kv_cache_dtype": "fp8"}, caches)
    assert "torch.int8" in kv_refusal({}, [torch.zeros(2, dtype=torch.int8)])
