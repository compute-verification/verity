"""The exact-int8 row's KV cache in the store: what the session needs to register it and to read it credited.

The int8 attention backend's cache (`engine.int8_attention_vllm.KV_LAYOUT`, vLLM's LBNHC) is each layer one region of
[pages][slots][KV heads][K ‖ V] bf16: a token-layer is 2 x 8 KV heads x 128 x 2 bytes = 4 KiB at Llama-3.1-8B, exactly one
store block, so the region's block j is physical slot j (page j // slots, slot j % slots) and its codec word w of that
block is index ``first_word + j * block_words + w``, the codec index pg_window decodes at word order 1 with ``store_base =
first_word`` (`engine.int8_window`).  The cache is bf16 (the int8 step quantizes each window as it reads it), so the session's
bf16-only KV pool takes it with ``kv_cache_dtype`` auto.

~~~text
token_layer_layout   the layer's `PageLayout` (binder.page_layout), refused unless one plane, slot-major, a slot one block
store_bases          each KV layer's region's first codec word, by layer index (region ``kv/<layer>``)
layer_keys           `int8_window.LayerKeys` of a layer from those and the session's bound offsets and plaintext flags
kv_refusal           why the session's pool would not take the engine's KV caches (None: it takes them)
~~~
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine.int8_window import LayerKeys
from verity_vllm.protocols.poub.binder import page_layout
from verity_vllm.protocols.poub.interface import Layout, PageLayout

BLOCK = 4096


def token_layer_layout(kv_cache: Any, pages: int, slots: int, head_size: int, block_bytes: int = BLOCK) -> PageLayout:
    """A layer of the int8 backend's cache (vLLM's view [pages, KV heads, slots, 2 head_size]) as the store registers it,
    each slot one ``block_bytes`` block: one plane, slot-major, a slot's K ‖ V for every KV head ``block_bytes`` bytes."""
    if why := A.page_layout_refusal(kv_cache, head_size):
        raise ValueError(f"the int8 KV cache's {why}")
    lay = page_layout(kv_cache, pages, slots)
    if lay.planes != 1 or not lay.slot_major or lay.slot_bytes != block_bytes:
        raise ValueError(f"the int8 KV cache is {lay.planes} planes, slot-major {lay.slot_major}, {lay.slot_bytes}-byte slots: "
                         f"not a {block_bytes}-byte block a token-layer")
    return lay


def store_bases(layout: Layout, pool: str = "kv", word_bytes: int | None = None) -> dict[int, int]:
    """Each KV layer's region's first codec word, by layer index (``<pool>/<layer>``, `Store.register_kv_pool`'s names);
    refused unless every one is a slot a block (``block_bytes`` the token-layer's).  With ``word_bytes``, the word is the
    session's codec's (block j's first at ``j * block_bytes / word_bytes``), not the region's own codec's."""
    out = {}
    for r in layout.regions:
        if r.kind == "kv" and r.name.startswith(f"{pool}/"):
            if r.block_bytes != BLOCK:
                raise ValueError(f"{r.name}: {r.block_bytes}-byte blocks, not one {BLOCK}-byte token-layer")
            out[int(r.name.split("/", 1)[1])] = r.first_word if word_bytes is None else r.first * (BLOCK // word_bytes)
    if not out:
        raise ValueError(f"the store holds no KV pool {pool!r}")
    return out


def layer_keys(bases: dict[int, int], bj: Callable[[int], Any] | None = None,
               plain: Callable[[int], Any] | None = None) -> Callable[[int], LayerKeys]:
    """`int8_window.CreditedRead`'s ``keys``: layer l's region's first word, its slots' bound offsets ``bj(l)`` and its plaintext
    flags ``plain(l)`` (each None or absent: the seed's key for every slot, no flags)."""
    return lambda layer: LayerKeys(bases[layer], None if bj is None else bj(layer), None if plain is None else plain(layer))


def kv_refusal(engine_args: dict[str, Any], kv_caches: Sequence[Any]) -> str | None:
    """Why the session's KV pool would not take this engine's caches: a ``kv_cache_dtype`` other than auto / bfloat16, or a
    cache tensor that is not bf16 (None: it takes them)."""
    want = str(engine_args.get("kv_cache_dtype") or "auto")
    if want not in ("auto", "bfloat16"):
        return f"kv_cache_dtype={want} (the session's KV pool is bf16)"
    dtypes = sorted({str(t.dtype) for t in kv_caches})
    if dtypes != ["torch.bfloat16"]:
        return f"KV cache dtype {dtypes} (the session's KV pool is bf16)"
    return None
