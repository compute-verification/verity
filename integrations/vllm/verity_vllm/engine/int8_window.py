"""The exact-int8 attention's credited cache read: each step's windows through memory accounting's `pg_window`
(``benchmarks/vllm_encoded/paged.cu``, loaded by the store's session as its ``paged`` module) over a cache the store holds
encoded in place, in place of `int8_attention.window_of`'s gather from the pages.

`CreditedRead` is what the session installs on the engine's int8 attention layers
(`int8_attention_vllm.install_credited_read`); without one, the backend reads the cache as it is.

~~~text
offsets   once a step (at its first layer; vLLM builds one attention metadata a step for every layer):
          `pg_window_offsets` of the requests' lengths, request r's window rows wo[r] .. wo[r + 1] - 1 of the step's
          working buffers, K and V bf16 [rows][KV heads][head_dim], allocated for the step and dropped with it (no
          plaintext copy of the cache outlives its step)
flags     before each layer's window, its slots the step's KV write filled are flagged plaintext (the write put plain K
          and V there; the session clears a slot's flag when its append's coin arrives and it encodes the slot in place)
window    once a layer: `pg_window` of the layer's cache into the buffers at word order 1 (a token-layer is
          [KV heads][K ‖ V], `int8_attention_vllm.KV_LAYOUT`): each word decoded at its codec index, the layer's region's
          first word plus the physical word (a layer to a region, a 4 KiB block a slot: `store.int8_kv`), under the
          slot's bound offset (`LayerKeys.bj`; None: the seed's key), or read as it is where the slot's plaintext flag is
          set (deferred residency); with ``encoded`` False, copied
~~~

The windows are the cache's plaintext K and V, so `int8_ops.window_attention`, reading each request's rows where
`int8_ops.paged_attention` gathers them from the pages, gives the same bits.  `int8_ops.credited_attention` runs the read as
two named ops, ``verity_int8::credited_step`` (the offsets and buffers) and ``verity_int8::credited_window`` (a layer's
flags and window), so a capture records the cache, block table and buffers each window read, and the int8 fold
(`observe.fold.patterns.int8`) binds each request's slice of ``verity_int8::kv_quant_window_batch`` to its KV slots as
it binds ``verity_int8::kv_quant_batch``'s.  The ops find their read by its ``handle``.
"""
from __future__ import annotations

import itertools
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

#: the codec's word order in a token-layer that pg_window reads: each head's K ‖ V, the cache's physical order
WORD_ORDER = 1

_READS: weakref.WeakValueDictionary[int, CreditedRead] = weakref.WeakValueDictionary()
_HANDLES = itertools.count(1)


@dataclass(frozen=True)
class LayerKeys:
    """What a layer's decode reads of the store: its region's first codec word, its slots' bound offsets (int32
    [physical slots][limbs] on the device, `paged.offsets`; None: every encoded slot under the seed's key) and its
    plaintext flags (uint8 [physical slots] on the device, nonzero: as it is; None: none flagged, and the step's writes are
    not flagged either)."""

    store_base: int
    bj: torch.Tensor | None = None
    plain: torch.Tensor | None = None


@dataclass
class Step:
    """A step's windows: its metadata (identity), requests, window offsets (device and host) and working buffers."""

    meta: Any
    batch: int
    lens: torch.Tensor
    wo: torch.Tensor
    offsets: list[int]
    k: torch.Tensor
    v: torch.Tensor


def read_of(handle: int) -> CreditedRead:
    """The live read of ``handle`` (`CreditedRead.handle`)."""
    read = _READS.get(handle)
    if read is None:
        raise LookupError(f"no credited read {handle} in this process")
    return read


class CreditedRead:
    """The int8 attention's cache read through ``paged``'s `window_offsets` and `window` on ``lib`` (a `paged.PagedLib`
    at the store's codec, its key set), each layer's store state from ``keys(layer index)``; ``encoded`` False copies
    (an identity store), ``rounds`` the codec's."""

    def __init__(self, paged: Any, lib: Any, keys: Callable[[int], LayerKeys], *, encoded: bool = True,
                 rounds: int = 3) -> None:
        self.paged, self.lib, self.keys, self.encoded, self.rounds = paged, lib, keys, encoded, rounds
        self.current: Step | None = None
        self.steps = 0
        self.windows = 0
        self.handle = next(_HANDLES)
        _READS[self.handle] = self

    def buffers(self, seq_lens: torch.Tensor, batch: int, kv_heads: int,
                head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A step's (lens int32 [batch], window offsets int32 [batch + 1], K, V bf16 [rows][kv_heads][head_dim]), fresh."""
        lens = seq_lens[:batch].to(torch.int32, copy=True).contiguous()
        wo = torch.empty(batch + 1, dtype=torch.int32, device=lens.device)
        self.paged.window_offsets(self.lib, torch, lens, batch, wo)
        rows = int(lens.clamp(min=0).sum())
        k = torch.empty((rows, kv_heads, head_dim), dtype=torch.bfloat16, device=lens.device)
        self.steps += 1
        return lens, wo, k, torch.empty_like(k)

    def read(self, kv_cache: torch.Tensor, layer: int, block_table: torch.Tensor, slot_mapping: torch.Tensor | None,
             lens: torch.Tensor, wo: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Layer ``layer``'s windows into ``k`` and ``v`` from ``kv_cache`` (vLLM's view [pages, KV heads, slots, 2 head_dim])
        under the step's block table, after flagging the slots ``slot_mapping`` wrote."""
        keys = self.keys(layer)
        plain = keys.plain if self.encoded else None
        if plain is not None and slot_mapping is not None:
            plain[slot_mapping[slot_mapping >= 0].long()] = 1
        lay = self.paged.layout(kv_cache, kv_cache.shape[1], self.lib.shape["wb"], (0, 1, 2))
        self.paged.window(self.lib, torch, kv_cache, lay, keys.store_base, 1, 0, int(self.encoded), self.rounds,
                          lens, block_table, wo, int(lens.numel()), k, v, order=WORD_ORDER,
                          bj=keys.bj if self.encoded else None, plain=plain)
        self.windows += 1
