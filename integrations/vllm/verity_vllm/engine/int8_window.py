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
          set (deferred residency, #2460); with ``encoded`` False, copied
~~~

The windows are the cache's plaintext K and V, so `int8_ops.window_attention`, reading each request's rows where
`int8_ops.paged_attention` gathers them from the pages, gives the same bits.  Its KV quantization is its own op
(``verity_int8::kv_quant_window``), which the int8 fold does not pair with the pages it read (the Match / Build side of a
credited read is not mapped yet).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

#: the codec's word order in a token-layer that pg_window reads: each head's K ‖ V, the cache's physical order
WORD_ORDER = 1


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


class CreditedRead:
    """The int8 attention's cache read through ``paged``'s `window_offsets` and `window` on ``lib`` (a `paged.PagedLib`
    at the store's codec, its key set), each layer's store state from ``keys(layer index)``; ``encoded`` False copies
    (an identity store), ``rounds`` the codec's."""

    def __init__(self, paged: Any, lib: Any, keys: Callable[[int], LayerKeys], *, encoded: bool = True,
                 rounds: int = 3) -> None:
        self.paged, self.lib, self.keys, self.encoded, self.rounds = paged, lib, keys, encoded, rounds
        self._step: Step | None = None
        self.steps = 0
        self.windows = 0

    def step(self, meta: Any, query_start_loc: torch.Tensor, seq_lens: torch.Tensor, kv_heads: int,
             head_dim: int) -> Step:
        """The step of attention metadata ``meta``: computed at its first layer, the same object at the others."""
        st = self._step
        if st is not None and st.meta is meta:
            return st
        self._step = None
        batch = query_start_loc.numel() - 1
        lens = seq_lens[:batch].to(torch.int32).contiguous()
        wo = torch.empty(batch + 1, dtype=torch.int32, device=lens.device)
        self.paged.window_offsets(self.lib, torch, lens, batch, wo)
        host = [0]
        for n in lens.tolist():
            host.append(host[-1] + max(int(n), 0))
        shape = (host[-1], kv_heads, head_dim)
        self._step = st = Step(meta, batch, lens, wo, host, torch.empty(shape, dtype=torch.bfloat16, device=lens.device),
                               torch.empty(shape, dtype=torch.bfloat16, device=lens.device))
        self.steps += 1
        return st

    def window(self, st: Step, kv_cache: torch.Tensor, layer: int, block_table: torch.Tensor,
               slot_mapping: torch.Tensor | None) -> None:
        """Layer ``layer``'s windows of step ``st`` into its buffers, from ``kv_cache`` (vLLM's view [pages, KV heads,
        slots, 2 head_dim]) under the step's block table, after flagging the slots ``slot_mapping`` wrote."""
        keys = self.keys(layer)
        plain = keys.plain if self.encoded else None
        if plain is not None and slot_mapping is not None:
            written = slot_mapping[slot_mapping >= 0].long()
            plain[written] = 1
        lay = self.paged.layout(kv_cache, kv_cache.shape[1], self.lib.shape["wb"], (0, 1, 2))
        self.paged.window(self.lib, torch, kv_cache, lay, keys.store_base, 1, 0, int(self.encoded), self.rounds,
                          st.lens, block_table, st.wo, st.batch, st.k, st.v, order=WORD_ORDER,
                          bj=keys.bj if self.encoded else None, plain=plain)
        self.windows += 1
