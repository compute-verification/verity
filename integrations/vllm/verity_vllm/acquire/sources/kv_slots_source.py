"""Each attention layer's KV slots as an acquisition source: per layer and step, the slot every token row's KV went to,
`<layer name>/slot_mapping` (int64 [rows], slot = page * block_size + offset), in the same step stream and run root as the
run's other values.  Opt-in (commit and tp-commit `--memory-map`, a TP rank's own layers), never the record: the memory map
(`pipeline.memory_map`) takes each block's page from them and assumes none.

The layers are the modules the engine bound a KV cache to (`layer_name`, `kv_cache`): attention's and MLA's.  DSA's indexer
cache (`DeepseekV32IndexerCache`, named by `prefix`, no `layer_name`) is none of them: its 132-byte rows pack across 4 KiB
blocks, and memory accounting has it uncredited until the map packs them.  A forward pre-hook on each layer reads its
layer's entry of the forward context's `slot_mapping`, the one its cache write reads (FlashAttention's `do_kv_cache_update`,
MLA's, and the sparse MLA layer's `fused_norm_rope`, which writes the indexer cache at the same slots), and commits a copy,
one per distinct tensor of the step.  `caches` keeps each layer's cache tensor: its shape and strides are the page layout
the map's blocks follow, and its bytes what the map's digests are checked against.  `own_names` are the names it commits: no
module's output, so no module the identity check counts and no value the Match record holds (`oracle_compare`'s ``own``).
"""
from __future__ import annotations

import time
from typing import Any, Mapping

MEMBER = "slot_mapping"


def kv_layers(model: Any) -> dict[str, Any]:
    """Layer name -> the module the engine bound that layer's KV cache to (the outermost, when nested modules carry one name)."""
    out: dict[str, Any] = {}
    for _path, m in model.named_modules():
        name, cache = getattr(m, "layer_name", None), getattr(m, "kv_cache", None)
        if isinstance(name, str) and hasattr(cache, "stride") and cache.numel() and name not in out:
            out[name] = m
    return out


class KvSlotsSource:
    short_name = "kv_slots"

    def __init__(self, committer: Any, model: Any) -> None:
        import torch

        self.torch, self.c = torch, committer
        self.layers = kv_layers(model)
        if not self.layers:
            raise ValueError("kv slots: the model has no module with a bound KV cache (layer_name, kv_cache)")
        self.caches: dict[str, Any] = {n: m.kv_cache for n, m in self.layers.items()}
        self.own_names = frozenset(f"{n}/{MEMBER}" for n in self.layers)
        self._handles: list[Any] = []
        self._copies: dict[int, Any] = {}
        self.stats: dict[str, Any] = {"layers": len(self.layers), "tensors": 0, "bytes": 0, "errors": 0}
        self.active = False

    def install(self) -> None:
        from vllm.forward_context import get_forward_context

        self._context = get_forward_context
        for name, m in self.layers.items():
            self._handles.append(m.register_forward_pre_hook(self._pre(name)))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._copies.clear()
        self.active = False

    def on_step_begin(self) -> None:
        self._copies.clear()

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def _pre(self, name: str) -> Any:
        me = self

        def pre(_module: Any, _args: tuple) -> None:
            job = me.c._cur
            if job is None:
                return
            sm = me._context().slot_mapping
            s = sm.get(name) if isinstance(sm, dict) else None
            if s is None:
                me.error(f"kv slots: {name} ran in step {job.step} with no slot mapping in the forward context ({type(sm).__name__})")
                return
            t0 = time.perf_counter()
            v = me._copies.get(id(s))
            if v is None:
                v = me._copies[id(s)] = s.to(me.torch.int64, copy=True)
            me.c._acquire(f"{name}/{MEMBER}", v, job)
            me.stats["tensors"] += 1
            me.stats["bytes"] += int(v.numel()) * 8
            spans = getattr(me.c, "spans", None)
            if spans is not None:
                spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=int(v.numel()) * 8)

        return pre

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        return {"per_token_bytes": 8 * len(self.layers), "per_seq_bytes": 0, "tensors_per_step": len(self.layers)}

    def declaration_classes(self) -> list[dict]:
        return []


def attach(com: Any, model: Any = None) -> KvSlotsSource:
    src = KvSlotsSource(com, model if model is not None else com.model)
    com.attach_source(src)
    print(f"[delta] kv slots source attached: {len(src.layers)} KV layers -> committer {com.name}", flush=True)
    return src


def caches_of(com: Any) -> dict[str, Any] | None:
    """The KV caches of the KV-slots source attached to committer ``com``; None without one."""
    return next((s.caches for s in getattr(com, "sources", ()) if getattr(s, "short_name", "") == KvSlotsSource.short_name), None)


__all__ = ["MEMBER", "KvSlotsSource", "kv_layers", "attach", "caches_of"]
