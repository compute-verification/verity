"""A credited read the engine installs at its build from ``VERITY_INT8_CREDITED_READ=MODE:PAGED_SO``, so that a row's own
stages (Build, Match, Commit) serve the int8 attention through `engine.int8_window` with no store session: every int8
attention layer reads its windows through pg_window (``PAGED_SO``, ``benchmarks/vllm_encoded/paged.cu`` built at P1024 with
the model's query heads) under the seed's key, layer l's slots at codec words ``(7 + l * slots) * token-layer words`` on.

~~~text
copy      pg_window copies the cache (``encoded`` False)
flagged   decoded, every slot flagged plaintext
encoded   decoded: before each layer's window every slot an earlier step wrote (its window flagged it from slot_mapping) is
          encoded in place under the seed's key and unflagged, as the session encodes a slot once its append's coin arrives,
          but for a slot this step's write refilled
~~~

Unset, nothing is installed.  A value that is not ``MODE:PATH`` with one of these modes, or a model without int8 attention
layers, refuses.
"""
from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any

import torch

from verity_vllm import config
from verity_vllm.engine.env import INT8_CREDITED_READ as ENV
from verity_vllm.engine.int8_window import CreditedRead, LayerKeys

MODES = ("copy", "flagged", "encoded")
ROUNDS = 3


def requested(value: str | None) -> tuple[str, str] | None:
    """(mode, paged.so path) of `ENV`'s ``value`` (`engine.env.int8_credited_read`); None when unset or empty."""
    if not value:
        return None
    mode, sep, so = value.partition(":")
    if mode not in MODES or not sep or not so:
        raise ValueError(f"{ENV}={value!r}: expected MODE:PAGED_SO with MODE one of {', '.join(MODES)}")
    return mode, so


def paged_module() -> ModuleType:
    """``benchmarks/vllm_encoded/paged.py``, loaded once as ``paged`` (the module `store.remote.session_module` loads)."""
    if "paged" in sys.modules:
        return sys.modules["paged"]
    spec = importlib.util.spec_from_file_location("paged", config.POUB_SESSION.parent / "vllm_encoded" / "paged.py")
    if spec is None or spec.loader is None:
        raise ImportError("no paged.py beside benchmarks/poub_session")
    module = importlib.util.module_from_spec(spec)
    sys.modules["paged"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["paged"]
        raise
    return module


class SettlingRead(CreditedRead):
    """`encoded`: before each layer's window, its slots an earlier step flagged encoded in place at their codec words and
    unflagged, but for the slots this step's write refilled (a freed block handed to a new request): the layer's write
    precedes its window, so those hold this step's plaintext and stay flagged (a layer's cache vLLM's view [pages, KV heads,
    slots, 2 head_dim] over [pages][slots][KV heads][K ‖ V])."""

    def __init__(self, paged: Any, lib: Any, bases: dict[int, int], flags: list[torch.Tensor], words: int) -> None:
        super().__init__(paged, lib, lambda layer: LayerKeys(bases[layer], None, flags[layer]), rounds=ROUNDS)
        self.bases, self.flags, self.words = bases, flags, words
        self.settled = 0

    def read(self, kv_cache: torch.Tensor, layer: int, block_table: torch.Tensor, slot_mapping: torch.Tensor | None,
             *rest: torch.Tensor) -> None:
        flags = self.flags[layer]
        due = flags.clone()
        if slot_mapping is not None:
            due[slot_mapping[slot_mapping >= 0].long()] = 0
        slots = due.nonzero().flatten().tolist()
        base, block, st = kv_cache.data_ptr(), kv_cache.stride(2) * kv_cache.element_size(), self.paged.stream(torch)
        for s in slots:
            p = base + s * block
            self.lib.call("kv_encode", p, p, self.words, self.bases[layer] + s * self.words, ROUNDS, 0, st)
        flags.zero_()
        self.settled += len(slots)
        super().read(kv_cache, layer, block_table, slot_mapping, *rest)


def install(engine: Any, value: str | None) -> dict[str, Any] | None:
    """Install the credited read `ENV`'s ``value`` requests on ``engine``'s int8 attention layers; its record (mode, library,
    layers, slots a layer), or None when none is requested."""
    want = requested(value)
    if want is None:
        return None
    mode, so = want
    from verity_vllm.engine.int8_attention_vllm import install_credited_read

    P = paged_module()
    lib = P.PagedLib("p1024", so)
    P.KB.set_key(lib, P.KB.lib_key(lib, ROUNDS))
    words = P.K.token_layer_words(lib.field)
    caches = list(engine.runner.kv_caches)
    nslots = caches[0].shape[0] * int(engine.runner.cache_config.block_size)
    bases = {layer: (7 + layer * nslots) * words for layer in range(len(caches))}
    if mode == "copy":
        read: CreditedRead = CreditedRead(P, lib, lambda layer: LayerKeys(bases[layer]), encoded=False, rounds=ROUNDS)
    elif mode == "flagged":
        ones = [torch.ones(nslots, dtype=torch.uint8, device=c.device) for c in caches]
        read = CreditedRead(P, lib, lambda layer: LayerKeys(bases[layer], None, ones[layer]), rounds=ROUNDS)
    else:
        flags = [torch.zeros(nslots, dtype=torch.uint8, device=c.device) for c in caches]
        read = SettlingRead(P, lib, bases, flags, words)
    layers = install_credited_read(engine.model, read)
    return {"mode": mode, "paged_so": so, "lib": dict(lib.shape), "layers": len(layers), "slots": nslots, "token_layer_words": words}
