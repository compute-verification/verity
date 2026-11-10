"""The exact-int8 parts (`TargetProfile.int8_exact`) in every process that builds the model, the rank processes of a
tensor-parallel engine included.

`register(parts)` puts the parts in this process's vLLM registries (the linear method, `engine.int8_vllm`; the attention backend,
`engine.int8_attention_vllm`) and names them in ``VERITY_INT8_PARTS``, which the ranks vLLM spawns inherit.  A rank builds its model
in its own process, where nothing the driver registered exists, but it imports the engine's `worker_extension_cls` first: this
module, imported by `pipeline.tp.rank_worker` and by `Int8Ranks` (the extension of an engine with no other, such as a Match control),
registers what ``VERITY_INT8_PARTS`` names when it is imported, before the rank builds its model.
"""
from __future__ import annotations

from collections.abc import Iterable

from verity_vllm import settings

ENV = "VERITY_INT8_PARTS"
PARTS = ("attention", "linears")
EXTENSION = "verity_vllm.engine.int8_ranks.Int8Ranks"


def register(parts: Iterable[str]) -> list[str]:
    """Register `parts` (a subset of `PARTS`) here and export them to the ranks; returns them, sorted.  No part: nothing, and the
    variable is removed."""
    parts = sorted(set(parts))
    if bad := [p for p in parts if p not in PARTS]:
        raise ValueError(f"int8 parts {bad} are not among {list(PARTS)}")
    if "linears" in parts:
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
    if "attention" in parts:
        from verity_vllm.engine import int8_attention_vllm
        int8_attention_vllm.register()
    if parts:
        settings.put(ENV, ",".join(parts))
    else:
        settings.unset(ENV)
    return parts


def register_from_env() -> list[str]:
    """Register what ``VERITY_INT8_PARTS`` names in this process (a rank's); returns it."""
    return register(settings.get(ENV))


class Int8Ranks:
    """The `worker_extension_cls` of a tensor-parallel engine that needs no other: no methods, only this module's import in each
    rank."""


register_from_env()
