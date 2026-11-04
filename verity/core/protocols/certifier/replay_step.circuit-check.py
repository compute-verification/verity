"""circuit-check's bindings of `verity.core.protocols.certifier.replay_step`'s Definition, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any

#: The trace corpus's smallkv engine (Llama-3.2-1B, 512 KV blocks, prefix caching, chunked prefill), which preempts
SMALLKV = {"max_num_seqs": 256, "max_num_batched_tokens": 2048, "max_model_len": 2048, "num_gpu_blocks": 512, "block_size": 16,
           "enable_chunked_prefill": True, "enable_prefix_caching": True, "scheduler_reserve_full_isl": True}


def _network_replay_roots() -> list[Any]:
    """The replay step (`verity.core.protocols.certifier.replay_step`) on the smallkv engine at the smallest size class
    that reaches every stage (4 running, 8 waiting, 4 arrivals; lowered whole), sync and then async with the walk
    entries of a 511-entry cache table (EW = 9), and at the class its difftest runs (16, 64, 16)."""
    from verity.core.protocols.certifier.replay_step import statics, unit

    return [unit(**statics(4, 8, 4, SMALLKV, 10)), unit(**statics(4, 8, 4, {**SMALLKV, "async_scheduling": True}, 10, 9)),
            unit(**statics(16, 64, 16, SMALLKV, 10))]
