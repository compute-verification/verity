"""circuit-check's bindings of `verity_vllm.compiler.tick_step`'s Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any

#: the served engine under its AsyncScheduler, whose class is (R, W, A) = (8, 1, 9)
SERVED = {"max_num_seqs": 8, "max_num_batched_tokens": 2048, "max_model_len": 512, "num_gpu_blocks": 101626, "block_size": 16,
          "enable_chunked_prefill": True, "enable_prefix_caching": False, "async_scheduling": True}
#: today's rule in microseconds (the difftest's constants), and the chosen rule in nanoseconds
RULES = ({"tau": 100_000, "sat": 30_000, "free": 40_000}, {"tau": 10**8, "sat": 6_303_000, "free": 6_303_000, "token": 18_126})


def _network_tick_roots() -> list[Any]:
    from verity_vllm.compiler.tick_step import statics, unit

    return [unit(**statics(8, 1, 9, SERVED, 9, **rule, quiet=quiet)) for rule in RULES for quiet in (True, False)]
