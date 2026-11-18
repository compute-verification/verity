"""circuit-check's bindings of `verity_vllm.pipeline.tick_slice`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_tick_slice_roots() -> list[Any]:
    """The compile slice under the certified tick at its toy window's class (`tick_slice.demo`), sync and async: two
    sequences, (R 2, W 1, A 3), five steps (six async) and three arrival slots, buckets of 10 time units carried at most three a step."""
    from verity_vllm.pipeline.tick_slice import demo, unit

    return [unit(**demo(asy)[0]) for asy in (False, True)]
