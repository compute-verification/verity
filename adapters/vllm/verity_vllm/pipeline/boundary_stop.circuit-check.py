"""circuit-check's bindings of `verity_vllm.pipeline.boundary_stop`'s Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _stop_roots() -> list[Any]:
    """`EgressStop_v1` at a small class with two stop ids and with none, and at T = 1,024 ids with GLM-5.3's three
    stop ids and a model of 131,072 ids."""
    from verity_vllm.pipeline.boundary_stop import egress_stop

    return [egress_stop(4, (3, 7), 64), egress_stop(4, (), 64), egress_stop(1024, (154820, 154827, 154829), 131072)]
