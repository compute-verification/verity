"""circuit-check's bindings of `verity_vllm.program.registry.pouw_rows`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The model Program's rows at a PoUW site, without and with the bias."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import pouw_rows

    return [bind(pouw_rows.NcpLinearRow, K=64, N=16), bind(pouw_rows.NcpLinearRowBias, K=64, N=16),
            bind(pouw_rows.NcpLinearCoordinate, K=64)]
