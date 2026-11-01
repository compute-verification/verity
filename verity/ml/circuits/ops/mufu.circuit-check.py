"""circuit-check's bindings of `verity.ml.circuits.ops.mufu`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """Every MUFU primitive, each checked against its batch kernel where it has one (all but `MufuTanh`)."""
    from verity.ml.circuits.ops import mufu as M

    return [M.MufuEx2Ftz, M.MufuRcpFtz, M.Fa2InvSum, M.MufuSqrtFtz, M.DivFullRcp, M.DivFullScaleA, M.RsqrtApprox, M.MufuTanh]
