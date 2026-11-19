"""circuit-check's bindings of `verity.circuits.boolean.benes`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _benes_roots() -> list[Any]:
    """The switch and the network at 36-bit inputs over M = 16 positions, the I/O tie's smallest routing: the recursion
    from D = 4 to its D = 1 base."""
    from verity.circuits.boolean.benes import IoTieRoute, IoTieSwitch
    from verity.circuits.defs import bind

    return [bind(IoTieSwitch, W=36), bind(IoTieRoute, W=36, D=4)]
