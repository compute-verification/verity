"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.scalar`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_constant_roots() -> list[Any]:
    """The Boolean constants GLM-5.3 TP8's B1 request Calls, 1 and 0x11f (`const_word`: made on demand, as the word constant is,
    so no module binds one ahead)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import scalar as BS

    return [bind(BS.const_word(32, 1)), bind(BS.const_word(32, 0x11F))]
