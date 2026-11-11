"""circuit-check's bindings of `verity.definitions.registry.gumbel_boolean`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The Gumbel select (`GumbelSelectF32_v2`) at two lanes (its family's smallest, and the noisy row's: about 600,000 gates
    each, every lane about 300,000) and at a full group plus a remainder; the stream key, setup, lane and their pieces are
    static-free; and libdevice's logf without flush-to-zero (`NvLogf_v5`), which no select calls."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import gumbel_boolean as GB

    return [
        bind(GB.GumbelSelectF32, V=2),
        bind(GB.GumbelSelectF32, V=65),
        bind(GB.NvLogfV5),
    ]
