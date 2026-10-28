"""circuit-check's bindings of `verity.core.protocols.certifier.charge_chain`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_charge_chain_roots() -> list[Any]:
    """The charge's carry chain (`verity.core.protocols.certifier.charge_chain`): a window's charge from its committed
    carry-in row, held to its carry-out row, at N = 2 slots (ChargeCount_v1's smallest with n_arrived's offset symbol)
    and at the compile slice's P·A = 14."""
    from verity.core.protocols.certifier.charge_chain import link_unit

    return [link_unit(2), link_unit(14)]
