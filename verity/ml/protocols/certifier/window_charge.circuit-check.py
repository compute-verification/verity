"""circuit-check's bindings of `verity.ml.protocols.certifier.window_charge`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_window_charge_roots() -> list[Any]:
    """Advice v1's whole window charged (`verity.ml.protocols.certifier.window_charge`): a window at one decision, one
    hit and one order entry (every symbol kind: the decision, the hit's bin and offset, the gap's and the rank step's bins
    and their one- and two-chunk offsets); at two decisions and two hits, its smallest whose n_hits reaches its offset
    symbol; and the link at one of each, from its committed carry-in row to its carry-out row. Each is within the
    whole-Definition lowering budget."""
    from verity.ml.protocols.certifier.window_charge import charge_unit, link_unit

    return [charge_unit(1, 1, 1), charge_unit(2, 2, 0), link_unit(1, 1, 1)]
