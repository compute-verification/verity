"""circuit-check's bindings of `verity.protocols.certifier.prover.window_charge`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_window_charge_roots() -> list[Any]:
    """Advice v1's whole window charged (`verity.protocols.certifier.prover.window_charge`): a window at one decision, one
    hit, one order entry and one length term (every symbol kind: the decision, the hit's bin and offset, the gap's and
    the rank step's bins and their one- and two-chunk offsets; and the length term's tail); at two decisions, two hits
    and two length terms, its smallest whose n_hits reaches its offset symbol and whose tail sums two terms; and the link
    at one of each, from its committed carry-in row to its carry-out row. Each is within the whole-Definition lowering
    budget."""
    from verity.protocols.certifier.prover.window_charge import charge_unit, link_unit

    return [charge_unit(1, 1, 1, 1), charge_unit(2, 2, 0, 2), link_unit(1, 1, 1, 1)]
