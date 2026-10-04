"""The replay construction: the auditor recomputes each drawn unit's trace. Light; ships with `verity`."""

from __future__ import annotations

from verity.circuits.circuit import Circuit, trace


class Replay:
    name = "replay"

    def verify_unit(self, circuit: Circuit, inputs: tuple[int, ...], claimed: object) -> bool:
        return trace(circuit, inputs) == claimed
