"""circuit-check's bindings of `verity.core.protocols.certifier.replay_chain`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_replay_chain_roots() -> list[Any]:
    """The state chain's three parts (`ReplayAdvance`, `ReplayLoad`, `ReplayClose`) at the tests' small shape and at
    ENC-REPLAY's sync KV-pressure trace's (R 7, W 2, A 3, Q 14, D 15, P 670, K 14), each well within the
    whole-Definition lowering budget."""
    from verity.core.protocols.certifier.replay_chain import Shape, advance_unit, close_unit, load_unit

    small = Shape(R=3, W=2, A=3, L=40, NB=20, IW=4, Q=5, D=6, P=3, K=3)
    trace = Shape(R=7, W=2, A=3, L=1024, NB=96, IW=4, Q=14, D=15, P=670, K=14)
    return [f(S) for S in (small, trace) for f in (advance_unit, load_unit, close_unit)]
