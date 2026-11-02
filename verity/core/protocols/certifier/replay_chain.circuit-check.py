"""circuit-check's bindings of `verity.core.protocols.certifier.replay_chain`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_replay_chain_roots() -> list[Any]:
    """The state chain's three parts (`ReplayAdvance`, `ReplayLoad`, `ReplayClose`) at the tests' small shape, at
    the replay's sync KV-pressure trace's (R 7, W 2, A 3, Q 14, D 15, P 670, K 14), and at the served vLLM run's engine
    class under its AsyncScheduler (R 8, W 1, A 9, Q 4, D 32, P 54, K 4, AS 1), with and without 17-bit walk
    entries, each well within the whole-Definition lowering budget."""
    from verity.core.protocols.certifier.replay_chain import Shape, advance_unit, close_unit, load_unit

    small = Shape(R=3, W=2, A=3, L=40, NB=20, IW=4, Q=5, D=6, P=3, K=3)
    trace = Shape(R=7, W=2, A=3, L=1024, NB=96, IW=4, Q=14, D=15, P=670, K=14)
    served = Shape(R=8, W=1, A=9, L=512, NB=101626, IW=9, Q=4, D=32, P=54, K=4, AS=1)
    walk = Shape(R=8, W=1, A=9, L=512, NB=101626, IW=9, Q=4, D=32, P=54, K=4, AS=1, EW=17)
    return [f(S) for S in (small, trace, served, walk) for f in (advance_unit, load_unit, close_unit)]
