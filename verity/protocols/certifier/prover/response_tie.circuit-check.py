"""circuit-check's bindings of `verity.protocols.certifier.prover.response_tie`'s Definitions, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_response_tie_roots() -> list[Any]:
    """The response tie's two proof blocks (`verity.protocols.certifier.prover.response_tie`: a response row of the
    network certifier's content rows computed from the egress row, the end row and, in a stop class, the stop word)
    at a row of 16 ids, a stop class's and a no-stop class's; at a stop class's row of 1,024 ids; and at the two
    sizes whose gates those lack: a no-stop class's row of 2 ids, the smallest, whose count has one bit below O's
    so one half of the comparison reads none, and a stop class's of 4,096, the largest, where the stop word has no
    bit past the count's. The largest is past the default whole-lowering budget: its correspondence and pin are
    checked with ``--lower-max-ands 200000``."""
    from verity.circuits.defs import bind
    from verity.protocols.certifier.prover.response_tie import ResponseTieNoStop, ResponseTieStop

    return [
        bind(ResponseTieStop, O=16),
        bind(ResponseTieNoStop, O=16),
        bind(ResponseTieStop, O=1024),
        bind(ResponseTieNoStop, O=2),
        bind(ResponseTieStop, O=4096),
    ]
