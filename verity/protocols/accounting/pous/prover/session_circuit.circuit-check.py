"""circuit-check's bindings of `verity.protocols.accounting.pous.prover.session_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """PoUS's session units (`verity.protocols.accounting.pous.prover.session_circuit`). The signature's unit at `gates.lms`'
    toy binding, N = 8 (`lms.TOY_N`), W = 1, H = 1: the toy verifier, 79 compressions, and q's 31 top bits held to zero.
    What it catches that the verifier's own binding can't is the wiring of the rows' bytes into the verifier's words: the
    key's I and root, the message, q's four bytes big-endian and their low H bits, C, the chain values and the path, and
    the top bits' check. The round's unit over a 264-byte row, which reaches the row digest's head, a middle compression and
    a tail with its rest, at both slots: the record's place in its row, its j and index big-endian against the public
    integers, and its digest. And R_even and R_odd as served, over the store's 2,056-byte rows, whose `digests` pins hold
    the two Programs the session's statements name."""
    from verity.circuits.defs import bind
    from verity.commitments.crypto.lms import TOY_N
    from verity.protocols.accounting.pous.prover.session_circuit import PousRound, PousSessionSig

    return [
        bind(PousSessionSig, N=TOY_N, W=1, H=1),
        bind(PousRound, N=264, SLOT=0),
        bind(PousRound, N=264, SLOT=1),
        bind(PousRound, N=2056, SLOT=0),
        bind(PousRound, N=2056, SLOT=1),
    ]
