"""circuit-check's bindings of `verity.ml.commitments.gates.attest`'s Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _attest_roots() -> list[Any]:
    """The period unit (`verity.ml.commitments.gates.attest`) at `gates.lms`' toy binding, N = 8 (`lms.TOY_N`), W = 1,
    H = 1, over a memory challenger party's context on a device role of 8 bytes (L = 161): the three compressions of
    `period_root_message` over the arguments' bits, then the toy verifier, 82 compressions, and the domain id's one
    compression from its constant midstate with its equality. What it catches that the verifier's own binding can't is the
    messages' wiring: the tag, I's bytes, the context's length prefix and bytes, the count's bytes big-endian in both, the
    root's prefix and bytes, the domain message's constant bytes, and the padding. The messages are the same at N = 64
    apart from I, whose words the verifier reads the same way at both."""
    from verity.ml.commitments.gates.attest import period
    from verity.ml.crypto.lms import TOY_N

    return [period(TOY_N, 1, 1, 161)]
