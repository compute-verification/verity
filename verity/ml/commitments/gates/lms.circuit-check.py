"""circuit-check's bindings of `verity.ml.commitments.gates.lms`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _lms_roots() -> list[Any]:
    """The LMS verifier (`verity.ml.commitments.gates.lms`) at a toy binding, N = 8 (`lms.TOY_N`, one SHA-512 word per
    value), W = 1, H = 1: 71 chains of one step (64 message digits and a checksum of 7), a five-block one-time public key, the
    leaf, one tree level and the comparison, 79 compressions. The toy is never deployed: the smallest production binding
    (N = 64, W = 1, H = 1) is 789 compressions, and correspondence on it does not finish; the package's tests keep its agreement
    with `verity.ml.crypto.lms` there. What the toy catches that the parts' own checks can't is the wiring: the digits
    fed to the chain steps' batch axes, each chain's index and step constants, the chain ends laid out into K's blocks, q's bits
    under the constant 1 as node numbers, and the side each level's swap takes. The parts are also bound at N = 64, a
    compression or two each, so production's layout keeps its own checks (its message hash and its tree nodes are two blocks,
    which no N below 26 and 45 reaches)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.lms import LmsChainStep, LmsEqual, LmsLeaf, LmsMessage, LmsNode, verifier
    from verity.ml.crypto.lms import TOY_N

    return [verifier(TOY_N, 1, 1), bind(LmsMessage, N=64, W=1), bind(LmsChainStep, N=64, W=1), bind(LmsLeaf, N=64),
            bind(LmsNode, N=64), bind(LmsEqual, N=64)]
