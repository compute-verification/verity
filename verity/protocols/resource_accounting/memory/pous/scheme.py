"""PoUS and the memory wipe as resource schemes (`verity.protocols.resource_accounting.scheme`), registered as ``pous``
and ``wipe``.

**PoUS** is decided by `PousCheck.RunProtocol.protocol` (`verifier/check/PousVerifier/RunProtocol.lean`), whose
`Core.Account` row is memory held in the GPUs' HBM, verified from below. It adds no block to the registered circuit yet:
its setup unit, `PousP2SetupCheck_v1` (`prover/setup_circuit.py`), is in no window, and what it certifies is a share ρ of
the store `C*`, not a cost per block, so it states what it requires of the proof layer instead (`requires`). Its
certificate is P2's at the store's block count (`verifier.period.certificate`): up to `2^23` blocks
`Pous.SecurityProofs.P2SlackFamilyUncond`, in abstract M1 with no named assumption; above, `periodHeldAt`, which takes
`P2RingRoundSecureWide` and `RingDeviceAt`. Onsite's soundness takes PoUS's target, `PousSound`, as a premise. A device's
whole chip and its side channels' capacity over a round are counted in the state the ring game bounds
(`ResourceAccounting.RealizesState`). The vLLM adapter attaches it as ``pous:p2-16448/v4``: it reads the declared build
(``declared``) and the registered weights once resident (``memory``), and computes each linear's block with the weight
decoded from `C*` (``blocks``).

**The wipe** is decided by `MemoryChallenger.WipeProtocol.protocol` (`attester/lean/MemoryAttester/WipeProtocol.lean`),
whose row bounds the foreign bits a server may still hold after an accepted wipe, from above: a device's game with no
block in the circuit and no adapter option. `Pous.SecurityProofs.StreamedWipe` is proved with no named assumption;
onsite's soundness takes the wipe's target, `WipeSound`, as a premise.
"""

from __future__ import annotations

from verity.commitments import domains
from verity.protocols.resource_accounting.scheme import Charge, Order, Scheme, register

TRUSTED_DEVICE = "Verity.Assumptions.TrustedDevice"

POUS = register(Scheme(
    name="pous",
    protocol=domains.POUS_P2_RUN,
    resource="memory",
    blocks=(),
    assumptions=("Verity.Vocab.PousSound", TRUSTED_DEVICE, "Pous.P2.Assumptions.P2RingRoundSecureWide",
                 "Pous.Period.RingDeviceAt"),
    log=(Order("weights", "salt"), Order("store", "challenges")),
    charges=(Charge("chip", "ResourceAccounting.RealizesState"),
             Charge("side-channels", "ResourceAccounting.RealizesState")),
    settings={"challenges": 320},
    option="pous:p2-16448/v4",
    hooks=("declared", "memory", "blocks"),
    requires={
        "one-labelling": "C*'s own tree over the memory attester's row digests (`VerifierKey.from_row_digests`); "
                         "`PousP2SetupCheck_v1` is in no window, and setup's tie decodes C* in Python each period",
        "log-order": "its own salt, `setup.sample_salt(source, commitment)` after the commitment to W, and the memory "
                     "attester's coin tree, its session header bound to setup's root",
        "log-times": "the memory attester's own clock: each answer within Δ + RTT of its index, in the rounds it signs",
        "coins-at-rate": "the memory attester's coin tree, committed before the session",
        "one-design": "its own: `challenges` indices of C*'s blocks, uniform, revealed one at a time",
    },
))

WIPE = register(Scheme(
    name="wipe",
    protocol=domains.WIPE_PROTOCOL,
    resource="memory",
    blocks=(),
    assumptions=("Verity.Vocab.WipeSound", TRUSTED_DEVICE),
    log=(Order("fill", "reveal"),),
    charges=(),
    requires={
        "one-labelling": "its own labels, SHA-512 under a key fresh for the server and the wipe, derived by the memory "
                         "attester",
        "log-order": "the memory attester's own: it flips its coins once the fill is complete",
        "log-times": "the memory attester's own clock: a deadline per batch, in the session it signs",
        "coins-at-rate": "the memory attester's own coins",
        "one-design": "its own: `k` indices of the fill, revealed one at a time",
    },
))

__all__ = ["POUS", "WIPE"]
