"""The verifier of sampled verification, as a party:

    1. receive the prover's registration: its computation as units, and its commitment to every unit's boundary
    2. check it against the verifier's own copies of the circuits
    3. draw units from the verifier's secret, bound to the registration, and send the draw
    4. receive a proof of each drawn unit, and verify each with the proof system
    5. record the profile: if every drawn unit verifies, at most `wrong_units` are wrong, except with probability δ

Today's references: `verity_sampled_proofs.one_stage` and `verity.proofs.profile.IntegrityProfile`.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping

from verity.primitives.circuit import Circuit
from verity.primitives.events import Party, Send, receive
from verity.primitives.randomness import derive
from verity.subprotocols.verification.sampling import registration as R
from verity.subprotocols.verification.sampling.events import Draw, IntegrityProfile, Proved, Registered
from verity.subprotocols.verification.sampling.law import Subset
from verity.subprotocols.verification.units import ProofSystem


def verifier(
    circuits: Mapping[str, Circuit], law: Subset, max_width: int, delta: float, proof_system: ProofSystem,
    prover: str = "prover",
) -> Party:
    secret = secrets.token_bytes(32)
    registration = (yield from receive(Registered)).body.registration
    digest, n = registration.digest(), registration.population

    refused = R.check(registration, circuits, max_width, proof_system.scheme.name)
    if refused:
        return (Send(IntegrityProfile(digest, False, n, n, delta, refused)),)

    units = law.draw(derive(secret, "verity/verification/units/v1", digest), n)
    proofs = (yield from receive(Proved, Send(Draw(units), to=(prover,)))).body.proofs

    reasons = tuple(
        f"unit {u}: " + ("not proved" if u not in proofs else "does not verify")
        for u in units if u not in proofs or not proof_system.verify(registration.unit(u, circuits), proofs[u])
    )
    return (Send(IntegrityProfile(digest, not reasons, n, law.bound(n, delta), delta, reasons)),)
