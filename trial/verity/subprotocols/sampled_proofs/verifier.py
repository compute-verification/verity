"""The verifier of sampled proofs, as a party:

    1. receive the prover's registration: its computation as units, and its commitment to every unit's boundary
    2. check it against the verifier's own copies of the circuits
    3. draw units from the verifier's secret, bound to the registration, and send the draw
    4. receive the drawn units, opened and proved, and check each opening and each proof
    5. record the profile: if every drawn unit verifies, at most `wrong_units` are wrong, except with probability δ

Today's references: `verity_sampled_proofs.one_stage` and `verity.proofs.profile.IntegrityProfile`.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.events import Party, Send, receive
from verity.primitives.randomness import derive
from verity.subprotocols.sampled_proofs import registration as R
from verity.subprotocols.sampled_proofs.construction import Construction
from verity.subprotocols.sampled_proofs.events import Draw, IntegrityProfile, Opened, Opening, Registered
from verity.subprotocols.sampled_proofs.law import Subset


def verifier(
    circuits: Mapping[str, Circuit], law: Subset, max_width: int, delta: float, scheme: CommitmentScheme,
    construction: Construction, prover: str = "prover",
) -> Party:
    secret = secrets.token_bytes(32)
    registration = (yield from receive(Registered)).body.registration
    digest, n = registration.digest(), registration.population

    refused = R.check(registration, circuits, max_width, scheme.name)
    if refused:
        return (Send(IntegrityProfile(digest, False, n, n, delta, refused)),)

    units = law.draw(derive(secret, "verity/sampled_proofs/units/v1", digest), n)
    openings = (yield from receive(Opened, Send(Draw(units), to=(prover,)))).body.openings

    reasons = tuple(f"unit {u}: {why}" for u in units if (why := _verify(registration, circuits, scheme, construction, u, openings.get(u))))
    return (Send(IntegrityProfile(digest, not reasons, n, law.bound(n, delta), delta, reasons)),)


def _verify(
    registration: R.Registration, circuits: Mapping[str, Circuit], scheme: CommitmentScheme, construction: Construction,
    u: int, opening: Opening | None,
) -> str | None:
    if opening is None:
        return "not opened"
    i, r = registration.unit(u)
    execution = registration.executions[i]
    circuit, region = circuits[execution.circuit], execution.partition[r]
    ins, outs = P.boundary(circuit, region)
    if set(opening.values) != {*ins, *outs}:
        return "does not open exactly its boundary"
    index = {w: j for j, w in enumerate(R.boundary_wires(circuit, execution.partition))}
    if not all(scheme.verify(execution.root, index[w], R.leaf(w, v), opening.proofs[w]) for w, v in opening.values.items()):
        return "an opening does not match the root"
    inputs, outputs = {w: opening.values[w] for w in ins}, {w: opening.values[w] for w in outs}
    if not construction.verify(circuit, region, inputs, outputs, opening.witness):
        return "does not verify"
    return None
