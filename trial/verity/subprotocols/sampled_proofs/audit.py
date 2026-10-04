"""The one-stage audit (the 3-pager's protocol):

    1. the prover commits to its computation as units: per execution, the circuit and a partition it chose
    2. it commits to every unit's inputs and outputs (`registration`), and the verifier checks the registration
    3. the verifier draws units, with coins drawn after the registration exists
    4. the prover opens each drawn unit's boundary and proves it (rematerializing what it needs by an untrusted replay)
    5. if every drawn unit verifies, at most `bound` units are wrong, except with probability δ

The prover's side answers `register()` (the registration and the circuits it names) and `open_unit(u)` (the drawn
unit's boundary values with their openings, and the construction's witness). Today's references:
`verity_sampled_proofs.one_stage` (registration, draw, audit record) and `verity.proofs.profile.IntegrityProfile`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.circuits import partition as P
from verity.primitives import registry
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.randomness import Challenger
from verity.subprotocols.sampled_proofs import registration as R
from verity.subprotocols.sampled_proofs.law import Subset
from verity.subprotocols.sampled_proofs.replay import Replay

SHIPPED = {Replay.name: Replay}


@dataclass(frozen=True)
class Opening:
    values: Mapping[int, int]  # every boundary wire of the unit: inputs and outputs
    proofs: Mapping[int, object]  # each value's opening against the execution's root
    witness: object  # the construction's: nothing for replay, a proof for a proof system


@dataclass(frozen=True)
class IntegrityProfile:
    accepted: bool
    delta: float  # the probability of accepting with more than `wrong_units` wrong units
    covered: frozenset[str]  # the circuits whose units the draw was over
    units: int
    wrong_units: int
    record: Mapping[str, object]  # the audit record: registration digest, draw, verdict per drawn unit


class Audit:
    def __init__(self, construction: str, parameters: Mapping[str, object], delta: float, scheme: CommitmentScheme):
        self.construction = registry.load("sampled_proofs", construction, SHIPPED)
        self.law = Subset(int(parameters.get("k", 8)))
        self.max_width = int(parameters.get("max_width", 4))
        self.delta, self.scheme = delta, scheme

    def open(self, challenger: Challenger) -> None:
        self.challenger = challenger

    def close(self, declared: Sequence[str], prover: object) -> IntegrityProfile:
        """`declared`: the circuit of each execution the account declares, in order."""
        registration, circuits = prover.register()
        refused = R.check(registration, circuits, declared, self.law, self.max_width, self.scheme.name)
        digest = registration.digest()
        if refused:
            return self._profile(False, registration, {"registration": digest, "refused": list(refused)})

        drawn = self.law.draw(self.challenger.draw("units", registration=digest), registration.population)
        verdicts = {u: self._verify(registration, circuits, u, prover.open_unit(u)) for u in drawn}
        record = {"registration": digest, "drawn": list(drawn), "verdicts": verdicts}
        return self._profile(all(verdicts.values()), registration, record)

    def _verify(self, registration: R.Registration, circuits: Mapping[str, object], u: int, opening: Opening) -> bool:
        i, r = registration.unit(u)
        execution = registration.executions[i]
        circuit = circuits[execution.circuit]
        region = execution.partition[r]
        ins, outs = P.boundary(circuit, region)
        index = {w: j for j, w in enumerate(R.boundary_wires(circuit, execution.partition))}
        if set(opening.values) != set(ins) | set(outs):
            return False
        for w, v in opening.values.items():
            if not self.scheme.verify(execution.root, index[w], R.leaf(w, v), opening.proofs[w]):
                return False
        inputs = {w: opening.values[w] for w in ins}
        outputs = {w: opening.values[w] for w in outs}
        return self.construction.verify_unit(circuit, region, inputs, outputs, opening.witness)

    def _profile(self, accepted: bool, registration: R.Registration, record: Mapping[str, object]) -> IntegrityProfile:
        n = registration.population
        return IntegrityProfile(
            accepted=accepted,
            delta=self.delta,
            covered=frozenset(e.circuit for e in registration.executions),
            units=n,
            wrong_units=self.law.bound(n, self.delta),
            record=record,
        )
