"""The prover of sampled verification, as a party: the reference an integration's own (vLLM's, on the GPU) must match.

It runs each execution once, commits to its boundary values and sends the registration, keeping only each execution's
inputs: traces are not stored. When the draw comes, it rematerializes each drawn unit's execution by replay and proves
the unit with the proof system.
"""

from __future__ import annotations

from collections.abc import Sequence

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, trace
from verity.primitives.events import Party, Send, receive
from verity.subprotocols.verification import units as U
from verity.subprotocols.verification.sampling import registration as R
from verity.subprotocols.verification.sampling.events import Draw, Proved, Registered


def prover(
    circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]], proof_system: U.ProofSystem,
    verifier: str = "verifier",
) -> Party:
    commit = proof_system.scheme.commit
    registration = R.Registration(tuple(
        R.Execution(circuit.id(), partition, commit(U.leaves(circuit, partition, trace(circuit, x)))) for x in inputs
    ))
    draw = (yield from receive(Draw, Send(Registered(registration), to=(verifier,)))).body

    def prove(u: int) -> object:
        i, _ = registration.locate(u)
        return proof_system.prove(registration.unit(u, {circuit.id(): circuit}), trace(circuit, inputs[i]))

    return (Send(Proved({u: prove(u) for u in draw.units}), to=(verifier,)),)
