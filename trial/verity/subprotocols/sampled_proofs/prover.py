"""The prover of sampled proofs, as a party: the reference an integration's own (vLLM's, on the GPU) must match.

It runs each execution once, commits to its boundary values and sends the registration, keeping only each execution's
inputs: traces are not stored. When the draw comes, it rematerializes each drawn unit's execution by replay, opens the
unit's boundary against the root it committed, and proves it.
"""

from __future__ import annotations

from collections.abc import Sequence

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, trace
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.events import Party, Send, receive
from verity.subprotocols.sampled_proofs import registration as R
from verity.subprotocols.sampled_proofs.construction import Construction
from verity.subprotocols.sampled_proofs.events import Draw, Opened, Opening, Registered


def prover(
    circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]], scheme: CommitmentScheme,
    construction: Construction, verifier: str = "verifier",
) -> Party:
    wires = R.boundary_wires(circuit, partition)

    def leaves(values: tuple[int, ...]) -> list[bytes]:
        return [R.leaf(w, values[w]) for w in wires]

    registration = R.Registration(tuple(R.Execution(circuit.id(), partition, scheme.commit(leaves(trace(circuit, x)))) for x in inputs))
    draw = (yield from receive(Draw, Send(Registered(registration), to=(verifier,)))).body

    def open_unit(u: int) -> Opening:
        i, r = registration.unit(u)
        values = trace(circuit, inputs[i])
        ins, outs = P.boundary(circuit, partition[r])
        opened = {w: values[w] for w in (*ins, *outs)}
        proofs = {w: scheme.open(leaves(values), wires.index(w)) for w in opened}
        return Opening(opened, proofs, construction.prove(circuit, partition[r], {w: values[w] for w in ins}))

    return (Send(Opened({u: open_unit(u) for u in draw.units}), to=(verifier,)),)
