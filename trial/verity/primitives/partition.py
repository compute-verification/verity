"""Partitions: a circuit's gates split into regions, the proof units of sampled proofs.

The builder chooses the partition; the verifier checks it has the properties the policy needs (`validate`: every gate
in exactly one region, and each region's boundary at most `max_width` wires). A region is checked through its boundary:
the wires it reads from outside (`inputs`) and the wires it produces that anything outside reads, or that the circuit
outputs (`outputs`). The baseline is the circuit's own, so it is on no boundary. Today's references: `verity.ir.cut`
(the width rule) and `verity.ir.partition_object`.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

from verity.primitives.circuit import OPS, Circuit

Partition = tuple[tuple[int, ...], ...]  # regions, each a tuple of gate indices


def wire_of(circuit: Circuit, gate: int) -> int:
    return circuit.inputs + gate


def boundary(circuit: Circuit, region: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(inputs, outputs) of `region`, each sorted by wire."""
    produced = {wire_of(circuit, g) for g in region}
    read = {w for g in region for w in circuit.gates[g][1:]}
    inside, readers, circuit_outputs = set(region), _readers(circuit), set(circuit.outputs)
    outputs = {w for w in produced if w in circuit_outputs or any(r not in inside for r in readers.get(w, ()))}
    return tuple(sorted(w for w in read - produced if w >= len(circuit.baseline))), tuple(sorted(outputs))


@functools.cache
def _readers(circuit: Circuit) -> dict[int, tuple[int, ...]]:
    """Each wire's reading gates."""
    readers: dict[int, list[int]] = {}
    for g, (_, a, b) in enumerate(circuit.gates):
        for w in {a, b}:
            readers.setdefault(w, []).append(g)
    return {w: tuple(gs) for w, gs in readers.items()}


def evaluate_region(circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int]) -> dict[int, int]:
    """Every wire `region` produces, from its boundary inputs."""
    wires = {**dict(enumerate(circuit.baseline)), **inputs}
    for g in sorted(region):
        op, a, b = circuit.gates[g]
        wires[wire_of(circuit, g)] = OPS[op](wires[a], wires[b])
    return {wire_of(circuit, g): wires[wire_of(circuit, g)] for g in region}


def validate(circuit: Circuit, partition: Partition, max_width: int) -> tuple[str, ...]:
    why = []
    seen = sorted(g for region in partition for g in region)
    if seen != list(range(len(circuit.gates))):
        why.append("the regions do not cover every gate exactly once")
    for i, region in enumerate(partition):
        ins, outs = boundary(circuit, region)
        if len(ins) + len(outs) > max_width:
            why.append(f"region {i}: {len(ins) + len(outs)} boundary wires, over the width {max_width}")
    return tuple(why)


def whole(circuit: Circuit) -> Partition:
    """The circuit as one region: enough for replay, which needs no finer partition."""
    return (tuple(range(len(circuit.gates))),)


def greedy(circuit: Circuit, max_width: int) -> Partition:
    """A builder's partition: consecutive gates, each region grown while its boundary stays within `max_width` wires."""
    regions: list[tuple[int, ...]] = []
    start = 0
    while start < len(circuit.gates):
        end = start + 1
        while end < len(circuit.gates) and sum(map(len, boundary(circuit, range(start, end + 1)))) <= max_width:
            end += 1
        regions.append(tuple(range(start, end)))
        start = end
    return tuple(regions)
