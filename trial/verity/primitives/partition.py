"""Partitions: a circuit's gates split into isolation units, and each isolation unit into regions.

Two readers need the circuit cut at two scales. NCI bounds each isolation unit: its gates, and the novel bits it reads.
Sampled proofs draw regions, the proof units, which must be narrow enough to commit and prove one at a time. Every
region lies in one isolation unit, so a wire that crosses an isolation unit's boundary crosses a region's, and the
commitment to the regions' boundaries holds every wire NCI reads (in the main repo, a `verity.ir.partition.Hierarchy`
of two levels, the finer refining the coarser).

The builder chooses the partition; the verifier checks it has the properties the policy needs (`validate`: every gate
in exactly one region, and each region's boundary at most `max_width` wires). A set of gates is checked through its
boundary: the wires it reads from outside (`inputs`) and the wires it produces that anything outside reads, or that
the circuit outputs (`outputs`). The baseline is the circuit's own, so it is on no boundary. Today's references:
`verity.ir.cut` (the width rule) and `verity.ir.partition_object`.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

from verity.primitives.circuit import OPS, Circuit

Region = tuple[int, ...]  # gate indices
Partition = tuple[tuple[Region, ...], ...]  # isolation units, each split into regions


def regions(partition: Partition) -> tuple[Region, ...]:
    """Every region, isolation unit by isolation unit."""
    return tuple(region for unit in partition for region in unit)


def gates(unit: Sequence[Region]) -> Region:
    """An isolation unit's gates."""
    return tuple(g for region in unit for g in region)


@functools.cache
def unit_of(partition: Partition) -> dict[int, int]:
    """Each gate's isolation unit."""
    return {g: u for u, unit in enumerate(partition) for g in gates(unit)}


def wire_of(circuit: Circuit, gate: int) -> int:
    return circuit.inputs + gate


def boundary(circuit: Circuit, gates: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(inputs, outputs) of a set of gates, each sorted by wire."""
    produced = {wire_of(circuit, g) for g in gates}
    read = {w for g in gates for w in circuit.gates[g][1:]}
    inside, readers, circuit_outputs = set(gates), _readers(circuit), set(circuit.outputs)
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
    seen = sorted(g for region in regions(partition) for g in region)
    if seen != list(range(len(circuit.gates))):
        why.append("the regions do not cover every gate exactly once")
    for i, region in enumerate(regions(partition)):
        ins, outs = boundary(circuit, region)
        if len(ins) + len(outs) > max_width:
            why.append(f"region {i}: {len(ins) + len(outs)} boundary wires, over the width {max_width}")
    return tuple(why)


def whole(circuit: Circuit) -> Partition:
    """The circuit as one isolation unit of one region: enough for replay, which needs no finer partition."""
    return ((tuple(range(len(circuit.gates))),),)


def greedy(circuit: Circuit, max_width: int) -> Partition:
    """The circuit as one isolation unit of consecutive regions, each grown while its boundary stays within
    `max_width` wires."""
    unit: list[Region] = []
    start = 0
    while start < len(circuit.gates):
        end = start + 1
        while end < len(circuit.gates) and sum(map(len, boundary(circuit, range(start, end + 1)))) <= max_width:
            end += 1
        unit.append(tuple(range(start, end)))
        start = end
    return (tuple(unit),)
