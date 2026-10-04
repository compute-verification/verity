"""Partitions: a circuit's gates split into regions, the proof units of sampled proofs.

The prover chooses the partition; the verifier checks it has the properties the policy needs (`validate`: every gate in
exactly one region, and each region's boundary at most `max_width` wires). A region is checked through its boundary: the
wires it reads from outside (`inputs`) and the wires it produces that anything outside reads, or that the circuit
outputs (`outputs`). Today's references: `verity.ir.cut` (the width rule) and `verity.ir.partition_object`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from verity.primitives.circuit import OPS, Circuit

Partition = tuple[tuple[int, ...], ...]  # regions, each a tuple of gate indices


def wire_of(circuit: Circuit, gate: int) -> int:
    return circuit.inputs + gate


def boundary(circuit: Circuit, region: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(inputs, outputs) of `region`, each sorted by wire."""
    produced = {wire_of(circuit, g) for g in region}
    read = {w for g in region for w in circuit.gates[g][1:]}
    inside = set(region)
    read_outside = {w for g, (_, a, b) in enumerate(circuit.gates) if g not in inside for w in (a, b)}
    outputs = produced & (read_outside | set(circuit.outputs))
    return tuple(sorted(read - produced)), tuple(sorted(outputs))


def evaluate_region(circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int]) -> dict[int, int]:
    """Every wire `region` produces, from its boundary inputs."""
    wires = dict(inputs)
    for g in sorted(region):
        op, a, b = circuit.gates[g]
        wires[wire_of(circuit, g)] = OPS[op](wires[a], wires[b])
    return {w: v for w, v in wires.items() if w not in inputs}


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


def digest(circuit: Circuit, partition: Partition) -> str:
    return hashlib.sha256(json.dumps([circuit.id(), partition], separators=(",", ":")).encode()).hexdigest()[:16]


def whole(circuit: Circuit) -> Partition:
    """The circuit as one region: replay's, which needs no finer partition."""
    return (tuple(range(len(circuit.gates))),)


def greedy(circuit: Circuit, max_width: int) -> Partition:
    """A prover's partition: consecutive gates, each region grown while its boundary stays within `max_width` wires."""
    readers: dict[int, list[int]] = {}
    for g, (_, a, b) in enumerate(circuit.gates):
        for w in {a, b}:
            readers.setdefault(w, []).append(g)
    outputs = set(circuit.outputs)
    regions: list[tuple[int, ...]] = []
    start = 0
    while start < len(circuit.gates):
        end = start + 1
        while end < len(circuit.gates) and _width(circuit, start, end + 1, readers, outputs) <= max_width:
            end += 1
        regions.append(tuple(range(start, end)))
        start = end
    return tuple(regions)


def _width(circuit: Circuit, start: int, end: int, readers: Mapping[int, list[int]], outputs: set[int]) -> int:
    produced = {wire_of(circuit, g) for g in range(start, end)}
    ins = {w for g in range(start, end) for w in circuit.gates[g][1:] if w not in produced}
    outs = {w for w in produced if w in outputs or any(not start <= r < end for r in readers.get(w, ()))}
    return len(ins) + len(outs)
