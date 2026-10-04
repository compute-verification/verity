"""Circuits: what they compute, how they are cut into units, and the builders that make them.

What a circuit is and what it computes is stated in `Circuits.lean`, beside this file; this implements it. A toy of
descriptor v1's Boolean profile, as `verity.ir.boolean` is: one flat list of two-input gates (`and`, `xor`, `not`, which
ignores its second operand) on bits, so a wire is a bit, and a count of wires a count of bits. A transcript is every
wire's value.

A circuit is one execution of a stream, and its inputs say where they come from: first the baseline, the same in every
execution (weights, constants); then its state, which earlier executions of the stream wrote (a decode step's KV
cache); then the rest, fresh.

A partition cuts a circuit's gates at two scales: into inference units (IUs), which NCI bounds (`Nci.Protocol` in the
main repo calls them isolation units), and each inference unit into regions, which are narrow enough to commit and
check one at a time. Every region lies in one inference unit, so a wire that crosses an inference unit's boundary
crosses a region's (in the main repo, a `verity.ir.partition.Hierarchy` of two levels). A set of gates meets the rest
of the circuit at its boundary: the wires it reads from outside, and the wires it produces that anything outside reads
or that the circuit outputs. The baseline is the circuit's own, so it is on no boundary. Today's references:
`verity.ir.cut` (the width rule) and `verity.ir.partition_object`.

A builder makes each execution's circuit and partition from that execution's advice: a bitstring of at most
`advice_bits`, the record of the runtime choices no one could fix ahead of time (how a scheduler batched requests, which
step a session continues, a preemption, an autotune pick). The builder here is Python, standing in for a provable
program (Rust under SP1, say) whose id is its program's.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

OPS = {
    "xor": lambda a, b: a ^ b,
    "and": lambda a, b: a & b,
    "not": lambda a, b: a ^ 1,
}


@dataclass(frozen=True)
class Circuit:
    inputs: int
    gates: tuple[tuple[str, int, int], ...]  # (op, wire, wire); wire i < inputs is input i, then one wire per gate
    outputs: tuple[int, ...]
    baseline: tuple[int, ...] = ()  # what its first inputs hold in every execution (weights, constants)
    state: tuple[tuple[int, int], ...] = ()  # per state input: (executions back, which of the writer's outputs)

    def __hash__(self) -> int:
        return self._hash

    @functools.cached_property
    def _hash(self) -> int:  # a circuit keys caches, and hashing its gates is linear in them
        return hash((self.inputs, self.gates, self.outputs, self.baseline, self.state))

    def id(self) -> str:
        body = {
            "inputs": self.inputs,
            "gates": self.gates,
            "outputs": self.outputs,
            "baseline": self.baseline,
            "state": self.state,
        }
        return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()[:16]

    def carried(self, wire: int) -> tuple[int, int] | None:
        """Where a state input's value was written, (executions back, which of its outputs); None for any other wire."""
        k = wire - len(self.baseline)
        return self.state[k] if 0 <= k < len(self.state) else None

    @classmethod
    def from_json(cls, obj: dict) -> Circuit:
        """The flat format `examples/data/*.json` carries: `inputs`, `gates` as [op, wire, wire], `outputs`, and the
        `constants` its first inputs hold, which are its baseline."""
        gates = tuple((op, int(a), int(b)) for op, a, b in obj["gates"])
        return cls(int(obj["inputs"]), gates, tuple(obj["outputs"]), tuple(obj.get("constants", ())))


def transcript(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    """Every wire's value, from the inputs after the baseline (its state, then the fresh ones)."""
    wires = [*circuit.baseline, *inputs]
    for op, a, b in circuit.gates:
        wires.append(OPS[op](wires[a], wires[b]))
    return tuple(wires)


def evaluate(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    wires = transcript(circuit, inputs)
    return tuple(wires[o] for o in circuit.outputs)


Region = tuple[int, ...]  # gate indices
Partition = tuple[tuple[Region, ...], ...]  # inference units, each split into regions


def regions(partition: Partition) -> tuple[Region, ...]:
    """Every region, inference unit by inference unit."""
    return tuple(region for unit in partition for region in unit)


def gates(unit: Sequence[Region]) -> Region:
    """An inference unit's gates."""
    return tuple(g for region in unit for g in region)


@functools.cache
def unit_of(partition: Partition) -> dict[int, int]:
    """Each gate's inference unit."""
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


def consistent(circuit: Circuit, gates: Sequence[int], values: Mapping[int, int]) -> bool:
    """Whether `values`, bits on exactly the boundary of `gates`, are what the gates compute: run from the values on
    their inputs, they give the values on their outputs."""
    ins, outs = boundary(circuit, gates)
    if set(values) != {*ins, *outs} or set(values.values()) - {0, 1}:
        return False
    produced = evaluate_region(circuit, gates, {w: values[w] for w in ins})
    return all(produced[w] == values[w] for w in outs)


def covers(circuit: Circuit, partition: Partition) -> bool:
    """Whether the regions hold every gate exactly once."""
    return sorted(g for region in regions(partition) for g in region) == list(range(len(circuit.gates)))


def validate(circuit: Circuit, partition: Partition, max_width: int) -> tuple[str, ...]:
    """What keeps `partition` from cutting `circuit` into regions of at most `max_width` boundary wires."""
    why = []
    if not covers(circuit, partition):
        why.append("the regions do not cover every gate exactly once")
    for i, region in enumerate(regions(partition)):
        ins, outs = boundary(circuit, region)
        if len(ins) + len(outs) > max_width:
            why.append(f"region {i}: {len(ins) + len(outs)} boundary wires, over the width {max_width}")
    return tuple(why)


def whole(circuit: Circuit) -> Partition:
    """The circuit as one inference unit of one region."""
    return ((tuple(range(len(circuit.gates))),),)


def greedy(circuit: Circuit, max_width: int) -> Partition:
    """The circuit as one inference unit of consecutive regions, each grown while its boundary stays within
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


class Builder(Protocol):
    advice_bits: int  # the most advice one execution may take

    def id(self) -> str: ...

    def build(self, advice: str) -> tuple[Circuit, Partition]:
        """The execution's circuit and partition; raises ValueError on advice it cannot read."""
        ...


@functools.cache
def built(builder: Builder, advice: str) -> tuple[Circuit, Partition]:
    """What `builder` builds from `advice`; raises ValueError if the advice is not a bitstring it may take, or what it
    builds holds a baseline that is not bits."""
    if not isinstance(advice, str) or len(advice) > builder.advice_bits or set(advice) - {"0", "1"}:
        raise ValueError(f"advice {advice!r} is not a bitstring of at most {builder.advice_bits} bits")
    circuit, partition = builder.build(advice)
    if set(circuit.baseline) - {0, 1}:
        raise ValueError(f"advice {advice!r} builds a baseline that is not bits")
    return circuit, partition
