"""Circuits: build, name and evaluate them, and give their transcripts (every wire's value).

What a circuit is and what it computes is stated in `Circuit.lean`, beside this file; this implements it and is held
to it by conformance vectors. A toy of descriptor v1: one flat list of two-input gates. Word gates (`add`, `mul`) act
on wires mod 2**32; the Boolean profile's gates (`and`, `xor`, `not`, which ignores its second operand) on bits, as
`verity.ir.boolean` does.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass

MASK = 2**32 - 1
OPS = {
    "add": lambda a, b: (a + b) & MASK,
    "mul": lambda a, b: (a * b) & MASK,
    "xor": lambda a, b: a ^ b,
    "and": lambda a, b: a & b,
    "not": lambda a, b: a ^ 1,
}


@dataclass(frozen=True)
class Circuit:
    inputs: int
    gates: tuple[tuple[str, int, int], ...]  # (op, wire, wire); wire i < inputs is input i, then one wire per gate
    outputs: tuple[int, ...]
    baseline: tuple[int, ...] = ()  # what its first inputs hold in every execution (weights, constants); the rest vary

    def __hash__(self) -> int:
        return self._hash

    @functools.cached_property
    def _hash(self) -> int:  # a circuit keys caches, and hashing its gates is linear in them
        return hash((self.inputs, self.gates, self.outputs, self.baseline))

    def id(self) -> str:
        body = {"inputs": self.inputs, "gates": self.gates, "outputs": self.outputs, "baseline": self.baseline}
        return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()[:16]

    @classmethod
    def from_json(cls, obj: dict) -> Circuit:
        """The flat format `examples/data/*.json` carries: `inputs`, `gates` as [op, wire, wire], `outputs`, and the
        `constants` its first inputs hold, which are its baseline."""
        gates = tuple((op, int(a), int(b)) for op, a, b in obj["gates"])
        return cls(int(obj["inputs"]), gates, tuple(obj["outputs"]), tuple(obj.get("constants", ())))


def transcript(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    """Every wire's value, from the inputs after the baseline."""
    wires = [*circuit.baseline, *inputs]
    for op, a, b in circuit.gates:
        wires.append(OPS[op](wires[a], wires[b]))
    return tuple(wires)


def evaluate(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    wires = transcript(circuit, inputs)
    return tuple(wires[o] for o in circuit.outputs)
