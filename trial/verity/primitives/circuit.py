"""Circuits: build, name, evaluate and trace them.

What a circuit is and what it computes is stated in `Circuit.lean`, beside this file; this implements it and is held
to it by conformance vectors. A toy of descriptor v1: one flat list of two-input gates. Word gates (`add`, `mul`) act on wires mod 2**32; the Boolean profile's gates (`and`,
`xor`, `not`, which ignores its second operand) on bits, as `verity.ir.boolean` does.
"""

from __future__ import annotations

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
    baseline: tuple[int, ...] = ()  # the inputs bound when it is registered (weights, constants); the rest per run

    def id(self) -> str:
        body = {"inputs": self.inputs, "gates": self.gates, "outputs": self.outputs, "baseline": self.baseline}
        return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()[:16]

    @classmethod
    def from_json(cls, obj: dict) -> Circuit:
        """The flat format `examples/data/*.json` carries: `inputs`, `gates` as [op, wire, wire], `outputs`, and the
        `constants` its first inputs hold, which are its baseline."""
        gates = tuple((op, int(a), int(b)) for op, a, b in obj["gates"])
        return cls(int(obj["inputs"]), gates, tuple(obj["outputs"]), tuple(range(len(obj.get("constants", ())))))


def trace(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    """Every wire's value: the replay's witness."""
    wires = list(inputs)
    for op, a, b in circuit.gates:
        wires.append(OPS[op](wires[a], wires[b]))
    return tuple(wires)


def evaluate(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    wires = trace(circuit, inputs)
    return tuple(wires[o] for o in circuit.outputs)
