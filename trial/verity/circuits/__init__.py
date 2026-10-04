"""The circuit library: build, name, evaluate and trace circuits.

What a circuit is and what it computes is accounting's definition (`verity/protocol/lean/Circuit.lean`); this library
implements it and is held to it by conformance vectors, never named by the configuration. A toy: gates over integer
wires mod 2**32.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

MASK = 2**32 - 1
OPS = {"add": lambda a, b: (a + b) & MASK, "mul": lambda a, b: (a * b) & MASK, "xor": lambda a, b: a ^ b}


@dataclass(frozen=True)
class Circuit:
    inputs: int
    gates: tuple[tuple[str, int, int], ...]  # (op, wire, wire); wire i < inputs is input i, then one wire per gate
    outputs: tuple[int, ...]

    def id(self) -> str:
        body = {"inputs": self.inputs, "gates": self.gates, "outputs": self.outputs}
        return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()[:16]


def trace(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    """Every wire's value: the replay's witness."""
    wires = list(inputs)
    for op, a, b in circuit.gates:
        wires.append(OPS[op](wires[a], wires[b]))
    return tuple(wires)


def evaluate(circuit: Circuit, inputs: tuple[int, ...]) -> tuple[int, ...]:
    wires = trace(circuit, inputs)
    return tuple(wires[o] for o in circuit.outputs)
