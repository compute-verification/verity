"""What the examples share: a real circuit, eight requests to serve through it, and printing a view.

The circuits are verity's FP32 adder and BF16 comparison, exported as Boolean circuits with reference vectors from
`verity.evaluation.bits.evaluate_bits` (`data/`). A request is a pair of random floats.
"""

import json
import random
import struct
from dataclasses import is_dataclass
from pathlib import Path

from verity.primitives.circuit import Circuit, evaluate
from verity.primitives.events import Event


def load(name: str) -> tuple[Circuit, list[tuple[int, ...]], frozenset[int]]:
    """The circuit, the requests, and the baseline wires (its constants)."""
    data = json.loads((Path(__file__).parent / "data" / f"{name}.json").read_text())
    circuit = Circuit.from_json(data)
    bits = (circuit.inputs - len(data["constants"])) // 2  # per operand

    def wires(a: int, b: int) -> tuple[int, ...]:
        """The constant wires, then each operand's bits, least significant first."""
        return (*data["constants"], *((a >> i) & 1 for i in range(bits)), *((b >> i) & 1 for i in range(bits)))

    def pattern(x: float) -> int:
        """FP32's bit pattern, or BF16's (its top half)."""
        return struct.unpack("<I", struct.pack("<f", x))[0] >> (32 - bits)

    for v in data["vectors"]:
        assert sum(bit << i for i, bit in enumerate(evaluate(circuit, wires(v["a"], v["b"])))) == v["out"], v
    print(f"{data['source']}: {len(circuit.gates)} gates, agrees with its {len(data['vectors'])} reference vectors")
    rng = random.Random(0)
    requests = [wires(pattern(rng.uniform(-100, 100)), pattern(rng.uniform(-100, 100))) for _ in range(8)]
    return circuit, requests, frozenset(range(len(data["constants"])))


def show(event: Event) -> None:
    fields = ", ".join(f"{name}={_brief(value)}" for name, value in vars(event.body).items())
    print(f"{event.sender} -> {', '.join(event.to) or 'record'}: {type(event.body).__name__}({fields})")


def _brief(value: object) -> str:
    if is_dataclass(value):
        return type(value).__name__
    if isinstance(value, tuple) and all(isinstance(v, str) for v in value):
        return repr(value)
    if isinstance(value, (tuple, list, dict, bytes, frozenset)) or hasattr(value, "items"):
        return f"<{len(value)}>"
    if isinstance(value, str) and len(value) > 16:
        return repr(value[:8] + "…")
    return repr(value)
