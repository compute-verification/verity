"""Integrity end to end on a real circuit: verity's FP32 adder or BF16 comparison, exported as a Boolean circuit with
reference vectors from `verity.evaluation.bits.evaluate_bits` (`data/`). The prover serves eight requests, each a pair
of random floats; `run` plays the actors against each other and prints the verifier's view.

    uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]

Replay samples whole executions and has a sealed replayer run them again; proofs sample regions and have the prover
prove them, with the transparent proof system or SP1.
"""

import json
import random
import struct
import sys
from pathlib import Path

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, evaluate
from verity.primitives.commitments import MerkleSha256
from verity.protocols.integrity.proofs.construction import Proofs
from verity.protocols.integrity.proofs.transparent import Transparent
from verity.protocols.integrity.replay import Replay
from verity.run import run

circuit_name, construction = (sys.argv[1:] + ["f32_add", "replay"][len(sys.argv) - 1:])[:2]
data = json.loads((Path(__file__).parent / "data" / f"{circuit_name}.json").read_text())
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
scheme = MerkleSha256()

if construction == "replay":
    replay = Replay(k=2, delta=0.01)
    actors = {**replay.verifier(scheme), **replay.replayer(scheme), **replay.prover(scheme, circuit, requests)}
else:
    if construction == "sp1":
        from verity.protocols.integrity.proofs.sp1.sp1 import Sp1 as system
    else:
        system = Transparent
    proofs = Proofs(k=64, max_width=16, delta=0.01, system=system())
    actors = {**proofs.verifier(scheme), **proofs.prover(scheme, circuit, P.greedy(circuit, 16), requests)}

for event in run(actors, view="verifier"):
    body = event.body
    sizes = {name: len(value) for name, value in vars(body).items() if isinstance(value, (tuple, dict)) and name != "reasons"}
    what = body if type(body).__name__ == "IntegrityProfile" else f"{type(body).__name__}({', '.join(f'{n}: {s}' for n, s in sizes.items())})"
    print(f"{event.sender} -> {', '.join(event.to) or 'record'}: {what}")
