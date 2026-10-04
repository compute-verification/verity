"""Sampled verification end to end on a real circuit: verity's FP32 adder or BF16 comparison, exported as a Boolean circuit
with reference vectors from `verity.evaluation.bits.evaluate_bits` (`data/`). The prover serves eight requests, each a
pair of random floats; `run` plays it against the verifier and prints the verifier's view.

    uv run python examples/verification.py [f32_add|bf16_gt_strict]
"""

import json
import random
import struct
import sys
from pathlib import Path

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, evaluate
from verity.primitives.commitments import MerkleSha256
from verity.protocol.run import run
from verity.subprotocols.verification.replay import Replay
from verity.subprotocols.verification.sampling.events import Draw, Proved, Registered
from verity.subprotocols.verification.sampling.law import Subset
from verity.subprotocols.verification.sampling.prover import prover
from verity.subprotocols.verification.sampling.verifier import verifier

data = json.loads((Path(__file__).parent / "data" / f"{sys.argv[1] if len(sys.argv) > 1 else 'f32_add'}.json").read_text())
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
partition = P.greedy(circuit, max_width=16)
proof_system = Replay(MerkleSha256())

parties = {
    "prover": prover(circuit, partition, requests, proof_system),
    "verifier": verifier({circuit.id(): circuit}, Subset(k=64), max_width=16, delta=0.01, proof_system=proof_system),
}
for event in run(parties, view="verifier"):
    match event.body:
        case Registered(registration):
            what = f"Registered({len(registration.executions)} executions, {registration.population} units)"
        case Draw(units):
            what = f"Draw({len(units)} units)"
        case Proved(proofs):
            what = f"Proved({len(proofs)} units)"
        case body:
            what = body
    print(f"{event.sender} -> {', '.join(event.to) or 'record'}: {what}")
