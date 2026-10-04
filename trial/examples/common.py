"""What the examples share: a real circuit, eight requests to serve through it, a scheduler that batches them into
steps and the builder of a step's circuit, and printing a view.

The circuits are verity's FP32 adder and BF16 comparison, exported as Boolean circuits with reference vectors from
`verity.evaluation.bits.evaluate_bits` (`data/`). A request is a pair of random floats.
"""

import hashlib
import json
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass, is_dataclass
from pathlib import Path

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, evaluate, transcript
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.events import Event
from verity.primitives.registration import Builder, Registration, Stream, built, commit


def load(name: str, n: int = 8, seed: int = 0) -> tuple[Circuit, list[tuple[int, ...]]]:
    """The circuit (its baseline is its constants) and `n` requests."""
    data = json.loads((Path(__file__).parent / "data" / f"{name}.json").read_text())
    circuit = Circuit.from_json(data)
    bits = (circuit.inputs - len(circuit.baseline)) // 2  # per operand

    def inputs(a: int, b: int) -> tuple[int, ...]:
        """Each operand's bits, least significant first."""
        return (*((a >> i) & 1 for i in range(bits)), *((b >> i) & 1 for i in range(bits)))

    def pattern(x: float) -> int:
        """FP32's bit pattern, or BF16's (its top half)."""
        return struct.unpack("<I", struct.pack("<f", x))[0] >> (32 - bits)

    for v in data["vectors"]:
        assert sum(bit << i for i, bit in enumerate(evaluate(circuit, inputs(v["a"], v["b"])))) == v["out"], v
    print(f"{data['source']}: {len(circuit.gates)} gates, agrees with its {len(data['vectors'])} reference vectors")
    rng = random.Random(seed)
    return circuit, [inputs(pattern(rng.uniform(-100, 100)), pattern(rng.uniform(-100, 100))) for _ in range(n)]


@dataclass(frozen=True)
class Batched:
    """The builder of a scheduler step: as many requests as the step's advice says, side by side through `circuit`,
    sharing its baseline, each partitioned as `partition` (so each request is its own isolation units). The advice is
    the batch size less one, since how the scheduler batches is a runtime choice no one fixes ahead of time."""

    circuit: Circuit
    partition: P.Partition
    advice_bits: int  # so at most 2**advice_bits requests a step

    def id(self) -> str:
        body = json.dumps(["examples/Batched/v0", self.circuit.id(), self.partition, self.advice_bits])
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def build(self, advice: str) -> tuple[Circuit, P.Partition]:
        c, n = self.circuit, int(advice or "0", 2) + 1
        b, own = len(c.baseline), c.inputs - len(c.baseline)  # baseline inputs, and each request's own
        total = b + n * own

        def wire(j: int, w: int) -> int:
            """Request j's copy of the circuit's wire w."""
            if w < b:
                return w
            if w < c.inputs:
                return b + j * own + (w - b)
            return total + j * len(c.gates) + (w - c.inputs)

        gates = tuple((op, wire(j, x), wire(j, y)) for j in range(n) for op, x, y in c.gates)
        outputs = tuple(wire(j, o) for j in range(n) for o in c.outputs)
        partition = tuple(
            tuple(tuple(j * len(c.gates) + g for g in region) for region in unit)
            for j in range(n)
            for unit in self.partition
        )
        return Circuit(total, gates, outputs, c.baseline), partition


def schedule(
    requests: Sequence[tuple[int, ...]], sizes: Sequence[int], builder: Batched
) -> list[tuple[str, tuple[int, ...]]]:
    """The scheduler's steps, `sizes[i]` requests in step i, in order: each step's advice and inputs."""
    steps, start = [], 0
    for n in sizes:
        steps.append((format(n - 1, f"0{builder.advice_bits}b"), sum(requests[start : start + n], ())))
        start += n
    return steps


def transcribe(builder: Builder, steps: Sequence[tuple[str, tuple[int, ...]]]) -> list[tuple[int, ...]]:
    """Each step run honestly, from its advice and fresh inputs: its transcript, its state read from the transcripts of
    the steps that wrote it."""
    transcripts: list[tuple[int, ...]] = []
    for advice, fresh in steps:
        circuit, i, state = built(builder, advice)[0], len(transcripts), []
        for back, k in circuit.state:
            writer = built(builder, steps[i - back][0])[0]
            state.append(transcripts[i - back][writer.outputs[k]])
        transcripts.append(transcript(circuit, (*state, *fresh)))
    return transcripts


def executed(
    builder: Builder, steps: Sequence[tuple[str, tuple[int, ...]]], scheme: CommitmentScheme
) -> tuple[Stream, list[tuple[int, ...]]]:
    """What the prover's runtime leaves: its stream, each step run and committed, and the transcripts it keeps."""
    transcripts = transcribe(builder, steps)
    executions = tuple(commit(builder, advice, values, scheme) for (advice, _), values in zip(steps, transcripts))
    return Stream(Registration(builder), executions), transcripts


def tries(stream: Stream) -> int:
    """The work the stream holds: one try per gate per execution."""
    return sum(len(stream.built(i)[0].gates) for i in range(len(stream.executions)))


def show(event: Event) -> None:
    fields = ", ".join(f"{name}={_brief(value)}" for name, value in vars(event.body).items())
    to = "record" if event.to == event.sender else event.to
    print(f"{event.sender} -> {to}: {type(event.body).__name__}({fields})")


def _brief(value: object) -> str:
    if is_dataclass(value):
        fields = vars(value)
        if fields and all(isinstance(v, (int, float)) for v in fields.values()):
            return f"{type(value).__name__}({', '.join(f'{k}={v}' for k, v in fields.items())})"
        return type(value).__name__
    if isinstance(value, tuple) and all(isinstance(v, str) for v in value):
        return repr(value)
    if isinstance(value, (tuple, list, dict, bytes, frozenset)) or hasattr(value, "items"):
        return f"<{len(value)}>"
    if isinstance(value, str) and len(value) > 16:
        return repr(value[:8] + "…")
    return repr(value)
