"""Value producer for ``DemoCore_v1{N,R}``: the ``verity.ir.evaluate`` reference interpreter over the whole program (no other numerics).

Writes the brief's interface-2 value records: JSONL, one row per gate, sorted by gate index,
``{"g": <gate>, "k": "<kind>", "w": <width bits>, "v": "<hex, LE words>"}`` — one file per RU (``interior/<rid>/0/<call>.jsonl``) for the
interior gates, and ``boundary/<rid>.jsonl`` for the RU output gates; ``inputs.jsonl`` for the Input gates.  Also ``inventory.json``
(gates required per RU = every VU output gate of the RU not at the RU boundary).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from verity.ir.evaluate import evaluate_region

from .circuit import Partition, build

__all__ = ["Values", "Transcript", "produce", "random_inputs", "words_hex"]

Transcript = dict[int, int]


def words_hex(words: Sequence[int], width: int = 16) -> str:
    nbytes = width // 8
    return b"".join(int(w).to_bytes(nbytes, "little") for w in words).hex()


def random_inputs(N: int, seed: int) -> dict[str, list[int]]:
    """Deterministic pseudo-random BF16 input words from a declared seed (the declared input x; ``sha256(seed || name || i)``).
    Restricted to finite, moderate magnitudes so the demonstration exercises the ordinary (non-NaN/inf) paths of both primitives;
    the primitives are total either way and the checker's semantics are the registered ones."""
    out: dict[str, list[int]] = {}
    for name in ("a", "b", "c"):
        ws = []
        for i in range(N):
            h = hashlib.sha256(f"demo-core/{seed}/{name}/{i}".encode()).digest()
            # sign, exponent in [120, 133] (|x| in ~[2^-7, 2^6]), 7 mantissa bits
            sign = h[0] & 1
            exp = 120 + (h[1] % 14)
            mant = h[2] & 0x7F
            ws.append((sign << 15) | (exp << 7) | mant)
        out[name] = ws
    return out


@dataclass(frozen=True)
class Values:
    partition: Partition
    inputs: dict[str, list[int]]
    transcript: Transcript

    def value(self, g: int) -> int:
        return int(self.transcript[g])

    def words(self, gates: Sequence[int]) -> list[int]:
        return [int(self.transcript[g]) for g in gates]

    def ru_output_words(self, r: int) -> list[int]:
        return self.words(self.partition.rus[r].returns)

    def ru_interior_gates(self, r: int) -> list[int]:
        ru = self.partition.rus[r]
        return [v.gate for v in ru.vus if not v.ru_output]

    def ru_interior_words(self, r: int) -> list[int]:
        return self.words(self.ru_interior_gates(r))

    def output_words(self) -> list[int]:
        return self.words(self.partition.outputs)

    def to_json(self) -> dict[str, Any]:
        return {"schema": "veritor/demo-core/values/v1", "program_digest": self.partition.program_digest, "partition_digest": self.partition.digest,
                "inputs": self.inputs, "outputs": self.output_words(), "gates": self.partition.gates,
                "transcript": {str(g): int(v) for g, v in sorted(self.transcript.items())}}


def produce(part: Partition, inputs: dict[str, list[int]], program=None) -> Values:
    """Evaluate the program on ``inputs`` (a, b, c as BF16 words) with the reference interpreter; every gate gets exactly one value."""
    if program is None:
        program, digest = build(part.N, part.R)
        if digest != part.program_digest:
            raise RuntimeError(f"program digest {digest} != partition's {part.program_digest}")
    t: Transcript = {}
    for name in ("a", "b", "c"):
        gates = part.inputs[name]
        ws = list(inputs[name])
        if len(ws) != len(gates):
            raise ValueError(f"input {name}: {len(ws)} words for {len(gates)} Input gates")
        for g, w in zip(gates, ws):
            if not (0 <= int(w) < 1 << 16):
                raise ValueError(f"input {name}: word {w} is not a 16-bit encoding")
            t[g] = int(w)
    circuit = program.circuit
    evaluate_region(circuit, circuit.region(0, program.gates), t)
    if len(t) != program.gates:
        raise RuntimeError(f"transcript has {len(t)} of {program.gates} gates")
    return Values(part, {k: list(v) for k, v in inputs.items()}, t)


def stage_eval(part: Partition, r: int, h: Sequence[int], b: Sequence[int], c: Sequence[int], program=None) -> list[int]:
    """Re-evaluate ONE stage f_r on given input words with the reference interpreter (the prover's consistency determination, R17a):
    the stage's region is evaluated from a transcript holding only its inputs; returns the output tuple's words."""
    if program is None:
        program, _ = build(part.N, part.R)
    ru = part.rus[r]
    t: Transcript = {}
    for g, w in zip(ru.args[:part.N], h):
        t[int(g)] = int(w)
    for g, w in zip(part.inputs["b"], b):
        t[int(g)] = int(w)
    for g, w in zip(part.inputs["c"], c):
        t[int(g)] = int(w)
    circuit = program.circuit
    evaluate_region(circuit, circuit.region(ru.lo, ru.hi), t)
    return [int(t[g]) for g in ru.returns]


def write_value_records(vals: Values, out: Path) -> dict[str, Any]:
    """Interface-2 JSONL value records + ``inventory.json``; returns the inventory."""
    part = vals.partition
    out.mkdir(parents=True, exist_ok=True)
    kind_of = {v.gate: v.kind for v in part.vus}
    (out / "inputs.jsonl").write_text("".join(json.dumps({"g": g, "k": f"Input16|{name}", "w": 16, "v": words_hex([vals.value(g)])}) + "\n"
                                              for name in ("a", "b", "c") for g in part.inputs[name]))
    inventory: dict[str, Any] = {"schema": "veritor/demo-core/inventory/v1", "program_digest": part.program_digest, "partition_digest": part.digest, "rus": {}}
    for ru in part.rus:
        rid = f"core-ru{ru.ru_index}"
        d = out / "interior" / rid / "0"
        d.mkdir(parents=True, exist_ok=True)
        interior = [v.gate for v in ru.vus if not v.ru_output]
        (d / f"{ru.name}.jsonl").write_text("".join(json.dumps({"g": g, "k": kind_of[g], "w": 16, "v": words_hex([vals.value(g)])}) + "\n" for g in interior))
        (out / "boundary").mkdir(exist_ok=True)
        (out / "boundary" / f"{rid}.jsonl").write_text("".join(json.dumps({"g": g, "k": kind_of[g], "w": 16, "v": words_hex([vals.value(g)])}) + "\n" for g in ru.returns))
        inventory["rus"][rid] = {"ru_index": ru.ru_index, "call": ru.name, "gate_interval": [ru.lo, ru.hi], "required_interior_gates": interior,
                                 "boundary_output_gates": list(ru.returns), "n_required": len(interior)}
    (out / "inventory.json").write_text(json.dumps(inventory, indent=1) + "\n")
    return inventory
