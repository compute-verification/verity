"""Run the protocol over real circuits: verity's FP32 adder and BF16 comparison, exported as Boolean circuits.

    uv run python examples/floats.py [--circuit f32_add|bf16_gt_strict] [--epochs N] [--requests R] [--k K]
                                     [--max-width W] [--mode honest|lazy] [--lazy-epochs 2] [--corrupt C]

Each request is a pair of random floats; the toy engine serves it by evaluating the circuit, commits each execution's
boundary wires, and the protocol samples and replays units. `data/*.json` carries the circuit and reference vectors from
`verity.evaluation.bits.evaluate_bits`; the script first checks the trial's evaluator agrees with them.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from fractions import Fraction
from pathlib import Path

from verity.circuits import partition as P
from verity.circuits.circuit import Circuit, evaluate
from verity.protocol.accounting.account import Device, Inventory, SpatialUnit
from verity.protocol.configuration import Choice, Configuration
from verity.protocol.run import Refused, run
from verity.subprotocols.nci.policy import NciBound
from verity.subprotocols.sampled_proofs.law import Subset

from toy_serving.engine import Engine

p = argparse.ArgumentParser()
p.add_argument("--circuit", default="f32_add", choices=("f32_add", "bf16_gt_strict"))
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--requests", type=int, default=8, help="requests served per epoch")
p.add_argument("--k", type=int, default=64, help="units drawn per epoch")
p.add_argument("--max-width", type=int, default=16, help="boundary wires per unit")
p.add_argument("--mode", default="honest", choices=("honest", "lazy"))
p.add_argument("--lazy-epochs", default="", help="with --mode lazy: the epochs it skips work in (default: all)")
p.add_argument("--corrupt", type=int, default=0, help="units per epoch whose committed output the engine flips")
args = p.parse_args()

data = json.loads((Path(__file__).parent / "data" / f"{args.circuit}.json").read_text())
circuit = Circuit.from_json(data)
width = (circuit.inputs - len(data["constants"])) // 2  # bits per operand


def wires(a: int, b: int) -> tuple[int, ...]:
    """Constant wires, then each operand's bits, least significant first."""
    return (*data["constants"], *((a >> i) & 1 for i in range(width)), *((b >> i) & 1 for i in range(width)))


def word(bits: tuple[int, ...]) -> int:
    return sum(bit << i for i, bit in enumerate(bits))


for v in data["vectors"]:
    got = word(evaluate(circuit, wires(v["a"], v["b"])))
    assert got == v["out"], (v, got)
print(f"{data['source']}: {circuit.inputs} inputs, {len(circuit.gates)} gates, "
      f"{len(data['vectors'])} reference vectors agree")


def pattern(x: float) -> int:
    if width == 32:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<I", struct.pack("<f", x))[0] >> 16  # bf16: the top half of an fp32


def requests(epoch: int) -> list[tuple[int, ...]]:
    rng = random.Random(epoch)
    return [wires(pattern(rng.uniform(-100, 100)), pattern(rng.uniform(-100, 100))) for _ in range(args.requests)]


partition = P.greedy(circuit, args.max_width)
work = len(circuit.gates)  # one work unit per gate
gpu = Device("gpu0", "h100-sxm", work=work * args.requests, memory_bits=1 << 20)
inventory = Inventory((gpu,), (SpatialUnit("rack0", ("gpu0",), ("eth0",)),))
print(f"partition: {len(partition)} regions of at most {args.max_width} boundary wires; "
      f"{len(partition) * args.requests} units per epoch, {args.k} drawn")

config = Configuration(
    sampled_proofs=Choice("replay", delta=0.01, parameters={"k": args.k, "max_width": args.max_width}),
    pouw=Choice("pearl-c", delta=0.01),
    pous=Choice("band-chain", delta=0.01),
    network_warden=Choice("constant-rate", parameters={"bits_per_unit": 7}),
    nci=Choice("disclosed"),
    policy=NciBound(gates=work * args.requests, alpha=Fraction(1)),
    commitments="merkle-sha256",
    challenges="auditor-secret",
    delta=0.05,
)
lazy = frozenset(int(e) for e in args.lazy_epochs.split(",") if e) or frozenset(range(1, args.epochs + 1))
engine = Engine("merkle-sha256", "replay", Subset(args.k), circuit=circuit, partition=partition, requests=requests,
                device=gpu, work=work, novel_bits=2 * width, mode=args.mode, lazy_epochs=lazy, corrupt=args.corrupt)
try:
    for report, record in run(config, engine, inventory, range(1, args.epochs + 1)):
        integrity = record.evidence.integrity
        print(f"epoch {record.epoch}: accounting={report.accounting.value} compliance={report.compliance.value} "
              f"units={integrity.units} wrong_units<={integrity.wrong_units} (δ={integrity.delta}) "
              f"record={record.digest()[:12]}")
        for r in record.reasons:
            print("    -", r)
except Refused as e:
    print("refused:")
    for r in e.reasons:
        print("  -", r)
    raise SystemExit(2)
