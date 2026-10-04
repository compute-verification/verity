"""Run one epoch: `python -m toy_serving [--integrity replay|sp1] [--mode honest|lazy|rollout] [--no-space]`."""

from __future__ import annotations

import argparse
from fractions import Fraction

from verity.protocol.configuration import Choice, Configuration
from verity.protocol.epoch import Refused, run
from verity.subprotocols.nci.policy import NciBound
from verity.subprotocols.sampled_proofs.law import Subset

from toy_serving.engine import INVENTORY, Engine

p = argparse.ArgumentParser()
p.add_argument("--integrity", default="replay")
p.add_argument("--mode", default="honest", choices=("honest", "lazy", "rollout"))
p.add_argument("--no-space", action="store_true")
p.add_argument("--k", type=int, default=6)
p.add_argument("--verbose", action="store_true")
args = p.parse_args()

config = Configuration(
    sampled_proofs=Choice(args.integrity, delta=0.01, parameters={"k": args.k, "max_width": 4}),
    pouw=Choice("pearl-c", delta=0.01),
    pous=None if args.no_space else Choice("band-chain", delta=0.01),
    network_warden=Choice("constant-rate", parameters={"bits_per_unit": 7}),
    nci=Choice("disclosed"),
    policy=NciBound(gates=1000, alpha=Fraction(10)),
    commitments="merkle-sha256",
    challenges="auditor-secret",
    delta=0.05,
)
try:
    engine = Engine("merkle-sha256", args.integrity, Subset(args.k), mode=args.mode, space=not args.no_space)
    report, record = run(config, engine, INVENTORY, epoch=1)
except Refused as e:
    print("refused:")
    for r in e.reasons:
        print("  -", r)
    raise SystemExit(2)
print(f"accounting={report.accounting.value} compliance={report.compliance.value} record={record.digest()[:16]}")
for r in record.reasons:
    print("  -", r)
if args.verbose:
    for name, profile in record.evidence.present().items():
        print(f"  {name}: {profile}")
