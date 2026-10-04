"""Run one epoch: `python -m toy_serving [--integrity replay|sp1] [--mode honest|lazy|rollout] [--no-space]`."""

from __future__ import annotations

import argparse
from fractions import Fraction

from verity.protocol.compliance.policy import NciBound
from verity.protocol.configuration import Choice, Configuration
from verity.protocol.epoch import Refused, run

from toy_serving import INVENTORY, Engine

p = argparse.ArgumentParser()
p.add_argument("--integrity", default="replay")
p.add_argument("--mode", default="honest", choices=("honest", "lazy", "rollout"))
p.add_argument("--no-space", action="store_true")
args = p.parse_args()

accounting = {
    "integrity": Choice("sampled_proofs", args.integrity, delta=0.01, parameters={"k": 4}),
    "work": Choice("pouw", "pearl-c", delta=0.01),
    "channel": Choice("network_warden", "constant-rate", parameters={"bits_per_unit": 7}),
}
if not args.no_space:
    accounting["space"] = Choice("pous", "band-chain", delta=0.01)
config = Configuration(
    variant="v0", accounting=accounting, compliance=Choice("nci", "disclosed"),
    commitments="sha256-leaves", challenges="auditor-secret",
    policy=(NciBound(gates=1000, alpha=Fraction(10)),), delta=0.05,
)
try:
    report, record = run(config, Engine(mode=args.mode, space=not args.no_space), INVENTORY, epoch=1)
except Refused as e:
    print("refused:")
    for r in e.reasons:
        print("  -", r)
    raise SystemExit(2)
print(f"accounting={report.accounting.value} compliance={report.compliance.value} record={record.digest()[:16]}")
for r in record.reasons:
    print("  -", r)
