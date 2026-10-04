"""Run the protocol over several epochs with the toy engine as the developer:

    python -m toy_serving [--epochs N] [--integrity replay|sp1] [--mode honest|lazy|rollout] [--lazy-epochs 2,3]
                          [--no-space] [--k K] [--verbose]
"""

from __future__ import annotations

import argparse
from fractions import Fraction

from verity.protocol.configuration import Choice, Configuration
from verity.protocol.run import Refused, run
from verity.subprotocols.nci.policy import NciBound
from verity.subprotocols.sampled_proofs.law import Subset

from toy_serving.engine import INVENTORY, Engine

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--integrity", default="replay")
p.add_argument("--mode", default="honest", choices=("honest", "lazy", "rollout"))
p.add_argument("--lazy-epochs", default="", help="with --mode lazy: the epochs it skips work in (default: all)")
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
lazy = frozenset(int(e) for e in args.lazy_epochs.split(",") if e) or frozenset(range(1, args.epochs + 1))
engine = Engine("merkle-sha256", args.integrity, Subset(args.k), mode=args.mode, space=not args.no_space, lazy_epochs=lazy)
try:
    for report, record in run(config, engine, INVENTORY, range(1, args.epochs + 1)):
        print(f"epoch {record.epoch}: accounting={report.accounting.value} compliance={report.compliance.value} "
              f"record={record.digest()[:12]} previous={(record.previous or '-')[:12]}")
        for r in record.reasons:
            print("    -", r)
        if args.verbose and record.evidence:
            for name, profile in record.evidence.present().items():
                print(f"    {name}: {profile}")
except Refused as e:
    print("refused:")
    for r in e.reasons:
        print("  -", r)
    raise SystemExit(2)
