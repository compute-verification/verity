"""machine-model lean-headline [--check]   write sm_120's headline instances as exact Lean data, the files its headline
                                          guarantees read (security/Definitions/Accounting/Headline/Sm120*.lean);
                                          --check writes nothing and exits 1 when a committed file is not what it
                                          would write"""
from __future__ import annotations

import argparse

from verity_machine_model.lean_headline import write
from verity_machine_model.model import DATA


def add(sub: argparse._SubParsersAction) -> None:
    c = sub.add_parser("lean-headline")
    c.add_argument("--check", action="store_true", help="write nothing; exit 1 when a committed file differs")


def run(args: argparse.Namespace) -> int:
    if args.data != DATA:
        raise SystemExit("lean-headline writes the instances of this package's own data: no --data, and no --device "
                         "but sm_120")
    return write(args.check)
