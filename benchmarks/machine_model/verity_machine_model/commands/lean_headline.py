"""machine-model lean-headline [--check | --numbers] [--json F]
                                          write every family's headline instances as exact Lean data, the files the
                                          headline guarantees read (security/Definitions/Accounting/Headline/ and
                                          HeadlineDc/, Sm*.lean); --check writes nothing and exits 1 when a committed
                                          file is not what it would write; --numbers prints the data-centre
                                          families' readings and writes no Lean file; --json F also writes those
                                          readings to F"""
from __future__ import annotations

import argparse

from verity_machine_model import lean_headline, lean_headline_dc
from verity_machine_model.model import DATA


def add(sub: argparse._SubParsersAction) -> None:
    c = sub.add_parser("lean-headline")
    mode = c.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="write nothing; exit 1 when a committed file differs")
    mode.add_argument("--numbers", action="store_true",
                      help="print the data-centre families' readings only; write no Lean file")
    c.add_argument("--json", help="also write the data-centre families' readings here as JSON")


def run(args: argparse.Namespace) -> int:
    if args.data != DATA:
        raise SystemExit("lean-headline writes every family's instances from this package's own data: no --data, and "
                         "no --device but sm_120")
    sm120 = 0 if args.numbers else lean_headline.write(args.check)
    return max(sm120, lean_headline_dc.write(args.check, args.numbers, args.json))
