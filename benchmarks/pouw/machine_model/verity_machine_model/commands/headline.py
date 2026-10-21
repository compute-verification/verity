"""machine-model headline [--reading R] [--json]
                                          the per-family headline (headline.py, data/headline.toml): each family's
                                          design at each reading (`sheet`, the DS vectors at q* = ½; `measured`,
                                          decision 89's q* = 1.22, the 1.22-slot issue floor (Theorem C, unproved
                                          in Lean), on the measured vectors), operands free, credit ♭: ε_skip,
                                          ε_mand, ε_machine, ε_luck, their sum (the core ε), γ, q*₀, the
                                          cheater's binding rows and each row's conditions (the floor it needs,
                                          its fillers' credit, its p = 1 reading); ε is per node (ASK-65 = A):
                                          the node's GPUs, host CPU and peer GPUs together did at least 1 − ε of
                                          the credited work, and a per-GPU reading is attribution's; --json for
                                          the record"""
from __future__ import annotations

import json

from verity_machine_model import headline as H


def add(sub) -> None:
    hl = sub.add_parser("headline")
    hl.add_argument("--reading", choices=("sheet", "measured"),
                    help="one reading (default both: sheet, the DS vectors at q* = ½; measured, q* = 1.22 on the "
                         "measured vectors)")
    hl.add_argument("--json", action="store_true")


def run(args) -> int:
    readings = (args.reading,) if args.reading else H.READINGS
    if args.json:
        print(json.dumps(H.record(readings), ensure_ascii=False, indent=1))
    else:
        print(H.text(readings), end="")
    return 0
