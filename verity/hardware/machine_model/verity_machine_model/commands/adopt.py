"""machine-model adopt CENSUS EPOCH          merge a measured census (an opcode probe's census.csv) at EPOCH, the last
                                          epoch, into data/census/EPOCH.csv: each (kind, row) keeps its faster line"""
from __future__ import annotations

from pathlib import Path

from verity_machine_model.adopt import adopt
from verity_machine_model.model import model


def add(sub) -> None:
    d = sub.add_parser("adopt")
    d.add_argument("census", type=Path)
    d.add_argument("epoch")


def run(args) -> int:
    out = adopt(args.census, args.epoch, args.data)
    m = model(args.data)
    prev = m.epochs[-2]
    print(f"{'kind':10s} {'row':14s} {'u at ' + prev:>14s} {'measured':>10s}")
    for a in out:
        before = f"{a.before:.1f}" if a.before is not None else "–"
        verdict = f"adopted from {args.epoch}" if a.adopted else "kept (the measurement is slower)"
        print(f"{a.kind:10s} {a.row:14s} {before:>14s} {a.measured:10.1f}  {verdict}")
    n = sum(a.adopted for a in out)
    print(f"{n} of {len(out)} lines adopted at {args.epoch}, {len(out) - n} kept")
    return 0
