"""machine-model census [EPOCH] [--kind K]   each instruction class's u on each row at EPOCH (default: the last)"""
from __future__ import annotations

from verity_machine_model.model import census, model


def add(sub) -> None:
    c = sub.add_parser("census")
    c.add_argument("epoch", nargs="?")
    c.add_argument("--kind")


def run(args) -> int:
    m = model(args.data)
    epoch = args.epoch or m.epochs[-1]
    if epoch not in m.epochs:
        raise SystemExit(f"no epoch {epoch!r}: one of {', '.join(m.epochs)}")
    c = census(m, epoch, args.data)
    for k in sorted(c):
        if args.kind and k != args.kind:
            continue
        print(f"{k}: " + ", ".join(f"{r} {u / m.cycle_u:.4g}" for r, u in c[k].items()) + "  (SMSP-cycles)")
    return 0
