"""machine-model list                        the named vectors: epoch, rows, credits and published τ"""
from __future__ import annotations

from verity_machine_model.model import vector, vectors


def add(sub) -> None:
    sub.add_parser("list")


def run(args) -> int:
    print(f"{'vector':16s} {'epoch':16s} {'rows':>4s} {'credits':>7s} {'τ':>7s} {'margin':>6s}  summary")
    for n in vectors(args.data):
        v = vector(n, args.data)
        tag = f" (superseded by {v.superseded_by})" if v.superseded_by else ""
        print(f"{n:16s} {v.epoch:16s} {len(v.rows):4d} {len(v.credits):7d} {v.published.get('tau', float('nan')):7.4f} "
              f"{v.published.get('margin', float('nan')):6.3f}  {v.summary}{tag}")
    return 0
