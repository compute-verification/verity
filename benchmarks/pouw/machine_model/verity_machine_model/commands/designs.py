"""machine-model designs [--json]            the design registry (data/designs): each design's name, its aliases, the
                                          noise on each product, fold, filler, proof block, coins and perplexity arm"""
from __future__ import annotations

import json
import textwrap

from verity_machine_model import eps as E
from verity_machine_model.commands._shared import registry


def add(sub) -> None:
    r = sub.add_parser("designs")
    r.add_argument("--json", action="store_true")


def run(args) -> int:
    ds = [E.design(n, args.data) for n in E.designs(args.data)]
    if args.json:
        print(json.dumps([registry(d) for d in ds], ensure_ascii=False, indent=1))
        return 0
    print(f"{'design':15s} {'title':22s} {'linears':>9s} {'lm_head':>9s} {'QKᵀ':>9s} {'P·V':>5s} {'fold':>9s} "
          f"{'filler':>6s} {'unit':>6s} {'coins':>5s} {'ppl':14s} aliases")
    for d in ds:
        n = d.noise
        print(f"{d.name:15s} {d.title:22s} {n['linears']:>9s} {n['lm_head']:>9s} {n['qk']:>9s} {n['pv']:>5s} "
              f"{d.fold_name + ' ' + f'{d.fold_cost():g}':>9s} {sum(f.per_word for f in d.filler):6g} {d.unit:>6s} "
              f"{d.coin_bits:5d} {d.ppl or '–':14s} {', '.join(d.aliases)}")
    print(textwrap.fill("noise: " + "; ".join(f"{k}: {v}" for k, v in E.NOISE.items()), width=120,
                        subsequent_indent="  "))
    return 0
