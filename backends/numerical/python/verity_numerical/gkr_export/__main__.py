"""``python -m verity_numerical.gkr_export OUT_DIR --units N [--seed S]``: circuit.txt, honest.bin (N random finite
units, each cross-checked against WP2 ``check_step`` and the circuit's own evaluator), negatives/<name>.bin and
negatives.txt (name, WP2 predicate, circuit evaluator verdict), and a manifest with the counts."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .circuit import (
    GOLDILOCKS,
    build_unit_circuit,
    export_circuit,
    random_finite_unit,
    unit_witness,
    verify_unit,
    write_witness,
)
from .negatives import negative_cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--units", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    ctx, paths = build_unit_circuit()
    export_circuit(ctx, out / "circuit.txt")
    rng = random.Random(args.seed)
    units = []
    for _ in range(args.units):
        c, a, b, y, h = random_finite_unit(rng)                # check_step accepts (raises otherwise)
        vals = unit_witness(ctx, paths, c, a, b, y, h)
        bad = verify_unit(ctx, vals)
        if bad is not None:
            print(f"circuit rejects an honest unit: {bad}", file=sys.stderr)
            return 1
        units.append(vals)
    write_witness(units, out / "honest.bin")
    neg_dir = out / "negatives"
    neg_dir.mkdir(exist_ok=True)
    lines = []
    n_neg = 0
    for name, c, a, b, y, h, predicate in negative_cases():
        vals = unit_witness(ctx, paths, c, a, b, y, h)
        verdict = verify_unit(ctx, vals)
        if verdict is None:
            print(f"circuit evaluator ACCEPTS negative {name} (WP2 rejects at {predicate})", file=sys.stderr)
            return 1
        write_witness([vals], neg_dir / f"{name}.bin")
        lines.append(f"{name}\t{predicate}\t{verdict}")
        n_neg += 1
    (out / "negatives.txt").write_text("\n".join(lines) + "\n")
    manifest = {
        "field": "goldilocks", "p": GOLDILOCKS, "columns": len(ctx.cols), "wires": len(ctx.wires),
        "products": sum(1 for w in ctx.wires if w[0] == "prod"), "assertions": len(ctx.asserts),
        "queries": len(ctx.queries), "max_depth": max(ctx.depth),
        "queries_by_table": {t: sum(1 for n, _ in ctx.queries if n == t) for t in {n for n, _ in ctx.queries}},
        "honest_units": len(units), "honest_cross_checked_against_check_step": len(units),
        "negatives": n_neg, "seed": args.seed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
