"""``python -m verity_numerical.gkr_export OUT_DIR --units N [--seed S] [--field babybear]``: circuit.txt, honest.bin
(N random finite units, each cross-checked against WP2 ``check_step`` and the circuit's own evaluator),
negatives/<name>.bin and negatives.txt (name, WP2 predicate, circuit evaluator verdict), and a manifest with the
counts.  ``--field babybear`` exports the embedded checker of :mod:`.babybear` (also cross-checked against the
embedded integer checker) for the ``--features babybear`` build."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from verity_numerical.checker import REAL

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
    ap.add_argument("--field", default="goldilocks", choices=("goldilocks", "babybear"))
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.field == "babybear":
        from . import babybear as bb

        modulus = bb.BABYBEAR
        ctx, paths = bb.build_unit_circuit_bb()
        witness, verify = bb.unit_witness_bb, bb.verify_unit_bb

        def make_unit(rng):
            c, a, b, y, h = random_finite_unit(rng)
            bb.check_embedded(REAL, c, a, b, y, h)                # the embedded integer checker accepts too
            return c, a, b, y, h
    else:
        modulus = GOLDILOCKS
        ctx, paths = build_unit_circuit()
        witness, verify, make_unit = unit_witness, verify_unit, random_finite_unit
    export_circuit(ctx, out / "circuit.txt", modulus=modulus)
    rng = random.Random(args.seed)
    units = []
    for _ in range(args.units):
        c, a, b, y, h = make_unit(rng)                          # check_step accepts (raises otherwise)
        vals = witness(ctx, paths, c, a, b, y, h)
        bad = verify(ctx, vals)
        if bad is not None:
            print(f"circuit rejects an honest unit: {bad}", file=sys.stderr)
            return 1
        units.append(vals)
    write_witness(units, out / "honest.bin", modulus=modulus)
    neg_dir = out / "negatives"
    neg_dir.mkdir(exist_ok=True)
    lines = []
    n_neg = 0
    for name, c, a, b, y, h, predicate in negative_cases():
        vals = witness(ctx, paths, c, a, b, y, h)
        verdict = verify(ctx, vals)
        if verdict is None:
            print(f"circuit evaluator ACCEPTS negative {name} (WP2 rejects at {predicate})", file=sys.stderr)
            return 1
        write_witness([vals], neg_dir / f"{name}.bin", modulus=modulus)
        lines.append(f"{name}\t{predicate}\t{verdict}")
        n_neg += 1
    (out / "negatives.txt").write_text("\n".join(lines) + "\n")
    manifest = {
        "field": args.field, "p": modulus, "columns": len(ctx.cols), "wires": len(ctx.wires),
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
