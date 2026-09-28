"""Compose the circuit prover's circuit for a template, the verifier's own copy: the lowering comes from an input set of the template, and
the statement binds the given partition and program digests. Run in the circuit prover's tree (its `verity_flock`).

    python compose_circuit.py --input-set DIR --out FILE [--partition H --program H]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity_flock import circuit as FC
from verity_numerical.bench.input_sets import InputSet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--program", default=None)
    a = ap.parse_args()
    s = InputSet.open(a.input_set)
    low = FC.lowering_for_set(s)
    bind = {k: v for k, v in (("partition", a.partition), ("program", a.program)) if v}
    text = FC.compose(low, s.subcircuit, **bind)
    Path(a.out).write_text(text)
    if low.units.cut:
        FC.IL.write_tables(Path(a.out).parent / "tables")
    out = {"subcircuit": s.subcircuit.id, "pin": FC.digest(text)}
    if hasattr(FC, "class_digest"):
        out["class_sha512"] = FC.class_digest(text)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
