"""Compose the circuit prover's circuit for a template, the verifier's own copy: the lowering comes from an input set of the template, and
the statement binds the given partition and program digests. Run in the circuit prover's tree (its `verity_flock`).

    python compose_circuit.py --input-set DIR --out FILE [--partition H --program H]
    python compose_circuit.py --template ID --out FILE --partition H --program H

``--template``: a serving-rows template whose units are bit rows (`verity_vllm.pipeline.serving_rows.CHUNKED`: a
``ScaledMmS8Coordinate``, a ``COORDS`` coordinate, a ``COLS`` column block, a ``SLICES`` slice, a ``KCHUNKS`` K-chunk), composed by C-Flock's `UnitStager` on the rows the window commits
(`verity_vllm.pipeline.chunk_rows.unit_rows`), bound to the partition and program; no input set is read.
A ``SPLITS`` part or combine (`verity_vllm.pipeline.split_rows`) reads its MUFU tables in lookup slots (`chunk_rows.reads`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity_flock import circuit as FC


def template(a: argparse.Namespace) -> int:
    from verity_flock.class_statement import UnitStager
    from verity_vllm.pipeline import chunk_rows as CR, serving_rows as SRW

    if not (a.partition and a.program):
        raise SystemExit("--template binds the statement: give --partition and --program")
    tp = SRW.template(a.template)
    in_rows, out_rows = CR.unit_rows(tp)
    st = UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=Path(a.out),
                    bind={"partition": a.partition, "program": a.program}, reads=CR.reads(tp))
    if st.meta.get("lookups"):
        FC.IL.write_tables(Path(a.out).parent / "tables")
    print(json.dumps({"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"]}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--program", default=None)
    a = ap.parse_args()
    if bool(a.input_set) == bool(a.template):
        ap.error("give either --input-set or --template")
    if a.template:
        return template(a)
    from verity_numerical.bench.input_sets import InputSet

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
