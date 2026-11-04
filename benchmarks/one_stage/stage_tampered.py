"""Stage an input set for the circuit prover with bit 0 of every instance's first output word flipped: the registered
roots then commit wrong outputs, so the proof of any sampled proof block must be refused. Run in the circuit prover's tree (its
`verity_flock`).

    python stage_tampered.py --input-set DIR --n N --out DIR [--partition H --program H --unit-indices FILE]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from verity_flock import circuit as FC
from verity_numerical.bench.input_sets import InputSet


class Flipped:
    def __init__(self, inner, port: str) -> None:
        self._inner, self._port = inner, port

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def port(self, name, lo, hi):
        words = np.asarray(self._inner.port(name, lo, hi)).copy().reshape(hi - lo, -1)
        if name == self._port:
            words[:, 0] ^= 1
        return words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--program", default=None)
    ap.add_argument("--unit-indices", default=None)
    a = ap.parse_args()
    s = InputSet.open(a.input_set)
    out_port = s.subcircuit.outputs[0].name
    flipped = Flipped(s, out_port)
    bind = {}
    if a.partition:
        bind = {"partition": a.partition, "program": a.program,
                "indices": json.loads(open(a.unit_indices).read()) if a.unit_indices else None}
    FC.stage(FC.lowering_for_set(s), flipped, 0, a.n, a.out, **bind)
    print(json.dumps({"tampered": out_port, "word": 0, "bit": 0, "n": a.n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
