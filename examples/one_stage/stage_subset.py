"""Stage a selection of an input set's instances for the circuit prover (in the order given), as its own population: the frame domains bind
the selection, because the set's name and content digest become `<name>@<tag>` and the SHA-256 of the original digest and the
selected indices. Run in the circuit prover's tree (its `verity_flock`).

    python stage_subset.py --input-set DIR --indices FILE --tag TAG --out DIR [--n N ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from verity_flock import circuit as FC
from verity_numerical.bench.input_sets import InputSet


class Selection:
    def __init__(self, inner, indices: list[int], tag: str) -> None:
        self._inner, self._idx = inner, np.asarray(indices, dtype=np.int64)
        self.name = f"{inner.name}@{tag}"
        self.content_digest = hashlib.sha256(json.dumps({"content_digest": inner.content_digest, "indices": list(map(int, indices))},
                                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.n = len(indices)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def port(self, name, lo, hi):
        words = np.asarray(self._inner.port(name, 0, self._inner.n))
        return words.reshape(self._inner.n, -1)[self._idx[lo:hi]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs="*", default=[])
    a = ap.parse_args()
    s = InputSet.open(a.input_set)
    sel = Selection(s, json.loads(open(a.indices).read()), a.tag)
    low = FC.lowering_for_set(s)
    for n in (a.n or [sel.n]):
        FC.stage(low, sel, 0, n, a.out)
    text = open(f"{a.out}/circuit.txt").read()
    print(json.dumps({"subcircuit": s.subcircuit.id, "unit": low.name, "pin": FC.digest(text), "name": sel.name,
                      "content_digest": sel.content_digest, "n": a.n or [sel.n]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
