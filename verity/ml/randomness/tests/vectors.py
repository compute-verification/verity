"""Writes ``vectors.json`` from `verity.ml.randomness`: the known answers ``test_randomness.py`` pins.

    python verity/ml/randomness/tests/vectors.py [--check]
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from verity.ml.randomness import derive, frame

OUT = Path(__file__).with_name("vectors.json")
FRAMES = (b"", b"ab", "é", 0, 255, 256)
CASES = (
    ("00", "verity/test/kat/v1", {}),
    ("626561636f6e2d726f756e642d3432", "verity/sampled-proofs/ru-sample/v1",
     {"program": {"bytes": "11" * 32}, "boundary_root": {"bytes": "22" * 32}, "epoch": {"int": 7}}),
    ("6465766963652d6b6579", "verity/pouw/noise-seed/v1", {"request": {"str": "r0"}, "device": {"str": "gpu0"}, "zero": {"int": 0}}),
)
STREAM, STREAM_BYTES, SHARD = ["s", 3], 80, 5
UNIFORM = ((1, 0), (7, 0), (7, 1), (1000, 2), (2**127 + 1, 3), (10**30, 4))
BERNOULLI = (("1/2", 0), ("1/3", 1), ("2/3", 2), ("0", 3), ("1", 4))
SUBSET = ((0, 5, 0), (3, 10, 1), (10, 10, 2), (4, 10**12, 3))


def _context(doc: dict) -> dict:
    return {n: bytes.fromhex(v["bytes"]) if "bytes" in v else v.get("str", v.get("int")) for n, v in doc.items()}


def build() -> dict:
    cases = []
    for source, domain, context in CASES:
        key = derive(bytes.fromhex(source), domain, _context(context))
        cases.append({
            "source": source, "domain": domain, "context": context, "key": key.value.hex(),
            "stream": {"index": STREAM, "bytes": key.stream(tuple(STREAM), STREAM_BYTES).hex()},
            "shard": {"index": SHARD, "key": key.shard(SHARD).value.hex()},
            "uniform": [[n, i, key.uniform(n, i)] for n, i in UNIFORM],
            "bernoulli": [[p, i, key.bernoulli(Fraction(p), i)] for p, i in BERNOULLI],
            "subset": [[k, n, i, list(key.subset(k, n, i))] for k, n, i in SUBSET],
        })
    return {"frame": [[repr(x), frame(x).hex()] for x in FRAMES], "cases": cases}


def main() -> int:
    text = json.dumps(build(), indent=1)
    if "--check" in sys.argv[1:]:
        return 0 if OUT.read_text() == text else 1
    OUT.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
