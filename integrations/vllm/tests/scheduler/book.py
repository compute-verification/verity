"""The registered book the compile slice's charge reads: the async-light stream's of the certifier's `arith_encode`
vectors (`arith_vectors.decisions`), a fixture of core's, where `arith_encode` is."""
from __future__ import annotations

import json
from pathlib import Path

ARITH = Path(__file__).resolve().parents[4] / "verity/core/protocols/certifier/tests/fixtures/arith_encode_vectors.json"


def book() -> dict:
    streams = json.loads(ARITH.read_text())["streams"]
    return next(s for s in streams if s["source"].startswith("async-light"))["decisions"]["book"]
