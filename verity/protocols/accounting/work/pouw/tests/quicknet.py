"""Recorded drand quicknet rounds (`vectors/quicknet.json`) for the tests that need a verified beacon."""

from __future__ import annotations

import json
from pathlib import Path

from verity.primitives.crypto import beacon as B

DATA = json.loads((Path(__file__).with_name("vectors") / "quicknet.json").read_text())
ROUNDS = {r["round"]: B.Round(r["round"], bytes.fromhex(r["signature"])) for r in DATA["rounds"]}
LATEST = ROUNDS[max(ROUNDS)]


def recorded_before(rnd: B.Round) -> int:
    """The latest start of a round slot at which a recording still lets `rnd` follow it: `rnd.round` −
    `B.MARGIN` is then the round due."""
    return B.QUICKNET.genesis + (rnd.round - B.MARGIN - 1) * B.QUICKNET.period
