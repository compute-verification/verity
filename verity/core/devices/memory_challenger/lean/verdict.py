"""`memory_challenger.verdict` computed by its Lean port, behind the Python's signature: `verdict(signed, commitment, openings,
expected, trusted=..., schemes=...)` sends the record to `memory-challenger-verdict` (`MemoryChallenger.Judge`, built here
with `lake` on first use) and returns its `Verdict`, the reasons the Python's word for word.  `memory_challenger.verdict`
stays the reference: `tests/test_verdict.py` holds this to it, and `benchmarks/pous/memory_challenger/live.py`'s `judge`
runs both on every session, records both, and rejects a session they disagree on.

The record goes as `to_json` writes its parts.  `expected` is read only at the labels the rounds name, so C's whole table
never crosses; `schemes` goes by id, each one the port verifies (`MemoryChallenger.KNOWN_SCHEMES`: `memory_challenger.LMS`
and the test signers' ids).  Where the Lean refuses the record (a scheme id it doesn't verify, a field `to_json` can't
have written, such as a negative integer), this raises `ValueError`: it fails closed, never to the Python."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import verity_difftest as D
from verity.core.protocols.pous import memory_challenger as mc

PACKAGE = Path(__file__).resolve().parent
EXE = "memory-challenger-verdict"


def record(signed: mc.SignedSession, commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening],
           expected: Callable[[int], bytes], *, trusted: frozenset[bytes] | set[bytes],
           schemes: Mapping[str, Any] = mc.SCHEMES) -> dict[str, Any]:
    """The Lean verdict's input: `verdict`'s arguments as JSON, `expected` at the in-range labels the rounds name."""
    digests = {}
    for r in signed.rounds:
        if 0 <= r.index < commitment.blocks and str(r.index) not in digests:
            d = expected(r.index)
            if isinstance(d, bytes | bytearray):
                digests[str(r.index)] = bytes(d).hex()
    return {"signed": signed.to_json(), "commitment": commitment.to_json(), "openings": [o.to_json() for o in openings],
            "expected": digests, "trusted": sorted(k.hex() for k in trusted), "schemes": sorted(schemes)}


@functools.cache
def server() -> D.Server:
    return D.Server([str(D.build(PACKAGE, EXE))])


def verdict(signed: mc.SignedSession, commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening],
            expected: Callable[[int], bytes], *, trusted: frozenset[bytes] | set[bytes],
            schemes: Mapping[str, Any] = mc.SCHEMES) -> mc.Verdict:
    """`memory_challenger.verdict`, by the Lean."""
    kind, body = server().call("verdict", record(signed, commitment, openings, expected, trusted=trusted, schemes=schemes))
    if kind != "output":
        raise ValueError(f"the Lean verdict refuses the record: {body}")
    out = json.loads(body)
    return mc.Verdict(out["accepted"], tuple(out["reasons"]), out["late"], out["wrong"])
