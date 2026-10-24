"""The streamed-label wipe's verdict decided by its Lean port, `streamed-wipe-verdict` (`MemoryChallenger.StreamedWipe`,
built here with `lake` on first use), with `streamed_wipe.verdict` beside it as the reference.

`judge(signed, commitment, trusted=..., schemes=..., decide=VERDICT)` returns the verdict a caller records and each side's
as it gave it (`{"accepted", "egress_mib", "reasons"}`), as `decide` names:

- `lean`: the Lean's verdict; the Python's isn't computed.
- `both`: both computed, and the Lean's verdict where they agree on every field; where they differ, a rejection naming
  both (`DISAGREE`).
- `python`: the Python's alone (`streamed_wipe.verdict`), and no Lean process.

Each side rejects a session whose batch size the guarantee doesn't cover (`WipeParams.covered`), however it ran.
`VERDICT` is the default, the one line that switches it. `tests/test_wipe_verdict.py` holds the Lean to the Python on
drawn sessions, adversarial edits of them and named negative controls; `benchmarks/pous/streamed_wipe/bench.py`'s session
step judges its session here.

The record goes as `to_json` writes its parts, `trusted` as sorted hex and `schemes` by id, each one the port verifies
(`MemoryChallenger.KNOWN_SCHEMES`: `memory_challenger.LMS` and the test signers' ids).  Where the Lean refuses the record
(a scheme id it doesn't verify, a field `to_json` can't have written), `verdict` raises `ValueError` and `judge` rejects
the session with the egress saturated, which bounds nothing: it fails closed, never to the Python."""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import verity_difftest as D
from verity.core.protocols.pous import streamed_wipe as sw

PACKAGE = Path(__file__).resolve().parent
EXE = "streamed-wipe-verdict"
VERDICT = "lean"
"""Who decides the verdict (module docstring): `lean`, `both` or `python`."""
VERDICTS = ("lean", "both", "python")
DISAGREE = "the Lean and Python verdicts disagree"


def record(signed: sw.SignedWipe, commitment: sw.WipeCommitment, *, trusted: frozenset[bytes] | set[bytes],
           schemes: Mapping[str, Any] = sw.SCHEMES) -> dict[str, Any]:
    """The Lean verdict's input: `verdict`'s arguments as JSON."""
    return {"signed": signed.to_json(), "commitment": commitment.to_json(), "trusted": sorted(k.hex() for k in trusted),
            "schemes": sorted(schemes)}


@functools.cache
def server() -> D.Server:
    return D.Server([str(D.build(PACKAGE, EXE))])


def lean(signed: sw.SignedWipe, commitment: sw.WipeCommitment, *, trusted: frozenset[bytes] | set[bytes],
         schemes: Mapping[str, Any] = sw.SCHEMES) -> dict[str, Any]:
    """The Lean's verdict as it gives it; `ValueError` where it refuses the record."""
    kind, body = server().call("verdict", record(signed, commitment, trusted=trusted, schemes=schemes))
    if kind != "output":
        raise ValueError(f"the Lean verdict refuses the record: {body}")
    return json.loads(body)


def python(signed: sw.SignedWipe, commitment: sw.WipeCommitment, *, trusted: frozenset[bytes] | set[bytes],
           schemes: Mapping[str, Any] = sw.SCHEMES) -> dict[str, Any]:
    """The Python's verdict, in the Lean's form."""
    v = sw.verdict(signed, commitment, trusted=trusted, schemes=schemes)
    return {"accepted": v.accepted, "egress_mib": v.egress_mib, "reasons": list(v.reasons)}


def _of(side: dict[str, Any]) -> sw.Verdict:
    return sw.Verdict(side["accepted"], tuple(side["reasons"]), side["egress_mib"])


def verdict(signed: sw.SignedWipe, commitment: sw.WipeCommitment, *, trusted: frozenset[bytes] | set[bytes],
            schemes: Mapping[str, Any] = sw.SCHEMES) -> sw.Verdict:
    """`streamed_wipe.verdict`, by the Lean."""
    return _of(lean(signed, commitment, trusted=trusted, schemes=schemes))


def judge(signed: sw.SignedWipe, commitment: sw.WipeCommitment, *, trusted: frozenset[bytes] | set[bytes],
          schemes: Mapping[str, Any] = sw.SCHEMES, decide: str = VERDICT) -> tuple[sw.Verdict, dict[str, dict[str, Any]]]:
    """The verdict `decide` names (module docstring), and each side's as it gave it, a side not run absent."""
    if decide not in VERDICTS:
        raise ValueError(f"the verdict is one of {VERDICTS}")
    kw = {"trusted": trusted, "schemes": schemes}
    sides: dict[str, dict[str, Any]] = {}
    if decide != "python":
        try:
            sides["lean"] = lean(signed, commitment, **kw)
        except ValueError as e:
            sides["lean"] = {"accepted": False, "egress_mib": sw.EGRESS_MAX_UNITS, "reasons": [str(e)]}
    if decide != "lean":
        sides["python"] = python(signed, commitment, **kw)
    if decide == "python":
        return _of(sides["python"]), sides
    if decide == "both" and sides["lean"] != sides["python"]:
        why = f"{DISAGREE}: lean {json.dumps(sides['lean'])}, python {json.dumps(sides['python'])}"
        return sw.Verdict(False, (why,), sw.EGRESS_MAX_UNITS), sides
    return _of(sides["lean"]), sides
