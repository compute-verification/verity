"""One-stage v2's per-member form outside Lean: the refusal's run on a case.

A window of several members is audited in v2's per-member form of ``proofs`` (the parameters' ``members``, the
evidence's ``statements``; `proofs.members_v2` writes it). `Spot.OneStageZk` parses the form and accepts nothing of it
(`checkMembersAt_refuses`). The executable's ``refusal PARAMS COMMITMENT EVIDENCE [BLOB ...]``
(`Spot/MembersRefusal.lean`) runs the audit's check on such a case and prints why it is refused: the reason of the
first check that fails, and `MULTI_REFUSED` under ``proofs:`` exactly where every check before the refusal passed
(`ALL_PASSED`). The line is a refusal's reason and no verdict (`REFUSAL`): the command never exits 0, and nothing here
reads its answer as an audit's.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .proofs import V2_EXE, LeanError, v2_executable

#: what the v2 executable's ``refusal`` says of what it prints (`OneStageZk.REFUSAL`)
REFUSAL = "the per-member form's refusal, no verdict"
#: `OneStageZk.MULTI_REFUSED`: the refusal of the per-member form's acceptance
MULTI_REFUSED = ("the per-member form (proofs.members) is refused until the guarantee for a sample over several members' "
                 "statements under one draw is in the lock: verify one statement (proofs {circuit, public, registered})")
#: `refusal`'s reason exactly where the sessions' verdicts and v1's whole check at the coins passed
ALL_PASSED = f"proofs: {MULTI_REFUSED}"


def refusal(params: Mapping[str, Any], commitment: bytes, evidence: bytes, blobs: Sequence[Path] = ()) -> tuple[str, bool]:
    """Why v2 refuses the case of parameters `params`, registration record `commitment` and evidence `evidence` (each
    the bytes the audit reads) on the files `blobs` (`V2_EXE`'s ``refusal``, `OneStageZk.membersOutcome`): the reason,
    and whether the check decided (False where it is undecided). The reason is `ALL_PASSED` exactly where every check
    before the per-member form's refusal passed. A case in the one statement's form is refused unchecked. Raises
    `LeanError` on an answer that is not a refusal's line."""
    with tempfile.TemporaryDirectory(prefix="verity-members-refusal-") as d:
        paths = [Path(d, n) for n in ("params.json", "commitment.bin", "evidence.bin")]
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        for path, b in zip(paths, (canonical, commitment, evidence)):
            path.write_bytes(b)
        p = subprocess.run([str(v2_executable()), "refusal", *map(str, paths), *map(str, blobs)], capture_output=True,
                           text=True)
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise LeanError(f"{V2_EXE} refusal answered no line (exit {p.returncode}): {p.stderr[-400:]}") from None
    if (not isinstance(got, dict) or sorted(got) != ["decided", "refusal", "run"] or got["run"] != REFUSAL
            or type(got["refusal"]) is not str or type(got["decided"]) is not bool
            or p.returncode != (1 if got["decided"] else 2)):
        raise LeanError(f"{V2_EXE} refusal answered a line that is not a refusal's (exit {p.returncode}): {got!r}"[:600])
    return got["refusal"], got["decided"]


def case_refusal(case: Mapping[str, Any]) -> tuple[str, bool]:
    """`refusal` on a case as a driver writes one: ``{"blobs": [path, ...], "commitment": hex, "evidence": hex,
    "params": {...}}``."""
    return refusal(case["params"], bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]),
                   [Path(b) for b in case["blobs"]])


__all__ = ["ALL_PASSED", "MULTI_REFUSED", "REFUSAL", "case_refusal", "refusal"]
