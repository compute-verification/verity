"""Candidate 130's instances through the per-member check whole (Lean's `checkMembersAt`, the twin's `members_run`):
members reading a tied parameter's run Input (`OneStageZk.heldReadsFiled`) with an empty registered file, or a file
naming other ports than the read, the member sampled or not."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_members_twin import A, B, Audit, M, _json, lms_verify, _never
from verity_difftest import canonical

pytestmark = pytest.mark.slow

OTHER = {"format": "verity/registered-reads/v2",
         "ports": {"p": {"value": "v", "leaves": 4, "domain": "dd" * 64, "root": "aa" * 64}}}


def _tied(a: Audit, **files) -> dict:
    params = a.params | {"parameters": [[n, t.to_json()] for n, t in a.s.program.fn.signature()[0]],
                         "tied": ["0.x", "1.x"]}
    ms = params["proofs"]["members"]
    return params | {"proofs": {"members": {k: m | ({"registered": files[k]} if k in files else {})
                                            for k, m in ms.items()}}}


def _both(a: Audit, at: Path, params: dict) -> tuple[str, str, str]:
    at.mkdir(parents=True, exist_ok=True)
    paths = []
    for d, b in a.files.items():
        (at / d[:16]).write_bytes(b)
        paths.append(at / d[:16])
    p = _json(params)
    verdict, why = M.check(p, canonical(a.rec).encode(), canonical(a.evidence).encode(), paths)
    try:
        run, refused = M.members_run(p, _json(a.rec), _json(a.evidence), M.blobs_of(paths), program=a.s.program,
                                     verify=lms_verify, verdict=_never)
        py = M.run_reason(run, refused)
    except M.Outside as e:
        py = f"Outside: {e}"
    return verdict, why, py


def test_the_instances_through_the_per_member_check(tmp_path):
    sampled = Audit("held-instances")
    unsampled = Audit("held-instances-unsampled", opened=(B,), sampled=(B,))
    cases = {
        "no-files": (sampled, {}),
        "empty-A": (sampled, {A: {}, B: OTHER}),
        "empty-v2-A": (sampled, {A: {"format": "verity/registered-reads/v2", "ports": {}}, B: OTHER}),
        "other-ports-sampled": (sampled, {A: OTHER, B: OTHER}),
        "other-ports-unsampled": (unsampled, {A: OTHER, B: OTHER}),
        "other-ports-untied": (sampled, None),
    }
    out = {}
    for k, (a, files) in cases.items():
        params = a.params | {"proofs": {"members": {n: m | {"registered": OTHER}
                                                    for n, m in a.params["proofs"]["members"].items()}}} \
            if files is None else _tied(a, **files)
        out[k] = dict(zip(("lean_verdict", "lean", "python"), _both(a, tmp_path / k, params)))
    print("HELD-INSTANCES " + json.dumps(out, indent=1))
    for k, v in out.items():
        assert v["lean_verdict"] == "reject", (k, v)
