"""Members reading a tied parameter's run Input through the per-member check whole (Lean's `checkMembersAt`, the twin's
`members_run`): with no file, an empty file or a file naming other ports than the read, sampled or not, with and without
the record's `input/n<w>` trees; two members sharing the node; a partition with no template ranges."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_members_twin import A, B, Audit, M, _json, lms_verify, _never
from verity_difftest import canonical

from verity.circuits import partition_object as PO

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
    rooted = Audit("held-instances-rooted", extra=("input/n0", "input/n1"))
    c1 = f"{A[:-3]}/c1"
    shared = {A: [[0, 1]], c1: [[1, 2]], B: [[2, 5]]}

    def sharing(extra):
        return lambda: Audit("held-instances-shared", ports=(f"{A}/out", f"{B}/out"), units=shared,
                             extra=(f"{c1}/out", *extra))

    cases = {
        "no-files": (lambda: sampled, {}),
        "empty-A": (lambda: sampled, {A: {}, B: OTHER}),
        "empty-v2-A": (lambda: sampled, {A: {"format": "verity/registered-reads/v2", "ports": {}}, B: OTHER}),
        "other-ports-sampled": (lambda: sampled, {A: OTHER, B: OTHER}),
        "other-ports-unsampled": (lambda: unsampled, {A: OTHER, B: OTHER}),
        "other-ports-untied": (lambda: sampled, None),
        "other-ports-rooted": (lambda: rooted, {A: OTHER, B: OTHER}),
        "empty-A-rooted": (lambda: rooted, {A: {}, B: OTHER}),
        "shared-no-files": (sharing(()), {}),
        "shared-files": (sharing(()), {A: OTHER, c1: OTHER, B: OTHER}),
        "shared-files-rooted": (sharing(("input/n0", "input/n1")), {A: OTHER, c1: OTHER, B: OTHER}),
        "no-template-ranges": (lambda: sampled, "Q_word"),
    }
    out = {}
    for k, (make, files) in cases.items():
        try:
            a = make()
        except (ValueError, KeyError, TypeError, AssertionError) as e:
            out[k] = {"fixture": f"{type(e).__name__}: {e}"}
            continue
        if files is None:
            params = a.params | {"proofs": {"members": {n: m | {"registered": OTHER}
                                                        for n, m in a.params["proofs"]["members"].items()}}}
        elif files == "Q_word":
            params = _tied(a, **{A: OTHER, B: OTHER})
            params = params | {"query": PO.query(16, 32)}
        else:
            params = _tied(a, **files)
        out[k] = dict(zip(("lean_verdict", "lean", "python"), _both(a, tmp_path / k, params)))
    print("HELD-INSTANCES " + json.dumps(out, indent=1))
    for k, v in out.items():
        if k != "other-ports-untied" and "fixture" not in v:
            assert v["lean_verdict"] == "reject", (k, v)
