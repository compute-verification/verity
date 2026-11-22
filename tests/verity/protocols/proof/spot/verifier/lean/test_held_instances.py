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
from test_difftest import PAIR16, TRIPLE16, Setup
from verity_difftest import canonical

from verity.circuits import partition_object as PO
from verity.circuits import population as PN
from verity.protocols.proof.spot.verifier.one_stage import partition as P

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


A16, B16, C16 = f"{PAIR16.id}/c0", f"{TRIPLE16.id}/c0", f"{PAIR16.id}/c1"


def _audit16(seed: str, units: dict | None = None, opened=(A16, B16), sampled=(A16, B16), extra=()) -> Audit:
    """A window of `PAIR16` and `TRIPLE16` members, whose template returns whole u16 words: C_S passes the door's
    lowering (`OneStageZk.admitted`), so a case ends past it."""
    units = units or {A16: [[0, 2]], B16: [[2, 5]]}
    ports = (f"{A16}/out", f"{B16}/out")
    setup = Setup(PN.mixed_population_program([(PAIR16, 2), (TRIPLE16, 3)]),
                  P.template_instances_query([PAIR16.id, TRIPLE16.id]), names=ports)
    return Audit(seed, ports=ports, opened=opened, closed=opened, sampled=sampled, setup=setup, units=units,
                 trees={PAIR16.id: 2, TRIPLE16.id: 3}, extra=extra)


def test_the_instances_reach_the_read_on_a_window_the_door_lowers(tmp_path):
    """The instances on a window whose C_S the door lowers: the untied control passes the lowering, and a member reading
    a tied parameter with a file naming other ports, sampled or not, is answered where the members check reads it."""
    lowered = _audit16("held16")
    unsampled = _audit16("held16-unsampled", opened=(B16,), sampled=(B16,))
    rooted = _audit16("held16-rooted", extra=("input/n0", "input/n1"))
    shared = {A16: [[0, 1]], C16: [[1, 2]], B16: [[2, 5]]}
    cases = {
        "control-untied": (lambda: lowered, None),
        "no-files": (lambda: lowered, {}),
        "empty-A": (lambda: lowered, {A16: {}, B16: OTHER}),
        "other-ports-sampled": (lambda: lowered, {A16: OTHER, B16: OTHER}),
        "other-ports-unsampled": (lambda: unsampled, {A16: OTHER, B16: OTHER}),
        "other-ports-rooted": (lambda: rooted, {A16: OTHER, B16: OTHER}),
        "shared-files": (lambda: _audit16("held16-shared", units=shared), {A16: OTHER, C16: OTHER, B16: OTHER}),
        "shared-files-rooted": (lambda: _audit16("held16-shared-rooted", units=shared, extra=("input/n0", "input/n1")),
                                {A16: OTHER, C16: OTHER, B16: OTHER}),
    }
    out = {}
    for k, (make, files) in cases.items():
        try:
            a = make()
        except (ValueError, KeyError, TypeError, AssertionError) as e:
            out[k] = {"fixture": f"{type(e).__name__}: {e}"}
            continue
        names = [n for n, _ in a.s.program.fn.signature()[0]]
        if files is None:
            params = a.params | {"proofs": {"members": {n: m | {"registered": OTHER}
                                                        for n, m in a.params["proofs"]["members"].items()}}}
        else:
            params = _tied(a, **files) | {"tied": names}
        out[k] = dict(zip(("lean_verdict", "lean", "python"), _both(a, tmp_path / k, params)))
    print("HELD16-INSTANCES " + json.dumps(out, indent=1))
    assert "fixture" not in out["control-untied"] and "admitted" not in out["control-untied"]["lean"], out
