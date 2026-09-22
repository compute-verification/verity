"""red-team-zk fixtures (``fixtures/redteam-zk/``): the recorded B-Ligero ``q`` leak.

``q_leak_*.json`` holds the public chain message ``q`` of a real ``--zk`` proof (HM96 coins on), the verifier's chain
coefficients ``rc`` recomputed from the transcript, the chain constants, the layout, and the recorded accumulators
of the proved VUs.  The test replays the torch-free recovery (``verity_numerical.redteam.ligero_q_leak``): fold
``q`` to ``H``, walk every VU from ``c_0 = 0``, and require that EVERY intermediate accumulator comes out exactly.
A prover that masks ``q`` on ``H`` makes this recovery fail (the 6 x 3 systems become inconsistent) -- when the fix
lands, regenerate the fixture with the fixed prover and flip ``expected_recovery`` in the manifest to ``"none"``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from verity_numerical.redteam import ligero_q_leak as core

ROOT = Path(__file__).resolve().parents[4] / "fixtures" / "redteam-zk"
MANIFEST = ROOT / "manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="fixtures/redteam-zk not present")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _fixtures() -> list[tuple[dict, dict]]:
    m = _manifest()
    return [(e, json.loads((ROOT / e["file"]).read_text())) for e in m["fixtures"]]


def test_manifest_files_exist_and_are_well_formed() -> None:
    m = _manifest()
    assert m["lane"] == "red-team-zk"
    for e, doc in _fixtures():
        assert doc["schema"] == "redteam-zk/q-leak/v1"
        cfg, lay = doc["config"], doc["layout"]
        D, l = cfg["D"], cfg["l"]
        assert len(doc["transcript"]["q"]) == D and all(len(row) == l + cfg["k"] - 1 for row in doc["transcript"]["q"])
        assert len(doc["verifier_rc"]) == D and all(len(fam) == l for d in doc["verifier_rc"] for fam in d) and all(len(d) == 7 for d in doc["verifier_rc"])
        assert len(doc["recorded_accumulators"]) == lay["n_vus"] and all(len(a) == lay["steps"] for a in doc["recorded_accumulators"])
        assert e["accumulators"] == (lay["steps"] - 1) * lay["n_vus"]


def test_q_on_H_recovers_every_accumulator() -> None:
    for e, doc in _fixtures():
        cfg, lay = doc["config"], doc["layout"]
        QH = [core.fold_to_H(row, cfg["l"]) for row in doc["transcript"]["q"]]
        rec = core.recover(QH, doc["verifier_rc"], doc["y16"], doc["chain_constants"]["Yc"], doc["chain_constants"]["Y16c"],
                           cfg["l"], lay["n_vus"], lay["steps"])
        match, total = core.matches_recorded(rec, doc["recorded_accumulators"], lay["steps"])
        assert total == e["accumulators"]
        if e["expected_recovery"] == "all":
            assert match == total, f"{e['file']}: {match}/{total} accumulators recovered (fixture says all leak)"
            assert all(rec["end_positions_consistent"])
            assert rec["components"] == doc["recovered_components"]
        else:
            assert match == 0, f"{e['file']}: q still leaks {match}/{total} accumulators"


def test_recovery_is_not_a_tautology() -> None:
    """Perturbing one value of q on H (one coefficient) breaks the recovery downstream of that column: the test
    genuinely reads the accumulators off q rather than off the recorded arrays."""
    _, doc = _fixtures()[0]
    cfg, lay = doc["config"], doc["layout"]
    q = [list(row) for row in doc["transcript"]["q"]]
    q[0][3] = (q[0][3] + 1) % core.P
    QH = [core.fold_to_H(row, cfg["l"]) for row in q]
    rec = core.recover(QH, doc["verifier_rc"], doc["y16"], doc["chain_constants"]["Yc"], doc["chain_constants"]["Y16c"],
                       cfg["l"], lay["n_vus"], lay["steps"])
    match, total = core.matches_recorded(rec, doc["recorded_accumulators"], lay["steps"])
    assert match < total
