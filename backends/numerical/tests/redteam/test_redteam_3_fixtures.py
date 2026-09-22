"""red-team-3 fixtures (``fixtures/redteam-3/``): the recorded verdicts of the two forgery batteries run against the
machinery that landed in wave 3 -- the B-AIR LogUp realisation (``verify_lookup``) and the GPU A Python verifier.

Every case carries ``expected`` (accept/reject) and the recorded ``verdict``; a case with ``expected == "reject"`` and
``verdict == "ACCEPTED"`` is an accepted forgery and must be listed in the manifest's ``accepted_forgeries`` (and in
``backends/numerical/reports/redteam_3.md``).  The test pins the battery to its manifest so a re-run that silently
changes a verdict, or a fixture that goes missing, fails loudly."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4] / "fixtures" / "redteam-3"
MANIFEST = ROOT / "manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="fixtures/redteam-3 not present")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _battery(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_manifest_lists_every_battery_file() -> None:
    m = _manifest()
    assert m["schema"] == "verity/redteam-3/v1"
    for b in m["batteries"]:
        assert (ROOT / b["file"]).exists(), b["file"]
        assert b["target"] in {"T1", "T2"}


def test_every_case_has_expected_and_verdict() -> None:
    m = _manifest()
    for b in m["batteries"]:
        d = _battery(b["file"])
        cases = d["results"]
        assert len(cases) == b["cases"], (b["file"], len(cases))
        for c in cases:
            assert c["expected"] in {"accept", "reject"}, c
            assert c["verdict"] in {"ACCEPTED", "REJECTED", "ERROR"}, c
            assert c.get("case"), c


def test_honest_controls_accepted() -> None:
    m = _manifest()
    for b in m["batteries"]:
        d = _battery(b["file"])
        ctrl = [c for c in d["results"] if c["expected"] == "accept"]
        assert ctrl, b["file"]
        for c in ctrl:
            assert c["verdict"] == "ACCEPTED", (b["file"], c["case"])


def test_accepted_forgeries_match_manifest() -> None:
    m = _manifest()
    for b in m["batteries"]:
        d = _battery(b["file"])
        forged = sorted(c["case"] for c in d["results"] if c["expected"] == "reject" and c["verdict"] == "ACCEPTED")
        assert forged == sorted(b["accepted_forgeries"]), (b["file"], forged)


def test_no_case_errored() -> None:
    m = _manifest()
    for b in m["batteries"]:
        d = _battery(b["file"])
        errs = [c["case"] for c in d["results"] if c["verdict"] == "ERROR"]
        assert errs == b.get("errors", []), (b["file"], errs)
