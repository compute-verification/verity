"""`global_match_checks.lift` pins the fold record GM-01 read (laptop-safe, synthetic documents).

`global_match.json.impl.fold_load` is keyed by fold DIRECTORY; the pins sit one level down.  A lift that filters the top level for
`*sha256` keys gets {} -> None on every row, so a candidate that read a different fold record compares equal."""
from __future__ import annotations

import json

from tests.regression import lift_expected as LE
from tests.regression import resolver
from tests.regression.checks import global_match_checks as GM

PINS = {"program_sha256": "p" * 64, "instances_sha256": "i" * 64}


def _gm(match_dir: str, pins: dict) -> dict:
    notes = {"source": "record files", "instances_schema": "verity-gen/instances/v2", "operand_refs_form": "progressions", **pins}
    return {"verdict": "PASS", "failed_checks": [], "checks": [{"id": "G1", "status": "PASS"}], "requests": [{}],
            "impl": {"fold_load": {match_dir: notes}, "config": {"fold_load": {match_dir: notes}}}}


def test_the_pins_are_one_level_down_and_the_directory_does_not_enter():
    rec = GM.lift(_gm("/workspace/cp/sweep/r19-int-10/row/match", PINS))
    assert rec["fold_record_pins"] == PINS
    old = {k: v for k, v in _gm("/x/match", PINS)["impl"]["fold_load"].items() if k.endswith("sha256")} or None
    assert old is None, "the old top-level filter: the keys are paths"
    assert GM.lift(_gm("/elsewhere/match", PINS))["fold_record_pins"] == PINS, "same bytes read on another machine: same pins"
    assert GM.lift({"verdict": "PASS"})["fold_record_pins"] is None


def test_a_candidate_that_read_another_fold_record_now_differs_by_name():
    rec = GM.lift(_gm("/a/match", PINS))
    cand = GM.lift(_gm("/b/match", {**PINS, "instances_sha256": "j" * 64}))
    assert resolver.diff(rec, cand) == [f'$.fold_record_pins.instances_sha256: expected "{"i" * 64}" != actual "{"j" * 64}"']


def test_every_frozen_contract_with_a_global_match_pins_its_fold_record():
    for p in sorted(resolver.EXPECTED_DIR.glob("*.json")):
        c = json.loads(p.read_text()).get("global_match_checks")
        if c is not None:
            assert set(c["fold_record_pins"] or {}) == {"program_sha256", "instances_sha256"}, p.name
        assert LE.dumps_expected(json.loads(p.read_text())) == p.read_text(), f"{p.name}: not in lift_expected's serialisation"
