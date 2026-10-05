"""BabyBear embedding rules, the forgery battery and the z3 harness (z3 marked slow: pod only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from verity_numerical.checker import REAL
from verity_numerical.redteam import babybear as BB
from verity_numerical.redteam import battery_babybear as BAT

ROOT = Path(__file__).resolve().parents[4]
FIXTURES = ROOT / BAT.FIXTURE_DIR

slow = pytest.mark.skipif(not os.environ.get("VERITY_SLOW"), reason="z3 sweeps run on the pod (VERITY_SLOW=1)")


@pytest.mark.slow
def test_rules_analysis_sound_with_all_rules():
    """Every predicate of every embedded gadget is read faithfully mod p under the full rule set."""
    rep = BB.rules_report(REAL)
    assert rep["embedded_unsound"] == [], rep["embedded_unsound"]
    assert set(rep["rules"]) == set(BB.RULES)


def test_embedded_checker_accepts_honest_golden():
    inst = BAT.honest_instances(REAL, limit=3)
    assert inst
    for _, x in inst:
        h = BB.gen_step_bb(REAL, x["c"], x["a"], x["b"])
        from verity_numerical.checker.step import step_output

        y = step_output(REAL, x["c"], x["a"], x["b"])
        ok, why = BB.accepts_step_bb(REAL, x["c"], x["a"], x["b"], y, h)
        assert ok, why


def _load():
    assert (FIXTURES / "manifest.json").exists(), "run python -m verity_numerical.redteam.battery_babybear"
    return BAT.load_battery(FIXTURES)


def test_battery_manifest_consistent():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    cases = _load()
    assert manifest["total"] == len(cases) >= 60
    assert manifest["field"]["p"] == BB.P
    assert all(c.expected == "reject" for c in cases)
    assert {c.cls for c in cases} >= {"wrap", "zero_swap", "hint_swap", "chain", "epilogue", "z3"}


def test_integer_checker_rejects_every_case():
    cases = [c for c in _load() if c.level != "gadget"]
    rep = BAT.run_battery(lambda c: BAT._int_accepts(c, REAL)[0], cases)
    assert rep.ok, rep.render()


def test_embedded_checker_rejects_every_case():
    cases = _load()

    def accepts(c):
        if c.level == "gadget":
            return BAT._gadget_accepts(c, REAL, BB.DEFAULT)[0]
        return BAT._embedded_accepts(c, REAL)[0]

    rep = BAT.run_battery(accepts, cases)
    assert rep.ok, rep.render()


def test_naive_modp_checker_has_teeth():
    """A checker asserting the WP2 predicates mod p without the rules accepts forgeries: the battery bites."""
    cases = [c for c in _load() if c.level != "gadget"]
    rep = BAT.run_battery(lambda c: BAT._naive_accepts(c, REAL)[0], cases)
    assert not rep.ok
    assert any(cid.startswith("chain") for cid in rep.accepted), rep.accepted


def test_dropping_each_necessary_rule_admits_its_z3_model():
    cases = [c for c in _load() if c.level == "gadget"]
    assert cases, "battery built without --z3 models"
    for c in cases:
        without = BB.Embedding(BB.ALL_RULES - {c.dropped_rule})
        assert BAT._gadget_accepts(c, REAL, without)[0], (c.gadget, c.dropped_rule)
        assert not BAT._gadget_accepts(c, REAL, BB.DEFAULT)[0], (c.gadget, c.dropped_rule)


def test_run_battery_report_shape():
    cases = _load()[:10]
    rep = BAT.run_battery(lambda c: False, cases)
    assert rep.ok and rep.total == 10 and not rep.accepted
    rep = BAT.run_battery(lambda c: True, cases)
    assert not rep.ok and len(rep.accepted) == 10
    assert "FAIL" in rep.render()


@slow
def test_z3_all_rules_unsat_small_gadgets():
    from verity_numerical.redteam import z3_babybear as Z

    for name in ("bf16_decode", "product", "state_out", "epilogue"):
        v = Z.run_gadget(name, BB.Embedding(BB.ALL_RULES), timeout_ms=60_000, procs=1)
        assert v.sat == 0 and v.unknown == 0, name
