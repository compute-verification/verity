"""The policy (`research.policy`, policy.toml): the runners report real verdicts, and what gates and what may stay red is its
config.  A policy of any other form is refused; an allowance covers a suite whole or test by test, a circuit-check failure by
its check code and family, and only until it expires; `research.merge` requires every job but those off the gate, and the
quick tier passes a run whose every failure `main`'s policy allows."""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from research import merge, policy, quick_tier
from research import queue as Q
from test_landing import _main, _move_main, _policy
from test_queue import World

LIVE, EXPIRED = "2099-01-01T00:00:00Z", "1970-01-01T00:00:00Z"   # the harness's clock reads 1970-01-12


TEST = "tests/test_x.py::test_y"


FP8 = "partition/gate-recomputed ScaledMmFp8Block_v1"


@pytest.mark.parametrize("text, why", [
    ('schema = "policy/v1"\n', "schema"),
    ('schema = "policy/v0"\ngate = ["x"]\n', "unknown keys"),
    ('schema = "policy/v0"\noff_gate = "circuit-check"\n', "off_gate must be a list"),
    (_policy(("suite pkg", "ci", LIVE)).replace('reason = "red; its fix is landing"\n', ""), "an allowance is id, owner"),
    (_policy(("pkg::test", "ci", LIVE)), "is none of"),
    (_policy(("circuit-check partition", "ci", LIVE)), "is none of"),
    (_policy(("suite pkg", "ci", LIVE), ("suite pkg", "ci", EXPIRED)), "listed twice"),
    (_policy(("suite pkg", "ci", "2026-11-02T18:00:00+01:00")), "isn't a UTC time"),
    (_policy(("suite pkg", "nobody", LIVE)), "owner 'nobody' isn't a handle"),
    ("not toml [", "not TOML"),
])
def test_a_policy_of_any_other_form_is_refused_with_what_is_wrong(text, why):
    with pytest.raises(policy.PolicyError, match=why):
        policy.parse(text, handles={"ci", "circuits"})


def test_an_allowance_covers_a_suite_whole_or_test_by_test_and_only_until_it_expires():
    now = policy._utc("2026-11-01T00:00:00Z")
    row = {"failed_tests": [TEST], "beyond_tests": False}
    each = policy.parse(_policy((f"suite pkg::{TEST}", "ci", "2026-11-02T18:00:00Z")))
    assert each.suite("pkg", row, now).covered
    assert not each.suite("pkg", {**row, "beyond_tests": True}, now).covered      # it failed for more than its tests
    assert not each.suite("pkg", {**row, "failed_tests": [TEST, "tests/test_x.py::test_z"]}, now).covered
    assert not each.suite("other", row, now).covered and not each.suite("pkg", {"failed_tests": []}, now).covered
    late = each.suite("pkg", row, policy._utc("2026-11-02T18:00:00Z"))
    assert not late.covered and [a.id for a in late.expired] == [f"suite pkg::{TEST}"]
    whole = policy.parse(_policy(("suite pkg", "ci", LIVE)))
    assert whole.suite("pkg", {"failed_tests": [], "beyond_tests": True}, now).covered


def test_a_circuit_check_failure_is_covered_by_its_check_code_and_family_and_each_must_be():
    log = (f"circuit-check: 3 target(s)\nFAIL {FP8}{{K=128,N=128,G=128}} (127 gate(s))\n"
           f"FAIL {FP8}{{K=256,N=128,G=128}} (255 gate(s))\n")
    assert policy.circuit_keys(log) == [FP8, FP8]
    pol, now = policy.parse(_policy((f"circuit-check {FP8}", "circuits", LIVE))), 0.0
    assert pol.circuit(log, now).covered
    assert not pol.circuit(log + "FAIL width/overflow Foo_v1{N=8}\n", now).covered
    assert not pol.circuit("circuit-check: exit 2, no report\n", now).covered      # a failure naming no target is never covered


def test_merge_requires_every_job_but_those_off_the_gate_and_refuses_an_unreadable_policy(tmp_path):
    world = World(tmp_path)
    ci = 'schema = "ci/v0"\n' + "".join(f'[[job]]\nname = "{s}"\nowner = "ci"\ngroup = 1\n' for s in ("pytest", "circuit-check"))
    _move_main(world, {"tools/check/ci.toml": ci})
    assert merge.declared_jobs(world.origin, _main(world)) == ["pytest", "circuit-check"]
    _move_main(world, {"tools/check/policy.toml": _policy()})
    assert merge.declared_jobs(world.origin, _main(world)) == ["pytest"]
    _move_main(world, {"tools/check/policy.toml": None,                   # ci.toml's own `gate = false`
                       "tools/check/ci.toml": ci.replace('name = "circuit-check"\n', 'name = "circuit-check"\ngate = false\n')})
    assert merge.declared_jobs(world.origin, _main(world)) == ["pytest"]
    _move_main(world, {"tools/check/policy.toml": "not toml ["})
    assert "not TOML" in merge.declared_jobs(world.origin, _main(world))


def _tier(*rows: dict) -> dict:
    return {"kind": quick_tier.KIND, "head": "h" * 40, "rc": 1 if rows else 0, "rows": list(rows)}


def test_the_quick_tier_passes_a_run_whose_every_failure_mains_policy_allows(tmp_path):
    world = World(tmp_path)
    row = {"name": "pkg", "status": "failed", "failed_tests": [TEST], "beyond_tests": False}
    other = {"name": "other", "status": "failed", "failed_tests": ["tests/test_o.py::test_o"], "beyond_tests": False}
    assert Q.allowed(world.origin, _main(world)) is None                         # no policy.toml: nothing is allowed
    _move_main(world, {"tools/check/policy.toml": _policy((f"suite pkg::{TEST}", "ci", LIVE))})
    covers = Q.allowed(world.origin, _main(world))
    assert quick_tier.judge(_tier(row), "h" * 40, covers=covers) is None
    assert quick_tier.judge(_tier(row, other), "h" * 40, covers=covers).startswith("pkg: ")
    assert quick_tier.judge(_tier(row), "h" * 40) is not None                      # the run judges itself with no policy
    _move_main(world, {"tools/check/policy.toml": _policy((f"suite pkg::{TEST}", "ci", EXPIRED))})
    assert quick_tier.judge(_tier(row), "h" * 40, covers=Q.allowed(world.origin, _main(world))) is not None


ALLOWED_ROW = {"name": "pkg", "status": "failed", "failed_tests": [TEST], "beyond_tests": False}
OTHER_ROW = {"name": "other", "status": "failed", "failed_tests": ["tests/test_o.py::test_o"], "beyond_tests": False}


def _covers(row: dict) -> bool:
    return row.get("failed_tests") == [TEST]


@pytest.mark.parametrize("rows, end, covers, refusal", [
    ((ALLOWED_ROW,), ("failed", 1, "failed"), _covers, None),
    ((), ("done", 0, "passed"), None, None),
    ((ALLOWED_ROW,), ("done", 1, "failed"), _covers, r"facts show a pass, but it ended done \(exit 1\) with validation failed"),
    ((ALLOWED_ROW,), ("failed", 1, "passed"), _covers, r"ended failed \(exit 1\) with validation passed"),
    ((ALLOWED_ROW,), ("failed", 2, "failed"), _covers, r"ended failed \(exit 2\) with validation failed"),
    ((ALLOWED_ROW,), ("cancelled", 1, "failed"), _covers, r"ended cancelled \(exit 1\)"),
    ((ALLOWED_ROW,), ("done", 0, "failed"), _covers, r"ended done \(exit 0\)"),
    ((ALLOWED_ROW,), ("failed", 1, "failed"), None, r"^pkg: "),
    ((ALLOWED_ROW, OTHER_ROW), ("failed", 1, "failed"), _covers, r"^pkg: .*; other: "),
    ((), ("failed", 1, "passed"), None, r"ended failed \(exit 1\) with validation passed"),
], ids=["every failure allowed, ended as the runner records exit 1", "no failure, ended done", "ended done with exit 1",
        "ended failed with validation passed", "ended failed with exit 2", "cancelled", "ended done with exit 0",
        "a failure nothing allows", "a failure the policy doesn't allow beside one it does", "no failure, ended failed"])
def test_the_quick_tier_passes_a_run_whose_every_failure_is_allowed_only_when_it_ended_as_the_runner_records_its_exit_1(
        rows, end, covers, refusal):
    """`quick_tier.py` judges itself with no policy, so a run whose every failure is allowed exits 1 with validation failed,
    and the runner records exit 1 as `failed`: that end, and the clean tier's `done`, exit 0 and validation passed, are the
    only ends of a pass; an uncovered failure, another exit, a validation that disagrees and a cancelled run each refuse."""
    state, rc, validation = end
    res = SimpleNamespace(doc={"tier": _tier(*rows)}, validation_status=validation)
    why = Q.judge_report(res, "r", "m", "h" * 40, {"state": state, "rc": rc}, covers)
    assert why is None if refusal is None else re.search(refusal, why), why
