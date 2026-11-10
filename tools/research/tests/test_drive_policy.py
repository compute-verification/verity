"""Drive's policy (`research.policy`, policy.toml): the runners report real verdicts, and what gates and what may stay red is
drive's config.  A failure an unexpired allowance covers isn't red: it names no culprit, reverts nothing and holds no promotion;
an expired one is as if absent, and its ring names it.  A job in `off_gate` holds no promotion; `research merge` and the quick
tier read the same file."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import research.drive as D
from research import merge, policy, quick_tier
from research import queue as Q
from research.drive import AFTER_LANDING
from test_drive_checked import _checked, _landed
from test_drive_review import GATED_TREE, Review, _main, _move_main, _owning, _rings
from test_queue import World, _git

LIVE, EXPIRED = "2099-01-01T00:00:00Z", "1970-01-01T00:00:00Z"   # the harness's clock reads 1970-01-12
TEST = "tests/test_x.py::test_y"
FP8 = "partition/gate-recomputed ScaledMmFp8Block_v1"


def _policy(*allow: tuple[str, str, str], off_gate: tuple[str, ...] = ("circuit-check",)) -> str:
    """policy.toml's text: ``off_gate`` and an allowance (id, owner, expires) for each of ``allow``."""
    return (f'schema = "{policy.SCHEMA}"\noff_gate = {json.dumps(list(off_gate))}\n'
            + "".join(f'[[allow]]\nid = "{i}"\nowner = "{o}"\nexpires = "{e}"\nreason = "red; its fix is landing"\n'
                      for i, o, e in allow))


# -- the file


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
    _move_main(world, {"tools/verity/check/ci.toml": ci})
    assert merge.declared_jobs(world.origin, _main(world)) == ["pytest", "circuit-check"]
    _move_main(world, {"tools/verity/check/policy.toml": _policy()})
    assert merge.declared_jobs(world.origin, _main(world)) == ["pytest"]
    _move_main(world, {"tools/verity/check/policy.toml": None,                   # ci.toml's own `gate = false`
                       "tools/verity/check/ci.toml": ci.replace('name = "circuit-check"\n', 'name = "circuit-check"\ngate = false\n')})
    assert merge.declared_jobs(world.origin, _main(world)) == ["pytest"]
    _move_main(world, {"tools/verity/check/policy.toml": "not toml ["})
    assert "not TOML" in merge.declared_jobs(world.origin, _main(world))


# -- the quick tier: main's policy at its base


def _tier(*rows: dict) -> dict:
    return {"kind": quick_tier.KIND, "head": "h" * 40, "rc": 1, "rows": list(rows)}


def test_the_quick_tier_passes_a_run_whose_every_failure_mains_policy_allows(tmp_path):
    world = World(tmp_path)
    row = {"name": "pkg", "status": "failed", "failed_tests": [TEST], "beyond_tests": False}
    other = {"name": "other", "status": "failed", "failed_tests": ["tests/test_o.py::test_o"], "beyond_tests": False}
    assert Q.allowed(world.origin, _main(world)) is None                         # no policy.toml: nothing is allowed
    _move_main(world, {"tools/verity/check/policy.toml": _policy((f"suite pkg::{TEST}", "ci", LIVE))})
    covers = Q.allowed(world.origin, _main(world))
    assert quick_tier.judge(_tier(row), "h" * 40, covers=covers) is None
    assert quick_tier.judge(_tier(row, other), "h" * 40, covers=covers).startswith("pkg: ")
    assert quick_tier.judge(_tier(row), "h" * 40) is not None                      # the run judges itself with no policy
    res = SimpleNamespace(doc={"tier": _tier(row)}, validation_status="failed")
    assert Q.judge_report(res, "r", "m", "h" * 40, {"state": "done", "rc": 1}, covers) is None
    assert "ended done (exit 0)" in Q.judge_report(res, "r", "m", "h" * 40, {"state": "done", "rc": 0}, covers)
    assert Q.judge_report(res, "r", "m", "h" * 40, {"state": "done", "rc": 1}) is not None
    _move_main(world, {"tools/verity/check/policy.toml": _policy((f"suite pkg::{TEST}", "ci", EXPIRED))})
    assert quick_tier.judge(_tier(row), "h" * 40, covers=Q.allowed(world.origin, _main(world))) is not None


# -- drive after landing


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _drive(world: World, monkeypatch, text: str) -> Review:
    """A review landing onto `checked` in a tree under drive policy ``text``; `pkg/` is pkg-owner's."""
    _owning(world, {"lane-1": [], "pkg-owner": ["pkg/"], "circuits": []})
    _move_main(world, {**GATED_TREE, policy.PATH: text})
    return Review(world, monkeypatch, land_into="checked")


def _run(h: Review, tmp_path, failed: dict[str, str] | None = None, suite: dict | None = None) -> None:
    """The after-landing run r-post-0 ended with exit 1: every step passed but ``failed`` ({step: its log}), and suite pkg's row
    ``suite`` (pytest's) failed."""
    d = tmp_path / "r-post-0"
    (d / "suites").mkdir(parents=True)
    failed = failed or {}
    recs = [{"step": s, "status": "failed" if s in failed else "passed", **({"log": f"/x/{s}.log"} if failed.get(s) else {})}
            for s in AFTER_LANDING]
    (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    for s, log in failed.items():
        (d / f"{s}.log").write_text(log)
    if suite is not None:
        (d / "suites.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed", "log": "/x/suites/pkg.log",
                                                    **suite}]))
        (d / "suites" / "pkg.log").write_text(f"{TEST.replace('::', ':2: in ')}\nE       assert False\n")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [d]


RED_TEST = {"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}
PKG_ROW = {"failed_tests": [TEST], "beyond_tests": False}


def test_a_failure_an_unexpired_allowance_covers_is_not_red_reverts_nothing_and_holds_no_promotion(world, monkeypatch, tmp_path):
    h = _drive(world, monkeypatch, _policy((f"suite pkg::{TEST}", "ci", LIVE)))
    tip = _landed(h, RED_TEST)
    _run(h, tmp_path, {"pytest": ""}, PKG_ROW)
    h.tick()
    pl = h.postland()
    assert not pl.get("red") and not _rings(h), h.posts
    assert pl["verified"] == tip and _main(world) == tip and pl["green"]["suite pkg"] == tip
    h.tick()
    assert _checked(world) == tip and not [p for p in h.posts if "reverted" in p], h.posts


def test_a_suite_failing_beyond_its_allowed_tests_is_red_as_before(world, monkeypatch, tmp_path):
    h = _drive(world, monkeypatch, _policy((f"suite pkg::{TEST}", "ci", LIVE)))
    main = _main(world)
    tip = _landed(h, RED_TEST)
    _run(h, tmp_path, {"pytest": ""}, {**PKG_ROW, "beyond_tests": True})
    h.tick()
    (ring,) = _rings(h)
    assert "suite pkg is red on `checked`" in ring and "Culprit: #1" in ring and "allowance" not in ring, ring
    assert _main(world) == main and _checked(world) == tip and h.postland()["red"]["suite pkg"]["culprit"][0] == 1


def test_an_expired_allowance_is_as_if_absent_and_the_red_it_no_longer_covers_names_it_and_rings_its_owner(world, monkeypatch,
                                                                                                         tmp_path):
    h = _drive(world, monkeypatch, _policy((f"suite pkg::{TEST}", "circuits", EXPIRED)))
    main = _main(world)
    tip = _landed(h, RED_TEST)
    _run(h, tmp_path, {"pytest": ""}, PKG_ROW)
    h.tick()
    (ring,) = _rings(h)
    assert "suite pkg is red on `checked`" in ring and "Culprit: #1" in ring, ring
    assert f"Its allowance in policy.toml expired, so it is red again: `suite pkg::{TEST}` (owner circuits, expires {EXPIRED}" \
        in ring and "@circuits" in ring.split(" queue: ")[0], ring
    r = h.postland()["red"]["suite pkg"]
    assert r["allowed_by"] == ["circuits"] and len(r["expired"]) == 1
    h.clock += D.FIX_FORWARD_S
    h.tick()
    (told,) = [p for p in h.posts if "#1 reverted on `checked`" in p]
    assert "@circuits" in told.split(" queue: ")[0], told
    assert _git(world.origin, "rev-parse", f"{_checked(world)}^") == tip and _main(world) == main


@pytest.mark.parametrize("extra, red", [("", False), ("FAIL width/overflow Foo_v1{N=8} (1 gate(s))\n", True)])
def test_a_circuit_check_failing_only_on_allowed_failures_is_not_red_and_one_more_is(world, monkeypatch, tmp_path, extra, red):
    h = _drive(world, monkeypatch, _policy((f"circuit-check {FP8}", "circuits", LIVE)))
    tip = _landed(h, {"pkg/x.circuit-check.py": "X = 1\n"})
    _run(h, tmp_path, {"circuit-check": f"FAIL {FP8}{{K=128,N=128,G=128}} (127 gate(s))\n{extra}"})
    h.tick()
    pl = h.postland()
    assert _main(world) == tip                                  # circuit-check is off the gate either way
    if red:
        assert pl["red"]["step circuit-check"]["culprit"][0] == 1 and _checked(world) == tip
    else:
        assert not pl.get("red") and _checked(world) == tip and pl["verified"] == tip, h.posts


@pytest.mark.parametrize("off", [True, False])
def test_a_red_job_drives_policy_takes_off_the_gate_holds_no_promotion(world, monkeypatch, tmp_path, off):
    gate = ("circuit-check", "flock-rust-tests") if off else ("circuit-check",)
    h = _drive(world, monkeypatch, _policy(off_gate=gate))
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    _run(h, tmp_path, {"flock-rust-tests": ""})
    h.tick()
    h.tick()
    assert _main(world) == (tip if off else main) and (h.postland().get("gated") == tip) is off
    assert any("step flock-rust-tests is red on `checked`" in p for p in h.posts), h.posts
