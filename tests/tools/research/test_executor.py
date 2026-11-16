"""The task executor's tasks/v1 side (`research.executor`): each task keyed (an unkeyed one by its check alone), each key's
verdicts folded, and the executor's verdicts compared with a run's records."""
from __future__ import annotations

import json

from research.executor import ERROR, Shadowed, agrees, compare, fold, keyed, old_verdicts, shadow_report, task_target
from research.landing import AFTER_LANDING
from research.store import vocab

K_SUITE, K_LEAN = "a" * 128, "b" * 128
#: the jobs the graph lists whole, unkeyed (a build, or a job that keeps no verdict of its own)
WHOLE = tuple(j for j in AFTER_LANDING if j not in ("pytest", "lean-audit", "flock-rust-tests", "circuit-check"))
N_TASKS = 5 + len(WHOLE)


def _graph(tip: str) -> dict:
    """A tip's tasks/v1 graph: a suite and a Lake package keyed by their inputs (the same at every tip here), a job keyed by
    the tip, circuit-check's two slices, unkeyed, and every other after-landing job whole, unkeyed."""
    return {"schema": "tasks/v1", "commit": tip, "kind": "after-landing", "tasks": [
        {"id": "pytest:pkg", "key": K_SUITE}, {"id": "lean-audit:Security", "key": K_LEAN},
        {"id": "flock-rust-tests", "key": tip[:32] * 4},
        {"id": "circuit-check:1/2", "key": None, "unkeyed": "per target"},
        {"id": "circuit-check:2/2", "key": None, "unkeyed": "per target"},
        *({"id": j, "key": None, "unkeyed": "no verdict of its own"} for j in WHOLE)]}


def test_each_task_is_keyed_and_an_unkeyed_one_by_its_check_alone():
    tip, other = "1" * 40, "2" * 40
    keys = {t["id"]: t["key"] for t in keyed(_graph(tip))["tasks"]}
    assert keys["pytest:pkg"] == K_SUITE and len(set(keys.values())) == N_TASKS
    assert keyed(_graph(other))["tasks"][3]["key"] != keys["circuit-check:1/2"], "an unkeyed task's key is its check's alone"
    assert keyed(_graph(other))["tasks"][3]["unkeyed"] == "per target"
    assert task_target("sha512:" + K_SUITE) == task_target(K_SUITE) == f"task:{K_SUITE}"


def test_error_is_not_run_and_clears_the_fail_before_it():
    """A false fail (a run that gave no result, labelled fail) written over as error reads as not run, and the key's rerun
    that passes is its verdict; error never reads as pass or fail."""
    assert fold(["fail", ERROR]) == fold([ERROR]) == "pending"
    assert fold(["fail", ERROR, "pass"]) == "pass" and fold([ERROR, "fail"]) == "fail"
    assert ERROR in vocab.TASK_VERDICTS, "the store takes the executor's error label"


def test_a_fail_stands_when_a_rerun_passes():
    """A flake must not go green on a rerun: a fail with no error label after it is the key's verdict whatever came later,
    and only an error written after it (a false fail relabelled) lets a later pass count."""
    assert fold(["fail", "pass"]) == "fail"
    assert fold(["pass", "fail", "pass"]) == "fail"
    assert fold(["fail", ERROR, "pass"]) == "pass"
    assert fold(["fail", ERROR, "pass", "fail", "pass"]) == "fail"
    assert fold(["pass", ERROR]) == "pending"
    assert fold(["fail", "pass", ERROR]) == "pending"
    assert fold([]) == "pending"


def test_the_comparison_is_per_task_where_the_run_records_one_and_per_job_where_it_records_the_job_whole():
    ids = {"pytest:pkg": "k1", "circuit-check:1/2": "k2", "circuit-check:2/2": "k3", "lean-unit-cut": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "circuit-check": "fail", "lean-unit-cut": "pass"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "fail", "k4": "pass"})
    assert got == {"units": 4, "agree": 4, "pending": [], "disagree": [], "missing": [], "executor_only": []}
    assert agrees(got)


def test_a_graph_missing_a_part_the_run_has_never_agrees_fully():
    ids = {"pytest:pkg": "k1", "lean-unit-cut": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "pytest:other": "pass", "lean-unit-cut": "pass", "ratchet-gate": "pass"}
    got = compare(ids, old, {"k1": "pass", "k4": "pass"})
    assert got["missing"] == ["pytest:other", "ratchet-gate"] and got["executor_only"] == [] and not agrees(got)
    assert {"unit": "ratchet-gate", "old": "pass", "executor": "missing"} in got["disagree"] and got["agree"] == 3
    text, ok = shadow_report([Shadowed("1" * 40, "2026-11-01T01:00:00Z", got)] * 3, 10)
    assert not ok and "missing in executor 2" in text and "    ratchet-gate: missing in executor (run pass)" in text, text


def test_an_executor_only_task_is_listed_and_disagrees():
    ids = {"pytest:pkg": "k1", "pytest:new": "k2", "lean-unit-cut": "k3", "circuit-check:1/1": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "circuit-check": "pass"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "pending", "k4": "pass"})
    # pytest's suites are recorded one by one, so pytest:new has no counterpart; lean-unit-cut the run didn't run at all
    assert got["executor_only"] == ["lean-unit-cut", "pytest:new"] and got["missing"] == [] and not agrees(got)
    assert {"unit": "lean-unit-cut", "old": "none", "executor": "pending"} in got["disagree"] and got["pending"] == []
    text, _ = shadow_report([Shadowed("1" * 40, "2026-11-01T01:00:00Z", got)], 10)
    assert "executor only 2" in text and "    pytest:new: executor only (executor pass)" in text, text
    assert not agrees({k: v for k, v in compare({"a": "k"}, {"a": "pass"}, {"k": "pass"}).items()
                       if k not in ("missing", "executor_only")}), "a record without coverage counts never agrees"


def test_the_runs_records_read_as_the_graph_cuts_them(tmp_path):
    """A suite reused from the cache passes; one a step skipped by name, and the preflight, have no verdict; a suite the graph
    splits by file is compared whole through its parts."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "steps.jsonl").write_text(json.dumps({"step": "preflight-lock", "status": "passed"}) + "\n"
                                   + json.dumps({"step": "pytest", "status": "failed"}) + "\n")
    (d / "suites.json").write_text(json.dumps([{"name": "kept", "status": "cached"}, {"name": "lean", "status": "skipped"},
                                               {"name": "big", "status": "failed"}]))
    old = old_verdicts([d])
    assert old == {"pytest": "fail", "pytest:kept": "pass", "pytest:big": "fail"}, old
    ids = {"pytest:kept": "k1", "pytest:big:1/2": "k2", "pytest:big:2/2": "k3"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "fail"})
    assert got == {"units": 3, "agree": 3, "pending": [], "disagree": [], "missing": [], "executor_only": []}, got
    assert agrees(got)
    assert compare(ids, old, {"k1": "pass", "k2": "pass"})["pending"] == ["pytest", "pytest:big"]


def test_shadow_reports_the_switch_gate_once_three_tips_in_a_row_agree():
    full = {"units": 2, "agree": 2, "pending": [], "disagree": [], "missing": [], "executor_only": []}
    off = {"units": 2, "agree": 1, "pending": [], "disagree": [{"unit": "x", "old": "pass", "executor": "fail"}],
           "missing": [], "executor_only": []}
    recs = [Shadowed(f"{i}" * 40, f"2026-11-01T0{i}:00:00Z", r) for i, r in ((4, full), (3, full), (2, full), (1, off))]
    assert shadow_report(recs, 2)[1] and "the newest 3 agree" in shadow_report(recs, 2)[0]
    assert not shadow_report(recs[:2] + recs[3:], 10)[1]
