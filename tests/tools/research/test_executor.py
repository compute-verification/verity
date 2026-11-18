"""The task executor's tasks/v1 side (`research.executor`): each task keyed (an unkeyed one by its check alone), and each key's
verdicts folded."""
from __future__ import annotations

from research.executor import ERROR, fold, keyed, task_target
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

