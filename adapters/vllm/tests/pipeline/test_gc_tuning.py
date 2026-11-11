"""verity_vllm.pipeline.gc_tuning: process-level cyclic-GC thresholds for the Build's derives and compose, the fold worker, the Match
checkers and the manifest build.

The setting is content-neutral by construction (the collector only frees unreachable cycles); these tests pin
(1) the env contract in fresh interpreters, (2) that "default" leaves the interpreter alone, (3) that the fold's
digest, instances, steps and report are identical under the interpreter defaults and the tuned thresholds,
(4) that the entry points call it, (5) that a tuned process's heap is frozen at exit, once, and "default"'s is not, and
(6) that `tuned` holds the thresholds for one block (a rank's Program load) and gives the caller's back.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys

import pytest

from verity_vllm.pipeline import gc_tuning
from tests.observe.test_fold_synthetic import memory, toy_log, toy_profile  # noqa: F401  (memory: fixture)
from verity_vllm.observe.fold.fold import fold

PROBE = ("import gc, json, sys\n"
         "from verity_vllm.pipeline.gc_tuning import tune\n"
         "before = gc.get_threshold(); r = tune(); after = gc.get_threshold()\n"
         "print(json.dumps({'before': before, 'ret': r, 'after': after}))\n")


def _probe(env_value: str | None) -> dict:
    env = dict(os.environ)
    env.pop(gc_tuning.ENV, None)
    if env_value is not None:
        env[gc_tuning.ENV] = env_value
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True).stdout
    import json
    return json.loads(out)


def _interpreter_default() -> list[int]:
    """A bare fresh interpreter's own thresholds, the A/B's baseline: CPython 3.14 moved them from (700, 10, 10) to (2000, 10, 10)."""
    import json
    probe = "import gc, json; print(json.dumps(gc.get_threshold()))"
    return json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout)


@pytest.mark.parametrize("value,expect", [
    (None, list(gc_tuning.TUNED)),          # unset -> tuned
    ("", list(gc_tuning.TUNED)),            # empty -> tuned
    ("default", None),                      # the A/B control: interpreter untouched
    ("OFF", None),
    ("700,10,10", [700, 10, 10]),           # explicit
    ("50000, 20, 100", [50000, 20, 100]),   # spaces tolerated
])
def test_env_contract_in_a_fresh_interpreter(value, expect):
    r = _probe(value)
    assert r["before"] == _interpreter_default(), "the interpreter's default thresholds are the baseline of the A/B"
    if expect is None:
        assert r["ret"] is None and r["after"] == r["before"]
    else:
        assert r["ret"] == expect and r["after"] == expect


EXIT_PROBE = ("import atexit, gc\n"
              "from verity_vllm.pipeline.gc_tuning import tune\n"
              "n0 = gc.get_freeze_count()\n"
              "atexit.register(lambda: print('frozen' if gc.get_freeze_count() > n0 else 'unfrozen', flush=True))\n"
              "tune(); tune()\n")


@pytest.mark.parametrize("value,expect", [(None, "frozen"), ("50000,20,100", "frozen"), ("default", "unfrozen")])
def test_a_tuned_process_freezes_its_heap_at_exit(value, expect):
    """atexit runs its hooks last-registered first, so a hook registered before `tune` sees the heap as the shutdown collections will."""
    env = dict(os.environ)
    env.pop(gc_tuning.ENV, None)
    if value is not None:
        env[gc_tuning.ENV] = value
    out = subprocess.run([sys.executable, "-c", EXIT_PROBE], env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == [expect]


def test_the_exit_freeze_is_registered_once(monkeypatch):
    hooks = []
    monkeypatch.setattr(gc_tuning.atexit, "register", hooks.append)
    monkeypatch.setattr(gc_tuning, "_freeze_registered", False)
    saved = gc.get_threshold()
    try:
        monkeypatch.setenv(gc_tuning.ENV, "default")
        gc_tuning.tune()
        assert hooks == []
        monkeypatch.setenv(gc_tuning.ENV, "")
        gc_tuning.tune()
        gc_tuning.tune()
    finally:
        gc.set_threshold(*saved)
    assert hooks == [gc.freeze]


@pytest.mark.parametrize("bad", ["1,2", "a,b,c", "1,2,3,4", "-1,2,3"])
def test_malformed_values_are_refused_loudly(bad, monkeypatch):
    monkeypatch.setenv(gc_tuning.ENV, bad)
    saved = gc.get_threshold()
    try:
        with pytest.raises(ValueError):
            gc_tuning.tune()
        assert gc.get_threshold() == saved
    finally:
        gc.set_threshold(*saved)


def test_fold_content_is_identical_under_default_and_tuned_thresholds(memory, monkeypatch):
    saved = gc.get_threshold()
    try:
        monkeypatch.setenv(gc_tuning.ENV, "default")
        gc.set_threshold(700, 10, 10)
        assert gc_tuning.tune() is None
        r_default = fold(toy_log(), toy_profile(), strict=True)
        monkeypatch.setenv(gc_tuning.ENV, "")
        assert gc_tuning.tune() == gc_tuning.TUNED and gc.get_threshold() == gc_tuning.TUNED
        r_tuned = fold(toy_log(), toy_profile(), strict=True)
    finally:
        gc.set_threshold(*saved)
    assert r_default.digest == r_tuned.digest
    assert r_default.instances == r_tuned.instances
    strip = lambda steps: [{k: v for k, v in s.items() if k != "seconds"} for s in steps]   # per-step wall: volatile
    assert strip(r_default.steps) == strip(r_tuned.steps)
    assert r_default.report.by_spec == r_tuned.report.by_spec and r_default.sampled == r_tuned.sampled
    assert len(r_default.body.nodes) == len(r_tuned.body.nodes)


def test_tuned_holds_the_thresholds_for_its_block_and_gives_the_callers_back(monkeypatch):
    saved = gc.get_threshold()
    try:
        gc.set_threshold(700, 10, 10)
        monkeypatch.setenv(gc_tuning.ENV, "")
        with gc_tuning.tuned() as th:
            assert th == gc_tuning.TUNED and gc.get_threshold() == gc_tuning.TUNED
        assert gc.get_threshold() == (700, 10, 10)
        with pytest.raises(RuntimeError, match="the step failed"):
            with gc_tuning.tuned():
                raise RuntimeError("the step failed")
        assert gc.get_threshold() == (700, 10, 10)
        monkeypatch.setenv(gc_tuning.ENV, "default")
        with gc_tuning.tuned() as th:
            assert th is None and gc.get_threshold() == (700, 10, 10)
        monkeypatch.setenv(gc_tuning.ENV, "50000,20,100")
        gc.set_threshold(*gc_tuning.TUNED)
        with gc_tuning.tuned():
            assert gc.get_threshold() == (50000, 20, 100)
        assert gc.get_threshold() == gc_tuning.TUNED
    finally:
        gc.set_threshold(*saved)


def test_a_ranks_program_load_runs_under_the_tuned_thresholds(monkeypatch):
    from verity_vllm.pipeline.tp import rank_replay
    seen = []
    monkeypatch.setattr(rank_replay, "_load", lambda *a: seen.append((a, gc.get_threshold())) or "loaded")
    monkeypatch.setenv(gc_tuning.ENV, "")
    saved = gc.get_threshold()
    assert rank_replay.load({"programs": {}}, 3, 8, definitions=False) == "loaded"
    assert seen == [(({"programs": {}}, 3, 8, False), gc_tuning.TUNED)] and gc.get_threshold() == saved


def test_the_entry_points_call_tune():
    import inspect
    from verity_vllm.pipeline import manifest
    from verity_vllm.pipeline import match as run_config
    from verity_vllm.check.match import batch_decomp
    from verity_vllm.check.match import global_match
    from verity_vllm.pipeline import build, global_program
    assert "gc_tuning import tune" in inspect.getsource(build.main)
    assert "gc_tuning import tune" in inspect.getsource(global_program.main)
    assert "gc_tuning import tune" in inspect.getsource(run_config.fold_worker)
    assert "gc_tuning import tune" in inspect.getsource(global_match.main)
    assert "gc_tuning import tune" in inspect.getsource(batch_decomp.main)
    assert "gc_tuning import tune" in inspect.getsource(manifest.main)
