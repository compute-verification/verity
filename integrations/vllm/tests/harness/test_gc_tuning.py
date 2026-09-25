"""verity_vllm.pipeline.gc_tuning: process-level cyclic-GC thresholds for the fold worker and the Match checkers.

The setting is content-neutral by construction (the collector only frees unreachable cycles); these tests pin
(1) the env contract in fresh interpreters, (2) that "default" leaves the interpreter alone, (3) that the fold's
digest, instances, steps and report are identical under the interpreter defaults and the tuned thresholds, and
(4) that the three entry points call it.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys

import pytest

from verity_vllm.pipeline import gc_tuning
from tests.observe.test_fold_synthetic import memory, toy_log, toy_profile  # noqa: F401  (memory: fixture)
from verity_vllm.observe.fold import fold

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
    assert r["before"] == [700, 10, 10], "CPython default thresholds are the baseline of the A/B"
    if expect is None:
        assert r["ret"] is None and r["after"] == r["before"]
    else:
        assert r["ret"] == expect and r["after"] == expect


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


def test_fold_content_is_identical_under_default_and_tuned_thresholds(memory, monkeypatch):  # noqa: F811
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


def test_the_three_entry_points_call_tune():
    import inspect
    from verity_vllm.pipeline import match as run_config
    from verity_vllm.correspondence import batch_decomp
    from verity_vllm.check import global_match
    assert "gc_tuning import tune" in inspect.getsource(run_config.fold_worker)
    assert "gc_tuning import tune" in inspect.getsource(global_match.main)
    assert "gc_tuning import tune" in inspect.getsource(batch_decomp.main)
