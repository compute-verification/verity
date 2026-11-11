"""`engine.step_end`: the scheduler's seam at each step's end, over stand-ins of vLLM's ``Scheduler``."""
from __future__ import annotations

import importlib
import importlib.abc
import sys
import types
from typing import Any

import pytest

from verity_vllm.engine import hooks
from verity_vllm.engine import step_end as se


class Scheduler:
    """``update_from_output`` as vLLM's: the engine core outputs it makes of a step's scheduler and model outputs."""

    def __init__(self, log: list) -> None:
        self.log = log

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> dict:
        self.log.append(("vllm", scheduler_output))
        return {0: (scheduler_output, model_runner_output)}


class Async(Scheduler):
    """``AsyncScheduler``: a subclass that inherits ``update_from_output``."""


@pytest.fixture
def seam():
    cls = type("Scheduler", (Scheduler,), {})
    s = se.StepSeam(scheduler=cls)
    s.cls = cls
    try:
        yield s
    finally:
        s.uninstall()
        assert "update_from_output" not in vars(cls)


def observer(log: list, name: str):
    return lambda step: log.append((name, step.scheduler_output, step.model_output, step.outputs, step.index))


def test_each_observer_sees_each_step_end_in_its_order_once_vllms_call_returned(seam):
    log: list = []
    seam.on_step("replay", lambda sched: observer(log, "replay"))
    seam.on_step("step_trace", lambda sched: observer(log, "step_trace"))
    sched = seam.cls(log)
    assert sched.update_from_output("s0", "m0") == {0: ("s0", "m0")}
    sched.update_from_output("s1", "m1")
    assert log == [("vllm", "s0"), ("step_trace", "s0", "m0", {0: ("s0", "m0")}, 0),
                   ("replay", "s0", "m0", {0: ("s0", "m0")}, 0),
                   ("vllm", "s1"), ("step_trace", "s1", "m1", {0: ("s1", "m1")}, 1),
                   ("replay", "s1", "m1", {0: ("s1", "m1")}, 1)]


def test_each_scheduler_makes_its_own_observers_and_counts_its_own_steps(seam):
    log: list = []
    made: list = []

    def make(sched: Any) -> Any:
        made.append(sched)
        return None if sched.log is not log else observer(log, "step_trace")

    seam.on_step("step_trace", make)
    a, b = seam.cls(log), seam.cls([])
    for so in ("a0", "a1"):
        a.update_from_output(so, None)
        b.update_from_output(so, None)
    assert made == [a, b]
    assert [x for x in log if x[0] == "step_trace"] == [("step_trace", "a0", None, {0: ("a0", None)}, 0),
                                                       ("step_trace", "a1", None, {0: ("a1", None)}, 1)]
    assert b.log == [("vllm", "a0"), ("vllm", "a1")]


def test_a_subclass_and_a_wrap_of_one_scheduler_both_reach_the_observers(seam):
    log: list = []
    seam.on_step("step_trace", lambda sched: observer(log, "step_trace"))
    sched = type("Async", (seam.cls, Async), {})(log)
    with hooks.Hooks(tag="instance") as h:
        h.wrap(sched, "update_from_output", lambda orig: lambda so, mo: ("instance", orig(so, mo)))
        assert sched.update_from_output("s0", "m0") == ("instance", {0: ("s0", "m0")})
    assert log == [("vllm", "s0"), ("step_trace", "s0", "m0", {0: ("s0", "m0")}, 0)]


def test_an_observers_error_fails_the_step(seam):
    def fail(step: se.StepEnd) -> None:
        raise RuntimeError("the observer failed")

    seam.on_step("step_trace", lambda sched: fail)
    with pytest.raises(RuntimeError, match="the observer failed"):
        seam.cls([]).update_from_output("s0", None)


def test_an_observer_is_refused_without_its_place_twice_or_after_a_scheduler_made_its_own(seam):
    with pytest.raises(ValueError, match="engine.step_end's STEPS"):
        seam.on_step("tick", lambda sched: None)
    seam.on_step("step_trace", lambda sched: None)
    with pytest.raises(ValueError, match="already added"):
        seam.on_step("step_trace", lambda sched: None)
    sched = seam.cls([])
    sched.update_from_output("s0", None)
    with pytest.raises(RuntimeError, match="before vLLM builds the engine"):
        seam.on_step("replay", lambda sched: None)


def test_it_wraps_vllms_scheduler_when_given_none(monkeypatch):
    mod = types.ModuleType("vllm.v1.core.sched.scheduler")
    mod.Scheduler = type("Scheduler", (Scheduler,), {})
    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    s = se.StepSeam()
    s.on_step("step_trace", lambda sched: None)
    try:
        assert "update_from_output" in vars(mod.Scheduler)
        assert [e.tag for e in hooks.live("step-end")] == ["step-end"]
    finally:
        s.uninstall()
    assert "update_from_output" not in vars(mod.Scheduler)


def test_importing_it_imports_no_vllm(monkeypatch):
    class NoVllm(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
            assert name.split(".")[0] != "vllm", f"engine.step_end imported {name}"

    for name in [m for m in sys.modules if m.split(".")[0] == "vllm"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NoVllm(), *sys.meta_path])
    importlib.reload(se)
    assert isinstance(se.SEAM, se.StepSeam)
