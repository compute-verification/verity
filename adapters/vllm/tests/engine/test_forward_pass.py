"""`engine.forward_pass`: the model runner's seam around each forward pass, over stand-ins of vLLM's ``GPUModelRunner``."""
from __future__ import annotations

import importlib
import importlib.abc
import sys
from typing import Any

import pytest

from verity_vllm.engine import forward_pass as fp


class Runner:
    """``execute_model`` and ``sample_tokens`` as vLLM's: a scheduler output named ``early...`` returns its output from
    ``execute_model``, any other leaves ``execute_model_state`` for the sample, and ``raise`` raises."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.execute_model_state: Any = None

    def execute_model(self, scheduler_output: str, intermediate_tensors: Any = None, dummy_run: bool = False) -> Any:
        self.log.append(("execute", scheduler_output, dummy_run))
        if scheduler_output == "raise":
            raise ValueError("the forward failed")
        if scheduler_output.startswith("early"):
            return ("early", scheduler_output)
        self.execute_model_state = ("state", scheduler_output)
        return None

    def sample_tokens(self, grammar_output: Any) -> Any:
        self.log.append(("sample", grammar_output))
        state, self.execute_model_state = self.execute_model_state, None
        if state is None:
            return None
        if grammar_output == "raise":
            raise ValueError("the sample failed")
        return ("sampled", state[1])


class Observer:
    def __init__(self, log: list, name: str) -> None:
        self.log, self.name = log, name

    def begin(self, runner: Any, scheduler_output: Any) -> None:
        self.log.append((self.name, "begin", scheduler_output))

    def end(self, done: fp.ForwardEnd) -> None:
        self.log.append((self.name, "end", done.scheduler_output, done.state, done.output,
                         str(done.error) if done.error else None, done.index))


@pytest.fixture
def seam():
    cls = type("GPUModelRunner", (Runner,), {})
    s = fp.ForwardSeam(runners=(cls,))
    s.cls = cls
    try:
        yield s
    finally:
        s.uninstall()
        assert not {"execute_model", "sample_tokens"} & set(vars(cls))


def observed(seam: fp.ForwardSeam, *names: str) -> list:
    log: list = []
    for name in names:
        seam.on_forward(name, lambda runner, name=name: Observer(log, name))
    return log


def test_observers_see_a_pass_begin_in_their_order_and_end_in_the_reverse_once_it_sampled(seam):
    log = observed(seam, "route", "capture")
    runner = seam.cls(log)
    assert runner.execute_model("s0") is None
    assert runner.sample_tokens("g0") == ("sampled", "s0")
    runner.execute_model("s1")
    runner.sample_tokens("g1")
    assert log[:6] == [("capture", "begin", "s0"), ("route", "begin", "s0"), ("execute", "s0", False), ("sample", "g0"),
                       ("route", "end", "s0", ("state", "s0"), ("sampled", "s0"), None, 0),
                       ("capture", "end", "s0", ("state", "s0"), ("sampled", "s0"), None, 0)]
    assert log[-1] == ("capture", "end", "s1", ("state", "s1"), ("sampled", "s1"), None, 1)


def test_a_pass_whose_execute_model_returns_its_output_ends_there(seam):
    log = observed(seam, "route")
    runner = seam.cls(log)
    assert runner.execute_model("early0") == ("early", "early0")
    assert runner.sample_tokens("g0") is None
    assert log == [("route", "begin", "early0"), ("execute", "early0", False),
                   ("route", "end", "early0", None, ("early", "early0"), None, 0), ("sample", "g0")]


def test_a_dummy_run_is_no_pass_and_a_runner_without_observers_runs_vllms_own(seam):
    log: list = []
    seam.on_forward("route", lambda runner: Observer(log, "route") if runner.log is log else None)
    runner, other = seam.cls(log), seam.cls([])
    runner.execute_model("s0", dummy_run=True)
    other.execute_model("s0")
    other.sample_tokens("g0")
    assert log == [("execute", "s0", True)]
    assert other.log == [("execute", "s0", False), ("sample", "g0")]


def test_a_pass_that_raises_ends_with_its_error_which_propagates(seam):
    log = observed(seam, "route")
    runner = seam.cls(log)
    with pytest.raises(ValueError, match="the forward failed"):
        runner.execute_model("raise")
    runner.execute_model("s1")
    with pytest.raises(ValueError, match="the sample failed"):
        runner.sample_tokens("raise")
    assert [x for x in log if x[1] == "end"] == [("route", "end", "raise", None, None, "the forward failed", 0),
                                                 ("route", "end", "s1", ("state", "s1"), None, "the sample failed", 1)]


def test_a_pass_whose_sample_never_came_ends_when_the_next_one_starts(seam):
    log = observed(seam, "route")
    runner = seam.cls(log)
    runner.execute_model("s0")
    runner.execute_model_state = None
    runner.execute_model("early1")
    assert [x for x in log if x[1] == "end"] == [
        ("route", "end", "s0", ("state", "s0"), None, "the runner's next execute_model came before this pass's sample_tokens",
         0),
        ("route", "end", "early1", None, ("early", "early1"), None, 1)]


def test_an_observer_is_refused_without_its_place(seam):
    with pytest.raises(ValueError, match="engine.forward_pass's FORWARDS"):
        seam.on_forward("tick", lambda runner: None)


def test_it_wraps_each_runner_module_once_imported_and_imports_none_itself(tmp_path, monkeypatch):
    sites = ("forward_pass_runner_v2", "forward_pass_runner_v1")
    for site in sites:
        (tmp_path / f"{site}.py").write_text("class GPUModelRunner:\n    def execute_model(self, so):\n        return so\n\n"
                                             "    def sample_tokens(self, g):\n        return g\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(fp, "RUNNERS", sites)
    s = fp.ForwardSeam()
    try:
        s.on_forward("route", lambda runner: None)
        assert not set(sites) & set(sys.modules)
        modules = [importlib.import_module(site) for site in sites]
        assert {m.GPUModelRunner.execute_model.__qualname__ for m in modules} == {
            "ForwardSeam._execute.<locals>.execute_model"}
        assert modules[0].GPUModelRunner().execute_model("s0") == "s0"
    finally:
        s.uninstall()
        for site in sites:
            sys.modules.pop(site, None)
    assert {m.GPUModelRunner.execute_model.__qualname__ for m in modules} == {"GPUModelRunner.execute_model"}
    assert {m.GPUModelRunner.sample_tokens.__qualname__ for m in modules} == {"GPUModelRunner.sample_tokens"}


def test_importing_it_imports_no_vllm(monkeypatch):
    class NoVllm(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
            assert name.split(".")[0] != "vllm", f"engine.forward_pass imported {name}"

    for name in [m for m in sys.modules if m.split(".")[0] == "vllm"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NoVllm(), *sys.meta_path])
    importlib.reload(fp)
    assert isinstance(fp.SEAM, fp.ForwardSeam)
