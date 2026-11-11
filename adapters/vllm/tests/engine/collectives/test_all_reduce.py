"""`engine.collectives.all_reduce`: the CUDA communicator's seam for its all-reduce, over stand-ins of vLLM's
``CudaCommunicator``."""
from __future__ import annotations

import importlib
import importlib.abc
import sys
import types
from typing import Any

import pytest

from verity_vllm.engine import hooks
from verity_vllm.engine.collectives import all_reduce as ar


class Communicator:
    """``all_reduce`` as vLLM's: one route for every tensor, logged."""

    def __init__(self, log: list, group: str = "tp") -> None:
        self.log, self.group = log, group

    def all_reduce(self, input_: Any) -> Any:
        self.log.append(("vllm", self.group, input_))
        return ("vllm", input_)


@pytest.fixture
def seam():
    cls = type("CudaCommunicator", (Communicator,), {})
    s = ar.AllReduceSeam(communicator=cls)
    s.cls = cls
    try:
        yield s
    finally:
        s.uninstall()
        assert "all_reduce" not in vars(cls)


def tree(log: list):
    """Sums odd tensors itself and leaves the rest on vLLM's routes."""
    def all_reduce(comm: Any, input_: int, vllm: Any) -> Any:
        log.append(("tree", comm.group, input_))
        return ("tree", input_) if input_ % 2 else vllm(input_)
    return all_reduce


def test_a_communicator_with_the_replacement_reduces_through_it_and_reaches_vllms_routes_for_the_rest(seam):
    log: list = []
    seam.on_all_reduce("tree", lambda comm: tree(log) if comm.group == "tp" else None)
    tp, pp = seam.cls(log), seam.cls(log, "pp")
    assert tp.all_reduce(1) == ("tree", 1)
    assert tp.all_reduce(2) == ("vllm", 2)
    assert pp.all_reduce(3) == ("vllm", 3)
    assert log == [("tree", "tp", 1), ("tree", "tp", 2), ("vllm", "tp", 2), ("vllm", "pp", 3)]


def test_a_wrap_above_the_seam_still_sees_every_call(seam):
    log: list = []
    seam.on_all_reduce("tree", lambda comm: tree(log))
    comm = seam.cls(log)
    with hooks.Hooks(tag="observer") as h:
        h.wrap(comm, "all_reduce", lambda orig: lambda x: log.append(("observed", x)) or orig(x))
        comm.all_reduce(1)
        comm.all_reduce(2)
    assert [x for x in log if x[0] == "observed"] == [("observed", 1), ("observed", 2)]


def test_a_second_replacement_and_one_without_its_place_are_refused(seam):
    with pytest.raises(ValueError, match="engine.collectives.all_reduce's ALL_REDUCES"):
        seam.on_all_reduce("nccl", lambda comm: None)
    seam.on_all_reduce("tree", lambda comm: None)
    with pytest.raises(ValueError, match="'tree' already does"):
        seam.on_all_reduce("tree", lambda comm: None)


def test_it_wraps_vllms_cuda_communicator_when_given_none(monkeypatch):
    mod = types.ModuleType("vllm.distributed.device_communicators.cuda_communicator")
    mod.CudaCommunicator = type("CudaCommunicator", (Communicator,), {})
    for name in ("vllm", "vllm.distributed", "vllm.distributed.device_communicators"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    s = ar.AllReduceSeam()
    s.on_all_reduce("tree", lambda comm: None)
    try:
        assert "all_reduce" in vars(mod.CudaCommunicator)
        assert mod.CudaCommunicator([]).all_reduce(4) == ("vllm", 4)
    finally:
        s.uninstall()
    assert "all_reduce" not in vars(mod.CudaCommunicator)


def test_importing_it_imports_no_vllm(monkeypatch):
    class NoVllm(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
            assert name.split(".")[0] != "vllm", f"engine.collectives.all_reduce imported {name}"

    for name in [m for m in sys.modules if m.split(".")[0] == "vllm"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NoVllm(), *sys.meta_path])
    importlib.reload(ar)
    assert isinstance(ar.SEAM, ar.AllReduceSeam)
