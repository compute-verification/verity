"""`engine.core_gate`: the engine core's seam between steps, over stand-ins of vLLM's ``EngineCoreProc`` whose input queue
follows a script and never waits."""
from __future__ import annotations

import importlib
import importlib.abc
import queue
import sys
import threading
import types
from collections import deque
from typing import Any

import pytest

from verity_vllm.engine import core_gate as cg
from verity_vllm.pipeline import tick_plugin as tp

EMPTY = object()
SHUTDOWN = "the engine core is shutting down"


class Inputs:
    """An input queue: ``now`` is what it holds, and each ``get`` that finds it empty takes the next of ``later``, raising
    `queue.Empty` for `EMPTY`, as a wait that timed out."""

    def __init__(self, log: list, now: list, later: list) -> None:
        self.log, self.now, self.later = log, deque(now), deque(later)

    def empty(self) -> bool:
        return not self.now

    def get_nowait(self) -> Any:
        return self.now.popleft()

    def get(self, timeout: float | None = None) -> Any:
        self.log.append(("get", timeout))
        if self.now:
            return self.now.popleft()
        assert self.later, "the loop waited on an input queue nothing more arrives on"
        got = self.later.popleft()
        if got is EMPTY:
            raise queue.Empty
        return got


class Aborts:
    def __init__(self, log: list) -> None:
        self.log, self.mutex, self.queue = log, threading.Lock(), self

    def clear(self) -> None:
        self.log.append(("aborts-cleared",))


class Sched:
    def __init__(self) -> None:
        self.work = False

    def has_requests(self) -> bool:
        return True


class Proc:
    """vLLM's ``EngineCoreProc`` as the seam drives it, each call logged."""

    def __init__(self, log: list, now: list = (), later: list = (), batches: list | None = None) -> None:
        self.log, self.scheduler, self.running = log, Sched(), True
        self.input_queue, self.aborts_queue = Inputs(log, list(now), list(later)), Aborts(log)
        self.batch_queue = None if batches is None else deque(batches)

    def _process_input_queue(self) -> None:
        self.log.append(("vllm-input",))

    def has_work(self) -> bool:
        self.log.append(("has_work", self.scheduler.work))
        return self.scheduler.work

    def is_running(self) -> bool:
        return self.running

    def _notify_idle_state_callbacks(self) -> None:
        self.log.append(("idle",))

    def _handle_client_request(self, kind: str, req: Any) -> None:
        self.log.append((kind, req))
        self.scheduler.work = self.scheduler.work or kind == "add"
        self.running = self.running and kind != "stop"

    def step(self) -> Any:
        self.log.append(("step", self.scheduler.has_requests()))
        return {}, True

    def step_with_batch_queue(self) -> Any:
        scheduling = self.scheduler.has_requests()
        self.log.append(("step", scheduling))
        if scheduling:
            self.batch_queue.appendleft("b")
        elif self.batch_queue:
            self.batch_queue.pop()
        return None, scheduling


class Gate:
    """A gate whose answers follow a script."""

    def __init__(self, log: list, name: str, ready: list, pending: list = (), poll_s: float = 0.25) -> None:
        self.log, self.name, self.poll_s = log, name, poll_s
        self.answers, self.pendings = deque(ready), deque(pending)

    def ready(self, scheduler: Any) -> bool:
        ok = self.answers.popleft()
        self.log.append((self.name, "ready", ok))
        return ok

    def pending(self) -> bool:
        p = self.pendings.popleft()
        self.log.append((self.name, "pending", p))
        return p

    def open(self, why: str) -> None:
        self.log.append((self.name, "open", why))


class Edge:
    def __init__(self, log: list, name: str, due: list) -> None:
        self.log, self.name, self.dues = log, name, deque(due)

    def due(self, core: Any) -> bool:
        return self.dues[0] if len(self.dues) == 1 else self.dues.popleft()

    def take(self, core: Any, drained: cg.Drained) -> None:
        self.log.append((self.name, "take", drained))
        self.dues = deque([False])


class Held(Edge):
    """An edge that waits until ready, its answers following a script whose last answer holds."""

    def __init__(self, log: list, name: str, due: list, ready: list, poll_s: float) -> None:
        super().__init__(log, name, due)
        self.readies, self.poll_s = deque(ready), poll_s

    def ready(self, core: Any) -> bool:
        ok = self.readies[0] if len(self.readies) == 1 else self.readies.popleft()
        self.log.append((self.name, "ready", ok))
        return ok


@pytest.fixture
def seam():
    """A seam over a fresh subclass of `Proc` (as both of vLLM's classes), its clock counting 1, 2, ...; its wraps come off
    afterwards."""
    cls = type("P", (Proc,), {})
    ticks = iter(range(1, 100))
    s = cg.EngineCoreGate(proc=cls, core=cls, clock=lambda: float(next(ticks)))
    s.cls = cls
    try:
        yield s
    finally:
        s.uninstall()
        assert not {"_process_input_queue", "has_work", "step", "step_with_batch_queue"} & set(vars(cls))


def tick_scenario(cls: type, attach: Any) -> list:
    """Four calls of the input loop under one scripted gate: waits that time out and a request that arrives, a gate that
    holds pending requests, an idle wait, and shutdown."""
    log: list = []
    core = cls(log, now=[("abort", "x")], later=[EMPTY, ("add", "r1"), ("add", "r2")])
    attach(core, Gate(log, "tick", ready=[False, False, True, True, True, True, True], pending=[True, False, False]))
    core._process_input_queue()
    core.scheduler.work = False
    core._process_input_queue()
    core._process_input_queue()
    core.running = False
    core._process_input_queue()
    return log


def test_with_one_gate_the_input_loop_makes_the_tick_plugins_calls(seam):
    tick = type("T", (Proc,), {"_process_input_queue": tp._gated_input(Proc._process_input_queue),
                                "has_work": tp._gated_has_work(Proc.has_work)})

    def on_scheduler(core: Any, gate: Gate) -> None:
        core.scheduler._verity_tick = gate

    seam.gate("tick", lambda core: core.scheduler._verity_tick)
    want = tick_scenario(tick, on_scheduler)
    assert tick_scenario(seam.cls, on_scheduler) == want
    assert want[:4] == [("abort", "x"), ("tick", "ready", False), ("get", 0.25), ("tick", "ready", False)]
    assert ("tick", "pending", True) in want and ("get", None) in want and ("aborts-cleared",) in want
    assert want[-1] == ("tick", "open", SHUTDOWN)


def test_every_gate_is_asked_in_its_order_and_the_loop_waits_the_least_poll_of_those_waiting(seam):
    log: list = []
    seam.gate("release", lambda core: Gate(log, "release", ready=[False, False, True], poll_s=0.1))
    seam.gate("tick", lambda core: Gate(log, "tick", ready=[False, True, True], poll_s=0.05))
    core = seam.cls(log, now=[("add", "r0")], later=[EMPTY, EMPTY])
    core._process_input_queue()
    assert log == [("add", "r0"), ("tick", "ready", False), ("release", "ready", False), ("get", 0.05),
                   ("tick", "ready", True), ("release", "ready", False), ("get", 0.1),
                   ("tick", "ready", True), ("release", "ready", True), ("has_work", True)]
    core.running = False
    core._process_input_queue()
    assert log[-2:] == [("tick", "open", SHUTDOWN), ("release", "open", SHUTDOWN)]


def test_a_core_without_gates_or_edges_runs_vllms_own(seam):
    seam.gate("tick", lambda core: None)
    seam.edge("cut", lambda core: None)
    log: list = []
    core = seam.cls(log, batches=[])
    core._process_input_queue()
    assert core.has_work() is False
    core.step_with_batch_queue()
    assert log == [("vllm-input",), ("has_work", False), ("step", True)]


def test_a_due_edge_drains_the_batches_in_flight_then_is_taken_before_the_next_schedule(seam):
    log: list = []
    seam.edge("cut", lambda core: Edge(log, "cut", due=[False, True]))
    core = seam.cls(log, batches=["b2", "b1"])
    core.step_with_batch_queue()
    core.step_with_batch_queue()
    core.step_with_batch_queue()
    assert "has_requests" not in vars(core.scheduler)
    core.step_with_batch_queue()
    core.step_with_batch_queue()
    core.step_with_batch_queue()
    assert log == [("step", True), ("step", False), ("step", False), ("step", False),
                   ("cut", "take", cg.Drained(due_t=1.0, drained_t=2.0, drain_calls=3)), ("step", True), ("step", True)]


def test_without_a_batch_queue_a_due_edge_is_taken_at_once(seam):
    log: list = []
    seam.edge("cut", lambda core: Edge(log, "cut", due=[True]))
    core = seam.cls(log)
    core.step()
    core.step()
    assert log == [("cut", "take", cg.Drained(due_t=1.0, drained_t=2.0, drain_calls=0)), ("step", True), ("step", True)]


def test_a_held_edge_holds_the_input_loop_once_drained_until_ready_and_requests_are_handled_meanwhile(seam):
    log: list = []
    seam.edge("cut", lambda core: Held(log, "cut", due=[True], ready=[False, False, False, True], poll_s=0.5))
    core = seam.cls(log, later=[("abort", "a1"), ("add", "r1")], batches=["b1"])
    core.step_with_batch_queue()
    core.step_with_batch_queue()
    core._process_input_queue()
    core.step_with_batch_queue()
    assert "has_requests" not in vars(core.scheduler)
    assert log == [("step", False),
                   ("cut", "ready", False), ("step", False),
                   ("cut", "ready", False), ("get", 0.5), ("abort", "a1"),
                   ("cut", "ready", False), ("get", 0.5), ("add", "r1"),
                   ("cut", "ready", True), ("has_work", True),
                   ("cut", "ready", True), ("cut", "take", cg.Drained(due_t=1.0, drained_t=2.0, drain_calls=1)),
                   ("step", True)]


def test_a_held_edge_waits_beside_a_gate_and_at_shutdown_is_not_taken(seam):
    log: list = []
    seam.gate("tick", lambda core: Gate(log, "tick", ready=[False, True], poll_s=0.25))
    seam.edge("cut", lambda core: Held(log, "cut", due=[True], ready=[False], poll_s=0.05))
    core = seam.cls(log, later=[EMPTY, ("stop", None)])
    core._process_input_queue()
    core.step()
    assert log == [("tick", "ready", False), ("cut", "ready", False), ("get", 0.05),
                   ("tick", "ready", True), ("cut", "ready", False), ("get", 0.05), ("stop", None),
                   ("tick", "open", SHUTDOWN),
                   ("cut", "ready", False), ("step", False)]


def test_edges_due_at_one_step_are_taken_in_their_order(seam):
    log: list = []
    seam.edge("boundary", lambda core: Edge(log, "boundary", due=[True]))
    seam.edge("cut", lambda core: Edge(log, "cut", due=[True]))
    seam.cls(log, batches=[]).step_with_batch_queue()
    assert [x[0] for x in log] == ["cut", "boundary", "step"]


def test_a_gate_or_edge_is_refused_without_its_place_twice_or_after_a_core_made_its_own(seam):
    with pytest.raises(ValueError, match="GATES or EDGES"):
        seam.gate("cut", lambda core: None)
    seam.gate("tick", lambda core: None)
    with pytest.raises(ValueError, match="already added"):
        seam.gate("tick", lambda core: None)
    core = seam.cls([])
    core._process_input_queue()
    with pytest.raises(RuntimeError, match="before the engine core is built"):
        seam.edge("cut", lambda core: None)


def test_it_wraps_vllms_engine_core_classes_when_given_none(monkeypatch):
    mod = types.ModuleType("vllm.v1.engine.core")
    mod.EngineCoreProc = type("EngineCoreProc", (Proc,), {})
    mod.EngineCore = type("EngineCore", (Proc,), {})
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    s = cg.EngineCoreGate()
    s.edge("cut", lambda core: None)
    try:
        assert {"_process_input_queue", "has_work"} <= set(vars(mod.EngineCoreProc))
        assert {"step", "step_with_batch_queue"} <= set(vars(mod.EngineCore))
    finally:
        s.uninstall()


def test_importing_it_imports_no_vllm(monkeypatch):
    class NoVllm(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
            assert name.split(".")[0] != "vllm", f"engine.core_gate imported {name}"

    for name in [m for m in sys.modules if m.split(".")[0] == "vllm"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NoVllm(), *sys.meta_path])
    importlib.reload(cg)
    assert isinstance(cg.SEAM, cg.EngineCoreGate)
