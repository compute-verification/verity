"""The engine core's one seam between steps (`EngineCoreGate`): gates that hold its input loop, and edges that act between
two steps once nothing is in flight.

vLLM's engine core process (``EngineCoreProc.run_busy_loop`` at the pin) alternates ``_process_input_queue``, which returns
once ``has_work()``, with one step (``step_fn``: ``EngineCore.step``, or ``EngineCore.step_with_batch_queue`` when batches
overlap). Two kinds of plugin act there:

- a `Gate` holds the input loop: no step runs while a gate isn't ready. Meanwhile the loop keeps handling client requests,
  waiting on the input queue at most the least ``poll_s`` of the gates still waiting, and ``has_work()`` holds while a
  gate has requests pending. At shutdown every gate is opened. With one gate the loop makes the calls
  `verity_vllm.pipeline.tick_plugin`'s does.
- an `Edge` acts between two steps: once it is due, each step drains the oldest batch in flight and schedules nothing, and
  once none is in flight every due edge is taken, in `EDGES` order, before the engine core schedules again. A `HeldEdge`
  also waits, once none is in flight, until it is ready: meanwhile the input loop holds as it does for a gate, handling
  client requests and waiting on the input queue at most its ``poll_s``, and a step reached meanwhile (at shutdown)
  schedules nothing. At shutdown the input loop returns without taking it.

A plugin (a ``vllm.general_plugins`` entry point, whose ``register`` vLLM runs at the top of ``EngineCore.__init__``)
adds each by name to `SEAM`, this process's seam, with a function that makes it for an engine core, or returns None where
that core has none. Each is made once per engine core, at its first input loop or step. The first gate or edge added wraps
``EngineCoreProc._process_input_queue``, ``EngineCoreProc.has_work``, ``EngineCore.step`` and
``EngineCore.step_with_batch_queue`` once, through one `verity_vllm.engine.hooks.Hooks`, so plugins compose in the order
`GATES` and `EDGES` declare rather than in the order their wraps were stacked. On an engine core with neither, each wrapped
method calls vLLM's own. This module imports vLLM only when it wraps.
"""
from __future__ import annotations

import queue
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from verity_vllm.engine.hooks import Hooks, patched

__all__ = ["EDGES", "GATES", "SEAM", "Drained", "Edge", "EngineCoreGate", "Gate", "HeldEdge"]

#: the gates' names, in the order each pass of the input loop asks them
GATES = ("tick", "release")
#: the edges' names, in the order the edges due at one step are taken: a cut before the boundary state, so that the state
#: taken is the one the next window starts from
EDGES = ("cut", "boundary")


class Gate(Protocol):
    """Holds the engine core's input loop (`verity_vllm.pipeline.serve_admit.TickGate` is one)."""

    poll_s: float

    def ready(self, scheduler: Any, /) -> bool:
        """Whether the next step may run. Asked on each pass of the input loop, so it may admit requests."""

    def pending(self) -> bool:
        """Whether the gate holds requests the scheduler hasn't seen yet, which keep the engine core stepping."""

    def open(self, why: str, /) -> None:
        """Let everything through from now on: the engine core is shutting down."""


@dataclass(frozen=True)
class Drained:
    """The drain before an edge: the clock when the first due edge fell due and when nothing was left in flight, and the
    steps that drained."""

    due_t: float
    drained_t: float
    drain_calls: int


class Edge(Protocol):
    """Acts on the engine core between two steps, once nothing is in flight."""

    def due(self, core: Any, /) -> bool:
        """Whether the edge falls before the next step. Once true, it stays true until the edge is taken."""

    def take(self, core: Any, drained: Drained, /) -> None:
        """Act on the drained engine core. It schedules again once every due edge's `take` has returned."""


class HeldEdge(Edge, Protocol):
    """An `Edge` that, once nothing is in flight, waits until it is ready, while the input loop keeps handling client
    requests."""

    poll_s: float

    def ready(self, core: Any, /) -> bool:
        """Whether the due edge may be taken. Asked on each pass of the input loop once nothing is in flight, and before
        its `take`. Once true, it stays true until the edge is taken. No step runs while it is false, so it turns true at a
        deadline the edge keeps (a longest hold after the drain) even when what it waits for never comes."""


@dataclass(eq=False)
class _Core:
    gates: tuple[Gate, ...]
    edges: tuple[Edge, ...]
    held: tuple[HeldEdge, ...]
    since: float | None = None
    drained_t: float | None = None
    calls: int = 0


class EngineCoreGate:
    """One process's seam: the gates and edges added by name, made per engine core, and the wraps that run them. ``proc``
    and ``core`` are the classes it wraps (vLLM's ``EngineCoreProc`` and ``EngineCore`` unless given), and ``clock`` times
    each drain."""

    def __init__(self, *, proc: type | None = None, core: type | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._classes = (proc, core)
        self.clock = clock
        self._makers: dict[str, Callable[[Any], Any]] = {}
        self._cores: weakref.WeakKeyDictionary[Any, _Core] = weakref.WeakKeyDictionary()
        self._hooks: Hooks | None = None

    def gate(self, name: str, make: Callable[[Any], Gate | None]) -> None:
        """Add the gate ``name``, one of `GATES`: ``make(core)`` makes it for an engine core, or returns None."""
        self._add(name, GATES, make)

    def edge(self, name: str, make: Callable[[Any], Edge | None]) -> None:
        """Add the edge ``name``, one of `EDGES`: ``make(core)`` makes it for an engine core, or returns None."""
        self._add(name, EDGES, make)

    def uninstall(self) -> None:
        """Take the wraps off and forget every gate, edge and engine core."""
        if self._hooks is not None:
            self._hooks.uninstall()
        self._hooks = None
        self._makers.clear()
        self._cores.clear()

    def _add(self, name: str, names: tuple[str, ...], make: Callable[[Any], Any]) -> None:
        if name not in names:
            raise ValueError(f"{name!r} is not one of {names}: give it its place in engine.core_gate's GATES or EDGES")
        if name in self._makers:
            raise ValueError(f"{name!r} is already added to this engine core seam")
        if len(self._cores):
            raise RuntimeError(f"{name!r} was added after an engine core in this process made its gates and edges; add "
                               "each from a plugin's register, before the engine core is built")
        self._install()
        self._makers[name] = make

    def _install(self) -> None:
        if self._hooks is not None:
            return
        proc, core = self._classes
        if proc is None or core is None:
            from vllm.v1.engine.core import EngineCore, EngineCoreProc
            proc, core = proc or EngineCoreProc, core or EngineCore
        hooks = Hooks(tag="engine-core-gate")
        hooks.wrap(proc, "_process_input_queue", self._input)
        hooks.wrap(proc, "has_work", self._has_work)
        hooks.wrap(core, "step", self._step)
        hooks.wrap(core, "step_with_batch_queue", self._step)
        self._hooks = hooks

    def _state(self, core: Any) -> _Core:
        st = self._cores.get(core)
        if st is None:
            made = {n: self._makers[n](core) for n in GATES + EDGES if n in self._makers}
            edges = tuple(made[n] for n in EDGES if made.get(n) is not None)
            st = _Core(tuple(made[n] for n in GATES if made.get(n) is not None), edges,
                       tuple(e for e in edges if hasattr(e, "ready")))
            self._cores[core] = st
        return st

    def _due(self, core: Any, st: _Core) -> list[Edge]:
        due = [e for e in st.edges if e.due(core)]
        if due and st.since is None:
            st.since, st.calls = self.clock(), 0
        return due

    def _drained(self, core: Any, st: _Core) -> bool:
        if core.batch_queue:
            return False
        if st.drained_t is None:
            st.drained_t = self.clock()
        return True

    def _waiting(self, core: Any, st: _Core, due: list[Edge]) -> list[HeldEdge]:
        """The due held edges that aren't ready, once nothing is in flight."""
        held = [e for e in st.held if any(e is d for d in due)]
        if not held or not self._drained(core, st):
            return []
        return [e for e in held if not e.ready(core)]

    def _input(self, orig: Callable[[Any], None]) -> Callable[[Any], None]:
        def _process_input_queue(core: Any) -> None:
            st = self._state(core)
            gates = st.gates
            if not gates and not st.held:
                return orig(core)
            while True:
                while not core.input_queue.empty():
                    core._handle_client_request(*core.input_queue.get_nowait())
                if not core.is_running():
                    for g in gates:
                        g.open("the engine core is shutting down")
                    return
                waiting: list[Gate | HeldEdge] = [g for g in gates if not g.ready(core.scheduler)]
                if st.held:
                    waiting += self._waiting(core, st, self._due(core, st))
                if not waiting and core.has_work():
                    return
                if not waiting:
                    core._notify_idle_state_callbacks()
                    if core.input_queue.empty():
                        with core.aborts_queue.mutex:
                            core.aborts_queue.queue.clear()
                try:
                    req = core.input_queue.get(timeout=min(g.poll_s for g in waiting) if waiting else None)
                except queue.Empty:
                    continue
                core._handle_client_request(*req)
        return _process_input_queue

    def _has_work(self, orig: Callable[[Any], bool]) -> Callable[[Any], bool]:
        def has_work(core: Any) -> bool:
            return orig(core) or any(g.pending() for g in self._state(core).gates)
        return has_work

    def _step(self, orig: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def step(core: Any) -> Any:
            st = self._state(core)
            due = self._due(core, st)
            if not due:
                return orig(core)
            if not self._drained(core, st) or self._waiting(core, st, due):
                st.calls += bool(core.batch_queue)
                with patched(core.scheduler, "has_requests", lambda: False):
                    return orig(core)
            since, drained_t = st.since, st.drained_t
            assert since is not None and drained_t is not None, "an edge is due and nothing is in flight: both clocks were read"
            drained = Drained(since, drained_t, st.calls)
            st.since = st.drained_t = None
            for e in due:
                e.take(core, drained)
            return orig(core)
        return step


#: this process's engine core seam, which each plugin adds its gates and edges to
SEAM = EngineCoreGate()
