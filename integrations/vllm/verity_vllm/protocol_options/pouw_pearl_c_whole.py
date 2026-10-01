"""Pearl-C's decode forward replayed as one CUDA graph: vLLM's own (Model Runner V2's FULL graphs of the decode batch),
captured with every call's graph steps (`pouw_pearl_c_graphs`) inside it, so a decode step launches one graph instead of one
per call, and runs no call's Python.

vLLM captures one set of graphs at start, over what its linears ran then (its own apply).  `WholeStep` keeps a set per key:
`VLLM` for the start's, used while no Pearl-C arm executes, and `GraphCalls.whole_key` (hashing, omitted steps, schedule) for
each Pearl-C pass kind, which `prepare` captures through vLLM's ``capture_model`` inside a scratch pass the first time the arm
is active.  On the runner's graph manager it patches, on the instance:

* ``dispatch``: the key's set as the manager's ``graphs``.  A step runs eagerly, with the descriptor vLLM gives a batch
  without graphs, when the key has no set (no pass open, a retained pass, graphs off, a schedule the graph path doesn't run)
  or its set lacks the batch;
* ``run_fullgraph``: before vLLM's replay, `GraphCalls.whole_replay` with the calls the capture recorded, which does the
  forward's host work: every call's slot in one upload, and the pass's records.

The calls inside a capture are those made while vLLM's stream captures; vLLM's warm-up forward just before, outside the
capture, makes their per-call graphs.  A captured forward ends where vLLM stores its graph (``graphs[desc] = graph``), which
`_Set` observes.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .pouw_pearl_c_device import ROW_ALIGN
from .pouw_pearl_c_graphs import GRAPH_ROWS

#: the key of vLLM's own graphs, captured at start
VLLM = ("vllm",)


class _Set(dict):
    """One key's graphs by vLLM's batch descriptor; storing one calls `stored` with its descriptor."""

    def __init__(self, stored: Callable[[Any], None]) -> None:
        super().__init__()
        self._stored = stored

    def __setitem__(self, desc: Any, graph: Any) -> None:
        super().__setitem__(desc, graph)
        self._stored(desc)


class WholeStep:
    """`run`'s decode forwards as vLLM's graphs, on `runner` (Model Runner V2's), through `hooks`.  ``counts``: steps replayed,
    steps run eagerly, and graphs captured, since the last `take`."""

    def __init__(self, run: Any, stack: Any, runner: Any, hooks: Any) -> None:
        mgr = runner.cudagraph_manager
        if mgr is None or not mgr.graphs:
            raise ValueError("vLLM captured no CUDA graphs at start: whole-step replay needs its FULL decode graphs "
                             "(cudagraph_mode FULL_DECODE_ONLY, enforce_eager off)")
        bad = sorted({d.num_tokens for d in mgr.graphs if d.num_tokens > GRAPH_ROWS or d.num_tokens % ROW_ALIGN})
        if bad:
            raise ValueError(f"vLLM's graphs of {bad} tokens: a forward is captured whole only when its calls run as graphs "
                             f"(at most {GRAPH_ROWS} rows) without filler rows (a multiple of {ROW_ALIGN})")
        self.run, self.stack, self.runner, self.mgr = run, stack, runner, mgr
        self.sets: dict[tuple, dict] = {VLLM: mgr.graphs}
        self.calls: dict[tuple, tuple] = {}
        self.counts = {"replayed": 0, "eager": 0, "captured": 0}
        hooks.wrap(mgr, "dispatch", self._dispatch)
        hooks.wrap(mgr, "run_fullgraph", self._run_fullgraph)

    def key(self) -> tuple | None:
        if not self.stack.executing:
            return VLLM
        if self.stack.open_pass is None:
            return None
        return self.run.whole_key()

    def take(self) -> dict[str, int]:
        out, self.counts = self.counts, dict.fromkeys(self.counts, 0)
        return out

    def _dispatch(self, original: Callable[..., Any]) -> Callable[..., Any]:
        mgr = self.mgr

        def dispatch(*a: Any, **kw: Any) -> Any:
            graphs = self.sets.get(self.key())
            if graphs:
                mgr.graphs = graphs
                desc = original(*a, **kw)
                if desc in graphs:
                    return desc
            captured, mgr._graphs_captured = mgr._graphs_captured, False
            try:
                desc = original(*a, **kw)
            finally:
                mgr._graphs_captured = captured
            self.counts["eager"] += 1
            return desc
        return dispatch

    def _run_fullgraph(self, original: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def run_fullgraph(desc: Any) -> Any:
            key = self.key()
            if key != VLLM:
                self.run.whole_replay(self.calls[(key, desc)])
            self.counts["replayed"] += 1
            return original(desc)
        return run_fullgraph

    def prepare(self) -> bool:
        """The active arm's graphs, captured if its key has none yet, inside a scratch pass that is then dropped from the
        stack's and the run's records.  Returns whether it captured."""
        if not self.stack.executing:
            return False
        with self.stack.pass_():
            key = self.key()
            fresh = key is not None and key not in self.sets
            if fresh:
                self._capture(key)
        self.stack.passes.pop()
        self.run.details["passes"].pop()
        return fresh

    def _capture(self, key: tuple) -> None:
        mgr, run = self.mgr, self.run

        def stored(desc: Any) -> None:
            self.calls[(key, desc)] = run.whole_capture_end()
            run.whole_capture_begin()

        graphs, held = _Set(stored), mgr.graphs
        mgr.graphs = graphs
        run.whole_capture_begin()
        try:
            self.runner.capture_model()
        finally:
            run.whole_capture_end()
            mgr.graphs = held
        self.sets[key] = graphs
        self.counts["captured"] += len(graphs)
