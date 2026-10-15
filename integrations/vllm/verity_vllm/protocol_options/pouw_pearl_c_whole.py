"""Pearl-C's decode forward replayed as one CUDA graph: vLLM's own (Model Runner V2's FULL graphs of the decode batch),
captured with every call's graph steps (`pouw_pearl_c_graphs`) inside it, so a decode step launches one graph instead of one
per call, and runs no call's Python.

vLLM captures one set of graphs at start, over what its linears ran then (its own apply).  `WholeStep` keeps a set per key:
`VLLM` for the start's, used while no Pearl-C arm executes, and `GraphCalls.whole_key` (hashing, omitted steps, schedule,
retained) for each Pearl-C pass kind, which `prepare` captures, the arm's passes and with hashing its retained passes, through
vLLM's ``capture_model`` inside a scratch pass the first time the arm is active.  vLLM's eager decode isn't its FULL graph's
bit for bit, so a retained pass replays the graph its timed pass did.  On the runner's graph manager it patches, on the
instance:

* ``dispatch``: the key's set as the manager's ``graphs``.  A step runs eagerly, with the descriptor vLLM gives a batch
  without graphs, when the key has no set (no pass open, `enabled` off, graphs off, a schedule the graph path doesn't run)
  or its set lacks the batch;
* ``run_fullgraph``: before vLLM's launch of the replay, `GraphCalls.whole_replay` with the calls the capture recorded,
  every call's slot in one upload; after it, while the GPU runs the forward, `GraphCalls.whole_replayed`, the rest of the
  forward's host work: the pass's records, a retained pass's files, and the next forward's slots staged.

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
    those of them whose slots were staged after the step before, steps run eagerly, and graphs captured, since the last
    `take`.  With ``enabled`` off the arm's forwards run eagerly in vLLM, each call as its own graph."""

    enabled = True

    def __init__(self, run: Any, stack: Any, runner: Any, hooks: Any) -> None:
        mgr = runner.cudagraph_manager
        if mgr is None or not mgr.graphs:
            raise ValueError("vLLM captured no CUDA graphs at start: whole-step replay needs its FULL decode graphs "
                             "(cudagraph_mode FULL_DECODE_ONLY, enforce_eager off)")
        bad = sorted({d.num_tokens for d in mgr.graphs if -(-d.num_tokens // ROW_ALIGN) * ROW_ALIGN > GRAPH_ROWS})
        if bad:
            raise ValueError(f"vLLM's graphs of {bad} tokens: a forward is captured whole only when its calls run as graphs "
                             f"(at most {GRAPH_ROWS} rows, filler rows to a multiple of {ROW_ALIGN} included)")
        self.run, self.stack, self.runner, self.mgr = run, stack, runner, mgr
        self.sets: dict[tuple, dict] = {VLLM: mgr.graphs}
        self.calls: dict[tuple, tuple] = {}
        self.counts = {"replayed": 0, "staged": 0, "eager": 0, "captured": 0}
        hooks.wrap(mgr, "dispatch", self._dispatch)
        hooks.wrap(mgr, "run_fullgraph", self._run_fullgraph)

    def key(self) -> tuple | None:
        if not self.stack.executing:
            return VLLM
        if not self.enabled or self.stack.open_pass is None:
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
            self.counts["replayed"] += 1
            if key == VLLM:
                return original(desc)
            calls = self.calls[(key, desc)]
            self.counts["staged"] += self.run.whole_replay(calls)
            out = original(desc)
            self.run.whole_replayed(calls)
            return out
        return run_fullgraph

    def prepare(self) -> bool:
        """The active arm's graphs, its passes' and its retained passes', each captured if its key has none yet, inside a
        scratch pass that is then dropped from the stack's and the run's records.  Returns whether it captured."""
        if not self.stack.executing or not self.enabled:
            return False
        with self.stack.pass_():
            keys = [k for k in dict.fromkeys(self.run.whole_key(r) for r in (False, True)) if k is not None and k not in self.sets]
            for key in keys:
                self._capture(key)
        self.stack.passes.pop()
        self.run.details["passes"].pop()
        return bool(keys)

    def _capture(self, key: tuple) -> None:
        mgr, run = self.mgr, self.run

        def stored(desc: Any) -> None:
            self.calls[(key, desc)] = run.whole_capture_end()
            run.whole_capture_begin(key)

        graphs, held = _Set(stored), mgr.graphs
        mgr.graphs = graphs
        run.whole_capture_begin(key)
        try:
            self.runner.capture_model()
        finally:
            run.whole_capture_end()
            mgr.graphs = held
        self.sets[key] = graphs
        self.counts["captured"] += len(graphs)
