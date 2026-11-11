"""The model runner's one seam around each forward pass (`ForwardSeam`): observers that see each pass begin and end.

vLLM's model runner (``GPUModelRunner`` of ``vllm.v1.worker.gpu.model_runner``, the default at the pin, and of
``vllm.v1.worker.gpu_model_runner``) runs one pass as ``execute_model(scheduler_output)``, which either returns the pass's
output itself (a step that runs no forward, or the intermediate tensors of a pipeline stage that isn't the last) or
returns None once it has left the sample's inputs in ``execute_model_state``, and then ``sample_tokens(grammar_output)``,
which returns the output. ``execute_model(..., dummy_run=True)``, a warm-up or capture run, is no pass.

A plugin (a ``vllm.general_plugins`` entry point, whose ``register`` runs in each worker process) adds an observer by name,
one of `FORWARDS`, to `SEAM`, this process's seam, with a function that makes it for a runner, or returns None where that
runner has none (`verity_vllm.engine.makers.Makers`). The first observer added wraps both methods of each runner class
once, through one `verity_vllm.engine.hooks.Hooks`, as each runner module is imported (`hooks.at_import`), so nothing is
imported for it. Each pass's observers see ``begin(runner, scheduler_output)`` in `FORWARDS` order before vLLM's
``execute_model``, and ``end(ForwardEnd)`` in the reverse order once the pass's output is returned: from
``execute_model`` itself, else from the ``sample_tokens`` that follows. A pass that raises ends with its error, which then
propagates, and a pass whose ``sample_tokens`` never came ends, with an error saying so, when the runner's next
``execute_model`` starts. An observer's own error propagates out of the pass. On a runner with none, each wrapped method
calls vLLM's own.
"""
from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from verity_vllm.engine.hooks import Hooks
from verity_vllm.engine.makers import Makers

__all__ = ["FORWARDS", "RUNNERS", "SEAM", "Forward", "ForwardEnd", "ForwardSeam"]

#: the observers' names, in the order each sees a pass begin (and the reverse, the order each sees it end)
FORWARDS = ("capture", "route", "replay")
#: the modules whose ``GPUModelRunner`` the seam wraps: vLLM's model runner, and its older one
RUNNERS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")


@dataclass(frozen=True)
class ForwardEnd:
    """One pass's end: the runner and what it was scheduled, its ``execute_model_state`` as ``execute_model`` left it
    (None where ``execute_model`` returned the output itself), its output (None when it raised), its error, and how many of
    this runner's passes came before it."""

    runner: Any
    scheduler_output: Any
    state: Any
    output: Any
    error: BaseException | None
    index: int


class Forward(Protocol):
    def begin(self, runner: Any, scheduler_output: Any, /) -> None:
        """See a pass begin, before vLLM's ``execute_model``."""

    def end(self, done: ForwardEnd, /) -> None:
        """See the pass end."""


@dataclass(eq=False)
class _Runner:
    observers: tuple[Forward, ...]
    passes: int = 0
    open: tuple[Any, Any] | None = None


class ForwardSeam:
    """One process's seam: the observers added by name, made per runner, and the wraps that run them. ``runners`` are the
    classes it wraps (each `RUNNERS` module's ``GPUModelRunner`` unless given)."""

    def __init__(self, *, runners: tuple[type, ...] | None = None) -> None:
        self._classes = runners
        self._observers: Makers[Forward] = Makers(FORWARDS, "engine.forward_pass's FORWARDS")
        self._runners: weakref.WeakKeyDictionary[Any, _Runner] = weakref.WeakKeyDictionary()
        self._hooks: Hooks | None = None

    def on_forward(self, name: str, make: Callable[[Any], Forward | None]) -> None:
        """Add the observer ``name``, one of `FORWARDS`: ``make(runner)`` makes it for a runner, or returns None."""
        self._observers.add(name, make)
        self._install()

    def uninstall(self) -> None:
        """Take the wraps off (and disarm the ones still waiting for their module), and forget every observer and runner."""
        if self._hooks is not None:
            self._hooks.uninstall()
        self._hooks = None
        self._observers.clear()
        self._runners.clear()

    def _install(self) -> None:
        if self._hooks is not None:
            return
        hooks = self._hooks = Hooks(tag="forward-pass")
        if self._classes is not None:
            for cls in self._classes:
                self._wrap(hooks, cls)
            return
        for site in RUNNERS:
            hooks.at_import(site, lambda module: self._wrap(hooks, module.GPUModelRunner))

    def _wrap(self, hooks: Hooks, cls: type) -> None:
        hooks.wrap(cls, "execute_model", self._execute)
        hooks.wrap(cls, "sample_tokens", self._sample)

    def _state(self, runner: Any) -> _Runner:
        st = self._runners.get(runner)
        if st is None:
            st = self._runners[runner] = _Runner(self._observers.made(runner))
        return st

    def _end(self, runner: Any, st: _Runner, output: Any, error: BaseException | None) -> None:
        if st.open is None:
            return
        (scheduler_output, state), st.open = st.open, None
        done = ForwardEnd(runner, scheduler_output, state, output, error, st.passes)
        st.passes += 1
        for o in reversed(st.observers):
            o.end(done)

    def _execute(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        def execute_model(runner: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            st = self._state(runner)
            if not st.observers or kwargs.get("dummy_run"):
                return orig(runner, scheduler_output, *args, **kwargs)
            if st.open is not None:
                self._end(runner, st, None, RuntimeError("the runner's next execute_model came before this pass's "
                                                         "sample_tokens"))
            for o in st.observers:
                o.begin(runner, scheduler_output)
            st.open = (scheduler_output, None)
            try:
                out = orig(runner, scheduler_output, *args, **kwargs)
            except BaseException as e:
                self._end(runner, st, None, e)
                raise
            state = getattr(runner, "execute_model_state", None)
            if out is None and state is not None:
                st.open = (scheduler_output, state)
            else:
                self._end(runner, st, out, None)
            return out
        return execute_model

    def _sample(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        def sample_tokens(runner: Any, *args: Any, **kwargs: Any) -> Any:
            st = self._state(runner)
            if st.open is None:
                return orig(runner, *args, **kwargs)
            try:
                out = orig(runner, *args, **kwargs)
            except BaseException as e:
                self._end(runner, st, None, e)
                raise
            self._end(runner, st, out, None)
            return out
        return sample_tokens


#: this process's forward seam, which each plugin adds its observers to
SEAM = ForwardSeam()
