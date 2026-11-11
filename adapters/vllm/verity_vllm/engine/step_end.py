"""The scheduler's one seam at each step's end (`StepSeam`): observers that see each step once the scheduler has taken the
model's output for it.

vLLM's engine core (``EngineCore.step``, and ``EngineCore.step_with_batch_queue`` once a batch completes, at the pin) ends
each step that reached the model with ``Scheduler.update_from_output(scheduler_output, model_runner_output)``, which
returns the engine core outputs by client. ``AsyncScheduler``, the scheduler under async scheduling, inherits it, and a
step with no requests reaches neither.

A plugin (a ``vllm.general_plugins`` entry point, whose ``register`` vLLM runs at the top of ``EngineCore.__init__``) adds
an observer by name, one of `STEPS`, to `SEAM`, this process's seam, with a function that makes it for a scheduler, or
returns None where that scheduler has none (`verity_vllm.engine.makers.Makers`). The first observer added wraps
``Scheduler.update_from_output`` once, through one `verity_vllm.engine.hooks.Hooks`: once vLLM's own call returns, each
observer made for the scheduler sees the step's `StepEnd`, in `STEPS` order, and the outputs return unchanged. An
observer's error propagates out of the step, so an observer that must not stop serving catches its own. On a scheduler
with none the wrap calls vLLM's own. This module imports vLLM only when it wraps.
"""
from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from verity_vllm.engine.hooks import Hooks
from verity_vllm.engine.makers import Makers

__all__ = ["SEAM", "STEPS", "StepEnd", "StepObserver", "StepSeam"]

#: the observers' names, in the order each sees a step's end
STEPS = ("step_trace", "replay")


@dataclass(frozen=True)
class StepEnd:
    """One step's end: the scheduler, what it scheduled, the model's output for it, the engine core outputs the scheduler
    made of them, and how many of this scheduler's step ends came before it."""

    scheduler: Any
    scheduler_output: Any
    model_output: Any
    outputs: Any
    index: int


class StepObserver(Protocol):
    def __call__(self, step: StepEnd, /) -> None:
        """See one step's end."""


class StepSeam:
    """One process's seam: the observers added by name, made per scheduler, and the wrap that runs them. ``scheduler`` is
    the class it wraps (vLLM's ``Scheduler`` unless given)."""

    def __init__(self, *, scheduler: type | None = None) -> None:
        self._cls = scheduler
        self._observers: Makers[StepObserver] = Makers(STEPS, "engine.step_end's STEPS")
        self._steps: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
        self._hooks: Hooks | None = None

    def on_step(self, name: str, make: Callable[[Any], StepObserver | None]) -> None:
        """Add the observer ``name``, one of `STEPS`: ``make(scheduler)`` makes it for a scheduler, or returns None."""
        self._observers.add(name, make)
        self._install()

    def uninstall(self) -> None:
        """Take the wrap off and forget every observer and scheduler."""
        if self._hooks is not None:
            self._hooks.uninstall()
        self._hooks = None
        self._observers.clear()
        self._steps.clear()

    def _install(self) -> None:
        if self._hooks is not None:
            return
        cls = self._cls
        if cls is None:
            from vllm.v1.core.sched.scheduler import Scheduler as cls
        hooks = Hooks(tag="step-end")
        hooks.wrap(cls, "update_from_output", self._update)
        self._hooks = hooks

    def _update(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        def update_from_output(scheduler: Any, scheduler_output: Any, model_runner_output: Any, *args: Any,
                               **kwargs: Any) -> Any:
            outputs = orig(scheduler, scheduler_output, model_runner_output, *args, **kwargs)
            observers = self._observers.made(scheduler)
            if observers:
                index = self._steps.get(scheduler, 0)
                self._steps[scheduler] = index + 1
                step = StepEnd(scheduler, scheduler_output, model_runner_output, outputs, index)
                for observe in observers:
                    observe(step)
            return outputs
        return update_from_output


#: this process's step seam, which each plugin adds its observers to
SEAM = StepSeam()
