"""One wrap of the model runner's ``execute_model`` and ``sample``, and the parts that run around them.

A module that acts at an engine step does not patch the runner: it subclasses `Part`, overrides the methods it needs and
calls `add(runner, part)`.  The two patches go on the runner object, through `hooks`, when its first part is added, and
come off when its last part is removed; a runner with no ``sample`` has ``execute_model`` wrapped alone.

**Which calls are steps.**  ``execute_model`` with ``dummy_run`` true (the keyword, or the third argument) or with no
scheduled token calls vLLM's method and no part.  Any other call opens a `Step`, which stays open until vLLM's
``sample`` returns for it or the next step opens.  ``sample`` while no step is open calls vLLM's method and no part.

**The order parts run in.**  By rank, nested; parts of one rank in the order they were added::

    execute_model   before_forward of each part, ADMIT first
                    enter each part's around_forward, ADMIT first
                        vLLM's execute_model
                    leave them, last entered first
                    after_forward of each part, PASS first
    sample          enter each part's around_sample, ADMIT first
                        vLLM's sample
                    leave them, last entered first
                    after_sample of each part, PASS first

Every ``around`` is entered after every ``before`` and left before any ``after``, so nothing of a part runs inside
another part's ``around`` except vLLM's call.  An ``after`` does not run when vLLM's call raised: the error goes to
vLLM's caller and the step stays open.  A part's own error goes to vLLM's caller too, and the parts after it are not
called.

**Parts come and go on a built engine.**  `add` at any time, removal in any order.  A step's parts are the ones that
were on when its ``execute_model`` began: a part added during a step joins at the next one, and a part removed during a
step is not called again.  Removing the last part takes the two patches off first, which `hooks` refuses while a patch
of the same method that another component installed later is still on: the removal then raises, with the part still on
and nothing changed.
"""
from __future__ import annotations

import enum
import itertools
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import Any

from verity_vllm.engine import hooks

#: the tag of the two patches (`hooks.live`)
TAG = "runner-step"


class Rank(enum.IntEnum):
    """Where a part runs among the others: a lower rank is further out, so its ``before`` runs earlier and its ``after``
    later."""

    ADMIT = 0
    PLAN = 1
    COMMIT = 2
    OBSERVE = 3
    PASS = 4


class Part:
    """What one module does at an engine step.  A subclass sets ``rank`` and overrides the methods it needs; each does
    nothing by default.  An ``around`` returns a context manager that is entered before vLLM's call and left after it, or
    None; its exit does not swallow an error."""

    rank: Rank

    def before_forward(self, step: Step) -> None:
        """Before vLLM's ``execute_model``, and before any part's ``around_forward`` is entered."""

    def around_forward(self, step: Step) -> AbstractContextManager[None] | None:
        """The context vLLM's ``execute_model`` runs in."""
        return None

    def after_forward(self, step: Step) -> None:
        """After vLLM's ``execute_model`` returned (``step.forwarded``) and every ``around_forward`` was left."""

    def around_sample(self, step: Step) -> AbstractContextManager[None] | None:
        """The context vLLM's ``sample`` runs in."""
        return None

    def after_sample(self, step: Step) -> None:
        """After vLLM's ``sample`` returned (``step.sampled``) and every ``around_sample`` was left."""


@dataclass(eq=False)
class Step:
    """One engine step: a call of ``execute_model`` that runs the model, and the ``sample`` that follows it."""

    runner: Any
    #: ``execute_model``'s first argument
    scheduler_output: Any
    #: the parts that were on when ``execute_model`` began, in the order they run
    parts: tuple[Part, ...]
    #: what ``execute_model`` returned; set before ``after_forward``
    forwarded: Any = None
    #: ``sample``'s three arguments; set before ``around_sample``
    hidden_states: Any = None
    input_batch: Any = None
    grammar_output: Any = None
    #: what ``sample`` returned; set before ``after_sample``
    sampled: Any = None


class _Wraps:
    """One runner's two patches, the parts on it and its open step."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.added: list[tuple[Rank, int, Part]] = []
        self.on: tuple[Part, ...] = ()
        self.step: Step | None = None
        self.hooks = hooks.Hooks(tag=TAG)
        self.hooks.wrap(runner, "execute_model", self._execute_model)
        if hasattr(runner, "sample"):
            self.hooks.wrap(runner, "sample", self._sample)

    def _each(self, step: Step, last_first: bool = False) -> Iterator[Part]:
        """The step's parts that are still on, looked up again before each one."""
        for part in reversed(step.parts) if last_first else step.parts:
            if any(part is p for p in self.on):
                yield part

    def _inside(self, step: Step, around: str, vllm: Callable[[], Any]) -> Any:
        with ExitStack() as stack:
            for part in self._each(step):
                context = getattr(part, around)(step)
                if context is not None:
                    stack.enter_context(context)
            return vllm()
        raise RuntimeError(f"a part's {around} swallowed the error of vLLM's call; an around's exit must let it through")

    def _execute_model(self, vllm: Callable[..., Any]) -> Callable[..., Any]:
        def execute_model(scheduler_output: Any, *a: Any, **kw: Any) -> Any:
            dummy = bool(kw.get("dummy_run", False)) or (len(a) >= 2 and bool(a[1]))
            if dummy or getattr(scheduler_output, "total_num_scheduled_tokens", 0) == 0:
                return vllm(scheduler_output, *a, **kw)
            step = self.step = Step(self.runner, scheduler_output, self.on)
            for part in self._each(step):
                part.before_forward(step)
            step.forwarded = self._inside(step, "around_forward", lambda: vllm(scheduler_output, *a, **kw))
            for part in self._each(step, last_first=True):
                part.after_forward(step)
            return step.forwarded

        return execute_model

    def _sample(self, vllm: Callable[..., Any]) -> Callable[..., Any]:
        def sample(*a: Any, **kw: Any) -> Any:
            step = self.step
            if step is None:
                return vllm(*a, **kw)
            step.hidden_states, step.input_batch, step.grammar_output = (
                a[i] if len(a) > i else kw.get(name) for i, name in enumerate(("hidden_states", "input_batch", "grammar_output")))
            step.sampled = self._inside(step, "around_sample", lambda: vllm(*a, **kw))
            self.step = None
            for part in self._each(step, last_first=True):
                part.after_sample(step)
            return step.sampled

        return sample


_WRAPS: dict[int, _Wraps] = {}
_ORDER = itertools.count()
_LOCK = threading.Lock()


def add(runner: Any, part: Part) -> Callable[[], None]:
    """Put ``part`` on ``runner`` from its next step on, and return the part's removal.  The removal is idempotent; it
    raises `RuntimeError`, with the part still on, when ``part`` is the runner's last and the two patches can't come off."""
    if not isinstance(getattr(part, "rank", None), Rank):
        raise TypeError(f"{type(part).__qualname__} sets no rank: a part's rank is one of {', '.join(r.name for r in Rank)}")
    entry = (part.rank, next(_ORDER), part)
    with _LOCK:
        wraps = _WRAPS.get(id(runner))
        if wraps is not None and any(part is p for p in wraps.on):
            raise ValueError(f"{type(part).__qualname__} is already on this runner")
        if wraps is None:
            wraps = _WRAPS[id(runner)] = _Wraps(runner)
        wraps.added.append(entry)
        wraps.on = tuple(p for _, _, p in sorted(wraps.added, key=lambda e: e[:2]))

    def remove() -> None:
        with _LOCK:
            if not any(entry is e for e in wraps.added):
                return
            if len(wraps.added) == 1:
                wraps.hooks.uninstall()
                del _WRAPS[id(runner)]
            wraps.added = [e for e in wraps.added if e is not entry]
            wraps.on = tuple(p for p in wraps.on if p is not part)

    return remove


def parts(runner: Any) -> tuple[Part, ...]:
    """The parts that are on ``runner``, in the order they run."""
    wraps = _WRAPS.get(id(runner))
    return wraps.on if wraps is not None else ()
