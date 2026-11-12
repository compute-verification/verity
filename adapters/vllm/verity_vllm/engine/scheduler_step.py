"""One wrap of each method of vLLM's in-process scheduler that a part acts at, and the parts that run around it.

A module that acts when a request is added, a step is scheduled, a step's output is processed or a request is finished
from outside does not patch the scheduler: it subclasses `Part`, overrides the ``around_`` methods it needs and calls
`add(scheduler, part)`.

**Which methods are patched.**  `METHODS`' four, each through `hooks` on the scheduler object, and each only while a
part that overrides its ``around_`` is on: a method's patch goes on with the first such part and comes off with the last.
A method no part acts at stays the scheduler's own, so a patch that another component puts on the scheduler's class
later is the one the scheduler calls (the store's binder wraps ``schedule`` and ``update_from_output`` there, and a
protocol installed again on a built engine installs it after the engine class's admission is on).

**The order parts run in.**  By rank (`runner_step.Rank`), nested: the ``around_`` of each part that overrides it,
ADMIT outermost, parts of one rank in the order they were added.  Each is handed ``call`` and the call's arguments;
``call`` runs the parts inside it and then vLLM's method, and what the ``around_`` returns is what its caller gets.  A
part may hand ``call`` other arguments, call it later, or not call it (an admission that refuses raises instead).

**Among other components' patches.**  A part joins its method's patch where that patch is.  A part added after another
component wrapped the same method on the same scheduler therefore runs inside that component's wrapper, where a wrap
made at that moment would run outside it.

**Parts come and go on a built engine.**  `add` at any time, removal in any order.  A call's parts are the ones that
were on when it began: a part added during a call joins at the next one, and a part removed during a call is not
entered after its removal.  Removing a part takes off the patches of the methods no other part acts at, which `hooks`
refuses while a patch of the same method that another component installed later is still on: the removal then raises,
with the part still on and nothing changed.  `remove` runs several removals as one, all of them or none.
"""
from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from typing import Any

from verity_vllm.engine import hooks
from verity_vllm.engine.runner_step import Rank

__all__ = ["METHODS", "Part", "Removal", "add", "parts", "remove"]

#: the tag of the patches (`hooks.live`)
TAG = "scheduler-step"
#: the scheduler's methods a part may act at, in the order their patches go on
METHODS = ("add_request", "schedule", "update_from_output", "finish_requests")


class Part:
    """What one module does at the scheduler's calls.  A subclass sets ``rank`` and overrides the methods it needs; each
    calls ``call`` with what it was handed by default."""

    rank: Rank

    def around_add_request(self, call: Callable[..., Any], request: Any, *a: Any, **kw: Any) -> Any:
        """Around vLLM's ``add_request(request)``."""
        return call(request, *a, **kw)

    def around_schedule(self, call: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        """Around vLLM's ``schedule()``, which returns the step's scheduler output."""
        return call(*a, **kw)

    def around_update_from_output(self, call: Callable[..., Any], scheduler_output: Any, model_runner_output: Any,
                                  *a: Any, **kw: Any) -> Any:
        """Around vLLM's ``update_from_output(scheduler_output, model_runner_output)``."""
        return call(scheduler_output, model_runner_output, *a, **kw)

    def around_finish_requests(self, call: Callable[..., Any], request_ids: Any, finished_status: Any, *a: Any, **kw: Any) -> Any:
        """Around vLLM's ``finish_requests(request_ids, finished_status)``: a finish from outside the scheduler."""
        return call(request_ids, finished_status, *a, **kw)


def _acts_at(part: Part, method: str) -> bool:
    """Whether ``part``'s class overrides the ``around_`` of ``method``."""
    return getattr(type(part), f"around_{method}") is not getattr(Part, f"around_{method}")


class _Wraps:
    """One scheduler's patches, at most one a method, and the parts on it."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.added: list[tuple[Rank, int, Part]] = []
        self.on: tuple[Part, ...] = ()
        self.patches: dict[str, hooks.Hooks] = {}

    def wrap(self, name: str) -> None:
        """Patch the scheduler's ``name``: the parts that act at it, then vLLM's method."""
        around = f"around_{name}"

        def make(vllm: Callable[..., Any]) -> Callable[..., Any]:
            def method(*a: Any, **kw: Any) -> Any:
                began = tuple(p for p in self.on if _acts_at(p, name))

                def run(i: int, *a: Any, **kw: Any) -> Any:
                    while i < len(began) and not any(began[i] is p for p in self.on):
                        i += 1
                    if i == len(began):
                        return vllm(*a, **kw)
                    return getattr(began[i], around)(lambda *a, **kw: run(i + 1, *a, **kw), *a, **kw)

                return run(0, *a, **kw)

            return method

        made = hooks.Hooks(tag=TAG)
        made.wrap(self.scheduler, name, make)
        self.patches[name] = made


def _covered(patch: Any) -> bool:
    """Whether a live patch of ``patch``'s attribute was installed after it."""
    later = itertools.dropwhile(lambda e: e is not patch, hooks.live())
    return any(e is not patch and isinstance(e, hooks.Patch) and e.owner is patch.owner and e.name == patch.name for e in later)


_WRAPS: dict[int, _Wraps] = {}
_ORDER = itertools.count()
_LOCK = threading.Lock()


class Removal:
    """One part's removal from one scheduler, as `add` returns it: calling it is `remove` of it alone."""

    def __init__(self, wraps: _Wraps, entry: tuple[Rank, int, Part]) -> None:
        self.wraps, self.entry = wraps, entry

    def __call__(self) -> None:
        remove(self)


def add(scheduler: Any, part: Part) -> Removal:
    """Put ``part`` on ``scheduler`` from the next call of each method it acts at, and return the part's removal.  The
    removal is idempotent; it raises `RuntimeError`, with the part still on, when a patch it would take off can't come
    off.  A scheduler that lacks a method the part acts at is refused (`AttributeError`) with no patch left on."""
    if not isinstance(getattr(part, "rank", None), Rank):
        raise TypeError(f"{type(part).__qualname__} sets no rank: a part's rank is one of {', '.join(r.name for r in Rank)}")
    entry = (part.rank, next(_ORDER), part)
    with _LOCK:
        wraps = _WRAPS.get(id(scheduler))
        if wraps is not None and any(part is p for p in wraps.on):
            raise ValueError(f"{type(part).__qualname__} is already on this scheduler")
        if wraps is None:
            wraps = _Wraps(scheduler)
        new = [name for name in METHODS if name not in wraps.patches and _acts_at(part, name)]
        try:
            for name in new:
                wraps.wrap(name)
        except BaseException:
            for name in reversed(new):
                if name in wraps.patches:
                    wraps.patches.pop(name).uninstall()
            raise
        _WRAPS[id(scheduler)] = wraps
        wraps.added.append(entry)
        wraps.on = tuple(p for _, _, p in sorted(wraps.added, key=lambda e: e[:2]))
    return Removal(wraps, entry)


def remove(*removals: Removal) -> None:
    """Run ``removals`` as one: every part of theirs that is still on comes off, or none does.  They are refused
    (`RuntimeError`, from `hooks`) when a patch that would come off once all of them are gone can't.  A module with
    more than one part on, on one scheduler or on several, takes them off through this, so a refusal leaves it whole."""
    with _LOCK:
        going = [r for r in removals if any(r.entry is e for e in r.wraps.added)]       # one that has run is skipped
        plan = []
        for wraps in {id(r.wraps): r.wraps for r in going}.values():
            left = [e for e in wraps.added if not any(e is r.entry for r in going)]
            plan.append((wraps, left, [name for name in wraps.patches if not any(_acts_at(p, name) for _, _, p in left)]))
        for wraps, _, off in plan:              # a covered one first, while every patch is on: `hooks` refuses it in its words
            for name in off:
                if any(_covered(patch) for patch in wraps.patches[name].entries):
                    wraps.patches[name].uninstall()
        for wraps, left, off in plan:
            for name in off:
                wraps.patches.pop(name).uninstall()
            wraps.added = left
            wraps.on = tuple(p for _, _, p in sorted(left, key=lambda e: e[:2]))
            if not left:
                del _WRAPS[id(wraps.scheduler)]


def parts(scheduler: Any) -> tuple[Part, ...]:
    """The parts that are on ``scheduler``, in the order they run."""
    wraps = _WRAPS.get(id(scheduler))
    return wraps.on if wraps is not None else ()
