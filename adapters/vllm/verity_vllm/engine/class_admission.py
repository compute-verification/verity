"""A served engine class's admission: what a window of the class (`profiles.engine_classes`) refuses while it is served.

`Admission(name).install(engine)` acts at the engine's in-process scheduler's ``add_request`` (`handle.scheduler`), as a part
of the scheduler's calls at their outermost rank (`scheduler_step`): a request whose prompt is over the engine's context
limit (the scheduler's ``max_model_len``: the launch's, or the
length vLLM fitted to the KV pool where the launch left it to vLLM) is refused before the scheduler holds it, since a longer
prompt has no position for its first output; a long prompt within it the class prefills in chunks, each a launch of
the class's request Program.  The scheduler counts the prompt it was handed, so the count is the engine's whatever form the prompt
came in.  `close_window` is a Commit window's close: it is refused while the scheduler holds a request, since the next window
compiles that request after its carried history (`predict.steps`), which reads the closing window's edge ports (`workload.Edge`),
values the served Commit doesn't commit.

A refusal is sticky: vLLM's frontend registers a request before its scheduler sees it, so once one is refused every later admission
and close is refused with the same reason, and the window ends there.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity_vllm.engine import handle, scheduler_step
from verity_vllm.engine.profiles import engine_classes as EC
from verity_vllm.engine.runner_step import Rank
from verity_vllm.errors import Refused


class Admission:
    """Engine class ``name``'s admission on one engine; ``limit`` is the engine's context limit once installed, and ``refused``
    the first refusal's reason, None while there is none."""

    def __init__(self, name: str) -> None:
        EC.of(name)
        self.name, self.limit = name, 0
        self.refused: str | None = None
        self.sched: Any = None
        self._removals: list[scheduler_step.Removal] = []   # one for each install

    def _refuse(self, why: str | None = None) -> None:
        self.refused = self.refused or why
        if self.refused is not None:
            raise Refused(self.refused, what=f"engine class {self.name}")

    def install(self, engine: Any) -> Admission:
        self.sched = handle.scheduler(engine)
        limit = getattr(self.sched, "max_model_len", None)
        if type(limit) is not int or limit <= 0:
            raise Refused(f"the engine's scheduler states no context limit (its max_model_len is {limit!r})",
                          what=f"engine class {self.name}")
        self.limit = limit
        self._removals.append(scheduler_step.add(self.sched, _Admits(self)))
        return self

    def close_window(self) -> None:
        """A Commit window closes: refused while the scheduler holds a request (in flight across the close)."""
        held = list(self.sched.requests)
        self._refuse(f"{len(held)} requests in flight at a Commit window's close, {held[0]!r} first: a class-{self.name} window "
                     f"closes with nothing in flight (a carried request's history reads the window's edge ports, which the "
                     f"served Commit doesn't commit)"
                     if held else None)

    def uninstall(self) -> None:
        scheduler_step.remove(*self._removals)  # refused with the admission still on, wherever it is on
        self._removals.clear()


class _Admits(scheduler_step.Part):
    """An admission's part on one scheduler (`scheduler_step`): the refusal comes before the parts of every later rank and
    before vLLM's ``add_request``."""

    rank = Rank.ADMIT

    def __init__(self, admission: Admission) -> None:
        self.admission = admission

    def around_add_request(self, call: Callable[..., Any], request: Any, *a: Any, **kw: Any) -> Any:
        me = self.admission
        n = int(getattr(request, "num_prompt_tokens", 0) or len(request.prompt_token_ids or ()))
        me._refuse(f"request {request.request_id}: a prompt of {n} tokens, over the {me.limit} a class-{me.name} window "
                   f"admits (its engine's max_model_len)" if n > me.limit else None)
        return call(request, *a, **kw)


__all__ = ["Admission"]
