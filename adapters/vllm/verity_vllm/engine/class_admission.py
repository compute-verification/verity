"""A served engine class's admission: what a window of the class (`profiles.engine_classes`) refuses while it is served.

`Admission(name).install(engine)` wraps the engine's in-process scheduler's ``add_request`` (`handle.scheduler`, through
`engine.hooks`): a request whose prompt is over the class's prompt limit (`profiles.engine_classes.prompt_limit`, its served
`max_model_len`) is refused before the scheduler holds it; a longer prompt within it the class prefills in chunks, each a launch of
the class's request Program.  The scheduler counts the prompt it was handed, so the count is the engine's whatever form the prompt
came in.  `close_window` is a Commit window's close: it is refused while the scheduler holds a request, since the next window
compiles that request after its carried history (`predict.steps`), which reads the closing window's edge ports (`workload.Edge`),
values the served Commit doesn't commit.

A refusal is sticky: vLLM's frontend registers a request before its scheduler sees it, so once one is refused every later admission
and close is refused with the same reason, and the window ends there.
"""
from __future__ import annotations

from typing import Any

from verity_vllm.engine import handle, hooks
from verity_vllm.engine.profiles import engine_classes as EC
from verity_vllm.errors import Refused


class Admission:
    """Engine class ``name``'s admission on one engine; ``refused`` is the first refusal's reason, None while there is none."""

    def __init__(self, name: str) -> None:
        self.name, self.limit = name, EC.prompt_limit(name)
        self.refused: str | None = None
        self.sched: Any = None
        self.hooks = hooks.Hooks(tag=f"engine-class:{name}")

    def _refuse(self, why: str | None = None) -> None:
        self.refused = self.refused or why
        if self.refused is not None:
            raise Refused(self.refused, what=f"engine class {self.name}")

    def install(self, engine: Any) -> Admission:
        self.sched = handle.scheduler(engine)
        me = self

        def add_request(orig):
            def wrapped(request, *a, **kw):
                n = int(getattr(request, "num_prompt_tokens", 0) or len(request.prompt_token_ids or ()))
                me._refuse(f"request {request.request_id}: a prompt of {n} tokens, over the {me.limit} a class-{me.name} window "
                           f"admits (its served max_model_len)" if n > me.limit else None)
                return orig(request, *a, **kw)
            return wrapped

        self.hooks.wrap(self.sched, "add_request", add_request)
        return self

    def close_window(self) -> None:
        """A Commit window closes: refused while the scheduler holds a request (in flight across the close)."""
        held = list(self.sched.requests)
        self._refuse(f"{len(held)} requests in flight at a Commit window's close, {held[0]!r} first: a class-{self.name} window "
                     f"closes with nothing in flight (a carried request's history reads the window's edge ports, which the "
                     f"served Commit doesn't commit)"
                     if held else None)

    def uninstall(self) -> None:
        self.hooks.uninstall()


__all__ = ["Admission"]
