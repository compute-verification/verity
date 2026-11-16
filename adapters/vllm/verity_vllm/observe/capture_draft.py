"""The capture's draft phase, opt-in (`ObserverConfig.draft`, off by default).  The V2 runner's `sample_tokens` calls the
speculator's `propose` after `sample` returns, outside both: with the draft phase the step is held open after `sample`,
`runner.speculator.propose` runs as the step's phase "draft", and the step closes after it (or at the next `execute_model`
or uninstall when no propose came).  The speculator model's parameters and buffers are roots named under `DRAFTER_PREFIX`
(a tensor it shares with the model stays the model's root) and its KV caches `kv_cache[drafter.<module>]`.  Its events are
buffered in the step like the rest, so they precede the step's `SampleEnd`.  Each function takes the `Capture`, typed
`Any` here because `capture` imports this module."""

from __future__ import annotations

import time
from typing import Any

import torch

from verity_vllm.observe.events import Note

__all__ = ["DRAFTER_PREFIX", "close_awaiting", "drafter_of", "patch_propose"]

#: the drafter's roots and KV caches are named under this prefix
DRAFTER_PREFIX = "drafter."


def drafter_of(capture: Any) -> torch.nn.Module | None:
    """The speculator's model when a recording capture asks for the draft phase; None otherwise, and with a Note when it
    asks and the runner has none."""
    if not (capture.cfg.draft and capture.cfg.mode == "record"):
        return None
    model = getattr(getattr(capture.runner, "speculator", None), "model", None)
    if isinstance(model, torch.nn.Module):
        return model
    capture._write(Note("draft phase asked for, but the runner has no speculator with a model: not observed", {}))
    return None


def patch_propose(capture: Any) -> None:
    """`runner.speculator.propose` observed as phase "draft" of the step held open for it, which closes after it; a propose
    outside a held step (the dummy run's) passes through."""
    speculator = capture.runner.speculator
    orig_propose = speculator.propose

    def propose(*a: Any, **kw: Any) -> Any:
        st = capture._cur
        if st is None or not st.awaiting_draft:
            return orig_propose(*a, **kw)
        t0 = time.perf_counter()
        try:
            return capture._run_observed(lambda: orig_propose(*a, **kw), phase="draft")
        finally:
            st.draft_s = time.perf_counter() - t0
            capture._close_step(st)

    capture._hooks.patch(speculator, "propose", propose)


def close_awaiting(capture: Any) -> None:
    """Close the step held open for a propose that didn't come: at the next `execute_model`, or at uninstall."""
    st = capture._cur
    if st is not None and st.awaiting_draft:
        capture._close_step(st)
