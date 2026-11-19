"""`serve-openai`'s entry: the serve (`pipeline.serve_openai`) with each committed window's circuit planned at its close."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity_vllm.pipeline.serve_openai import Options
from verity_vllm.pipeline.serve_openai import main as serve

__all__ = ["Options", "main", "plan_of"]


def plan_of(out: Path, opts: Any, flush: Callable[[], Any]) -> Any:
    """The serve's `close_plan.ClosePlan`, at the window class ``--window-class`` declares (none: each window says so)."""
    from .close_plan import ClosePlan
    from .window_class import WindowClass

    cls = WindowClass(*(int(p) for p in opts.window_class.split(","))) if opts.window_class else None
    return ClosePlan(out, cls, flush=flush)


def main(opts: Options) -> int:
    return serve(opts, plan_of=plan_of)
