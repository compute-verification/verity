"""Pearl-C's pass boundaries under serving: the PoUW option ends its device run's pass when each engine step's forward
returns, and before any call the open pass can't hold.

`hook_steps` wraps ``execute_model`` on every vLLM ``GPUModelRunner`` class the process has imported (`model_runners`)
through the option's tagged `Hooks`, so plain serving commits one pass per step with no caller ending it.  A step whose
calls overflow one pass's buffers (`PearlCDeviceRun.capacity` tiles, `CALL_CAPACITY` calls, its screened rows; a prefill
step over about 24K tokens of Llama-3.1-8B) is cut between two calls into as many passes as it needs (`room`), each with
its own transcript over its own calls, so no pass overflows and each is audited on its own; with a transcript directory
each pass is retained in its own (`pass_dir`).  A caller that drives forwards outside a runner still ends its own passes (`Option.end_pass`,
`PearlCDeviceRun.end_pass`), and one called right after a step ends nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

#: vLLM's GPU model runner modules (V1's and Model Runner V2's), each with its ``GPUModelRunner`` (as `store.binder.RUNNERS`)
MODEL_RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")


def model_runners() -> tuple[type, ...]:
    """The ``GPUModelRunner`` classes of `MODEL_RUNNERS` that this process has imported, without importing any: once an
    engine is built, its runner's class is among them."""
    found = (getattr(sys.modules.get(m), "GPUModelRunner", None) for m in MODEL_RUNNERS)
    return tuple(dict.fromkeys(c for c in found if c is not None))


def step_ends(run: Any):
    """The wrapper of a model runner's ``execute_model`` that ends ``run``'s pass once the step's forward has returned
    (`PearlCDeviceRun.end_pass` with ``step``).  This is the step's own boundary under sync and async scheduling alike.
    With async scheduling, vLLM executes step s + 1 before it updates the scheduler from step s, so
    ``Scheduler.update_from_output`` would close one pass over two forwards.  The wrappers made from one call share a depth,
    so a call inside another (a dummy run's, a subclass's runner wrapping its base's) ends the pass once.  A forward that
    raises leaves its pass open."""
    depth = [0]

    def make(orig):
        def execute_model(runner: Any, *a: Any, **kw: Any) -> Any:
            depth[0] += 1
            try:
                out = orig(runner, *a, **kw)
            finally:
                depth[0] -= 1
            if not depth[0]:
                run.end_pass(step=True)
            return out
        return execute_model
    return make


def room(run: Any, tiles: int, rows: int, calls: int) -> Any:
    """The pass a call of ``tiles`` tiles and ``rows`` screened rows runs in: ``run``'s open pass, or, when that one holds
    calls and can't take this one (``run.capacity`` tiles, ``calls`` calls, ``run.flag_buf``'s rows) and ``run.split_full``
    is set, the next, ``run.end_pass`` ending it.  Pearl-C under `pouw.install` runs eagerly, so a cut between two calls of
    one step is a cut between two committed calls.  ``RuntimeError`` when the call doesn't fit even so."""
    full = lambda p: p.tiles + tiles > run.capacity or len(p.matmuls) >= calls or p.flag_rows + rows > len(run.flag_buf)
    if run.hashing and run.split_full and run.cur.matmuls and full(run.cur):
        run.end_pass()
    if run.hashing and full(run.cur):
        raise RuntimeError(f"{run.entry}: the pass is over its capacity ({run.capacity} tiles, {calls} calls, "
                           f"{len(run.flag_buf)} screened rows); end it and begin another")
    return run.cur


def pass_dir(run: Any) -> str | None:
    """The next pass's retain directory under ``run.retain_dir`` (``pass-NNNNN``, numbered in the order passes begin), None
    without one."""
    if run.retain_dir is None:
        return None
    run.retained += 1
    return str(run.retain_dir / f"pass-{run.retained - 1:05d}")


def hook_steps(hooks: Any, run: Any, transcript_dir: str | None = None) -> None:
    """End ``run``'s pass at each engine step (`step_ends`) on every runner class `model_runners` finds, through ``hooks``,
    and before a call the open pass can't hold (`room`), retain each pass under ``transcript_dir`` when given (`pass_dir`),
    and record in ``run.details["pass_ends"]`` which path ends its passes."""
    runners, make = model_runners(), step_ends(run)
    for cls in runners:
        hooks.wrap(cls, "execute_model", make)
    run.split_full, run.retain_dir = True, Path(transcript_dir) if transcript_dir else None
    if run.retain_dir is not None and not run.cur.matmuls:
        run.begin()                 # the pass the run began before it had a retain directory, retained from its start
    run.details["pass_ends"] = ("each engine step, at the model runner's execute_model" if runners else
                                "the caller's end_pass (no model runner in this process)") + "; and before a call the pass can't hold"
