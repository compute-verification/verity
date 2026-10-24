"""Pearl-C's pass boundary under serving: the PoUW option ends its device run's pass when each engine step's forward returns.

`hook_steps` wraps ``execute_model`` on every vLLM ``GPUModelRunner`` class the process has imported (`model_runners`)
through the option's tagged `Hooks`, so plain serving commits one pass per step with no caller ending it.  A caller that
drives forwards outside a runner still ends its own passes (`Option.end_pass`, `PearlCDeviceRun.end_pass`), and one
called right after a step ends nothing.
"""

from __future__ import annotations

import sys
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


def hook_steps(hooks: Any, run: Any) -> None:
    """End ``run``'s pass at each engine step (`step_ends`) on every runner class `model_runners` finds, through ``hooks``,
    and record in ``run.details["pass_ends"]`` which path ends its passes."""
    runners, make = model_runners(), step_ends(run)
    for cls in runners:
        hooks.wrap(cls, "execute_model", make)
    run.details["pass_ends"] = ("each engine step, at the model runner's execute_model" if runners else
                                "the caller's end_pass (no model runner in this process)")
