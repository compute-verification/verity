"""Pearl-C's pass boundary under serving: the PoUW option ends its device run's pass when each engine step's forward returns.

`hook_steps` wraps ``execute_model`` on every vLLM ``GPUModelRunner`` class the process has imported (`model_runners`)
through the option's tagged `Hooks`, so plain serving commits one pass per step with no caller ending it, and each pass is
one whole forward: every weight's calls, as `audit.Epoch.start` holds a pass's workload to every committed weight.  A step
is held to what one pass's buffers take (`step_tokens`: `PearlCDeviceRun.capacity` tiles, `CALL_CAPACITY` calls, its
screened rows; 24,960 tokens of Llama-3.1-8B), and one scheduled over it is refused before its forward runs.  With a
transcript directory each pass is retained in its own (`pass_dir`), complete for `verify_run.py` alone.  A caller that
drives forwards outside a runner still ends its own passes (`Option.end_pass`, `PearlCDeviceRun.end_pass`), and one called
right after a step ends nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw.protocol import CALL_RANGE

#: vLLM's GPU model runner modules (V1's and Model Runner V2's), each with its ``GPUModelRunner`` (as `store.binder.RUNNERS`)
MODEL_RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")


def model_runners() -> tuple[type, ...]:
    """The ``GPUModelRunner`` classes of `MODEL_RUNNERS` that this process has imported, without importing any: once an
    engine is built, its runner's class is among them."""
    found = (getattr(sys.modules.get(m), "GPUModelRunner", None) for m in MODEL_RUNNERS)
    return tuple(dict.fromkeys(c for c in found if c is not None))


def step_tokens(run: Any, calls: int) -> int:
    """The most tokens one engine step's forward can carry in one pass of ``run``: the forward calls each weight once on
    the step's rows, so T rows take ⌈T/TILE⌉ tiles of each weight's n/TILE, and T screened rows, per call; a multiple of
    TILE, and 0 when even one tile row doesn't fit (or the weights are more than ``calls``)."""
    tile = run.kernel.run.TILE
    per = sum(wt.n // tile for wt in run.weights.values())
    if not per or len(run.weights) > calls:
        return 0
    return min(tile * (run.capacity // per), tile * (len(run.flag_buf) // len(run.weights) // tile))


def room(run: Any, tiles: int, rows: int, calls: int) -> Any:
    """``run``'s open pass, where a call of ``tiles`` tiles and ``rows`` screened rows fits it (``run.capacity`` tiles,
    ``calls`` calls, ``run.flag_buf``'s rows); ``RuntimeError`` where it doesn't."""
    p = run.cur
    if run.hashing and (p.tiles + tiles > run.capacity or len(p.matmuls) >= calls or p.flag_rows + rows > len(run.flag_buf)):
        raise RuntimeError(f"{run.entry}: the pass is over its capacity ({run.capacity} tiles, {calls} calls, "
                           f"{len(run.flag_buf)} screened rows); end it and begin another")
    return p


def call_index(run: Any, calls: int = 1) -> int:
    """The index of ``run``'s next call, numbered across the whole epoch (the install's salt), never per pass: -h3's row
    seed binds the salt and the index but no pass, so two passes of one salt sharing an index would repeat their seeds and
    tickets.  The pass's own buffers stay positional (``index - run.cur.base``); a pass begun without a base
    (`finish`'s) takes `EpochCalls.epoch_calls`.  ``RuntimeError`` where ``calls`` more calls would reach
    `protocol.CALL_RANGE`, which the verifier refuses."""
    p = run.cur
    if p.base is None:
        p.base = run.epoch_calls()
    index = p.base + len(p.matmuls)
    if index + calls > CALL_RANGE:
        raise RuntimeError(f"{run.entry}: the epoch is at call {index}, and {calls} more would reach 2^32 calls, the range of "
                           "-h3's row seed (`protocol.CALL_RANGE`); end the epoch: reinstall under a new salt")
    run._used = max(run._used, index + calls)
    return index


class EpochCalls:
    """A device run's call numbering across its epoch (`call_index`): where each pass's first call is numbered from."""
    _used = 0                         # one past the highest index `call_index` has given out
    repeat_at: int | None = None      # the next pass's first index, once (`pass_base`'s ``at``)

    def epoch_calls(self) -> int:
        """One past the highest call index any pass of the epoch has used (0 before the first call): where the next pass
        numbers from."""
        cur = self.cur
        if cur is not None and cur.base is not None:
            self._used = max(self._used, cur.base + len(cur.matmuls))
        return self._used

    def pass_base(self, at: int | None = None) -> int:
        """The next pass's first index: `epoch_calls`, or ``at`` (else `repeat_at`, which it clears) for a check that repeats
        an earlier pass's calls to compare their bytes; the two share indices, so they are never credited together
        (`credit.passes_distinct` refuses them)."""
        used, at, self.repeat_at = self.epoch_calls(), self.repeat_at if at is None else at, None
        return used if at is None else at


def pass_dir(run: Any) -> str | None:
    """The next pass's retain directory under ``run.retain_dir`` (``pass-NNNNN``, numbered in the order passes begin), None
    without one."""
    if run.retain_dir is None:
        return None
    run.retained += 1
    return str(run.retain_dir / f"pass-{run.retained - 1:05d}")


def step_ends(run: Any, limit: int | None = None):
    """The wrapper of a model runner's ``execute_model`` that ends ``run``'s pass once the step's forward has returned
    (`PearlCDeviceRun.end_pass` with ``step``).  This is the step's own boundary under sync and async scheduling alike.
    With async scheduling, vLLM executes step s + 1 before it updates the scheduler from step s, so
    ``Scheduler.update_from_output`` would close one pass over two forwards.  The wrappers made from one call share a depth,
    so a call inside another (a dummy run's, a subclass's runner wrapping its base's) ends the pass once.  A forward that
    raises leaves its pass open.  With ``limit``, a step whose scheduler output schedules more tokens is refused
    (``RuntimeError``) before its forward runs."""
    depth = [0]

    def make(orig):
        def execute_model(runner: Any, *a: Any, **kw: Any) -> Any:
            tokens = getattr(a[0] if a else kw.get("scheduler_output"), "total_num_scheduled_tokens", None)
            if not depth[0] and limit is not None and isinstance(tokens, int) and tokens > limit:
                raise RuntimeError(f"{run.entry}: a step of {tokens} tokens is over one pass ({limit} tokens a step, "
                                   f"`pouw_pass.step_tokens`); serve with max_num_batched_tokens at most {limit}")
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


def hook_steps(hooks: Any, run: Any, calls: int, transcript_dir: str | None = None) -> None:
    """End ``run``'s pass at each engine step (`step_ends`) on every runner class `model_runners` finds, through ``hooks``,
    refusing a step over `step_tokens` (with ``calls`` calls a pass, in ``run.details["max_step_tokens"]``; none where the
    run holds no device buffers), retain each
    pass under ``transcript_dir`` when given (`pass_dir`), and record in ``run.details["pass_ends"]`` which path ends its
    passes."""
    limit = step_tokens(run, calls) if hasattr(run, "flag_buf") else None
    runners, make = model_runners(), step_ends(run, limit)
    for cls in runners:
        hooks.wrap(cls, "execute_model", make)
    run.retain_dir = Path(transcript_dir) if transcript_dir else None
    if run.retain_dir is not None and not run.cur.matmuls:
        run.begin()                 # the pass the run began before it had a retain directory, retained from its start
    run.details["max_step_tokens"] = limit
    run.details["pass_ends"] = ("each engine step, at the model runner's execute_model" if runners else
                                "the caller's end_pass (no model runner in this process)")
