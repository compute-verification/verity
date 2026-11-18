"""The adapter's admission and release hook: vLLM's admission, scheduling and output points, patched once, as the events
of the ``steps`` hook (`verity.hooks`) to this process's exchange (`verity_vllm.messages`), and the schemes' requests on
starting inputs and releasing outputs back. It names no scheme.

`install(scheduler)` puts one `Hook` on a V1 scheduler, a part of its calls at the outermost rank (`scheduler_step`,
``Rank.ADMIT``), so every later part (a step trace's) sees a request from the moment the hook adds it. The hook is also
the engine core's gate (`engine.core_gate.Gate`): `register` adds it to the engine core process's seam (`core_gate.SEAM`,
gate ``admission``) for every engine core that process builds, and serve-openai's engine loop, which steps an in-process
engine itself, asks its `Hook.ready` before each step.

An input is a request, by its key (``key`` reads it from vLLM's request id), and an output a token. What each vLLM point
emits, and what it asks:

- ``add_request``: `InputArrived`, with the request's count of prompt ids; the request goes into the scheduler once every
  gate a scheme filed (`StartOnBoundary`) starts it (`Exchange.start`), and one still held when it is finished from
  outside never reaches the scheduler, which is never handed its id;
- before each step, on each pass of the engine core's input loop: every gate is asked (`Exchange.ready`, with the
  scheduler's running, waiting and finished requests as `StepDue`) and holds the step while one says no;
- ``schedule()``: `StepBegan` for a step that scheduled a token, with its requests, its tokens less draft tokens and the
  requests left waiting;
- ``update_from_output``: `OutputReleased` for each new token, refused while a scheme holds its request's release at that
  step (`HoldRelease`), `OutputsComplete` for each request that finished, then `StepEnded`;
- ``finish_requests``: `OutputsComplete` for each request it ends; where vLLM names none, every gate is opened first.
"""
from __future__ import annotations

import collections
import re
import time
from collections.abc import Callable
from typing import Any

from verity import hooks as H
from verity_vllm import messages
from verity_vllm.engine import scheduler_step
from verity_vllm.engine.runner_step import Rank

__all__ = ["POLL_S", "Hook", "WindowGroups", "install", "non_draft_tokens", "register", "request_key"]

#: seconds between the engine's looks at the gates while one holds the step
POLL_S = 0.002
_ENGINE_ID = re.compile(r"^(?:cmpl|chatcmpl)-(.+)-\d+-[0-9a-f]{8}$")


def request_key(request_id: str) -> str:
    """The client's X-Request-Id inside vLLM's OpenAI server's engine request id (``cmpl-{X-Request-Id}-0-{8 hex}``), else
    the id unchanged."""
    m = _ENGINE_ID.match(request_id)
    return m.group(1) if m else request_id


def non_draft_tokens(out: Any) -> tuple[int, int]:
    """A scheduler output's tokens less its draft tokens, and its draft tokens: the sum of its rows' n as the replay's
    step forms it, the batch's token limit less the budget it leaves (`replay_step.reference`), which models no draft."""
    drafts = sum(len(d) for d in (getattr(out, "scheduled_spec_decode_tokens", None) or {}).values())
    return out.total_num_scheduled_tokens - drafts, drafts


def _waiting(sched: Any) -> int:
    return len(sched.waiting) + len(getattr(sched, "skipped_waiting", None) or ())


class Hook(scheduler_step.Part):
    """One scheduler's admission and release hook (the module docstring), and its engine core's gate."""

    rank = Rank.ADMIT
    poll_s = POLL_S

    def __init__(self, sched: Any, exchange: messages.Exchange, key: Callable[[str], str] = request_key) -> None:
        self.sched, self.ex, self.key = sched, exchange, key
        self.step = 0                                       # the next step that schedules a token
        self.began: collections.deque[int] = collections.deque()    # steps scheduled whose outputs haven't come
        self.held: dict[str, str] = {}                      # vLLM's request id -> its input, until a gate starts it
        self.live: dict[str, str] = {}                      # vLLM's request id -> its input, in the scheduler
        self.sent: dict[str, int] = {}                      # vLLM's request id -> its tokens released so far
        self._removal: scheduler_step.Removal | None = None

    def around_add_request(self, call: Callable[..., Any], request: Any, *a: Any, **kw: Any) -> Any:
        rid = request.request_id
        input_ = self.held[rid] = self.key(rid)
        if self.ex.wants("steps"):
            n = int(getattr(request, "num_prompt_tokens", 0) or len(getattr(request, "prompt_token_ids", None) or ()))
            self.ex.emit(H.InputArrived(input_, n, self.step))

        def start() -> None:
            if self.held.pop(rid, None) is not None:
                self.live[rid] = input_
                call(request, *a, **kw)

        self.ex.start(input_, start)

    def around_schedule(self, call: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        out = call(*a, **kw)
        if out.total_num_scheduled_tokens > 0:
            tokens, _ = non_draft_tokens(out)
            inputs = tuple(self.key(r) for r in getattr(out, "num_scheduled_tokens", None) or ())
            self.ex.emit(H.StepBegan(self.step, inputs, tokens, _waiting(self.sched)))
            self.began.append(self.step)
            self.step += 1
        return out

    def around_update_from_output(self, call: Callable[..., Any], scheduler_output: Any, model_runner_output: Any,
                                  *a: Any, **kw: Any) -> Any:
        step = self.began.popleft() if self.began else self.step - 1
        outs = call(scheduler_output, model_runner_output, *a, **kw)
        wants = self.ex.wants("steps")
        for batch in (outs or {}).values():
            for o in getattr(batch, "outputs", None) or ():
                rid, k = o.request_id, len(o.new_token_ids or ())
                input_ = self.live.get(rid) or self.key(rid)
                if k and self.ex.held(input_, step):
                    raise NotImplementedError(f"vllm/adapter-names-no-scheme: a scheme holds input {input_}'s release "
                                              f"at step {step}, and the adapter keeps no released token back yet")
                index = self.sent.get(rid, 0)
                self.sent[rid] = index + k
                if wants:
                    for j in range(k):
                        self.ex.emit(H.OutputReleased(input_, index + j, step))
                if o.finish_reason is not None:
                    self.live.pop(rid, None)
                    self.sent.pop(rid, None)
                    if wants:
                        self.ex.emit(H.OutputsComplete(input_, step))
        if wants:
            self.ex.emit(H.StepEnded(step))
        return outs

    def around_finish_requests(self, call: Callable[..., Any], request_ids: Any, finished_status: Any,
                               *a: Any, **kw: Any) -> Any:
        if request_ids is None:
            self.open("every request finished")
            ended = [*self.held.values(), *self.live.values()]
            self.held.clear()
            self.live.clear()
            self.sent.clear()
            rest = None
        else:
            rest, ended = [], []
            for rid in (request_ids,) if isinstance(request_ids, str) else request_ids:
                self.sent.pop(rid, None)
                if rid in self.held:
                    ended.append(self.held.pop(rid))
                    continue
                if rid in self.live:
                    ended.append(self.live.pop(rid))
                rest.append(rid)
        if self.ex.wants("steps"):
            for input_ in ended:
                self.ex.emit(H.OutputsComplete(input_, self.step))
        return call(rest, finished_status, *a, **kw)

    # --- the engine core's gate (`core_gate.Gate`) ----------------------------------------------------------------------

    def ready(self, scheduler: Any = None, /) -> bool:
        """Whether the next step may run: every gate is asked (`Exchange.ready`), and may start requests."""
        s, waiting = self.sched, _waiting(self.sched)
        return self.ex.ready(H.StepDue(self.step, len(s.running) + waiting, waiting, len(s.finished_req_ids)))

    def pending(self) -> bool:
        return self.ex.pending()

    def open(self, why: str, /) -> None:
        self.ex.open(why)

    def uninstall(self) -> None:
        if self._removal is not None:
            self._removal()
            self._removal = None


def install(sched: Any, exchange: messages.Exchange | None = None, key: Callable[[str], str] = request_key) -> Hook:
    """Put the admission and release hook on ``sched`` (a vLLM V1 scheduler) for ``exchange`` (this process's by default)
    until it is uninstalled. Where a scheme filed a gate, refused for a scheduler policy other than FCFS, whose waiting
    queue is the order the gates start requests in, and for one of several data-parallel engines."""
    ex = exchange or messages.current()
    if any(isinstance(r, H.StartOnBoundary) for _, r in ex.requests):
        policy = getattr(getattr(sched, "scheduler_config", None), "policy", None)
        if policy != "fcfs":
            raise RuntimeError(f"a gate on starting inputs needs the FCFS scheduler, whose waiting queue is the order it "
                               f"starts them in; this one's policy is {policy!r}")
        dp = getattr(getattr(getattr(sched, "vllm_config", None), "parallel_config", None), "data_parallel_size", 1)
        if dp != 1:
            raise RuntimeError(f"a gate on starting inputs orders one engine's admissions; this one is one of {dp} "
                               f"data-parallel engines")
    hook = Hook(sched, ex, key)
    hook._removal = scheduler_step.add(sched, hook)
    return hook


def register(attach: Callable[[messages.Exchange], Any], key: Callable[[str], str] = request_key) -> None:
    """In an engine core process (a vLLM general plugin's ``register``): for each engine core, ``attach`` files its
    schemes' requests on this process's exchange and subscribes them, and the hook goes on the core's scheduler, held
    through the core's input loop (`core_gate.SEAM`'s gate ``admission``)."""
    from verity_vllm.engine.core_gate import SEAM

    def make(core: Any) -> Hook:
        ex = messages.current()
        attach(ex)
        return install(core.scheduler, ex, key)

    SEAM.gate("admission", make)


class WindowGroups:
    """A gate (`hooks.Gate`, filed with `hooks.StartOnBoundary`) that starts each group of inputs on a window's boundary,
    a window being the steps between two moments the workload holds nothing: the inputs that arrive within ``gather_s``
    of the first into a workload holding nothing are one group, and the step waits for them, so all start at the
    window's first step; one that arrives later waits until the workload holds nothing again, so the current window is
    waited out and never cut short."""

    def __init__(self, gather_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        if not gather_s > 0:
            raise ValueError(f"a group gathers for a positive time, not {gather_s!r} s")
        self.gather_s, self.clock = float(gather_s), clock
        self.until: float | None = None
        self.idle = True                                    # the workload held nothing at the last step's ask
        self.later: dict[str, Callable[[], None]] = {}      # input -> its start, in arrival order

    def _start_later(self) -> None:
        later, self.later = self.later, {}
        self.idle = self.idle and not later
        for start in later.values():
            start()

    def add(self, input: str, start: Callable[[], None]) -> None:
        if self.until is None and self.idle and not self.later:
            self.until = self.clock() + self.gather_s
        if self.until is not None and self.clock() < self.until:
            self.idle = False
            start()
        else:
            self.later[input] = start

    def pending(self) -> bool:
        return bool(self.later) or self.until is not None

    def ready(self, due: H.StepDue) -> bool:
        if self.until is not None and self.clock() < self.until:
            return False
        self.until = None
        self.idle = not (due.active or due.completing)
        if self.later and self.idle:
            self.until = self.clock() + self.gather_s
            self._start_later()
            return False
        return True

    def open(self, why: str) -> None:
        self.until = None
        self._start_later()
