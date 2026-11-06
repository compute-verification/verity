"""The serve-tax arms' changes to vLLM's own OpenAI server, as a vLLM general plugin (`register`, entry point group
``vllm.general_plugins``, run in each of the server's processes): with ``VERITY_SERVETAX`` set (a JSON object: ``out``, the
directory its files go to, ``window_s``, and a true value for each change), every V1 scheduler built in the process gets

- ``record``: the scheduler's step trace (`verity_vllm.engine.step_trace.StepTrace`, the recorder serve-openai uses), its
  records stamped with the wall clock and written by a thread of its own to ``step_trace.PID.jsonl``;
- ``release_order``: `ReleaseGate`, the requests added to the scheduler in the certifier's release order
  (`serve_admit.ReleaseOrder`), their release numbers read from their X-Request-Id (``{run}-{n}``, as `serve_tax`'s client
  numbers them, standing in for the certifier's headers);
- ``tokens``: `TokenCommit`, each window's tokens (every prompt id at its request's arrival and every sampled id) committed
  in ``rows-sha512`` rows on a thread of its own when the window closes, its roots in ``tokens.PID.jsonl``.

The tick's plugin (`verity_vllm.pipeline.tick_plugin`) is a separate entry point, loaded after this one, so its gate wraps
the scheduler outside these.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ENV = "VERITY_SERVETAX"
#: how long the engine core waits on its input queue at a time while the release gate holds requests and nothing runs
POLL_S = 0.01

_PLUGIN: list[Any] = []


def config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    raw = (os.environ if environ is None else environ).get(ENV)
    if not raw:
        return None
    cfg = json.loads(raw)
    if not isinstance(cfg, dict) or not cfg.get("out"):
        raise ValueError(f"{ENV}: a JSON object with out and the changes to make, not {raw[:80]!r}")
    return cfg


class ReleaseGate:
    """The scheduler's ``add_request`` in release order: each request's (run, n) from its engine id (``key``), submitted to
    a `serve_admit.ReleaseOrder` that forwards it to ``add`` once every earlier number of its run has come or passed (an
    abort of a held one, or ``hold_s`` without it). The order's skip timers fire from `poll`, which the engine core calls
    at each schedule and while it waits on its input with requests held."""

    def __init__(self, add: Callable[[Any], Any], abort: Callable[[list[str]], Any], *, hold_s: float = 5.0,
                 key: Callable[[str], tuple[str, int] | None], clock: Callable[[], float] = time.monotonic) -> None:
        from verity_vllm.pipeline.serve_admit import RELEASE_HEADER, RUN_HEADER, ReleaseOrder

        self.headers = (RELEASE_HEADER, RUN_HEADER)
        self.key_of, self.clock = key, clock
        self.timers: list[tuple[float, Callable[[], None]]] = []
        self.order = ReleaseOrder(lambda adm: add(adm.request), lambda engine_id: abort([engine_id]), hold_s=hold_s,
                                  schedule=lambda delay, fn: self.timers.append((self.clock() + delay, fn)))

    def add(self, request: Any) -> None:
        k = self.key_of(request.request_id)
        release = None
        if k is not None:
            release = self.order.key({self.headers[0]: str(k[1]), self.headers[1]: k[0]})
        self.order.submit(_Admission(request, request.request_id, release))

    def finished(self, ids: str | Iterable[str] | None) -> list[str] | None:
        """Of the requests the engine core finishes from outside (aborts), those not held here, for the scheduler."""
        if ids is None:
            return None
        ids = [ids] if isinstance(ids, str) else list(ids)
        held = {adm.engine_id for adm in self.order.held.values()}
        for rid in ids:
            if rid in held:
                self.order.abort(rid)
        return [rid for rid in ids if rid not in held]

    def poll(self) -> None:
        now = self.clock()
        due = [t for t in self.timers if t[0] <= now]
        if due:
            self.timers = [t for t in self.timers if t[0] > now]
            for _, fn in due:
                fn()

    def held(self) -> bool:
        return bool(self.order.held)


class _Admission:
    __slots__ = ("request", "engine_id", "release")

    def __init__(self, request: Any, engine_id: str, release: tuple[str, int] | None) -> None:
        self.request, self.engine_id, self.release = request, engine_id, release


#: vLLM's OpenAI server's engine request id around the client's X-Request-Id: ``cmpl-{X-Request-Id}-{choice}-{8 hex}``
ENGINE_ID = re.compile(r"^(?:cmpl|chatcmpl)-(.+)-\d+-[0-9a-f]{8}$")


def release_key(engine_id: str) -> tuple[str, int] | None:
    """(run, n) from an X-Request-Id ``{run}-{n}`` inside the server's engine id."""
    m = ENGINE_ID.match(engine_id)
    run, _, n = (m.group(1) if m else engine_id).rpartition("-")
    return (run, int(n)) if run and n.isdigit() else None


class TokenCommit:
    """Each window's tokens, 4 bytes each (little-endian u32) in the order the engine core saw them, committed as
    ``rows-sha512`` rows under a fresh salt key (`step_rows.keyed_tensor_root`) on a thread of its own when the window
    closes (`tick`, at the first schedule ``window_s`` after it opened, and at exit). ``sink`` gets each window's record."""

    def __init__(self, window_s: float, sink: Callable[[dict], None], clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s, self.sink, self.clock = float(window_s), sink, clock
        self.window = 0
        self.opened = clock()
        self.tokens: list[int] = []
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._work, name="token-commit", daemon=True)
        self.thread.start()

    def prompt(self, request: Any) -> None:
        self.tokens.extend(request.prompt_token_ids or ())

    def sampled(self, outputs: Any) -> None:
        for eco in (outputs or {}).values():
            for o in getattr(eco, "outputs", None) or ():
                if o.new_token_ids:
                    self.tokens.extend(o.new_token_ids)

    def tick(self) -> None:
        if self.clock() - self.opened >= self.window_s:
            self.close()

    def close(self) -> None:
        tokens, self.tokens = self.tokens, []
        self.jobs.put((self.window, self.opened, self.clock(), tokens))
        self.window += 1
        self.opened = self.clock()

    def _work(self) -> None:
        import array

        from verity_vllm.commit import step_rows

        while (job := self.jobs.get()) is not None:
            window, opened, closed, tokens = job
            t = time.perf_counter()
            data = array.array("I", tokens).tobytes()
            entry = ["tokens", "u32", [len(tokens)], len(data)]
            root = step_rows.keyed_tensor_root(bytes(step_rows.ID_BYTES), window, entry, data, os.urandom(step_rows.KEY_BYTES))
            self.sink({"window": window, "tokens": len(tokens), "bytes": len(data), "rows": -(-len(data) // step_rows.ROW_BYTES),
                       "root": root.hex(), "open_s": opened, "close_s": closed, "hash_s": round(time.perf_counter() - t, 6)})

    def stop(self) -> None:
        if self.tokens:
            self.close()
        self.jobs.put(None)
        self.thread.join()


def _lines(path: Path) -> Callable[[dict], None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a")
    lock = threading.Lock()

    def write(rec: dict) -> None:
        with lock:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
    return write


def install(sched: Any, cfg: Mapping[str, Any]) -> list[Any]:
    """The changes ``cfg`` names on one scheduler, innermost first: the step trace, then the tokens' commit, then the
    release gate (outside both, so that they see a request when the gate adds it)."""
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    out = Path(cfg["out"])
    pid = os.getpid()
    done: list[Any] = []
    if cfg.get("record"):
        trace = StepTrace(out / f"step_trace.{pid}.jsonl", clock=time.time, writer=True).install(sched)
        done.append(trace)
    hooks = Hooks(tag="serve-tax")
    if cfg.get("tokens"):
        tc = TokenCommit(float(cfg.get("window_s", 600)), _lines(out / f"tokens.{pid}.jsonl"))

        def add_request(orig):
            def wrapped(request, *a, **kw):
                tc.prompt(request)
                return orig(request, *a, **kw)
            return wrapped

        def update_from_output(orig):
            def wrapped(so, mro, *a, **kw):
                outs = orig(so, mro, *a, **kw)
                tc.sampled(outs)
                return outs
            return wrapped

        def schedule(orig):
            def wrapped(*a, **kw):
                tc.tick()
                return orig(*a, **kw)
            return wrapped

        hooks.wrap(sched, "add_request", add_request)
        hooks.wrap(sched, "update_from_output", update_from_output)
        hooks.wrap(sched, "schedule", schedule)
        atexit.register(tc.stop)
        done.append(tc)
    if cfg.get("release_order"):
        add, finish = sched.add_request, sched.finish_requests
        gate = ReleaseGate(add, lambda ids: finish(ids, _aborted()), key=release_key, hold_s=float(cfg.get("hold_s", 5.0)))

        def gated_add(orig):
            return lambda request, *a, **kw: gate.add(request)

        def gated_schedule(orig):
            def wrapped(*a, **kw):
                gate.poll()
                return orig(*a, **kw)
            return wrapped

        def gated_finish(orig):
            return lambda request_ids, finished_status: orig(gate.finished(request_ids), finished_status)

        hooks.wrap(sched, "add_request", gated_add)
        hooks.wrap(sched, "schedule", gated_schedule)
        hooks.wrap(sched, "finish_requests", gated_finish)
        hooks.patch(sched, "_verity_release", gate)
        done.append(gate)
    done.append(hooks)
    return done


def _aborted() -> Any:
    from vllm.v1.request import RequestStatus

    return RequestStatus.FINISHED_ABORTED


def _init(init: Callable[..., None], cfg: Mapping[str, Any]) -> Callable[..., None]:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        self._verity_servetax = install(self, cfg)
    return __init__


def _input(orig: Callable[[Any], None]) -> Callable[[Any], None]:
    """`EngineCoreProc._process_input_queue` while the release gate holds requests and nothing runs: wait on the input
    queue ``POLL_S`` at a time, firing the gate's skip timers between, instead of blocking until the next request."""
    def _process_input_queue(self: Any) -> None:
        gate = getattr(self.scheduler, "_verity_release", None)
        while gate is not None and gate.held() and not self.has_work() and self.is_running():
            try:
                self._handle_client_request(*self.input_queue.get(timeout=POLL_S))
            except queue.Empty:
                pass
            gate.poll()
        return orig(self)
    return _process_input_queue


def register() -> None:
    """Install `config`'s changes on every V1 scheduler this process builds. Idempotent."""
    cfg = config()
    if cfg is None or _PLUGIN:
        return
    from verity_vllm.engine.hooks import Hooks
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCoreProc

    hooks = Hooks(tag="serve-tax")
    hooks.wrap(Scheduler, "__init__", lambda init: _init(init, cfg))
    if cfg.get("release_order"):
        hooks.wrap(EngineCoreProc, "_process_input_queue", _input)
    _PLUGIN.append(hooks)
