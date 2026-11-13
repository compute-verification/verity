"""The carried pool's cut at each window edge, under vLLM's engine core: `register`, a vLLM general plugin, adds the edge
``"cut"`` to the engine core's seam (`verity_vllm.engine.core_gate.SEAM`), a `HeldEdge`. Once an edge is due, the seam
drains every step in flight and schedules nothing; then

1. it **holds** until every open request's outputs are released (`Cut.released`, its ``e``), for at most ``hold_s``:
   the seam's input loop asks `Cut.ready` every ``poll_s`` and handles client requests meanwhile, while the engine core's
   output thread and the API server deliver what it emitted;
2. `Cut.take` **cuts**: it preempts every running request in vLLM's own victim order (``running.pop()``, then
   ``Scheduler._preempt_request`` with ``drop_stale_output``, as ``reset_prefix_cache(reset_running_requests=True)`` does
   without resetting the cache), so no block is held and each request continues in the next window as an arrival of its
   prompt and its ``e`` outputs, hitting its own cached blocks;
3. **requeues** every waiting request in release order: the order of the scheduler's request table
   (``Scheduler.requests``, in the order ``add_request`` received them), which under the certified tick (`tick_plugin`,
   whose gate adds requests in release order) is the release order;
4. keeps the prefix cache, and records the edge (`Cut.edges`, and a line of ``cut.PID.jsonl`` under ``log``).

The engine core's next step schedules window w + 1's first step. With ``VERITY_POOL_ROW`` (the keys' token width CTW)
too, the scheduler's blocks are keyed by BlockKey_v1 from its first request, and each cut exports the pool's row and root
(`engine.pool_row.export`) under ``log``'s ``pool_rows/``.

Each edge's record holds its index, the scheduler step it falls before, the drain (its seconds and steps, from the
seam's `Drained`), the wait for ``e`` after it (``wait_s``, over ``waits`` polls), the cut's and the export's time, and
the cut set in preemption order: each request's id, release number, prompt length P, outputs n and ``e``, its computed
tokens and blocks before the cut, and ``a_O``, the outputs it emitted in the window (n less the ``e`` it carried in). A
request cut with n > e (the hold timed out) is listed in ``over``; its continuation carries n outputs, which the network
certifier has not released. A request whose sampling reads its own outputs (frequency or presence penalties,
``min_tokens``, logits processors) is listed in ``reads_outputs``: vLLM's own preemption keeps its outputs as outputs,
so it serves as uncut, but a Definition that reads the continuation as a prompt of P + e tokens doesn't compute the
tokens it samples.

``VERITY_CUT`` is a JSON object: ``every_s`` (an edge every so many seconds of the engine core's clock from its first
step; one that falls while the engine is idle is taken at its next step) or ``every_steps`` (every so many scheduler
steps), ``hold_s`` (30 by default), ``log`` (a directory) and ``feed`` (a file). **Learning e.** With ``feed``, `SentFeed`,
the OpenAI server's ASGI ``--middleware`` (inside `tick_plugin.TickFeed`), appends each streamed completion's count of
output token ids as its bytes leave the server, keyed by its X-Request-Id, and the engine core reads them every step
(`Sent`): e is what the server has sent, an upper bound on what the network certifier in front of it has released, which
takes its place once the certifier posts its released counts in the same records. Without ``feed``, e is the count the
engine emitted, so the hold is the drain alone. Refused at the engine core's first step: a scheduling policy other than
FCFS, a KV connector, and more than one data-parallel engine.

vLLM loads its general plugins in each of its processes, the API server's included, so this module imports only the
stdlib, `verity_vllm.settings`, `verity_vllm.errors`, `serve_admit` and `tick_plugin`, and `register` imports the seam,
and through it vLLM, only once ``VERITY_CUT`` is set.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.errors import Refused
from verity_vllm.pipeline.serve_admit import request_key
from verity_vllm.pipeline.tick_plugin import SAMPLED_PATHS, FeedReader, route_of

__all__ = ["CUT_ENV", "Cut", "Sent", "SentFeed", "cut_config", "make", "register"]

#: `register`'s configuration, a JSON object (`cut_config`)
CUT_ENV = "VERITY_CUT"
#: the pool row's setting, the keys' token width CTW (`engine.pool_row`)
POOL_ENV = "VERITY_POOL_ROW"


def cut_config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """``VERITY_CUT``'s JSON object (in ``environ``, by default the process environment), None where it isn't set: exactly
    one of ``every_s`` (a positive number) and ``every_steps`` (a positive integer), ``hold_s`` (30 by default), ``log``
    and ``feed`` (optional)."""
    cfg = settings.get(CUT_ENV, environ)
    if cfg is None:
        return None
    if not isinstance(cfg, dict) or ("every_s" in cfg) == ("every_steps" in cfg):
        raise ValueError(f"{CUT_ENV}: a JSON object with one of every_s and every_steps, not {json.dumps(cfg)[:80]!r}")
    if unknown := set(cfg) - {"every_s", "every_steps", "hold_s", "log", "feed"}:
        raise ValueError(f"{CUT_ENV}: unknown keys {sorted(unknown)}")
    every_s, every_steps = cfg.get("every_s"), cfg.get("every_steps")
    if every_s is not None and not (type(every_s) in (int, float) and every_s > 0):
        raise ValueError(f"{CUT_ENV}: every_s is a positive number of seconds, not {every_s!r}")
    if every_steps is not None and not (type(every_steps) is int and every_steps > 0):
        raise ValueError(f"{CUT_ENV}: every_steps is a positive integer, not {every_steps!r}")
    hold_s = cfg.get("hold_s", 30)
    if not (type(hold_s) in (int, float) and hold_s >= 0):
        raise ValueError(f"{CUT_ENV}: hold_s is a number of seconds, not {hold_s!r}")
    return {"every_s": None if every_s is None else float(every_s), "every_steps": every_steps, "hold_s": float(hold_s),
            "log": cfg.get("log"), "feed": cfg.get("feed")}


class Sent:
    """Each open response's sent output count, from the records `SentFeed` appends (``pull``, a `tick_plugin.FeedReader`):
    `poll` reads what came since the last call, and calling it on an engine request gives its count, 0 before any."""

    def __init__(self, pull: Callable[[], list[dict[str, Any]]]) -> None:
        self.pull, self.sent = pull, {}

    def poll(self) -> None:
        for rec in self.pull():
            self.sent[str(rec["key"])] = int(rec["sent"])

    def __call__(self, request: Any) -> int:
        return self.sent.get(request_key(request.request_id), 0)


_TOKEN_IDS = re.compile(rb'"token_ids":\s*\[([^\]]*)\]')


class SentFeed:
    """vLLM's OpenAI server's side of learning e (an ASGI ``--middleware``): with ``VERITY_CUT``'s ``feed`` set, a
    completions or chat request with an X-Request-Id has the output token ids of its streamed events counted (the
    ``token_ids`` that ``return_token_ids`` streams) as each body message leaves, and the running count appended to the
    feed, ``{"key": X-Request-Id, "sent": count}``, each time it grows. Without the setting it passes everything."""

    def __init__(self, app: Any) -> None:
        self.app = app
        cfg = cut_config()
        feed = None if cfg is None else cfg.get("feed")
        self.fd = None if not feed else os.open(feed, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Any], send: Callable[[dict[str, Any]], Any]) -> None:
        if self.fd is None or scope.get("type") != "http" or route_of(scope) not in SAMPLED_PATHS:
            return await self.app(scope, receive, send)
        key = next((v.decode("latin-1") for k, v in scope.get("headers") or () if k.lower() == b"x-request-id"), None)
        if not key:
            return await self.app(scope, receive, send)
        sent, tail, fd = 0, b"", self.fd

        async def counted(message: dict[str, Any]) -> None:
            nonlocal sent, tail
            await send(message)
            if message.get("type") != "http.response.body":
                return
            data = tail + message.get("body", b"")
            i = data.rfind(b"\n\n")
            end = 0 if i < 0 else i + 2
            events, tail = data[:end], data[end:]
            n = sum(len(m.group(1).split(b",")) for m in _TOKEN_IDS.finditer(events) if m.group(1).strip())
            if n:
                sent += n
                os.write(fd, (json.dumps({"key": key, "sent": sent}, separators=(",", ":")) + "\n").encode())
        return await self.app(scope, receive, counted)


def _reads_outputs(request: Any) -> bool:
    p = getattr(request, "sampling_params", None)
    if p is None:
        return False
    return bool(getattr(p, "frequency_penalty", 0) or getattr(p, "presence_penalty", 0) or getattr(p, "min_tokens", 0)
                or getattr(p, "logits_processors", None))


def _blocks(sched: Any, request_id: str) -> int:
    try:
        return sum(len(g) for g in sched.kv_cache_manager.get_block_ids(request_id))
    except (AttributeError, KeyError):
        return 0


class Cut:
    """One engine core's edges, an `engine.core_gate.HeldEdge`: `due` before each step; once the seam has drained, `ready`
    on each pass of its input loop, true once every running request's outputs are released or ``hold_s`` after the first
    ask, and true from then until `take`. ``released(request)`` is the request's released output count ``e`` (by default,
    what the engine emitted), and ``poll()``, called by `due` and `ready`, brings it up to date (`Sent`);
    ``export(scheduler, k)`` exports the pool's row at edge k after the cut (by default, none). ``clock`` is the engine
    core's (injected in tests), ``poll_s`` the longest the seam's input loop waits on its queue between two asks, ``log``
    a JSON-lines file."""

    def __init__(self, *, every_s: float | None = None, every_steps: int | None = None, hold_s: float = 30.0,
                 released: Callable[[Any], int] | None = None, poll: Callable[[], None] | None = None,
                 export: Callable[[Any, int], Any] | None = None, clock: Callable[[], float] = time.monotonic,
                 poll_s: float = 0.002, log: Path | None = None) -> None:
        if (every_s is None) == (every_steps is None):
            raise ValueError("an edge every so many seconds or every so many steps, one of the two")
        self.every_s, self.every_steps, self.hold_s = every_s, every_steps, float(hold_s)
        self.released = released or (lambda r: r.num_output_tokens)
        self.poll, self.export, self.clock, self.poll_s = poll, export, clock, poll_s
        self.file = None if log is None else open(log, "a", encoding="utf-8")
        self.numbers = itertools.count()
        self.seq: dict[str, int] = {}
        self.carried: dict[str, int] = {}
        self.edges: list[dict[str, Any]] = []
        self.start: float | None = None
        self.next_due: float = float(every_steps if every_steps is not None else every_s)
        self.due_at: float | None = None
        self.held_at: float | None = None
        self.ready_at: float | None = None
        self.waits = 0

    def _at(self, sched: Any, now: float) -> float:
        return sched.current_step if self.every_steps is not None else now - self.start

    def due(self, core: Any) -> bool:
        if self.poll is not None:
            self.poll()
        now = self.clock()
        if self.start is None:
            self.start = now
        if self.due_at is None and self._at(core.scheduler, now) >= self.next_due:
            self.due_at = now
        return self.due_at is not None

    def ready(self, core: Any) -> bool:
        if self.ready_at is not None:
            return True
        if self.poll is not None:
            self.poll()
        now = self.clock()
        if self.held_at is None:
            self.held_at = now
        if now - self.held_at >= self.hold_s or all(r.num_output_tokens <= self.released(r) for r in core.scheduler.running):
            self.ready_at = now
            return True
        self.waits += 1
        return False

    def take(self, core: Any, drained: Any) -> None:
        sched = core.scheduler
        if busy := [r.request_id for r in sched.running
                    if getattr(r, "num_in_flight_tokens", 0) or getattr(r, "num_output_placeholders", 0)]:
            raise RuntimeError(f"the cut found {len(busy)} requests in flight ({busy[0]} first) after the seam's drain")
        if self.ready_at is None:
            raise RuntimeError("the cut was taken before it was ready: the seam asks a held edge's ready() first")
        wait_s, waits = self.ready_at - self.held_at, self.waits
        self.held_at = self.ready_at = None
        self.waits = 0
        self._cut(sched, drained, wait_s, waits)

    def _cut(self, sched: Any, drained: Any, wait_s: float, waits: int) -> None:
        k = len(self.edges)
        for rid in sched.requests:
            self.seq.setdefault(rid, next(self.numbers))
        cut, over, reads = [], [], []
        t1 = self.clock()
        while sched.running:
            r = sched.running.pop()
            n, e = r.num_output_tokens, self.released(r)
            rid = r.request_id
            cut.append({"id": rid, "seq": self.seq.get(rid), "P": r.num_prompt_tokens, "n": n, "e": e,
                        "computed": r.num_computed_tokens, "blocks": _blocks(sched, rid),
                        "a_O": n - self.carried.get(rid, 0)})
            if n > e:
                over.append(rid)
            if _reads_outputs(r):
                reads.append(rid)
            sched._preempt_request(r, t1, drop_stale_output=True)
            self.carried[rid] = n
        sched.prev_step_scheduled_req_ids.clear()
        waiting = list(sched.waiting)
        sched.waiting.remove_requests(waiting)
        for r in sorted(waiting, key=lambda r: self.seq.get(r.request_id, -1)):
            sched.waiting.add_request(r)
        t2 = self.clock()
        row = self.export(sched, k) if self.export is not None else None
        t3 = self.clock()
        rec = {"edge": k, "step": sched.current_step, "due": self.next_due, "at": self._at(sched, self.due_at),
               "drain_s": drained.drained_t - drained.due_t, "drain_calls": drained.drain_calls,
               "wait_s": wait_s, "waits": waits, "cut_s": t2 - t1, "export_s": t3 - t2, "cut": cut,
               "requeued": [r.request_id for r in sched.waiting],
               "skipped": [r.request_id for r in getattr(sched, "skipped_waiting", ())], "over": over,
               "reads_outputs": reads, "pool_row": row}
        self.edges.append(rec)
        self.due_at = None
        every = self.every_steps if self.every_steps is not None else self.every_s
        while self.next_due <= self._at(sched, t1):
            self.next_due += every
        live = {r.request_id for r in sched.waiting}
        self.carried = {rid: n for rid, n in self.carried.items() if rid in live}
        self.seq = {rid: i for rid, i in self.seq.items() if rid in sched.requests}
        if self.file is not None:
            self.file.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.file.flush()


def make(cfg: Mapping[str, Any]) -> Callable[[Any], Cut]:
    """``cfg``'s (`cut_config`'s) edge maker for the seam: ``make(cfg)(core)`` refuses an engine core it can't cut,
    installs BlockKey_v1 with ``VERITY_POOL_ROW`` set, and returns the core's `Cut`. Refused: a scheduling policy other
    than FCFS, a KV connector, more than one data-parallel engine."""
    log = Path(cfg["log"]) if cfg.get("log") else None
    ctw = settings.get(POOL_ENV)

    def for_core(core: Any) -> Cut:
        sched = core.scheduler
        policy = getattr(getattr(sched, "scheduler_config", None), "policy", None)
        if policy != "fcfs":
            raise RuntimeError(f"{CUT_ENV} needs the FCFS scheduler, whose waiting queue it requeues in release order; "
                               f"this one's policy is {policy!r}")
        if getattr(sched, "connector", None) is not None:
            raise RuntimeError(f"{CUT_ENV} preempts every running request at each edge, which a KV connector's pending "
                               "transfers don't allow")
        dp = getattr(getattr(getattr(sched, "vllm_config", None), "parallel_config", None), "data_parallel_size", 1)
        if dp != 1:
            raise RuntimeError(f"{CUT_ENV} cuts one engine's requests; this one is one of {dp} data-parallel engines")
        export = None
        if ctw:
            from verity_vllm.engine import pool_row as PR
            PR.install_keys(ctw, sched.block_size)
            rows = None if log is None else log / "pool_rows"
            if rows is not None:
                rows.mkdir(parents=True, exist_ok=True)

            def export(s: Any, k: int) -> Any:
                try:
                    return PR.export(s.kv_cache_manager.block_pool, s.block_size,
                                     None if rows is None else rows / f"e{k}.json")
                except Refused as e:
                    return {"refused": str(e)[:500]}
        if log is not None:
            log.mkdir(parents=True, exist_ok=True)
        sent = Sent(FeedReader(cfg["feed"])) if cfg.get("feed") else None
        return Cut(every_s=cfg["every_s"], every_steps=cfg["every_steps"], hold_s=cfg["hold_s"], export=export,
                   released=sent, poll=None if sent is None else sent.poll,
                   log=None if log is None else log / f"cut.{os.getpid()}.jsonl")
    return for_core


_REGISTERED: list[Any] = []


def register() -> None:
    """The cut at each window edge, as a vLLM general plugin (entry point group ``vllm.general_plugins``, run in each of
    its processes): with ``VERITY_CUT`` set (`cut_config`), this process's engine core seam takes the edge ``"cut"``
    (`make`). Idempotent."""
    cfg = cut_config()
    if cfg is None or _REGISTERED:
        return
    from verity_vllm.engine.core_gate import SEAM
    SEAM.edge("cut", make(cfg))
    _REGISTERED.append(SEAM)
