"""serve-openai's engine loop (`EngineLoop`): the one thread that steps the engine, the process's main thread, between the
HTTP side's admissions and the Commit's windows (`verity_vllm.pipeline.serve_openai`)."""
from __future__ import annotations

import collections
import queue
import time
from collections.abc import Callable
from typing import Any

from verity_vllm.engine.served_seed import SeedCoins, served
from verity_vllm.pipeline.serve_admit import Admission, Delta, TickGate
from verity_vllm.pipeline.serve_commit import CommitWindows, ingress_of


class EngineLoop:
    """The one thread that steps the engine: admissions come through `submit`, outputs go back through each admission's
    ``deliver``. ``engine`` is vLLM's ``LLMEngine`` (``add_request``, ``abort_request``, ``step``,
    ``has_unfinished_requests``); ``commit`` a `CommitWindows` or None. With ``cap`` (``--window-requests``), at most ``cap``
    requests are in the engine at once: later ones wait in ``held``, in admission order, and go in as running ones finish.
    With ``gate`` (``--admit tick``, a `serve_admit.TickGate` installed on ``sched``, the engine's scheduler), its records
    come through `tick`, in queue order with the admissions, and the loop steps only once the gate is ready.

    Each request goes into the engine with the seed `engine.served_seed.served` gives it: a stochastic one that sent none is
    seeded from the current seed coin of ``coins`` (the Commit's, else its own) and its ``user``, a fresh coin when it comes
    into an engine holding nothing with no window open (`SeedCoins.admitting`)."""

    _STOP = object()

    def __init__(self, engine: Any, commit: CommitWindows | None = None, cap: int = 0,
                 clock: Callable[[], float] = time.time, gate: TickGate | None = None, sched: Any = None,
                 coins: SeedCoins | None = None) -> None:
        self.engine, self.commit, self.cap, self.clock, self.gate, self.sched = engine, commit, int(cap), clock, gate, sched
        own = getattr(commit, "coins", None)
        if own is not None and coins is not None and coins is not own:
            raise ValueError("the engine loop seeds from its Commit's seed coins, whose windows record them: pass them to the Commit")
        self.coins = own or coins or SeedCoins()
        self.q: queue.Queue = queue.Queue()
        self.live: dict[str, Admission] = {}
        self.held: collections.deque[Admission] = collections.deque()
        self.error: str | None = None
        self.draining: str | None = None
        self.steps = 0

    def submit(self, adm: Admission) -> None:
        adm.times.setdefault("submitted", self.clock())
        self.q.put(adm)

    def abort(self, engine_id: str) -> None:
        self.q.put(("abort", engine_id))

    def stop(self) -> None:
        self.q.put(self._STOP)

    def drain(self, why: str) -> None:
        """Stop once the requests in flight have finished; every request admitted after this is refused with ``why``."""
        self.q.put(("drain", why))

    def tick(self, rec: dict[str, Any]) -> None:
        self.q.put(("tick", rec))

    def _take(self, item: Any) -> bool:
        """Apply one queued item; False for the stop."""
        if item is self._STOP:
            return False
        if isinstance(item, tuple) and item[0] == "tick":
            self.gate.take(item[1])
            return True
        if isinstance(item, tuple) and item[0] == "drain":
            self.draining = self.draining or str(item[1])
            if self.gate is not None:
                self.gate.open(f"draining: {self.draining}")
            return True
        if isinstance(item, tuple):
            eid = item[1]
            if self.live.pop(eid, None) is not None:
                self.engine.abort_request([eid])
                if self.commit is not None:
                    self.commit.aborted(eid, "client")
                self._release()
            else:
                self.held = collections.deque(a for a in self.held if a.engine_id != eid)
            return True
        if self.draining is not None:
            item.deliver(Delta([], "", error=f"the endpoint is stopping: {self.draining}"))
            return True
        if self.cap and (self.held or len(self.live) >= self.cap):
            self.held.append(item)
        else:
            self._add(item)
        return True

    def _add(self, item: Admission) -> None:
        try:
            idle = not self.engine.has_unfinished_requests() and not (self.commit is not None and self.commit.is_open)
            item.params, rule = served(item.params, item.user, self.coins.admitting(idle))
            item.ingress = ingress_of(item.brought, item.params, rule)
            self.engine.add_request(request_id=item.engine_id, prompt={"prompt_token_ids": list(item.prompt_ids)}, params=item.params)
        except Exception as e:  # noqa: BLE001  (the engine's refusal of one request goes back to it)
            item.deliver(Delta([], "", error=f"{type(e).__name__}: {e}"))
            return
        item.times["admitted"] = self.clock()
        self.live[item.engine_id] = item
        if self.commit is not None:
            self.commit.admitted(item)

    def _release(self) -> None:
        """Held requests into the engine, in admission order, while it has room under ``cap``; those held before a drain
        are still served."""
        while self.held and len(self.live) < self.cap:
            self._add(self.held.popleft())

    def _deliver(self, out: Any) -> None:
        adm = self.live.get(out.request_id)
        if adm is None:
            return
        for c in out.outputs[:1]:
            d = Delta(list(c.token_ids), c.text or "", c.finish_reason, c.stop_reason,
                      list(out.prompt_token_ids) if out.prompt_token_ids is not None else None)
            if self.commit is not None:
                self.commit.streamed(adm.engine_id, d.token_ids, d.finish_reason, d.stop_reason)
            adm.deliver(d)
        if out.finished:
            self.live.pop(out.request_id, None)

    def run(self) -> None:
        running = True
        try:
            while running:
                idle = not self.engine.has_unfinished_requests()
                if idle and self.commit is not None:
                    self.commit.idle()
                if idle and self.draining is not None:
                    break
                item = self.q.get() if idle else self._poll()
                while item is not None:
                    if not self._take(item):
                        running = False
                        break
                    item = self._poll()
                if not running or not self.engine.has_unfinished_requests():
                    continue
                if self.gate is not None and not self.gate.ready(self.sched):
                    try:
                        running = self._take(self.q.get(timeout=self.gate.poll_s))
                    except queue.Empty:
                        pass
                    continue
                if self.commit is not None:
                    self.commit.before_step(self.live)
                for out in self.engine.step():
                    self._deliver(out)
                if self.commit is not None:
                    self.commit.stepped()
                self.steps += 1
                self._release()
        except BaseException as e:
            self.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            for adm in list(self.live.values()):
                adm.deliver(Delta([], "", error=self.error or "the server stopped"))
                if self.commit is not None:
                    self.commit.aborted(adm.engine_id, "server")
            self.live.clear()
            for adm in self.held:
                adm.deliver(Delta([], "", error=self.error or "the server stopped"))
            self.held.clear()
            if self.commit is not None and self.commit.is_open:
                self.commit.close()

    def _poll(self) -> Any:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None
