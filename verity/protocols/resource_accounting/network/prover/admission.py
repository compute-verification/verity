"""The network attester's admission on the served node: its release order (`ReleaseOrder`) and the certified tick
(`TickGate`), the readers of its release headers and its watermark (`release_of`, `Watermark`), and the
frontends that feed them: a front that hands its engine each input (`TickAdmit`) and a front in another process than
its engine (`FrontFeed`). The adapter hands in what is its own: its submit and abort of an input, the configuration it
read, and the exchange of `verity.hooks` events and requests the gate and the frontend attach to, both built and
attached by one entry (`Admitter`). Nothing here names a serving stack.

`ReleaseOrder` is a reorder buffer in front of the adapter's ``submit``: it forwards request n of the release order
once the frontend holds every earlier-released request, each forwarded, refused or dropped, and holds it until then. There
is no hold to a bucket boundary (the boundary saves no step bits).

**What the network attester sends.** Each request it releases carries ``x-verity-release: N``, N its 0-based index in the release
order of its run (the network attester's ``Proxy.requests_released`` before the release counts it): dense, each N once, every
released request numbered, a request a fault drops before its release unnumbered. With ``x-verity-release-run: ID`` (the
network attester's run's id, hex) the numbering is per run: a new ID starts a new order at 0, and what the old run still held is
forwarded at once. The network attester copies both onto the POST it sends the endpoint (`benchmarks/network_traces/
network_attester_serve.py serve --watermark`). The network attester also sends
``x-verity-release-bucket: A``, the release's bucket counted from 0 at the run's origin, and the watermark: one
``POST /verity/watermark`` a run, naming it, whose chunked body has the line ``t n`` (`watermark_line`) once bucket t has
closed, the releases of buckets up to t being exactly those numbered below n, followed on that line by each release of
bucket t that its server connection lost on the way; its end ends the run. A frontend takes one such stream a run, and
none that carries ``x-verity-release``, which the network attester puts on every request it releases and never on its
own stream (`open_watermark`): any other POST to that route is answered 400, and feeds the gate nothing.

What a request without the header gets: it is forwarded at once (``unordered``), as the served plan's probes are. A
malformed or repeated ordinal is refused (`ValueError`, the server's 400). A released request that never reaches the
frontend, or whose body doesn't arrive, would hold every later one: once a later one has been held ``hold_s`` behind it,
it is skipped and the held ones go on; if it comes after all it is forwarded at once (``late``). Both are counted
(`summary`), and so are the runs.

**The certified tick** (``tick``, `TickGate`). Step k has a logical time T_k, an integer of microseconds, and before it the
gate adds exactly the requests whose release bucket a has closed by then, (a + 1)·τ_b ≤ T_k, in release order. After a step
that computed a slot, T advances by τ_sat if a started input is left waiting for a slot (`hooks.StepBegan.queued`) and by
τ_free if none is; when nothing is
running or waiting, T jumps to the close of the next release's bucket, so T_0 is the first release's. τ_b is the network
attester's bucket and τ_sat, τ_free the engine class's (`Tick`), registered with the class: a configuration, not a
run-time choice. A release's bucket is the watermark's (the header is only checked against it), and the gate starts step k
once it holds the watermark's line for the last bucket closed by T_k and every request numbered below that line's n but
the ones a line names lost. The rest of each step is the serving stack's own scheduler, unpatched: the gate holds a
numbered request outside it until its step (`hooks.StartOnBoundary`: an input starts only when `ready` lets its step
run), so the scheduler sees each request from its admission on, and the batches are a function of the certified releases and the
registered τ, which the encoder's replay recomputes (`advice_code.replay`, ``tick``) without timing advice. A release a
watermark line names lost is skipped (``lost``): the gate doesn't wait for it, the later ones are admitted in order, and
its request, should it reach the engine all the same, before that line or after it, is kept out of the scheduler
(``dropped``), so no step depends on whether or when a lost release's bytes arrive. What isn't certified is added at once
and counted (`TickGate.summary`): a request without the header (``unordered``), one of an older run (``stale``), one that
comes after it was skipped (``late``). A released request that doesn't reach the engine, and that no line names lost,
holds the later ones ``hold_s``, then is skipped; a watermark the gate has waited on ``hold_s`` without a new line, or that
breaks, opens the gate (``opened``: every request is added at once from then on), as a shutdown or a drain does. The
gate requires that inputs start in the order it adds them, on one stream of steps; the adapter refuses a scheduler that
can't give it that. Two frontends feed the gate: `TickAdmit`, where the front hands its engine each input itself (its
watermark route in the server), and, where the server's front runs in another process than its engine, `FrontFeed` on
the front's hooks, which appends the releases and the watermark to a feed file that the engine's gate reads (`engine`).
A request the gate drops is its frontend's to end, and each refuses one that comes after its line with one sentence
(`LOST`): `TickAdmit` answers the one in the engine with that error and aborts it; `FrontFeed` hands the refusal to the
adapter's front (`hooks.RefuseInput`), which ends the request in its server.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from verity import hooks as H

RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER = "x-verity-release", "x-verity-release-run", "x-verity-release-bucket"
WATERMARK_PATH = "/verity/watermark"
#: what either frontend answers a request whose release a watermark line named lost
LOST = RELEASE_HEADER + ": {n} of run {run!r} was lost on the way to the server, as the watermark says"


class ReleaseOrder:
    """Forward each `Admission` to ``forward`` (the adapter's ``submit``) in release order.
    `key` reads a request's headers as the frontend receives it; `submit` and `abort` stand in for the adapter's;
    `passed` says a numbered request won't be submitted (refused, or its connection ended). All of it runs on the
    front's event loop. ``schedule(delay_s, fn)`` arms the skip (default: the running loop's ``call_later``)."""

    #: the release headers `key` reads, the release's number and its run
    headers = (RELEASE_HEADER, RUN_HEADER)

    def __init__(self, forward: Callable[[Any], None], abort: Callable[[str], None], *, hold_s: float,
                 schedule: Callable[[float, Callable[[], None]], Any] | None = None) -> None:
        self.forward, self.abort_forward, self.hold_s, self._schedule = forward, abort, float(hold_s), schedule
        self.run: str | None = None
        self.runs: list[str] = []
        self.next = 0
        self.arrived: set[int] = set()           # this run's ordinals the frontend has received, at or past `next`
        self.held: dict[int, Any] = {}           # ordinal -> its admission, waiting for an earlier one
        self.gone: set[int] = set()              # ordinals past `next` that won't be submitted
        self.skipped: set[int] = set()
        self.armed: tuple[str | None, int] | None = None
        self.counts = {"ordered": 0, "unordered": 0, "late": 0, "stale": 0}
        self.max_held = 0

    def key(self, headers: Mapping[str, str]) -> tuple[str, int] | None:
        """The request's (run, ordinal) from its lower-cased headers, None without one; `ValueError` on a malformed or
        repeated ordinal. A new run starts here, forwarding what the old one still holds."""
        raw = headers.get(RELEASE_HEADER)
        if raw is None:
            return None
        if not raw.strip().isdecimal():
            raise ValueError(f"{RELEASE_HEADER}: a non-negative integer, not {raw!r}")
        run, n = headers.get(RUN_HEADER, ""), int(raw)
        if run != self.run:
            if run in self.runs:
                self.counts["stale"] += 1
                return None
            self._start(run)
        if n in self.arrived or (n < self.next and n not in self.skipped):
            raise ValueError(f"{RELEASE_HEADER}: {n} came twice in run {run!r}")
        if n >= self.next:
            self.arrived.add(n)
        return run, n

    def submit(self, adm: Any) -> None:
        k = getattr(adm, "release", None)
        if k is None or k[0] != self.run:
            self.counts["unordered"] += 1
            return self.forward(adm)
        if k[1] < self.next:
            self.skipped.discard(k[1])
            self.counts["late"] += 1
            return self.forward(adm)
        self.held[k[1]] = adm
        self.max_held = max(self.max_held, len(self.held))
        self._flush()

    def abort(self, engine_id: str) -> None:
        for n, adm in self.held.items():
            if adm.engine_id == engine_id:
                del self.held[n]
                self.gone.add(n)
                return self._flush()
        self.abort_forward(engine_id)

    def passed(self, key: tuple[str, int] | None) -> None:
        if key is None or key[0] != self.run or key[1] < self.next or key[1] in self.held:
            return
        self.gone.add(key[1])
        self._flush()

    def _start(self, run: str) -> None:
        for n in sorted(self.held):
            self.forward(self.held[n])
            self.counts["ordered"] += 1
        self.run, self.next, self.armed = run, 0, None
        self.runs.append(run)
        self.arrived, self.held, self.gone, self.skipped = set(), {}, set(), set()

    def _flush(self) -> None:
        while True:
            n = self.next
            if n in self.held:
                self.forward(self.held.pop(n))
                self.counts["ordered"] += 1
            elif n in self.gone:
                self.gone.discard(n)
            else:
                break
            self.arrived.discard(n)
            self.next += 1
        if self.held and self.armed != (self.run, self.next):
            self.armed = (self.run, self.next)
            schedule = self._schedule or asyncio.get_running_loop().call_later
            schedule(self.hold_s, lambda at=self.armed: self._expire(at))

    def _expire(self, at: tuple[str | None, int]) -> None:
        """``hold_s`` after a later request began waiting on ordinal ``at``: skip it if it still hasn't come."""
        if (self.run, self.next) != at or not self.held:
            return
        self.skipped.add(self.next)
        self.gone.add(self.next)
        self._flush()

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "skipped": len(self.skipped) + self.counts["late"], "skipped_unseen": sorted(self.skipped)[:20],
                "held_now": len(self.held), "max_held": self.max_held, "runs": len(self.runs), "hold_s": self.hold_s}


def watermark_line(line: bytes, last: tuple[int, ...] | None = None) -> tuple[int, int, tuple[int, ...]]:
    """(t, n, lost) of one watermark line, ``t n``, any lost releases and its newline, as the network attester's own
    reader reads it (`verity.protocols.resource_accounting.network.verifier.release.read_watermark`): bucket t has closed, the releases of
    buckets up to t are exactly those numbered below n, and each of ``lost`` is a release of bucket t that its server
    connection lost on the way, which won't come. `ValueError` unless it follows ``last``, the line before it (its
    (t, n)): t one past last's (0 first), n at least last's, and the lost rising from last's n to below n, so no line
    names a release an earlier line counted and each lost one is named once."""
    # Imported where a line is read, not with the module, which an adapter loads in each of its serving processes and
    # which imports only the stdlib and this package.
    from verity.protocols.resource_accounting.network.verifier.release import (
        read_watermark,
    )
    return read_watermark(line, last)


def release_of(headers: Mapping[str, str]) -> tuple[str, int, int | None] | None:
    """(run, N, bucket) of a request's lower-cased headers, None without ``x-verity-release``; `ValueError` on a malformed
    N or bucket. The run is ``x-verity-release-run`` (empty without it), the bucket None without its header."""
    raw = headers.get(RELEASE_HEADER)
    if raw is None:
        return None
    if not raw.strip().isdecimal():
        raise ValueError(f"{RELEASE_HEADER}: a non-negative integer, not {raw!r}")
    bucket = headers.get(BUCKET_HEADER)
    if bucket is not None and not bucket.strip().isdecimal():
        raise ValueError(f"{BUCKET_HEADER}: a non-negative integer, not {bucket!r}")
    return headers.get(RUN_HEADER, "").strip(), int(raw), None if bucket is None else int(bucket)


@dataclass(frozen=True)
class Tick:
    """The certified tick's constants, in microseconds: the network attester's bucket τ_b, and the engine class's step
    times τ_sat (a step that leaves a started input waiting for a slot) and τ_free (one that leaves none)."""

    bucket_us: int
    sat_us: int
    free_us: int

    def __post_init__(self) -> None:
        for name in ("bucket_us", "sat_us", "free_us"):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f"tick {name}: a positive integer of microseconds, not {v!r}")

    @classmethod
    def of_ms(cls, bucket_ms: float, sat_ms: float, free_ms: float) -> Tick:
        return cls(*(int(round(float(ms) * 1000)) for ms in (bucket_ms, sat_ms, free_ms)))

    def close(self, a: int) -> int:
        """When bucket a closes: (a + 1)·τ_b."""
        return (a + 1) * self.bucket_us

    def closed(self, t: int) -> int:
        """The last bucket closed by logical time t, −1 for none."""
        return t // self.bucket_us - 1

    def step(self, saturated: bool) -> int:
        return self.sat_us if saturated else self.free_us

    def record(self) -> dict[str, int]:
        return {"bucket_us": self.bucket_us, "sat_us": self.sat_us, "free_us": self.free_us}


class TickGate:
    """The certified tick's gate on starting inputs (`verity.hooks.Gate`) and a subscriber of the ``steps`` hook, which
    `attach` files and subscribes on an exchange that the adapter's admission hook drives (the module docstring).

    The frontend's records come through `take`: a numbered request's release ``{"key", "run", "n", "bucket"}`` (``key``
    the request's key as the hook reads it), before the request reaches the engine; ``{"run", "n", "passed": true}`` for
    one that won't reach it; each watermark line ``{"run", "t", "n"}``, with ``"lost": [N, ...]`` where it names lost
    releases; and the watermark's end ``{"run", "end": true}`` or its break ``{"run", "error": why}``. ``pull()``, if
    given, returns the records that came since its last call (the adapter's reader of a feed file). The rest come from the hook, on
    the engine's thread: `add` when a request comes, `ready` before each step, and the ``steps`` events (`on`): a step
    that began advances T, and a held request completed from outside is dropped. ``clock`` (seconds) times only the holds and the waits, ``log`` gets one dict per admission (each
    request's steps held, ``held_steps``) and per event.

    ``token_ns`` is the rule's token term: after a step that computed, T also advances that many nanoseconds for each
    slot it computed (`hooks.StepBegan.slots`), on a clock of nanoseconds that T is the floor of in µs, so nothing
    rounds; 0, the default, has none. T catches up only as the registered rule does: to the next release's bucket close
    when no input is active (`hooks.StepDue.active` 0).

    After a line that names N lost, N is in ``lost`` for the rest of its run and the gate holds no request of N's: the
    one it held is dropped, never started, and so is one that comes later (`add`); the hook keeps it from the
    scheduler until the frontend's abort of it comes. ``next`` steps over N as soon as it reaches it, with no hold, so the
    releases after N are admitted in order at their buckets. An open gate orders nothing, lost or not: it has started what
    it held, and starts a lost release's request at once like any other."""

    def __init__(self, tick: Tick, *, hold_s: float, clock: Callable[[], float] = time.monotonic,
                 pull: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
                 log: Callable[[dict[str, Any]], Any] | None = None, token_ns: int = 0) -> None:
        self.tick, self.hold_s, self.clock, self.pull, self.log = tick, float(hold_s), clock, pull, log
        self.token_ns = token_ns
        self.run: str | None = None
        self.runs: list[str] = []
        self.keys: dict[str, tuple[str, int]] = {}      # a release's key -> (run, N)
        self.opened: str | None = None
        self.counts = {"admitted": 0, "unordered": 0, "stale": 0, "late": 0, "skipped": 0, "lost": 0, "dropped": 0,
                       "aborted": 0, "past_end": 0, "bucket_mismatch": 0, "watermark_errors": 0}
        self.steps = 0
        self.waits = {"n": 0, "s": 0.0, "max_s": 0.0}
        self.max_held = 0
        self.wait_start: float | None = None
        self.progress = 0.0
        self._reset()

    def _reset(self) -> None:
        self.T = self.next = self.ns = 0
        self.lines: list[int] = []                      # bucket t -> the watermark's n
        self.ended = False
        self.held: dict[int, tuple[str, Callable[[], None]]] = {}       # N -> (its key, the hook's start)
        self.held_ids: dict[str, int] = {}              # key -> N
        self.held_at: dict[int, int] = {}               # N -> the step count when its request came
        self.held_steps: list[int] = []
        self.buckets: dict[int, int] = {}               # N -> its x-verity-release-bucket
        self.gone: set[int] = set()                     # N past `next` that won't reach the engine
        self.lost: set[int] = set()                     # N a watermark line named lost: never waited for, never added
        self.skipped: set[int] = set()
        self.missing: dict[int, float] = {}             # N -> when the gate began waiting on it

    def _event(self, kind: str, **kw: Any) -> None:
        if self.log is not None:
            self.log({"event": kind, "run": self.run, **kw, "t": self.clock()})

    def _start(self, run: str) -> None:
        if self.run is not None:
            for n in sorted(self.held):
                self._add(n)
            self._event("run", next=run)
        self.run = run
        self.runs.append(run)
        self._reset()

    def take(self, rec: Mapping[str, Any]) -> None:
        run = str(rec.get("run") or "")
        if run != self.run:
            if run in self.runs:
                if "key" in rec:
                    self.keys[str(rec["key"])] = (run, int(rec["n"]))
                return
            self._start(run)
        if "key" in rec:
            n = int(rec["n"])
            self.keys[str(rec["key"])] = (run, n)
            if rec.get("bucket") is not None:
                self.buckets[n] = int(rec["bucket"])
        elif rec.get("passed"):
            n = int(rec["n"])
            if n >= self.next and n not in self.held:
                self.gone.add(n)
        elif "t" in rec:
            t, n = int(rec["t"]), int(rec["n"])
            if t != len(self.lines) or n < (self.lines[-1] if self.lines else 0):
                self.counts["watermark_errors"] += 1
                self._event("watermark", line=[t, n], why="out of order")
                return
            self.lines.append(n)
            self.progress = self.clock()
            for k in rec.get("lost") or ():
                self._lose(int(k))
        elif rec.get("end"):
            self.ended = True
        elif "error" in rec:
            self.counts["watermark_errors"] += 1
            self.open(f"the watermark broke: {rec['error']}")

    def _lose(self, n: int) -> None:
        """A watermark line names release n lost: the gate won't wait for it, and drops its request if it holds it."""
        self.lost.add(n)
        self.counts["lost"] += 1
        held = self.held.pop(n, None)
        if held is not None:
            self.held_ids.pop(held[0], None)
            self.counts["dropped"] += 1
        self._event("lost", n=n, dropped=held is not None)

    def _pull(self) -> None:
        if self.pull is not None:
            for rec in self.pull():
                self.take(rec)

    def attach(self, exchange: Any, name: str | None = None) -> Callable[[], None]:
        """File this gate on ``exchange`` (the adapter's exchange of `verity.hooks` events and requests) under scheme
        ``name`` (the network attester's declaration's by default), and subscribe it to the ``steps`` hook; returns
        the subscription's removal."""
        name = name or scheme_name()
        exchange.request(name, H.StartOnBoundary(self))
        return exchange.subscribe(name, "steps", self.on)

    def add(self, key: str, start: Callable[[], None]) -> None:
        """A request came: ``start`` it now where the gate doesn't order it, else at its step, and never where its
        release the watermark named lost (``dropped``)."""
        k = self.keys.get(key)
        if k is None and self.opened is None:
            self._pull()
            k = self.keys.get(key)
        why = ("opened" if self.opened is not None else "unordered" if k is None else "stale" if k[0] != self.run
               else "dropped" if k[1] in self.lost else "late" if k[1] < self.next else None)
        if why is not None:
            if why != "opened":
                self.counts[why] += 1
                self._event(why, key=key, n=None if k is None else k[1])
            if why == "late":
                self.skipped.discard(k[1])
            if why != "dropped":
                start()
            return
        n = k[1]
        self.held[n] = (key, start)
        self.held_ids[key] = n
        self.held_at[n] = self.steps
        self.gone.discard(n)
        self.missing.pop(n, None)
        self.max_held = max(self.max_held, len(self.held))

    def _add(self, n: int) -> None:
        key, start = self.held.pop(n)
        self.held_ids.pop(key, None)
        self.counts["admitted"] += 1
        self.held_steps.append(self.steps - self.held_at.pop(n, self.steps))
        start()

    def on(self, event: Any) -> None:
        """A ``steps`` event: a step that computes advances T; a held request completed from outside is dropped, as its N
        won't come."""
        if isinstance(event, H.StepBegan):
            if self.run is not None and self.opened is None and event.slots > 0:
                self.T += self.tick.step(event.queued > 0)
                self.steps += 1
                self.ns += event.slots * self.token_ns
                self.T += self.ns // 1000
                self.ns %= 1000
        elif isinstance(event, H.OutputsComplete):
            n = self.held_ids.pop(event.input, None)
            if n is not None:
                del self.held[n]
                self.gone.add(n)
                self.counts["aborted"] += 1

    def ready(self, due: H.StepDue) -> bool:
        """Before a step: admit what T has closed. False while the gate waits on the watermark or on a released request
        (the engine waits too, asking again), True once the step may run."""
        self._pull()
        if self.opened is not None or self.run is None:
            return self._go()
        while self.next in self.gone or self.next in self.lost:
            self.gone.discard(self.next)
            self.next += 1
        quiet = not due.active
        if quiet and due.completing:
            return self._go()                   # the step that frees them comes first, as in the replay
        if quiet:
            a = bisect.bisect_right(self.lines, self.next)      # the next release's bucket, if the watermark has it
            if a == len(self.lines):
                if not self.held:
                    return self._go()
                return self._past_end() if self.ended else self._wait()
            if self.tick.close(a) > self.T:
                self.T, self.ns = self.tick.close(a), 0
        b = self.tick.closed(self.T)
        if b >= len(self.lines) and not self.ended:
            return self._wait()
        target = self.lines[min(b, len(self.lines) - 1)] if b >= 0 and self.lines else 0
        first, self.held_steps = self.next, []
        while self.next < target:
            n = self.next
            if n in self.held:
                if n in self.buckets and self.buckets[n] != bisect.bisect_right(self.lines, n):
                    self.counts["bucket_mismatch"] += 1
                self._add(n)
            elif n in self.gone or n in self.lost:
                self.gone.discard(n)
            elif self.clock() - self.missing.setdefault(n, self.clock()) >= self.hold_s:
                self.skipped.add(n)
                self.counts["skipped"] += 1
                self._event("skipped", n=n)
            else:
                self._admitted(first, b)
                return self._wait(watermark=False)
            self.missing.pop(n, None)
            self.next += 1
        self._admitted(first, b)
        if self.ended and b >= len(self.lines) - 1:
            return self._past_end()
        return self._go()

    def _admitted(self, first: int, b: int) -> None:
        if self.next > first and self.log is not None:
            self.log({"step": self.steps, "T": self.T, "bucket": b, "admitted": [first, self.next],
                      "held_steps": self.held_steps, "t": self.clock()})

    def _past_end(self) -> bool:
        """The run has ended and every release it numbered is in: what the gate still holds is added now."""
        for n in sorted(self.held):
            self.counts["past_end"] += 1
            self._add(n)
        return self._go()

    def _wait(self, watermark: bool = True) -> bool:
        now = self.clock()
        if self.wait_start is None:
            self.wait_start = now
        if watermark and now - max(self.progress, self.wait_start) >= self.hold_s:
            self.open(f"no watermark line for {self.hold_s:g} s, waiting on bucket {self.tick.closed(self.T)} "
                      f"(the last line is {len(self.lines) - 1})")
            return self._go()
        return False

    def _go(self) -> bool:
        if self.wait_start is not None:
            dt = self.clock() - self.wait_start
            self.wait_start = None
            self.waits["n"] += 1
            self.waits["s"] += dt
            self.waits["max_s"] = max(self.waits["max_s"], dt)
        return True

    def open(self, why: str) -> None:
        """Order nothing from now on: what the gate holds is added now, in release order, and every later request at once."""
        if self.opened is None:
            self.opened = why
            self._event("opened", why=why)
        for n in sorted(self.held):
            self._add(n)

    def pending(self) -> bool:
        """The gate holds a request, or waits on the watermark or a release: the engine has work though its scheduler may
        have none."""
        return bool(self.held) or self.wait_start is not None

    def record(self) -> dict[str, int]:
        return self.tick.record()

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "tick": self.tick.record(), "hold_s": self.hold_s, "runs": len(self.runs), "steps": self.steps,
                "T_us": self.T, "next": self.next, "buckets": len(self.lines), "ended": self.ended, "held_now": len(self.held),
                "max_held": self.max_held, "skipped_unseen": sorted(self.skipped)[:20], "opened": self.opened,
                "waits": dict(self.waits), **({"token_ns": self.token_ns} if self.token_ns else {})}


class Watermark:
    """One watermark stream read into gate records (`TickGate.take`, through ``emit``): `feed` its body's bytes as they
    come, then `close` it, with why it broke if it did. A line's record is ``{"run", "t", "n"}``, with ``"lost"``, the
    releases it names lost, where it names any. A line `watermark_line` refuses closes the stream."""

    def __init__(self, run: str, emit: Callable[[dict[str, Any]], Any]) -> None:
        self.run, self.emit = run, emit
        self.last: tuple[int, int] | None = None
        self.buf = b""
        self.error: str | None = None
        self.closed = False

    def feed(self, data: bytes) -> bool:
        """False once the stream is closed."""
        if self.closed:
            return False
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            try:
                t, n, lost = watermark_line(line + b"\n", self.last)
            except ValueError as e:
                return self.close(str(e))
            self.last = (t, n)
            self.emit({"run": self.run, "t": t, "n": n, **({"lost": list(lost)} if lost else {})})
        return True

    def close(self, why: str | None = None) -> bool:
        """End the stream: its end record, or its error (``why``, or a line it ended inside). True for a clean end."""
        if self.closed:
            return self.error is None
        if why is None and self.buf:
            why = f"the watermark ended inside a line ({self.buf[:40]!r})"
        self.closed, self.error = True, why
        self.emit({"run": self.run, "end": True} if why is None else {"run": self.run, "error": why})
        return why is None


def open_watermark(headers: Mapping[str, str], runs: set[str], emit: Callable[[dict[str, Any]], Any]) -> Watermark:
    """The watermark stream a ``POST /verity/watermark`` of these lower-cased headers opens for the run they name, into
    ``emit``; ``runs`` holds the runs whose stream this frontend has had, and gets this one. `ValueError` (the frontend's
    400), with nothing emitted, for the two POSTs that aren't the network attester's stream. One that carries
    ``x-verity-release`` is a client's request: the network attester puts that header on every request it releases
    and never on its own stream. And a run's second: the network attester sends one a run, so a second, whether
    the first is still open or has ended, would be read beside it or after it, and its lost releases acted on."""
    if RELEASE_HEADER in headers:
        raise ValueError(f"{WATERMARK_PATH}: a POST that carries {RELEASE_HEADER} is a client's request, not the network "
                         f"attester's watermark")
    run = headers.get(RUN_HEADER, "").strip()
    if run in runs:
        raise ValueError(f"{WATERMARK_PATH}: run {run!r} has had its watermark stream, and a run has one")
    runs.add(run)
    return Watermark(run, emit)


def scheme_name() -> str:
    """The name the network attester's scheme registers under."""
    from verity.protocols.resource_accounting.network.scheme import NETWORK_ATTESTER
    return NETWORK_ATTESTER.name


class TickAdmit:
    """The frontend of a front that hands its engine each input: `key` reads a request's release headers (`ValueError` on
    a malformed or repeated N, the server's 400), `submit` hands the gate the release (through ``take``) just before the
    request itself (``forward``), `passed` says a numbered request won't be submitted, and the
    watermark's route (`attach`, `serve_watermark`) feeds `watermark` streams, one a run and none of a client's (`open_watermark`). A release
    a watermark line names lost is the gate's to drop and this frontend's to answer, since the engine never finishes a
    request its scheduler never saw: a request of it that is in the engine is answered with that error and aborted, and
    one that comes later is refused (`key`). A watermark that breaks forgets its run's lost releases: the gate is open
    then, and adds every request at once. The frontend sees no other opening of the gate (a watermark that stalls past
    the hold, the engine's drain), so after one a line still has it abort or refuse the requests of the releases it
    names lost, which the open gate adds at once like any other: no admission order rests on which, since an open gate
    orders nothing. All of it runs on the front's event loop."""

    def __init__(self, forward: Callable[[Any], None], abort: Callable[[str], None],
                 take: Callable[[dict[str, Any]], None]) -> None:
        self.forward, self.abort, self.take = forward, abort, take
        self.seen: dict[str, set[int]] = {}
        self.buckets: dict[tuple[str, int], int | None] = {}
        self.submitted: dict[tuple[str, int], Any] = {}     # (run, N) -> its admission, until its answer ends
        self.lost: dict[str, set[int]] = {}                 # run -> the N its watermark named lost
        self.streams: set[str] = set()                      # the runs whose watermark stream has come
        self.counts = {"numbered": 0, "unordered": 0, "passed": 0, "lost": 0, "watermarks": 0}
        self.refuse: Callable[[str, str], None] | None = None

    def attach(self, exchange: Any, name: str | None = None) -> None:
        """Serve the watermark's route on ``exchange`` (the adapter's exchange of `verity.hooks` events and requests)
        under scheme ``name`` (the network attester's declaration's by default), and refuse there the requests of the
        releases it names lost."""
        name = name or scheme_name()
        exchange.request(name, H.ServeRoute(WATERMARK_PATH, self.serve_watermark))
        self.refuse = lambda input, why: exchange.request(name, H.RefuseInput(input, why))

    def serve_watermark(self, first_line: str, headers: Mapping[str, str], body: Iterable[bytes]) -> tuple[int, str]:
        """The watermark's route (`hooks.Route`): its chunked body's lines go to the gate as they come, and its end is
        answered 200 (400 for a malformed stream). A POST that isn't a run's one stream (`watermark`) is answered 400
        with its body unread."""
        if not first_line.startswith("POST "):
            return 405, f"{WATERMARK_PATH} takes POST"
        try:
            w = self.watermark(headers)
        except ValueError as e:
            return 400, str(e)
        if "chunked" not in headers.get("transfer-encoding", "").lower():
            w.close("the watermark's body is chunked")
        try:
            for data in body:
                if w.closed:
                    break
                w.feed(data)
            if not w.closed:
                w.close()
        except H.Broken as e:
            w.close(f"the watermark broke off: {e}")
        return (200, "") if w.error is None else (400, w.error)

    def key(self, headers: Mapping[str, str]) -> tuple[str, int] | None:
        rel = release_of(headers)
        if rel is None:
            return None
        run, n, bucket = rel
        if n in self.lost.get(run, ()):
            raise ValueError(LOST.format(n=n, run=run))
        seen = self.seen.setdefault(run, set())
        if n in seen:
            raise ValueError(f"{RELEASE_HEADER}: {n} came twice in run {run!r}")
        seen.add(n)
        self.buckets[(run, n)] = bucket
        return run, n

    def submit(self, adm: Any) -> None:
        k = getattr(adm, "release", None)
        if k is None:
            self.counts["unordered"] += 1
        else:
            self.counts["numbered"] += 1
            self.take({"key": adm.engine_id, "run": k[0], "n": k[1], "bucket": self.buckets.pop(k, None)})
            self.submitted[k] = adm
        self.forward(adm)

    def passed(self, key: tuple[str, int] | None) -> None:
        if key is None:
            return
        if self.submitted.pop(key, None) is not None:
            return
        self.buckets.pop(key, None)
        self.counts["passed"] += 1
        self.take({"run": key[0], "n": key[1], "passed": True})

    def watermark(self, headers: Mapping[str, str]) -> Watermark:
        """The stream a watermark POST of these lower-cased headers opens (`open_watermark`); `ValueError`, the server's
        400, for a client's and for a run's second. A client's is the request of the release it carries, which won't
        be submitted: its release is read and passed as a refused completion's is, so the gate doesn't wait for it."""
        if RELEASE_HEADER in headers:
            self.passed(self.key(headers))
        w = open_watermark(headers, self.streams, self._record)
        self.counts["watermarks"] += 1
        return w

    def _record(self, rec: dict[str, Any]) -> None:
        """One watermark record, to the gate: before it, a line's lost releases are marked, and after it the requests of
        them that are in the engine are answered and aborted, so the gate drops each before its abort comes."""
        run = rec["run"]
        lost = [int(n) for n in rec.get("lost") or ()]
        if lost:
            self.lost.setdefault(run, set()).update(lost)
            self.counts["lost"] += len(lost)
        if "error" in rec:
            self.lost.pop(run, None)
        self.take(rec)
        for n in lost:
            adm = self.submitted.get((run, n))
            if adm is not None:
                if self.refuse is None:
                    raise RuntimeError("a TickAdmit refuses through the exchange it is attached to (attach)")
                self.refuse(adm.engine_id, LOST.format(n=n, run=run))

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "runs": len(self.seen)}


def tick_config(cfg: Any, source: str) -> dict[str, Any] | None:
    """The certified tick's configuration from ``cfg``, the JSON object the adapter read under ``source``, None for None:
    ``bucket_us``, ``sat_us`` and ``free_us`` (the `Tick`), ``hold_ms`` (5000 by default), ``log`` (a directory for
    each gate's ``tick.PID.jsonl``, optional), and the rule's token term ``token_ns`` (`TickGate`; 0 by default)."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"{source}: a JSON object with bucket_us, sat_us, free_us, not {json.dumps(cfg)[:80]!r}")
    token_ns = cfg.get("token_ns", 0)
    if type(token_ns) is not int or token_ns < 0:
        raise ValueError(f"{source}: token_ns a non-negative integer, not {token_ns!r}")
    return {"tick": Tick(cfg.get("bucket_us"), cfg.get("sat_us"), cfg.get("free_us")),
            "hold_s": float(cfg.get("hold_ms", 5000)) / 1000, "log": cfg.get("log"), "token_ns": token_ns}


#: what `Admitter` adds requests in: the network attester's release order, or that order at the certified tick's steps
ORDERS = ("release-order", "tick")


class Admitter:
    """The network attester's admission on one served node, the one entry an adapter attaches it by: built from its
    configuration and installed on the adapter's exchange before the engine is built, as memory's `Binder` is.

    ``cfg`` is a JSON object: ``order``, one of `ORDERS` (``tick`` by default), ``hold_ms``, and under ``tick`` the rest of
    `tick_config`'s (its ``log`` is the adapter's to write: the gate reads ``pull`` and writes ``log``). ``release-order`` makes the frontend, `ReleaseOrder`, and no gate. ``tick`` makes the gate (`gate`, a
    `verity.hooks.Gate`): `TickGate`, with the rule's token term; and, given ``submit``,
    the frontend `TickAdmit`, which hands the gate its records through ``take`` (the gate's own `TickGate.take` by
    default; where the gate runs on another thread, a call that queues them to it). `ValueError` names what ``cfg``
    lacks. `install` files and subscribes both under `scheme_name`."""

    def __init__(self, cfg: Mapping[str, Any], *, source: str = "the network attester's admission",
                 submit: Callable[[Any], None] | None = None, abort: Callable[[str], None] | None = None,
                 take: Callable[[dict[str, Any]], None] | None = None,
                 pull: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
                 log: Callable[[dict[str, Any]], Any] | None = None, clock: Callable[[], float] = time.monotonic,
                 schedule: Callable[[float, Callable[[], None]], Any] | None = None) -> None:
        order = cfg.get("order", "tick") if isinstance(cfg, Mapping) else None
        if order not in ORDERS:
            raise ValueError(f"{source}: order one of {list(ORDERS)}, not {order!r}")
        if (order == "release-order" or submit is not None) and (submit is None or abort is None):
            raise ValueError(f"{source}: the {order} frontend forwards each request through submit and abort")
        self.gate: TickGate | None = None
        self.front: ReleaseOrder | TickAdmit | None = None
        self.installed = False
        self._off: list[Callable[[], None]] = []
        if order == "release-order":
            self.front = ReleaseOrder(submit, abort, hold_s=float(cfg.get("hold_ms", 5000)) / 1000, schedule=schedule)
            return
        tc = tick_config({k: v for k, v in cfg.items() if k != "order"}, source)
        kw = {"hold_s": tc["hold_s"], "clock": clock, "pull": pull, "log": log}
        self.gate = TickGate(tc["tick"], token_ns=tc["token_ns"], **kw)
        if submit is not None:
            self.front = TickAdmit(submit, abort, take or self.gate.take)

    def install(self, exchange: Any, name: str | None = None) -> Admitter:
        """File and subscribe on ``exchange`` (the adapter's: ``request(name, request)``, and ``subscribe(name, hook,
        fn)`` returning the removal) under ``name`` (`scheme_name` by default), before the engine is built: the gate
        (`TickGate.attach`: `hooks.StartOnBoundary` and the ``steps`` hook) and the frontend's watermark route and
        refusals (`TickAdmit.attach`: `hooks.ServeRoute`, `hooks.RefuseInput`). ``release-order`` files nothing."""
        if self.installed:
            raise RuntimeError("the network attester's admission is installed")
        self.installed = True
        name = name or scheme_name()
        if self.gate is not None:
            self._off.append(self.gate.attach(exchange, name))
        if isinstance(self.front, TickAdmit):
            self.front.attach(exchange, name)
        return self

    def uninstall(self) -> None:
        """Remove the gate's subscription; what `install` filed stays filed, as the exchange takes no request back."""
        offs, self._off = self._off, []
        for off in offs:
            off()


def feed_file(directory: str, name: str | None = None) -> str:
    """The file in a server's feed directory (`hooks.FRONT`'s ``ctx["feed"]``) that the scheme ``name``'s front appends
    to and its engine reads."""
    return os.path.join(directory, f"{name or scheme_name()}.jsonl")


class FeedReader:
    """The records appended to a feed file (JSON lines) since the last call; a partial last line waits for its end."""

    def __init__(self, path: str) -> None:
        self.path, self.fd, self.tail = path, None, b""

    def __call__(self) -> list[dict[str, Any]]:
        if self.fd is None:
            try:
                self.fd = os.open(self.path, os.O_RDONLY)
            except FileNotFoundError:
                return []
        data = self.tail
        while chunk := os.read(self.fd, 1 << 16):
            data += chunk
        *lines, self.tail = data.split(b"\n")
        return [json.loads(x) for x in lines if x.strip()]


def _log_to(directory: str | None) -> Callable[[dict[str, Any]], None] | None:
    if not directory:
        return None
    f = open(os.path.join(directory, f"tick.{os.getpid()}.jsonl"), "a", encoding="utf-8")

    def log(rec: dict[str, Any]) -> None:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()
    return log


def engine(settings: Mapping[str, Any], feed: str, name: str | None = None) -> Admitter:
    """The engine's side (`hooks.ENGINE`): the tick gate (`Admitter` under ``tick`` with ``settings``, `tick_config`'s),
    reading the records the front appends to its feed file (`feed_file` in ``feed``)."""
    return Admitter({**settings, "order": "tick"}, source=f"{name or scheme_name()}'s settings",
                    pull=FeedReader(feed_file(feed, name)), log=_log_to(settings.get("log")))


class FrontFeed:
    """The front's side (`hooks.FRONT`) where the engine runs in another process: `TickAdmit` on the front's hooks,
    appending its records to the feed file (`feed_file`) the engine's gate reads. Each input that arrives with a release
    is that release, its key the input; a malformed or repeated one is refused (`hooks.RefuseInput`); one complete at
    the front at step 0 never reached the engine and is passed; the watermark's route and its lost releases' refusals
    are `TickAdmit.attach`'s."""

    def __init__(self, feed: str, name: str | None = None) -> None:
        self.path = feed_file(feed, name)
        self.admit = TickAdmit(lambda adm: None, lambda key: None, self._append)
        self.inputs: dict[str, tuple[str, int]] = {}

    def _append(self, rec: dict[str, Any]) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, json.dumps(rec, separators=(",", ":")).encode() + b"\n")
        finally:
            os.close(fd)

    def attach(self, exchange: Any, name: str | None = None) -> FrontFeed:
        name = name or scheme_name()
        self.admit.attach(exchange, name)
        exchange.subscribe(name, "steps", self.on)
        return self

    def on(self, ev: Any) -> None:
        if isinstance(ev, H.InputArrived):
            try:
                k = self.admit.key(ev.headers)
            except ValueError as e:
                assert self.admit.refuse is not None
                self.admit.refuse(ev.input, str(e))
                return
            if k is not None:
                self.inputs[ev.input] = k
                self.admit.submit(_Release(k, ev.input))
        elif isinstance(ev, H.OutputsComplete) and ev.input in self.inputs:
            k = self.inputs.pop(ev.input)
            self.admit.submitted.pop(k, None)
            if ev.step == 0:
                self.admit.passed(k)

    def summary(self) -> dict[str, Any]:
        return self.admit.summary()


@dataclass(frozen=True)
class _Release:
    release: tuple[str, int]
    engine_id: str

