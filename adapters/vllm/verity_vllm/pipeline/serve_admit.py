"""serve-openai's admissions: what a request brings to the engine (`Admission`, and `Delta`, what goes back), and the order
and the step they are added in (``--admit``). ``arrival`` adds each request to the engine as soon as the frontend has parsed
it, in the API server's body-read order; ``release-order`` adds them in the network certifier's release order, so the advice
needn't carry the permutation between the two; ``tick`` also fixes the step each is added at, by the certified tick, so it
needn't carry when each arrived either.

`ReleaseOrder` is a reorder buffer in front of the engine loop's ``submit``: it forwards request n of the release order
once the frontend holds every earlier-released request, each forwarded, refused or dropped, and holds it until then. There
is no hold to a bucket boundary (the boundary saves no step bits).

**What the network certifier sends.** Each request it releases carries ``x-verity-release: N``, N its 0-based index in the release
order of its run (the certifier's ``Proxy.requests_released`` before the release counts it): dense, each N once, every
released request numbered, a request a fault drops before its release unnumbered. With ``x-verity-release-run: ID`` (the
certifier run's id, hex) the numbering is per run: a new ID starts a new order at 0, and what the old run still held is
forwarded at once. The network certifier copies both onto the POST it sends the endpoint (`benchmarks/network_traces/
active_replay.LiveServer`, and `warden_serve.py serve --watermark` beside it). The network certifier also sends
``x-verity-release-bucket: A``, the release's bucket counted from 0 at the run's origin, and the watermark: one
``POST /verity/watermark`` a run, naming it, whose chunked body has the line ``t n`` (`watermark_line`) once bucket t has
closed, the releases of buckets up to t being exactly those numbered below n, followed on that line by each release of
bucket t that its server connection lost on the way; its end ends the run.

What a request without the header gets: it is forwarded at once (``unordered``), as the served plan's probes are. A
malformed or repeated ordinal is refused (`ValueError`, the server's 400). A released request that never reaches the
frontend, or whose body doesn't arrive, would hold every later one: once a later one has been held ``hold_s`` behind it,
it is skipped and the held ones go on; if it comes after all it is forwarded at once (``late``). Both are counted
(`summary`), and so are the runs.

**A step's rows**. The advice numbers requests by their first row, so the order of the rows inside a step is a
choice the engine must not make. vLLM's scheduler (FCFS) lists a step's requests as its running ones in running order, then
the ones it newly schedules in waiting-queue order, which is the admission order; Model Runner V2 then lays the step out by
its own `sort_batch_req_ids` (vLLM d9105ea8: decode-length rows first, every other row by its row count, shortest first),
which put a later-admitted shorter prompt's first row ahead of an earlier one's. Under ``release-order``,
`keep_scheduler_rows` replaces that sort with `scheduler_order`: requests verifying draft tokens first (none without
speculative decoding), then every row in the scheduler's order. A step is then its running requests in running order (the
decodes, then the prefill a chunk continues, if any: under FCFS it is the latest scheduled), then its new requests' first
rows in admission order. vLLM sorts decodes first for the attention backends that split a batch into decodes and prefills
(``split_decodes_and_prefills``, the ones with a ``reorder_batch_threshold``: FlashInfer, MLA, Mamba); FLASH_ATTN, the
served backend, isn't one, so no kernel needs the decodes first, and `keep_scheduler_rows` refuses an engine with one.
Batch invariance keeps each row's values independent of its place.

**The certified tick** (``tick``, `TickGate`). Step k has a logical time T_k, an integer of microseconds, and before it the
gate adds exactly the requests whose release bucket a has closed by then, (a + 1)·τ_b ≤ T_k, in release order. After a step
that scheduled a token, T advances by τ_sat if vLLM's waiting queue is nonempty and by τ_free if it is empty; when nothing is
running or waiting, T jumps to the close of the next release's bucket, so T_0 is the first release's. τ_b is the network
certifier's bucket and τ_sat, τ_free the engine class's (`Tick`), registered with the class: a configuration, not a
run-time choice. A release's bucket is the watermark's (the header is only checked against it), and the gate starts step k
once it holds the watermark's line for the last bucket closed by T_k and every request numbered below that line's n but
the ones a line names lost. The rest of each step is vLLM's own scheduler, unpatched: the gate holds a numbered request
outside it until its step (`install`, on the scheduler's ``add_request``, ``schedule`` and ``finish_requests``), so the
scheduler sees each request from its admission on, and the batches are a function of the certified releases and the
registered τ, which the encoder's replay recomputes (`advice_code.replay`, ``tick``) without timing advice. A release a
watermark line names lost is skipped (``lost``): the gate doesn't wait for it, the later ones are admitted in order, and
its request, should it reach the engine all the same, before that line or after it, is kept out of the scheduler
(``dropped``), so no step depends on whether or when a lost release's bytes arrive. What isn't certified is added at once
and counted (`TickGate.summary`): a request without the header (``unordered``), one of an older run (``stale``), one that
comes after it was skipped (``late``). A released request that doesn't reach the engine, and that no line names lost,
holds the later ones ``hold_s``, then is skipped; a watermark the gate has waited on ``hold_s`` without a new line, or that
breaks, opens the gate (``opened``: every request is added at once from then on), as a shutdown or a drain does. Refused: a
scheduler policy other than FCFS (the waiting queue must be the admission order) and more than one data-parallel engine.
Two frontends feed the gate: serve-openai's (`TickAdmit`, its watermark route in the server), and vLLM's own OpenAI server
through `verity_vllm.pipeline.tick_plugin` (its ``--middleware``, which writes the releases and the watermark to a feed
file, and a vLLM general plugin whose gate in the engine core process reads it). A request the gate drops is its
frontend's to end: serve-openai's answers it with an error and aborts it, and refuses one that comes after its line
(`TickAdmit`); under vLLM's own server nothing here does, and the request ends when vLLM aborts it on its connection's
close, a lost release's connection being one that is closing, failed or ended.
"""
from __future__ import annotations

import asyncio
import bisect
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.engine.hooks import Hooks

ADMIT = ("arrival", "release-order", "tick")
RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER = "x-verity-release", "x-verity-release-run", "x-verity-release-bucket"
WATERMARK_PATH = "/verity/watermark"
#: seconds between the gate's looks at its feed while the engine waits on it
POLL_S = 0.002
_ENGINE_ID = re.compile(r"^(?:cmpl|chatcmpl)-(.+)-\d+-[0-9a-f]{8}$")


@dataclass
class Delta:
    """One engine output of a request: the new token ids and text since the last, and its finish."""

    token_ids: list[int]
    text: str
    finish_reason: str | None = None
    stop_reason: Any = None
    prompt_token_ids: list[int] | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None or self.error is not None


@dataclass
class Admission:
    http_id: str
    engine_id: str
    prompt_ids: list[int]
    params: Any
    deliver: Callable[[Delta], None]
    brought: dict[str, Any]                  # the sampling fields it sent, as it sent them
    user: str | None = None
    release: tuple[str, int] | None = None   # (the network certifier's run, the request's place in its release order)
    times: dict[str, float] = field(default_factory=dict)   # wall-clock stamps (telemetry): head, received, submitted, admitted
    ingress: dict[str, Any] = field(default_factory=dict, init=False)   # its ingress record, once the engine loop seeds and adds it


def engine_id_of(http_id: str) -> str:
    """vLLM's engine request id of a completion's one prompt: ``f"{request_id}-{i}"``."""
    return f"{http_id}-0"


def new_http_id() -> str:
    return f"cmpl-{uuid.uuid4().hex}"


class ReleaseOrder:
    """Forward each `Admission` to ``forward`` (the engine loop's ``submit``) in release order.
    `key` reads a request's headers as the frontend receives it; `submit` and `abort` stand in for the engine loop's;
    `passed` says a numbered request won't be submitted (refused, or its connection ended). All of it runs on the HTTP
    server's event loop. ``schedule(delay_s, fn)`` arms the skip (default: the running loop's ``call_later``)."""

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
        if not raw.strip().isdigit():
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


def scheduler_order(num_tokens_per_req: Mapping[str, int], draft_tokens: Mapping[str, Sequence[int]],
                    decode_query_len: int) -> list[str]:
    """Model Runner V2's ``sort_batch_req_ids`` under ``release-order``: the step's requests in the scheduler's order (its
    ``num_scheduled_tokens``), the ones verifying draft tokens first as vLLM keeps them; ``decode_query_len`` is unused."""
    return sorted(num_tokens_per_req, key=lambda r: not draft_tokens.get(r))


def keep_scheduler_rows(runner: Any, scheduler: Any) -> Hooks:
    """Lay each step of ``runner`` (Model Runner V2) out in ``scheduler``'s order (`scheduler_order`, the module docstring),
    until the returned hooks are uninstalled. Refused for a runner whose module has no ``sort_batch_req_ids`` (V1's), for
    one whose attention backends aren't built yet or one of which needs the decodes first (a ``reorder_batch_threshold``),
    and for a scheduler policy other than FCFS, whose waiting queue is not the admission order."""
    module = sys.modules.get(type(runner).__module__)
    if not callable(getattr(module, "sort_batch_req_ids", None)):
        raise RuntimeError(f"--admit release-order keeps the scheduler's row order through Model Runner V2's sort_batch_req_ids, "
                           f"which {type(runner).__module__} has none of: serve with VLLM_USE_V2_MODEL_RUNNER=1 (engine.env)")
    groups = getattr(runner, "attn_groups", None)
    if groups is None:
        raise RuntimeError("--admit release-order reads the runner's attention backends (attn_groups), which aren't built yet: "
                           "keep the scheduler's rows once the engine is up")
    split = sorted({type(b).__name__ for gs in groups for g in gs for b in getattr(g, "metadata_builders", ())
                    if getattr(b, "reorder_batch_threshold", None) is not None})
    if split:
        raise RuntimeError(f"--admit release-order keeps the scheduler's row order, but attention backend {', '.join(split)} "
                           f"splits each batch into decodes and prefills and needs the decodes first (reorder_batch_threshold)")
    policy = getattr(getattr(scheduler, "scheduler_config", None), "policy", None)
    if policy != "fcfs":
        raise RuntimeError(f"--admit release-order needs the FCFS scheduler, whose waiting queue is the admission order; "
                           f"this one's policy is {policy!r}")
    hooks = Hooks(tag="admit-rows")
    hooks.patch(module, "sort_batch_req_ids", scheduler_order)
    return hooks


# --- the certified tick ------------------------------------------------------------------------------------------------


def watermark_line(line: bytes, last: tuple[int, ...] | None = None) -> tuple[int, int, tuple[int, ...]]:
    """(t, n, lost) of one watermark line, ``t n``, any lost releases and its newline, as the network certifier's own
    reader reads it (`verity.protocols.certifier.verifier.release.read_watermark`): bucket t has closed, the releases of
    buckets up to t are exactly those numbered below n, and each of ``lost`` is a release of bucket t that its server
    connection lost on the way, which won't come. `ValueError` unless it follows ``last``, the line before it (its
    (t, n)): t one past last's (0 first), n at least last's, and the lost rising from last's n to below n, so no line
    names a release an earlier line counted and each lost one is named once."""
    # Imported where a line is read, not with the module, which `tick_plugin` loads in each of vLLM's processes and
    # which imports only the stdlib and this package.
    from verity.protocols.certifier.verifier.release import read_watermark
    return read_watermark(line, last)


def release_of(headers: Mapping[str, str]) -> tuple[str, int, int | None] | None:
    """(run, N, bucket) of a request's lower-cased headers, None without ``x-verity-release``; `ValueError` on a malformed
    N or bucket. The run is ``x-verity-release-run`` (empty without it), the bucket None without its header."""
    raw = headers.get(RELEASE_HEADER)
    if raw is None:
        return None
    if not raw.strip().isdigit():
        raise ValueError(f"{RELEASE_HEADER}: a non-negative integer, not {raw!r}")
    bucket = headers.get(BUCKET_HEADER)
    if bucket is not None and not bucket.strip().isdigit():
        raise ValueError(f"{BUCKET_HEADER}: a non-negative integer, not {bucket!r}")
    return headers.get(RUN_HEADER, "").strip(), int(raw), None if bucket is None else int(bucket)


@dataclass(frozen=True)
class Tick:
    """The certified tick's constants, in microseconds: the network certifier's bucket τ_b, and the engine class's step
    times τ_sat (a step after which vLLM's waiting queue is nonempty) and τ_free (one after which it is empty)."""

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


def _quiet(sched: Any) -> bool:
    return not (sched.waiting or getattr(sched, "skipped_waiting", None) or sched.running)


class TickGate:
    """``--admit tick`` in front of one vLLM V1 scheduler (the module docstring).

    The frontend's records come through `take`: a numbered request's release ``{"key", "run", "n", "bucket"}`` (``key``
    the request's id as `install`'s ``key`` reads it), before the request reaches the engine; ``{"run", "n", "passed":
    true}`` for one that won't reach it; each watermark line ``{"run", "t", "n"}``, with ``"lost": [N, ...]`` where it
    names lost releases; and the watermark's end ``{"run", "end": true}`` or its break ``{"run", "error": why}``.
    ``pull()``, if given, returns the records that came since its last call (a `tick_plugin.FeedReader`). The rest runs
    on the engine's thread: `add` is the scheduler's ``add_request``, `finished` its ``finish_requests``, `ready` comes
    before each engine iteration and `scheduled` after each ``schedule()``. ``clock`` (seconds) times only the holds and
    the waits, ``log`` gets one dict per admission and per event.

    After a line that names N lost, N is in ``lost`` for the rest of its run and the gate holds no request of N's: the
    one it held is dropped, never added to the scheduler, and so is one that comes later (`add`). ``next`` steps over N
    as soon as it reaches it, with no hold, so the releases after N are admitted in order at their buckets. An open gate
    orders nothing, lost or not: it has added what it held, and adds a lost release's request at once like any other.
    The frontend still owns a dropped request's answer and its abort: the gate only keeps it from the scheduler."""

    def __init__(self, tick: Tick, *, hold_s: float, clock: Callable[[], float] = time.monotonic,
                 pull: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
                 log: Callable[[dict[str, Any]], Any] | None = None, poll_s: float = POLL_S) -> None:
        self.tick, self.hold_s, self.clock, self.pull, self.log, self.poll_s = tick, float(hold_s), clock, pull, log, float(poll_s)
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
        self.T = 0
        self.next = 0
        self.lines: list[int] = []                      # bucket t -> the watermark's n
        self.ended = False
        self.held: dict[int, tuple[Any, Callable[[Any], Any]]] = {}     # N -> (request, the scheduler's add)
        self.held_ids: dict[str, int] = {}
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
            self.held_ids.pop(held[0].request_id, None)
            self.counts["dropped"] += 1
        self._event("lost", n=n, dropped=held is not None)

    def _pull(self) -> None:
        if self.pull is not None:
            for rec in self.pull():
                self.take(rec)

    def add(self, request: Any, key: str, add: Callable[[Any], Any]) -> Any:
        """The scheduler's ``add_request(request)``, ``add``: now for a request the gate doesn't order, else at its step,
        and never for one whose release the watermark named lost (``dropped``)."""
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
            return None if why == "dropped" else add(request)
        n = k[1]
        self.held[n] = (request, add)
        self.held_ids[request.request_id] = n
        self.gone.discard(n)
        self.missing.pop(n, None)
        self.max_held = max(self.max_held, len(self.held))
        return None

    def _add(self, n: int) -> None:
        request, add = self.held.pop(n)
        self.held_ids.pop(request.request_id, None)
        self.counts["admitted"] += 1
        add(request)

    def finished(self, ids: str | Iterable[str] | None) -> Iterable[str] | None:
        """``finish_requests``' ids less the held requests among them, which are dropped (their N won't come); for every
        request (None), the held ones are added first."""
        if ids is None:
            self.open("every request finished")
            return None
        rest = []
        for rid in (ids,) if isinstance(ids, str) else ids:
            n = self.held_ids.pop(rid, None)
            if n is None:
                rest.append(rid)
                continue
            del self.held[n]
            self.gone.add(n)
            self.counts["aborted"] += 1
        return rest

    def ready(self, sched: Any) -> bool:
        """Before an engine iteration: add what T has closed. False while the gate waits on the watermark or on a released
        request (the engine waits too, polling every ``poll_s``), True once the iteration may run."""
        self._pull()
        if self.opened is not None or self.run is None:
            return self._go()
        while self.next in self.gone or self.next in self.lost:
            self.gone.discard(self.next)
            self.next += 1
        quiet = _quiet(sched)
        if quiet and sched.finished_req_ids:
            return self._go()                   # the step that frees them comes first, as in the replay
        if quiet:
            a = bisect.bisect_right(self.lines, self.next)      # the next release's bucket, if the watermark has it
            if a == len(self.lines):
                if not self.held:
                    return self._go()
                return self._past_end() if self.ended else self._wait()
            self.T = max(self.T, self.tick.close(a))
        b = self.tick.closed(self.T)
        if b >= len(self.lines) and not self.ended:
            return self._wait()
        target = self.lines[min(b, len(self.lines) - 1)] if b >= 0 and self.lines else 0
        first = self.next
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
            self.log({"step": self.steps, "T": self.T, "bucket": b, "admitted": [first, self.next], "t": self.clock()})

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

    def scheduled(self, sched: Any, out: Any) -> None:
        """After ``schedule()``: a step that scheduled a token advances T."""
        if self.run is not None and self.opened is None and getattr(out, "total_num_scheduled_tokens", 0) > 0:
            self.T += self.tick.step(bool(sched.waiting or getattr(sched, "skipped_waiting", None)))
            self.steps += 1

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

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "tick": self.tick.record(), "hold_s": self.hold_s, "runs": len(self.runs), "steps": self.steps,
                "T_us": self.T, "next": self.next, "buckets": len(self.lines), "ended": self.ended, "held_now": len(self.held),
                "max_held": self.max_held, "skipped_unseen": sorted(self.skipped)[:20], "opened": self.opened,
                "waits": dict(self.waits)}


def request_key(engine_id: str) -> str:
    """The client's X-Request-Id inside vLLM's OpenAI server's engine request id (``cmpl-{X-Request-Id}-0-{8 hex}``), else
    the id unchanged: `tick_plugin.TickFeed`'s key for a release."""
    m = _ENGINE_ID.match(engine_id)
    return m.group(1) if m else engine_id


def install(sched: Any, gate: TickGate, key: Callable[[str], str] = request_key) -> Hooks:
    """Put ``gate`` between the engine core and ``sched`` (a vLLM V1 scheduler) until the returned hooks are uninstalled:
    its ``add_request`` goes through `TickGate.add` (``key`` reads a request's release key from its engine id), its
    ``finish_requests`` through `TickGate.finished`, and each ``schedule()`` advances T; ``sched._verity_tick`` is the
    gate. Installed after any other wrapper of these (a step trace's), so that they see a request when it is added.
    Refused for a policy other than FCFS and for more than one data-parallel engine."""
    policy = getattr(getattr(sched, "scheduler_config", None), "policy", None)
    if policy != "fcfs":
        raise RuntimeError(f"--admit tick needs the FCFS scheduler, whose waiting queue is the admission order; this one's "
                           f"policy is {policy!r}")
    dp = getattr(getattr(getattr(sched, "vllm_config", None), "parallel_config", None), "data_parallel_size", 1)
    if dp != 1:
        raise RuntimeError(f"--admit tick orders one engine's admissions; this one is one of {dp} data-parallel engines")
    hooks = Hooks(tag="admit-tick")

    def add_request(add: Callable[..., Any]) -> Callable[..., Any]:
        def gated(request: Any, *args: Any, **kwargs: Any) -> Any:
            return gate.add(request, key(request.request_id), lambda r: add(r, *args, **kwargs))
        return gated

    def schedule(orig: Callable[..., Any]) -> Callable[..., Any]:
        def ticked(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            gate.scheduled(sched, out)
            return out
        return ticked

    def finish_requests(orig: Callable[..., Any]) -> Callable[..., Any]:
        def finish(request_ids: Any, finished_status: Any) -> Any:
            return orig(gate.finished(request_ids), finished_status)
        return finish

    hooks.wrap(sched, "add_request", add_request)
    hooks.wrap(sched, "schedule", schedule)
    hooks.wrap(sched, "finish_requests", finish_requests)
    hooks.patch(sched, "_verity_tick", gate)
    return hooks


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


class TickAdmit:
    """serve-openai's frontend under ``--admit tick``: `key` reads a request's release headers (`ValueError` on a malformed
    or repeated N, the server's 400), `submit` hands the engine loop's gate the release (through ``take``, the loop's
    queue) just before the request itself (``forward``), `passed` says a numbered request won't be submitted, and the
    server's watermark route feeds `watermark` streams. A release a watermark line names lost is the gate's to drop and
    this frontend's to answer, since the engine never finishes a request its scheduler never saw: a request of it that
    is in the engine is answered with that error and aborted, and one that comes later is refused (`key`). A watermark
    that breaks forgets its run's lost releases: the gate is open then, and adds every request at once. The frontend
    sees no other opening of the gate (a watermark that stalls past the hold, the engine loop's drain), so after one a
    line still has it abort or refuse the requests of the releases it names lost, which the open gate adds at once like
    any other: no admission order rests on which, since an open gate orders nothing. All of it runs on the HTTP server's
    event loop."""

    LOST = RELEASE_HEADER + ": {n} of run {run!r} was lost on the way to the server, as the watermark says"

    def __init__(self, forward: Callable[[Any], None], abort: Callable[[str], None],
                 take: Callable[[dict[str, Any]], None]) -> None:
        self.forward, self.abort, self.take = forward, abort, take
        self.seen: dict[str, set[int]] = {}
        self.buckets: dict[tuple[str, int], int | None] = {}
        self.submitted: dict[tuple[str, int], Any] = {}     # (run, N) -> its admission, until its answer ends
        self.lost: dict[str, set[int]] = {}                 # run -> the N its watermark named lost
        self.counts = {"numbered": 0, "unordered": 0, "passed": 0, "lost": 0, "watermarks": 0}

    def key(self, headers: Mapping[str, str]) -> tuple[str, int] | None:
        rel = release_of(headers)
        if rel is None:
            return None
        run, n, bucket = rel
        if n in self.lost.get(run, ()):
            raise ValueError(self.LOST.format(n=n, run=run))
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

    def watermark(self, run: str) -> Watermark:
        self.counts["watermarks"] += 1
        return Watermark(run, self._record)

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
                adm.deliver(Delta([], "", error=self.LOST.format(n=n, run=run)))
                self.abort(adm.engine_id)

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "runs": len(self.seen)}