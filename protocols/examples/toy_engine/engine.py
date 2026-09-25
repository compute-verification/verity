"""A toy CPU inference engine: request queue, batched decode steps, one small IR Program per batch size, a fake KV
cache and weights in memory regions, and a simulated clock.  It knows nothing about protocols; it exposes hook
points and a clock that timers can interrupt.

Each decode step, per request:  h = x + kv          (ToyMix)
                                 y_m = <h, W_m>      (PoUW's NoisedDot, one tile per row; y' is the noised output)
                                 kv' = kv + y        (ToyAccum), written back to the request's KV page
and the request's next x is y.  Everything is mod 2^16 and evaluated by the reference evaluator.

Hooks, in the order one step emits them:  step_start -> (compute; timers fire at yield points) -> commit -> step_end.
Also: admit (request enters the queue), alloc (KV page allocated or freed), idle (queue empty until `until`).
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from verity.evaluation import evaluate_region
from verity.ir import Array, Program, Tuple, Value, array_of, bind, composite, query, tuple_of
from verity_pouw import NoisedDot, U16Add

V16 = Value(16)


def _vec(n):
    return Array(n, V16)


@composite("ToyMix", 1, ["K"], lambda S: ((("x", _vec(S.K)), ("kv", _vec(S.K))), _vec(S.K)))
def ToyMix(B, S, x, kv):
    return array_of([B.call(U16Add, x[i], kv[i]) for i in range(S.K)])


@composite("ToyAccum", 1, ["K"], lambda S: ((("kv", _vec(S.K)), ("y", _vec(S.K))), _vec(S.K)))
def ToyAccum(B, S, kv, y):
    return array_of([B.call(U16Add, kv[i], y[i]) for i in range(S.K)])


@composite("ToyToken", 1, ["K"], lambda S: (
    (("x", _vec(S.K)), ("kv", _vec(S.K)), ("W", Array(S.K, _vec(S.K))), ("e", _vec(S.K)), ("f", _vec(S.K))),
    Tuple(Array(S.K, Tuple(V16, V16)), _vec(S.K))))
def ToyToken(B, S, x, kv, W, e, f):
    h = B.call(bind(ToyMix, K=S.K), x, kv)
    tiles = B.batch(bind(NoisedDot, K=S.K), h, W, e, f, axes=(None, 0, None, None))
    y = array_of([tiles[m][0] for m in range(S.K)])
    return tuple_of(tiles, B.call(bind(ToyAccum, K=S.K), kv, y))


@composite("ToyStep", 1, ["B", "K"], lambda S: (
    tuple((n, Array(S.B, _vec(S.K))) for n in ("X", "KV")) + (("W", Array(S.K, _vec(S.K))),)
    + tuple((n, Array(S.B, _vec(S.K))) for n in ("E", "F")),
    Array(S.B, Tuple(Array(S.K, Tuple(V16, V16)), _vec(S.K)))))
def ToyStep(B, S, X, KV, W, E, F):
    return B.batch(bind(ToyToken, K=S.K), X, KV, W, E, F, axes=(0, 0, None, 0, 0))


class Conflict(Exception):
    """Two parties want one resource: memory, the clock, or a property of the step loop."""


@dataclass
class Region:
    name: str
    tier: str
    owner: str
    words: np.ndarray  # (sectors, sector_words) uint16


class Memory:
    """Memory regions in tiers ("hbm", "host").  `offloaded` sectors are really held in host memory (they read at the
    host rate); `dropped` sectors are gone (they read as zeros).  Both model a cheating deployment."""

    def __init__(self, sector_words: int, capacity: dict[str, int], rates: dict[str, float], sector_bytes: int) -> None:
        self.sector_words, self.capacity, self.rates = sector_words, capacity, rates
        self.sector_bytes = sector_bytes  # real bytes a toy sector stands for, for timing
        self.regions: dict[str, Region] = {}
        self.offloaded: set[tuple[str, int]] = set()
        self.dropped: set[tuple[str, int]] = set()
        self.peak: dict[str, int] = {}

    def used(self, tier: str) -> int:
        return sum(len(r.words) for r in self.regions.values() if r.tier == tier)

    def allocate(self, name: str, tier: str, sectors: int, owner: str, words: np.ndarray | None = None) -> Region:
        old = len(self.regions[name].words) if name in self.regions else 0
        free = self.capacity[tier] - self.used(tier) + old
        if sectors > free:
            holders = ", ".join(f"{r.name} ({r.owner}) {len(r.words)}" for r in self.regions.values() if r.tier == tier)
            raise Conflict(f"memory: {name} ({owner}) needs {sectors} sectors of {tier}, {free} free; holders: {holders}")
        if words is None:
            words = np.zeros((sectors, self.sector_words), np.uint16)
        self.regions[name] = Region(name, tier, owner, words)
        self.peak[name] = max(self.peak.get(name, 0), sectors)
        return self.regions[name]

    def read(self, region: str, sector: int) -> tuple[np.ndarray, float]:
        r = self.regions[region]
        words = np.zeros_like(r.words[sector]) if (region, sector) in self.dropped else r.words[sector]
        tier = "host" if (region, sector) in self.offloaded else r.tier
        return words, self.sector_bytes / self.rates[tier]


class Clock:
    """Simulated time plus timers.  The engine calls `fire()` at its yield points; a timer's handler sees how late it
    runs (`now - scheduled`) and may advance the clock for time it holds exclusively."""

    def __init__(self) -> None:
        self.now = 0.0
        self._timers: list = []

    def at(self, t: float, fn: Callable[[float], None]) -> None:
        heapq.heappush(self._timers, (t, len(self._timers), fn))

    def advance(self, dt: float) -> None:
        self.now += dt

    def fire(self, until: float | None = None) -> None:
        """Run due timers.  While idle (`until` given) each runs at its own time, up to `until`."""
        while self._timers and self._timers[0][0] <= (self.now if until is None else until):
            t, _, fn = heapq.heappop(self._timers)
            if until is not None:
                self.now = max(self.now, t)
            fn(t)


class Hooks:
    """The application-owned hook registry: handlers run in registration order."""

    NAMES = ("admit", "alloc", "step_start", "commit", "step_end", "idle")

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {n: [] for n in self.NAMES}

    def on(self, name: str, fn: Callable) -> None:
        self._handlers[name].append(fn)

    def emit(self, name: str, event) -> None:
        for fn in self._handlers[name]:
            fn(event)


@dataclass
class Request:
    id: str
    prompt: np.ndarray
    tokens: int
    arrival: float
    header: bytes
    x: np.ndarray | None = None
    page: int | None = None
    last_step: tuple[str, int] | None = None  # (step id, batch position) of the previous token
    out: list = field(default_factory=list)


@dataclass
class Step:
    id: str
    requests: list[Request]
    program: Program
    placement: dict[str, tuple[int, int]]  # request -> its ToyToken's gate interval
    previous: dict[str, tuple[str, int] | None]
    now: float
    noise: dict[str, tuple[list[int], list[int]]] = field(default_factory=dict)
    values: dict[int, int] = field(default_factory=dict)


@dataclass
class Admit:
    request: Request
    now: float


@dataclass
class Alloc:
    page: int
    owner: str | None  # None: freed
    words: np.ndarray


@dataclass
class Commit:
    step: Step
    values: dict[int, int]
    kv_writes: list[tuple[int, np.ndarray]]
    now: float


@dataclass
class Idle:
    now: float
    until: float


@dataclass(frozen=True)
class EngineConfig:
    K: int = 8
    batch: int = 8
    compute_rate: float = 1e8  # MACs per simulated second
    launch: float = 1e-6  # per-step overhead, never preemptible
    preemptible: bool = True  # yield to timers between the requests of a step, not only between steps

    @property
    def chunk(self) -> float:
        return 3 * self.K * self.K / self.compute_rate  # noised product plus correction, per request

    @property
    def yield_latency(self) -> float:
        """The longest a timer can wait for this engine: its longest non-preemptible stretch."""
        return max(self.launch, self.chunk) if self.preemptible else self.launch + self.batch * self.chunk


class Engine:
    def __init__(self, cfg: EngineConfig, memory: Memory, clock: Clock, hooks: Hooks) -> None:
        self.cfg, self.memory, self.clock, self.hooks = cfg, memory, clock, hooks
        self.programs: dict[int, Program] = {}
        self.steps: list[Step] = []
        self.free_pages = deque(range(len(memory.regions["kv"].words)))
        self.busy = 0.0

    def program(self, b: int) -> Program:
        if b not in self.programs:
            self.programs[b] = Program(bind(ToyStep, B=b, K=self.cfg.K))
        return self.programs[b]

    def gates(self, program: Program, param: int) -> list[int]:
        refs = program.input_coll(param).refs
        return [refs[i][2] for i in range(refs.count)]

    def run(self, requests: list[Request], until: float) -> None:
        pending, queue, running = deque(sorted(requests, key=lambda r: r.arrival)), deque(), []
        while pending or queue or running:
            while pending and pending[0].arrival <= self.clock.now:
                self._admit(pending.popleft(), queue)
            while queue and len(running) < self.cfg.batch and self.free_pages:
                r = queue.popleft()
                r.page = self.free_pages.popleft()
                words = self.memory.regions["kv"].words[r.page]
                words[:] = 0
                self.hooks.emit("alloc", Alloc(r.page, r.id, words))
                running.append(r)
            if not running:
                self._idle(pending[0].arrival)
                continue
            self._step(running)
            for r in [r for r in running if len(r.out) == r.tokens]:
                running.remove(r)
                self.free_pages.append(r.page)
                self.hooks.emit("alloc", Alloc(r.page, None, self.memory.regions["kv"].words[r.page]))
        if self.clock.now < until:
            self._idle(until)

    def _admit(self, r: Request, queue: deque) -> None:
        r.x = r.prompt.copy()
        self.hooks.emit("admit", Admit(r, self.clock.now))
        queue.append(r)

    def _idle(self, until: float) -> None:
        self.hooks.emit("idle", Idle(self.clock.now, until))
        self.clock.fire(until)
        self.clock.now = max(self.clock.now, until)

    def _step(self, batch: list[Request]) -> None:
        cfg, K = self.cfg, self.cfg.K
        program = self.program(len(batch))
        tokens = query(program.circuit, "//ToyToken")
        step = Step(f"s{len(self.steps)}", list(batch), program, {r.id: tokens[j].interval for j, r in enumerate(batch)},
                    {r.id: r.last_step for r in batch}, self.clock.now)
        self.hooks.emit("step_start", step)
        self.busy += cfg.launch + len(batch) * cfg.chunk
        self.clock.advance(cfg.launch)
        self.clock.fire()
        for _ in batch:
            self.clock.advance(cfg.chunk)
            if cfg.preemptible:
                self.clock.fire()
        self.clock.fire()

        kv = self.memory.regions["kv"].words
        W = self.memory.regions["weights"].words.ravel()[:K * K]
        zeros = [0] * K
        inputs = (
            [v for r in batch for v in r.x.tolist()],
            [v for r in batch for v in kv[r.page][:K].tolist()],
            W.tolist(),
            [v for r in batch for v in step.noise.get(r.id, (zeros, zeros))[0]],
            [v for r in batch for v in step.noise.get(r.id, (zeros, zeros))[1]],
        )
        values = step.values
        for param, vals in enumerate(inputs):
            values.update(zip(self.gates(program, param), vals))
        evaluate_region(program.circuit, program.circuit[0:program.circuit.size], values)

        writes = []
        for j, r in enumerate(batch):
            ret = tokens[j].returns
            kv[r.page][:K] = [values[g] for g in ret[2 * K:3 * K]]
            writes.append((r.page, kv[r.page]))
        self.hooks.emit("commit", Commit(step, values, writes, self.clock.now))
        for j, r in enumerate(batch):
            ret = tokens[j].returns
            r.x = np.array([values[g] for g in ret[0:2 * K:2]], np.uint16)
            r.out.append(r.x)
            r.last_step = (step.id, j)
        self.steps.append(step)
        self.hooks.emit("step_end", step)
