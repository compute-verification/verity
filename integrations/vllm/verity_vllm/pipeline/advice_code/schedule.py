"""What the scheduler replay schedules (`replay`, over the KV pool `pool`): the registered engine configuration
(`Engine`) and certified tick (`Tick`), each request's certified I/O (`RequestIO`) and its scheduler state (`RequestState`, with
its block hashes, `BlockHashes`), and the steps the engine executed (`Step`, its rows (request, start, n)).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BlockHashes", "cdiv", "Engine", "FINISHED", "PREEMPTED", "RequestIO", "RequestState", "Row", "RUNNING", "Step",
    "Tick", "WAITING"]


# a request's status (`vllm.v1.request.RequestStatus`)
WAITING, RUNNING, PREEMPTED, FINISHED = "waiting", "running", "preempted", "finished"


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class BlockHashes:
    """Block hashes as interned ids: equal ids iff equal chained (parent, tokens), so there are no collisions."""

    def __init__(self) -> None:
        self._ids: dict[tuple[int, tuple[int, ...]], int] = {}

    def __call__(self, parent: int, block: tuple[int, ...]) -> int:
        return self._ids.setdefault((parent, block), len(self._ids) + 1)


@dataclass(frozen=True)
class Engine:
    """The configuration vLLM's scheduler reads, as `verity_vllm.engine.step_trace.scheduler_config` records it."""

    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    num_gpu_blocks: int
    block_size: int = 16
    long_prefill_token_threshold: int = 0
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    async_scheduling: bool = False
    watermark: float = 0.0
    scheduler_reserve_full_isl: bool = True
    policy: str = "fcfs"

    def __post_init__(self) -> None:
        if self.policy != "fcfs":
            raise ValueError(f"policy {self.policy!r}: the replay models FCFS only")

    @classmethod
    def from_header(cls, config: dict[str, Any]) -> Engine:
        asy = config.get("scheduler") == "AsyncScheduler" if config.get("scheduler") else config.get("async_scheduling")
        return cls(max_num_seqs=int(config["max_num_seqs"]), max_num_batched_tokens=int(config["max_num_batched_tokens"]),
                   max_model_len=int(config["max_model_len"]), num_gpu_blocks=int(config["num_gpu_blocks"]),
                   block_size=int(config.get("block_size", 16)),
                   long_prefill_token_threshold=int(config.get("long_prefill_token_threshold") or 0),
                   enable_chunked_prefill=bool(config.get("enable_chunked_prefill", True)),
                   enable_prefix_caching=bool(config.get("enable_prefix_caching", False)), async_scheduling=bool(asy),
                   watermark=float(config.get("watermark", 0.0) or 0.0),
                   scheduler_reserve_full_isl=bool(config.get("scheduler_reserve_full_isl", True)),
                   policy=str(config.get("policy", "fcfs")))


@dataclass(frozen=True)
class Tick:
    """The certified tick's registered constants, in microseconds: the network certifier's bucket τ_b, and the engine
    class's step times τ_sat (a step after which the waiting queue is nonempty) and τ_free (one after which it is empty).
    The serving side's is `verity_vllm.pipeline.serve_admit.Tick`; this stdlib copy keeps the replay loadable alone."""

    bucket_us: int
    sat_us: int
    free_us: int

    def __post_init__(self) -> None:
        for name in ("bucket_us", "sat_us", "free_us"):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f"tick {name}: a positive integer of microseconds, not {v!r}")

    def close(self, a: int) -> int:
        """When bucket a closes: (a + 1)·τ_b."""
        return (a + 1) * self.bucket_us


@dataclass(frozen=True)
class RequestIO:
    """One request as the certified I/O shows it. `output_len` counts every token the request emitted, its stop token
    included; `eos` is whether it stopped on a stop token there (else it ran to `max_tokens` or `max_model_len`). `tokens`,
    the prompt then the output, is read only with prefix caching on."""

    prompt_len: int
    max_tokens: int
    output_len: int
    eos: bool
    tokens: tuple[int, ...] | None = None


Row = tuple[int, int, int]          # (request index, first position, tokens scheduled)


@dataclass
class Step:
    """One executed step: its rows in the scheduler's order (running, then new and resumed), and the rest of what
    `SchedulerOutput` says of it."""

    rows: list[Row]
    new: list[int] = field(default_factory=list)
    resumed: list[int] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)
    cache_hit: dict[int, int] = field(default_factory=dict)
    sources: dict[int, list[tuple[int, int]]] = field(default_factory=dict)   # a hit's runs (request, tokens) by owner

    def key(self) -> tuple[Row, ...]:
        return tuple(sorted(self.rows))


class RequestState:
    """A request's scheduler state (`vllm.v1.request.Request`, the fields the served subset reads)."""

    def __init__(self, i: int, io: RequestIO, hashes: BlockHashes | None, block_size: int) -> None:
        self.i = i
        self.io = io
        self.P = io.prompt_len
        self.n_out = 0
        self.num_computed = 0
        self.placeholders = 0
        self.status = WAITING
        self.is_prefill_chunk = False
        self.in_flight = 0
        self.stale = 0
        self.drop_stale = False
        self.preemptions = 0
        self.finish: str | None = None
        self._hashes = hashes
        self._B = block_size
        self.block_hashes: list[int] = []
        if hashes is not None:
            if io.tokens is None or len(io.tokens) < io.prompt_len + io.output_len:
                raise ValueError(f"request {i}: prefix caching is on, so the replay needs its tokens (prompt then output)")
            self._extend_hashes()

    @property
    def num_tokens(self) -> int:
        return self.P + self.n_out

    def _extend_hashes(self) -> None:
        toks = self.io.tokens
        assert toks is not None and self._hashes is not None
        while (len(self.block_hashes) + 1) * self._B <= self.num_tokens:
            k = len(self.block_hashes)
            parent = self.block_hashes[-1] if self.block_hashes else 0
            self.block_hashes.append(self._hashes(parent, tuple(toks[k * self._B:(k + 1) * self._B])))

    def append_output(self) -> None:
        self.n_out += 1
        if self._hashes is not None:
            self._extend_hashes()

    def fork(self) -> RequestState:
        r = copy.copy(self)
        r.block_hashes = list(self.block_hashes)
        return r
