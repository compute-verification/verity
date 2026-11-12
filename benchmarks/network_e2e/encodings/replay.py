"""The scheduler replay's traces, and the timing input coded inside the replay.

The replay is `verity_vllm.pipeline.advice_code.replay` (`Engine`, `Tick`, `replay`, `compare`, over the KV pool
`pool.BlockPool`; its docstring states what it models and refuses), re-exported here for the modules and tests that load this
one by path. This module reads step traces as `Trace`s (`read_step_trace`, a vLLM run's; `read_corpus_trace`, the trace
corpus's) and codes the replay's timing input.

**The residual advice** is the timing input, coded online inside the replay. A request's visibility matters only at an
*open* step: one whose waiting loop reached the queue's end with room to admit it. At every other step it would be blocked
whether it had arrived or not, and when the engine is quiet (nothing running or waiting) the next arrival is what starts
the next step, so neither costs a bit. So the advice is one binary decision per open step, whether the next request had
arrived (`decisions`): `gamma_values` sends each request's count of open steps passed over as Elias gamma, and `encode`
codes the decisions with an adaptive binary arithmetic code inside the replay, one bit string per certifier window. Its
contexts (`CONTEXTS`) are the request's passes so far, its certified ingress bucket against the release buckets around
the step (`_Clock`), and its certified egress: the replay, forked at the open step and run on with the request arriving
now or a step later and nothing after it, says which of its steps emit its outputs, and an earlier request's output from
each of those steps says which bucket it left in (`_Egress`). `decode` replays from that advice and the certified I/O
alone, and
`table` counts the bits per certifier window against I1 v0's (`advice.bits_by_window`, else `v0_widths`).

**With a draft** the replay takes one more input, the drafts each row kept, and `draft_table` reports a trace's two
classes of advice: the arrivals (the residual advice above) and the kept count (`kept_bits`: charged at its measured
entropy, or 0 where the audited block derives it).
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.pipeline.advice_code.pool import BlockPool as BlockPool
from verity_vllm.pipeline.advice_code.replay import (
    compare, drive as drive, Emitted, Match as Match, Open, Queue as Queue, Replay, replay,
    Scheduler as Scheduler,
)
from verity_vllm.pipeline.advice_code.schedule import (
    cdiv, DraftEngine as DraftEngine, Engine, BlockHashes as BlockHashes, RequestState as RequestState, RequestIO, Row as Row,
    RUNNING as RUNNING, Step, Tick,
)

STEP_TRACE = "verity-vllm/step-trace/v1"


# --- traces ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Releases:
    """A tick-admitted run's certified releases, as the network certifier (or the trace corpus's stand-in for it) gave
    them: the registered tick, the certifier run's origin (unix seconds; bucket a is [origin + a·τ_b, origin + (a + 1)·τ_b)),
    and each request's release number N and bucket a, in the trace's engine order."""

    tick: Tick
    origin_unix: float
    n: tuple[int, ...]
    bucket: tuple[int, ...]


@dataclass
class Trace:
    """A served run's step trace (`verity-vllm/step-trace/v1`), its requests in the order they reached the engine, with
    the certifier's view of its timing when the trace has a clock: each request's ingress time and each step's release
    time (its output's tokens leaving through the egress certifier), or each kept output's own (`output_t`) where the
    recorder saw them leave one by one; and `releases`, a run served under the certified tick's releases."""

    engine: Engine
    ids: list[str]
    requests: list[RequestIO]
    before_step: list[int]          # each request's admission step, as the engine core added it
    steps: list[Step]
    emits: list[bool]               # whether each step emitted a token
    ingress_t: list[float | None]
    release_t: list[float | None]
    header: dict[str, Any] = field(default_factory=dict)
    output_t: list[list[float]] | None = None
    releases: Releases | None = None
    passed: list[int] | None = None  # the empty schedules each request arrived after at its step (`replay`'s `passed`)
    emitted: list[dict[int, int]] | None = None  # the tokens each step's requests emitted, where the trace records them

    def first_scheduled(self) -> list[int]:
        first = [-1] * len(self.requests)
        for k, st in enumerate(self.steps):
            for i, _, _ in st.rows:
                if first[i] < 0:
                    first[i] = k
        return first

    def clocked(self) -> bool:
        return all(t is not None for t in self.ingress_t) and all(t is not None for t in self.release_t)

    def kept_steps(self) -> list[list[int]]:
        """Each request's kept outputs as the steps that emitted them: a row emits when it reaches the request's frontier
        (its prompt and the outputs kept so far), up to its output length. With a draft a row emits its token and the
        drafts it kept, at positions that depend on the rows before it, so the steps are read from the trace's own
        count of what each emitted."""
        kept: list[list[int]] = [[] for _ in self.requests]
        if self.engine.num_speculative_tokens:
            if self.emitted is None:
                raise ValueError("with a draft, the steps a request's outputs left in are the trace's `emitted`, which "
                                 "this trace does not record")
            for s, em in enumerate(self.emitted):
                for i, k in em.items():
                    kept[i].extend([s] * k)
            return kept
        for s, st in enumerate(self.steps):
            for i, start, n in st.rows:
                if start + n == self.requests[i].prompt_len + len(kept[i]) and len(kept[i]) < self.requests[i].output_len:
                    kept[i].append(s)
        return kept

    def output_times(self) -> list[list[float]]:
        """When each kept output left: its own time where the trace has one, else its step's release."""
        if self.output_t is not None:
            return self.output_t
        return [[self.release_t[s] for s in ks] for ks in self.kept_steps()]  # type: ignore[misc]

    def origin(self, bucket_s: float) -> int:
        """The session's first bucket, its earliest ingress's, which the certified record's buckets count from (the
        advice module's session-global buckets); 0 without a clock."""
        return min(bucket(t, bucket_s) for t in self.ingress_t) if self.clocked() else 0  # type: ignore[arg-type]

    def certified(self, bucket_s: float = 0.1) -> Certified:
        """What the verifier holds of this run without advice (`Certified`)."""
        if not self.clocked():
            return Certified(self.engine, list(self.requests), bucket_s, None, None)
        o = self.origin(bucket_s)
        return Certified(self.engine, list(self.requests), bucket_s,
                         [bucket(t, bucket_s) - o for t in self.ingress_t],                              # type: ignore[arg-type]
                         [[bucket(t, bucket_s) - o for t in ts] for ts in self.output_times()])


@dataclass(frozen=True)
class Certified:
    """What the verifier holds without advice: the registered engine, each request's certified I/O in the order it reached
    the engine, and, with a clock, each request's ingress bucket and the bucket each of its kept outputs left in, counted
    from the session's first (`Trace.origin`). The decoder reads nothing else."""

    engine: Engine
    requests: list[RequestIO]
    bucket_s: float
    ingress: list[int] | None
    release: list[list[int]] | None

    def windows(self, window_buckets: int) -> list[int]:
        """Each request's certifier window (its ingress bucket's); one window without a clock."""
        return [0] * len(self.requests) if self.ingress is None else [b // window_buckets for b in self.ingress]


def load_step_trace(records: Iterable[dict[str, Any]], *, tokens: dict[str, Sequence[int]] | None = None,
                    ingress_lag_s: float = 0.0, release_lag_s: float = 0.0) -> Trace:
    """A `Trace` from `verity_vllm.engine.step_trace` records. Output lengths and stops come from the trace's `emitted` and
    `finished` (in a served run the committed sampled tokens and the certified egress say the same); with a draft (the
    header's `speculative`), so do the drafts each row kept, its `emitted` less its own token. An `arrive` or
    `step` record's `t` is the engine's clock; a record's `ingress_t` / `release_t`, where the recorder has the
    certifier's, wins, and otherwise the engine's clock less `ingress_lag_s` (plus `release_lag_s`) stands in for it."""
    recs = list(records)
    if not recs or recs[0].get("kind") != "header" or recs[0].get("schema") != STEP_TRACE:
        raise ValueError(f"not a {STEP_TRACE} trace")
    engine = Engine.from_header(recs[0]["config"])
    arrivals = [r for r in recs if r["kind"] == "arrive"]
    ids = [r["request"] for r in arrivals]
    index = {rid: k for k, rid in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("a request arrives twice")
    steps_recs = sorted((r for r in recs if r["kind"] == "step"), key=lambda r: r["step"])
    if [r["step"] for r in steps_recs] != list(range(len(steps_recs))):
        raise ValueError("the trace's steps are not 0..S-1 each once")
    emitted = [0] * len(ids)
    kept: list[list[int]] = [[] for _ in ids]
    finish: list[str | None] = [None] * len(ids)
    steps = []
    for r in steps_recs:
        steps.append(Step(rows=[(index[a], int(b), int(c)) for a, b, c in r["rows"]], new=[index[a] for a in r.get("new", ())],
                          resumed=[index[a] for a in r.get("resumed", ())], preempted=[index[a] for a in r.get("preempted", ())],
                          cache_hit={index[a]: int(k) for a, k in (r.get("cache_hit") or {}).items()},
                          drafts={index[a]: int(k) for a, k in (r.get("drafts") or {}).items()}))
        for a, k in (r.get("emitted") or {}).items():
            emitted[index[a]] += int(k)
            kept[index[a]].append(int(k) - 1)
        for a, f in (r.get("finished") or {}).items():
            finish[index[a]] = f["reason"]
    for r in recs:
        if r["kind"] == "finish":
            finish[index[r["request"]]] = r["reason"]
    requests = []
    for k, a in enumerate(arrivals):
        sp = a.get("sampling") or {}
        if finish[k] not in ("stop", "length"):
            raise ValueError(f"request {ids[k]}: finish {finish[k]!r}; the replay models stop and length only")
        if sp.get("stop"):
            raise ValueError(f"request {ids[k]}: stop strings finish from the frontend, outside the scheduler")
        max_tokens = sp.get("max_tokens")
        toks = tuple(tokens[ids[k]]) if tokens is not None and ids[k] in tokens else None
        requests.append(RequestIO(prompt_len=int(a["prompt_len"]), max_tokens=int(max_tokens) if max_tokens is not None
                                  else engine.max_model_len - int(a["prompt_len"]), output_len=emitted[k],
                                  eos=finish[k] == "stop", tokens=toks,
                                  kept=tuple(kept[k]) if engine.num_speculative_tokens else None))

    def at(r: dict[str, Any], key: str, lag: float) -> float | None:
        if r.get(key) is not None:
            return float(r[key])
        return None if r.get("t") is None else float(r["t"]) + lag

    return Trace(engine=engine, ids=ids, requests=requests, before_step=[int(a["before_step"]) for a in arrivals], steps=steps,
                 passed=[int(a.get("after_empty") or 0) for a in arrivals],
                 emitted=[{index[a]: int(k) for a, k in (r.get("emitted") or {}).items()} for r in steps_recs],
                 emits=[bool(r.get("emitted")) for r in steps_recs],
                 ingress_t=[at(a, "ingress_t", -ingress_lag_s) for a in arrivals],
                 release_t=[at(r, "release_t", release_lag_s) for r in steps_recs], header=recs[0])


def records(path: str | Path) -> list[dict[str, Any]]:
    """A step trace's records, from JSON lines or gzipped JSON lines (`.gz`)."""
    p = Path(path)
    text = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_step_trace(path: str | Path, **kw: Any) -> Trace:
    return load_step_trace(records(path), **kw)


def span(records: Iterable[dict[str, Any]], mine: Callable[[str], bool]) -> list[dict[str, Any]]:
    """The records of the requests `mine` picks out of a longer serve's step trace, as a trace of their own from step 0:
    for a span the engine began with nothing running or in flight and an empty prefix cache, the state the replay starts
    from. The empty schedules the engine made before the span's first arrival are the serve's before it, so the arrivals
    before the span's first step count theirs from the first arrival's. Refuses a span whose steps are not consecutive or
    hold another request's rows: such a span is not a serve of its own."""
    recs = list(records)
    arrive = [r for r in recs[1:] if r["kind"] == "arrive" and mine(r["request"])]
    if not arrive:
        raise ValueError("none of the trace's arrivals is the span's")
    first, empty = int(arrive[0]["before_step"]), int(arrive[0].get("after_empty") or 0)
    steps = [r for r in recs[1:] if r["kind"] == "step" and any(mine(row[0]) for row in r["rows"])]
    if [r["step"] for r in steps] != list(range(first, first + len(steps))):
        raise ValueError(f"the span's steps are not {first}..{first + len(steps) - 1} each once")
    for r in steps:
        named = [row[0] for row in r["rows"]] + [i for k in ("emitted", "finished", "preempted") for i in r.get(k) or ()]
        if not all(map(mine, named)):
            raise ValueError(f"step {r['step']} holds requests outside the span")
    out = recs[:1]
    for r in arrive:
        b = int(r["before_step"]) - first
        out.append({**r, "before_step": b, "after_empty": int(r.get("after_empty") or 0) - (0 if b else empty)})
    out += [{**r, "step": r["step"] - first} for r in steps]
    out += [{**r, "after_step": r["after_step"] - first} for r in recs[1:] if r["kind"] == "finish" and mine(r["request"])]
    return out


def _jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def read_corpus_trace(d: str | Path, *, num_gpu_blocks: int | None = None, block_size: int | None = None,
                      egress: str = "engine", ingress: str = "client") -> Trace:
    """One trace of the trace corpus (`benchmarks/network_e2e/traces`, store kind `vllm-trace-corpus/v1`): its
    directory's `steps.jsonl`, `requests.jsonl` and `meta.json` (the server's argv), and the KV pool where not given: its
    block size and block count from `engine.json` (the hook's record of the scheduler's), else 16-token blocks and the
    count `server.log`'s KV cache size gives. A pool of more than one KV cache group is refused: the replay models one.
    The hook records no arrivals, so a request's engine order and admission step are its first scheduling (FCFS schedules
    new requests in arrival order), and a step's release is the time its output was back. With `egress="client"` each
    kept output left when the client got the streamed chunk that carried it (its `chunks`), which is what an egress
    certifier on the link sees.

    A request's certified ingress is the client's send time (`ingress="client"`, a passive certifier on the link),
    or with `ingress="warden"` the server's own arrival stamp (`arrival`, its body read), the order vLLM schedules in.
    That is not what design v3's warden certifies: behind it the engine still follows the body read, and the warden's
    release order (a, ordinal) leaves order exceptions to charge (the corpus's `<trace>.released` traces carry it as
    their ingress). `ingress="warden"` is the floor a server admitting in release order at the next bucket would give.

    A trace served under ``--admit tick`` (meta.json's ``tick``: τ and the certifier run's origin) has each request's
    release number and bucket from its network certifier, a warden trace's warden or the corpus's stand-in for it
    (`requests.jsonl`'s ``release`` and ``bucket``), as `Trace.releases`."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    argv = meta["server_argv"]

    def opt(name: str) -> str | None:
        return argv[argv.index(name) + 1] if name in argv else None

    pool = json.loads((d / "engine.json").read_text()) if (d / "engine.json").exists() else {}
    if (pool.get("kv_cache_groups") or 1) != 1:
        raise ValueError(f"{d / 'engine.json'}: {pool['kv_cache_groups']} KV cache groups; the replay models one pool")
    if block_size is None:
        block_size = int(pool.get("block_size") or 16)
    if num_gpu_blocks is None and pool.get("num_gpu_blocks"):
        num_gpu_blocks = int(pool["num_gpu_blocks"])
    if num_gpu_blocks is None:
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", (d / "server.log").read_text(errors="replace"))
        if m is None:
            raise ValueError(f"{d / 'server.log'}: no 'GPU KV cache size' line; give num_gpu_blocks")
        L = int(opt("--max-model-len") or 0)
        # vLLM logs int(num_blocks / cdiv(L, block_size) * L) (`get_kv_cache_capacity`), which this inverts exactly
        num_gpu_blocks = cdiv(int(m.group(1).replace(",", "")) * cdiv(L, block_size), L)
    engine = Engine(max_num_seqs=int(opt("--max-num-seqs") or 0), max_num_batched_tokens=int(opt("--max-num-batched-tokens") or 0),
                    max_model_len=int(opt("--max-model-len") or 0), num_gpu_blocks=num_gpu_blocks, block_size=block_size,
                    enable_chunked_prefill="--no-enable-chunked-prefill" not in argv,
                    enable_prefix_caching="--enable-prefix-caching" in argv,
                    async_scheduling="--no-async-scheduling" not in argv,
                    long_prefill_token_threshold=int(opt("--long-prefill-token-threshold") or 0))
    raw = [s for s in _jsonl(d / "steps.jsonl") if s["total_num_scheduled_tokens"]]
    reqs = {r["id"]: r for r in _jsonl(d / "requests.jsonl")}
    order, seen = [], set()
    for s in raw:
        for r in s["new"]:
            if r["id"] not in seen and not r.get("resumed"):
                seen.add(r["id"])
                order.append(r["id"])
    if missing := [rid for rid in order if rid not in reqs]:
        raise ValueError(f"{len(missing)} scheduled requests the engine never freed (unfinished at the drain?): {missing[:4]}")
    index = {rid: k for k, rid in enumerate(order)}
    requests = []
    for rid in order:
        r = reqs[rid]
        if r["finish_reason"] not in ("stop", "length") or (r["finish_reason"] == "stop") != bool(r["eos"]):
            raise ValueError(f"request {rid}: finish {r['finish_reason']!r} (eos {r['eos']}); the replay models EOS and length")
        requests.append(RequestIO(prompt_len=int(r["prompt_len"]), max_tokens=int(r["max_tokens"]), output_len=int(r["output_len"]),
                                  eos=bool(r["eos"]), tokens=tuple(r["prompt_ids"]) + tuple(r["output_ids"])))
    steps, emits, release = [], [], []
    for s in raw:
        rows = {r["id"]: (index[r["id"]], int(r["num_computed_tokens"]), int(r["num_scheduled_tokens"]))
                for r in s["new"] + s["cached"] if r["num_scheduled_tokens"]}
        batch = [rid for rid in s.get("batch") or () if rid in rows]
        ids = batch if sorted(batch) == sorted(rows) else list(rows)
        steps.append(Step(rows=[rows[rid] for rid in ids],
                          new=[index[r["id"]] for r in s["new"] if not r.get("resumed")],
                          resumed=[index[r["id"]] for r in s["new"] + s["cached"] if r.get("resumed")],
                          preempted=[index[rid] for rid in s.get("preempted") or () if rid in index],
                          cache_hit={index[r["id"]]: int(r["num_computed_tokens"]) for r in s["new"]
                                     if not r.get("resumed") and r["num_computed_tokens"]}))
        emits.append(any(s.get("sampled") or ()))
        release.append(s.get("t_output"))
    first = [-1] * len(order)
    for k, st in enumerate(steps):
        for i, _, _ in st.rows:
            if first[i] < 0:
                first[i] = k
    if ingress == "client":
        ingress_t = [reqs[rid].get("send_unix") or reqs[rid].get("arrival") for rid in order]
    elif ingress == "warden":
        ingress_t = [reqs[rid].get("arrival") for rid in order]
    else:
        raise ValueError(f"ingress {ingress!r}: client or warden")
    output_t = None
    if egress == "client":
        output_t = []
        for rid, io in zip(order, requests):
            r = reqs[rid]
            ts = [r["send_unix"] + float(s) for s, n in r.get("chunks") or () for _ in range(int(n))]
            if len(ts) != io.output_len:
                raise ValueError(f"request {rid}: its chunks stream {len(ts)} tokens of {io.output_len}")
            output_t.append(ts)
    elif egress != "engine":
        raise ValueError(f"egress {egress!r}: engine or client")
    releases = None
    if (tk := meta.get("tick")) is not None:
        if unnumbered := [rid for rid in order if reqs[rid].get("release") is None or reqs[rid].get("bucket") is None]:
            raise ValueError(f"{len(unnumbered)} requests of a tick trace without a release number and bucket: "
                             f"{unnumbered[:4]}")
        releases = Releases(Tick(int(tk["bucket_us"]), int(tk["sat_us"]), int(tk["free_us"])), float(tk["origin_unix"]),
                            tuple(int(reqs[rid]["release"]) for rid in order), tuple(int(reqs[rid]["bucket"]) for rid in order))
    return Trace(engine=engine, ids=order, requests=requests, before_step=first, steps=steps, emits=emits,
                 ingress_t=ingress_t, release_t=release, output_t=output_t, releases=releases,
                 header={"kind": "header", "schema": "vllm-trace-corpus/v1", "config": dict(engine.__dict__), "source": str(d),
                         "egress": egress, "ingress": ingress})


def order_bits(trace: Trace) -> float:
    """What the engine's arrival order costs beyond the certified ingress order: per request in engine order, an adaptive
    bit for whether it is the earliest certified ingress not yet admitted, and Elias gamma of its rank among those when
    not (ideal code length, bits)."""
    if not trace.clocked():
        return 0.0
    left = sorted(range(len(trace.requests)), key=lambda i: (trace.ingress_t[i], i))
    model, cost = _KT(), 0.0
    for i in range(len(trace.requests)):
        rank = left.index(i)
        f0, f1 = model.freqs(())
        cost -= math.log2((f1 if rank == 0 else f0) / (f0 + f1))
        model.update((), rank == 0)
        if rank:
            cost += gamma_bits(rank)
        left.pop(rank)
    return cost


# --- the residual advice ----------------------------------------------------------------------------------------------------


def bucket(t: float, bucket_s: float) -> int:
    return math.floor(t / bucket_s + 1e-9)


@dataclass(frozen=True)
class Decision:
    """One open step's question for the next request: had it arrived (`arrived`)? `passed` counts the request's earlier
    open steps, at which it hadn't."""

    request: int
    step: int
    arrived: bool
    passed: int


def decisions(trace: Trace) -> tuple[list[Decision], Replay]:
    """The trace's decisions in the order the replay asks them (by step, then request), from each request's first
    scheduled step. Refuses a trace the replay doesn't reproduce."""
    first = trace.first_scheduled()
    if any(f < 0 for f in first):
        raise ValueError(f"requests never scheduled: {[trace.ids[i] for i, f in enumerate(first) if f < 0][:8]}")
    rp = replay(trace.engine, trace.requests, decide=lambda i, k, _: k >= first[i])
    m = compare(rp.steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch} ({m.matched}/{m.steps} steps match)")
    out = [Decision(i, k, not forced and p == len(asked) - 1, p)
           for i, (asked, forced) in enumerate(zip(rp.asked, rp.forced)) for p, k in enumerate(asked)]
    return sorted(out, key=lambda d: (d.step, d.request)), rp


def gamma_bits(v: int) -> int:
    """Elias gamma's length for v >= 1."""
    return 2 * v.bit_length() - 1


def gamma_values(rp: Replay) -> list[int | None]:
    """The plainest residual: per request, the open steps it was passed over, sent as Elias gamma of v + 1. A request the
    engine took because it went quiet was passed over at every open step it was asked at (v = their count), and None
    when there were none (0 bits)."""
    return [None if f and not a else len(a) - (0 if f else 1) for a, f in zip(rp.asked, rp.forced)]


def _clocked(cert: Certified) -> tuple[list[int], list[list[int]]]:
    if cert.ingress is None or cert.release is None:
        raise ValueError("the trace has no certifier clock (ingress_t and release_t)")
    return cert.ingress, cert.release


class _Clock:
    """The certified timing a decision is coded in: request i's ingress bucket against the release buckets around the
    step. The engine core scheduled it after processing the outputs it has (those of every step but the one in flight
    under async scheduling) and before any later output left, so the newest output it has processed left at about when it
    scheduled or before (`prev`), and the first output it hasn't, of a request in the engine or of i, after (`next`)."""

    def __init__(self, cert: Certified) -> None:
        self.b_in, self.out = _clocked(cert)

    def context(self, i: int, o: Open) -> tuple[int, int]:
        s = o.sched
        p = max((self.out[r.i][min(r.n_out, len(self.out[r.i])) - 1] for r in s.reqs if r.n_out and self.out[r.i]),
                default=None)
        ahead = [self.out[i][0]] if self.out[i] else []
        for r in (*s.running, *s.waiting, *s.skipped):
            m = r.n_out + r.placeholders
            if m < len(self.out[r.i]):
                ahead.append(self.out[r.i][m])
        n = min(ahead, default=None)
        return (2 if p is None else max(-1, min(1, self.b_in[i] - p)), 2 if n is None else int(self.b_in[i] > n))


class _Egress:
    """Request i's certified egress against the engine run on from the open step (`Open.whatif`), i arriving now or
    before the next step and nothing after it. Under FCFS that run is the served one for i and every request ahead of
    it: a later arrival goes behind them in the running order, so takes none of their budget, and is preempted before
    them. So under the true hypothesis each of i's outputs leaves in the bucket of an earlier request's output from the
    same step, which the certified egress shows; of any of them, since a frontend streams one step's outputs over some
    milliseconds. They are compared up to i's first change of bucket, where some earlier request's must change too (2: no
    earlier request's output to compare with)."""

    def __init__(self, cert: Certified) -> None:
        _, self.out = _clocked(cert)

    def lines_up(self, i: int, em: Emitted, edge: int | None) -> int:
        a, checked = self.out[i], 0
        for e in em:
            t = e.get(i)
            if t is None:
                continue
            bs = {self.out[j][m] for j, m in e.items() if j < i and m < len(self.out[j])}
            if bs:
                if (a[t] not in bs) if edge is None or t < edge else (bs == {a[0]}):
                    return 0
                checked += 1
        return 1 if checked else 2

    def context(self, i: int, o: Open) -> tuple[int, int]:
        a = self.out[i]
        if not a:
            return (2, 2)
        edge = next((t for t in range(1, len(a)) if a[t] != a[t - 1]), None)
        want = len(a) if edge is None else edge + 1
        return (self.lines_up(i, o.whatif(True, want), edge), self.lines_up(i, o.whatif(False, want), edge))


# --- an adaptive binary arithmetic code for the decisions (Witten, Neal and Cleary's coder, KT estimates per context) --

_FULL, _HALF, _QTR = 1 << 32, 1 << 31, 1 << 30


class _KT:
    """Krichevsky-Trofimov counts per context, from `prior`'s where given (pseudo-observations of 0s and 1s)."""

    def __init__(self, prior: dict[Any, tuple[int, int]] | None = None) -> None:
        self.n: dict[Any, tuple[int, int]] = dict(prior or {})

    def freqs(self, ctx: Any) -> tuple[int, int]:
        n0, n1 = self.n.get(ctx, (0, 0))
        return 2 * n0 + 1, 2 * n1 + 1

    def update(self, ctx: Any, bit: bool) -> None:
        n0, n1 = self.n.get(ctx, (0, 0))
        self.n[ctx] = (n0 + (not bit), n1 + bool(bit))


class _Encoder:
    def __init__(self) -> None:
        self.lo, self.hi, self.pending, self.count = 0, _FULL - 1, 0, 0
        self.bits: list[int] = []

    def put(self, bit: bool, f0: int, f1: int) -> None:
        split = self.lo + (self.hi - self.lo + 1) * f0 // (f0 + f1)
        if bit:
            self.lo = split
        else:
            self.hi = split - 1
        self.count += 1
        while True:
            if self.hi < _HALF:
                self._emit(0)
            elif self.lo >= _HALF:
                self._emit(1)
                self.lo -= _HALF
                self.hi -= _HALF
            elif self.lo >= _QTR and self.hi < 3 * _QTR:
                self.pending += 1
                self.lo -= _QTR
                self.hi -= _QTR
            else:
                break
            self.lo, self.hi = 2 * self.lo, 2 * self.hi + 1

    def _emit(self, b: int) -> None:
        self.bits.append(b)
        self.bits.extend([1 - b] * self.pending)
        self.pending = 0

    def finish(self) -> list[int]:
        if self.count:
            self.pending += 1
            self._emit(0 if self.lo < _QTR else 1)
        return self.bits


class _Decoder:
    def __init__(self, bits: Sequence[int]) -> None:
        self.bits, self.pos, self.lo, self.hi, self.code = bits, 0, 0, _FULL - 1, 0
        for _ in range(32):
            self.code = 2 * self.code + self._next()

    def _next(self) -> int:
        b = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def get(self, f0: int, f1: int) -> bool:
        split = self.lo + (self.hi - self.lo + 1) * f0 // (f0 + f1)
        bit = self.code >= split
        if bit:
            self.lo = split
        else:
            self.hi = split - 1
        while True:
            if self.hi < _HALF:
                pass
            elif self.lo >= _HALF:
                self.lo -= _HALF
                self.hi -= _HALF
                self.code -= _HALF
            elif self.lo >= _QTR and self.hi < 3 * _QTR:
                self.lo -= _QTR
                self.hi -= _QTR
                self.code -= _QTR
            else:
                break
            self.lo, self.hi, self.code = 2 * self.lo, 2 * self.hi + 1, 2 * self.code + self._next()
        return bit


def windows(trace: Trace, *, bucket_s: float, window_buckets: int) -> tuple[list[int], list[int]]:
    """Each request's certifier window (its ingress bucket's) and each step's (its release bucket's); one window without a
    clock."""
    if not trace.clocked():
        return [0] * len(trace.requests), [0] * len(trace.steps)
    o = trace.origin(bucket_s)
    return ([(bucket(t, bucket_s) - o) // window_buckets for t in trace.ingress_t],   # type: ignore[arg-type]
            [(bucket(t, bucket_s) - o) // window_buckets for t in trace.release_t])   # type: ignore[arg-type]


@dataclass
class Code:
    """The residual advice under the adaptive code: one bit string per certifier window (a request's decisions go in its
    ingress window's), and what an ideal coder would spend (`ideal`, bits)."""

    contexts: str                   # one of CONTEXTS
    bucket_s: float
    window_buckets: int
    bits: dict[int, list[int]]
    ideal: dict[int, float]


#: the adaptive code's contexts: `plain`, the request's passes so far (`min(passed, 2)`); `clocked`, those and its ingress
#: bucket against the release buckets around the step (`_Clock`); `egress`, whether its release buckets line up with
#: the replay's forks where it arrived now or a step later (`_Egress`), and the clock's where those can't tell
CONTEXTS = ("plain", "clocked", "egress")
#: the egress verdicts that tell the decision (`_Egress.context`: lines up now, a step later), each a prior of `_PRIOR`
#: observations of its decision in the adaptive code
_TELLS = {(1, 0): True, (0, 1): False, (0, 0): False}
_PRIOR = 8


def _model(contexts: str) -> _KT:
    return _KT({c: (0, _PRIOR) if v else (_PRIOR, 0) for c, v in _TELLS.items()} if contexts == "egress" else None)


def _coder(cert: Certified, contexts: str) -> Callable[[int, int, Open], tuple[int, ...]]:
    if contexts not in CONTEXTS:
        raise ValueError(f"contexts {contexts!r}: one of {CONTEXTS}")
    clock = _Clock(cert) if contexts != "plain" else None
    egress = _Egress(cert) if contexts == "egress" else None

    def context(i: int, passed: int, o: Open) -> tuple[int, ...]:
        if clock is None:
            return (min(passed, 2),)
        if egress is None:
            return (min(passed, 2),) + clock.context(i, o)
        e = egress.context(i, o)
        return e if e in _TELLS else e + clock.context(i, o)

    return context


def encode(trace: Trace, *, contexts: str = "egress", bucket_s: float = 0.1, window_buckets: int = 600) -> Code:
    """Code each decision, as the replay asks it, with the adaptive binary code in `contexts`. Refuses a trace the replay
    doesn't reproduce."""
    first = trace.first_scheduled()
    if any(f < 0 for f in first):
        raise ValueError(f"requests never scheduled: {[trace.ids[i] for i, f in enumerate(first) if f < 0][:8]}")
    cert = trace.certified(bucket_s)
    context = _coder(cert, contexts)
    rw = cert.windows(window_buckets)
    model, enc, ideal = _model(contexts), {}, {}
    passed = [0] * len(trace.requests)

    def decide(i: int, k: int, o: Open) -> bool:
        bit = k >= first[i]
        ctx = context(i, passed[i], o)
        f0, f1 = model.freqs(ctx)
        w = rw[i]
        enc.setdefault(w, _Encoder()).put(bit, f0, f1)
        ideal[w] = ideal.get(w, 0.0) - math.log2((f1 if bit else f0) / (f0 + f1))
        model.update(ctx, bit)
        passed[i] += 1
        return bit

    m = compare(replay(trace.engine, trace.requests, decide=decide).steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch} ({m.matched}/{m.steps} steps match)")
    return Code(contexts, bucket_s, window_buckets, {w: e.finish() for w, e in enc.items()}, ideal)


def decode(cert: Certified, code: Code) -> Replay:
    """Replay from the residual advice and what the verifier holds without it (`Certified`, at the code's bucket)."""
    if cert.bucket_s != code.bucket_s:
        raise ValueError(f"the code is in {code.bucket_s} s buckets, the certified record in {cert.bucket_s} s")
    context = _coder(cert, code.contexts)
    rw = cert.windows(code.window_buckets)
    model, dec = _model(code.contexts), {}
    passed = [0] * len(cert.requests)

    def decide(i: int, k: int, o: Open) -> bool:
        ctx = context(i, passed[i], o)
        f0, f1 = model.freqs(ctx)
        w = rw[i]
        if w not in dec:
            dec[w] = _Decoder(code.bits.get(w, []))
        bit = dec[w].get(f0, f1)
        model.update(ctx, bit)
        passed[i] += 1
        return bit

    return replay(cert.engine, cert.requests, decide=decide)


# --- bits per certifier window ----------------------------------------------------------------------------------------


def width(n: int) -> int:
    """Bits for a value in [0, n)."""
    return math.ceil(math.log2(n)) if n > 1 else 0


def v0_widths(trace: Trace, *, window_buckets: int) -> dict[str, int]:
    """I1 v0's fields at fixed width (a stand-in for `advice.bits`): per request its id (an index), prompt_len,
    output_len and finish; per step its row count and release bucket; per row its request, kind, start, n, cache_hit and
    preempted."""
    e = trace.engine
    R = max(len(trace.requests), 1)
    req = width(R) + width(e.max_model_len + 1) + width(max((r.max_tokens for r in trace.requests), default=1) + 1) + 1
    row = width(R) + 1 + width(e.max_model_len) + width(e.max_num_batched_tokens + 1) + width(e.max_model_len) + 1
    step = width(e.max_num_seqs + 1) + width(window_buckets)
    return {"request": req, "row": row, "step": step}


def v0_advice(trace: Trace, v0: Any, *, bucket_s: float = 0.1) -> Any:
    """The trace as advice v0, built with its module `v0` (`verity.protocols.certifier.verifier.advice`): the
    trace's rows in its batch order, each hit's sources from the replay's block map, and each kept output released in its
    step's release bucket. v0 validates it as it builds it."""
    rp = replay(trace.engine, trace.requests, admit=trace.before_step, passed=trace.passed)
    m = compare(rp.steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch}, so it has no sources to give")
    first = trace.first_scheduled()
    order = sorted(range(len(trace.requests)), key=lambda i: (first[i], [r[0] for r in trace.steps[first[i]].rows].index(i)))
    idx = {i: k for k, i in enumerate(order)}
    reqs = tuple(v0.RequestAdvice(id=trace.ids[i], prompt_len=trace.requests[i].prompt_len,
                                  output_len=trace.requests[i].output_len, finish="eos" if trace.requests[i].eos else "length")
                 for i in order)
    kept, times, o = trace.kept_steps(), trace.output_times(), trace.origin(bucket_s)
    steps, release, at = [], [], [0] * len(reqs)
    for s, (st, mine) in enumerate(zip(trace.steps, rp.steps)):
        rows = []
        for i, start, n in st.rows:
            q = reqs[idx[i]]
            rows.append(v0.Row(request=idx[i], kind="decode" if n == 1 and start >= q.prompt_len else "prefill", start=start,
                               n=n, cache_hit=mine.cache_hit.get(i, 0), preempted=i in mine.resumed,
                               sources=tuple(v0.Source(idx[o], k) for o, k in mine.sources.get(i, ()))))
            if at[i] < len(kept[i]) and kept[i][at[i]] == s:
                t = times[i][at[i]]
                at[i] += 1
                release.append(0 if t is None else bucket(t, bucket_s) - o)
        steps.append(v0.StepAdvice(rows=tuple(rows)))
    return v0.Advice(requests=reqs, steps=tuple(steps), release=tuple(release))


def v0_parts(v0: Any, adv: Any) -> dict[str, int] | None:
    """I1 v0's bits by record (`header`, `requests`, `steps`, `release`), from the advice module's layout (`_layout`,
    None without one). The residual advice replaces only the step records: the verifier reads the rest from the
    certified record."""
    layout = getattr(v0, "_layout", None)
    if layout is None:
        return None
    lay = layout(adv)
    return {"header": lay.header, "requests": sum(lay.requests), "steps": sum(lay.steps),
            "release": lay.release * len(adv.release)}


def framed_bits(n: int) -> int:
    """A window's residual of `n` bits with the advice module's `uvar` length prefix (a u6 width, then the length), for
    a transport that doesn't frame windows."""
    return n + 6 + n.bit_length()


def table(trace: Trace, *, bucket_s: float = 0.1, window_buckets: int = 600, v0: Any = None) -> dict[str, Any]:
    """Per certifier window: I1 v0's bits (`advice.bits_by_window` when the module is given as `v0`, else
    `v0_widths`), the steps the replay matched, and the residual advice's bits under each code (Elias gamma; the adaptive
    code, unclocked and clocked), each code checked by decoding it back to the trace's steps."""
    truth = replay(trace.engine, trace.requests, admit=trace.before_step, passed=trace.passed)
    adv = v0_advice(trace, v0, bucket_s=bucket_s) if v0 is not None else None
    charge = v0.bits_by_window(adv, window_buckets) if adv is not None else None
    ds, rp = decisions(trace)
    g = gamma_values(rp)
    codes = {"adaptive" if c == "plain" else f"adaptive_{c}": encode(trace, contexts=c, bucket_s=bucket_s,
                                                                     window_buckets=window_buckets)
             for c in CONTEXTS if c == "plain" or trace.clocked()}
    cert = trace.certified(bucket_s)
    decoded = {name: compare(decode(cert, c).steps, trace.steps).__dict__ for name, c in codes.items()}
    rw, sw = windows(trace, bucket_s=bucket_s, window_buckets=window_buckets)
    v0w = v0_widths(trace, window_buckets=window_buckets)
    rows = []
    for w in sorted(set(rw) | set(sw)):
        ks = [k for k, x in enumerate(sw) if x == w]
        rs = [i for i, x in enumerate(rw) if x == w]
        row = {"window": w, "requests": len(rs), "steps": len(ks), "rows": sum(len(trace.steps[k].rows) for k in ks),
               "matched": sum(1 for k in ks if k < len(truth.steps) and truth.steps[k].key() == trace.steps[k].key()),
               "forced": sum(1 for i in rs if rp.forced[i]), "decisions": sum(1 for d in ds if rw[d.request] == w),
               "v0_bits": charge.get(w, 0) if charge is not None else
                          len(rs) * v0w["request"] + sum(v0w["step"] + len(trace.steps[k].rows) * v0w["row"] for k in ks),
               "gamma_bits": sum(gamma_bits(g[i] + 1) for i in rs if g[i] is not None)}
        for name, c in codes.items():
            row[f"{name}_bits"] = len(c.bits.get(w, []))
            row[f"{name}_ideal"] = round(c.ideal.get(w, 0.0), 2)
        rows.append(row)
    return {"engine": dict(trace.engine.__dict__), "requests": len(trace.requests), "steps": len(trace.steps),
            "match_before_step": compare(truth.steps, trace.steps).__dict__, "decoded": decoded,
            "v0": "advice.bits_by_window" if charge is not None else {"widths": v0w},
            "v0_parts": v0_parts(v0, adv) if adv is not None else None, "bucket_s": bucket_s,
            "window_buckets": window_buckets, "windows": rows, "order_bits": round(order_bits(trace), 2),
            "passed": [x for x in g if x is not None]}


def kept_bits(trace: Trace) -> dict[str, Any]:
    """With a draft, what the drafts each row kept (`RequestIO.kept`) cost as advice, both ways. `charged`: the count
    sent for every row that verified drafts and emitted, at its measured entropy over those rows (`entropy`, bits a
    row, at most `max`, log2(K + 1)). `derived`: 0, where the audited block computes the count from committed values and
    nothing is sent. `counts[k]` is the rows that kept k. The rows and the drafts each carried are the replay's, so it
    refuses a trace the replay doesn't reproduce."""
    K = trace.engine.num_speculative_tokens
    if not K:
        raise ValueError("the trace's engine serves with no draft, so no row keeps one")
    if trace.emitted is None:
        raise ValueError("the drafts a row kept are the trace's `emitted` less one, which this trace does not record")
    rp = replay(trace.engine, trace.requests, admit=trace.before_step, passed=trace.passed)
    m = compare(rp.steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch} ({m.matched}/{m.steps} steps match)")
    counts = [0] * (K + 1)
    for st, em in zip(rp.steps, trace.emitted):
        for i, d in st.drafts.items():
            if d and em.get(i):
                counts[em[i] - 1] += 1
    rows = sum(counts)
    entropy = -sum(c / rows * math.log2(c / rows) for c in counts if c) if rows else 0.0
    return {"K": K, "rows": rows, "counts": counts, "entropy": entropy, "max": math.log2(K + 1), "charged": rows * entropy,
            "derived": 0}


def draft_table(trace: Trace, *, contexts: Sequence[str] = ("plain",), bucket_s: float = 0.1,
                window_buckets: int = 600) -> dict[str, Any]:
    """A trace served with a draft, as the two classes of advice its replay takes, in bits over the trace: `arrivals`,
    the residual advice under each of `contexts` (`encode`, each decoded back to the trace's steps), which a serve under
    the certified tick does not send; and `kept_count` (`kept_bits`). Refuses a trace the replay doesn't reproduce."""
    kept = kept_bits(trace)
    cert = trace.certified(bucket_s)
    arrivals = {}
    for c in contexts:
        code = encode(trace, contexts=c, bucket_s=bucket_s, window_buckets=window_buckets)
        back = compare(decode(cert, code).steps, trace.steps)
        if back.first_mismatch is not None:
            raise ValueError(f"the residual advice ({c}) decodes to another schedule from step {back.first_mismatch}")
        arrivals[c] = sum(len(b) for b in code.bits.values())
    return {"requests": len(trace.requests), "steps": len(trace.steps), "tick": trace.releases is not None,
            "arrivals": arrivals, "kept_count": kept}


#: advice v0's module (`verity.protocols.certifier.verifier.advice`)
V0_PATH = Path(__file__).resolve().parents[3] / "verity" / "protocols" / "certifier" / "verifier" / "advice.py"


def load_v0(spec: str | Path | None) -> Any:
    """The advice module from `spec`, a path to its `advice.py`, else from `V0_PATH` when it is there, else None
    (`table` falls back to `v0_widths`)."""
    import importlib.util
    import sys
    if not spec:
        if not V0_PATH.exists():
            return None
        spec = V0_PATH
    s = importlib.util.spec_from_file_location("advice_v0", spec)
    assert s is not None and s.loader is not None
    sys.modules["advice_v0"] = mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def read_any(p: Path, *, egress: str = "engine", ingress: str = "client") -> Trace:
    if p.is_dir():
        return read_corpus_trace(p, egress=egress, ingress=ingress)
    if egress != "engine" or ingress != "client":
        raise ValueError(f"{p}: a step trace has only its steps' release times (egress engine) and its own ingress")
    tok = p.with_suffix(".tokens.json")
    return read_step_trace(p, tokens=json.loads(tok.read_text()) if tok.exists() else None)


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Bits per certifier window: I1 v0 against the replay's residual advice.")
    ap.add_argument("traces", nargs="+", type=Path,
                    help="step traces (*.jsonl; <name>.tokens.json beside one with prefix caching) or trace-corpus "
                         "trace dirs")
    ap.add_argument("--bucket-ms", type=float, default=100.0)
    ap.add_argument("--window-buckets", type=int, default=600)
    ap.add_argument("--v0", help="path to the advice module's advice.py (default: V0_PATH in the tree, else fixed "
                                 "widths)")
    ap.add_argument("--egress", choices=("engine", "client"), default="engine",
                    help="a trace-corpus output's release time: its step's (engine) or its streamed chunk's (client)")
    ap.add_argument("--ingress", choices=("client", "warden"), default="client",
                    help="a trace-corpus request's certified ingress: the client's send time (client) or the engine's "
                         "arrival stamp (warden: the floor a server admitting in the warden's release order would "
                         "give; the warden certifies the release order, which a trace's .released view carries)")
    ap.add_argument("--out", type=Path, help="write {trace: table} as JSON")
    a = ap.parse_args(argv)
    v0 = load_v0(a.v0)
    out: dict[str, Any] = {}
    per_window = []
    drafted = []
    print("| trace | requests | steps | replay matched | I1 v0 bits | v0 step records | gamma | adaptive | adaptive clocked "
          "| adaptive egress | best bits/request | order bits |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in a.traces:
        name = p.stem if p.is_file() else f"{p.resolve().parent.name}/{p.name}"
        tr = read_any(p, egress=a.egress, ingress=a.ingress)
        if tr.engine.num_speculative_tokens:
            # I1 v0 has no row for a draft's, so a trace served with one has its own table
            try:
                d = out[name] = draft_table(tr, contexts=[c for c in CONTEXTS if c == "plain" or tr.clocked()],
                                            bucket_s=a.bucket_ms / 1000, window_buckets=a.window_buckets)
            except (ValueError, RuntimeError) as e:
                out[name] = {"error": str(e), "requests": len(tr.requests), "steps": len(tr.steps)}
                drafted.append(f"| {name} | {len(tr.requests)} | {len(tr.steps)} | {e} |" + " |" * 7)
                continue
            k, n = d["kept_count"], max(d["requests"], 1)
            drafted.append(f"| {name} | {d['requests']} | {d['steps']} | {k['K']} | "
                           f"{' / '.join(str(d['arrivals'].get(c, '-')) for c in CONTEXTS)} | {k['rows']} | "
                           f"{k['entropy']:.3f} | {k['max']:.3f} | {k['charged']:.0f} | {k['charged'] / n:.2f} | "
                           f"{k['derived']} |")
            continue
        try:
            tb = out[name] = table(tr, bucket_s=a.bucket_ms / 1000, window_buckets=a.window_buckets, v0=v0)
        except (ValueError, RuntimeError) as e:
            try:
                m = compare(replay(tr.engine, tr.requests, admit=tr.before_step, passed=tr.passed).steps, tr.steps)
                where = f"{m.matched}/{m.steps} (first mismatch {m.first_mismatch})"
                match: dict[str, Any] = m.__dict__
            except (ValueError, RuntimeError) as e2:
                where, match = f"replay refused: {e2}", {"error": str(e2)}
            out[name] = {"error": str(e), "requests": len(tr.requests), "steps": len(tr.steps), "match_before_step": match}
            print(f"| {name} | {len(tr.requests)} | {len(tr.steps)} | {where} | {e} |" + " |" * 7, flush=True)
            continue
        s = {k: sum(w[k] for w in tb["windows"] if w.get(k) is not None) for k in tb["windows"][0]}
        m = tb["match_before_step"]
        best = min(s[k] for k in ("adaptive_bits", "adaptive_clocked_bits", "adaptive_egress_bits") if k in s)
        steps = tb["v0_parts"]["steps"] if tb["v0_parts"] else "-"
        print(f"| {name} | {tb['requests']} | {tb['steps']} | {m['matched']}/{m['steps']} | "
              f"{s['v0_bits']} | {steps} | {s['gamma_bits']} | {s['adaptive_bits']} | {s.get('adaptive_clocked_bits', '-')} | "
              f"{s.get('adaptive_egress_bits', '-')} | {best / max(tb['requests'], 1):.2f} | {tb['order_bits']} |", flush=True)
        code = "adaptive_egress_bits" if "adaptive_egress_bits" in s else "adaptive_bits"
        for w in tb["windows"]:
            per_window.append(f"| {name} | {w['window']} | {w['requests']} | {w['steps']} | {w['matched']}/{w['steps']} | "
                              f"{w['decisions']} | {w['forced']} | {w['v0_bits']} | {w[code]} | {framed_bits(w[code])} |")
    print("\n| trace | window | requests | steps | replay matched | decisions | forced | I1 v0 bits | residual bits "
          "| residual framed |\n|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(per_window))
    if drafted:
        print("\n| trace, served with a draft | requests | steps | K | arrivals: bits (" + " / ".join(CONTEXTS) + ") "
              "| rows that verified drafts | kept count: entropy, bits a row | at most | charged, bits | a request "
              "| derived, bits |\n|---|---|---|---|---|---|---|---|---|---|---|")
        print("\n".join(drafted))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

