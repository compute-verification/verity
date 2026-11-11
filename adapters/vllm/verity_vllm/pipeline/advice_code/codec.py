"""The advice code (advice v1): I1 v0 (`verity.protocols.certifier.verifier.advice`) sent as only what the certified record,
the scheduler replay (`replay`) and the derived rules (`derived`) leave, in a prefix-free, self-delimiting code
(`window_code`) under the registered codebook. A registration that selects it has the compiler
(`verity_vllm.program.predict.steps`) read these bits, decoded here, and its cap ℓ bound them.

`encode(advice, record, registered) -> bytes` and `decode(bytes, record, registered) -> I1.Advice`, with
`decode(encode(a)) == a`. `decode` is total: any byte string decodes to one I1 v0 or is refused by name (`AdviceCodeError`),
deterministically, and a string longer than the cap ℓ (`Registered.cap_bits`) is refused unread.

**Certified, 0 bits** (`Record`, from the network certifier's signed records; the compile unit holds them as witness):
- the network certifier's connection ordinals, in the order its own schedule admitted them (`Record.requests`);
- each request's ingress bytes, an id request's fields (`id_request_fields`): its `user` (I1's id), its `prompt` ids and
  its `max_tokens`; and its release, the bucket a of its connection's last ingress cell (`Request.end`), from which the
  server holds its whole body;
- each request's output ids, its id events' `token_id`s in order, and each output's release bucket (`Request.release`),
  which the network certifier's fixed schedule sets (egress cell i in bucket a + D + ⌊i/r⌋).
The reference order is the network certifier's release order, the requests by (a, ordinal) (`release_order`).

**Sent, the order**: the engine doesn't admit in that order. It admits in the API server's body-read order, which
comes after the release and overtakes it under load. So the engine's arrival order (`arrival_order`, recovered from the
advice) is sent as the permutation from release order to it (`_deviations`): for each arrival that isn't the earliest
released request not yet arrived, its position and its rank among those, in `window_code`'s `order` field as (position,
(rank,)). `streams` (`Decoded.streams`) is that permutation as the I/O tie's egress side reads it: each I1 request's
network certifier ordinal.

**Replayed, 0 bits**: `replay.replay` runs vLLM's scheduler over the requests in arrival order under the
registered engine (`Registered.engine`, its KV pool `num_gpu_blocks` included), and gives every step's rows (request,
start, n), new and resumed requests, cache hits (its block pool's LRU, eviction included) and preemptions, from one bit
per open step: whether the next request had arrived.

**Derived, 0 bits**: the batch order (`Registered.row_order`: Model Runner V2's stable sort of the scheduled rows by n,
or the scheduler's own order), I1's numbering by first row, and the derived rules' walk (`derived.walk_steps`) over the
replayed rows, which recomputes every start and n by vLLM's rules and checks them against the replay, holds each hit
under the cumulative pool bound to the longest cached prefix, and gives `kind`, the canonical `sources`, and the request
records (the tie: prompt and output lengths against the certified ids, `finish` by the stop rule).

**Sent**, as `window_charge.Window`s (`residual`), and nothing else:
- `arrived`: the replay's decisions, one per open step: whether the next request had arrived.
- `order`: the order's deviations from release order, as above.
- `hits`: past the pool bound, every prefix-cache lookup the replay's scheduler reads (`pool.BlockPool.hit`: each candidate its
  waiting loop reaches, the one whose allocation ends the loop included, and `would_admit`'s for an open step's
  arrival), in the order it reads them, as its shortfall: the longest prefix a cache that never evicts holds
  (`derived.Prefixes.cached` over the rows scheduled before the lookup) less the replay's hit, in whole blocks. Where
  the bound holds (`derived.walk_steps`'s: sync scheduling, and the blocks of the requests admitted through the step at most
  the pool less its null block) the shortfall is 0 by the bound and is derived at 0 bits. Past it, the hit is the
  prover's choice of eviction, so it is sent and charged, and no choice goes uncharged (the sampled leak bound's `hAs`).
The resumes and the order of new requests that `derived.walk_steps` would otherwise read are the replay's, so v1 sends none.
The KV pool's size is registered before the run, never sent: a prover that chose it could declare a bigger pool and buy
derived hits with it.

**Sent, the lengths** (`Registered.send_lengths`, `lengths`): while no proof holds the length rule, each request's
output_len and finish, as the plain tail of the window its body lands in, held to the certified ids and the stop rule.

**Under the certified tick** (`Registered.tick`, an engine served with ``--admit tick``), the engine adds each request
once a logical clock has passed its release bucket's close, in release order, and the replay runs that rule from the
record's `end`s and the registered τ (`replay.replay`'s ``tick``). It asks nothing, so there are no decisions, and v1
requires the arrival order to be release order, so there are no order entries. The registered caps hold neither
(`TICK_CAPS`, and a registered book's caps must too), so a window's timing fields code in 0 bits and it sends only its
hits.

**The windows** (the certifier PROTOCOL's A5: no later than the first window whose rows the value changes). Step s's
window is the earliest window an output at step s or later is released in (the last window where none follows). An
arrival's ingress window is the earliest release window of it and every later arrival, so that it never decreases
where the arrivals deviate from release order. A decision about arrival i at step k goes to the earlier of step k's
window and i's ingress window; an order entry for arrival p to the earlier of p's ingress window and the window of the
first step that reads p (a decision about it, a lookup of it, or its admission); and a sent hit at step s to step s's
window. Each is nondecreasing in the replay's order, so each field's values, window after window, are the order the
replay reads them in, and the decoder replays them straight through.

**The string**, MSB first, then zero bits to a whole byte: `window_code.encode(the windows, the registered Book)`, with W the
record's windows (`windows`: its last bucket's, plus one). `window_code` codes each window's counts before its values and
terminates each window's segment, so the code finds every window's end itself, with no length prefix. A window's charge
is its segment's bits (`window_code.Encoded.lens`; `bits_by_window`), a function of the string alone, and its length
terms', whose widths the record's `max_tokens` fix.
The strings are prefix-free for every W, so a string
charged c bits in all is one of at most 2^c (Kraft; the sampled leak bound's `hAs`). The codebook (`Registered.book`,
`window_code.Book`) is fit on other runs, never this one, and its digest is registered before the first window. `decode`
takes only the canonical string (`window_code.decode`: each segment the coder's own codeword for the symbols it decodes to),
zero padding of fewer than 8 bits and no byte past it, and then holds the windows to the replay's: the decisions the
replay asks, in order, and the hits and windows it gives; and the order to the one `arrival_order` recovers from the
advice it decodes to, so that each advice has one string.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, TypeVar

from verity.protocols.certifier.verifier import advice as I1
from verity_vllm.pipeline.advice_code import arith as C
from verity_vllm.pipeline.advice_code import derived as D
from verity_vllm.pipeline.advice_code import lengths as LN
from verity_vllm.pipeline.advice_code import pool as P
from verity_vllm.pipeline.advice_code import replay as R
from verity_vllm.pipeline.advice_code import schedule as S
from verity_vllm.pipeline.advice_code import window_code as CV

__all__ = ["advice_of_replay", "AdviceCodeError", "arrival_order", "bits_by_window", "book_of", "CAPS", "charged",
    "check_record", "Decision", "decode", "decode_bits", "decode_full", "Decoded", "encode", "length_tails", "length_widths",
    "lengths", "Lookup", "Record", "Registered", "release_order", "replay_record", "Request", "residual", "ROW_ORDERS",
    "streams", "string_of", "TICK_CAPS", "windows"]

T = TypeVar("T")

ROW_ORDERS = ("batch", "scheduling")


class AdviceCodeError(ValueError):
    """A string v1 refuses, or an advice or record it can't carry."""


@dataclass(frozen=True)
class Request:
    """One certified request (the module's docstring, "Certified"): its `user` id, prompt ids and `max_tokens`, its
    last-frame mark `end`, and its output ids, each with its release bucket. Buckets count from the session's first."""

    id: str
    prompt: tuple[int, ...]
    max_tokens: int
    output: tuple[int, ...]
    end: int
    release: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("prompt", "output", "release"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.id) is not str or not self.id:
            raise AdviceCodeError(f"a request's id is a non-empty str, not {self.id!r}")
        ints = (self.max_tokens, self.end, *self.prompt, *self.output, *self.release)
        if any(type(x) is not int or x < 0 for x in ints):
            raise AdviceCodeError(f"request {self.id}: its fields are ints ≥ 0")
        if not self.prompt or not self.output or self.max_tokens < 1:
            raise AdviceCodeError(f"request {self.id}: a prompt, an output and max_tokens of at least one")
        if len(self.release) != len(self.output):
            raise AdviceCodeError(f"request {self.id}: {len(self.release)} release buckets for {len(self.output)} outputs")
        if any(b < a for a, b in zip(self.release, self.release[1:])):
            raise AdviceCodeError(f"request {self.id}: its outputs' release buckets decrease")


@dataclass(frozen=True)
class Record:
    """The certified record: the requests in the network certifier's admission order."""

    requests: tuple[Request, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        if not self.requests or any(type(q) is not Request for q in self.requests):
            raise AdviceCodeError("a record is one or more Requests")
        if len({q.id for q in self.requests}) != len(self.requests):
            raise AdviceCodeError("two requests share an id")


def release_order(record: Record) -> list[int]:
    """The certified reference order: the network certifier's release order, each request's last ingress cell landing, by
    (`end`, ordinal)."""
    return sorted(range(len(record.requests)), key=lambda k: (record.requests[k].end, k))


def arrival_order(advice: I1.Advice, record: Record, row_order: str) -> list[int]:
    """The engine's arrival order (ordinals) that v1 replays the advice under. FCFS admits new requests in arrival
    order, so it is by first step, and within a step by the scheduling order. Under the scheduler's row order that is
    the step's own. Under the batch order (a stable sort by n) it is the order that puts a request after any request of
    the step its hit's sources name and the step's partial prefill last, and keeps each equal-n chain's batch order: in
    the step itself, and in each later step among those of its new requests still running since (the scheduler keeps
    its running requests in admission order, and a resumed one goes to the end), read until the order is unique or
    fewer than two are left. Where the rows leave a choice, it takes the earliest in release order."""
    ordinal = {q.id: k for k, q in enumerate(record.requests)}
    if sorted(q.id for q in advice.requests) != sorted(ordinal):
        raise AdviceCodeError("the advice's requests aren't the record's")
    rank = {k: r for r, k in enumerate(release_order(record))}
    of = [ordinal[q.id] for q in advice.requests]
    last: dict[int, int] = {}
    for s, st in enumerate(advice.steps):
        for x in st.rows:
            last[x.request] = s
    seen: set[int] = set()
    out: list[int] = []
    for s, st in enumerate(advice.steps):
        new = [x for x in st.rows if x.request not in seen]
        seen.update(x.request for x in new)
        if row_order == "scheduling" or len(new) < 2:
            out += [of[x.request] for x in new]
            continue
        after: dict[int, set[int]] = {x.request: set() for x in new}

        def chain(rows: Sequence[Any]) -> None:
            by_n: dict[int, list[int]] = {}
            for x in rows:
                by_n.setdefault(x.n, []).append(x.request)
            for c in by_n.values():
                for a, b in zip(c, c[1:]):
                    after[b].add(a)

        chain(new)
        for x in new:
            after[x.request] |= {u.request for u in x.sources if u.request in after and u.request != x.request}
            if x.start + x.n < advice.requests[x.request].prompt_len:
                after[x.request] |= {y.request for y in new if y.request != x.request}
        running, k = {q for q in after if last[q] > s}, s + 1
        while len(running) > 1 and _topo(after, lambda q: rank[of[q]])[1] is False:
            rows = advice.steps[k].rows
            running -= {x.request for x in rows if x.preempted}
            chain([x for x in rows if x.request in running])
            running = {q for q in running if last[q] > k}
            k += 1
        order, _ = _topo(after, lambda q: rank[of[q]])
        if order is None:
            raise AdviceCodeError(f"step {s}: no arrival order of its new requests fits their rows")
        out += [of[q] for q in order]
    if sorted(out) != list(range(len(record.requests))):
        raise AdviceCodeError("the advice's requests aren't the record's")
    return out


def _topo(after: dict[int, set[int]], key: Callable[[int], int]) -> tuple[list[int] | None, bool | None]:
    """The topological order of `after` (each request's predecessors) taking the least `key` among those free to go
    next, and whether it is the only one; (None, None) on a cycle."""
    left = {q: set(a) for q, a in after.items()}
    out: list[int] = []
    unique = True
    while left:
        free = [q for q, a in left.items() if not a]
        if not free:
            return None, None
        q = min(free, key=key)
        unique = unique and len(free) == 1
        out.append(q)
        del left[q]
        for a in left.values():
            a.discard(q)
    return out, unique


def streams(advice: I1.Advice, record: Record) -> tuple[int, ...]:
    """The order permutation as the I/O tie's egress side reads it (`walk.order` through the arrival order): for each I1
    request q, the network certifier's ingress ordinal of the stream that carries it."""
    ordinal = {q.id: k for k, q in enumerate(record.requests)}
    return tuple(ordinal[q.id] for q in advice.requests)


class Decoded(NamedTuple):
    """What a string carries under the record: the advice; `streams`, each I1 request's network certifier ordinal (the I/O tie's
    egress side checks each cell's stream against its request q through it); and the arrival order, the ordinals in the
    order the engine took them."""

    advice: I1.Advice
    streams: tuple[int, ...]
    arrival: tuple[int, ...]


def _deviations(record: Record, arrival: Sequence[int]) -> list[tuple[int, int]]:
    """The permutation from release order to the arrival order, as (position, rank): each arrival that isn't the
    earliest in release order of the requests not yet arrived, and its rank among them."""
    left, out = release_order(record), []
    for p, k in enumerate(arrival):
        r = left.index(k)
        if r:
            out.append((p, r))
        left.pop(r)
    return out


def _arrival(record: Record, deviations: Sequence[tuple[int, int]]) -> list[int]:
    """`_deviations`' inverse; refuses a position or rank past the requests left, and a rank of 0."""
    left, out, at = release_order(record), [], dict(deviations)
    if len(at) != len(deviations) or any(not 0 <= p < len(left) or r < 1 for p, r in deviations):
        raise AdviceCodeError("an order entry names no arrival, or ranks it first")
    for p in range(len(left)):
        r = at.get(p, 0)
        if r >= len(left):
            raise AdviceCodeError(f"arrival {p}: rank {r} among {len(left)} requests left")
        out.append(left.pop(r))
    return out


def windows(record: Record, T: int) -> int:
    """W: the windows the record covers, through its last bucket."""
    return max(max(q.end, *q.release) for q in record.requests) // T + 1


@dataclass(frozen=True)
class Registered:
    """What v1 reads besides the string and the record, fixed before the run: the engine's scheduler config, its KV pool
    `num_gpu_blocks` included (public, never in the advice: a prover that set it could buy derived hits with a bigger
    pool), the stop ids, the runner's row order, the window T in buckets, the advice code's codebook (`window_code.Book`,
    whose digest the registration holds; None for the uniform book), ℓ, the certified tick the engine admits by
    (`replay.Tick`, its τ the engine class's; None for an engine that admits on arrival), and whether the string sends
    and charges each request's output_len and finish (the module's "Sent, the lengths").

    `send_lengths` is on, in every registration (class G's included), until a proof holds both halves of the length
    rule on main: the stop ids, a stop-id rule in each request Program, and the `max_tokens` cap. Until then the
    certified record's output_len is the prover's choice and v1 charges it; a registration that turns it off names
    the proof of each half."""

    engine: Any  # replay.Engine
    stop_ids: tuple[int, ...]
    row_order: str = "batch"
    T: int = 600
    book: Any = None
    cap_bits: int = 1 << 23
    tick: Any = None  # replay.Tick
    send_lengths: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_ids", tuple(self.stop_ids))
        e = self.engine
        if not self.stop_ids or self.row_order not in ROW_ORDERS or self.T < 1:
            raise AdviceCodeError(f"a registration has stop ids, a row order of {ROW_ORDERS} and T ≥ 1")
        if type(self.send_lengths) is not bool:
            raise AdviceCodeError(f"a registration's send_lengths is a bool, not {self.send_lengths!r}")
        if self.book is not None and type(self.book) is not CV.Book:
            raise AdviceCodeError(f"a registration's book is a window_code.Book, not a {type(self.book).__name__}")
        if self.tick is not None:
            if type(self.tick) is not S.Tick:
                raise AdviceCodeError(f"a registration's tick is a replay.Tick, not a {type(self.tick).__name__}")
            if self.book is not None and (self.book.caps.arrived or self.book.caps.order):
                raise AdviceCodeError("under the certified tick the registered book codes no decision and no order entry: its "
                                      "caps' arrived and order are 0")
        if e.max_num_seqs < 1 or e.max_num_batched_tokens < 1 or e.max_model_len < 2 or e.block_size < 1:
            raise AdviceCodeError("the engine schedules at least one sequence and token a step")
        if not e.enable_chunked_prefill and e.max_num_batched_tokens < e.max_model_len:
            raise AdviceCodeError("without chunked prefill the token budget holds a whole max_model_len prompt (vLLM's own check)")

    def decode(self, data: bytes, record: Record, cap_bits: int | None = None) -> I1.Advice:
        """The module's `decode` of `data` against `record` under this registration, at ℓ = `cap_bits` where it is given
        (the compiler's registration's): the advice code as `steps.Registered.code` reads it."""
        if type(record) is not Record:
            raise AdviceCodeError(f"v1 decodes against a certified Record, not a {type(record).__name__}")
        return decode(data, record, self if cap_bits is None else replace(self, cap_bits=cap_bits))

    def config(self, engine: Any) -> Any:
        """The derived rules' `Config` for `engine`."""
        return D.Config(block_size=engine.block_size, num_gpu_blocks=engine.num_gpu_blocks,
                        max_num_batched_tokens=engine.max_num_batched_tokens, max_model_len=engine.max_model_len,
                        eos_token_id=self.stop_ids[0], stop_token_ids=self.stop_ids[1:],
                        scheduler="async" if engine.async_scheduling else "sync",
                        long_prefill_token_threshold=engine.long_prefill_token_threshold,
                        max_num_seqs=engine.max_num_seqs, enable_prefix_caching=engine.enable_prefix_caching,
                        row_order=self.row_order)


def check_record(record: Record, reg: Registered, engine: Any) -> Any:
    """The derived config, once the record and the pool are ones the replay runs to the end on: every output the stop
    rule's, within max_model_len, and the pool holding any one request alone, async's two steps ahead included."""
    cfg = reg.config(engine)
    L, B = engine.max_model_len, engine.block_size
    for q in record.requests:
        if len(q.prompt) + len(q.output) > L or D.stop_rule(q.output, len(q.prompt), q.max_tokens, cfg) is None:
            raise AdviceCodeError(f"request {q.id}: its output isn't one the stop rule ends, within max_model_len {L}")
    least = max(S.cdiv(min(len(q.prompt) + len(q.output) + 2, L), B) for q in record.requests) + 1
    if engine.num_gpu_blocks < least:
        raise AdviceCodeError(f"the registered pool of {engine.num_gpu_blocks} blocks holds no request alone (needs {least})")
    return cfg


# --- the residual ------------------------------------------------------------------------------------------------------

def _ios(record: Record, engine: Any, order: Sequence[int], stops: frozenset[int]) -> list[Any]:
    out = []
    for k in order:
        q = record.requests[k]
        out.append(S.RequestIO(prompt_len=len(q.prompt), max_tokens=q.max_tokens, output_len=len(q.output),
                               eos=q.output[-1] in stops,
                               tokens=q.prompt + q.output if engine.enable_prefix_caching else None))
    return out


@dataclass(frozen=True)
class Lookup:
    """A prefix-cache lookup the replay's scheduler read: a request (in arrival order) with K known tokens, and the hit
    the replay's block pool gave it, in whole blocks."""

    request: int
    K: int
    hit: int


Decision = tuple[int, int, bool]  # (request, step, whether it had arrived)


@contextmanager
def _recording() -> Iterator[list[tuple[list[Lookup], Any]]]:
    """Every lookup the replay's scheduler reads (`BlockPool.hit`: each candidate the waiting loop reaches, the one whose
    allocation ends it included, and `would_admit`'s for an open step's arrival), by `schedule()` call, with the step
    the call made."""
    calls: list[tuple[list[Lookup], Any]] = []
    hit0, schedule0 = P.BlockPool.hit, R.Scheduler.schedule

    def hit(kv: Any, r: Any) -> list[int]:
        h = hit0(kv, r)
        if calls:
            calls[-1][0].append(Lookup(r.i, r.num_tokens, len(h)))
        return h

    def schedule(sched: Any, on_open: Any = None) -> Any:
        calls.append(([], None))
        out = schedule0(sched, on_open)
        calls[-1] = (calls[-1][0], out[0])
        return out

    P.BlockPool.hit, R.Scheduler.schedule = hit, schedule
    try:
        yield calls
    finally:
        P.BlockPool.hit, R.Scheduler.schedule = hit0, schedule0


def replay_record(record: Record, reg: Registered, engine: Any, order: Sequence[int],
                  bit_of: Callable[[int, int], bool]) -> tuple[Any, list[Decision], list[list[Lookup]]]:
    """The replay, each decision `bit_of(request, step)` (under the certified tick, the tick's rule from each request's
    `end`, and no decisions): its steps, its decisions in order, and each executed step's lookups in the order the
    scheduler read them (`would_admit`'s and the waiting loop's own of one request read once; a `schedule()` that
    scheduled nothing is no step)."""
    ios = _ios(record, engine, order, reg.config(engine).stops)
    decisions: list[Decision] = []

    def decide(i: int, k: int, _o: Any) -> bool:
        b = bool(bit_of(i, k))
        decisions.append((i, k, b))
        return b

    with _recording() as calls:
        if reg.tick is not None:
            rp = R.replay(engine, ios, tick=reg.tick, released=[record.requests[k].end for k in order])
        else:
            rp = R.replay(engine, ios, decide=decide)
    looks = [[x for k, x in enumerate(ls) if k == 0 or x != ls[k - 1]] for ls, step in calls
             if step is not None and step.rows]
    if len(looks) != len(rp.steps):
        raise AdviceCodeError(f"{len(looks)} scheduling calls with rows for the replay's {len(rp.steps)} steps")
    return rp, decisions, looks


def _sent(record: Record, engine: Any, order: Sequence[int], rp: Any,
          looks: Sequence[Sequence[Lookup]]) -> list[tuple[int, int]]:
    """The hits v1 sends, as (step, shortfall): every lookup of a step the no-eviction bound doesn't cover, as the
    longest prefix a cache that never evicts holds less the replay's hit, in whole blocks. The bound is
    `derived.walk_steps`'s: sync scheduling, and the blocks of every request admitted so far, this step's included, at most
    the pool less its null block. The cache that never evicts is `derived.Prefixes` over the rows scheduled before the
    lookup, running rows first."""
    if not engine.enable_prefix_caching:
        return []
    B = engine.block_size
    reqs = [record.requests[k] for k in order]
    prefixes = D.Prefixes([q.prompt + q.output for q in reqs], B)
    life = [S.cdiv(len(q.prompt) + len(q.output), B) for q in reqs]
    derive, filled, out = not engine.async_scheduling, 0, []
    for s, (st, ls) in enumerate(zip(rp.steps, looks, strict=True)):
        filled += sum(life[i] for i in st.new)
        derive = derive and filled <= engine.num_gpu_blocks - 1
        rows = {i: (start, start + n) for i, start, n in st.rows}
        admitted = [i for i in rows if i in set(st.new) | set(st.resumed)]
        for i, (start, end) in rows.items():
            if i not in admitted:
                prefixes.add(i, start, end)
        for x in ls:
            if not derive:
                short = prefixes.cached(x.request, x.K) // B - x.hit
                if short < 0:
                    raise AdviceCodeError(f"step {s}: the replay hits {x.hit} blocks for request {x.request}, past the "
                                          f"{x.hit + short} a cache that never evicts holds")
                out.append((s, short))
            if x.request in admitted:
                prefixes.add(x.request, *rows[x.request])
                admitted.remove(x.request)
        for i in admitted:
            prefixes.add(i, *rows[i])
    return out


def charged(record: Record, reg: Registered, order: Sequence[int], rp: Any, W: int) -> tuple[list[int], list[int]]:
    """A5's windows, as I1's `bits_by_window` charges: each step's, the earliest window an output at that step or later
    is released in (the last window where none follows); and each arrival's ingress window, the earliest of its own
    and every later arrival's last frame's, so that it never decreases where the arrivals deviate from release order."""
    reqs = [record.requests[k] for k in order]
    kept, own = [0] * len(reqs), []
    for st in rp.steps:
        w = None
        for i, start, n in st.rows:
            q = reqs[i]
            if start + n == len(q.prompt) + kept[i] and kept[i] < len(q.output):
                b = q.release[kept[i]] // reg.T
                w = b if w is None else min(w, b)
                kept[i] += 1
        own.append(w)
    earliest, at = [W - 1] * len(own), None
    for s in range(len(own) - 1, -1, -1):
        if own[s] is not None:
            at = own[s] if at is None else min(at, own[s])
        if at is not None:
            earliest[s] = at
    win = [q.end // reg.T for q in reqs]
    for i in range(len(win) - 2, -1, -1):
        win[i] = min(win[i], win[i + 1])
    return earliest, win


def _residual(record: Record, reg: Registered, engine: Any, order: Sequence[int], rp: Any,
              decisions: Sequence[Decision], looks: Sequence[Sequence[Lookup]], W: int) -> list[Any]:
    """The windows v1 sends (`window_charge.Window`). A decision goes to the earlier of its arrival's ingress window
    and the earliest window that shows its step's rows; an order entry to the earlier of its arrival's ingress window
    and the earliest window that shows the rows of the first step that reads that arrival (a lookup, a decision or its
    admission); and a sent hit's shortfall to the earliest window that shows its step's. Each is nondecreasing in the
    replay's order, so each field's values, window after window, are the replay's."""
    earliest, win = charged(record, reg, order, rp, W)

    def at(k: int) -> int:
        return earliest[k] if k < len(earliest) else W - 1

    if reg.send_lengths:
        LN.check_admitted(record.requests, order, rp.admitted, at, reg.T, AdviceCodeError)

    arrived: list[list[bool]] = [[] for _ in range(W)]
    hits: list[list[int]] = [[] for _ in range(W)]
    ranks: list[list[tuple[int, tuple[int]]]] = [[] for _ in range(W)]
    touch = list(rp.admitted)
    for i, k, _ in decisions:
        touch[i] = min(touch[i], k)
    for s, ls in enumerate(looks):
        for x in ls:
            touch[x.request] = min(touch[x.request], s)
    for field, values in (("decision", [(min(at(k), win[i]), b) for i, k, b in decisions]),
                          ("order entry", [(min(at(touch[p]), win[p]), (p, (r,))) for p, r in _deviations(record, order)]),
                          ("hit", [(earliest[s], short) for s, short in _sent(record, engine, order, rp, looks)])):
        last = 0
        for w, v in values:
            if w < last:
                raise AdviceCodeError(f"a {field} charged to window {w}, after one charged to window {last}")
            last = w
            {"decision": arrived, "order entry": ranks, "hit": hits}[field][w].append(v)
    return [CV.Window(arrived=tuple(a), hits=tuple(h), order=tuple(o)) for a, h, o in zip(arrived, hits, ranks)]


# --- the string --------------------------------------------------------------------------------------------------------

#: v1's caps where the registration gives no book: an order entry ranks one arrival
CAPS = CV.Caps(order_len=1)
#: and under the certified tick, where no window holds a decision or an order entry
TICK_CAPS = replace(CAPS, arrived=0, order=0)


def book_of(reg: Registered) -> Any:
    if reg.book is None:
        return CV.Book.uniform(CAPS if reg.tick is None else TICK_CAPS)
    return reg.book


def lengths(record: Record, reg: Registered) -> list[list[int]]:
    """Each window's length terms (`lengths.terms`); none in any window when the registration sends no lengths."""
    if not reg.send_lengths:
        return [[] for _ in range(windows(record, reg.T))]
    return LN.terms(record.requests, release_order(record), reg.T, windows(record, reg.T))


def length_widths(record: Record, reg: Registered) -> list[list[int]]:
    return LN.widths(record.requests, lengths(record, reg))


def length_tails(record: Record, reg: Registered) -> list[list[tuple[int, int]]]:
    """Each window's tail as `string_of` takes it: (value, width) of each length term's output_len and finish."""
    return LN.tails(record.requests, lengths(record, reg), reg.config(reg.engine))


def string_of(wins: Sequence[Any], reg: Registered,
              tails: Sequence[Sequence[tuple[int, int]]] | None = None) -> tuple[list[int], list[int]]:
    """The string's bits, unpadded, and each window's charge: its segment's bits and its tail's (the length terms)."""
    enc = CV.encode(wins, book_of(reg), tails)
    return enc.bitlist(), list(enc.lens)


def _frames(bits: Sequence[int], reg: Registered, widths: Sequence[Sequence[int]], *,
            padded: bool) -> tuple[list[Any], list[list[int]]]:
    """The windows `bits` codes and their tails of `widths`, refusing any bits but the code's own and, when `padded`,
    fewer than 8 zero bits after them. The code is prefix-free, so at most one padding decodes."""
    err: Exception | None = None
    for pad in range(8 if padded else 1):
        if pad > len(bits) or any(bits[len(bits) - pad:]):
            break
        data, n = CV.pack(list(bits[:len(bits) - pad]))
        try:
            return CV.decode_tailed(data, n, book_of(reg), widths)
        except ValueError as e:
            err = err or e
    raise AdviceCodeError(f"no {len(widths)} windows end the string: {err}")


def _parse(data: Any, reg: Registered, record: Record) -> tuple[list[Any], list[list[int]]]:
    """A byte string's windows and tails; refuses all but those and their padding."""
    if not isinstance(data, (bytes, bytearray)):
        raise AdviceCodeError(f"v1 decodes bytes, not a {type(data).__name__}")
    if 8 * len(data) > reg.cap_bits:
        raise AdviceCodeError(f"{8 * len(data)} bits, over the cap ℓ = {reg.cap_bits}")
    return _frames(C.BitReader.from_bytes(bytes(data), 8 * len(data)).bits, reg, length_widths(record, reg), padded=True)


def bits_by_window(data: bytes, record: Record, reg: Registered) -> dict[int, int]:
    """Each window's charge: its segment's bits, a function of the string alone, and its length terms'. They sum to the
    string's bits less padding."""
    def run() -> dict[int, int]:
        wins, tails = _parse(data, reg, record)
        widths = length_widths(record, reg)
        return dict(enumerate(string_of(wins, reg, [list(zip(t, n)) for t, n in zip(tails, widths)])[1]))
    return _total(run)


# --- I1 from the replay ------------------------------------------------------------------------------------------------

def advice_of_replay(record: Record, reg: Registered, cfg: Any, order: Sequence[int], rp: Any) -> I1.Advice:
    """I1 v0 from the replayed steps: the batch order, I1's numbering by first row, then the derived rules' walk, which
    must give the replay's every start and n back, and the tie's request fields; each kept output's release from the
    record."""
    reqs = [record.requests[k] for k in order]
    batch = [sorted(st.rows, key=lambda r: r[2]) if reg.row_order == "batch" else list(st.rows) for st in rp.steps]
    num: dict[int, int] = {}
    first_of: list[int] = []  # I1 request -> engine index
    for rows in batch:
        for i, _, _ in rows:
            if i not in num:
                num[i] = len(first_of)
                first_of.append(i)
    if len(first_of) != len(reqs):
        raise AdviceCodeError(f"the replay schedules {len(first_of)} of the record's {len(reqs)} requests")
    rows = tuple(tuple(num[i] for i, _, _ in b) for b in batch)
    at = [{i: (start, n) for i, start, n in b} for b in batch]
    resumed = [set(st.resumed) for st in rp.steps]
    sampling = [0] * len(reqs)
    for b in batch:
        for i, start, n in b:
            if start + n >= len(reqs[i].prompt) + sampling[i]:
                sampling[i] += 1

    def hit_of(s: int, t: int, cached: int | None) -> int:
        if t not in at[s]:
            raise AdviceCodeError(f"step {s}: the walk admits request {t}, which the replay doesn't run there")
        h = at[s][t][0]
        if cached is not None and h != cached:
            raise AdviceCodeError(f"step {s}: the replay's hit of {h} for request {t} where nothing can have been evicted, and "
                                  f"the longest cached prefix is {cached}")
        return h

    def lag_of(i: int) -> bool:
        return sampling[first_of[i]] - len(reqs[first_of[i]].output) == 1

    def n_of(s: int, k: int) -> int:
        raise AdviceCodeError("the walk asks for an n")

    held = D.Holdings(tuple(D.CertifiedRequest(q.id, q.prompt, q.max_tokens) for q in reqs), tuple(q.output for q in reqs))
    prompt_len = tuple(len(reqs[i].prompt) for i in first_of)
    pool = D.pool_bound([[(num[i], start, n) for i, start, n in b] for b in batch], cfg)
    walk = D.walk_steps(rows, prompt_len, cfg, held=held, pool=pool, hit_of=hit_of,
                        resumed_of=lambda s, k: first_of[rows[s][k]] in resumed[s], n_of=n_of, lag_of=lag_of,
                        order_of=lambda s, numbers: tuple(first_of[i] for i in numbers))
    if walk.order != tuple(first_of):
        raise AdviceCodeError("the walk numbers the requests otherwise than the replay's first rows")
    for s, (mine, b) in enumerate(zip(walk.steps, batch, strict=True)):
        if [(x.start, x.n) for x in mine.rows] != [(start, n) for _, start, n in b]:
            raise AdviceCodeError(f"step {s}: the walk's rows {[(x.start, x.n) for x in mine.rows]} aren't the replay's "
                                  f"{[(start, n) for _, start, n in b]}")
    requests = D.tie_fields(walk, D.Residual(rows, prompt_len, ()), held, cfg)
    for i in walk.lag_requests:
        if lag_of(i) and requests[i].finish != "eos":
            raise AdviceCodeError(f"request {i}: a decode after its last output, which no stop id ended")
    kept = [0] * len(requests)
    release = []
    for st in walk.steps:
        for x in st.rows:
            q = requests[x.request]
            if x.start + x.n == q.prompt_len + kept[x.request] and kept[x.request] < q.output_len:
                release.append(reqs[first_of[x.request]].release[kept[x.request]])
                kept[x.request] += 1
    return I1.Advice(tuple(requests), walk.steps, tuple(release))


# --- encode and decode -------------------------------------------------------------------------------------------------

def _residual_of(advice: I1.Advice, record: Record, reg: Registered) -> tuple[list[Any], Any, list[int]]:
    """The windows of the advice's residual, the replay that gives them, and the arrival order."""
    if type(advice) is not I1.Advice:
        raise AdviceCodeError(f"v1 encodes an I1 Advice, not a {type(advice).__name__}")
    engine = reg.engine
    check_record(record, reg, engine)
    order = arrival_order(advice, record, reg.row_order)
    if reg.tick is not None and order != release_order(record):
        raise AdviceCodeError("under the certified tick the engine admits in release order, and this advice's arrival order "
                              "isn't it")
    index = {record.requests[k].id: e for e, k in enumerate(order)}
    first = [-1] * len(order)
    for s, st in enumerate(advice.steps):
        for x in st.rows:
            e = index[advice.requests[x.request].id]
            if first[e] < 0:
                first[e] = s
    rp, decisions, looks = replay_record(record, reg, engine, order, lambda i, k: k >= first[i])
    return _residual(record, reg, engine, order, rp, decisions, looks, windows(record, reg.T)), rp, order


def residual(advice: I1.Advice, record: Record, reg: Registered) -> list[Any]:
    """The windows `encode` codes (`window_charge.Window`s: the decisions, the order's deviations and the sent hits):
    what a codebook is fit on."""
    return _total(lambda: _residual_of(advice, record, reg)[0])


def encode(advice: I1.Advice, record: Record, reg: Registered, *, check: bool = True) -> bytes:
    """The advice's v1 string under the record; refuses an advice v1 can't carry (one whose schedule isn't the
    replay's under its arrival order, whose rows aren't in the registered order, or whose sources aren't the canonical
    ones). `check` decodes the string back and holds it to the advice."""
    wins, rp, order = _residual_of(advice, record, reg)
    data = CV.pack(string_of(wins, reg, length_tails(record, reg))[0])[0]
    if check:
        index = {record.requests[k].id: e for e, k in enumerate(order)}
        theirs = [[(index[advice.requests[x.request].id], x.start, x.n) for x in st.rows] for st in advice.steps]
        m = R.compare(rp.steps, [S.Step(rows=r) for r in theirs])
        if m.first_mismatch is not None:
            raise AdviceCodeError(f"the advice's schedule isn't the replay's from step {m.first_mismatch} "
                                  f"({m.matched}/{m.steps} steps match)")
        try:
            got = decode(data, record, reg)
        except AdviceCodeError as e:
            raise AdviceCodeError(f"the advice is outside v1: {e}") from None
        if got != advice:
            raise AdviceCodeError("the advice is outside v1: its string decodes to another advice (rows out of the "
                                  "registered order, or sources not the canonical ones)")
    return data


def decode(data: bytes, record: Record, reg: Registered) -> I1.Advice:
    """The I1 v0 the string `data` carries under the record, or `AdviceCodeError`."""
    return decode_full(data, record, reg).advice


def decode_full(data: bytes, record: Record, reg: Registered) -> Decoded:
    """`decode`, with the order the string sends (`Decoded`)."""
    return _total(lambda: _decode(*_parse(data, reg, record), record, reg))


def decode_bits(bits: Sequence[int], record: Record, reg: Registered) -> I1.Advice:
    """`decode` of the bit string itself, unpadded: refused unless its windows end exactly at its end."""
    def run() -> I1.Advice:
        if len(bits) > reg.cap_bits:
            raise AdviceCodeError(f"{len(bits)} bits, over the cap ℓ = {reg.cap_bits}")
        if any(b not in (0, 1) for b in bits):
            raise AdviceCodeError("a bit string holds 0s and 1s")
        return _decode(*_frames(list(bits), reg, length_widths(record, reg), padded=False), record, reg).advice
    return _total(run)


def _total(run: Callable[[], T]) -> T:
    try:
        return run()
    except AdviceCodeError:
        raise
    except (ValueError, RuntimeError) as e:
        raise AdviceCodeError(str(e)) from None


def _decode(wins: Sequence[Any], tails: Sequence[Sequence[int]], record: Record, reg: Registered) -> Decoded:
    engine = reg.engine
    cfg = check_record(record, reg, engine)
    LN.check(tails, record.requests, lengths(record, reg), cfg, AdviceCodeError)
    if any(w.resumes or w.lag for w in wins):
        raise AdviceCodeError("a window sends resumes or lag bits, which v1 replays or derives")
    if reg.tick is not None and any(w.arrived or w.order for w in wins):
        raise AdviceCodeError("a window sends decisions or order entries, which the certified tick fixes")
    entries = [(p, idx) for w in wins for p, idx in w.order]
    if any(len(idx) != 1 for _, idx in entries):
        raise AdviceCodeError("an order entry ranks one arrival")
    order = _arrival(record, [(p, idx[0]) for p, idx in entries])
    arrived = [b for w in wins for b in w.arrived]
    used = 0

    def bit_of(i: int, k: int) -> bool:
        nonlocal used
        if used == len(arrived):
            raise AdviceCodeError(f"the replay asks more than the string's {len(arrived)} decisions")
        used += 1
        return arrived[used - 1]

    rp, decisions, looks = replay_record(record, reg, engine, order, bit_of)
    if used != len(arrived):
        raise AdviceCodeError(f"the replay asks {used} of the string's {len(arrived)} decisions")
    if _residual(record, reg, engine, order, rp, decisions, looks, len(wins)) != list(wins):
        raise AdviceCodeError("the windows aren't the replay's residual: a hit that isn't the replay's, or a value charged to "
                              "another window than A5's")
    advice = advice_of_replay(record, reg, cfg, order, rp)
    if arrival_order(advice, record, reg.row_order) != order:
        raise AdviceCodeError("the order isn't the one the advice's rows give (another arrival order gives the same schedule)")
    return Decoded(advice, streams(advice, record), tuple(order))
