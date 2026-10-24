"""Advice v1: I1 v0 (`verity.core.protocols.certifier.advice`) sent as only what the certified record, the scheduler
replay (ENC-REPLAY, `replay.py`) and the derived rules (ENC-DERIVED, `derived.py`) leave, in ENC-CODE's prefix-free code
(`code.py`).

`encode(advice, record, registered) -> bytes` and `decode(bytes, record, registered) -> I1.Advice`, with
`decode(encode(a)) == a`. `decode` is total: any byte string decodes to one I1 v0 or is refused by name (`V1Error`),
deterministically, and a string longer than the cap ℓ (`Registered.cap_bits`) is refused unread.

**Certified, 0 bits** (`Record`, from the warden's signed records under design v3; the compile unit holds them as
witness):
- the admission order: the warden's connection ordinals, in the order its own schedule admitted them (`Record.requests`);
- each request's ingress bytes, an id request's fields (`id_request_fields`): its `user` (I1's id), its `prompt` ids and
  its `max_tokens`; and its last-frame mark, the bucket of its connection's ingress `END` cell (`Request.end`);
- each request's output ids, its id events' `token_id`s in order, and each output's release bucket (`Request.release`),
  which the warden's fixed schedule sets (egress cell i in bucket s + D + ⌊i/r⌋).
The engine sees a request once its last ingress frame is out, so its arrival order is the requests by (`end`, ordinal)
(`engine_order`), and a request's decisions are charged to the window of its `end`.

**Replayed, 0 bits**: ENC-REPLAY's `replay` runs vLLM's scheduler over those requests under the registered engine
(`Registered.engine`), and gives every step's rows (request, start, n), new and resumed requests, cache hits (its
block pool's LRU, eviction included) and preemptions, from one bit per open step: whether the next request had arrived.

**Derived, 0 bits**: the batch order (`Registered.row_order`: Model Runner V2's stable sort of the scheduled rows by n,
or the scheduler's own order), I1's numbering by first row, and ENC-DERIVED's walk (`derived._walk`) over the replayed
rows, which recomputes every start and n by vLLM's rules and checks them against the replay, holds each hit under the
cumulative pool bound to the longest cached prefix, and gives `kind`, the canonical `sources`, and the request records
(the tie: prompt and output lengths against the certified ids, `finish` by the stop rule).

**Sent**: the replay's decisions, and nothing else: past the pool bound the replay fixes the hits and the resumes the
walk would otherwise read, and the order the walk would send is the replay's. Each decision is one symbol of ENC-CODE's
arithmetic coder (`code.ArithEncoder`), at ENC-REPLAY's adaptive estimate for its context (`Registered.contexts`, see
`replay.CONTEXTS`), with one terminated segment per window. The registered pool's size `num_gpu_blocks` is sent too
unless the deployment pins it (vLLM sizes it at startup); I1 v0 doesn't carry it, so a string is its advice's encoding
under the pool its header sends (`registered_of`).

**The string**, MSB first, then zero bits to a whole byte:

    header   Elias-gamma(num_gpu_blocks), unless pinned
    window   for w = 0 … W − 1: Elias-gamma(ℓ_w + 1) | the window's ℓ_w-bit segment

with W the record's windows (`windows`: its last bucket's, plus one). A window's charge is its frame, Elias-gamma's
bits and the segment's (`bits_by_window`); the header goes to window 0. So the charge is a function of the string alone,
never of the record. The frames make the set of strings prefix-free for every W, so a string charged c bits in all is
one of at most 2^c (LEAK-LEAN's `bitsCount_of_prefixFree`), and window w's frame decodes from the frames before it.
`decode` takes only the canonical string: a segment that is the coder's own codeword for the symbols it decodes to,
exactly that long, zero padding of fewer than 8 bits and no byte past it.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from verity.core.protocols.certifier import advice as I1

HERE = Path(__file__).resolve().parent
T = TypeVar("T")


def _load(name: str) -> Any:
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


R, C, D = _load("replay"), _load("code"), _load("derived")

ROW_ORDERS = ("batch", "scheduling")


class V1Error(ValueError):
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
            raise V1Error(f"a request's id is a non-empty str, not {self.id!r}")
        ints = (self.max_tokens, self.end, *self.prompt, *self.output, *self.release)
        if any(type(x) is not int or x < 0 for x in ints):
            raise V1Error(f"request {self.id}: its fields are ints ≥ 0")
        if not self.prompt or not self.output or self.max_tokens < 1:
            raise V1Error(f"request {self.id}: a prompt, an output and max_tokens of at least one")
        if len(self.release) != len(self.output):
            raise V1Error(f"request {self.id}: {len(self.release)} release buckets for {len(self.output)} outputs")
        if any(b < a for a, b in zip(self.release, self.release[1:])):
            raise V1Error(f"request {self.id}: its outputs' release buckets decrease")


@dataclass(frozen=True)
class Record:
    """The certified record: the requests in the warden's admission order."""

    requests: tuple[Request, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        if not self.requests or any(type(q) is not Request for q in self.requests):
            raise V1Error("a record is one or more Requests")
        if len({q.id for q in self.requests}) != len(self.requests):
            raise V1Error("two requests share an id")


def engine_order(record: Record) -> list[int]:
    """The requests in the order the engine sees them: by last-frame mark, then by admission."""
    return sorted(range(len(record.requests)), key=lambda k: (record.requests[k].end, k))


def windows(record: Record, T: int) -> int:
    """W: the windows the record covers, through its last bucket."""
    return max(max(q.end, *q.release) for q in record.requests) // T + 1


@dataclass(frozen=True)
class Registered:
    """What v1 reads besides the string and the record, fixed before the run: the engine's scheduler config (its
    `num_gpu_blocks` the session's when not `pinned`, and then sent), the stop ids, the runner's row order, the
    window T in buckets, the decisions' contexts, the largest pool a header may name, and ℓ."""

    engine: Any  # replay.Engine
    stop_ids: tuple[int, ...]
    row_order: str = "batch"
    T: int = 600
    contexts: str = "plain"
    pinned: bool = False
    max_gpu_blocks: int = 1 << 20
    cap_bits: int = 1 << 23

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_ids", tuple(self.stop_ids))
        e = self.engine
        if not self.stop_ids or self.row_order not in ROW_ORDERS or self.contexts not in R.CONTEXTS or self.T < 1:
            raise V1Error(f"a registration has stop ids, a row order of {ROW_ORDERS}, contexts of {R.CONTEXTS} and T ≥ 1")
        if e.max_num_seqs < 1 or e.max_num_batched_tokens < 1 or e.max_model_len < 2 or e.block_size < 1:
            raise V1Error("the engine schedules at least one sequence and token a step")
        if not e.enable_chunked_prefill and e.max_num_batched_tokens < e.max_model_len:
            raise V1Error("without chunked prefill the token budget holds a whole max_model_len prompt (vLLM's own check)")

    def config(self, engine: Any) -> Any:
        """ENC-DERIVED's `Config` for `engine`."""
        return D.Config(block_size=engine.block_size, num_gpu_blocks=engine.num_gpu_blocks,
                        max_num_batched_tokens=engine.max_num_batched_tokens, max_model_len=engine.max_model_len,
                        eos_token_id=self.stop_ids[0], stop_token_ids=self.stop_ids[1:],
                        scheduler="async" if engine.async_scheduling else "sync",
                        long_prefill_token_threshold=engine.long_prefill_token_threshold,
                        max_num_seqs=engine.max_num_seqs, enable_prefix_caching=engine.enable_prefix_caching,
                        row_order=self.row_order)


def _check(record: Record, reg: Registered, engine: Any) -> Any:
    """The derived config, once the record and the pool are ones the replay runs to the end on: every output the stop
    rule's, within max_model_len, and the pool holding any one request alone, async's two steps ahead included."""
    cfg = reg.config(engine)
    L, B = engine.max_model_len, engine.block_size
    for q in record.requests:
        if len(q.prompt) + len(q.output) > L or D.stop_rule(q.output, len(q.prompt), q.max_tokens, cfg) is None:
            raise V1Error(f"request {q.id}: its output isn't one the stop rule ends, within max_model_len {L}")
    least = max(R.cdiv(min(len(q.prompt) + len(q.output) + 2, L), B) for q in record.requests) + 1
    if not least <= engine.num_gpu_blocks <= reg.max_gpu_blocks:
        raise V1Error(f"num_gpu_blocks {engine.num_gpu_blocks} outside [{least}, {reg.max_gpu_blocks}]: the pool holds "
                      f"each request alone, and no more than the registration's largest")
    return cfg


# --- the decisions -----------------------------------------------------------------------------------------------------

Symbol = tuple[int, int, int]  # (bit, f0, f1)


def _scaled(f0: int, f1: int) -> tuple[int, int]:
    """ENC-REPLAY's estimate within ENC-CODE's coder's total, halved and kept at least 1."""
    while f0 + f1 > C.MAX_TOTAL:
        f0, f1 = (f0 + 1) >> 1, (f1 + 1) >> 1
    return f0, f1


def _ios(record: Record, engine: Any, order: Sequence[int], stops: frozenset[int]) -> list[Any]:
    out = []
    for k in order:
        q = record.requests[k]
        out.append(R.RequestIO(prompt_len=len(q.prompt), max_tokens=q.max_tokens, output_len=len(q.output),
                               eos=q.output[-1] in stops,
                               tokens=q.prompt + q.output if engine.enable_prefix_caching else None))
    return out


def _run(record: Record, reg: Registered, engine: Any, order: Sequence[int], W: int,
         bit_of: Callable[[int, int, int, int, int], int]) -> tuple[Any, list[list[Symbol]]]:
    """The replay, each decision `bit_of(request, step, window, f0, f1)`; and each window's symbols."""
    ios = _ios(record, engine, order, reg.config(engine).stops)
    reqs = [record.requests[k] for k in order]
    cert = R.Certified(engine, ios, 0.0, [q.end for q in reqs], [list(q.release) for q in reqs])
    context, model = R._coder(cert, reg.contexts), R._model(reg.contexts)
    win = [q.end // reg.T for q in reqs]
    passed = [0] * len(reqs)
    syms: list[list[Symbol]] = [[] for _ in range(W)]

    def decide(i: int, k: int, o: Any) -> bool:
        ctx = context(i, passed[i], o)
        f0, f1 = _scaled(*model.freqs(ctx))
        bit = bit_of(i, k, win[i], f0, f1)
        syms[win[i]].append((bit, f0, f1))
        model.update(ctx, bool(bit))
        passed[i] += 1
        return bool(bit)

    return R.replay(engine, ios, decide=decide), syms


def _segment(syms: Sequence[Symbol]) -> bytearray:
    w = C.BitWriter()
    enc = C.ArithEncoder(w)
    for bit, f0, f1 in syms:
        enc.encode(f0 if bit else 0, f1 if bit else f0, f0 + f1)
    enc.finish()
    return w.bits


# --- the frames --------------------------------------------------------------------------------------------------------

def _pack(blocks: int | None, segments: Sequence[bytearray]) -> bytes:
    w = C.BitWriter()
    if blocks is not None:
        C.put_gamma(w, blocks)
    for seg in segments:
        C.put_gamma(w, len(seg) + 1)
        w.bits.extend(seg)
    return w.to_bytes()


Frames = tuple[int | None, list[bytearray], list[int]]  # the header, each window's segment, each window's charge


def _frames(bits: bytearray, reg: Registered, W: int) -> tuple[Frames, int]:
    """The frames at the start of `bits`, and where they end."""
    r = C.BitReader(bits)
    blocks = None if reg.pinned else C.get_gamma(r)
    head = r.pos
    segments, charge = [], []
    for _ in range(W):
        at = r.pos
        n = C.get_gamma(r) - 1
        if n > len(bits) - r.pos:
            raise V1Error(f"truncated: a {n}-bit segment in {len(bits) - r.pos} bits")
        segments.append(bits[r.pos:r.pos + n])
        r.pos += n
        charge.append(r.pos - at)
    charge[0] += head
    return (blocks, segments, charge), r.pos


def _parse(data: Any, reg: Registered, W: int) -> Frames:
    """A byte string's frames; refuses all but the frames and their padding."""
    if not isinstance(data, (bytes, bytearray)):
        raise V1Error(f"v1 decodes bytes, not a {type(data).__name__}")
    if 8 * len(data) > reg.cap_bits:
        raise V1Error(f"{8 * len(data)} bits, over the cap ℓ = {reg.cap_bits}")
    bits = C.BitReader.from_bytes(bytes(data), 8 * len(data)).bits
    frames, end = _frames(bits, reg, W)
    if len(bits) - end >= 8 or any(bits[end:]):
        raise V1Error("bytes past the last window, or nonzero padding")
    return frames


def bits_by_window(data: bytes, record: Record, reg: Registered) -> dict[int, int]:
    """Each window's charge: its frame's bits, the header's in window 0. They sum to the string's bits less padding."""
    try:
        return dict(enumerate(_parse(data, reg, windows(record, reg.T))[2]))
    except ValueError as e:
        raise V1Error(str(e)) from None


# --- I1 from the replay ------------------------------------------------------------------------------------------------

def _advice(record: Record, reg: Registered, cfg: Any, order: Sequence[int], rp: Any) -> I1.Advice:
    """I1 v0 from the replayed steps: the batch order, I1's numbering by first row, then ENC-DERIVED's walk, which must
    give the replay's every start and n back, and the tie's request fields; each kept output's release from the
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
        raise V1Error(f"the replay schedules {len(first_of)} of the record's {len(reqs)} requests")
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
            raise V1Error(f"step {s}: the walk admits request {t}, which the replay doesn't run there")
        h = at[s][t][0]
        if cached is not None and h != cached:
            raise V1Error(f"step {s}: the replay's hit of {h} for request {t} where nothing can have been evicted, and "
                          f"the longest cached prefix is {cached}")
        return h

    def lag_of(i: int) -> bool:
        return sampling[first_of[i]] - len(reqs[first_of[i]].output) == 1

    def n_of(s: int, k: int) -> int:
        raise V1Error("the walk asks for an n")

    held = D.Holdings(tuple(D.CertifiedRequest(q.id, q.prompt, q.max_tokens) for q in reqs), tuple(q.output for q in reqs))
    prompt_len = tuple(len(reqs[i].prompt) for i in first_of)
    pool = D.pool_bound([[(num[i], start, n) for i, start, n in b] for b in batch], cfg)
    walk = D._walk(rows, prompt_len, cfg, held=held, pool=pool, hit_of=hit_of,
                   resumed_of=lambda s, k: first_of[rows[s][k]] in resumed[s], n_of=n_of, lag_of=lag_of,
                   order_of=lambda s, numbers: tuple(first_of[i] for i in numbers))
    if walk.order != tuple(first_of):
        raise V1Error("the walk numbers the requests otherwise than the replay's first rows")
    for s, (mine, b) in enumerate(zip(walk.steps, batch, strict=True)):
        if [(x.start, x.n) for x in mine.rows] != [(start, n) for _, start, n in b]:
            raise V1Error(f"step {s}: the walk's rows {[(x.start, x.n) for x in mine.rows]} aren't the replay's "
                          f"{[(start, n) for _, start, n in b]}")
    requests = D.tie_fields(walk, D.Residual(rows, prompt_len, ()), held, cfg)
    for i in walk.lag_requests:
        if lag_of(i) and requests[i].finish != "eos":
            raise V1Error(f"request {i}: a decode after its last output, which no stop id ended")
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

def encode(advice: I1.Advice, record: Record, reg: Registered, *, check: bool = True) -> bytes:
    """The advice's v1 string under the record; refuses an advice v1 can't carry (one whose schedule isn't the
    replay's under some arrivals, whose rows aren't in the registered order, or whose sources aren't the canonical
    ones). `check` decodes the string back and holds it to the advice."""
    if type(advice) is not I1.Advice:
        raise V1Error(f"v1 encodes an I1 Advice, not a {type(advice).__name__}")
    engine = reg.engine
    _check(record, reg, engine)
    order = engine_order(record)
    index = {record.requests[k].id: e for e, k in enumerate(order)}
    if sorted(q.id for q in advice.requests) != sorted(index):
        raise V1Error("the advice's requests aren't the record's")
    first = [-1] * len(order)
    for s, st in enumerate(advice.steps):
        for x in st.rows:
            e = index[advice.requests[x.request].id]
            if first[e] < 0:
                first[e] = s
    W = windows(record, reg.T)
    rp, syms = _run(record, reg, engine, order, W, lambda i, k, w, f0, f1: int(k >= first[i]))
    data = _pack(None if reg.pinned else engine.num_gpu_blocks, [_segment(s) for s in syms])
    if check:
        theirs = [[(index[advice.requests[x.request].id], x.start, x.n) for x in st.rows] for st in advice.steps]
        m = R.compare(rp.steps, [R.Step(rows=r) for r in theirs])
        if m.first_mismatch is not None:
            raise V1Error(f"the advice's schedule isn't the replay's from step {m.first_mismatch} "
                          f"({m.matched}/{m.steps} steps match)")
        try:
            got = decode(data, record, reg)
        except V1Error as e:
            raise V1Error(f"the advice is outside v1: {e}") from None
        if got != advice:
            raise V1Error("the advice is outside v1: its string decodes to another advice (rows out of the "
                          "registered order, or sources not the canonical ones)")
    return data


def decode(data: bytes, record: Record, reg: Registered) -> I1.Advice:
    """The I1 v0 the string `data` carries under the record, or `V1Error`."""
    return _total(lambda: _decode(_parse(data, reg, windows(record, reg.T)), record, reg))


def decode_bits(bits: Sequence[int], record: Record, reg: Registered) -> I1.Advice:
    """`decode` of the bit string itself, unpadded: refused unless its frames end exactly at its end."""
    def run() -> I1.Advice:
        if len(bits) > reg.cap_bits:
            raise V1Error(f"{len(bits)} bits, over the cap ℓ = {reg.cap_bits}")
        if any(b not in (0, 1) for b in bits):
            raise V1Error("a bit string holds 0s and 1s")
        frames, end = _frames(bytearray(bits), reg, windows(record, reg.T))
        if end != len(bits):
            raise V1Error(f"{len(bits) - end} bits past the last window")
        return _decode(frames, record, reg)
    return _total(run)


def registered_of(data: bytes, record: Record, reg: Registered) -> Registered:
    """`reg` with the pool the string's header sends, under which the string is its advice's encoding (`reg` itself
    when the pool is pinned)."""
    return _total(lambda: _with_pool(reg, _parse(data, reg, windows(record, reg.T))[0]))


def _with_pool(reg: Registered, blocks: int | None) -> Registered:
    return reg if blocks is None else replace(reg, engine=replace(reg.engine, num_gpu_blocks=blocks))


def _total(run: Callable[[], T]) -> T:
    try:
        return run()
    except V1Error:
        raise
    except (ValueError, RuntimeError) as e:
        raise V1Error(str(e)) from None


def _decode(frames: Frames, record: Record, reg: Registered) -> I1.Advice:
    W = windows(record, reg.T)
    blocks, segments, _ = frames
    engine = _with_pool(reg, blocks).engine
    cfg = _check(record, reg, engine)
    order = engine_order(record)
    decs = [C.ArithDecoder(C.BitReader(seg)) for seg in segments]

    def bit_of(i: int, k: int, w: int, f0: int, f1: int) -> int:
        d = decs[w]
        bit = int(d.target(f0 + f1) >= f0)
        d.consume(f0 if bit else 0, f1 if bit else f0, f0 + f1)
        return bit

    rp, syms = _run(record, reg, engine, order, W, bit_of)
    for w, (d, seg) in enumerate(zip(decs, segments)):
        if d.finish() != len(seg) or _segment(syms[w]) != seg:
            raise V1Error(f"window {w}: its {len(seg)}-bit segment isn't the code's own for the decisions it decodes to")
    return _advice(record, reg, cfg, order, rp)


# --- from a trace ------------------------------------------------------------------------------------------------------

def record_of(trace: Any, stop_ids: Sequence[int], *, bucket_s: float = 0.1) -> Record:
    """A trace's certified record (`replay.Trace`, clocked, with each request's tokens): its engine order standing for
    the warden's admission order, each ingress bucket for the last-frame mark, and the trace's release buckets."""
    if not trace.clocked():
        raise V1Error("the trace has no clock, so no certified buckets")
    cert = trace.certified(bucket_s)
    if any(b < a for a, b in zip(cert.ingress, cert.ingress[1:])):
        raise V1Error("the engine took the requests out of their ingress order")
    out = []
    for k, io in enumerate(trace.requests):
        if io.tokens is None:
            raise V1Error(f"request {trace.ids[k]}: the trace has no tokens")
        output = tuple(io.tokens[io.prompt_len:io.prompt_len + io.output_len])
        if (output[-1] in stop_ids) != io.eos:
            raise V1Error(f"request {trace.ids[k]}: stopped {'on' if io.eos else 'without'} a stop id, its last id "
                          f"{output[-1]}")
        out.append(Request(trace.ids[k], tuple(io.tokens[:io.prompt_len]), io.max_tokens, output, cert.ingress[k],
                           tuple(cert.release[k])))
    return Record(tuple(out))


def advice_of(trace: Any, record: Record) -> I1.Advice:
    """The trace as I1 v0: its rows in the order its runner ran them, I1's numbering by first row, each admitting row's
    hit its start and its sources the canonical ones, and each kept output released in the record's bucket. The trace's
    requests are the record's, in order."""
    steps_rows = [list(st.rows) for st in trace.steps]
    num: dict[int, int] = {}
    order: list[int] = []
    for rows in steps_rows:
        for i, _, _ in rows:
            if i not in num:
                num[i] = len(order)
                order.append(i)
    reqs = [record.requests[i] for i in order]
    requests = tuple(I1.RequestAdvice(q.id, len(q.prompt), len(q.output), "eos" if trace.requests[i].eos else "length")
                     for i, q in zip(order, reqs))
    prefixes = D.Prefixes([q.prompt + q.output for q in reqs])
    seen: set[int] = set()
    kept = [0] * len(order)
    steps, release = [], []
    for st, rows in zip(trace.steps, steps_rows):
        for i, start, n in rows:
            prefixes.add(num[i], start, start + n)
        out = []
        for i, start, n in rows:
            j, q = num[i], requests[num[i]]
            back = i in st.resumed
            hit = start if j not in seen or back else 0
            seen.add(j)
            out.append(I1.Row(j, "decode" if n == 1 and start >= q.prompt_len else "prefill", start, n, hit, back,
                              prefixes.sources(j, hit)))
            if start + n == q.prompt_len + kept[j] and kept[j] < q.output_len:
                release.append(reqs[j].release[kept[j]])
                kept[j] += 1
        steps.append(I1.StepAdvice(tuple(out)))
    return I1.Advice(requests, tuple(steps), tuple(release))
