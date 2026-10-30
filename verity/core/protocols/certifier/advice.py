"""Advice v0: what serving decided, as one canonical bit string per certifier session.

The precommitted AOT compiler turns the advice into each executed engine step's circuit; the certifier certifies the
I/O; sampled proofs hold the committed values to the compiled circuits and to the certified I/O. v0 is the committed
form (hm96 hiding rows), private, and `bits` splits its length among egress windows. The charge, the content advice
`A_c` of the certifier's accepted-runs bound, is `code_v1`'s compressed length of the advice, not v0's.

**What it says** (vLLM v1, the pinned 0.28.1rc1.dev472's `Scheduler.schedule` and `SchedulerOutput`):
- `requests`, in order of their first row: each request's id (the body's `user`), prompt and output lengths and how it
  finished. `output_len` counts the tokens vLLM kept, the stopping EOS included.
- `steps`, one per forward that ran: its rows in the forward's batch order. A row is one request's `n` scheduled
  tokens at positions `start` … `start + n − 1` (`num_computed_tokens`, `num_scheduled_tokens`). v1 has no prefill or
  decode phase: a row's sample is kept exactly when the row reaches the request's known tokens (prompt plus kept
  outputs), and is otherwise discarded (a partial prefill chunk, or the one row async scheduling runs past an EOS).
  `kind` is `decode` exactly when n = 1 and start ≥ prompt_len. A row that admits its request (its first row, or its
  first after a preemption, `preempted`) starts at its prefix-cache hit (`cache_hit`), whose KV `sources` names:
  runs of positions 0 … cache_hit − 1, each the KV a request computed there in this step or an earlier one.
- `release`, one per kept output in `outputs` order (step, then row): the session-global egress bucket in which the
  output's frame is released (window = bucket // T).

**Canonical.** An `Advice` is valid or doesn't exist: construction refuses (`MalformedAdvice`) types, ranges, order and
schedule semantics, so every encoded string decodes to one advice and every advice has one encoding; `decode` refuses
any other byte string, non-minimal widths and nonzero padding included.

**The bit layout**, MSB first, `u(w)` a w-bit unsigned integer, `uvar` a u6 width L then u(L), L the value's bit
length; every width is the least that holds its largest value; w_req = bitlen(len(requests) − 1):

    header     version u8 (0) | n_requests uvar | n_steps uvar | w_len u6 | w_rel u6 | w_src u6
    request    id_bytes uvar (≥ 1) | id, id_bytes × u8, UTF-8 | prompt_len u(w_len) | output_len u(w_len) | finish u1 (0 eos)
    step       rows − 1 u(w_req) | rows × [ request u(w_req) | kind u1 (0 prefill) | start u(w_len) | n u(w_len)
               | cache_hit u(w_len) | preempted u1 | when cache_hit > 0: sources − 1 u(w_src),
               sources × (request u(w_req) | n u(w_len)) ]
    release    Σ output_len × u(w_rel)
    padding    zero bits to a whole byte, never charged

**By window.** Each record goes to the earliest egress window it can show in: an output's release to its own window,
a step's rows to the earliest window of an output at that step or later, a request to its first step's, the header to
the first output's. So `bits` per window sums to `total_bits`, and windows 0 … m − 1 hold every record their frames
read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

SCHEMA = "verity/advice/v0"
VERSION = 0
KINDS = ("prefill", "decode")
FINISHES = ("eos", "length")
WIDTH_BITS = 6
LIMIT = 1 << ((1 << WIDTH_BITS) - 1)  # every integer field is below 2^63, so its width fits a u6


class MalformedAdvice(ValueError):
    """Advice that has no meaning, refused rather than repaired."""


def _int(name: str, value: Any, least: int = 0) -> None:
    if type(value) is not int or not least <= value < LIMIT:
        raise MalformedAdvice(f"{name} must be an int in [{least}, 2^63), not {value!r}")


def _tuple(name: str, value: Any, kind: type) -> tuple:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise MalformedAdvice(f"{name} must be a sequence of {kind.__name__}, not {type(value).__name__}")
    items = tuple(value)
    for item in items:
        if type(item) is not kind:
            raise MalformedAdvice(f"{name} holds a {type(item).__name__}, not a {kind.__name__}")
    return items


@dataclass(frozen=True, slots=True)
class RequestAdvice:
    """One request: `id` is its body's `user`, `finish` is `eos` (a stop token ended it) or `length` (its cap did)."""

    id: str
    prompt_len: int
    output_len: int
    finish: str

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id:
            raise MalformedAdvice(f"a request's id must be a non-empty str, not {self.id!r}")
        try:
            self.id.encode("utf-8")
        except UnicodeEncodeError as e:
            raise MalformedAdvice(f"a request's id must be UTF-8 encodable: {e}") from None
        _int("prompt_len", self.prompt_len, 1)
        _int("output_len", self.output_len, 1)
        if type(self.finish) is not str or self.finish not in FINISHES:
            raise MalformedAdvice(f"finish must be one of {FINISHES}, not {self.finish!r}")


@dataclass(frozen=True, slots=True)
class Source:
    """`n` consecutive positions of a cache hit whose KV request `request` computed."""

    request: int
    n: int

    def __post_init__(self) -> None:
        _int("a source's request", self.request)
        _int("a source's n", self.n, 1)


@dataclass(frozen=True, slots=True)
class Row:
    """One request's tokens in one step: positions `start` … `start + n − 1`. A row that admits its request starts at
    its prefix-cache hit, `cache_hit` tokens whose KV `sources` names, in position order."""

    request: int
    kind: str
    start: int
    n: int
    cache_hit: int = 0
    preempted: bool = False
    sources: tuple[Source, ...] = ()

    def __post_init__(self) -> None:
        _int("a row's request", self.request)
        if type(self.kind) is not str or self.kind not in KINDS:
            raise MalformedAdvice(f"kind must be one of {KINDS}, not {self.kind!r}")
        _int("start", self.start)
        _int("n", self.n, 1)
        _int("cache_hit", self.cache_hit)
        if type(self.preempted) is not bool:
            raise MalformedAdvice(f"preempted must be a bool, not {self.preempted!r}")
        sources = _tuple("sources", self.sources, Source)
        object.__setattr__(self, "sources", sources)
        if sum(s.n for s in sources) != self.cache_hit:
            raise MalformedAdvice(f"the sources cover {sum(s.n for s in sources)} positions, the cache hit {self.cache_hit}")
        if any(a.request == b.request for a, b in zip(sources, sources[1:])):
            raise MalformedAdvice("two adjacent sources name one request: they are one run")
        if (self.cache_hit or self.preempted) and self.start != self.cache_hit:
            raise MalformedAdvice(f"a row with a cache hit or a resume starts at its hit, {self.cache_hit}, not {self.start}")


@dataclass(frozen=True, slots=True)
class StepAdvice:
    """One forward: its rows in the batch order it ran them, each request at most once."""

    rows: tuple[Row, ...]

    def __post_init__(self) -> None:
        rows = _tuple("rows", self.rows, Row)
        object.__setattr__(self, "rows", rows)
        if not rows:
            raise MalformedAdvice("a step has at least one row (a schedule of no tokens runs no forward)")
        if len({r.request for r in rows}) != len(rows):
            raise MalformedAdvice("a step schedules each request at most once")


class Output(NamedTuple):
    """A kept sample: step `step`'s row `row` sampled output token `index` of request `request`."""

    step: int
    row: int
    request: int
    index: int


class _Replay(NamedTuple):
    outputs: tuple[Output, ...]
    first: tuple[int, ...]  # each request's first step


def _cover(intervals: list[list[int]], lo: int, hi: int) -> None:
    """Add [lo, hi) to a sorted list of disjoint, non-adjacent intervals."""
    i = 0
    while i < len(intervals) and intervals[i][1] < lo:
        i += 1
    j = i
    while j < len(intervals) and intervals[j][0] <= hi:
        lo, hi = min(lo, intervals[j][0]), max(hi, intervals[j][1])
        j += 1
    intervals[i:j] = [[lo, hi]]


def _covered(intervals: list[list[int]], lo: int, hi: int) -> bool:
    return any(a <= lo and hi <= b for a, b in intervals)


def _replay(requests: tuple[RequestAdvice, ...], steps: tuple[StepAdvice, ...]) -> _Replay:
    """Run the schedule's position rules; refuse a row that breaks one. Per request: `computed` the positions the next
    row continues from (vLLM's num_computed_tokens), `known` the prompt plus kept outputs, `kept` its kept outputs."""
    R = len(requests)
    computed, known, kept = [0] * R, [0] * R, [0] * R
    made: list[list[list[int]]] = [[] for _ in range(R)]  # the positions whose KV each request has computed
    first: list[int] = []
    outputs: list[Output] = []
    for s, step in enumerate(steps):
        for j, row in enumerate(step.rows):
            r = row.request
            where = f"step {s} row {j}"
            if r >= R:
                raise MalformedAdvice(f"{where}: request {r} of {R}")
            req = requests[r]
            if r == len(first):
                first.append(s)
                known[r] = req.prompt_len
                if row.preempted:
                    raise MalformedAdvice(f"{where}: a request's first row can't resume it")
                if row.start != row.cache_hit:
                    raise MalformedAdvice(f"{where}: request {r}'s first row starts at its cache hit, {row.cache_hit}")
            elif r > len(first):
                raise MalformedAdvice(f"{where}: request {r} before request {len(first)}: requests are in order of first row")
            elif row.preempted:
                if kept[r] == req.output_len:
                    raise MalformedAdvice(f"{where}: request {r} has finished and can't be preempted")
            elif row.cache_hit:
                raise MalformedAdvice(f"{where}: a cache hit only on a row that admits its request")
            elif row.start != computed[r]:
                raise MalformedAdvice(f"{where}: request {r} continues at {computed[r]}, not {row.start}")
            end = row.start + row.n
            if end > known[r]:
                raise MalformedAdvice(f"{where}: positions to {end}, past request {r}'s {known[r]} known tokens")
            kind = "decode" if row.n == 1 and row.start >= req.prompt_len else "prefill"
            if row.kind != kind:
                raise MalformedAdvice(f"{where}: kind {row.kind!r}, where n {row.n} at {row.start} of a {req.prompt_len}-token "
                                      f"prompt is {kind!r}")
            computed[r] = end
            _cover(made[r], row.start, end)
            if end == known[r] and kept[r] < req.output_len:
                outputs.append(Output(s, j, r, kept[r]))
                kept[r] += 1
                known[r] += 1
        for j, row in enumerate(step.rows):
            at = 0
            for src in row.sources:
                if src.request >= len(first) or not _covered(made[src.request], at, at + src.n):
                    raise MalformedAdvice(f"step {s} row {j}: request {src.request} computed no KV at positions {at} … "
                                          f"{at + src.n - 1} by this step")
                if src.request == row.request and not row.preempted:
                    raise MalformedAdvice(f"step {s} row {j}: a request reuses its own KV only on a resume")
                at += src.n
    if len(first) != R:
        raise MalformedAdvice(f"requests {len(first)} … {R - 1} have no row")
    for r, req in enumerate(requests):
        if kept[r] != req.output_len:
            raise MalformedAdvice(f"request {r} kept {kept[r]} of its {req.output_len} outputs")
    return _Replay(tuple(outputs), tuple(first))


@dataclass(frozen=True, slots=True)
class Advice:
    """One certifier session's advice: its requests, its steps, and the egress bucket each kept output is released in."""

    requests: tuple[RequestAdvice, ...]
    steps: tuple[StepAdvice, ...]
    release: tuple[int, ...]

    def __post_init__(self) -> None:
        requests = _tuple("requests", self.requests, RequestAdvice)
        steps = _tuple("steps", self.steps, StepAdvice)
        release = _tuple("release", self.release, int)
        for b in release:
            _int("a release bucket", b)
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "release", release)
        if len({q.id for q in requests}) != len(requests):
            raise MalformedAdvice("two requests share an id")
        replay = _replay(requests, steps)
        if len(release) != len(replay.outputs):
            raise MalformedAdvice(f"{len(release)} release buckets for {len(replay.outputs)} kept outputs")
        last: dict[int, int] = {}
        for o, b in zip(replay.outputs, release):
            if b < last.get(o.request, 0):
                raise MalformedAdvice(f"request {o.request}'s output {o.index} is released at bucket {b}, before its "
                                      f"output {o.index - 1}'s {last[o.request]}")
            last[o.request] = b


def outputs(advice: Advice) -> tuple[Output, ...]:
    """The kept outputs, step by step and row by row: `advice.release[k]` is output k's bucket."""
    return _replay(advice.requests, advice.steps).outputs


class _Writer:
    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.held = 0
        self.bits = 0

    def put(self, value: int, width: int) -> None:
        if value >> width:
            raise AssertionError(f"{value} doesn't fit {width} bits")
        self.acc = (self.acc << width) | value
        self.held += width
        self.bits += width
        while self.held >= 8:
            self.held -= 8
            self.out.append(self.acc >> self.held)
            self.acc &= (1 << self.held) - 1

    def uvar(self, value: int) -> None:
        self.put(value.bit_length(), WIDTH_BITS)
        self.put(value, value.bit_length())

    def finish(self) -> bytes:
        if self.held:
            self.out.append(self.acc << (8 - self.held))
        return bytes(self.out)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.size = 8 * len(data)

    def left(self) -> int:
        return self.size - self.pos

    def get(self, width: int) -> int:
        if width == 0:
            return 0
        end = self.pos + width
        if end > self.size:
            raise MalformedAdvice(f"truncated: a {width}-bit field at bit {self.pos} of {self.size}")
        lo, hi = self.pos >> 3, (end + 7) >> 3
        value = (int.from_bytes(self.data[lo:hi], "big") >> (8 * hi - end)) & ((1 << width) - 1)
        self.pos = end
        return value

    def uvar(self) -> int:
        width = self.get(WIDTH_BITS)
        value = self.get(width)
        if value.bit_length() != width:
            raise MalformedAdvice(f"a {width}-bit length-prefixed field holds {value}: its width isn't its bit length")
        return value


class _Widths(NamedTuple):
    req: int
    len: int
    rel: int
    src: int


def _widths(advice: Advice) -> _Widths:
    lens = [x for q in advice.requests for x in (q.prompt_len, q.output_len)]
    lens += [x for st in advice.steps for row in st.rows for x in (row.start, row.n, row.cache_hit, *(s.n for s in row.sources))]
    src = max((len(row.sources) - 1 for st in advice.steps for row in st.rows if row.cache_hit), default=0)
    return _Widths(max(len(advice.requests) - 1, 0).bit_length(), max(lens, default=0).bit_length(),
                   max(advice.release, default=0).bit_length(), src.bit_length())


class _Layout(NamedTuple):
    data: bytes
    header: int
    requests: tuple[int, ...]
    steps: tuple[int, ...]
    release: int  # bits per release bucket


def _layout(advice: Advice) -> _Layout:
    w = _widths(advice)
    out = _Writer()
    out.put(VERSION, 8)
    out.uvar(len(advice.requests))
    out.uvar(len(advice.steps))
    out.put(w.len, WIDTH_BITS)
    out.put(w.rel, WIDTH_BITS)
    out.put(w.src, WIDTH_BITS)
    header = out.bits
    requests = []
    for q in advice.requests:
        mark = out.bits
        raw = q.id.encode("utf-8")
        out.uvar(len(raw))
        for byte in raw:
            out.put(byte, 8)
        out.put(q.prompt_len, w.len)
        out.put(q.output_len, w.len)
        out.put(FINISHES.index(q.finish), 1)
        requests.append(out.bits - mark)
    steps = []
    for st in advice.steps:
        mark = out.bits
        out.put(len(st.rows) - 1, w.req)
        for row in st.rows:
            out.put(row.request, w.req)
            out.put(KINDS.index(row.kind), 1)
            out.put(row.start, w.len)
            out.put(row.n, w.len)
            out.put(row.cache_hit, w.len)
            out.put(int(row.preempted), 1)
            if row.cache_hit:
                out.put(len(row.sources) - 1, w.src)
                for src in row.sources:
                    out.put(src.request, w.req)
                    out.put(src.n, w.len)
        steps.append(out.bits - mark)
    for b in advice.release:
        out.put(b, w.rel)
    return _Layout(out.finish(), header, tuple(requests), tuple(steps), w.rel)


def encode(advice: Advice) -> bytes:
    """The advice's canonical bit string, zero-padded to whole bytes."""
    if type(advice) is not Advice:
        raise MalformedAdvice(f"encode takes an Advice, not a {type(advice).__name__}")
    return _layout(advice).data


def decode(data: bytes) -> Advice:
    """The advice `data` encodes; refuses anything but a canonical encoding of a valid advice."""
    if not isinstance(data, (bytes, bytearray)):
        raise MalformedAdvice(f"decode takes bytes, not a {type(data).__name__}")
    data = bytes(data)
    rd = _Reader(data)
    if (version := rd.get(8)) != VERSION:
        raise MalformedAdvice(f"advice version {version}, not {VERSION}")
    n_requests, n_steps = rd.uvar(), rd.uvar()
    w_len, w_rel, w_src = rd.get(WIDTH_BITS), rd.get(WIDTH_BITS), rd.get(WIDTH_BITS)
    w_req = max(n_requests - 1, 0).bit_length()
    if 15 * n_requests + 2 * n_steps > rd.left():
        raise MalformedAdvice(f"truncated: {n_requests} requests and {n_steps} steps in {rd.left()} bits")
    requests = []
    for _ in range(n_requests):
        size = rd.uvar()
        if 8 * size > rd.left():
            raise MalformedAdvice(f"truncated: a {size}-byte id in {rd.left()} bits")
        try:
            rid = bytes(rd.get(8) for _ in range(size)).decode("utf-8")
        except UnicodeDecodeError as e:
            raise MalformedAdvice(f"a request's id isn't UTF-8: {e}") from None
        requests.append(RequestAdvice(rid, rd.get(w_len), rd.get(w_len), FINISHES[rd.get(1)]))
    steps, rows_total = [], 0
    for _ in range(n_steps):
        rows = []
        for _ in range(rd.get(w_req) + 1):
            request = rd.get(w_req)
            kind = KINDS[rd.get(1)]
            start = rd.get(w_len)
            n = rd.get(w_len)
            hit = rd.get(w_len)
            preempted = bool(rd.get(1))
            sources = tuple(Source(rd.get(w_req), rd.get(w_len)) for _ in range(rd.get(w_src) + 1)) if hit else ()
            rows.append(Row(request, kind, start, n, hit, preempted, sources))
        rows_total += len(rows)
        steps.append(StepAdvice(tuple(rows)))
    n_outputs = sum(q.output_len for q in requests)
    if n_outputs > rows_total:
        raise MalformedAdvice(f"{n_outputs} outputs from {rows_total} rows")
    release = tuple(rd.get(w_rel) for _ in range(n_outputs))
    if rd.left() >= 8 or rd.get(rd.left()):
        raise MalformedAdvice("bytes past the advice, or nonzero padding")
    advice = Advice(tuple(requests), tuple(steps), release)
    if _widths(advice) != (w_req, w_len, w_rel, w_src):
        raise MalformedAdvice(f"widths (len {w_len}, release {w_rel}, sources {w_src}) aren't the least that hold the advice's "
                              f"values, {_widths(advice)[1:]}")
    return advice


def total_bits(advice: Advice) -> int:
    """v0's length in bits, its padding left out (the charge `A_c` is `code_v1`'s compressed length)."""
    lay = _layout(advice)
    return lay.header + sum(lay.requests) + sum(lay.steps) + lay.release * len(advice.release)


def bits_by_window(advice: Advice, T: int) -> dict[int, int]:
    """v0's bits in each egress window of T buckets that holds any, by the earliest window each record can show in (the
    module's docstring, "By window"). The values sum to `total_bits`."""
    _int("T", T, 1)
    lay = _layout(advice)
    replay = _replay(advice.requests, advice.steps)
    charge: dict[int, int] = {}

    def add(window: int, n: int) -> None:
        if n:
            charge[window] = charge.get(window, 0) + n

    if not advice.release:
        add(0, total_bits(advice))
        return charge
    windows = [b // T for b in advice.release]
    own: list[int | None] = [None] * len(advice.steps)
    for o, w in zip(replay.outputs, windows):
        own[o.step] = w if own[o.step] is None else min(own[o.step], w)
    earliest, at = [0] * len(advice.steps), None
    for s in range(len(advice.steps) - 1, -1, -1):
        if own[s] is not None:
            at = own[s] if at is None else min(at, own[s])
        earliest[s] = max(windows) if at is None else at  # a step no output follows shows nowhere: the last window
    add(earliest[0], lay.header)
    for r, n in enumerate(lay.requests):
        add(earliest[replay.first[r]], n)
    for s, n in enumerate(lay.steps):
        add(earliest[s], n)
    for w in windows:
        add(w, lay.release)
    return charge


def bits(advice: Advice, window: int, T: int) -> int:
    """v0's bits in egress window `window` of T buckets (`bits_by_window`)."""
    _int("window", window)
    return bits_by_window(advice, T).get(window, 0)


def to_json(advice: Advice) -> dict:
    """The advice as JSON, for debugging; `from_json` reads it back."""
    return {"schema": SCHEMA,
            "requests": [{"id": q.id, "prompt_len": q.prompt_len, "output_len": q.output_len, "finish": q.finish}
                         for q in advice.requests],
            "steps": [{"rows": [{"request": r.request, "kind": r.kind, "start": r.start, "n": r.n, "cache_hit": r.cache_hit,
                                 "preempted": r.preempted, "sources": [{"request": s.request, "n": s.n} for s in r.sources]}
                                for r in st.rows]} for st in advice.steps],
            "release": list(advice.release)}


def _keys(doc: Any, keys: tuple[str, ...], what: str) -> Mapping:
    if not isinstance(doc, Mapping) or set(doc) != set(keys):
        raise MalformedAdvice(f"{what} must be an object with exactly {sorted(keys)}, not {doc!r:.200}")
    return doc


def _list(doc: Any, what: str) -> list:
    if not isinstance(doc, list):
        raise MalformedAdvice(f"{what} must be a list, not {type(doc).__name__}")
    return doc


def from_json(doc: Any) -> Advice:
    """The advice `to_json` wrote; refuses any other key or shape."""
    d = _keys(doc, ("schema", "requests", "steps", "release"), "advice")
    if d["schema"] != SCHEMA:
        raise MalformedAdvice(f"schema {d['schema']!r}, not {SCHEMA!r}")
    requests = [RequestAdvice(**_keys(q, ("id", "prompt_len", "output_len", "finish"), "a request"))
                for q in _list(d["requests"], "requests")]
    steps = []
    for st in _list(d["steps"], "steps"):
        rows = []
        for r in _list(_keys(st, ("rows",), "a step")["rows"], "rows"):
            r = _keys(r, ("request", "kind", "start", "n", "cache_hit", "preempted", "sources"), "a row")
            sources = tuple(Source(**_keys(s, ("request", "n"), "a source")) for s in _list(r["sources"], "sources"))
            rows.append(Row(r["request"], r["kind"], r["start"], r["n"], r["cache_hit"], r["preempted"], sources))
        steps.append(StepAdvice(tuple(rows)))
    return Advice(tuple(requests), tuple(steps), tuple(_list(d["release"], "release")))
