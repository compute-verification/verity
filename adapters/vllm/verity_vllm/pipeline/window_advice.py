"""A served Commit window's advice: I1 v0 (`verity.core.protocols.certifier.advice`) with the state of the requests the window
carries across its open and its close.

A Commit window closes at idle or after ``window_steps`` engine steps (`serve_commit.CommitWindows`), so under load a request
runs in two windows or more. Its state passes from window to window as the compile slice's state chain passes it
(`replay_chain`: a running slot's id, P, O, outputs so far and tokens computed; window w's first state is window w − 1's last):

- `carried`, the window's first requests, in the order the last window held them: each one's record (`RequestAdvice`, as in
  the window that admitted it: id, prompt and output lengths, finish) and its state at the open (`Carried`): ``C`` its
  prefix-cache hit when it was admitted (its Program's), ``computed`` where its next row continues (vLLM's
  num_computed_tokens, or where a preempted request had reached), ``kept`` its kept outputs, and ``made`` the runs of positions
  whose KV it computed (a later source may name them).
- The requests the window admits follow, in order of first row, and its steps and release buckets are I1's.
- The replay is I1's (`_replay`) from the carried state, and a request may end the window unfinished. `close` is the state the
  window leaves: every request with outputs still to keep (the next window must carry it) and every request whose last kept
  output is in the window's last step (async scheduling may run one row past it there, so the next window may carry it).
  `follows` holds a window to the last one's close.

A request's Program is its whole request's, in every window it runs in (`predict.steps`): each window places its own rows.

**A client abort.** A request's finish may be ``abort`` (`RequestAdvice`): its client went away after its ``output_len``
outputs. The endpoint applies an abort only between engine steps (`serve_openai`'s loop, `engine.abort_request`, in process), so
vLLM frees the request (FINISHED_ABORTED: its seat among the running requests and its KV blocks, kept in the prefix cache as any
finished request's) after the step in flight, whose row for it is the one async scheduling runs past its last kept output, as
after a stop. Where it's freed and what it frees are those rows', so the record carries the finish alone.

**The bit layout.** A window that carries nothing in, leaves nothing to carry and aborts no request is I1 v0's, byte for byte
(version 0). One that aborts no request and carries one is version 1, I1 v0's layout with the carried state after the header:

    header   version u8 (1) | n_carried uvar | n_requests uvar | n_steps uvar | w_len u6 | w_rel u6 | w_src u6
    carried  n_carried × [ C u(w_len) | computed u(w_len) | kept u(w_len) | runs uvar | runs × (lo u(w_len) | hi u(w_len)) ]
    request, step, release, padding   as I1 v0's, w_len the least width holding the carried values too

and one that aborts a request is version 2, version 1's layout with each request's finish a u2 (eos 0, length 1, abort 2), one
advice bit more a request. So every window has one encoding, and `decode` refuses any other string: a version 1 one that
version 0 can say, a version 2 one that aborts nothing, a finish of 3.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from verity.core.protocols.certifier import advice as I1
from verity.core.protocols.certifier.advice import (  # the advice's parts, as `predict.batch._solo` reads them
    KINDS,
    WIDTH_BITS,
    MalformedAdvice,
    Output,
    Row,
    Source,
    StepAdvice,
)

SCHEMA = "verity-vllm/window-advice/v1"
VERSION = 1
ABORTS = 2  # the version of a window that aborts a request
FINISHES = (*I1.FINISHES, "abort")


def _int(name: str, value: Any, least: int = 0) -> None:
    if type(value) is not int or not least <= value < I1.LIMIT:
        raise MalformedAdvice(f"{name} must be an int in [{least}, 2^63), not {value!r}")


def _seq(name: str, value: Any, kind: type | tuple[type, ...]) -> tuple:
    kinds = kind if isinstance(kind, tuple) else (kind,)
    what = " or ".join(k.__name__ for k in kinds)
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise MalformedAdvice(f"{name} must be a sequence of {what}, not {type(value).__name__}")
    items = tuple(value)
    for x in items:
        if type(x) not in kinds:
            raise MalformedAdvice(f"{name} holds a {type(x).__name__}, not a {what}")
    return items


@dataclass(frozen=True, slots=True)
class RequestAdvice:
    """One request, as I1's (`advice.RequestAdvice`: its id, prompt and output lengths), whose `finish` is `eos`, `length` or
    `abort` (its client aborted it after its `output_len` outputs, freed at the step of its last row)."""

    id: str
    prompt_len: int
    output_len: int
    finish: str

    def __post_init__(self) -> None:
        if type(self.finish) is not str or self.finish not in FINISHES:
            raise MalformedAdvice(f"finish must be one of {FINISHES}, not {self.finish!r}")
        I1.RequestAdvice(self.id, self.prompt_len, self.output_len, I1.FINISHES[0] if self.finish == "abort" else self.finish)

    def i1(self) -> I1.RequestAdvice:
        if self.finish not in I1.FINISHES:
            raise MalformedAdvice(f"request {self.id!r} is aborted, which I1 v0 can't say (its finishes are {I1.FINISHES})")
        return I1.RequestAdvice(self.id, self.prompt_len, self.output_len, self.finish)


def _request(q: RequestAdvice | I1.RequestAdvice) -> RequestAdvice:
    return q if type(q) is RequestAdvice else RequestAdvice(q.id, q.prompt_len, q.output_len, q.finish)


def _cover(runs: list[list[int]], lo: int, hi: int) -> None:
    """Add [lo, hi) to sorted, disjoint, non-adjacent runs."""
    i = 0
    while i < len(runs) and runs[i][1] < lo:
        i += 1
    j = i
    while j < len(runs) and runs[j][0] <= hi:
        lo, hi = min(lo, runs[j][0]), max(hi, runs[j][1])
        j += 1
    runs[i:j] = [[lo, hi]]


def _covered(runs: list[list[int]], lo: int, hi: int) -> bool:
    return any(a <= lo and hi <= b for a, b in runs)


@dataclass(frozen=True, slots=True)
class Carried:
    """A carried request's state at the window's open: its admission's prefix-cache hit ``C``, ``computed`` where its next row
    continues, ``kept`` its kept outputs, and ``made`` the disjoint, non-adjacent runs [lo, hi) of positions whose KV it
    computed, in order."""

    C: int
    computed: int
    kept: int
    made: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        _int("a carried request's C", self.C)
        _int("a carried request's computed", self.computed)
        _int("a carried request's kept", self.kept)
        made = tuple(tuple(x) for x in _seq("made", self.made, tuple))
        object.__setattr__(self, "made", made)
        for x in made:
            if len(x) != 2:
                raise MalformedAdvice(f"a run of made positions is (lo, hi), not {x!r}")
            _int("a run's lo", x[0])
            _int("a run's hi", x[1], x[0] + 1)
        if any(a[1] >= b[0] for a, b in zip(made, made[1:])):
            raise MalformedAdvice(f"made runs {made} aren't disjoint, non-adjacent and in order")


class Close(NamedTuple):
    """A request the window leaves to the next: its index here, its state at the close, and whether the next window must carry
    it (outputs still to keep) or may (its last kept output is in this window's last step)."""

    request: int
    state: Carried
    must: bool


class _Replay(NamedTuple):
    outputs: tuple[Output, ...]
    first: tuple[int, ...]  # each request's first step here (a carried one's: 0)
    close: tuple[Close, ...]


def _replay(carried: tuple[Carried, ...], requests: tuple[RequestAdvice, ...], steps: tuple[StepAdvice, ...]) -> _Replay:
    """I1's position rules (`advice._replay`) from the carried requests' state, the first ``len(carried)`` requests, with no
    request held to finishing in the window."""
    R, n = len(requests), len(carried)
    if n > R:
        raise MalformedAdvice(f"{n} carried requests of {R}")
    computed, known, kept = [0] * R, [0] * R, [0] * R
    made: list[list[list[int]]] = [[] for _ in range(R)]
    C, last = [0] * R, [-1] * R
    for r, c in enumerate(carried):
        q = requests[r]
        known[r] = q.prompt_len + c.kept
        if not c.C < q.prompt_len or c.kept > q.output_len or c.computed > known[r] or any(hi > known[r] for _, hi in c.made):
            raise MalformedAdvice(f"carried request {r}: C {c.C}, computed {c.computed}, kept {c.kept} and made {c.made} don't fit "
                                  f"a {q.prompt_len}-token prompt with {q.output_len} outputs")
        computed[r], kept[r], C[r], made[r] = c.computed, c.kept, c.C, [list(x) for x in c.made]
    first: list[int] = [0] * n
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
                known[r], C[r] = req.prompt_len, row.cache_hit
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
                last[r] = s
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
    close = tuple(Close(r, Carried(C[r], computed[r], kept[r], tuple(map(tuple, made[r]))), kept[r] < q.output_len)
                  for r, q in enumerate(requests) if kept[r] < q.output_len or (steps and last[r] == len(steps) - 1))
    return _Replay(tuple(outputs), tuple(first), close)


@dataclass(frozen=True, slots=True)
class Advice:
    """One Commit window's advice: its requests (the carried ones first), its steps, each kept output's egress bucket, and the
    carried requests' state at the open."""

    requests: tuple[RequestAdvice, ...]
    steps: tuple[StepAdvice, ...]
    release: tuple[int, ...]
    carried: tuple[Carried, ...] = ()

    def __post_init__(self) -> None:
        requests = tuple(map(_request, _seq("requests", self.requests, (RequestAdvice, I1.RequestAdvice))))
        steps = _seq("steps", self.steps, StepAdvice)
        release = _seq("release", self.release, int)
        carried = _seq("carried", self.carried, Carried)
        for b in release:
            _int("a release bucket", b)
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "release", release)
        object.__setattr__(self, "carried", carried)
        if len({q.id for q in requests}) != len(requests):
            raise MalformedAdvice("two requests share an id")
        replay = _replay(carried, requests, steps)
        if len(release) != len(replay.outputs):
            raise MalformedAdvice(f"{len(release)} release buckets for {len(replay.outputs)} kept outputs")
        last: dict[int, int] = {}
        for o, b in zip(replay.outputs, release):
            if b < last.get(o.request, 0):
                raise MalformedAdvice(f"request {o.request}'s output {o.index} is released at bucket {b}, before its "
                                      f"output {o.index - 1}'s {last[o.request]}")
            last[o.request] = b

    @property
    def version(self) -> int:
        """2 when the window aborts a request, else 0 when it is I1 v0's (nothing carried in, nothing left to carry), else 1."""
        if any(q.finish == "abort" for q in self.requests):
            return ABORTS
        return 0 if not self.carried and not any(c.must for c in close(self)) else VERSION

    def i1(self) -> I1.Advice:
        """The window as I1 v0's `Advice`; only a version 0 window is one."""
        if self.version:
            raise MalformedAdvice("a window that carries a request in or out, or aborts one, isn't I1 v0 advice")
        return I1.Advice(tuple(q.i1() for q in self.requests), self.steps, self.release)


def outputs(advice: Advice) -> tuple[Output, ...]:
    """The window's kept outputs, step by step and row by row: `advice.release[k]` is output k's bucket."""
    return _replay(advice.carried, advice.requests, advice.steps).outputs


def close(advice: Advice) -> tuple[Close, ...]:
    """The state the window leaves to the next (`Close`), in request order."""
    return _replay(advice.carried, advice.requests, advice.steps).close


def follows(last: Advice, advice: Advice) -> None:
    """Refuse ``advice`` unless it opens where ``last``, the window before it, closed: its carried requests are ``last``'s
    `close` in order, each with the record and state it left with, every one that must carry among them."""
    left = list(close(last))
    for k, (q, c) in enumerate(zip(advice.requests, advice.carried)):
        while left and last.requests[left[0].request].id != q.id:
            if left[0].must:
                raise MalformedAdvice(f"request {last.requests[left[0].request].id!r} ends the last window with outputs still to keep, "
                                      f"and this window doesn't carry it")
            left.pop(0)
        if not left:
            raise MalformedAdvice(f"carried request {k} ({q.id!r}) isn't one the last window left, in its order")
        was = left.pop(0)
        if (last.requests[was.request], was.state) != (q, c):
            raise MalformedAdvice(f"carried request {k} ({q.id!r}) opens as {q}, {c}, not as the last window left it: "
                                  f"{last.requests[was.request]}, {was.state}")
    if must := [last.requests[x.request].id for x in left if x.must]:
        raise MalformedAdvice(f"request {must[0]!r} ends the last window with outputs still to keep, and this window doesn't carry it")


class _Widths(NamedTuple):
    req: int
    len: int
    rel: int
    src: int


def _widths(advice: Advice) -> _Widths:
    """The least widths that hold the window's values: I1 v0's, with the carried state among the lengths."""
    lens = [x for q in advice.requests for x in (q.prompt_len, q.output_len)]
    lens += [x for st in advice.steps for row in st.rows for x in (row.start, row.n, row.cache_hit, *(s.n for s in row.sources))]
    lens += [x for c in advice.carried for x in (c.C, c.computed, c.kept, *(y for run in c.made for y in run))]
    src = max((len(row.sources) - 1 for st in advice.steps for row in st.rows if row.cache_hit), default=0)
    return _Widths(max(len(advice.requests) - 1, 0).bit_length(), max(lens, default=0).bit_length(),
                   max(advice.release, default=0).bit_length(), src.bit_length())


class _Writer:
    def __init__(self) -> None:
        self.out, self.acc, self.held, self.bits = bytearray(), 0, 0, 0

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
        self.data, self.pos, self.size = data, 0, 8 * len(data)

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


class _Layout(NamedTuple):
    data: bytes
    header: int
    requests: tuple[int, ...]
    steps: tuple[int, ...]
    release: int


def _layout(advice: Advice) -> _Layout:
    """A version 1 or 2 window's bit string and each part's length in bits."""
    w = _widths(advice)
    version = advice.version
    out = _Writer()
    out.put(version, 8)
    out.uvar(len(advice.carried))
    out.uvar(len(advice.requests))
    out.uvar(len(advice.steps))
    out.put(w.len, WIDTH_BITS)
    out.put(w.rel, WIDTH_BITS)
    out.put(w.src, WIDTH_BITS)
    for c in advice.carried:
        for x in (c.C, c.computed, c.kept):
            out.put(x, w.len)
        out.uvar(len(c.made))
        for lo, hi in c.made:
            out.put(lo, w.len)
            out.put(hi, w.len)
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
        out.put(FINISHES.index(q.finish), 2 if version == ABORTS else 1)
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
    """The window's canonical bit string: I1 v0's for a version 0 window."""
    if type(advice) is not Advice:
        raise MalformedAdvice(f"encode takes a window Advice, not a {type(advice).__name__}")
    return _layout(advice).data if advice.version else I1.encode(advice.i1())


def decode(data: bytes) -> Advice:
    """The window ``data`` encodes; refuses anything but a canonical encoding of a valid window."""
    if not isinstance(data, (bytes, bytearray)):
        raise MalformedAdvice(f"decode takes bytes, not a {type(data).__name__}")
    data = bytes(data)
    rd = _Reader(data)
    version = rd.get(8)
    if version == I1.VERSION:
        a = I1.decode(data)
        return Advice(a.requests, a.steps, a.release)
    if version not in (VERSION, ABORTS):
        raise MalformedAdvice(f"window advice version {version}, not {I1.VERSION}, {VERSION} or {ABORTS}")
    w_finish = 2 if version == ABORTS else 1
    n_carried, n_requests, n_steps = rd.uvar(), rd.uvar(), rd.uvar()
    w_len, w_rel, w_src = rd.get(WIDTH_BITS), rd.get(WIDTH_BITS), rd.get(WIDTH_BITS)
    w_req = max(n_requests - 1, 0).bit_length()
    if 3 * w_len * n_carried + 15 * n_requests + 2 * n_steps > rd.left():
        raise MalformedAdvice(f"truncated: {n_carried} carried, {n_requests} requests and {n_steps} steps in {rd.left()} bits")
    carried = []
    for _ in range(n_carried):
        C, computed, kept = rd.get(w_len), rd.get(w_len), rd.get(w_len)
        runs = rd.uvar()
        if 2 * w_len * runs > rd.left():
            raise MalformedAdvice(f"truncated: {runs} made runs in {rd.left()} bits")
        carried.append(Carried(C, computed, kept, tuple((rd.get(w_len), rd.get(w_len)) for _ in range(runs))))
    requests = []
    for _ in range(n_requests):
        size = rd.uvar()
        if 8 * size > rd.left():
            raise MalformedAdvice(f"truncated: a {size}-byte id in {rd.left()} bits")
        try:
            rid = bytes(rd.get(8) for _ in range(size)).decode("utf-8")
        except UnicodeDecodeError as e:
            raise MalformedAdvice(f"a request's id isn't UTF-8: {e}") from None
        prompt_len, output_len, finish = rd.get(w_len), rd.get(w_len), rd.get(w_finish)
        if finish >= len(FINISHES):
            raise MalformedAdvice(f"request {len(requests)}'s finish {finish} names none of {FINISHES}")
        requests.append(RequestAdvice(rid, prompt_len, output_len, FINISHES[finish]))
    steps = []
    for _ in range(n_steps):
        rows = []
        for _ in range(rd.get(w_req) + 1):
            request, kind, start, n, hit, preempted = rd.get(w_req), KINDS[rd.get(1)], rd.get(w_len), rd.get(w_len), rd.get(w_len), bool(rd.get(1))
            sources = tuple(Source(rd.get(w_req), rd.get(w_len)) for _ in range(rd.get(w_src) + 1)) if hit else ()
            rows.append(Row(request, kind, start, n, hit, preempted, sources))
        steps.append(StepAdvice(tuple(rows)))
    n_outputs = len(_replay(tuple(carried), tuple(requests), tuple(steps)).outputs)
    if w_rel * n_outputs > rd.left():
        raise MalformedAdvice(f"truncated: {n_outputs} release buckets in {rd.left()} bits")
    release = tuple(rd.get(w_rel) for _ in range(n_outputs))
    if rd.left() >= 8 or rd.get(rd.left()):
        raise MalformedAdvice("bytes past the advice, or nonzero padding")
    advice = Advice(tuple(requests), tuple(steps), release, tuple(carried))
    if not advice.version:
        raise MalformedAdvice("a window that carries nothing in or out and aborts nothing is I1 v0's, version 0")
    if advice.version != version:
        raise MalformedAdvice(f"a window that aborts no request is version {advice.version}, not {version}")
    if _widths(advice) != (w_req, w_len, w_rel, w_src):
        raise MalformedAdvice(f"widths (len {w_len}, release {w_rel}, sources {w_src}) aren't the least that hold the window's "
                              f"values, {_widths(advice)[1:]}")
    return advice


def total_bits(advice: Advice) -> int:
    """The window's length in bits, its padding left out."""
    if not advice.version:
        return I1.total_bits(advice.i1())
    lay = _layout(advice)
    return lay.header + sum(lay.requests) + sum(lay.steps) + lay.release * len(advice.release)


def bits_by_window(advice: Advice, T: int) -> dict[int, int]:
    """The window's bits in each egress window of T buckets that holds any, by the earliest one each record can show in (I1's
    rule, `advice.bits_by_window`; the carried state goes with the header). The values sum to `total_bits`."""
    if not advice.version:
        return I1.bits_by_window(advice.i1(), T)
    _int("T", T, 1)
    lay = _layout(advice)
    replay = _replay(advice.carried, advice.requests, advice.steps)
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
        earliest[s] = max(windows) if at is None else at
    add(earliest[0], lay.header)
    for r, n in enumerate(lay.requests):
        add(earliest[replay.first[r]], n)
    for s, n in enumerate(lay.steps):
        add(earliest[s], n)
    for w in windows:
        add(w, lay.release)
    return charge


def to_json(advice: Advice) -> dict:
    """The window as JSON, for debugging: I1's `to_json` for a version 0 window, with ``carried`` and this schema otherwise (an
    aborted request's finish ``abort``); `from_json` reads either back."""
    if not advice.version:
        return I1.to_json(advice.i1())
    return {"schema": SCHEMA,
            "carried": [{"C": c.C, "computed": c.computed, "kept": c.kept, "made": [list(x) for x in c.made]} for c in advice.carried],
            "requests": [{"id": q.id, "prompt_len": q.prompt_len, "output_len": q.output_len, "finish": q.finish} for q in advice.requests],
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
    """The window `to_json` wrote; refuses any other key or shape."""
    if isinstance(doc, dict) and doc.get("schema") == I1.SCHEMA:
        a = I1.from_json(doc)
        return Advice(a.requests, a.steps, a.release)
    d = _keys(doc, ("schema", "carried", "requests", "steps", "release"), "window advice")
    if d["schema"] != SCHEMA:
        raise MalformedAdvice(f"schema {d['schema']!r}, not {SCHEMA!r} or {I1.SCHEMA!r}")
    carried = tuple(Carried(**{**_keys(c, ("C", "computed", "kept", "made"), "a carried request"),
                               "made": tuple(tuple(x) for x in _list(c["made"], "made"))}) for c in _list(d["carried"], "carried"))
    requests = tuple(RequestAdvice(**_keys(q, ("id", "prompt_len", "output_len", "finish"), "a request"))
                     for q in _list(d["requests"], "requests"))
    steps = []
    for st in _list(d["steps"], "steps"):
        rows = []
        for r in _list(_keys(st, ("rows",), "a step")["rows"], "rows"):
            r = _keys(r, ("request", "kind", "start", "n", "cache_hit", "preempted", "sources"), "a row")
            sources = tuple(Source(**_keys(s, ("request", "n"), "a source")) for s in _list(r["sources"], "sources"))
            rows.append(Row(r["request"], r["kind"], r["start"], r["n"], r["cache_hit"], r["preempted"], sources))
        steps.append(StepAdvice(tuple(rows)))
    advice = Advice(requests, tuple(steps), tuple(_list(d["release"], "release")), carried)
    if not advice.version:
        raise MalformedAdvice(f"a window that carries nothing in or out and aborts nothing is I1 v0's ({I1.SCHEMA!r})")
    return advice
