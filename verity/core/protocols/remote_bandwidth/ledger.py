"""The remote bandwidth audit's declared outputs (`PROTOCOL.md` §7, "Declared outputs") from an accepted session's
record: γ against both R, per window and over the session, and the memory line, a session total in three forms, each
with its ε and δ.  They are computed here and nowhere else: the benchmark and every other consumer reads them from
`declared`.

The memory line's forms are the conclusions of `Accounting.Guarantees.Remote.Held` (the snapshot), `HeldOverTime`
(time-held) and `HeldOverTimeLinks` (link-charged), `Security/Models/Accounting/Guarantees/RemoteHeld.lean`, read at
the record.  In the statements' terms, with b a block's bytes and every rate in blocks a ns:

- N, the registered blocks; P, the credited passes; `planned`, their registered receipts (each pass's `lengths` less
  its post-seed reads); W, the most of them wrong (`audit.pooled`'s, at the record's δ); S, S_chip/b;
- X = s/E + β, what can bring a block back into node memory a ns: the node's squaring ceiling over the codec's
  squarings a block (`audit.Rebuild`), plus its disks' read ceiling, plus, link-charged, its boundary links' β_l;
- B = B_r + B_d, what can come back at once beyond that rate, in blocks: one maximal rebuild batch and one maximal
  disk read (plus, link-charged, one link window B_l), `Node`'s bursts;
- snapshot: `planned − W − (X·span + B) − P·S` blocks over the passes, `span` the record's;
- paced, with each window a group g ≤ C, its challenge at s_g, its n_g passes, e_g = max(its last root's arrival,
  s_(g+1)) (the closing window C's: its last arrival), L = max_g (e_g − s_g), and the credited span Σ = [e_0, s_C):
  time-held is `Σ·(N − S) − 2L·(X·(Σ + 2L) + B) − fill` block-ns over Σ, where `fill` spreads W over the groups at
  v_g = (s_(g+1) − s_g)/n_g (L/n_C in C), highest first, at most n_g·N in group g.

The memory line counts registered blocks only.  δ is the horizon's (the record's δ and AuditRO's 2^-slack_log2 a
pass) plus, under sketch-sha512, `timing.sketch_failure` over the session: the statements' δc and δs."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from verity.core.protocols.profile import IntegrityProfile

from . import audit, receipts, timing

SNAPSHOT, TIME_HELD, LINK_CHARGED = "snapshot", "time-held", "link-charged"
FORMS = (SNAPSHOT, TIME_HELD, LINK_CHARGED)
"""The memory line's forms, in order: `Held`'s, `HeldOverTime`'s and `HeldOverTimeLinks`'s."""


def _rate(x: object, what: str) -> float:
    if type(x) not in (int, float) or not 0 <= x < math.inf:
        raise ValueError(f"{what} is a finite nonnegative rate")
    return float(x)


@dataclass(frozen=True, slots=True)
class Node:
    """The verifier's capacities for the node a session audits, each protocol structure, never the prover's report: its
    read bandwidth R at spec, the headline, and measured (every GPU's HBM and the host's DRAM, bytes a second), its
    on-chip storage S_chip (bytes), its squaring ceiling at each codec width (squarings a second), its SHA-512
    compressions a second (the sketch's term), and the read ceilings of its own disks and of its boundary links
    (bytes a second, 0 charging none).  Each way back into node memory also has a burst, the bytes it can bring at once
    beyond its rate (`SquaringCeiling`, `Disks` and `Links` hold over every interval as rate·length + burst): one
    maximal rebuild batch, one maximal disk read and one link window.  A burst of 0 states the rate down to one
    nanosecond, which no device with a nonzero rate meets.  `stated` takes `timing`'s, which a reported read rate can
    only raise."""

    spec_bytes_per_s: float
    read_bytes_per_s: float
    chip_bytes: int
    squarings_per_s: Mapping[str, float]
    sha512_per_s: float = timing.SHA512_PER_S
    disk_bytes_per_s: float = 0.0
    link_bytes_per_s: float = 0.0
    rebuild_burst_bytes: float = 0.0
    disk_burst_bytes: float = 0.0
    link_burst_bytes: float = 0.0

    def __post_init__(self) -> None:
        for name in ("spec_bytes_per_s", "read_bytes_per_s"):
            if not 0 < _rate(getattr(self, name), name):
                raise ValueError(f"{name} is positive")
        if type(self.chip_bytes) is not int or self.chip_bytes < 0:
            raise ValueError("chip_bytes is a nonnegative integer")
        for name in ("sha512_per_s", "disk_bytes_per_s", "link_bytes_per_s", "rebuild_burst_bytes",
                     "disk_burst_bytes", "link_burst_bytes"):
            _rate(getattr(self, name), name)
        squarings = dict(self.squarings_per_s)
        for width, rate in squarings.items():
            _rate(rate, f"the squaring ceiling at {width}")
        object.__setattr__(self, "squarings_per_s", MappingProxyType(squarings))

    @classmethod
    def stated(cls, *, reported_read_bytes_per_s: float = 0.0, disk_bytes_per_s: float = 0.0,
               link_bytes_per_s: float = 0.0, disk_burst_bytes: float | None = None,
               link_burst_bytes: float | None = None) -> Node:
        """`timing`'s node: R measured is the larger of `timing.READ_BYTES_PER_S` and the prover's reported read, R
        at spec the larger of `timing.SPEC_BYTES_PER_S` and that, so a prover reporting a slower read can't shrink
        a term (MEM-24).  Its rebuild burst is S_chip, what a batch in flight holds on the chips; its disk burst one
        second at the disks' ceiling, which is a one-second peak; its link burst `timing.LINK_WINDOW_BYTES` when the
        links are charged."""
        measured = max(timing.READ_BYTES_PER_S, _rate(reported_read_bytes_per_s, "a reported read"))
        disk = disk_bytes_per_s if disk_burst_bytes is None else disk_burst_bytes
        link = (timing.LINK_WINDOW_BYTES if link_bytes_per_s else 0.0) if link_burst_bytes is None else link_burst_bytes
        return cls(max(timing.SPEC_BYTES_PER_S, measured), measured, timing.CHIP_BYTES, timing.SQUARINGS_PER_S,
                   timing.SHA512_PER_S, disk_bytes_per_s, link_bytes_per_s, float(timing.CHIP_BYTES), disk, link)

    def rebuild(self, codec: audit.Codec) -> audit.Rebuild:
        """μ_rebuild's model at the record's codec: the node's squaring ceiling at its width."""
        if codec.field.name not in self.squarings_per_s:
            raise ValueError(f"the node states no squaring ceiling at {codec.field.name}, the record's codec's width")
        return audit.Rebuild(codec, self.squarings_per_s[codec.field.name])

    def to_json(self) -> dict:
        return {"chip_bytes": self.chip_bytes, "disk_burst_bytes": self.disk_burst_bytes,
                "disk_bytes_per_s": self.disk_bytes_per_s, "link_burst_bytes": self.link_burst_bytes,
                "link_bytes_per_s": self.link_bytes_per_s, "read_bytes_per_s": self.read_bytes_per_s,
                "rebuild_burst_bytes": self.rebuild_burst_bytes, "sha512_per_s": self.sha512_per_s,
                "spec_bytes_per_s": self.spec_bytes_per_s, "squarings_per_s": dict(self.squarings_per_s)}


def sketch(record: audit.Record, node: Node) -> float:
    """`RateAuditSketch`'s failure probability for a sketch-sha512 record, every compression the node could make over
    the span a guess or a key (`timing.sketch_failure`); 0 for another receipt kind."""
    if record.receipt != receipts.SKETCH_SHA512:
        return 0.0
    compressions = math.ceil(node.sha512_per_s * record.span_ns / 1e9)
    return float(timing.sketch_failure(compressions, record.passes, record.rows))


def windows(record: audit.Record) -> list[tuple[int, int]]:
    """The spans γ is declared over, each `(from, to)` in ns: each window's own, `[b_(t−1), D_t]`, a stated window
    apart; or a paced session's fixed windows of the stated T (`audit.Pace.window_ns`) from its first challenge, the
    last ending at its last deadline.  Both are protocol structure: when a paced session's coins left is a value."""
    if record.pace is None:
        return [(w.released_ns, w.deadline_ns) for w in record.windows]
    T = record.pace.window_ns
    return [(a, min(a + T, record.end_ns)) for a in range(record.start_ns, record.end_ns, T)]


@dataclass(frozen=True, slots=True)
class Gamma:
    """γ against one R, `capacity_bytes_per_s`: the session's terms (`audit.Ledger`, read per window at Wδ, `gamma`,
    and over the horizon at δ, `gamma_horizon`) and `windows`, the same over each span γ is declared over (`windows`:
    a paced session's fixed windows), each as a fraction of R over its own span.  `error` and `error_horizon` are the
    ledger's with the sketch's term added."""

    capacity_bytes_per_s: float
    ledger: audit.Ledger
    windows: tuple[audit.WindowLedger, ...]
    sketch: float

    @property
    def gamma(self) -> float:
        return self.ledger.gamma

    @property
    def gamma_horizon(self) -> float:
        return self.ledger.gamma_horizon

    @property
    def error(self) -> float:
        return self.ledger.error + self.sketch

    @property
    def error_horizon(self) -> float:
        return self.ledger.error_horizon + self.sketch

    def to_json(self) -> dict:
        return {"capacity_bytes_per_s": self.capacity_bytes_per_s,
                "error": {"horizon": self.error_horizon, "windows": self.error},
                "gamma": self.gamma, "gamma_horizon": self.gamma_horizon,
                "windows": [{"gamma": w.gamma, "gamma_horizon": w.gamma_horizon, "span_ns": w.span_ns,
                             "window": w.window} for w in self.windows]}


def _fixed(record: audit.Record, lg: audit.Ledger, chip_bytes: int, capacity: float) -> tuple[audit.WindowLedger, ...]:
    """The ledger's window terms over a paced record's fixed windows (`windows`).  A pass is in the fixed window that
    holds it whole, from its challenge's release to its root's arrival, and in none if it straddles a boundary; each
    term is its passes': their credited receipts, chip and counting, and of each window they are from, its wrong
    receipts' bytes, read per window and over the horizon, at most their receipts', and its floor."""
    if (why := _arrivals(record)) is not None:
        raise ValueError(f"γ per fixed window: {why}")
    spans, b, start, T = windows(record), record.block_bytes, record.start_ns, record.pace.window_ns
    held: list[dict[int, list[int]]] = [{} for _ in spans]
    for t, w in enumerate(record.windows):
        m = (w.released_ns - start) // T
        for i, arrived in enumerate(w.arrived_ns):
            if arrived <= spans[m][1]:
                held[m].setdefault(t, []).append(i)

    def charged(parts: list[tuple[audit.WindowLedger, int]], name: str) -> float:
        return sum(min(getattr(p, name) * capacity * p.span_ns / 1e9, n * b) for p, n in parts)

    out = []
    for m, ((u, v), passes) in enumerate(zip(spans, held)):
        whole = capacity * (v - u) / 1e9
        ps = [(record.windows[t], i) for t, js in passes.items() for i in js]
        reads = sum(w.lengths[i] for w, i in ps) * b
        if reads > whole:
            raise ValueError(f"fixed window {m} certifies {reads} bytes read, more than the capacity allows in its "
                             "span: the capacity is understated")
        parts = [(lg.windows[t], sum(record.windows[t].lengths[i] for i in js)) for t, js in passes.items()]
        credited = sum(w.credited(False)[i] for w, i in ps) * b
        named = sum(audit.named_bits(w.lengths[i]) + audit.SLACK_LOG2 for w, i in ps)
        out.append(audit.WindowLedger(
            m, v - u, timing=1 - credited / whole, sample=charged(parts, "sample") / whole,
            chip=len(ps) * chip_bytes / whole, counting=named / 8 / whole, rebuild=lg.rebuild, host=lg.host,
            link=lg.link, carrier=charged(parts, "carrier") / whole, horizon=charged(parts, "horizon") / whole,
            floor=max((p.floor for p, _ in parts), default=0.0)))
    return tuple(out)


def gamma(record: audit.Record, node: Node, capacity_bytes_per_s: float, *,
          wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound) -> Gamma:
    """γ against `capacity_bytes_per_s`, over the registered store only (nothing appended after the codec seed is
    credited), with the node's S_chip and squaring ceiling (`audit.ledger`), and per window as declared (`windows`):
    a paced record's needs each root's arrival, in order."""
    lg = audit.ledger(record, capacity_bytes_per_s=capacity_bytes_per_s, chip_bytes=node.chip_bytes,
                      rebuild=node.rebuild(record.codec), wrong=wrong)
    ws = lg.windows if record.pace is None else _fixed(record, lg, node.chip_bytes, capacity_bytes_per_s)
    return Gamma(capacity_bytes_per_s, lg, ws, sketch(record, node))


@dataclass(frozen=True, slots=True)
class Line:
    """One form of the memory line: the registered bytes the session shows held in node memory are at least
    `held_bytes` = (1 − ε)·N·b, except with probability `delta`.  `lost_blocks` is the blocks of N it charges, on
    average over the passes (snapshot) or over the credited span (the paced forms), and ε their share of N; or
    `refused`, why the session has no line of this form."""

    form: str
    blocks: int
    block_bytes: int
    delta: float
    lost_blocks: float | None = None
    refused: str | None = None

    @property
    def epsilon(self) -> float | None:
        return None if self.lost_blocks is None else min(1.0, max(0.0, self.lost_blocks / self.blocks))

    @property
    def held_bytes(self) -> float | None:
        return None if self.epsilon is None else (1 - self.epsilon) * self.blocks * self.block_bytes

    def to_json(self) -> dict:
        if self.refused is not None:
            return {"form": self.form, "refused": self.refused}
        return {"delta": self.delta, "epsilon": self.epsilon, "form": self.form, "held_bytes": self.held_bytes,
                "lost_blocks": self.lost_blocks}


def _arrivals(record: audit.Record) -> str | None:
    """Why a paced record's arrivals can't be read, or None: every root's, in order from its window's challenge to its
    deadline, on one clock."""
    if any(len(w.arrived_ns) != w.passes for w in record.windows):
        return "a paced record is read at each root's arrival, and this record has none: read a session's own record"
    for w in record.windows:
        if any(a > b for a, b in zip((w.released_ns, *w.arrived_ns), (*w.arrived_ns, w.deadline_ns))):
            return (f"window {w.window}'s arrivals are not in order from its challenge to its deadline: a paced "
                    "record's are read on one clock")
    return None


def _paced(record: audit.Record) -> str | None:
    """Why the record has no time-held line, or None: it must be paced, with a closing window after the first, every
    root's arrival in order (`_arrivals`), every pass reading every registered block (`Covers`: a block a pass leaves
    out can be absent across it with neither a wrong read nor a rebuild), and a credited span."""
    if record.pace is None:
        return "an unpaced session has no time-held line: its windows' coins come a stated window apart"
    if len(record.windows) < 2:
        return ("a record of one window has no time-held line: the credited span runs from the first window's end to "
                "the closing window's challenge")
    if (why := _arrivals(record)) is not None:
        return why
    for w in record.windows:
        if (short := next((n for n in w.credited(False) if n != record.blocks), None)) is not None:
            return (f"a pass of window {w.window} read {short} of the {record.blocks} registered blocks: the "
                    "time-held line needs every pass to read every registered block")
    if record.windows[-1].released_ns <= max(record.windows[0].arrived_ns[-1], record.windows[1].released_ns):
        return "the credited span is empty: the closing window's challenge left before the first window ended"
    return None


def memory(record: audit.Record, node: Node) -> tuple[Line, Line, Line]:
    """The memory line's three forms (`FORMS`)."""
    N, b, P = record.blocks, record.block_bytes, record.passes
    W = audit.pooled(record)[0]
    delta = record.windows[0].profile.delta + P * 2.0 ** -audit.SLACK_LOG2 + sketch(record, node)
    S = node.chip_bytes / b
    rebuild = node.rebuild(record.codec)
    X = rebuild.squarings_per_s / record.codec.encode_squarings / 1e9 + node.disk_bytes_per_s / b / 1e9
    X_link = X + node.link_bytes_per_s / b / 1e9
    Br, Bd, Bl = node.rebuild_burst_bytes / b, node.disk_burst_bytes / b, node.link_burst_bytes / b
    planned = sum(sum(w.credited(False)) for w in record.windows)
    snapshot = Line(SNAPSHOT, N, b, delta, N - (planned - W - (X * record.span_ns + (Br + Bd)) - P * S) / P)
    if (why := _paced(record)) is not None:
        return snapshot, Line(TIME_HELD, N, b, delta, refused=why), Line(LINK_CHARGED, N, b, delta, refused=why)
    ws = record.windows
    C = len(ws) - 1
    s = [w.released_ns for w in ws]
    n = [w.passes for w in ws]
    e = [max(w.arrived_ns[-1], s[g + 1]) if g < C else w.arrived_ns[-1] for g, w in enumerate(ws)]
    L = max(eg - sg for eg, sg in zip(e, s))
    span = s[C] - e[0]
    v = [(s[g + 1] - s[g]) / n[g] if g < C else L / n[g] for g in range(C + 1)]
    left, fill = W, 0.0
    for g in sorted(range(C + 1), key=lambda g: v[g], reverse=True):
        took = min(left, n[g] * N)
        fill, left = fill + took * v[g], left - took
        if left <= 0:
            break

    def lost(x: float, B: float) -> float:
        return S + (2 * L * (x * (span + 2 * L) + B) + fill) / span

    return (snapshot, Line(TIME_HELD, N, b, delta, lost(X, Br + Bd)),
            Line(LINK_CHARGED, N, b, delta, lost(X_link, Br + Bd + Bl)))


@dataclass(frozen=True, slots=True)
class Declared:
    """An accepted session's declared outputs: its verdict (every window accepted: a rejected session's is the
    `audit.Verdict` that ended it, and it declares nothing else), γ against the node's specified R (`spec`, the
    headline) and its measured R (`measured`), and the memory line's forms (`memory`, in `FORMS`' order)."""

    windows: int
    spec: Gamma
    measured: Gamma
    memory: tuple[Line, Line, Line]

    def to_json(self) -> dict:
        return {"accepted": True, "gamma": {"measured": self.measured.to_json(), "spec": self.spec.to_json()},
                "memory": {x.form: x.to_json() for x in self.memory}, "windows": self.windows}


def declared(record: audit.Record, node: Node, *,
             wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound) -> Declared:
    """`record`'s declared outputs on `node`.  `wrong` is each window's bound on its wrong receipts
    (`IntegrityProfile.bound`, computed once a window for both readings)."""
    bounds = {id(w.profile): wrong(w.profile) for w in record.windows}
    once = lambda p: bounds[id(p)]  # noqa: E731
    return Declared(len(record.windows), gamma(record, node, node.spec_bytes_per_s, wrong=once),
                    gamma(record, node, node.read_bytes_per_s, wrong=once), memory(record, node))
