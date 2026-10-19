"""The remote bandwidth audit's declared outputs (`PROTOCOL.md` §7, "Declared outputs") from an accepted session's
record: γ against both R, per window and over the session, and the memory line, a session total in three forms, each
with its ε and δ.  They are computed here and nowhere else: the benchmark and every other consumer reads them from
`declared`.

The memory line's forms are the conclusions of `Accounting.Guarantees.Remote.Held` (the snapshot), `HeldOverTime`
(time-held) and `HeldOverTimeLinks` (link-charged), `Security/Models/Accounting/Guarantees/RemoteHeld.lean`, read at
the record.  In the statements' terms, with b a block's bytes and every rate in blocks a ns:

- N, the registered blocks; P, the credited passes; `planned`, their registered receipts (each pass's `lengths` less
  its post-seed reads); W, the most of them wrong (`audit.pooled`'s, at the record's δ); S, S_chip/b;
- X, what can bring a block back into node memory a ns: the node's squaring ceiling over the codec's squarings a block
  (`audit.Rebuild`), plus its disks' read ceiling, plus, link-charged, its boundary links';
- snapshot: `planned − W − X·span − P·S` blocks over the passes, `span` the record's;
- paced, with each window a group g ≤ C, its challenge at s_g, its n_g passes, e_g = max(its last root's arrival,
  s_(g+1)) (the closing window C's: its last arrival), L = max_g (e_g − s_g), and the credited span Σ = [e_0, s_C):
  time-held is `Σ·(N − S) − 2L·X·(Σ + 2L) − fill` block-ns over Σ, where `fill` spreads W over the groups at
  v_g = (s_(g+1) − s_g)/n_g (L/n_C in C), highest first, at most n_g·N in group g.

The memory line counts registered blocks only.  δ is the horizon's (the record's δ and AuditRO's 2^-slack_log2 a
pass) plus, under sketch-sha512, `timing.sketch_failure` over the session: the statements' δc and δs."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
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
    (bytes a second, 0 charging none).  `stated` takes `timing`'s, which a reported read rate can only raise."""

    spec_bytes_per_s: float
    read_bytes_per_s: float
    chip_bytes: int
    squarings_per_s: Mapping[str, float]
    sha512_per_s: float = timing.SHA512_PER_S
    disk_bytes_per_s: float = 0.0
    link_bytes_per_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("spec_bytes_per_s", "read_bytes_per_s"):
            if not 0 < _rate(getattr(self, name), name):
                raise ValueError(f"{name} is positive")
        if type(self.chip_bytes) is not int or self.chip_bytes < 0:
            raise ValueError("chip_bytes is a nonnegative integer")
        for name in ("sha512_per_s", "disk_bytes_per_s", "link_bytes_per_s"):
            _rate(getattr(self, name), name)
        squarings = dict(self.squarings_per_s)
        for width, rate in squarings.items():
            _rate(rate, f"the squaring ceiling at {width}")
        object.__setattr__(self, "squarings_per_s", MappingProxyType(squarings))

    @classmethod
    def stated(cls, *, reported_read_bytes_per_s: float = 0.0, disk_bytes_per_s: float = 0.0,
               link_bytes_per_s: float = 0.0) -> Node:
        """`timing`'s node: R measured is the larger of `timing.READ_BYTES_PER_S` and the prover's reported read, R
        at spec the larger of `timing.SPEC_BYTES_PER_S` and that, so a prover reporting a slower read can't shrink
        a term (MEM-24)."""
        measured = max(timing.READ_BYTES_PER_S, _rate(reported_read_bytes_per_s, "a reported read"))
        return cls(max(timing.SPEC_BYTES_PER_S, measured), measured, timing.CHIP_BYTES, timing.SQUARINGS_PER_S,
                   timing.SHA512_PER_S, disk_bytes_per_s, link_bytes_per_s)

    def rebuild(self, codec: audit.Codec) -> audit.Rebuild:
        """μ_rebuild's model at the record's codec: the node's squaring ceiling at its width."""
        if codec.field.name not in self.squarings_per_s:
            raise ValueError(f"the node states no squaring ceiling at {codec.field.name}, the record's codec's width")
        return audit.Rebuild(codec, self.squarings_per_s[codec.field.name])

    def to_json(self) -> dict:
        return {"chip_bytes": self.chip_bytes, "disk_bytes_per_s": self.disk_bytes_per_s,
                "link_bytes_per_s": self.link_bytes_per_s, "read_bytes_per_s": self.read_bytes_per_s,
                "sha512_per_s": self.sha512_per_s, "spec_bytes_per_s": self.spec_bytes_per_s,
                "squarings_per_s": dict(self.squarings_per_s)}


def sketch(record: audit.Record, node: Node) -> float:
    """`RateAuditSketch`'s failure probability for a sketch-sha512 record, every compression the node could make over
    the span a guess or a key (`timing.sketch_failure`); 0 for another receipt kind."""
    if record.receipt != receipts.SKETCH_SHA512:
        return 0.0
    compressions = math.ceil(node.sha512_per_s * record.span_ns / 1e9)
    return float(timing.sketch_failure(compressions, record.passes, record.rows))


def windows(record: audit.Record) -> list[range]:
    """The record's windows as γ declares them: each window alone, or a paced session's stretches
    (`audit.Pace.stretches`)."""
    n = len(record.windows)
    return [range(t, t + 1) for t in range(n)] if record.pace is None else record.pace.stretches(n)


@dataclass(frozen=True, slots=True)
class Gamma:
    """γ against one R, `capacity_bytes_per_s`: the session's terms (`audit.Ledger`, read per window at Wδ, `gamma`,
    and over the horizon at δ, `gamma_horizon`) and `windows`, the same over each window as declared (`windows`: a
    paced session's stretches), each as a fraction of R over its own span.  `error` and `error_horizon` are the
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


def _stretch(record: audit.Record, lg: audit.Ledger, ws: range, index: int, capacity: float) -> audit.WindowLedger:
    """The ledger's window terms over windows `ws` together, its span from the first's challenge to the last's
    deadline: each term's bytes summed over them, the timing term from their credited receipts."""
    parts, recs = [lg.windows[t] for t in ws], [record.windows[t] for t in ws]
    if len(parts) == 1:
        return replace(parts[0], window=index)
    span = recs[-1].deadline_ns - recs[0].released_ns
    whole = capacity * span / 1e9

    def summed(name: str) -> float:
        return sum(getattr(p, name) * capacity * p.span_ns / 1e9 for p in parts) / whole

    credited = sum(sum(w.credited(False)) for w in recs) * record.block_bytes
    return audit.WindowLedger(index, span, timing=1 - credited / whole, sample=summed("sample"), chip=summed("chip"),
                              counting=summed("counting"), rebuild=lg.rebuild, host=lg.host, link=lg.link,
                              carrier=summed("carrier"), horizon=summed("horizon"),
                              floor=max(p.floor for p in parts))


def gamma(record: audit.Record, node: Node, capacity_bytes_per_s: float, *,
          wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound) -> Gamma:
    """γ against `capacity_bytes_per_s`, over the registered store only (nothing appended after the codec seed is
    credited), with the node's S_chip and squaring ceiling (`audit.ledger`), and per window as declared."""
    lg = audit.ledger(record, capacity_bytes_per_s=capacity_bytes_per_s, chip_bytes=node.chip_bytes,
                      rebuild=node.rebuild(record.codec), wrong=wrong)
    return Gamma(capacity_bytes_per_s, lg,
                 tuple(_stretch(record, lg, ws, m, capacity_bytes_per_s) for m, ws in enumerate(windows(record))),
                 sketch(record, node))


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


def _paced(record: audit.Record) -> str | None:
    """Why the record has no time-held line, or None: it must be paced, with a closing window after the first, every
    root's arrival, in order from its window's challenge to its deadline, every pass reading every registered block
    (`Covers`: a block a pass leaves out can be absent across it with neither a wrong read nor a rebuild), and a
    credited span."""
    if record.pace is None:
        return "an unpaced session has no time-held line: its windows' coins come a stated window apart"
    if len(record.windows) < 2:
        return ("a record of one window has no time-held line: the credited span runs from the first window's end to "
                "the closing window's challenge")
    if any(len(w.arrived_ns) != w.passes for w in record.windows):
        return "the time-held line reads each root's arrival, and this record has none: read a session's own record"
    for w in record.windows:
        if any(a > b for a, b in zip((w.released_ns, *w.arrived_ns), (*w.arrived_ns, w.deadline_ns))):
            return (f"window {w.window}'s arrivals are not in order from its challenge to its deadline: the time-held "
                    "line reads them on one clock")
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
    planned = sum(sum(w.credited(False)) for w in record.windows)
    snapshot = Line(SNAPSHOT, N, b, delta, N - (planned - W - X * record.span_ns - P * S) / P)
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

    def lost(x: float) -> float:
        return S + (2 * L * x * (span + 2 * L) + fill) / span

    return snapshot, Line(TIME_HELD, N, b, delta, lost(X)), Line(LINK_CHARGED, N, b, delta, lost(X_link))


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
