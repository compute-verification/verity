"""The remote bandwidth audit's declared outputs (`PROTOCOL.md` §7, "Declared outputs") from an accepted session's
record: γ against both R, per window and over the session, and the memory line, a session total in three forms, each
with its ε and δ.  They are computed here, the memory line's charges and paced forms by `line`'s arithmetic in floats,
and nowhere else: the benchmark and every other consumer reads them from `declared`.

The memory line's forms are the conclusions of `Accounting.Guarantees.Remote.Held` (the snapshot), `HeldOverTime`
(time-held) and `HeldOverTimeLinks` (link-charged), `Security/Definitions/Accounting/Guarantees/RemoteHeld.lean`, read at
the record.  In the statements' terms, with b a block's bytes and every rate in blocks a ns:

- N, the registered blocks; P, the credited passes; `planned`, their registered receipts (each pass's `lengths` less
  its post-seed reads); W, the most of them wrong (`audit.pooled`'s, at the record's δ); S, S_chip/b rounded up;
- X = s/E + β, what can bring a block back into node memory a ns: the node's squaring ceiling over the codec's
  squarings a block (`audit.Rebuild`), plus its disks' read ceiling, plus, link-charged, its boundary links' β_l;
- B = B_r + B_d, what can come back at once beyond that rate, in blocks: one maximal rebuild batch and one maximal
  disk read (plus, link-charged, one link window B_l), `Node`'s bursts (X and B: `line.charges_of`);
- snapshot: `planned − W − (X·span + B) − P·S` blocks over the passes, `span` the record's;
- paced, with each window a group g ≤ C and the credited span Σ = [e_0, s_C): time-held is
  `Σ·(N − S) − 2L·(X·(Σ + 2L) + B) − fill` block-ns over Σ, the line's stretch held over time with the whole session
  one stretch (`line.session_lost`: e_g, L and `fill` as `line` defines them).

The memory line counts registered blocks only.  δ is the horizon's (the record's δ and AuditRO's 2^-slack_log2 a
pass) plus the coin commitment's hiding terms (`hiding`), the receipts' binding term (`binding`), under sketch-sha512
`timing.sketch_failure` over the session, and the draw's rejection (`rejection`): the statements' δc and δs, and
`LineHeld`'s Σ_t hide_t, bind and Σ_t k_t/2^64 at the record's `samples`.  `LineHeld`'s error also carries the
windows' PRF advantages, which are named (`PRF_CLAIM`) and in no δ here.  γ's errors carry `PooledDraw`'s terms beyond
δ by name (`draw`), which have no hiding term since the prover is sent the coin commitment only at the end, the
advantages named with their claim and priced in neither."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from verity.core.protocols.profile import IntegrityProfile

from . import audit, coins, line, receipts, timing

SNAPSHOT = "snapshot"
TIME_HELD, LINK_CHARGED = line.FORMS
FORMS = (SNAPSHOT, *line.FORMS)
"""The memory line's forms, in order: `Held`'s, `HeldOverTime`'s and `HeldOverTimeLinks`'s."""
REJECTION_LOG2 = 64
"""The draw's rejection a draw, in bits: a draw below U reads a word of `bitlen(U) + 64` bits or more (`Key._uniform`,
`Accounting.Remote.Draw.width`), so a uniform first word is rejected with probability below 2^-64."""
DELTA_PARTS = ("profile", "slack", "hiding", "binding", "sketch", "rejection")
"""The memory line's δ parts (`delta_parts`), in the order `memory` adds them."""
PRF_CLAIM = "prf/sha-512"
"""The claim that prices `PooledDraw`'s and `LineHeld`'s Σ_t Adv(prf t), one oracle-access distinguisher a window
reading its k_t first blocks: a term with no numeric value, so it is named beside δ and γ's errors and in none of
them."""


def _rate(x: object, what: str) -> float:
    if type(x) not in (int, float) or not 0 <= x < math.inf:
        raise ValueError(f"{what} is a finite nonnegative rate")
    return float(x)


@dataclass(frozen=True, slots=True)
class Node:
    """The verifier's capacities for the node a session audits, each protocol structure, never the prover's report: its
    read bandwidth R at spec, the headline, and measured (every GPU's HBM and the host's DRAM, bytes a second), its
    on-chip storage S_chip (bytes), its squaring ceiling at each codec width (squarings a second), its SHA-512
    compressions a second and the outside-compute factor F over them (the sketch's term, `timing.SKETCH_OUTSIDE`: the
    prover's compressions over a session at most F times the node's), and the read ceilings of its own disks and of
    its boundary links (bytes a second, 0 charging none).  Each way back into node memory also has a burst, the bytes it can bring at once
    beyond its rate (`SquaringCeiling`, `Disks` and `Links` hold over every interval as rate·length + burst): one
    maximal rebuild batch, one maximal disk read and one link window.  A burst of 0 states the rate down to one
    nanosecond, which no device with a nonzero rate meets.  Beside the read ceilings are the write ceilings, one GPU's
    and the host DRAM's (bytes a second), how fast a fill can write labels into each tier, which a wipe's timing reads;
    the node states no DPU's (`timing.HOST_WRITE_BYTES_PER_S`).  `stated` takes `timing`'s, which a reported read rate
    can only raise."""

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
    gpu_write_bytes_per_s: float = timing.GPU_WRITE_BYTES_PER_S
    host_write_bytes_per_s: float = timing.HOST_WRITE_BYTES_PER_S
    sketch_outside: int = timing.SKETCH_OUTSIDE

    def __post_init__(self) -> None:
        for name in ("spec_bytes_per_s", "read_bytes_per_s", "gpu_write_bytes_per_s", "host_write_bytes_per_s"):
            if not 0 < _rate(getattr(self, name), name):
                raise ValueError(f"{name} is positive")
        if type(self.chip_bytes) is not int or self.chip_bytes < 0:
            raise ValueError("chip_bytes is a nonnegative integer")
        if type(self.sketch_outside) is not int or self.sketch_outside < 1:
            raise ValueError("sketch_outside, the outside-compute factor F, is a positive integer")
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
               link_burst_bytes: float | None = None, sketch_outside: int = timing.SKETCH_OUTSIDE) -> Node:
        """`timing`'s node: R measured is the larger of `timing.READ_BYTES_PER_S` and the prover's reported read, R
        at spec the larger of `timing.SPEC_BYTES_PER_S` and that, so a prover reporting a slower read can't shrink
        a term.  Its rebuild burst is S_chip, what a batch in flight holds on the chips; its disk burst one
        second at the disks' ceiling, which is a one-second peak; its link burst `timing.LINK_WINDOW_BYTES` when the
        links are charged."""
        measured = max(timing.READ_BYTES_PER_S, _rate(reported_read_bytes_per_s, "a reported read"))
        disk = disk_bytes_per_s if disk_burst_bytes is None else disk_burst_bytes
        link = (timing.LINK_WINDOW_BYTES if link_bytes_per_s else 0.0) if link_burst_bytes is None else link_burst_bytes
        return cls(max(timing.SPEC_BYTES_PER_S, measured), measured, timing.CHIP_BYTES, timing.SQUARINGS_PER_S,
                   timing.SHA512_PER_S, disk_bytes_per_s, link_bytes_per_s, float(timing.CHIP_BYTES), disk, link,
                   timing.GPU_WRITE_BYTES_PER_S, timing.HOST_WRITE_BYTES_PER_S, sketch_outside)

    def rebuild(self, codec: audit.Codec) -> audit.Rebuild:
        """μ_rebuild's model at the record's codec: the node's squaring ceiling at its width."""
        if codec.field.name not in self.squarings_per_s:
            raise ValueError(f"the node states no squaring ceiling at {codec.field.name}, the record's codec's width")
        return audit.Rebuild(codec, self.squarings_per_s[codec.field.name])

    def to_json(self) -> dict:
        return {"chip_bytes": self.chip_bytes, "disk_burst_bytes": self.disk_burst_bytes,
                "disk_bytes_per_s": self.disk_bytes_per_s, "gpu_write_bytes_per_s": self.gpu_write_bytes_per_s,
                "host_write_bytes_per_s": self.host_write_bytes_per_s, "link_burst_bytes": self.link_burst_bytes,
                "link_bytes_per_s": self.link_bytes_per_s, "read_bytes_per_s": self.read_bytes_per_s,
                "rebuild_burst_bytes": self.rebuild_burst_bytes, "sha512_per_s": self.sha512_per_s,
                "sketch_outside": self.sketch_outside, "spec_bytes_per_s": self.spec_bytes_per_s,
                "squarings_per_s": dict(self.squarings_per_s)}


def sketch(record: audit.Record, node: Node) -> float:
    """`SketchBound`'s δ for a sketch-sha512 record (`Accounting.Guarantees.Remote.SketchBoundDeployed`), every
    compression the prover could make over the span a guess or a key: Q = F·C, C the node's over the span and F its
    `sketch_outside` (`timing.sketch_failure`); 0 for another receipt kind. It is the δs of the memory line's
    `SketchBoundAt` premise (`LineHeld`): `SketchBoundDeployed` gives the sketch bound at it on the line's trace read on
    the sketch outcomes, and `Accounting.Guarantees.Remote.SketchBoundLine` carries it to the line's coins."""
    if record.receipt != receipts.SKETCH_SHA512:
        return 0.0
    return float(timing.sketch_failure(compressions(record, node), record.passes, record.rows, blocks=record.blocks,
                                       outside=node.sketch_outside))


OPENING_HASHES = 64 + 5
"""The SHA-512 evaluations one receipt opening's check makes at most (`Hash.RowLeaf.outerHashed` and `innerHashed`): the
path's nodes, at most 64 since a rank is a u64, the frame leaf, the hm96 leaf, its salt's digest and its key's, and the
row digest."""


def rejection(record: audit.Record) -> float:
    """The draw's rejection, `PooledDraw`'s and `LineHeld`'s Σ_t k_t/2^64: 2^-`REJECTION_LOG2` a receipt draw t
    requested, k_t the record's `samples` in every draw (the k at which `DrawsOfRecord` holds), one draw a window, or
    ⌈W/g⌉ draws at `groups_per_coin` g.  Not the receipts drawn (`profile.drawn`, min(k_t, U_t), short of k_t in a draw
    that opens all its receipts): the statement's error is a constant, and a draw's U_t is the prover's.  Refused for a
    record that states no `samples`.  Exact: the draws are an integer."""
    if record.samples is None:
        raise ValueError("the record states no samples, the receipts every draw requested, at which the memory line's "
                         "δ charges the draw's rejection (Σ_t k_t/2^64): give the record its session's Params.samples")
    return coins.openings_coins(len(record.windows), record.groups_per_coin) * record.samples * 2.0 ** -REJECTION_LOG2


def hiding(record: audit.Record) -> float:
    """The coin commitment's hiding terms over the record's n windows, `LineHeld`'s Σ_t hide_t (`Committed.hide`):
    window t's two distances between coin t's hm96 leaf and a leaf committing zeros, each within 2^-192 under the
    pinned key, n·2^-191 in all (claims `statistical-hiding`, `common-reference-string`, `hash-derived-key`).  The
    prover is sent the commitment only at the session's end (`coins.py`), so `PooledDraw` reads no hiding term and
    `draw` charges none; the memory line's δ keeps it while `LineHeld` states it."""
    return hiding_over(len(record.windows))


def hiding_over(windows: int) -> float:
    """`hiding` over a session of `windows` windows."""
    return 2 * 2.0 ** receipts.HM96.statistical_distance_log2(windows, key_failure_log2=64)


def compressions(record: audit.Record, node: Node) -> int:
    """C, the node's SHA-512 compressions over the record's span, at its stated rate."""
    return math.ceil(node.sha512_per_s * record.span_ns / 1e9)


def window_receipts(pace: audit.Pace | None, blocks: int, append_coins: int, max_blocks: int) -> int:
    """|L_t|, the most receipts a window of a session can have at any outcome, from its protocol structure: its passes,
    at most the pace's k (a (k + 1)-th root is refused as `order`) or, unpaced, fewer than `audit.STEP_LIMIT` (each
    pass's step is above the last's and below it), each receipting at most the store's extent once (a plan repeats no
    block): the registered `blocks` when the session reserved no append coins, so no append takes it past them, and
    the codec's `max_blocks` when it did."""
    passes = audit.STEP_LIMIT - 1 if pace is None else pace.k
    return passes * (blocks if append_coins == 0 else max_blocks)


def binding_over(draws: int, receipts: int, samples: int, queries: int) -> float:
    """`binding` over a session of `draws` draws, at most `receipts` receipts a draw (`window_receipts` times the
    windows a draw covers) and a draw cap of `samples`, its prover making at most `queries` SHA-512 compressions over
    the span."""
    s = min(samples, receipts)
    return draws * math.sqrt(receipts * s) * (2 * queries + 2 * s * OPENING_HASHES) / 2 ** 256.5


def binding(record: audit.Record, node: Node) -> float:
    """The receipts' binding term (`Accounting.Remote.Draw.Receipts.binding`): the probability that some opened
    receipt was presented with an accepted row other than the one its root holds (`Receipts.held`, the plurality of
    the rows it shows over its window's continuations from the cut before its coin).  Window t's fork finder runs the
    prover twice from that cut on independent openings coins and checks both runs' openings, q_t = 2·F·C +
    2·s_t·`OPENING_HASHES` SHA-512 calls, F·C the most compressions the prover makes over the span (`sketch`'s); under
    claim `cr/sha-512` it finds a collision with probability ε_t ≤ q_t²/2^513.
    `Accounting.Guarantees.Remote.ReceiptsBindingOfCR` proves the term at most Σ_t √(|L_t|·s_t·(ε_t + split_t)),
    L_t a set holding every receipt window t opens at every outcome, s_t the session's draw cap and split_t the chance
    that the two runs' roots differ.  Both are the session's bounds, not one outcome's U_t and s'_t, which follow the
    prover's play: this charges them at `window_receipts` and the record's `samples`.  At `groups_per_coin` g > 1 the
    term's t is a draw, one openings coin over g windows (`LineHeld`'s `Receipts` over `Node.draws`): ⌈W/g⌉ of them,
    each opening at most g·|L_t| receipts, its windows' together, at the cap s over them.  Where the cut fixes the
    roots, split_t = 0 and the term is Σ_t √(|L_t|·s_t)·q_t/2^256.5 (`Binding.binding_le_of_fixed`), this charge.  The
    prover is sent no coin commitment before the session's end (`coins.Released`, `audit.AuditRun.disclose`), so
    nothing it holds before a window's openings coin is released depends on that coin; the statement that the play
    then fixes each window's roots by its cut is not yet in Lean, and this charge takes split_t as 0 (the gap
    `binding-split`, `PROTOCOL.md` §8)."""
    if record.samples is None or record.append_coins is None:
        raise ValueError("the binding term is charged at the session's bounds, its draw cap and its append coins, "
                         "which a record no session made doesn't state")
    g = record.groups_per_coin
    L = window_receipts(record.pace, record.blocks, record.append_coins, record.codec.max_blocks)
    return binding_over(coins.openings_coins(len(record.windows), g), g * L, record.samples,
                        node.sketch_outside * compressions(record, node))


@dataclass(frozen=True, slots=True)
class SamplingTerms:
    """`PooledDraw`'s terms beyond its δ, each by name: the rejection (`rejection`) and the receipts' binding term
    (`binding`), what reading the record's openings as the deployed draw over receipts its roots hold costs.  `total`
    is their sum.  The windows' PRF advantages Σ_t Adv(prf t) are the third term, which `PRF_CLAIM` prices with no
    numeric value: `to_json` names it, and `total` holds none of it."""

    rejection: float
    binding: float

    @property
    def total(self) -> float:
        return self.rejection + self.binding

    def to_json(self) -> dict:
        return {"binding": self.binding, "rejection": self.rejection,
                "named": {"prf_advantage": {"claim": PRF_CLAIM, "term": "Σ_t Adv(prf t)", "value": None,
                                            "in_total": False}}}


def draw(record: audit.Record, node: Node) -> SamplingTerms:
    """The record's draw terms on `node`."""
    return SamplingTerms(rejection(record), binding(record, node))


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
    """γ against one R, `capacity_bytes_per_s`: the session's terms (`audit.GammaTerms`, read per window at Wδ, `gamma`,
    and over the horizon at δ, `gamma_horizon`) and `windows`, the same over each span γ is declared over (`windows`:
    a paced session's fixed windows), each as a fraction of R over its own span.  `error` and `error_horizon` are the
    ledger's with the sketch's term and the draw's (`draw`: `PooledDraw`'s rejection and binding terms, the PRF
    advantages named) added: both readings read the record's openings, each window's or pooled."""

    capacity_bytes_per_s: float
    ledger: audit.GammaTerms
    windows: tuple[audit.WindowLedger, ...]
    sketch: float
    draw: SamplingTerms

    @property
    def gamma(self) -> float:
        return self.ledger.gamma

    @property
    def gamma_horizon(self) -> float:
        return self.ledger.gamma_horizon

    @property
    def error(self) -> float:
        return self.ledger.error + self.sketch + self.draw.total

    @property
    def error_horizon(self) -> float:
        return self.ledger.error_horizon + self.sketch + self.draw.total

    def to_json(self) -> dict:
        return {"capacity_bytes_per_s": self.capacity_bytes_per_s,
                "error": {"draw": self.draw.to_json(), "horizon": self.error_horizon, "sketch": self.sketch,
                          "windows": self.error},
                "gamma": self.gamma, "gamma_horizon": self.gamma_horizon,
                "windows": [{"gamma": w.gamma, "gamma_horizon": w.gamma_horizon, "span_ns": w.span_ns,
                             "window": w.window} for w in self.windows]}


def _fixed(record: audit.Record, lg: audit.GammaTerms, chip_bytes: int,
           capacity: float) -> tuple[audit.WindowLedger, ...]:
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
    return Gamma(capacity_bytes_per_s, lg, ws, sketch(record, node), draw(record, node))


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


def delta_parts(record: audit.Record, node: Node) -> dict[str, float]:
    """The memory line's δ (`memory`'s `Line.delta`, its parts added in `DELTA_PARTS`' order) in its six parts: the
    windows' profile δ, AuditRO's 2^-SLACK_LOG2 a pass over the record's passes, the coin commitment's hiding terms
    (`hiding`), the receipts' binding term at the session's bounds (`binding`), SketchBound's δs (`sketch`) and the
    draw's rejection (`rejection`)."""
    return {"profile": record.windows[0].profile.delta, "slack": record.passes * 2.0 ** -audit.SLACK_LOG2,
            "hiding": hiding(record), "binding": binding(record, node), "sketch": sketch(record, node),
            "rejection": rejection(record)}


def memory(record: audit.Record, node: Node) -> tuple[Line, Line, Line]:
    """The memory line's three forms (`FORMS`): each form's charge `line.charges_of`'s and the paced forms
    `line.session_lost`'s, in floats."""
    N, b, P = record.blocks, record.block_bytes, record.passes
    W = audit.pooled(record)[0]
    delta = (record.windows[0].profile.delta + P * 2.0 ** -audit.SLACK_LOG2 + hiding(record) + binding(record, node)
             + sketch(record, node))
    S = -(-node.chip_bytes // b)
    c = line.charges_of(node.rebuild(record.codec).squarings_per_s / record.codec.encode_squarings,
                        node.disk_bytes_per_s, node.link_bytes_per_s, node.rebuild_burst_bytes, node.disk_burst_bytes,
                        node.link_burst_bytes, b)
    planned = sum(sum(w.credited(False)) for w in record.windows)
    delta += rejection(record)
    X, B = c[TIME_HELD].rate, c[TIME_HELD].burst
    snapshot = Line(SNAPSHOT, N, b, delta, N - (planned - W - (X * record.span_ns + B) - P * S) / P)
    if (why := _paced(record)) is not None:
        return snapshot, Line(TIME_HELD, N, b, delta, refused=why), Line(LINK_CHARGED, N, b, delta, refused=why)
    ws = record.windows
    times = [w.released_ns for w in ws], [w.arrived_ns[-1] for w in ws], [w.passes for w in ws]
    return (snapshot, *(Line(f, N, b, delta, line.session_lost(*times, blocks=N, block_bytes=b,
                                                               chip_bytes=node.chip_bytes, wrong=W, charge=c[f]))
                        for f in line.FORMS))


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
