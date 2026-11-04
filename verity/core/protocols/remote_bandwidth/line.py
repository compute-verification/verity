"""The memory line a paced session shows held over time, the line of record a gate run publishes: the conclusions of
`Accounting.Guarantees.Remote.HeldOverTime` (time-held) and `HeldOverTimeLinks` (link-charged),
`Security/Definitions/Accounting/Guarantees/RemoteHeld.lean`, read over each stretch of a session's groups, in exact
rationals.  It is computed here and in its Lean, `lean/RemoteAudit/Line.lean`, held to this by the difftest
(`lean/tests/test_line.py`, every refusal compared word for word): `benchmarks/remote_session/verifier.py`'s
`paced_line` (its diagnostic, on the verifier's clock) and `auditor_line.py` (the line of record, on the challenger's
clock) read it.  The declared outputs' paced forms (`ledger.memory`) are the same arithmetic in floats, over the whole
session as one stretch (`session_lost`); floats are otherwise for display only, outside this module.

A paced session is groups g = 0..C, group g's coin σ_g released at s_g, its n_g passes ending at its last root's
arrival a_g, and the registered chunks of each kind its passes read (`Group`).  e_g = max(a_g, s_(g+1)) (a_C for the
last) and L_g = e_g − s_g hold its reads; Δ_g = s_(g+1) − s_g (L_C for the last).  Stretch m is the `per` groups
[lo, hi) = [m·per, (m + 1)·per), for each m before the last full stretch, so the group after it (hi) is in the
session: it spans Σ from e_0 (m = 0: before group 0's reads a chunk may never have been there) or s_lo to s_hi, its P
passes are its groups', and L = max L_g over the groups lo − 1 to hi.  Each form charges X chunks a ns and B chunks at
once beyond that rate (`Charge`, from the node's rates and bursts: `charges`), so R = X·(Σ + 2L) + B chunks come back
from outside the store while the stretch's reads are served:

- held over time: at most S + (2L·R + fill)/Σ registered chunks are absent at an instant, on average over Σ, S the
  on-chip storage in chunks rounded up (`ChipCeiling` counts whole chunks, and a chunk split between a chip and memory
  is one of them) and `fill` the wrong receipts W spread over the groups lo..hi at v_g = Δ_g/n_g chunk-ns
  each (L/n_hi in hi), highest first, at most n_g·N in group g;
- the snapshot: at most (W + R)/P + S are absent at a pass, on average over the passes.

W is the pooled bound of the session's record (`pooled.bound`, the Lean's `RemoteAudit.Pooled`: its windows' receipts
and draws at δ), so the line reads it from the record's windows, never from a float another module computed.  The
line of record reads its groups' passes, its windows and its registered chunks off the Lean verifier's record of the
session (`line_of_record`, the Lean's `lineOfRecordAt` at the record's `groups_per_coin` g, `lineOfRecord` at g = 1),
and each group's times and reads beside it (`Seen`), refused unless the record is of the registration given with it
(`unregistered`, the Lean's `lineOfRegistered`).

Each kind is credited its mean chunks a pass less those, in bytes, and at least 0: weights (in HBM), HBM, and filler
(HBM less the weights, and the host tier).  A reading's line is its worst stretch by weights credited, the first on a
tie, and its shares are of the node's bytes.  Both forms need every pass to read every registered chunk (`Covers`): a
chunk a pass leaves out can be absent across it with neither a wrong read nor a rebuild.

The JSON form (`inputs_json`, `run_json`; `record_inputs_json`, `record_run_json` for the record's) is the Lean's:
integers, each rational a string `n` or `n/d`, and the windows a list of [receipts, draws].  A float given here is
the rational its shortest decimal writes, which is what its JSON text says (`number`), except δ, which is the float's
exact value (`pooled.number`): the value the session sampled at."""
from __future__ import annotations

import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from . import audit, codec, pooled
from . import receipts as _receipts
from .audit import RECEIPT_LIMIT

TIME_HELD, LINK_CHARGED = "time-held", "link-charged"
FORMS = (TIME_HELD, LINK_CHARGED)
"""The forms, `HeldOverTime`'s and `HeldOverTimeLinks`'s: the declared outputs' after the snapshot (`ledger.FORMS`)."""
TIME, SNAPSHOT = "time", "snapshot"
READINGS = (TIME, SNAPSHOT)
"""Each form's two readings of a stretch: held over time, and at the passes."""
NS = 10 ** 9
RATES = ("squarings_per_s", "disk_bytes_per_s", "link_bytes_per_s", "rebuild_burst_bytes", "disk_burst_bytes",
         "link_burst_bytes")
_RATIONAL = re.compile(r"-?[0-9]+(/[0-9]+)?")


class Refused(ValueError):
    """The inputs have no line, and why: the Lean's reason, word for word."""


class Malformed(ValueError):
    """The JSON form isn't the line's inputs."""


def number(x: object) -> Fraction:
    """`x` exactly: an integer, a `Fraction`, a float as the rational its shortest decimal writes (its JSON text), or
    a string `n` or `n/d` with d positive."""
    if isinstance(x, Fraction):
        return x
    if type(x) is int:
        return Fraction(x)
    if type(x) is float:
        if x != x or x in (float("inf"), float("-inf")):
            raise Malformed(f"not a finite number: {x!r}")
        return Fraction(repr(x))
    if isinstance(x, str) and _RATIONAL.fullmatch(x):
        num, _, den = x.partition("/")
        if den and int(den) == 0:
            raise Malformed(f"a rational's denominator is positive: {x!r}")
        return Fraction(int(num), int(den or 1))
    raise Malformed(f"a rational is an integer or a string n or n/d: {x!r}")


def text(q: Fraction) -> str:
    """`q` as the JSON form writes it, `n` or `n/d` in lowest terms (Lean's `toString` of a `Rat`)."""
    return str(q)


GROUP = ("released_ns", "last_ns", "passes", "least", "weights", "hbm", "host")


@dataclass(frozen=True, slots=True)
class Group:
    """One group: its coin's release s_g and its last root's arrival a_g (ns, on one clock), its passes n_g, the
    registered chunks its least pass read (`least`: the least any pass read in HBM plus the least in the host tier,
    the verifier's `held`), and the registered chunks its passes read in all, by kind: weights (in HBM), all of HBM,
    and the host tier."""

    released_ns: int
    last_ns: int
    passes: int
    least: int
    weights: int
    hbm: int
    host: int

    def __post_init__(self) -> None:
        for name in GROUP:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise Malformed(f"a group's {name} is a nonnegative integer")


@dataclass(frozen=True, slots=True)
class Charge:
    """What brings a registered chunk back into node memory other than a read of it there: at most `rate` chunks a ns
    and `burst` chunks at once beyond it (the statements' X and B)."""

    rate: Fraction
    burst: Fraction


@dataclass(frozen=True, slots=True)
class Rates:
    """The node's ways back into its memory (`ledger.Node`'s): its squaring ceiling over a chunk's encode in squarings
    (rebuilds), its disks' and its boundary links' read ceilings (bytes a second), and each one's burst (bytes)."""

    squarings_per_s: Fraction
    encode_squarings: int
    disk_bytes_per_s: Fraction = Fraction(0)
    link_bytes_per_s: Fraction = Fraction(0)
    rebuild_burst_bytes: Fraction = Fraction(0)
    disk_burst_bytes: Fraction = Fraction(0)
    link_burst_bytes: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        for name in RATES:
            object.__setattr__(self, name, number(getattr(self, name)))
        if type(self.encode_squarings) is not int or self.encode_squarings < 0:
            raise Malformed("a block's encode in squarings is a nonnegative integer")

    def to_json(self) -> dict:
        return {"encode_squarings": self.encode_squarings, **{name: text(getattr(self, name)) for name in RATES}}


def charges(rates: Rates, block_bytes: int) -> dict[str, Charge]:
    """Each form's charge in chunks at `rates` (`charges_of`)."""
    return charges_of(rates.squarings_per_s / rates.encode_squarings, rates.disk_bytes_per_s, rates.link_bytes_per_s,
                      rates.rebuild_burst_bytes, rates.disk_burst_bytes, rates.link_burst_bytes, block_bytes)


def charges_of(rebuilds_per_s, disk_bytes_per_s, link_bytes_per_s, rebuild_burst_bytes, disk_burst_bytes,
               link_burst_bytes, block_bytes: int) -> dict[str, Charge]:
    """Each form's charge in chunks: time-held X = s/E + β and B = B_r + B_d (rebuilds and the node's disks,
    `HeldOverTime`), link-charged X + β_l and B + B_l (its boundary links too, `HeldOverTimeLinks`), from the chunks
    the node rebuilds a second s/E and its ceilings and bursts in bytes; exact from `Rates`, and in floats, each step
    rounded as written, for the declared outputs (`ledger.memory`)."""
    b = block_bytes
    x = rebuilds_per_s / NS + disk_bytes_per_s / b / NS
    burst = rebuild_burst_bytes / b + disk_burst_bytes / b
    return {TIME_HELD: Charge(x, burst),
            LINK_CHARGED: Charge(x + link_bytes_per_s / b / NS, burst + link_burst_bytes / b)}


@dataclass(frozen=True, slots=True)
class Reading:
    """A stretch read one way: the registered chunks it charges as absent (`lost_blocks`) and each kind's bytes
    credited, at least 0."""

    lost_blocks: Fraction
    weights: Fraction
    hbm: Fraction
    filler: Fraction

    def to_json(self) -> dict:
        return {"filler": text(self.filler), "hbm": text(self.hbm), "lost_blocks": text(self.lost_blocks),
                "weights": text(self.weights)}


@dataclass(frozen=True, slots=True)
class Stretch:
    """Stretch m, groups [lo, hi): its span Σ, passes P, L (`l_max_ns`) and its groups' L summed, the mean chunks of
    each kind a pass read, the chunks charged back R (`rebuild_blocks`), the wrong receipts' absent chunk-ns (`fill`),
    and its two readings."""

    stretch: int
    lo: int
    hi: int
    span_ns: int
    passes: int
    l_max_ns: int
    l_sum_ns: int
    held: tuple[Fraction, Fraction, Fraction]
    rebuild_blocks: Fraction
    fill: Fraction
    time: Reading
    snapshot: Reading

    def reading(self, name: str) -> Reading:
        return self.time if name == TIME else self.snapshot

    def to_json(self) -> dict:
        w, h, host = self.held
        return {"fill": text(self.fill), "held": {"hbm": text(h), "host": text(host), "weights": text(w)},
                "hi": self.hi, "l_max_ns": self.l_max_ns, "l_sum_ns": self.l_sum_ns, "lo": self.lo,
                "passes": self.passes, "rebuild_blocks": text(self.rebuild_blocks), SNAPSHOT: self.snapshot.to_json(),
                "span_ns": self.span_ns, "stretch": self.stretch, TIME: self.time.to_json()}


def _check(groups: Sequence[Group], per: int, blocks: int, block_bytes: int, node_bytes: int, wrong: Fraction,
           rates: Rates | None) -> None:
    if block_bytes < 1:
        raise Refused("a block is one or more bytes")
    if blocks < 1:
        raise Refused("the store is one or more blocks")
    if node_bytes < 1:
        raise Refused("the node is one or more bytes")
    if rates is not None:
        if rates.encode_squarings < 1:
            raise Refused("a block's encode is one or more squarings")
        for name in RATES:
            if getattr(rates, name) < 0:
                raise Refused(f"{name} is nonnegative")
    if wrong < 0:
        raise Refused("the wrong receipts are nonnegative")
    if per < 1:
        raise Refused("a stretch is one or more groups")
    for g, x in enumerate(groups):
        if x.passes < 1:
            raise Refused(f"group {g} has no pass")
        if x.last_ns < x.released_ns:
            raise Refused(f"group {g}'s last root came before its coin")
        if g and x.released_ns < groups[g - 1].released_ns:
            raise Refused(f"group {g}'s coin left before group {g - 1}'s")
        if x.least != blocks:
            raise Refused(f"group {g}'s least pass read {x.least} of the {blocks} registered blocks: the paced line "
                          "needs every pass to read every registered block, since a block a pass leaves out can be "
                          "absent across it with neither a wrong read nor a rebuild")
        if x.hbm + x.host != x.passes * blocks:
            raise Refused(f"group {g}'s {x.passes} passes read {x.hbm + x.host} registered blocks in all, not "
                          f"{x.passes * blocks}")
        if x.weights > x.hbm:
            raise Refused(f"group {g}'s passes read more weights than HBM, which holds them")
    if len(groups) // per < 2:
        raise Refused(f"the session's {len(groups)} groups at {per} a stretch leave no stretch before the closing one")


def stretches(groups: Sequence[Group], *, per: int, blocks: int, block_bytes: int, chip_bytes: int,
              wrong: object, charge: Charge, node_bytes: int = 1) -> tuple[Stretch, ...]:
    """Every stretch of `per` groups but the last full one, at `charge`; refused (`Refused`) as the Lean refuses."""
    W = number(wrong)
    _check(groups, per, blocks, block_bytes, node_bytes, W, None)
    if charge.rate < 0 or charge.burst < 0:
        raise Refused("a charge is nonnegative")
    return _stretches(groups, per, blocks, block_bytes, chip_bytes, W, charge)


def _times(s: Sequence[int], last_ns: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """Each group's e_g, L_g and Δ_g, its coin released at s_g and its last root arriving at `last_ns[g]`."""
    G = len(s)
    e = [max(a, s[g + 1]) if g + 1 < G else a for g, a in enumerate(last_ns)]
    L = [eg - sg for eg, sg in zip(e, s)]
    return e, L, [s[g + 1] - s[g] if g + 1 < G else L[g] for g in range(G)]


def _held(s: Sequence[int], n: Sequence[int], e: Sequence[int], L: Sequence[int], gap: Sequence[int], m: int, per: int,
          N: int, S: int, W, charge: Charge, ratio=Fraction) -> tuple:
    """Stretch m of `per` groups held over time: its span Σ, L, R, fill and S + (2L·R + fill)/Σ, each v_g
    `ratio`(Δ_g, n_g): exact at `Fraction`, and in floats at `operator.truediv` with W and `charge` floats."""
    lo, hi = m * per, (m + 1) * per
    span = s[hi] - (e[0] if m == 0 else s[lo])
    if span <= 0:
        raise Refused(f"stretch {m}'s span is empty: group {hi}'s coin left no later than "
                      + ("group 0's end" if m == 0 else f"group {lo}'s coin"))
    l_max = max(L[max(0, lo - 1):hi + 1])
    R = charge.rate * (span + 2 * l_max) + charge.burst
    v = {g: ratio(l_max if g == hi else gap[g], n[g]) for g in range(lo, hi + 1)}
    left, fill = W, ratio(0, 1)
    for g in sorted(range(lo, hi + 1), key=lambda g: v[g], reverse=True):
        if left <= 0:
            break
        took = min(left, n[g] * N)
        fill, left = fill + took * v[g], left - took
    return span, l_max, R, fill, S + (2 * l_max * R + fill) / span


def session_lost(released_ns: Sequence[int], last_ns: Sequence[int], passes: Sequence[int], *, blocks: int,
                 block_bytes: int, chip_bytes: int, wrong: float, charge: Charge) -> float:
    """The chunks a paced session's groups g = 0..C charge as absent held over time with the whole session one
    stretch, groups [0, C) against the closing group C (stretch 0 at `per` = C, which `stretches` leaves out as the
    closing stretch), in floats: the declared outputs' paced forms (`ledger.memory`, whose record has a nonempty
    credited span, `_paced`)."""
    e, L, gap = _times(released_ns, last_ns)
    return _held(released_ns, passes, e, L, gap, 0, len(released_ns) - 1, blocks, -(-chip_bytes // block_bytes),
                 wrong, charge, operator.truediv)[-1]


def _stretches(groups: Sequence[Group], per: int, N: int, b: int, chip_bytes: int, W: Fraction,
               charge: Charge) -> tuple[Stretch, ...]:
    G = len(groups)
    s, n = [x.released_ns for x in groups], [x.passes for x in groups]
    e, L, gap = _times(s, [x.last_ns for x in groups])
    S = -(-chip_bytes // b)
    out = []
    for m in range(G // per - 1):
        lo, hi = m * per, (m + 1) * per
        span, l_max, R, fill, lost = _held(s, n, e, L, gap, m, per, N, S, W, charge)
        P = sum(n[lo:hi])
        held = tuple(Fraction(sum(k(x) for x in groups[lo:hi]), P)
                     for k in (lambda x: x.weights, lambda x: x.hbm, lambda x: x.host))

        def reading(lost: Fraction) -> Reading:
            w, h, host = held
            return Reading(lost, max(Fraction(0), w - lost) * b, max(Fraction(0), h - lost) * b,
                           max(Fraction(0), h - w + host - lost) * b)

        out.append(Stretch(m, lo, hi, span, P, l_max, sum(L[lo:hi]), held, R, fill, reading(lost),
                           reading((W + R) / P + S)))
    return tuple(out)


def worst(xs: Sequence[Stretch], reading: str) -> Stretch:
    """The stretch whose `reading` credits the fewest weights bytes, the first of them on a tie."""
    out = xs[0]
    for x in xs[1:]:
        if x.reading(reading).weights < out.reading(reading).weights:
            out = x
    return out


@dataclass(frozen=True, slots=True)
class Form:
    """One form's charge and stretches."""

    name: str
    charge: Charge
    stretches: tuple[Stretch, ...]

    def worst(self, reading: str) -> Stretch:
        return worst(self.stretches, reading)

    def to_json(self, node_bytes: int) -> dict:
        out = {"burst": text(self.charge.burst), "rate": text(self.charge.rate),
               "stretches": [x.to_json() for x in self.stretches]}
        for name in READINGS:
            x = self.worst(name)
            r = x.reading(name)
            out[name] = {"filler_share": text(r.filler / node_bytes), "hbm_share": text(r.hbm / node_bytes),
                         "stretch": x.stretch, "weights_share": text(r.weights / node_bytes)}
        return out


@dataclass(frozen=True, slots=True)
class Line:
    """A session's line in both forms (`FORMS`), against `node_bytes` of node memory, with at most `wrong` of its
    receipts wrong (`pooled.bound`'s W)."""

    node_bytes: int
    forms: tuple[Form, Form]
    wrong: Fraction

    def form(self, name: str) -> Form:
        return next(f for f in self.forms if f.name == name)

    def share(self, form: str, reading: str = TIME, kind: str = "weights") -> Fraction:
        """The `kind`'s share of the node that `form`'s worst stretch credits in `reading`."""
        return getattr(self.form(form).worst(reading).reading(reading), kind) / self.node_bytes

    def to_json(self) -> dict:
        return {**{f.name: f.to_json(self.node_bytes) for f in self.forms}, "wrong": text(self.wrong)}


def line(groups: Sequence[Group], *, per: int, blocks: int, block_bytes: int, chip_bytes: int, node_bytes: int,
         windows: Sequence[tuple[int, int]], delta: object, rates: Rates) -> Line:
    """The session's line in both forms on a node of `rates`, `node_bytes` of memory and `chip_bytes` of on-chip
    storage, with at most W of its receipts wrong: the pooled bound of its record's `windows` (each window's receipts
    and draws) at `delta` (`pooled.bound`, a float δ its exact value); refused (`Refused`) as the Lean refuses."""
    try:
        W = pooled.bound(windows, delta).wrong
    except pooled.Refused as e:
        raise Refused(str(e)) from None
    _check(groups, per, blocks, block_bytes, node_bytes, W, rates)
    c = charges(rates, block_bytes)
    return Line(node_bytes, tuple(Form(f, c[f], _stretches(groups, per, blocks, block_bytes, chip_bytes, W, c[f]))
                                  for f in FORMS), W)


# --- the JSON form, the Lean's (`RemoteAudit.Difftest`'s `line`) ------------------------------------------------------

def inputs_json(groups: Sequence[Group], *, per: int, blocks: int, block_bytes: int, chip_bytes: int, node_bytes: int,
                windows: Sequence[tuple[int, int]], delta: object, rates: Rates) -> dict:
    """`line`'s inputs in the JSON form."""
    return {"block_bytes": block_bytes, "blocks": blocks, "chip_bytes": chip_bytes, "node_bytes": node_bytes,
            "per": per, "rates": rates.to_json(), "windows": [[u, s] for u, s in windows],
            "delta": text(pooled.number(delta)), "groups": [{k: getattr(x, k) for k in GROUP} for x in groups]}


def _field(j: Mapping, k: str) -> Any:
    if not isinstance(j, Mapping) or k not in j:
        raise Malformed(f"missing {k}")
    return j[k]


def _nat(j: Mapping, k: str) -> int:
    v = _field(j, k)
    if type(v) is not int or v < 0:
        raise Malformed(f"{k}: not a nonnegative integer")
    return v


def _rational(j: Mapping, k: str) -> Fraction:
    v = _field(j, k)
    if type(v) is not int and not isinstance(v, str):
        raise Malformed(f"{k}: not an integer or a string")
    return number(v)


def of_json(j: Mapping) -> dict:
    """The JSON form's inputs as `line`'s keyword arguments, `groups` among them (`Malformed` if they aren't)."""
    gs = _field(j, "groups")
    if not isinstance(gs, list):
        raise Malformed("groups: not a list")
    r = _field(j, "rates")
    try:
        windows, delta = pooled.of_json({k: _field(j, k) for k in ("windows", "delta")})
    except pooled.Malformed as e:
        raise Malformed(str(e)) from None
    return {"groups": [Group(*(_nat(x, k) for k in GROUP)) for x in gs],
            "per": _nat(j, "per"), "blocks": _nat(j, "blocks"), "block_bytes": _nat(j, "block_bytes"),
            "chip_bytes": _nat(j, "chip_bytes"), "node_bytes": _nat(j, "node_bytes"), "windows": windows,
            "delta": delta,
            "rates": Rates(*(_rational(r, k) for k in RATES[:1]), _nat(r, "encode_squarings"),
                           *(_rational(r, k) for k in RATES[1:]))}


def run_json(j: Mapping) -> dict:
    """The line of the JSON form's inputs in the JSON form, or `{"refused": why}`."""
    args = of_json(j)
    try:
        return line(**args).to_json()
    except Refused as e:
        return {"refused": str(e)}


# --- the line read off the Lean verifier's record (`RemoteAudit.Line.lineOfRecordAt`, the Difftest's `line_record`) ---

SEEN = ("released_ns", "last_ns", "least", "weights", "hbm", "host")


@dataclass(frozen=True, slots=True)
class Seen:
    """What the auditor saw of one window beside the record of it: its coin's release and its last root's arrival (ns,
    on one clock), the registered chunks its least pass read, and the registered chunks its passes read in all, by
    kind (`Group`'s, but its passes, which are the record's)."""

    released_ns: int
    last_ns: int
    least: int
    weights: int
    hbm: int
    host: int

    def __post_init__(self) -> None:
        for name in SEEN:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise Malformed(f"a window's seen {name} is a nonnegative integer")


def receipts(window: Mapping) -> int:
    """U, a window's receipts in the record: its passes' lengths summed."""
    return sum(window["lengths"])


SHARED = ("a record drawn more than one window a coin (groups_per_coin) at one window in flight has no line: no "
          "property of the code covers its draws there")
"""Why a record drawn more than one window a coin (`audit.Params.groups_per_coin`) at one window in flight
(`audit.Pace.in_flight` 1) has no line: no property of the code covers its samples there (`LiveIsRun` is at two)."""


def draws_of(g: int, ws: Sequence[Mapping]) -> list[tuple[int, int]]:
    """The samples of a record drawn `g` windows a coin, as the pooled bound reads them (`audit.pooled_windows`): draw c is
    windows cg … min((c + 1)g, W) − 1, its receipts U and its samples theirs summed."""
    return [(sum(receipts(w) for w in ws[a:a + g]), sum(w["drawn"] for w in ws[a:a + g])) for a in range(0, len(ws), g)]


def of_summary(record: Mapping, samples: int, seen: Sequence[Seen],
               g: int = 1) -> tuple[list[Group], list[tuple[int, int]]]:
    """The groups and windows the line reads of an accepted session's record (the Lean verifier's, as its `outcome`
    request writes it) and what the auditor saw of each of its windows: group t is window t, its passes the record's
    and its times and reads `seen[t]`'s, and window t's receipts and draws are the record's (Σ lengths_t, drawn_t).
    Refused unless the auditor saw each of the record's windows and each window drew min(s, U_t) of its receipts at
    the session's `samples` s, as the verifier's sample does. Drawn `g` > 1 windows a coin (`groups_per_coin`), the
    windows are the record's samples (`draws_of`), each refused unless it sampled min(s, U) of its U receipts and has fewer
    than `audit.RECEIPT_LIMIT` (2^64), as `receive` holds each window to: the memory model's sample reads at most
    2^64 − 1 receipts (`Properties.RemoteAudit.Line.receiptMax`), and a sample's windows' receipts need not be."""
    ws = record["windows"]
    if len(seen) != len(ws):
        raise Refused(f"the auditor saw {len(seen)} windows, not the record's {len(ws)}")
    if g <= 1:
        for t, w in enumerate(ws):
            if w["drawn"] != min(samples, receipts(w)):
                raise Refused(f"window {t} drew {w['drawn']} of its {receipts(w)} receipts, not "
                              f"min({samples}, {receipts(w)})")
        windows = [(receipts(w), w["drawn"]) for w in ws]
    else:
        windows = draws_of(g, ws)
        for c, (u, d) in enumerate(windows):
            if d != min(samples, u):
                raise Refused(f"draw {c} drew {d} of its {u} receipts, not min({samples}, {u})")
        for c, (u, _) in enumerate(windows):
            if u >= RECEIPT_LIMIT:
                raise Refused(f"draw {c} has {u} receipts, not fewer than 2^64")
    return [Group(x.released_ns, x.last_ns, len(w["lengths"]), x.least, x.weights, x.hbm, x.host)
            for w, x in zip(ws, seen)], windows


CLOCKS = ("record", "challenger", "replay")
"""Whose clock a line's times seen are on (`RemoteAudit.Line.lineOfRecordOn`): the record's own, which holds them
(`times_off`), the challenger's release log's, or a replay's at another round trip (`gate_whatif`), which it
doesn't."""


def last_ns(window: Mapping) -> int:
    """A window's last root's arrival in the record: its last pass's, or its coin's release with none."""
    return window["arrived_ns"][-1] if window["arrived_ns"] else window["released_ns"]


def times_off(record: Mapping, seen: Sequence[Seen]) -> str | None:
    """Why the times `seen` aren't the record's own, or None: the first window whose coin's release or last root's
    arrival seen isn't the record's (`released_ns`, `last_ns`)."""
    for t, (w, x) in enumerate(zip(record["windows"], seen)):
        if (x.released_ns, x.last_ns) != (w["released_ns"], last_ns(w)):
            return (f"window {t} was seen released at {x.released_ns} and last at {x.last_ns}, not at the record's "
                    f"{w['released_ns']} and {last_ns(w)}")
    return None


def unregistered(registration: audit.Registration, record: Mapping) -> str | None:
    """Why `record` isn't of `registration`, or None: it names another registration's digest, or other registered
    chunks than the registration's (the Lean's `Line.unregistered`)."""
    if record["registration"] != registration.digest.hex():
        return "the record is not of the registration given"
    if record["blocks"] != registration.blocks:
        return f"the record's {record['blocks']} registered blocks are not the registration's {registration.blocks}"
    return None


def line_of_record(record: Mapping, *, registration: audit.Registration, samples: int, seen: Sequence[Seen],
                   per: int, block_bytes: int, chip_bytes: int, node_bytes: int, delta: object, rates: Rates,
                   clock: str = "record") -> Line:
    """The line of an accepted session's record: `line` at its groups and windows (`of_summary`, at the record's
    `groups_per_coin` g, 1 when it doesn't name it: above 1 the windows are its samples) and its registered chunks (the
    record's `blocks`, the registration the session ran on). Refused for a record drawn more than one window a coin at
    one window in flight (`SHARED`), then for a record not of `registration` (`unregistered`, the Lean's
    `lineOfRegistered`), whose chunks the line would credit. On the record's `clock` the times seen are held to the
    record's, refused after `of_summary`'s refusals where they differ (`times_off`); on another (`CLOCKS`) the record
    doesn't hold them."""
    g = record.get("groups_per_coin", 1)
    if g != 1 and (record.get("pace") or {}).get("in_flight", 2) == 1:
        raise Refused(SHARED)
    if (why := unregistered(registration, record)) is not None:
        raise Refused(why)
    groups, windows = of_summary(record, samples, seen, g)
    if clock == "record" and (why := times_off(record, seen)) is not None:
        raise Refused(why)
    return line(groups, per=per, blocks=record["blocks"], block_bytes=block_bytes, chip_bytes=chip_bytes,
                node_bytes=node_bytes, windows=windows, delta=delta, rates=rates)


def record_inputs_json(record: Mapping, *, registration: Mapping, samples: int, seen: Sequence[Seen], per: int,
                       block_bytes: int, chip_bytes: int, node_bytes: int, delta: object, rates: Rates,
                       clock: str = "record", session: Mapping | None = None) -> dict:
    """`line_of_record`'s inputs in the JSON form, the record as the Lean wrote it and `registration` as the Lean
    reads one (`period.registration_json`'s form, `registration_of`); with `session`, the session's transcript, from
    which the Lean verifier's `line_record` makes the record itself (`Shared.run g` at its `groups_per_coin` g,
    `Session.run` at 1) and refuses a `record` that isn't the one it makes (line.py reads `record`)."""
    if clock not in CLOCKS:
        raise ValueError(f"clock: one of {', '.join(CLOCKS)}, not {clock!r}")
    return {"record": record, "registration": registration, "samples": samples,
            "seen": [{k: getattr(x, k) for k in SEEN} for x in seen],
            "per": per, "block_bytes": block_bytes, "chip_bytes": chip_bytes, "node_bytes": node_bytes,
            "delta": text(pooled.number(delta)), "rates": rates.to_json(), "clock": clock,
            **({} if session is None else {"session": session})}


def _nats(j: Mapping, k: str) -> list[int]:
    v = _field(j, k)
    if not isinstance(v, list) or any(type(x) is not int or x < 0 for x in v):
        raise Malformed(f"{k}: not a list of nonnegative integers")
    return v


def _hex(v: object, k: str) -> None:
    if v is not None and not (isinstance(v, str) and re.fullmatch("(?:[0-9a-f]{2})*", v)):
        raise Malformed(f"{k}: not null or lowercase hex")


def _bytes(j: Mapping, k: str) -> bytes:
    v = _field(j, k)
    if v is None:
        raise Malformed(f"{k}: not lowercase hex")
    _hex(v, k)
    return bytes.fromhex(v)


def _str(j: Mapping, k: str) -> str:
    if not isinstance(v := _field(j, k), str):
        raise Malformed(f"{k}: not a string")
    return v


def _bool(j: Mapping, k: str) -> bool:
    if type(v := _field(j, k)) is not bool:
        raise Malformed(f"{k}: not true or false")
    return v


def registration_of(j: object) -> audit.Registration:
    """The registration a record is given with, as the Lean reads one (`Difftest.registrationOf`,
    `Registration.make?`): `audit.register`'s checks on a registration whose seed is given, `bind` refused
    (`Malformed` if it isn't one)."""
    c = _field(j, "codec")
    kind, field, rounds, masked = _str(j, "receipt"), _str(c, "field"), _nat(c, "rounds"), _bool(c, "masked")
    node, blocks, root, seed = _bytes(j, "node"), _nat(j, "blocks"), _bytes(j, "payload_root"), _bytes(j, "codec_seed")
    bind, layout, rows = _bool(j, "bind"), _str(j, "layout"), _nat(j, "rows")
    try:
        if field not in codec.FIELDS:
            raise ValueError(f"a codec's field is one of {sorted(codec.FIELDS)}")
        cd = codec.Codec(codec.FIELDS[field], rounds, masked)
        _receipts.receipt_kind(kind)
        if _receipts.store_blocks(blocks) > cd.max_blocks:
            raise ValueError(f"a store under {cd.name} is at most {cd.max_blocks} blocks")
        if len(root) != _receipts.DIGEST_BYTES:
            raise ValueError(f"the payload root is {_receipts.DIGEST_BYTES} bytes")
        if len(seed) != codec.SEED_BYTES:
            raise ValueError(f"a codec seed is {codec.SEED_BYTES} bytes")
        if bind:
            raise ValueError("a registration that binds PoUW's tile outputs is not covered")
        if layout != _receipts.LAYOUT:
            raise ValueError(f"the block layout is {_receipts.LAYOUT}")
        _receipts.sketch_rows(kind, rows)
        return audit.Registration(audit._node(node), blocks, root, seed, cd, kind, False, layout, rows)
    except ValueError as e:
        raise Malformed(f"registration: {e}") from None


def _record(j: object) -> Mapping:
    """The record as the Lean reads it back (`Difftest.summaryOf`): every field it holds, of its type."""
    if not isinstance(j, Mapping):
        raise Malformed("record: not an object")
    ws = _field(j, "windows")
    if not isinstance(ws, list):
        raise Malformed("windows: not a list")
    for w in ws:
        for k in ("lengths", "steps", "post_seed", "stale", "arrived_ns"):
            _nats(w, k)
        heads = _field(w, "heads")
        if not isinstance(heads, list):
            raise Malformed("heads: not a list")
        for h in heads:
            _hex(h, "heads")
        for k in ("released_ns", "deadline_ns", "opened_ns", "drawn"):
            _nat(w, k)
    for k in ("blocks", "appended", "released", "lifetime"):
        _nat(j, k)
    pace = _field(j, "pace")
    if pace is not None:
        _nat(pace, "k")
        _nat(pace, "window_ns")
        if "in_flight" in pace:
            _nat(pace, "in_flight")
    _hex(_field(j, "appends"), "appends")
    _bytes(j, "registration")
    if "groups_per_coin" in j:
        _nat(j, "groups_per_coin")
    return j


def of_record_json(j: Mapping) -> dict:
    """The record form's inputs as `line_of_record`'s arguments (`Malformed` if they aren't)."""
    seen, r = _field(j, "seen"), _field(j, "rates")
    if not isinstance(seen, list):
        raise Malformed("seen: not a list")
    clock = j.get("clock", "record")
    if clock not in CLOCKS:
        raise Malformed(f"clock: record, challenger or replay, not {clock}")
    try:
        delta = pooled.of_json({"windows": [], "delta": _field(j, "delta")})[1]
    except pooled.Malformed as e:
        raise Malformed(str(e)) from None
    return {"record": _record(_field(j, "record")), "registration": registration_of(_field(j, "registration")),
            "samples": _nat(j, "samples"), "seen": [Seen(*(_nat(x, k) for k in SEEN)) for x in seen],
            "per": _nat(j, "per"),
            "block_bytes": _nat(j, "block_bytes"), "chip_bytes": _nat(j, "chip_bytes"),
            "node_bytes": _nat(j, "node_bytes"), "delta": delta,
            "rates": Rates(*(_rational(r, k) for k in RATES[:1]), _nat(r, "encode_squarings"),
                           *(_rational(r, k) for k in RATES[1:])), "clock": clock}


def record_run_json(j: Mapping) -> dict:
    """The line of the record form's inputs in the JSON form, or `{"refused": why}`."""
    args = of_record_json(j)
    try:
        return line_of_record(**args).to_json()
    except Refused as e:
        return {"refused": str(e)}
