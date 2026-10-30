"""leak-bounds: K, the bits a covert sender in the server can move through the network certifier's accepted transcripts
over a monitoring period, for one link (`per_link`), summed over a datacenter's links (`datacenter`; the census's
reference datacenter has `datacenter_links`) and over the world's (`world`, the adversary resource model's), each beside
its share of one copy of the weights (`WEIGHTS`) and ε, the union of its scans' δ. It is the one calculator of K:
`chain_record.py leak` reads it on a warden run, and the slowdown ρ is derived from both bounds.

A link is one network certifier on one uplink (its load a `LinkLoad`, made by `link`), certifying ``rate`` requests a
second of one scan class (`ingress_scan.Class`, or a window class's `scan`). Each of its link-windows is 2T transfer
blocks, T buckets of τ seconds each way, each proved as its own proof block, every one sampled, its scan at δ_scan. The
scan's ``prog`` is the outputs of a compute audit, the tied audit (`NetworkCheck.Period` refuses an untied ``prog`` by
name, `wired`). K's terms are those of `NetworkCheck.Period`'s rows form (`rowsTerms`, `Scan.terms`) in the generator
reading of the scan's statement, in three parts (`Terms`):

* timing: the jitter term at D_j = `JITTER_BOUND`, ⌈log2 D_j^T⌉ = T bits a link-window, one a bucket (`JITTER`). The
  schedules and the fail cells are no term: the scan fixes each lane's length and renders no fail cell, and the record
  shows the schedule;
* the side channels: the tied audit's unregistered free inputs, which the rows form charges to the link-window
  (`tiedAudits`);
* the checking term, what the sampled proofs leave unchecked (`LinkLoad.checking`): 0 with every proof block sampled,
  E_w a link-window when the tied audit is sampled in part (`faultyEntries`), and in the pooled form its F incorrect
  proof blocks.

Which audit the scan is tied to is the reading (`READINGS`). The record's scan is tied to scan_zk's `SlotEntry_v1`
stand-in, whose free inputs are the slot entries themselves, NR·T / EVERY of NE bits a link-window
(`ingress_scan.entry_bits`): the whole response content, which the prover sets and nothing checks. Its side channels
are no bound, so its content is reported as unchecked (`UNCHECKED`) and added to no K. A served audit's side channels
are its own free inputs (``audit_free``, `chain_record.free_inputs`), and its checking term is E_w when it is sampled
in part.

A link runs m = s / (T·τ) link-windows and certifies rate·s requests over a monitoring period of s seconds
(`PERIODS`). ε = m·δ_scan is the rows form's `eps`: the union of the scans' δ, what K fails with at most beside the
scans' knowledge and collision errors; a tied audit sampled in part adds its δ (`chargedOf`). K takes one of two forms:

* δ None, the per-period form: `NetTiming.Guarantees.PeriodLeakAccepted`'s count. Each link-window's audit fails with
  at most its own error, which the statement leaves to the caller's union bound (`NetTiming.periodLeak_le_sum`).
* δ given, the pooled form: `NetTiming.Guarantees.PooledLeakAccepted`'s count at q = 1. Its checking term adds
  F·log2(N + 1) + b·F, where F = ⌈ln(1/δ)⌉ (`fmax`), N is the period's proof blocks and b = NS + NV, even with every
  proof block sampled, and it fails with at most δ beside ε. It is conditional: hesc and hleak await SampledKSLive
  and P2.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

from verity.core.protocols.certifier import ingress_scan as IS, warden as W
from verity.core.protocols.pous import wipe_monitoring as WM

#: the monitoring periods' seconds, by name
PERIODS: dict[str, int] = dict(WM.PERIODS)
DELTAS: tuple[Fraction, ...] = WM.DELTAS
#: a bucket's seconds when a link names none: the warden's default τ
TAU = Fraction(W.WardenParams().bucket_ms, 1000)
#: a link-window's scan's δ when a link names none: `scan_zk --delta`'s default
SCAN_DELTA = Fraction(1, 2 ** 20)
#: the jitter bound D_j that K is charged at (`NetTiming.JitterBounded`, one bound over every step): a statement of k
#: steps, one a bucket, pays ⌈log2 D_j^k⌉ = k bits. Each bucket leaves at a time fixed before it is made
#: (`warden.Warden(paced=…)`, `active.Proxy(paced=…)`), yet the host's state after the server's writes for the next
#: bucket still moves a bucket's arrival offset, so J isn't met at D_j = 1
JITTER_BOUND = 2
JITTER = "J assumed at D_j = 2; at D_j = 1 not met (next-bucket host-state channel)"
CENSUS = Path(__file__).resolve().parents[1] / "census"


class Weights(NamedTuple):
    """One copy of the weights a share is of: its bits, and what they are."""
    bits: int
    label: str


#: Llama-3.2-1B's parameters (128,256 × 2,048 tied embedding, 16 layers of 60,821,504, the final norm) in BF16. The
#: served checkpoint is int8, so a share is of this BF16 copy, as its label says.
WEIGHTS = Weights(1_235_814_400 * 16, "Llama-3.2-1B in BF16")
#: each reading of the audit a link-window's scan is tied to, as a report labels it
READINGS = {"stand-in": "tied to scan_zk's `SlotEntry_v1` stand-in (the record's): its free inputs are the slot "
                        "entries, the whole response content, unchecked",
            "served": "tied to a served compute audit: its unregistered free inputs, and E_w a link-window when it is "
                      "sampled in part"}
#: how a report labels the stand-in tie's content, which no bound adds
UNCHECKED = "unchecked (stand-in tie)"
STATEMENTS = {"per_period": "NetTiming.Guarantees.PeriodLeakAccepted",
              "pooled": "PooledLeakAccepted, conditional: hesc and hleak await SampledKSLive and P2",
              "datacenter": "the sum over links of each link's statement, each at δ / links: our composition, which no "
                            "Lean statement covers"}


class Terms(NamedTuple):
    """K's terms over a span, in bits: ``timing``, the jitter term; the side channels (``side``, None where none is
    given: the stand-in tie, or a served audit whose free inputs aren't); and the checking term (``checking``).
    ``unchecked`` is the stand-in tie's content, which no bound adds."""
    timing: float
    side: float | None
    checking: float
    unchecked: float = 0

    @property
    def bits(self) -> float | None:
        """K, timing, the side channels and the checking term summed: None where the side channels aren't given."""
        return None if self.side is None else self.timing + self.side + self.checking


class Bound(NamedTuple):
    """K in bits and its share of ``weights`` (`Weights.label`), each None where the side channels aren't given, and
    its ``terms``. ``delta`` is what the bound fails with at most beside ``eps``, None in the per-period form; ``eps``
    is the union of its scans' δ (the rows form's `eps`, m·δ_scan summed over the links); and ``statement`` is what it
    rests on (`STATEMENTS`)."""
    bits: float | None
    share: float | None
    delta: Fraction | float | None
    eps: Fraction | float
    statement: str
    weights: str
    terms: Terms


@dataclass(frozen=True)
class LinkLoad:
    """What one network certifier on one uplink certifies: its scan class, its certified requests a second, its
    link-window's T buckets each way of ``tau`` seconds and its scan's δ; and in the served reading its tied audit's
    unregistered free inputs' bits a link-window (``audit_free``, None when not given) and whether that audit is
    sampled whole (``audit_whole``)."""
    scan: IS.Class
    rate: float | Fraction
    T: int
    tau: float | Fraction
    scan_delta: float | Fraction = SCAN_DELTA
    audit_free: int | None = None
    audit_whole: bool = True

    @property
    def widths(self) -> dict[str, int]:
        return {"NS": IS.state_bits(self.scan), "NE": IS.entry_bits(self.scan), "NV": IS.view_bits(self.scan)}

    @property
    def entries(self) -> int:
        """A link-window's slot entries, NR at each of its T / EVERY boundaries (`scan_zk.entries_of`)."""
        return self.scan.NR * (self.T // self.scan.EVERY)

    @property
    def e_w(self) -> int:
        """E_w, a link-window's slot entries' bits (`Scan.entries`): the stand-in tie's content, and what a tied audit
        sampled in part can set."""
        return self.entries * IS.entry_bits(self.scan)

    @property
    def b(self) -> int:
        """A proof block's committed bits, an incorrect proof block's charge in the pooled form."""
        return IS.state_bits(self.scan) + IS.view_bits(self.scan)

    @property
    def jitter(self) -> int:
        """A link-window's jitter term, over its scan's T steps (`jitter_bits`)."""
        return jitter_bits(self.T)

    @property
    def block_s(self) -> float | Fraction:
        """A link-window's seconds."""
        return self.T * self.tau

    def blocks(self, period: str | float | Fraction) -> float | Fraction:
        """m, the link-windows over ``period``: exact unless ``period`` or τ is a float."""
        s = seconds(period)
        exact = not (isinstance(s, float) or isinstance(self.block_s, float))
        return Fraction(s) / self.block_s if exact else s / self.block_s

    def eps(self, period: str | float | Fraction) -> float | Fraction:
        """ε = m·δ_scan over ``period``."""
        return self.blocks(period) * self.scan_delta

    def checking(self, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None) -> float:
        """The checking term over ``period`` in ``reading``, what the sampled proofs leave unchecked: 0 with every proof
        block sampled; E_w a link-window (`e_w`, `faultyEntries`) when the served tied audit is sampled in part, which
        overcounts the rows form's, charged only at a bound of at least one incorrect proof block; and with ``delta``
        the pooled form's F incorrect proof blocks, F·(b + log2(N + 1)). Every checking charge is made here, so the
        terms and the reports read it from one place."""
        _reading(reading)
        blocks = self.blocks(period)
        bits = 0.0 if reading == "stand-in" or self.audit_whole else float(blocks * self.e_w)
        if delta is None:
            return bits
        f = fmax(delta)
        return bits + f * math.log2(blocks * 2 * self.T + 1) + self.b * f

    def terms(self, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None) -> Terms:
        """K's terms over ``period`` (a `PERIODS` name, or its seconds) in ``reading`` (`READINGS`): the per-period
        form, or with ``delta`` the pooled form."""
        blocks = self.blocks(period)
        side = None if _reading(reading) == "stand-in" or self.audit_free is None else float(blocks * self.audit_free)
        return Terms(float(blocks * self.jitter), side, self.checking(period, reading, delta),
                     float(blocks * self.e_w) if reading == "stand-in" else 0.0)

    def bits(self, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None) -> float | None:
        """K over ``period`` in ``reading`` (`terms`), None where the side channels aren't given."""
        return self.terms(period, reading, delta).bits


def _reading(reading: str) -> str:
    if reading not in READINGS:
        raise ValueError(f"a reading is one of {', '.join(READINGS)}, not {reading!r}")
    return reading


def seconds(period: str | float | Fraction) -> float | Fraction:
    if isinstance(period, str):
        if period not in PERIODS:
            raise ValueError(f"a monitoring period is one of {', '.join(PERIODS)} or its seconds, not {period!r}")
        return PERIODS[period]
    if not period > 0:
        raise ValueError(f"a monitoring period's seconds are positive, not {period!r}")
    return period


def jitter_bits(steps: int) -> int:
    """K's jitter term over `steps` steps at `JITTER_BOUND`, ⌈log2 D_j^steps⌉ (`NetworkCheck.Period`'s)."""
    return (JITTER_BOUND ** steps - 1).bit_length()


def fmax(delta: float | Fraction) -> int:
    """F = ⌈ln(1/δ)⌉: the incorrect proof blocks the pooled form pays for at q = 1."""
    if not 0 < delta < 1:
        raise ValueError(f"δ is in (0, 1), not {delta!r}")
    return math.ceil(math.log(1 / delta))


def _reference(name: str) -> dict:
    return next(d for d in json.loads((CENSUS / name).read_text()) if d["role"] == "reference")


def datacenter_links() -> int:
    """The census's reference datacenter's links (`deployments.json`): one network certifier on each machine's
    uplink."""
    return _reference("deployments.json")["machines"]


def world_links() -> int:
    """The world's links: the adversary resource model's compute (`adversary.json`, its total FLOP over its monitoring
    period at an H100-equivalent's FLOP/s) in machines of the reference datacenter's GPUs, rounded up, one network
    certifier on each machine's uplink."""
    world, dc = _reference("adversary.json"), _reference("deployments.json")
    return math.ceil(world["total_flop"] / (world["flop_per_s_per_h100"] * world["monitoring_period_s"]
                                            * dc["gpus_per_machine"]))


def link(cls: Any, rate: float | Fraction, *, T: int | None = None, tau: float | Fraction = TAU,
         scan_delta: float | Fraction = SCAN_DELTA, audit_free: int | None = None, audit_whole: bool = True) -> LinkLoad:
    """The link of class ``cls`` certifying ``rate`` requests a second. ``cls`` is a scan class (`ingress_scan.Class`,
    which needs ``T``) or a window class (its `scan`, and its ``buckets`` the default T). ``audit_free`` and
    ``audit_whole`` are the served tied audit's (`LinkLoad`)."""
    if isinstance(cls, IS.Class):
        scan = cls
    elif isinstance(getattr(cls, "scan", None), IS.Class):
        scan, T = cls.scan, cls.buckets if T is None else T
    else:
        raise TypeError(f"a link's class is an ingress_scan.Class or a window class with its scan, not {cls!r}")
    if T is None:
        raise ValueError("a scan class's link needs T, its link-window's buckets each way (a window class's buckets)")
    if type(T) is not int or T < 1 or T % scan.EVERY or not tau > 0 or not rate >= 0:
        raise ValueError(f"a link's T is a positive multiple of EVERY = {scan.EVERY}, its τ positive and its rate at "
                         f"least 0, not T = {T!r}, τ = {tau!r} and rate = {rate!r}")
    if not 0 < scan_delta < 1:
        raise ValueError(f"the scan's δ is in (0, 1), not {scan_delta!r}")
    if audit_free is not None and (type(audit_free) is not int or audit_free < 0):
        raise ValueError(f"the tied audit's free inputs are a count of bits, at least 0, or None, not {audit_free!r}")
    if type(audit_whole) is not bool:
        raise TypeError(f"whether the tied audit is sampled whole is a bool, not {audit_whole!r}")
    return LinkLoad(scan, rate, T, tau, scan_delta, audit_free, audit_whole)


def _bound(terms: Terms, delta: float | Fraction | None, eps: float | Fraction, statement: str,
           weights: Weights) -> Bound:
    bits = terms.bits
    return Bound(bits, None if bits is None else bits / weights.bits, delta, eps, statement, weights.label, terms)


def per_link(cls: Any, rate: float | Fraction, period: str | float | Fraction, reading: str,
             delta: float | Fraction | None = None, *, T: int | None = None, tau: float | Fraction = TAU,
             scan_delta: float | Fraction = SCAN_DELTA, audit_free: int | None = None, audit_whole: bool = True,
             weights: Weights = WEIGHTS) -> Bound:
    """K over a monitoring period for one link (`link`'s arguments), the count `PeriodLeakAccepted` bounds for one
    network certifier on one uplink (δ None), or `PooledLeakAccepted`'s pooled at ``delta``."""
    one = link(cls, rate, T=T, tau=tau, scan_delta=scan_delta, audit_free=audit_free, audit_whole=audit_whole)
    statement = STATEMENTS["per_period" if delta is None else "pooled"]
    return _bound(one.terms(period, reading, delta), delta, one.eps(period), statement, weights)


def datacenter(links: Sequence[LinkLoad] | Mapping[LinkLoad, int], period: str | float | Fraction, reading: str,
               delta: float | Fraction | None = None, *, weights: Weights = WEIGHTS) -> Bound:
    """K over a monitoring period summed over a datacenter's ``links`` (each a `LinkLoad`, or each with its count),
    each at δ / links (`STATEMENTS`'s ``datacenter``), term by term.

    This is our composition: no Lean statement covers it. `PeriodLeakAccepted` and `PooledLeakAccepted` are each
    about one device on one uplink. Summing their bits takes the secrets a sender moves through every link at once to
    number at most the product of the links' counts. That holds except where some link's statement fails, a union
    bound, so a datacenter at δ takes each link at δ / links and pays each link's own incorrect proof blocks,
    F = ⌈ln(links / δ)⌉ of them, and its ε is the links' summed. Two shortcuts would be unsound:

    * Pooling by weight copy, one pooled count over every link that serves a copy of the weights, paying the incorrect
      proof blocks once at δ: `PooledLeakAccepted` counts the incorrect proof blocks of one device's record, and links
      are separate devices with separate records, so no statement pools their proof blocks into one period's.
    * Dividing by the copies: every copy is the same secret, so the share is of one copy (``weights``) whatever the
      links.

    In the per-period form there is no δ to split, and the sum is the links' per-period counts. Each link's
    link-windows' errors still add up across the datacenter, in ε."""
    pairs = list(links.items()) if isinstance(links, Mapping) else [(l, 1) for l in links]
    if not pairs:
        raise ValueError("a datacenter has at least one link")
    if not all(isinstance(l, LinkLoad) for l, _ in pairs):
        raise TypeError("a datacenter's links are LinkLoads (`link`)")
    if not all(type(n) is int and n >= 1 for _, n in pairs):
        raise ValueError("a datacenter's link counts are positive integers")
    counts: Counter[LinkLoad] = Counter()
    for l, n in pairs:
        counts[l] += n
    if delta is not None:
        fmax(delta)
    each = None if delta is None else delta / counts.total()
    parts = [(n, l.terms(period, reading, each)) for l, n in counts.items()]
    sides = [t.side for _, t in parts]
    terms = Terms(*(None if field == "side" and None in sides else math.fsum(n * getattr(t, field) for n, t in parts)
                    for field in Terms._fields))
    eps = sum(n * l.eps(period) for l, n in counts.items())
    return _bound(terms, delta, eps, STATEMENTS["datacenter"], weights)


def world(one: LinkLoad, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None, *,
          weights: Weights = WEIGHTS) -> Bound:
    """K over a monitoring period summed over the world's links (`world_links`), each ``one``: `datacenter` at the
    adversary resource model's size."""
    return datacenter({one: world_links()}, period, reading, delta, weights=weights)
