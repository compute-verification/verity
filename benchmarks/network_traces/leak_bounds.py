"""leak-bounds: K, the bits a covert sender in the server can move through the network certifier's accepted transcripts
over a monitoring period, for one link (`per_link`) and summed over a datacenter's links (`datacenter`), each beside its
share of the weights. It is the one calculator of K: `chain_record.py leak` reads it on a warden run, and the slowdown ρ
is derived from both bounds.

A link is one network certifier on one uplink (`Link`, made by `link`), certifying ``rate`` requests a second of one
scan class (`ingress_scan.Class`, or a window class's `scan`). Its sampling blocks are each 2T transfer blocks, T
buckets of τ seconds each way, every one proved as its own proof block and every proof block drawn. K's terms are
those of the generator reading of the scan's statement:

* log2 #Ss = 0;
* the fail cell's position, log2(egress_cells + 1) a request;
* the sampling block's free inputs, which the scan's ``prog`` rows carry (`READINGS`): T·NP a sampling block as
  registered, or an entry's 33 + KO·IDB bits a request (`ingress_scan.entry_layout`) if ``prog`` is an input only at
  its anchored entries. Tied, the rows are a compute audit's outputs, so the T·NP bits are the tied audit's free
  inputs, which the rows form's K charges to the sampling block (`NetworkCheck.Period.tiedAudits`);
* no faulty proof block, every one being drawn;
* Dj = 1, so the jitter term is 0 (assumed, `NetTiming.JitterBounded`).

A link runs s / (T·τ) sampling blocks and certifies rate·s requests over a monitoring period of s seconds (`PERIODS`).
K takes one of two forms:

* δ None, the per-period form: `NetTiming.Guarantees.PeriodLeakAccepted`'s count, which has no δ term with every proof
  block drawn. Each sampling block's audit fails with at most its own error, which the statement leaves to the
  caller's union bound (`NetTiming.periodLeak_le_sum`).
* δ given, the pooled form: `NetTiming.Guarantees.HorizonLeakAccepted`'s count at q = 1. It is the per-period count
  plus F·log2(N + 1) + b·F, where F = ⌈ln(1/δ)⌉ (`fmax`), N is the period's proof blocks and b = NS + NV, and it fails
  with at most δ beside its per-window terms. It is conditional: hesc and hleak await SampledKSLive and P2.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, NamedTuple

from verity.core.protocols.certifier import ingress_scan as IS, warden as W
from verity.core.protocols.pous import wipe_monitoring as WM

#: the monitoring periods' seconds, by name
PERIODS: dict[str, int] = dict(WM.PERIODS)
DELTAS: tuple[Fraction, ...] = WM.DELTAS
#: a bucket's seconds when a link names none: the warden's default τ
TAU = Fraction(W.WardenParams().bucket_ms, 1000)
#: Llama-3.2-1B's parameters (128,256 × 2,048 tied embedding, 16 layers of 60,821,504, the final norm), BF16
WEIGHTS_BITS = 1_235_814_400 * 16
#: each reading of the sampling block's free inputs (the scan's `prog`, the tied audit's free inputs), as a report
#: labels it
READINGS = {"registered": "`prog` as registered today (the record's K)",
            "anchors": "`prog` only at its anchored entries (conditional on the generator-form lowering)"}
STATEMENTS = {"per_period": "NetTiming.Guarantees.PeriodLeakAccepted",
              "pooled": "HorizonLeakAccepted, conditional: hesc and hleak await SampledKSLive and P2",
              "datacenter": "the sum over links of each link's statement, each at δ / links: our composition, which no "
                            "Lean statement covers"}


class Bound(NamedTuple):
    """K in bits and its share of the weights. ``delta`` is what the bound fails with at most beside its statement's
    per-window errors, None in the per-period form, and ``statement`` is what it rests on (`STATEMENTS`)."""
    bits: float
    share: float
    delta: Fraction | float | None
    statement: str


@dataclass(frozen=True)
class Link:
    """One network certifier on one uplink: its scan class, its certified requests a second, its sampling block's T
    buckets each way of ``tau`` seconds, and its egress cells a connection."""
    scan: IS.Class
    rate: float | Fraction
    T: int
    tau: float | Fraction
    egress_cells: int

    @property
    def widths(self) -> dict[str, int]:
        return {"NS": IS.state_bits(self.scan), "NP": IS.prog_bits(self.scan), "NV": IS.view_bits(self.scan)}

    @property
    def b(self) -> int:
        """A proof block's committed bits, a faulty proof block's charge in the pooled form."""
        return IS.state_bits(self.scan) + IS.view_bits(self.scan)

    @property
    def cap(self) -> float:
        """The fail cell's position, a request's schedule term."""
        return math.log2(self.egress_cells + 1)

    @property
    def block_s(self) -> float | Fraction:
        """A sampling block's seconds."""
        return self.T * self.tau

    def free(self, reading: str) -> dict[str, int]:
        """The free inputs' bits a sampling block and a request in ``reading`` (`READINGS`)."""
        if reading not in READINGS:
            raise ValueError(f"a reading is one of {', '.join(READINGS)}, not {reading!r}")
        if reading == "registered":
            return {"block": self.T * IS.prog_bits(self.scan), "request": 0}
        return {"block": 0, "request": sum(w for _, w in IS.entry_layout(self.scan))}

    def bits(self, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None) -> float:
        """K over ``period`` (a `PERIODS` name, or its seconds) in ``reading``: the per-period form, or with ``delta``
        the pooled form."""
        blocks = seconds(period) / self.block_s
        requests = self.rate * seconds(period)
        free = self.free(reading)
        k = requests * self.cap + blocks * free["block"] + requests * free["request"]
        if delta is None:
            return k
        f = fmax(delta)
        return k + f * math.log2(blocks * 2 * self.T + 1) + self.b * f


def seconds(period: str | float | Fraction) -> float | Fraction:
    if isinstance(period, str):
        if period not in PERIODS:
            raise ValueError(f"a monitoring period is one of {', '.join(PERIODS)} or its seconds, not {period!r}")
        return PERIODS[period]
    if not period > 0:
        raise ValueError(f"a monitoring period's seconds are positive, not {period!r}")
    return period


def fmax(delta: float | Fraction) -> int:
    """F = ⌈ln(1/δ)⌉: the faulty proof blocks the pooled form pays for at q = 1."""
    if not 0 < delta < 1:
        raise ValueError(f"δ is in (0, 1), not {delta!r}")
    return math.ceil(math.log(1 / delta))


def link(cls: Any, rate: float | Fraction, *, T: int | None = None, tau: float | Fraction = TAU,
         egress_cells: int | None = None) -> Link:
    """The link of class ``cls`` certifying ``rate`` requests a second. ``cls`` is a scan class (`ingress_scan.Class`,
    which needs ``T``) or a window class (its `scan`, and its ``buckets`` the default T). ``egress_cells`` is by
    default the scan's KO + 1, the most `ingress_scan.Class.serves` takes."""
    if isinstance(cls, IS.Class):
        scan = cls
    elif isinstance(getattr(cls, "scan", None), IS.Class):
        scan, T = cls.scan, cls.buckets if T is None else T
    else:
        raise TypeError(f"a link's class is an ingress_scan.Class or a window class with its scan, not {cls!r}")
    if T is None:
        raise ValueError("a scan class's link needs T, its sampling block's buckets each way (a window class's "
                         "buckets)")
    egress_cells = scan.KO + 1 if egress_cells is None else egress_cells
    if type(T) is not int or T < 1 or not tau > 0 or not rate >= 0:
        raise ValueError(f"a link's T is a positive integer, its τ positive and its rate at least 0, not T = {T!r}, "
                         f"τ = {tau!r} and rate = {rate!r}")
    if type(egress_cells) is not int or not 1 <= egress_cells <= scan.KO + 1:
        raise ValueError(f"egress_cells is an integer from 1 to KO + 1 = {scan.KO + 1}, not {egress_cells!r}")
    return Link(scan, rate, T, tau, egress_cells)


def per_link(cls: Any, rate: float | Fraction, period: str | float | Fraction, reading: str,
             delta: float | Fraction | None = None, *, T: int | None = None, tau: float | Fraction = TAU,
             egress_cells: int | None = None, weights_bits: int = WEIGHTS_BITS) -> Bound:
    """K over a monitoring period for one link (`link`'s arguments), the count `PeriodLeakAccepted` bounds for one
    network certifier on one uplink (δ None), or `HorizonLeakAccepted`'s pooled at ``delta``."""
    bits = link(cls, rate, T=T, tau=tau, egress_cells=egress_cells).bits(period, reading, delta)
    return Bound(bits, bits / weights_bits, delta, STATEMENTS["per_period" if delta is None else "pooled"])


def datacenter(links: Sequence[Link], period: str | float | Fraction, reading: str,
               delta: float | Fraction | None = None, *, weights_bits: int = WEIGHTS_BITS) -> Bound:
    """K over a monitoring period summed over a datacenter's ``links``, each at δ / len(links) (`STATEMENTS`'s
    ``datacenter``).

    This is our composition: no Lean statement covers it. `PeriodLeakAccepted` and `HorizonLeakAccepted` are each
    about one device on one uplink. Summing their bits takes the secrets a sender moves through every link at once to
    number at most the product of the links' counts. That holds except where some link's statement fails, a union
    bound, so a datacenter at δ takes each link at δ / len(links) and pays each link's own faulty proof blocks,
    F = ⌈ln(len(links) / δ)⌉ of them. Two shortcuts would be unsound:

    * Pooling by weight copy, one pooled count over every link that serves a copy of the weights, paying the faulty
      proof blocks once at δ: `HorizonLeakAccepted` counts the faulty proof blocks of one device's record, and links
      are separate devices with separate records, so no statement pools their proof blocks into one period's.
    * Dividing by the copies: every copy is the same secret, so the share is of one copy (``weights_bits``) whatever
      the links.

    In the per-period form there is no δ to split, and the sum is the links' per-period counts. Each link's
    sampling blocks' errors still add up across the datacenter."""
    if not links:
        raise ValueError("a datacenter has at least one link")
    if not all(isinstance(l, Link) for l in links):
        raise TypeError("a datacenter's links are Links (`link`)")
    if delta is not None:
        fmax(delta)
    each = None if delta is None else delta / len(links)
    bits = math.fsum(n * l.bits(period, reading, each) for l, n in Counter(links).items())
    return Bound(bits, bits / weights_bits, delta, STATEMENTS["datacenter"])
