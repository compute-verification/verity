"""leak-bounds: K, the bits a covert sender in the server can move through the network certifier's accepted transcripts
over a monitoring period, for one link (`per_link`), summed over a datacenter's links (`datacenter`; the census's
reference datacenter has `datacenter_links`) and over the world's (`world`, the adversary resource model's), each beside
its share of one copy of the weights (`WEIGHTS`) and ε, the union of its scans' δ. It is the one calculator of K:
`chain_record.py leak` reads it on a warden run, and the slowdown ρ is derived from both bounds.

A link is one network certifier on one uplink (its load a `LinkLoad`, made by `link`), certifying ``rate`` requests a
second of one scan class (`ingress_scan.Class`, or a window class's `scan`). Its sampling blocks are each 2T transfer
blocks, T buckets of τ seconds each way, every one proved as its own proof block, every proof block drawn, each sampling
block's scan at δ_scan. K's terms are those of `NetworkCheck.Period`'s rows form (`rowsTerms`) in the generator reading
of the scan's statement:

* the schedules and the fail cells, ⌈log2(cap + 1)⌉ each a request (`capsTerm`, charged twice as `#Ss ≤ ∏_c (cap_c +
  1)`), a connection's cap its lane's egress_cells;
* the sampling block's free inputs (`READINGS`). As registered, the scan's ``prog`` is its tied compute audit's
  outputs, NR·T / EVERY slot entries of NE bits (`ingress_scan.entry_bits`), so they are that audit's free inputs,
  which the rows form charges to the sampling block (`tiedAudits`). If ``prog`` is an input only at its anchored
  entries, they are an entry's 33 + KO·IDB bits a request (`ingress_scan.entry_layout`);
* no faulty proof block, every one being drawn;
* Dj = 1, so the jitter term is 0 (assumed, `NetTiming.JitterBounded`).

A link runs m = s / (T·τ) sampling blocks and certifies rate·s requests over a monitoring period of s seconds
(`PERIODS`). ε = m·δ_scan is the rows form's `eps`: the union of the scans' δ, what K fails with at most beside the
scans' knowledge and collision errors. K takes one of two forms:

* δ None, the per-period form: `NetTiming.Guarantees.PeriodLeakAccepted`'s count, which has no δ term with every proof
  block drawn. Each sampling block's audit fails with at most its own error, which the statement leaves to the
  caller's union bound (`NetTiming.periodLeak_le_sum`).
* δ given, the pooled form: `NetTiming.Guarantees.HorizonLeakAccepted`'s count at q = 1. It is the per-period count
  plus F·log2(N + 1) + b·F, where F = ⌈ln(1/δ)⌉ (`fmax`), N is the period's proof blocks and b = NS + NV, and it fails
  with at most δ beside ε. It is conditional: hesc and hleak await SampledKSLive and P2.
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
#: a sampling block's scan's δ when a link names none: `scan_zk --delta`'s default
SCAN_DELTA = Fraction(1, 2 ** 20)
CENSUS = Path(__file__).resolve().parents[1] / "census"


class Weights(NamedTuple):
    """One copy of the weights a share is of: its bits, and what they are."""
    bits: int
    label: str


#: Llama-3.2-1B's parameters (128,256 × 2,048 tied embedding, 16 layers of 60,821,504, the final norm) in BF16. The
#: served checkpoint is int8, so a share is of this BF16 copy, as its label says.
WEIGHTS = Weights(1_235_814_400 * 16, "Llama-3.2-1B in BF16")
#: each reading of the sampling block's free inputs (the scan's `prog`, the tied audit's free inputs), as a report
#: labels it
READINGS = {"registered": "`prog` as registered, the tied audit's slot entries (the record's K)",
            "anchors": "`prog` only at its anchored entries (conditional on the generator-form lowering)"}
STATEMENTS = {"per_period": "NetTiming.Guarantees.PeriodLeakAccepted",
              "pooled": "HorizonLeakAccepted, conditional: hesc and hleak await SampledKSLive and P2",
              "datacenter": "the sum over links of each link's statement, each at δ / links: our composition, which no "
                            "Lean statement covers"}


class Bound(NamedTuple):
    """K in bits and its share of ``weights`` (`Weights.label`). ``delta`` is what the bound fails with at most beside
    ``eps``, None in the per-period form; ``eps`` is the union of its scans' δ (the rows form's `eps`, m·δ_scan summed
    over the links); and ``statement`` is what it rests on (`STATEMENTS`)."""
    bits: float
    share: float
    delta: Fraction | float | None
    eps: Fraction | float
    statement: str
    weights: str


@dataclass(frozen=True)
class LinkLoad:
    """What one network certifier on one uplink certifies: its scan class, its certified requests a second, its
    sampling block's T buckets each way of ``tau`` seconds, its egress cells a connection and its scan's δ."""
    scan: IS.Class
    rate: float | Fraction
    T: int
    tau: float | Fraction
    egress_cells: int
    scan_delta: float | Fraction = SCAN_DELTA

    @property
    def widths(self) -> dict[str, int]:
        return {"NS": IS.state_bits(self.scan), "NE": IS.entry_bits(self.scan), "NV": IS.view_bits(self.scan)}

    @property
    def entries(self) -> int:
        """A sampling block's slot entries, NR at each of its T / EVERY boundaries (`scan_zk.entries_of`)."""
        return self.scan.NR * (self.T // self.scan.EVERY)

    @property
    def b(self) -> int:
        """A proof block's committed bits, a faulty proof block's charge in the pooled form."""
        return IS.state_bits(self.scan) + IS.view_bits(self.scan)

    @property
    def cell(self) -> int:
        """⌈log2(egress_cells + 1)⌉, a connection's cap's term (`clog2 (cap + 1)`)."""
        return self.egress_cells.bit_length()

    @property
    def cap(self) -> int:
        """A request's schedules and fail-cell terms, one `cell` each (`rowsTerms`'s `schedules` and `cells`)."""
        return 2 * self.cell

    @property
    def block_s(self) -> float | Fraction:
        """A sampling block's seconds."""
        return self.T * self.tau

    def free(self, reading: str) -> dict[str, int]:
        """The free inputs' bits a sampling block and a request in ``reading`` (`READINGS`)."""
        if reading not in READINGS:
            raise ValueError(f"a reading is one of {', '.join(READINGS)}, not {reading!r}")
        if reading == "registered":
            return {"block": self.entries * IS.entry_bits(self.scan), "request": 0}
        return {"block": 0, "request": sum(w for _, w in IS.entry_layout(self.scan))}

    def blocks(self, period: str | float | Fraction) -> float | Fraction:
        """m, the sampling blocks over ``period``: exact unless ``period`` or τ is a float."""
        s = seconds(period)
        exact = not (isinstance(s, float) or isinstance(self.block_s, float))
        return Fraction(s) / self.block_s if exact else s / self.block_s

    def eps(self, period: str | float | Fraction) -> float | Fraction:
        """ε = m·δ_scan over ``period``."""
        return self.blocks(period) * self.scan_delta

    def bits(self, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None) -> float:
        """K over ``period`` (a `PERIODS` name, or its seconds) in ``reading``: the per-period form, or with ``delta``
        the pooled form."""
        blocks = self.blocks(period)
        requests = self.rate * seconds(period)
        free = self.free(reading)
        k = float(requests * self.cap + blocks * free["block"] + requests * free["request"])
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
         egress_cells: int | None = None, scan_delta: float | Fraction = SCAN_DELTA) -> LinkLoad:
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
    if type(T) is not int or T < 1 or T % scan.EVERY or not tau > 0 or not rate >= 0:
        raise ValueError(f"a link's T is a positive multiple of EVERY = {scan.EVERY}, its τ positive and its rate at "
                         f"least 0, not T = {T!r}, τ = {tau!r} and rate = {rate!r}")
    if type(egress_cells) is not int or not 1 <= egress_cells <= scan.KO + 1:
        raise ValueError(f"egress_cells is an integer from 1 to KO + 1 = {scan.KO + 1}, not {egress_cells!r}")
    if not 0 < scan_delta < 1:
        raise ValueError(f"the scan's δ is in (0, 1), not {scan_delta!r}")
    return LinkLoad(scan, rate, T, tau, egress_cells, scan_delta)


def per_link(cls: Any, rate: float | Fraction, period: str | float | Fraction, reading: str,
             delta: float | Fraction | None = None, *, T: int | None = None, tau: float | Fraction = TAU,
             egress_cells: int | None = None, scan_delta: float | Fraction = SCAN_DELTA,
             weights: Weights = WEIGHTS) -> Bound:
    """K over a monitoring period for one link (`link`'s arguments), the count `PeriodLeakAccepted` bounds for one
    network certifier on one uplink (δ None), or `HorizonLeakAccepted`'s pooled at ``delta``."""
    one = link(cls, rate, T=T, tau=tau, egress_cells=egress_cells, scan_delta=scan_delta)
    bits = one.bits(period, reading, delta)
    statement = STATEMENTS["per_period" if delta is None else "pooled"]
    return Bound(bits, bits / weights.bits, delta, one.eps(period), statement, weights.label)


def datacenter(links: Sequence[LinkLoad] | Mapping[LinkLoad, int], period: str | float | Fraction, reading: str,
               delta: float | Fraction | None = None, *, weights: Weights = WEIGHTS) -> Bound:
    """K over a monitoring period summed over a datacenter's ``links`` (each a `LinkLoad`, or each with its count),
    each at δ / links (`STATEMENTS`'s ``datacenter``).

    This is our composition: no Lean statement covers it. `PeriodLeakAccepted` and `HorizonLeakAccepted` are each
    about one device on one uplink. Summing their bits takes the secrets a sender moves through every link at once to
    number at most the product of the links' counts. That holds except where some link's statement fails, a union
    bound, so a datacenter at δ takes each link at δ / links and pays each link's own faulty proof blocks,
    F = ⌈ln(links / δ)⌉ of them, and its ε is the links' summed. Two shortcuts would be unsound:

    * Pooling by weight copy, one pooled count over every link that serves a copy of the weights, paying the faulty
      proof blocks once at δ: `HorizonLeakAccepted` counts the faulty proof blocks of one device's record, and links
      are separate devices with separate records, so no statement pools their proof blocks into one period's.
    * Dividing by the copies: every copy is the same secret, so the share is of one copy (``weights``) whatever the
      links.

    In the per-period form there is no δ to split, and the sum is the links' per-period counts. Each link's
    sampling blocks' errors still add up across the datacenter, in ε."""
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
    bits = math.fsum(n * l.bits(period, reading, each) for l, n in counts.items())
    eps = sum(n * l.eps(period) for l, n in counts.items())
    return Bound(bits, bits / weights.bits, delta, eps, STATEMENTS["datacenter"], weights.label)


def world(one: LinkLoad, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None, *,
          weights: Weights = WEIGHTS) -> Bound:
    """K over a monitoring period summed over the world's links (`world_links`), each ``one``: `datacenter` at the
    adversary resource model's size. Each link certifies ``one``'s rate, and requests past it are uncertified, so K
    doesn't count them."""
    return datacenter({one: world_links()}, period, reading, delta, weights=weights)
