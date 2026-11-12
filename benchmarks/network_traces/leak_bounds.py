"""leak-bounds: K, the bits a covert sender in the server can move through the network certifier's accepted transcripts
over a monitoring period, beside its share of one copy of the served model's weights (`WEIGHTS`) and ε, the union of
its scans' δ. It is the one calculator of K, and it scales K in two ways:

* by requests (`table`, which `verity bench certifier_term` prints): the network certifier's own terms at the load
  of a node, the census's reference datacenter and the world, all a scope serves counted by each check's own rule
  (At a scope's load, below);
* by links (`per_link`, `datacenter`, `at_link_rate`): one link at the rate a run certified, and as many of them
  summed, whatever the scope they stand for serves. `chain_record.py leak` reads it on a network certifier run, and
  the slowdown ρ is derived from both bounds.

A link is one network certifier on one uplink (its load a `LinkLoad`, made by `link`), certifying ``rate`` requests a
second of one scan class (`ingress_scan.Class`, or a window class's `scan`). Its sampling blocks are each 2T transfer
blocks, T buckets of τ seconds each way, every one proved as its own proof block, every proof block sampled, each sampling
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
  block sampled. Each sampling block's audit fails with at most its own error, which the statement leaves to the
  caller's union bound (`NetTiming.periodLeak_le_sum`).
* δ given, the pooled form: `NetTiming.Guarantees.PooledLeakAccepted`'s count at q = 1. It is the per-period count
  plus F·log2(N + 1) + b·F, where F = ⌈ln(1/δ)⌉ (`fmax`), N is the period's proof blocks and b = NS + NV, and it fails
  with at most δ beside ε. It is conditional: hesc and hleak await SampledKSLive and P2.

**At a scope's load.** A scope (`scopes`) is a node, one machine of the world fleet's GPUs; the census's reference
datacenter; or the world, rho's world fleet, which is the census's inference at the fleet's serving MFU. Each machine
is behind one network certifier. Over a monitoring period a scope serves GPU-seconds × MFU × a GPU's peak FLOP/s /
(a request's tokens × 2 × the served MACs a token) requests of the served model (`MODEL`), the rule
`benchmarks/spot_figures/figure1.py`'s `scopes` plans with, each on one connection. Which request that is is one of
`COUNTS`, and every row names its one: by default the sampling plan's (`plan`), so that the rows count the requests
the compute audit's plan prices; the census's where a caller asks; or none, where a caller gives the world's requests
a year. No run has certified a class for that model, so the class is assumed (`Assumed`): the most ids a response
holds, an id's bits, and an execution phase's buckets. `table` counts each of the eight `ROWS` by its check's own
rule, and never by requests what a check counts by capacity:

* the scan, `rowsTerms`'s terms (`terms`). The network certifier's own are the schedules and the fail cells,
  ⌈log2(egress_cells + 1)⌉ each a connection of the period's parameters, counted by request, and the jitter,
  ⌈log2 D_j^T⌉ an execution phase of each link, which is a bit a bucket at D_j = 2 and 0 at D_j = 1. The term of
  record is the one at D_j = 2 (`OF_RECORD`), two release schedules, and it is printed first. The term at D_j = 1 is
  what a network certifier with a set-time card, a network card that releases at a set time, would be charged. The
  card is simulated, an input of its own (`inputs`' ``set_time_card``, marked ``simulated``, its source the clause
  it rests on, `SET_TIME_CLAUSE`): none is deployed, so a figure at D_j = 1 is printed only under its label
  (`SET_TIME`). The entries, the tied compute audits' free inputs, are the compute side's. The check charges every
  slot's entry at its whole width (`ingress_scan.entry_bits`), lanes × T / EVERY of them a link's execution phase, a
  count by capacity that needs the class's lanes; with none given the entries are one a request, the least the check
  can charge;
* N2, the clock form's terms (`clock_terms`): the schedules are ⌈log2 #Σ_sync^m⌉ over a link's m execution phases,
  counted by link whatever it serves; its parameters list no connection, so it has no fail cell; and the jitter is
  the scan's, a step a bucket;
* the scan under each of `CHANGES`, the three changes to its check, and under all three: a response's length held to
  its Program's output count, and that count to its ids by a stop block, a proof block that checks where a response's
  ids stop, sampled on a served audit, so that the schedules are 0 and the connections that fail are charged pooled
  over the period (`failed`); the release times taken once, by the compute side's tick, so that the jitter is 0 at
  either D_j, which rests on the simulated set-time card, so a row under this change carries the label at either
  D_j, in the document and in its printed name; and the entries charged through the compute audit, whose own bound
  then stands in their place and is not counted here;
* the design, which no run has behind it: head checks, a lengths unit on the network certifier's records, and a stop
  block sampled on a served audit. The lengths unit holds a response's length to its Program's output count and the
  stop block holds that count to the ids, so the lengths are held; the entries are the compute audit's; and its term
  is the jitter, the pooled failures and the count (below);
* the design with no stop block, the same before a stop block has been sampled on a served audit: nothing holds the
  output count to the ids, so the server picks it and each connection is still charged its schedule, as the scan
  charges it. Its failed connections, jitter, count and entries are the design's.

**The count.** Under the design the network certifier signs, for each execution phase of a link, a head over its
tables, and a content class's tables are the rows it holds and no other. So the head states how many of the execution
phase's releases each class holds: one of R + 1 values a class, R being the most connections the schedule releases in
an execution phase (`phase_releases`). A link's execution phase is charged classes · log2(R + 1) bits for it
(`counted`), by link whatever it serves, on the rows whose check reads that head (`Kind.count`): the design's two.
The schedule's classes, delay and ingress buckets are inputs, by default those of the lengths rows' cost runner
(`COST_RUNNER`). Its lanes are the input the entries read: with none given the count is at the least lanes whose
schedule releases what a link serves in an execution phase, the least the rule can charge at this load.

`inputs` marks each input measured, modeled, assumed or simulated (`MARKS`) with its source, and `main` prints the
table and its JSON.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

from verity.protocols.certifier.prover import ingress_scan as IS
from verity.protocols.certifier.verifier import calibration
from verity.protocols.certifier.verifier import warden as W
from verity.protocols import reporting

#: the monitoring periods' seconds, by name
PERIODS: dict[str, int] = dict(reporting.PERIODS)
DELTAS: tuple[Fraction, ...] = reporting.DELTAS
#: a bucket's seconds when a link names none: the network certifier's default τ
TAU = Fraction(W.WardenParams().bucket_ms, 1000)
#: a sampling block's scan's δ when a link names none: `scan_zk --delta`'s default
SCAN_DELTA = Fraction(1, 2 ** 20)
ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "benchmarks" / "census"
#: the files the scale is read from: the census's world, reference datacenter and GPU peak, rho's world fleet and its
#: machine's GPUs, and the served model's workload card
ADVERSARY, DEPLOYMENTS, HARDWARE = (f"benchmarks/census/{name}.json"
                                    for name in ("adversary", "deployments", "hardware"))
RHO, RHO_HARDWARE = "verity/protocols/inference_only/rho/data/reference.toml", "verity/protocols/inference_only/rho/data/hardware.toml"
WORKLOAD = "benchmarks/spot_figures/cards/workload.glm-5.3.json"
#: the break-even planner the sampling plan is computed with, and the case of it a request is read on: the proof
#: server's line at its last step, at the 128k serving plan's request
PLANNER, PLAN_CASE = "benchmarks/spot_figures/breakeven", "P5+poolwin600"
#: the served model, rho's world fleet and the census's GPU peak, the H200's dense FP8 rate
MODEL, WORLD, PEAK = "GLM-5.3", "world-h200", "h100-sxm5-80gb/e4m3"
#: a GB's bits, and FP4's bits a parameter
GB, FP4 = 8 * 10 ** 9, 4


class Weights(NamedTuple):
    """One copy of the weights a share is of: its bits, and what they are."""
    bits: int
    label: str


def _json(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def _toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text())


#: one copy of the served model's weights in FP4: the workload card's parameters, 4 bits each
WEIGHTS = Weights(_json(WORKLOAD)["value"]["params"] * FP4, f"{MODEL} in FP4")
#: each reading of the sampling block's free inputs (the scan's `prog`, the tied audit's free inputs), as a report
#: labels it
READINGS = {"registered": "`prog` as registered, the tied audit's slot entries (the record's K)",
            "anchors": "`prog` only at its anchored entries (conditional on the generator-form lowering)"}
STATEMENTS = {"per_period": "NetTiming.Guarantees.PeriodLeakAccepted",
              "pooled": "PooledLeakAccepted, conditional: hesc and hleak await SampledKSLive and P2",
              "datacenter": "the sum over links of each link's statement, each at δ / links: our composition, which no "
                            "Lean statement covers"}
#: the three changes to the scan's check, by name, and what each zeroes of what the check charges today
CHANGES = {"lengths": "a response's length held to its Program's output count, and that count held to its ids by a "
                      "stop block, a proof block that checks where a response's ids stop, sampled on a served "
                      "audit: the schedules are 0, and the connections that fail are charged pooled over the "
                      "period, which of them failed and where each one's stream ended",
           "timing": "the release times taken once, by the compute side's tick: the jitter is 0, which rests on a "
                     "set-time card, a network card of the network certifier that releases at a set time; the card "
                     "is simulated, and none is deployed",
           "entries": "the entries charged through the compute audit: the compute side's own bound stands in their "
                      "place, and is not counted here"}


class Terms(NamedTuple):
    """A check's terms over a period, in bits: the network certifier's own (the schedules, the fail cells, the jitter
    and the count) and the entries, the tied compute audits' free inputs, which are the compute side's and None where
    its own bound stands in their place. The count is what the heads of the links' execution phases state of each
    content class's held rows (`counted`), 0 on a row whose check reads no such head."""
    schedules: Any
    cells: Any
    jitter: Any
    entries: Any
    count: Any = 0

    @property
    def certifier(self) -> Any:
        """The network certifier's own term."""
        return self.schedules + self.cells + self.jitter + self.count


def failed(connections: Any, failures: Any, cells: int) -> float:
    """What a period's failed connections are charged where the lengths are held, in bits, pooled over the period:
    log2 of the ways at most ``failures`` of its ``connections`` connections fail, each at one of the ``cells`` + 1
    places its stream can end, Σ_{j ≤ F} C(N, j)·(c + 1)^j, taken at its bounds ((c + 1)·e·N / F)^F and (c + 2)^N,
    whichever is less. It is 0 with no failure."""
    if not failures:
        return 0.0
    n, f = float(connections), float(failures)
    return min(f * (math.log2(math.e * n / f) + math.log2(cells + 1)), n * math.log2(cells + 2))


def phase_releases(lanes: int, buckets: int, delay: int, ingress_buckets: int) -> int:
    """R, the most connections a schedule releases in one execution phase: ``lanes`` · ⌈(``buckets`` +
    ``ingress_buckets`` − 1) / (``delay`` + 1)⌉. A release is at most ``ingress_buckets`` − 1 buckets after its
    connection's admission, so an execution phase's releases are of connections admitted in ``buckets`` +
    ``ingress_buckets`` − 1 buckets in a row; and a connection holds its lane through its admission's bucket +
    ``delay``, so those admitted in any ``delay`` + 1 buckets in a row hold a lane each. It is a bound: at some
    schedules no load releases as many. A schedule that signs its releases' tables has at most ``delay`` ingress
    buckets."""
    if (not all(type(x) is int for x in (lanes, buckets, delay, ingress_buckets)) or lanes < 0 or buckets < 1
            or not 1 <= ingress_buckets <= delay):
        raise ValueError(f"a schedule's lanes are an integer from 0, its execution phase's buckets a positive integer "
                         f"and its ingress buckets an integer from 1 to its delay, not {lanes!r} lanes, {buckets!r} "
                         f"buckets, {ingress_buckets!r} ingress buckets and a delay of {delay!r}")
    return lanes * -(-(buckets + ingress_buckets - 1) // (delay + 1))


def _jitter(dj: int, steps: int) -> int:
    """⌈log2 dj^steps⌉, a scan's or an audit's jitter term."""
    if type(dj) is not int or dj < 1 or type(steps) is not int or steps < 1:
        raise ValueError(f"the jitter bound and an execution phase's buckets are positive integers, not D_j = {dj!r} "
                         f"and {steps!r}")
    return (dj ** steps - 1).bit_length()


def terms(connections: Any, scans: Any, *, cells: int, steps: int, dj: int = 1, entries: Any = 0,
          changes: Iterable[str] = (), failures: Any = 0, stop: bool = True) -> Terms:
    """`rowsTerms`'s terms over ``connections`` connections of at most ``cells`` egress cells, each charged
    ⌈log2(cells + 1)⌉ bits for its schedule and again for its fail cell, and ``scans`` scans of ``steps`` buckets
    each at the jitter bound ``dj`` (⌈log2 dj^steps⌉ a scan), with ``entries`` the tied audits' free inputs. Under
    ``changes`` (`CHANGES`): with the lengths held the schedules are 0 and the ``failures`` failed connections are
    charged pooled (`failed`), with the timing taken once the jitter is 0, and with the entries charged through the
    compute audit they are None. ``stop`` is whether a stop block holds a response's output count to its ids, which
    only the lengths' change reads: without one the server picks the count the length is held to, so the schedules
    stay each connection's, and only the fail cells give way to the pooled charge."""
    changes = frozenset(changes)
    if changes - CHANGES.keys():
        raise ValueError(f"a change is one of {', '.join(CHANGES)}, not {', '.join(sorted(changes - CHANGES.keys()))}")
    if type(cells) is not int or cells < 0:
        raise ValueError(f"a connection's egress cells are an integer from 0, not {cells!r}")
    if not (connections >= 0 and scans >= 0 and 0 <= failures <= connections):
        raise ValueError(f"connections and scans are at least 0 and the failed connections among them, not "
                         f"{connections!r}, {scans!r} and {failures!r} failed")
    held, cell, jitter = "lengths" in changes, cells.bit_length(), _jitter(dj, steps)
    return Terms(0 if held and stop else connections * cell,
                 failed(connections, failures, cells) if held else connections * cell,
                 0 if "timing" in changes else scans * jitter, None if "entries" in changes else entries)


def clock_terms(links: Any, phases: Any, *, syncs: int, steps: int, dj: int = 1) -> Terms:
    """The clock form's terms (the monitoring-period check's `terms`, beside its `rowsTerms`) over ``links`` links of
    ``phases`` execution phases each: the schedules, ⌈log2 syncs^phases⌉ a link, every assignment of a sync of Σ_sync
    to its execution phases; no fail cell, its parameters listing no connection; and the jitter, ⌈log2 dj^steps⌉ an
    execution phase, a step a bucket as the scan's. The entries are None: N2 holds counts, and what its link carries
    is the compute side's."""
    if type(syncs) is not int or syncs < 1 or not (links >= 0 and phases >= 0):
        raise ValueError(f"Σ_sync holds a positive integer of syncs, and links and execution phases are at least 0, "
                         f"not {syncs!r}, {links!r} and {phases!r}")
    exact = syncs & (syncs - 1) == 0
    a_link = math.ceil(phases * (syncs.bit_length() - 1) if exact else phases * math.log2(syncs))
    return Terms(links * a_link, 0, links * phases * _jitter(dj, steps), None)


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
        """A proof block's committed bits, an incorrect proof block's charge in the pooled form."""
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

    def terms(self, period: str | float | Fraction, reading: str) -> Terms:
        """K's terms over ``period`` in ``reading`` (`terms`, at D_j = 1): a request is a connection, and the
        entries are the free inputs of its sampling blocks and requests."""
        blocks, requests, free = self.blocks(period), self.rate * seconds(period), self.free(reading)
        return terms(requests, blocks, cells=self.egress_cells, steps=self.T,
                     entries=blocks * free["block"] + requests * free["request"])

    def bits(self, period: str | float | Fraction, reading: str, delta: float | Fraction | None = None) -> float:
        """K over ``period`` (a `PERIODS` name, or its seconds) in ``reading``: the per-period form, or with ``delta``
        the pooled form."""
        t = self.terms(period, reading)
        k = float(t.certifier + t.entries)
        if delta is None:
            return k
        f = fmax(delta)
        return k + f * math.log2(self.blocks(period) * 2 * self.T + 1) + self.b * f


def seconds(period: str | float | Fraction) -> float | Fraction:
    if isinstance(period, str):
        if period not in PERIODS:
            raise ValueError(f"a monitoring period is one of {', '.join(PERIODS)} or its seconds, not {period!r}")
        return PERIODS[period]
    if not period > 0:
        raise ValueError(f"a monitoring period's seconds are positive, not {period!r}")
    return period


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
    """The world's links: the machines of rho's world fleet (`RHO`'s ``fleet.world-h200``, the census's inference at
    the fleet's serving MFU), one network certifier on each machine's uplink."""
    return round(_toml(RHO)["fleet"][WORLD]["count"]["value"])


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
    network certifier on one uplink (δ None), or `PooledLeakAccepted`'s pooled at ``delta``."""
    one = link(cls, rate, T=T, tau=tau, egress_cells=egress_cells, scan_delta=scan_delta)
    bits = one.bits(period, reading, delta)
    statement = STATEMENTS["per_period" if delta is None else "pooled"]
    return Bound(bits, bits / weights.bits, delta, one.eps(period), statement, weights.label)


def datacenter(links: Sequence[LinkLoad] | Mapping[LinkLoad, int], period: str | float | Fraction, reading: str,
               delta: float | Fraction | None = None, *, weights: Weights = WEIGHTS) -> Bound:
    """K over a monitoring period summed over a datacenter's ``links`` (each a `LinkLoad`, or each with its count),
    each at δ / links (`STATEMENTS`'s ``datacenter``).

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


def at_link_rate(one: LinkLoad, links: int, period: str | float | Fraction, reading: str,
                 delta: float | Fraction | None = None, *, weights: Weights = WEIGHTS) -> Bound:
    """The scaling by links: K over a monitoring period summed over ``links`` links (`datacenter_links`,
    `world_links`), each ``one``, which is `datacenter` over that many of it. Every link certifies ``one``'s rate
    whatever the scope the links stand for serves, and the scope's requests past that rate are uncertified, so K
    counts none of them: it is a run's rate held on every link, and not the scope's K, which `table` counts over
    every request the scope serves."""
    return datacenter({one: links}, period, reading, delta, weights=weights)


# --- at a scope's load: the network certifier's term, scaled by requests ---------------------------------------------

#: what an input is: measured by a run, modeled from other figures, assumed, or simulated, where a stand-in runs in
#: place of a device that is not deployed and the figure is what that device would be charged
MARKS = ("measured", "modeled", "assumed", "simulated")
#: a file's status for a figure, as its mark
_MARK = {"measured": "measured", "modeled": "modeled", "derived": "modeled", "datasheet": "assumed",
         "assumed": "assumed", "ruled": "assumed"}
#: the jitter bounds each row is counted at: D_j = 1, where the jitter is 0, and D_j = 2, a bit a bucket
JITTERS = (1, 2)
#: the jitter bound of the term of record: two release schedules
OF_RECORD = 2
#: the label every figure that rests on a set-time card is printed under: a figure at D_j = 1, what a network
#: certifier whose network card releases at a set time would be charged, and at either D_j a figure of a row whose
#: release times are taken once (`CHANGES`' timing). The card is simulated (`SET_TIME_CLAUSE`)
SET_TIME = "with a simulated set-time card"
#: the mark of the set-time card and of the figure printed under its label, one of `MARKS`
SIMULATED = "simulated"
#: what a figure under `SET_TIME` rests on, the source of the card's input and of that figure
SET_TIME_CLAUSE = ("`trusted-host`, set-time card clause: the network certifier's card releases each bucket at the "
                   "tick its record names, whatever the host and the server send it. It holds only of a calibrated "
                   "hardware card; no such card is deployed, and the software pacer that stands in for it does not "
                   "meet it.")
SCOPES = ("node", "datacenter", "world")
#: the assumed class's defaults: the most ids a response holds, an id's bits and an execution phase's buckets
RESPONSE_IDS, ID_BITS, BUCKETS = 4096, 18, 1280
#: the scan's slot stride in buckets, the scan class's default
EVERY: int = IS.Class.__dataclass_fields__["EVERY"].default
#: the lengths rows' cost runner, whose default schedule (`cost`'s) the count's inputs are read at
COST_RUNNER = "benchmarks/network_traces/connection_rows_run.py"
#: that schedule: its content classes, its delay D in buckets, which its ingress buckets equal, its lanes and its
#: execution phase's buckets. The count is at the first two where a caller names none; the last two are the schedule
#: `render` prints the count at beside the command's own
CLASSES, DELAY, COST_LANES, COST_BUCKETS = 6, 30, 16, 600
#: the GB the term is read against
TARGET_GB = 300
#: the request a scope's requests are counted of, which every row names: one a caller chooses (`REQUESTS`), or none
#: where the caller gives the world's requests a year
COUNTS = {
    "plan": "a scope serves the sampling plan's request, the 128k serving plan's as the break-even planner prices "
            "it (--requests plan, the default)",
    "census": "a scope serves the census's request, at the workload card's served MACs a token (--requests census)",
    "given": "the world's requests a year are given, of no stated request (--world-requests)",
}
REQUESTS = ("plan", "census")


class Kind(NamedTuple):
    """One of `ROWS`: where it stands (a check on main, a change to the scan's check, or a design), the changes it
    makes to the scan's check (`CHANGES`; None for the clock form, which is another check), what it counts and by
    which rule, whether a stop block holds a response's output count to its ids (`terms`' ``stop``), and whether its
    check reads the head that states each content class's held rows, so that it is charged the count (`counted`)."""
    standing: str
    changes: tuple[str, ...] | None
    counts: str
    stop: bool = True
    count: bool = False


#: a scope's eight rows over a monitoring period: the two checks on main, the scan under each of `CHANGES` and under
#: all three, and the design, without its stop block and with it
ROWS = {
    "scan": Kind("on main", (),
                 "the rows form on the network certifier's own run, as the check charges it: a schedule and a fail "
                 "cell for each connection of the period's parameters, counted by request, and the jitter of each "
                 "link's execution phase"),
    "N2": Kind("on main", None,
               "the clock form: the schedules, ⌈log2 #Σ_sync^m⌉ over a link's m execution phases, counted by link "
               "whatever it serves; no fail cell, its parameters listing no connection; and the jitter of each "
               "link's execution phase"),
    **{name: Kind("a change to the scan", (name,), what) for name, what in CHANGES.items()},
    "all three": Kind("a change to the scan", tuple(CHANGES),
                      "the three changes together, with the stop block the lengths' change rests on and the "
                      "simulated set-time card the timing's change rests on"),
    "design, no stop block": Kind(
        "a design with no run behind it", ("lengths", "entries"),
        "the design before a stop block, a proof block that checks where a response's ids stop, has been sampled on "
        "a served audit: the lengths unit holds a response's length to its Program's output count, but nothing "
        "holds that count to the ids, so the server picks it and each connection is still charged its schedule, as "
        "the scan charges it; its failed connections' pooled charge, its jitter, its count and its entries are the "
        "design's",
        stop=False, count=True),
    "design": Kind("a design with no run behind it", ("lengths", "entries"),
                   "head checks, a lengths unit on the network certifier's records, and a stop block sampled on a "
                   "served audit: the lengths are held, the stop block holding a response's output count to its "
                   "ids, and the entries are the compute audit's, so its term is the jitter, the failed "
                   "connections' pooled charge and the count, what the head of each link's execution phase states "
                   "of the rows each content class holds, counted by link",
                   count=True),
}


@dataclass(frozen=True)
class Assumed:
    """What `rowsTerms` reads of a scan class, for a model no run has certified one for: at most KO output ids a
    response, of IDB bits each (`ingress_scan.Class`'s names, so that `ingress_scan.entry_bits` lays its slot entry
    out), in execution phases of T buckets of ``tau`` seconds."""
    KO: int
    IDB: int
    T: int
    tau: float | Fraction

    def __post_init__(self) -> None:
        if not all(type(x) is int and x >= 1 for x in (self.KO, self.IDB, self.T)) or self.IDB > 32 or not self.tau > 0:
            raise ValueError(f"a class holds at least 1 id a response, of 1 to 32 bits, in execution phases of at "
                             f"least 1 bucket of positive seconds, not KO = {self.KO!r}, IDB = {self.IDB!r}, "
                             f"T = {self.T!r} and τ = {self.tau!r}")

    @property
    def cells(self) -> int:
        """egress_cells, the most cells a response of KO ids takes, KO + 1 (`ingress_scan.Class.serves`)."""
        return self.KO + 1

    @property
    def cell(self) -> int:
        """⌈log2(egress_cells + 1)⌉, a connection's bits for its schedule and again for its fail cell."""
        return self.cells.bit_length()

    @property
    def entry(self) -> int:
        """A slot entry's bits."""
        return IS.entry_bits(self)

    @property
    def phase_s(self) -> float | Fraction:
        """An execution phase's seconds."""
        return self.T * self.tau


class Input(NamedTuple):
    """An input of the scaling by requests: its value and unit, its mark (`MARKS`) and where it is read from."""
    value: Any
    unit: str
    mark: str
    source: str


def plan() -> dict[str, Input]:
    """A request as the sampling plan prices it, and the world's requests a year of it: the planner's case
    `PLAN_CASE` as the planner builds it (`sweep.build`, whose `reprice.workload` sets the 128k serving plan's tokens
    a request and served MACs a token and counts each scope's requests by `figure1.scopes`). The planner is loaded by
    path, and builds Figure 1's lines from the cards as it loads."""
    if (path := str(ROOT / PLANNER)) not in sys.path:
        sys.path.insert(0, path)
    import sweep
    _, cards, *_, at = sweep.build(sweep.cases()[PLAN_CASE])
    tokens, by = cards["tokens_per_request"], f"{PLANNER}/sweep.py's case {PLAN_CASE}"
    return {
        "request_tokens": Input(tokens.value, "token", _MARK[tokens.status], f"{by}: {tokens.source}"),
        "token_macs": Input(at["served_macs"], "MAC", "modeled",
                            f"{by}: the workload card's served MACs a token with the indexer's blocks at the 128k "
                            "serving plan's keys (reprice.workload)"),
        "world_requests": Input(at["scopes"]["world year"].requests, "request/year", "modeled",
                                f"{by}: its world year's requests (reprice.workload), by the rule "
                                "benchmarks/spot_figures/figure1.py's scopes plans with"),
    }


def inputs(*, response_ids: int = RESPONSE_IDS, id_bits: int = ID_BITS, buckets: int = BUCKETS,
           bucket_s: float | Fraction = TAU, failure_rate: float | Fraction = 0, requests: str = "plan",
           world_requests: float | None = None, scan_delta: float | Fraction = SCAN_DELTA,
           syncs: int | None = None, lanes: int | None = None, classes: int = CLASSES, delay: int = DELAY,
           ingress_buckets: int | None = None) -> dict[str, Input]:
    """Every input of the scaling by requests, each marked (`MARKS`) with its source.

    The scale is read from the files: the world's machines and their serving MFU (rho's world fleet), a machine's
    GPUs (rho's hardware), the reference datacenter's machines and a GPU's peak (the census), and the weights (the
    workload card). The world's requests a year follow from them and a request's tokens and served MACs a token:
    the sampling plan's (``requests`` = ``plan``, read from the planner by `plan`) or the census's tokens at the
    workload card's MACs (``census``); or they are ``world_requests`` where a caller gives them, whatever
    ``requests`` is. ``request_count`` names which of `COUNTS` it is. The class is assumed, no run having certified
    one for the served model: ``response_ids``, the most ids a response holds, ``id_bits``, an execution phase's
    ``buckets`` of ``bucket_s`` seconds (by default the network certifier's), one connection a request and one link a
    machine, and ``lanes``, the connections a link holds at once, which the entries and the count read: with None the
    entries are counted one a request and the count at the least lanes that serve a link's load (`counted`).
    ``classes``, ``delay`` and ``ingress_buckets`` are the rest of the schedule the count is at: its content classes,
    the buckets from a connection's admission to its first egress cell's, and the buckets a release is within of its
    connection's admission, by default the delay's; each is by default the lengths rows' cost runner's
    (`COST_RUNNER`). ``syncs`` is N2's #Σ_sync, by default the most a calibration chooses. ``failure_rate`` is
    the share of connections that fail, which only the rows with the lengths held read, and ``scan_delta`` each
    execution phase's scan's δ. ``set_time_card`` is the card every figure under `SET_TIME` rests on: it is
    simulated, the one input so marked (`SIMULATED`), and its source is the clause it rests on (`SET_TIME_CLAUSE`)."""
    if requests not in REQUESTS:
        raise ValueError(f"the request counted is one of {', '.join(REQUESTS)}, not {requests!r}")
    counted = "given" if world_requests is not None else requests
    census, dc = _reference("adversary.json"), _reference("deployments.json")
    peak = next(e for e in _json(HARDWARE) if e["id"] == PEAK)
    fleet, card = _toml(RHO)["fleet"][WORLD], _json(WORKLOAD)
    count, mfu = fleet["count"], fleet["mfu_serve"]
    gpus = _toml(RHO_HARDWARE)["location"][fleet["location"]]["devices"]
    if dc["gpus_per_machine"] != gpus["value"]:
        raise ValueError(f"a machine has one count of GPUs: {DEPLOYMENTS} gives {dc['gpus_per_machine']} and "
                         f"{RHO_HARDWARE} {gpus['value']}")
    work = "" if not card["provisional"] else ", provisional"
    macs = card["value"]["served_macs_per_token"]
    chosen = inspect.signature(calibration.choose_sync_set).parameters
    out = {
        "world_machines": Input(round(count["value"]), "machine", _MARK[count["status"]],
                                f"{RHO} fleet.{WORLD}.count: {count['source']}"),
        "datacenter_machines": Input(dc["machines"], "machine", "modeled", f"{DEPLOYMENTS} {dc['id']}.machines"),
        "machine_gpus": Input(gpus["value"], "GPU", _MARK[gpus["status"]],
                              f"{RHO_HARDWARE} location.{fleet['location']}.devices, the data sheet's: "
                              f"{gpus['source']}"),
        "mfu_serve": Input(mfu["value"], "", _MARK[mfu["status"]],
                           f"{RHO} fleet.{WORLD}.mfu_serve, {mfu['low']} to {mfu['high']}: {mfu['source']}"),
        "peak_flops": Input(peak["flop_per_second"], "FLOP/s", "assumed",
                            f"{HARDWARE} {PEAK}.flop_per_second: the data sheet's dense FP8 peak, the H200's too "
                            "(one die, one clock)"),
        "request_tokens": Input(census["tokens_per_request"], "token", "modeled",
                                f"{ADVERSARY} {census['id']}.tokens_per_request"),
        "token_macs": Input(macs, "MAC", _MARK[card["status"]], f"{WORKLOAD} served_macs_per_token{work}"),
        "weights": Input(WEIGHTS.bits, "bit", _MARK[card["status"]],
                         f"{WORKLOAD} params{work}, {FP4} bits each: {WEIGHTS.label}"),
    }
    if counted == "plan":
        out |= plan()
    elif counted == "census":
        flop = out["world_machines"].value * gpus["value"] * PERIODS["year"] * mfu["value"] * peak["flop_per_second"]
        out["world_requests"] = Input(
            flop / (census["tokens_per_request"] * 2 * macs), "request/year", "modeled",
            "the world's GPU-seconds a year x mfu_serve x peak_flops / (request_tokens x 2 x token_macs), the rule "
            "benchmarks/spot_figures/figure1.py's scopes plans with")
    else:
        out["world_requests"] = Input(world_requests, "request/year", "assumed", "given (--world-requests)")
    out |= {
        "request_count": Input(counted, "", "assumed", COUNTS[counted]),
        "connections": Input(1, "connection/request", "assumed", "each request on its own connection"),
        "links": Input(1, "link/machine", "assumed", "one network certifier on each machine's uplink"),
        "response_ids": Input(response_ids, "id", "assumed",
                              f"no certified {MODEL} class: the most ids a response holds (--response-ids)"),
        "id_bits": Input(id_bits, "bit", "assumed",
                         f"no certified {MODEL} class: an id of the vocabulary "
                         "(adapters/vllm/tests/check/fixtures/glm_moe_dsa/config.json's vocab_size)"),
        "phase_buckets": Input(buckets, "bucket", "assumed",
                               f"no certified {MODEL} class: an execution phase's buckets (--phase-buckets)"),
        "bucket_s": Input(bucket_s, "s", "assumed",
                          f"no certified {MODEL} class: a bucket's seconds, by default the network certifier's "
                          "(--bucket-s)"),
        "lanes": Input(lanes, "lane", "assumed",
                       "none given (--lanes): the entries are counted one a request, the least the check can charge, "
                       "and the count at the least lanes whose schedule releases what a link serves in an execution "
                       "phase" if lanes is None else
                       f"no certified {MODEL} class: the connections a link holds at once, each lane with a slot "
                       f"entry every {EVERY} buckets, the scan class's default stride (--lanes)"),
        "classes": Input(classes, "class", "assumed",
                         f"no certified {MODEL} class: the content classes a schedule files a link's releases in, "
                         f"whose held rows the count is of; by default {COST_RUNNER} cost's (--classes)"),
        "delay": Input(delay, "bucket", "assumed",
                       f"no certified {MODEL} class: the buckets from a connection's admission to its first egress "
                       f"cell's; by default {COST_RUNNER} cost's (--delay)"),
        "ingress_buckets": Input(delay if ingress_buckets is None else ingress_buckets, "bucket", "assumed",
                                 f"no certified {MODEL} class: the buckets a release is within of its connection's "
                                 f"admission; by default the delay's, as {COST_RUNNER} cost sets it "
                                 "(--ingress-buckets)"),
        "syncs": Input(chosen["n_delta"].default * chosen["n_rho"].default if syncs is None else syncs, "sync",
                       "assumed",
                       "N2's #Σ_sync, the most a calibration chooses by default: "
                       "verity/protocols/certifier/verifier/calibration.py choose_sync_set's n_delta x n_rho"
                       if syncs is None else "N2's #Σ_sync, given (--syncs)"),
        "failure_rate": Input(failure_rate, "failure/connection", "assumed",
                              "none measured: the share of connections that fail, which the rows with the lengths "
                              "held charge pooled (--failure-rate)"),
        "scan_delta": Input(scan_delta, "", "assumed", "each execution phase's scan's δ: `scan_zk --delta`'s default"),
        "set_time_card": Input(SIMULATED, "", SIMULATED, SET_TIME_CLAUSE),
    }
    if off := sorted(k for k, i in out.items() if i.mark not in MARKS or not i.source):
        raise ValueError(f"an input is marked one of {', '.join(MARKS)}, with its source: {', '.join(off)}")
    return out


class Scope(NamedTuple):
    """A scope: its machines, each behind one network certifier, and its GPUs."""
    name: str
    machines: int
    gpus: int


def scopes(inp: Mapping[str, Input]) -> list[Scope]:
    """`SCOPES`: a node, the census's reference datacenter and the world, each of machines of the world fleet's
    GPUs."""
    sizes = (1, inp["datacenter_machines"].value, inp["world_machines"].value)
    return [Scope(name, n, n * inp["machine_gpus"].value) for name, n in zip(SCOPES, sizes)]


class Row(NamedTuple):
    """One scope over one monitoring period in one of `ROWS`: the requests it serves, its links' execution phases,
    the terms at each jitter bound of `JITTERS`, and ε, its execution phases' scans' δ summed."""
    scope: Scope
    period: str
    name: str
    requests: Fraction
    phases: Fraction
    terms: dict[int, Terms]
    eps: Fraction


def _class(inp: Mapping[str, Input]) -> Assumed:
    return Assumed(inp["response_ids"].value, inp["id_bits"].value, inp["phase_buckets"].value,
                   Fraction(inp["bucket_s"].value))


class Count(NamedTuple):
    """What a link's execution phase is charged for the count, and the schedule it is counted at. The head of an
    execution phase states how many of its releases each of ``classes`` content classes holds: one of ``releases``
    + 1 values a class, ``releases`` being the most the schedule releases in an execution phase (`phase_releases`),
    so ``bits`` is ``classes`` · log2(``releases`` + 1). ``lanes`` are the schedule's: a caller's where ``given``,
    and otherwise the least whose schedule releases ``load``, the connections a link serves in an execution phase."""
    classes: int
    lanes: int
    given: bool
    releases: int
    load: Fraction
    bits: float


def counted(inp: Mapping[str, Input]) -> Count:
    """The count's charge at ``inp``'s schedule (`Count`): its classes, its delay and ingress buckets, its execution
    phase's buckets and its lanes. With no lanes given the schedule is not known, so the count is at the least lanes
    whose schedule releases what a link serves in an execution phase: any schedule that serves this load releases at
    least as many, so this is the least the rule can charge at it."""
    cls, lanes, classes = _class(inp), inp["lanes"].value, inp["classes"].value
    if type(classes) is not int or classes < 0:
        raise ValueError(f"a schedule's content classes are an integer from 0, not {classes!r}")
    world = scopes(inp)[-1]
    load = (Fraction(inp["world_requests"].value) * inp["connections"].value * cls.phase_s
            / (PERIODS["year"] * world.machines * inp["links"].value))
    a_lane = phase_releases(1, cls.T, inp["delay"].value, inp["ingress_buckets"].value)
    given = lanes is not None
    if not given:
        lanes = -(-math.ceil(load) // a_lane)
    releases = phase_releases(lanes, cls.T, inp["delay"].value, inp["ingress_buckets"].value)
    return Count(classes, lanes, given, releases, load, classes * math.log2(releases + 1))


def table(inp: Mapping[str, Input]) -> list[Row]:
    """Each check's terms (`ROWS`) at each scope's load (`scopes`) over each monitoring period (`PERIODS`), from
    ``inp`` (`inputs`), in exact arithmetic but for the failed connections' charge and the count. A scope serves the
    world's requests a year in proportion to its GPU-seconds, so every row counts all its scope serves: the scan's
    connections by request, the links' execution phases by link, and the entries by capacity where the lanes are
    given, never by requests what a check counts by capacity. The count (`counted`) is by link, for each of its
    execution phases, on the rows whose check reads the head that states it (`Kind.count`)."""
    cls, lanes, syncs = _class(inp), inp["lanes"].value, inp["syncs"].value
    failing, scan_delta = Fraction(inp["failure_rate"].value), Fraction(inp["scan_delta"].value)
    if not (0 <= failing <= 1 and 0 < scan_delta < 1 and inp["world_requests"].value >= 0):
        raise ValueError(f"the failure rate is in [0, 1], the scan's δ in (0, 1) and the world's requests at least 0, "
                         f"not {failing!r}, {scan_delta!r} and {inp['world_requests'].value!r}")
    if lanes is not None and (type(lanes) is not int or lanes < 1 or cls.T % EVERY):
        raise ValueError(f"a link's lanes are a positive integer and its execution phase's buckets a multiple of the "
                         f"slot stride's {EVERY}, not {lanes!r} lanes and {cls.T} buckets")
    world, a_phase = scopes(inp)[-1], counted(inp).bits
    a_gpu_second = Fraction(inp["world_requests"].value) / (world.gpus * PERIODS["year"])
    rows = []
    for scope in scopes(inp):
        for period, s in PERIODS.items():
            requests = a_gpu_second * scope.gpus * s
            connections = requests * inp["connections"].value
            links, each = scope.machines * inp["links"].value, Fraction(s) / cls.phase_s
            phases = links * each
            entries = requests * cls.entry if lanes is None else phases * lanes * (cls.T // EVERY) * cls.entry
            for name, kind in ROWS.items():
                if kind.changes is None:
                    at = {dj: clock_terms(links, each, syncs=syncs, steps=cls.T, dj=dj) for dj in JITTERS}
                else:
                    at = {dj: terms(connections, phases, cells=cls.cells, steps=cls.T, dj=dj, entries=entries,
                                    changes=kind.changes, failures=failing * connections, stop=kind.stop)
                          for dj in JITTERS}
                if kind.count:
                    at = {dj: t._replace(count=float(phases) * a_phase) for dj, t in at.items()}
                rows.append(Row(scope, period, name, requests, phases, at, phases * scan_delta))
    return rows


def afforded(bits: float, connections: Any, cells: int) -> float:
    """The largest failure rate at which the failed connections' pooled charge (`failed`) over ``connections``
    connections is at most ``bits``."""
    n = float(connections)
    if not n or failed(n, n, cells) <= bits:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if failed(n, n * mid, cells) <= bits else (lo, mid)
    return lo


def _size(bits: Any) -> dict | None:
    return None if bits is None else {"bits": float(bits), "GB": float(bits / GB)}


def document(inp: Mapping[str, Input], *, target_gb: float | Fraction = TARGET_GB, commit: str | None = None,
             run: str | None = None, only: Iterable[str] = (), periods: Iterable[str] = ()) -> dict:
    """`table`'s rows as a document: each row's terms in bits and GB, the network certifier's term at each jitter
    bound beside ``target_gb`` and the weights, an execution phase of one link and a request; what each row counts
    and where it stands (`ROWS`); the inputs; and at the top the term of record, the term over the world's year as
    the scan's check charges it today at two release schedules (D_j = `OF_RECORD`), as ``value`` in ``unit``, with
    the ``commit`` and the ``run`` that computed it. Beside it ``with_set_time_card`` holds the same term at D_j = 1,
    its unit, the label it is printed under (`SET_TIME`) and its ``mark``, the card's own (``inp``'s
    ``set_time_card``: simulated). A row's own ``with_set_time_card`` is that label where its changes include the
    timing's, so that its jitter is 0 at either D_j and every figure of it rests on the card, and None on every
    other row. A row's ``count`` is its count's charge, 0 where its check reads no head that states one, and the
    document's ``count`` says how it is counted (`Count`): the schedule, R as ``releases``, the bits a class and a
    link's execution phase, whether the schedule releases what a link serves (``holds_load``), the rows it is on, its
    charge over the world's year beside the target, and the same rule at the cost runner's own schedule
    (``cost_runner``). ``failure_rate_at_target`` is the largest failure rate at which the design's term over the
    world's year, with its release times taken once, is at most the target: the failed connections' charge and the
    count; it is None where the count alone is over the target. ``only`` and ``periods`` keep the rows of those
    scopes and monitoring periods; ``covers`` is the share of its scope's compute a row counts, and
    ``request_count`` which of `COUNTS` its requests are, the document's one."""
    only, periods, target = set(only) or set(SCOPES), set(periods) or set(PERIODS), Fraction(target_gb) * GB
    if not target > 0 or only - set(SCOPES) or periods - set(PERIODS):
        raise ValueError(f"the target is positive, a scope one of {', '.join(SCOPES)} and a monitoring period one of "
                         f"{', '.join(PERIODS)}, not {target_gb!r}, {sorted(only)} and {sorted(periods)}")
    cls, lanes, request_count = _class(inp), inp["lanes"].value, inp["request_count"].value

    def term(r: Row, dj: int) -> dict:
        t = Fraction(r.terms[dj].certifier)
        return {**_size(t), "of_target": float(t / target), "of_weights": float(t / WEIGHTS.bits),
                "bits_a_phase": float(t / r.phases), "bits_a_request": float(t / r.requests) if r.requests else None}

    rows = [{"scope": r.scope.name, "period": r.period, "row": r.name, "standing": ROWS[r.name].standing,
             "with_set_time_card": SET_TIME if "timing" in (ROWS[r.name].changes or ()) else None,
             "machines": r.scope.machines, "gpu_seconds": r.scope.gpus * PERIODS[r.period],
             "request_count": request_count, "requests": float(r.requests), "phases": float(r.phases), "covers": 1.0,
             "eps": float(r.eps), "schedules": _size(r.terms[1].schedules), "cells": _size(r.terms[1].cells),
             "count": _size(r.terms[1].count),
             "jitter": {str(dj): _size(r.terms[dj].jitter) for dj in JITTERS},
             "term": {str(dj): term(r, dj) for dj in JITTERS}, "entries": _size(r.terms[1].entries)}
            for r in table(inp)]
    head = next(r for r in rows if (r["scope"], r["period"], r["row"]) == ("world", "year", "scan"))
    connections = head["requests"] * inp["connections"].value
    count, delay, ingress = counted(inp), inp["delay"].value, inp["ingress_buckets"].value
    held = count.bits * head["phases"]
    at_cost = phase_releases(COST_LANES, COST_BUCKETS, DELAY, DELAY)
    return {"schema": "certifier-term/v2",
            "name": f"the network certifier's term for {MODEL} over the world's year, as the scan's check charges "
                    f"it today at two release schedules (D_j = {OF_RECORD})",
            "value": head["term"][str(OF_RECORD)]["GB"], "unit": "GB",
            "with_set_time_card": {"value": head["term"]["1"]["GB"], "unit": "GB", "label": SET_TIME,
                                   "mark": inp["set_time_card"].mark},
            "commit": commit, "run": run, "model": MODEL, "request_count": request_count,
            "target": {"value": float(target_gb), "unit": "GB", "bits_a_phase": float(target) / head["phases"],
                       "bits_a_request": float(target) / head["requests"] if head["requests"] else None},
            "weights": {"value": WEIGHTS.bits / GB, "unit": "GB", "label": WEIGHTS.label},
            "phase": {"seconds": float(cls.phase_s), "buckets": cls.T,
                      "requests": head["requests"] / head["phases"], "connection_bits": cls.cell,
                      "sync_bits": clock_terms(1, 1, syncs=inp["syncs"].value, steps=cls.T).schedules,
                      "entry_bits": cls.entry},
            "count": {"rows": [name for name, kind in ROWS.items() if kind.count], "classes": count.classes,
                      "lanes": count.lanes, "lanes_given": count.given, "delay": delay, "ingress_buckets": ingress,
                      "releases": count.releases, "holds_load": count.releases >= count.load,
                      "bits_a_class": math.log2(count.releases + 1), "bits_a_phase": count.bits,
                      "world_year": {**_size(held), "of_target": held / float(target)},
                      "cost_runner": {"lanes": COST_LANES, "buckets": COST_BUCKETS, "delay": DELAY,
                                      "ingress_buckets": DELAY, "releases": at_cost,
                                      "bits_a_phase": count.classes * math.log2(at_cost + 1)}},
            "failure_rate_at_target": (afforded(float(target) - held, connections, cls.cells)
                                       if held <= target else None),
            "entries_rule": "one entry a request, the least the check can charge: it charges every slot's entry, a "
                            "count by capacity that needs the class's lanes" if lanes is None else
                            f"every slot's entry, by capacity: {lanes} lanes x {cls.T // EVERY} strides a link's "
                            "execution phase, whatever it serves",
            "legend": {name: {"standing": k.standing, "counts": k.counts} for name, k in ROWS.items()},
            "inputs": {k: {"value": float(i.value) if isinstance(i.value, Fraction) else i.value, "unit": i.unit,
                           "mark": i.mark, "source": i.source} for k, i in inp.items()},
            "rows": [r for r in rows if r["scope"] in only and r["period"] in periods]}


def _n(x: float | str | None, none: str = "none") -> str:
    """A number to four significant figures, a name as it is, and ``none`` for None."""
    if x is None:
        return none
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1e9 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def _pipes(head: Sequence[str], body: Sequence[Sequence[str]], text: int) -> list[str]:
    """A pipe table, its first ``text`` columns set left and the rest right."""
    widths = [max(len(r[i]) for r in (head, *body)) for i in range(len(head))]

    def line(r: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(w) if i < text else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))) + " |"

    rule = ["-" * w if i < text else "-" * (w - 1) + ":" for i, w in enumerate(widths)]
    return [line(head), line(rule), *map(line, body)]


def render(doc: dict) -> str:
    """``doc`` (`document`) as text: what each row counts, the inputs, the scopes, and the terms in GB and in
    bits, the columns of the term of record (D_j = `OF_RECORD`) before those at D_j = 1."""
    ph, tg, theirs = doc["phase"], doc["target"], "the compute side's"
    at = [str(dj) for dj in sorted(JITTERS, key=lambda dj: dj != OF_RECORD)]
    counted, seen = doc["request_count"], {(r["scope"], r["period"]): r for r in doc["rows"]}
    card, ct = doc["with_set_time_card"], doc["count"]
    cost, rate = ct["cost_runner"], doc["failure_rate_at_target"]
    # the scan's schedules and fail cells are the same at each jitter bound and its jitter is 0 at D_j = 1, so what
    # the term of record holds past the labelled term is the jitter alone
    jitter = doc["value"] - card["value"]

    def entries(r: dict, unit: str) -> str:
        return _n(r["entries"] and r["entries"][unit], theirs)

    def short(label: str | None) -> str:
        """A label's short form after a comma, as a header and a row's name carry it, and nothing for no label."""
        return f", {label.removeprefix('with a ')}" if label else ""

    def head(what: str, dj: str) -> str:
        """A column's header at a jitter bound: at D_j = 1 it carries the label's short form."""
        return f"{what}, D_j = {dj}" + short(None if int(dj) == OF_RECORD else card["label"])

    def name(r: dict) -> str:
        """A row's name in the tables: a row that rests on the set-time card at either D_j carries the label's short
        form, as a header does."""
        return r["row"] + short(r["with_set_time_card"])

    lines = [
        f"# {doc['name'][0].upper()}{doc['name'][1:].split(' over ')[0]}, at a scope's load", "",
        f"commit {doc['commit'] or 'unknown'}, run {doc['run'] or 'none'}", "",
        "Each row counts all its scope serves over the monitoring period, by its check's own rule. The term is the "
        "schedules, the fail cells, the jitter and, on a row whose check reads the head that states it, the count. "
        "The entries are the compute side's.",
        f"The term of record is the one at D_j = {OF_RECORD}, two release schedules. The term at D_j = 1 is what a "
        f"network certifier whose network card releases at a set time would be charged. That card is {card['mark']}, "
        "the input set_time_card, whose source is what the term rests on: none is deployed, so it is printed as "
        f"\"{card['label']}\".",
        "A row whose release times are taken once has a jitter of 0 at either D_j, which rests on the same card, so "
        "its name carries the label.",
        f"Request count {counted}: {COUNTS[counted]}. Every row names its request count, and rows of two request "
        "counts don't compare.",
        "The rows of a scope and a monitoring period:",
        *(f"- {name} ({k['standing']}): {k['counts']}" for name, k in doc["legend"].items()),
        f"The entries as the scan's check charges them today: {doc['entries_rule']}.",
        f"A link's execution phase is {_n(ph['seconds'])} s of {_n(ph['buckets'])} buckets: {_n(ph['requests'])} "
        f"requests, each charged {ph['connection_bits']} bits for its schedule and again for its fail cell; N2's "
        f"schedules are {_n(ph['sync_bits'])} bits; the jitter is {_n(ph['buckets'])} bits at D_j = {OF_RECORD}, and "
        f"0 at D_j = 1 {card['label']}; a slot entry is {_n(ph['entry_bits'])} bits.",
        f"The count, on the rows {' and '.join(f'`{name}`' for name in ct['rows'])}: the head of a link's execution "
        "phase states how many of "
        f"its releases each of {_n(ct['classes'])} content classes holds, one of R + 1 values a class, so a link's "
        f"execution phase is charged {_n(ct['classes'])} x log2(R + 1) = {_n(ct['bits_a_phase'])} bits, whatever it "
        f"serves. R = {_n(ct['releases'])} is the most connections a schedule of {_n(ct['lanes'])} lanes releases in "
        f"an execution phase: {_n(ct['lanes'])} x ⌈({_n(ph['buckets'])} + {_n(ct['ingress_buckets'])} - 1) / "
        f"({_n(ct['delay'])} + 1)⌉. "
        + ("The lanes are given (--lanes)" if ct["lanes_given"] else
           f"No lanes are given (--lanes), so they are the least whose schedule releases the {_n(ph['requests'])} "
           "requests a link serves in an execution phase, the least the rule can charge at this load")
        + ("." if ct["holds_load"] else
           f": that schedule releases fewer than the {_n(ph['requests'])} requests a link serves in an execution "
           "phase, so it does not serve this load.")
        + f" At the lengths rows' cost runner's schedule, {_n(cost['lanes'])} lanes and an execution phase of "
          f"{_n(cost['buckets'])} buckets, R = {_n(cost['releases'])} and the count is {_n(cost['bits_a_phase'])} "
          "bits a link's execution phase.",
        f"The target is {_n(tg['value'])} GB: over the world's year, {_n(tg['bits_a_phase'])} bit a link's execution "
        f"phase and {_n(tg['bits_a_request'])} bit a request. The weights are one copy of {doc['weights']['label']}, "
        f"{_n(doc['weights']['value'])} GB.",
        "With the lengths held by a stop block and the release times taken once by a network certifier whose network "
        f"card releases at a set time ({card['mark']}), the design's term over the world's year is the failed "
        "connections' charge and the count. "
        + (f"The count alone is {_n(ct['world_year']['GB'])} {doc['unit']}, {_n(ct['world_year']['of_target'])} "
           "times the target, so the world's year meets the target at no failure rate." if rate is None else
           f"The count is {_n(ct['world_year']['GB'])} {doc['unit']}, and with it the world's year meets the target "
           f"at a failure rate of at most {_n(rate)}.")
        + f" Without that card the jitter alone, at D_j = {OF_RECORD}, is {_n(jitter)} {doc['unit']} over the "
          f"world's year, {_n(jitter / tg['value'])} times the target.",
        "ε is the scan's execution phases' δ summed, a union bound: at 1 or more it bounds nothing.", "",
        "## Inputs", "",
        *_pipes(("input", "value", "unit", "mark", "source"),
                [(k, _n(i["value"]), i["unit"], i["mark"], i["source"]) for k, i in doc["inputs"].items()], 5), "",
        "## The scopes: the compute each row covers, all its scope serves", "",
        *_pipes(("scope", "period", "request count", "machines", "GPU-seconds", "requests", "execution phases", "ε"),
                [(*k, *(_n(r[c]) for c in ("request_count", "machines", "gpu_seconds", "requests", "phases", "eps")))
                 for k, r in seen.items()], 3), "",
        "## The terms in GB", "",
        *_pipes(("scope", "period", "row", "request count", "schedules", "cells", "count",
                 *(head("jitter", dj) for dj in at),
                 *(head(c, dj) for dj in at for c in ("term", "x target", "x weights")),
                 "entries, the compute side's"),
                [(r["scope"], r["period"], name(r), r["request_count"], _n(r["schedules"]["GB"]),
                  _n(r["cells"]["GB"]), _n(r["count"]["GB"]), *(_n(r["jitter"][dj]["GB"]) for dj in at),
                  *(_n(r["term"][dj][k]) for dj in at for k in ("GB", "of_target", "of_weights")),
                  entries(r, "GB")) for r in doc["rows"]], 4), "",
        "## The terms in bits", "",
        *_pipes(("scope", "period", "row", "request count", "schedules", "cells", "count",
                 *(head("jitter", dj) for dj in at),
                 *(head("term", dj) for dj in at), "entries, the compute side's",
                 *(head("a link's phase", dj) for dj in at), *(head("a request", dj) for dj in at)),
                [(r["scope"], r["period"], name(r), r["request_count"], _n(r["schedules"]["bits"]),
                  _n(r["cells"]["bits"]), _n(r["count"]["bits"]), *(_n(r["jitter"][dj]["bits"]) for dj in at),
                  *(_n(r["term"][dj]["bits"]) for dj in at),
                  entries(r, "bits"), *(_n(r["term"][dj]["bits_a_phase"]) for dj in at),
                  *(_n(r["term"][dj]["bits_a_request"]) for dj in at)) for r in doc["rows"]], 4)]
    return "\n".join(lines)


def _head() -> str | None:
    """The checkout's commit, None outside a git checkout."""
    git = shutil.which("git")
    done = git and subprocess.run([git, "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done and done.returncode == 0 else None


def main(argv: Sequence[str] | None = None) -> int:
    """`verity bench certifier_term`: print the network certifier's term at each scope's load (`render`), or with
    ``--json`` its document (`document`)."""
    ap = argparse.ArgumentParser(
        prog="verity bench certifier_term",
        description=f"The network certifier's term for {MODEL} at the load of a node, the census's reference "
                    "datacenter and the world, over a week, a month and a year, in bits and GB: as each check on "
                    "main charges it by its own rule, under each of three changes to the scan's check, and for the "
                    "design that has no run behind it, without its stop block and with it: eight rows a scope and "
                    f"monitoring period. The term of record is the one at D_j = {OF_RECORD}, two release schedules; "
                    f"the term at D_j = 1 is printed as \"{SET_TIME}\".")
    ap.add_argument("--json", action="store_true",
                    help="print the document (value, the scan's term over the world's year at D_j = "
                         f"{OF_RECORD}; unit, commit, run, inputs, rows)")
    ap.add_argument("--response-ids", type=int, default=RESPONSE_IDS, metavar="N",
                    help=f"the most ids a response holds (default {RESPONSE_IDS})")
    ap.add_argument("--id-bits", type=int, default=ID_BITS, metavar="N", help=f"an id's bits (default {ID_BITS})")
    ap.add_argument("--phase-buckets", type=int, default=BUCKETS, metavar="T",
                    help=f"an execution phase's buckets (default {BUCKETS})")
    ap.add_argument("--bucket-s", type=Fraction, default=TAU, metavar="S",
                    help=f"a bucket's seconds (default {TAU}, the network certifier's)")
    ap.add_argument("--lanes", type=int, metavar="N",
                    help="the connections a link holds at once: the entries are then every slot's, by capacity, and "
                         "the count is at these lanes (default: none; the entries are one a request, the least, and "
                         "the count at the least lanes whose schedule releases what a link serves)")
    ap.add_argument("--classes", type=int, default=CLASSES, metavar="N",
                    help=f"the content classes whose held rows the count is of (default {CLASSES})")
    ap.add_argument("--delay", type=int, default=DELAY, metavar="D",
                    help="the buckets from a connection's admission to its first egress cell's, which the count's "
                         f"R reads (default {DELAY})")
    ap.add_argument("--ingress-buckets", type=int, metavar="I",
                    help="the buckets a release is within of its connection's admission, which the count's R reads "
                         "(default: the delay's)")
    ap.add_argument("--syncs", type=int, metavar="N",
                    help="N2's #Σ_sync (default: the most a calibration chooses)")
    ap.add_argument("--failure-rate", type=Fraction, default=Fraction(0), metavar="F",
                    help="the share of connections that fail, charged pooled where the lengths are held (default 0)")
    ap.add_argument("--requests", choices=REQUESTS, default=REQUESTS[0],
                    help="the request a scope's requests are counted of: the sampling plan's, read from the "
                         f"break-even planner, or the census's at the workload card's MACs (default {REQUESTS[0]})")
    ap.add_argument("--world-requests", type=float, metavar="N",
                    help="the world's requests a year, in place of either request's count")
    ap.add_argument("--scan-delta", type=Fraction, default=SCAN_DELTA, metavar="D",
                    help=f"each execution phase's scan's δ (default {SCAN_DELTA})")
    ap.add_argument("--target-gb", type=float, default=TARGET_GB, metavar="GB",
                    help=f"the GB the term is read against (default {TARGET_GB})")
    ap.add_argument("--scope", action="append", choices=SCOPES, help="only this scope's rows (repeatable)")
    ap.add_argument("--period", action="append", choices=tuple(PERIODS),
                    help="only this monitoring period's rows (repeatable)")
    ap.add_argument("--commit", help="the commit the document names (default: the checkout's HEAD)")
    ap.add_argument("--run", help="the run the document names (under `research run`, its id)")
    a = ap.parse_args(argv)
    inp = inputs(response_ids=a.response_ids, id_bits=a.id_bits, buckets=a.phase_buckets, bucket_s=a.bucket_s,
                 failure_rate=a.failure_rate, requests=a.requests, world_requests=a.world_requests,
                 scan_delta=a.scan_delta, syncs=a.syncs, lanes=a.lanes, classes=a.classes, delay=a.delay,
                 ingress_buckets=a.ingress_buckets)
    doc = document(inp, target_gb=a.target_gb, commit=a.commit or _head(), run=a.run, only=a.scope or (),
                   periods=a.period or ())
    print(json.dumps(doc, indent=1, ensure_ascii=False) if a.json else render(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
