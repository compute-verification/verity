"""Protocol soundness, the envelopes: each subprotocol's property of the actual circuit, its slack and its δ (the
working draft's §8).

An envelope `𝒜_a` narrows the physical model's class `𝒜_phys` if its verifier accepts, except with probability `δ_a`
summed over serving blocks. Each one here carries its parameters, the slack it leaves (shares of capacity by route, the
uncertified gates `U` no inference block constrains) and its `delta`, split into the coins' draws and the
cryptography's advantage:

* `Pouw`: the certified declared gates were evaluated, `|A[X,J] ∩ W_cert| ≥ (1 − ε_W)|W_cert|`. Its slack is `undecl`
  (monitored capacity running no certified declared work: the host CPUs, the GPUs outside the floor credit `residual`
  and the clock above the credited one, together the audited residual `r̂`) and `skip` (its deficit `ε_W` and the
  certificate slack `ε'`).
  Its δ is `p / ε'` with `p = δ_s + δ_bind + δ_ks` per window, by Markov over every window at once: `ε'` is the share
  whose certificates may be wrong, and `δ` the chance more are.
* `Detection`, sampled proofs with terminal detection: their slack is `break` (the verdict lag `JΔ/H` and, sampled,
  the faulty blocks accepted before detection `θ`) and `exfil.faulty`, Lean's `χ` (the compute that bits injected by
  accepted faulty blocks buy, at `chord` products a bit; not double counting, §8 "Channels"). Its δ is the escape
  bound's.
* `Wipe`: at each wipe a server keeps at most `k_wipe` bits of the earlier content of its whole memory `M_s` (`per =
  "server"`, the ruled operating point), or, the per-device sensitivity, each device
  `residual_bits` and each host `host_residual_bits` (`per = "device"`); a boot disk the wipe doesn't reach keeps
  `boot_disk_bits` more a server (today's wipe, `regime.onsite-today`). Its δ is `2⁻¹²⁸ + Adv_PRF` per game,
  one game per server (or device) per wipe, summed over the horizon.
* `Certifier`: a serving block behind certifiers reads at most `(r_in + r_side) Δ_wipe` unregistered bits from outside,
  `r_in` the certified prompt ingress and `r_side` the timing and side channels no certifier reads. `r_in` is
  `intake · α Φ`, at most `α Φ` since a server's ingress carries only its inference blocks' declared inputs;
  with `declared` (`regime.onsite-w63`), each server's declared inputs at a cadence the warden form's public parameters
  set. Its δ is 0: the certifier is the auditor's device (`trusted-host`), and the side channels an assumed cap.
* The NCI check is a registration check, not an envelope: the registered circuit's `(c, k)`-partition into inference
  blocks (`Policy`), registered once.

A **regime** is the envelopes that run: `Remote` (PoUW, detection and the NCI check; no wipe or certifier, so its
serving block is a server over the horizon and only the inference cut bounds it) and `Onsite` (with the wipe and the
certifier, whose serving block between two wipes gets the serving cut too). `Regime.partition` gives the serving blocks
(`theorem`), each with the cuts the envelopes make valid, and `Regime.delta` their δ summed over envelopes and serving
blocks. Its `layout` is what a protocol-abiding attack gets: the inference blocks and the uncertified gates a strategy
that never cheats can use (`ABIDING`), at the training MFU.

A regime's table is flat: each key is a field of one of its envelopes (or the regime's own `form`), and a table may name
a `base` table and override only what differs (`regime`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

from rho.hardware import Fleet, Horizon, Location, Units
from rho.params import Registry
from rho.theorem import Blocks, Budget, Cut, Partition, ServingBlock

#: the routes of the uncertified gates, the prefix of each slack term's key (`ε_undecl`, `ε_skip`, `ε_break`, and
#: Lean's `χ` as `exfil`)
ROUTES = ("undecl", "skip", "break", "exfil")


@dataclass(frozen=True)
class Policy:
    """The NCI check's registration: inference blocks of `gamma` FLOPs (`c` gates through `Units`), each reading at most
    `k` unregistered bits."""

    name: str
    gamma: float
    k: float
    #: the inference overhead: the honest inference workload's circuit under this partition over its unpartitioned
    #: size, a factor beside the training overhead. NaN where it isn't modeled (its source says so).
    inference_overhead: float = math.nan


@dataclass(frozen=True)
class Context:
    fleet: Fleet
    location: Location
    policy: Policy
    horizon: Horizon
    units: Units

    @property
    def c(self) -> float:
        """`c`: an inference block's gates."""
        return self.units.gates(self.policy.gamma)

    @property
    def capacity(self) -> float:
        """The monitored fleet's gates over the horizon at capacity."""
        return self.fleet.count * self.location.phi * self.horizon.seconds


@dataclass(frozen=True)
class Delta:
    """`Σ_a δ_a`, over envelopes and serving blocks, by envelope as `(coin, crypto)`. `Pr[Break]`, a forgery of the
    devices' logged coins (LMS), is named beside it and not bounded here."""

    by: Mapping[str, tuple[float, float]]

    @property
    def coin(self) -> float:
        return math.fsum(c for c, _ in self.by.values())

    @property
    def crypto(self) -> float:
        return math.fsum(c for _, c in self.by.values())

    def json(self) -> dict[str, Any]:
        return {"coin": self.coin, "crypto": self.crypto, "by": {k: {"coin": c, "crypto": x} for k, (c, x) in
                                                                  self.by.items()}}


@dataclass(frozen=True)
class Pouw:
    """PoUW's envelope. Each field is `benchmarks/nci/tiers.py`'s parameter of the same name."""

    #: the GPUs' share of capacity outside the floor credit (`tiers` `residual`)
    residual: float
    #: `ε_W`: PoUW's deficit, a share of certified work credited without its running
    eps_w: float
    #: `ε'`: the share of capacity whose certificates may be wrong (Markov over the windows)
    eps_agg: float
    #: a window's certificate is wrong with probability `p = δ_s + δ_bind + δ_ks` (`PouwSound`): `δ_s` from the coins'
    #: draw, `δ_bind + δ_ks` from the cryptography (commitment binding, the proof system's knowledge soundness)
    window_delta_coin: float = 0.0
    window_delta_crypto: float = 0.0
    #: the GPUs' capacity above the clock PoUW credits, as a share
    clock_share: float = 0.0
    #: the host CPUs counted at `host` times their peak (`Location.host_phi`): 0 under hosts carry no compute
    #: (`GLOSSARY.md`), 1 without that assumption
    host: float = 0.0

    CITES = ("Nci.Assumptions.PouwSound", "Nci.Guarantees.CertificateSlack")

    def audited(self, loc: Location) -> float:
        """`r̂ = 1 − W_cert/C`: the GPUs' uncredited share, their clock above the credited one, and the host CPUs."""
        return min(1.0, self.residual + self.clock_share + self.host * loc.host_phi / loc.phi)

    def slack(self, loc: Location) -> dict[str, float]:
        r = self.audited(loc)
        return {"undecl.host": self.host * loc.host_phi / loc.phi, "undecl.gpu": self.residual,
                "undecl.clock": self.clock_share, "skip.pouw": self.eps_w * (1 - r), "skip.certs": self.eps_agg}

    def delta(self, ctx: Context) -> tuple[float, float]:
        return self.window_delta_coin / self.eps_agg, self.window_delta_crypto / self.eps_agg


SAMPLING = ("proof_share", "overhead", "replay", "step_flop", "p_evade", "luck", "step_bits", "floor_steps", "chord",
            "chord_kappa")


@dataclass(frozen=True)
class Detection:
    """Terminal detection, the verdict lag and (with `sampled`) sampled proofs (tiers T5); off, every unit is proved
    (`q = 1`, `χ = θ = 0`). Each field is `tiers.py`'s parameter of the same name (`proof_share` is its `share`)."""

    #: `J`: identities a detection doesn't end; `Δ`: the verdict lag in seconds
    identities: float
    lag_s: float
    sampled: bool = False
    #: the sampled proofs' compute share, proving cost per unit of step work, replay cost per drawn step, work and
    #: output bits per step, failure probability `δ`, luck factor, and steps sampled at a floor of one
    proof_share: float = math.nan
    overhead: float = math.nan
    replay: float = math.nan
    step_flop: float = math.nan
    p_evade: float = math.nan
    luck: float = math.nan
    step_bits: float = math.nan
    floor_steps: float = math.nan
    #: counted multiplications per injected bit (`tiers.chord` at the injected bits), at `chord_kappa` products per
    #: FLOP: what turns `χ`'s products into the uncertified gates a cut carries
    chord: float = math.nan
    chord_kappa: float = math.nan
    #: `χ` given directly, in place of the sampled proofs' closed form (None: the closed form)
    cheat: float | None = None

    CITES = ("Nci.Assumptions.TerminalDetection", "Nci.Assumptions.SampledDetection", "Nci.Guarantees.VerdictLag")

    def __post_init__(self) -> None:
        unset = [k for k in SAMPLING if self.sampled and not math.isfinite(getattr(self, k))]
        if unset:
            raise ValueError(f"sampled proofs need {unset}")

    def q(self) -> float:
        """The detection probability per faulty block: the proof share over the cost of proving a unit."""
        return self.proof_share / (self.overhead + self.replay) if self.sampled else 1.0

    def injected(self) -> float:
        """Bits the accepted faulty proof blocks inject: `J ln(1/δ) (1/q + floor) · step_bits`."""
        if not self.sampled:
            return 0.0
        return self.identities * math.log(1 / self.p_evade) * (1 / self.q() + self.floor_steps) * self.step_bits

    def slack(self, ctx: Context) -> dict[str, float]:
        u, c = ctx.units, ctx.capacity
        out = {"break.lag": min(1.0, self.identities * self.lag_s / ctx.horizon.seconds)}
        if self.sampled:
            steps = self.luck * self.identities * math.log(1 / self.p_evade) / self.q()
            out["break.escape"] = steps * u.gates(self.step_flop) / c
        cheat = self.cheat if self.cheat is not None else (
            u.gates(2 * self.chord * self.injected() / self.chord_kappa) / c if self.sampled else 0.0)
        if cheat:
            out["exfil.faulty"] = min(1.0, cheat)
        return out

    def delta(self, ctx: Context) -> tuple[float, float]:
        return (self.p_evade if self.sampled else 0.0), 0.0


PER = ("server", "device")


@dataclass(frozen=True)
class Wipe:
    """The wipe every `wipe_s` seconds."""

    wipe_s: float = 86400.0
    #: `server` (`k_wipe` per server, of its whole memory `M_s`) or `device` (the sensitivity: `residual_bits` per
    #: device and `host_residual_bits` per host, the host's memory, DPU, BMC and flash unwiped)
    per: str = "server"
    #: `k_wipe`: bits a wipe leaves of one server's whole memory
    k_wipe: float = 0.0
    residual_bits: float = 0.0
    host_residual_bits: float = 0.0
    #: bits each server's boot disk keeps through a wipe that doesn't reach it (today's), on top of either form
    boot_disk_bits: float = 0.0
    #: one game's failure: the coins' `2⁻¹²⁸` and the pseudorandom function's advantage (the streamed wipe's retention game)
    game_delta_coin: float = 2.0**-128
    prf_advantage: float = 0.0

    #: the per-server bound against `M_s` is a draft; per device, the streamed wipe
    SERVER = ("draft:wipe-per-server",)
    DEVICE = ("Pous.Guarantees.StreamedWipeMeets2600GB",)

    def __post_init__(self) -> None:
        if self.per not in PER:
            raise ValueError(f"wipe per {self.per!r} is not one of {PER}")

    def cites(self) -> tuple[str, ...]:
        return self.SERVER + self.DEVICE if self.per == "server" else self.DEVICE

    def state(self, loc: Location) -> float:
        """`k_wipe` of a location's serving block: what the wipe leaves of its servers' memory, and their boot disks."""
        disks = loc.hosts * self.boot_disk_bits
        if self.per == "server":
            return loc.hosts * self.k_wipe + disks
        return loc.devices * self.residual_bits + loc.hosts * self.host_residual_bits + disks

    def games(self, ctx: Context) -> float:
        """Wipe games over the horizon: one per server (or device) per wipe."""
        loc = ctx.location
        return ctx.fleet.count * (loc.hosts if self.per == "server" else loc.devices) * ctx.horizon.seconds / self.wipe_s

    def delta(self, ctx: Context) -> tuple[float, float]:
        n = self.games(ctx)
        return n * self.game_delta_coin, n * self.prf_advantage


#: the warden form's public parameters the declared I/O bound reads (`Certifier.declared`)
DECLARED_IO = ("cell_bytes", "cell_rate", "lanes", "cells", "id_bits", "hold", "outputs", "bucket_s", "window_buckets")


def clog2(x: float) -> int:
    """`⌈log₂ x⌉` for `x ≥ 1`, as `NetworkCheck.Period.clog2`."""
    return (int(x) - 1).bit_length()


#: `%8.6f` of a number in [0, 10): 10^7 values (`ingress_scan.RATIO_BITS`)
RATIO_BITS = clog2(10 ** 7)
#: a name the scan checks only for form (`io_request._name_ok`) and doesn't store: a quote, up to 62 of the 93 printable
#: ASCII characters but `"` and `\`, a quote and spaces
NAME_BITS = clog2(sum(93 ** k for k in range(63)))
#: the bits of a warden-form id request besides its prompt ids that the ingress scan leaves free, each field's
#: (`verity/core/protocols/certifier/io_request.py`'s `layout` and checks; `requestBits` of the Lean statement)
REQUEST_HEAD = {
    "connection ordinal": 32,     # the cell header's u32, which `ingress_scan.step` checks only above the last
    "content-length": 32,         # a u32, free while the request is unfinished
    "max_tokens": 32,             # a u32
    "seed": 32,                   # a u32
    "temperature": RATIO_BITS,
    "top_p": RATIO_BITS,
    "model": NAME_BITS,
    "user": NAME_BITS,
}
#: a cell's header (kind u8, ordinal u32, length u32), a prompt id's field (`[` or `,`, then a u32 in 10 digits), and
#: the first prompt field's offset in a request's first cell (`io_request.layout(B).start`, the same at every `B`
#: whose first cell holds the head), bytes
CELL_HEADER, ID_FIELD, PROMPT_START = 9, 11, 367


@dataclass(frozen=True)
class Certifier:
    """A network certifier on each host's links, and the channels it doesn't read.

    With `declared`, a server's ingress and egress carry only its inference blocks' declared inputs and outputs, and an
    inference block runs on a single server (ruled). At the warden form's public parameters (the `WardenParams` of
    `benchmarks/network_traces/scan_zk.py`'s setups: `B` bytes a cell, `r` slots a lane, `lanes`, `C` cells a request,
    `IDB` bits an id, `D`, `T`) a request's declared input is at most

        requestBits = Σ REQUEST_HEAD + (K0 + (C − 1) K) IDB,   K0 = ⌊(B − 9 − 367) / 11⌋,   K = ⌊(B − 9) / 11⌋

    bits: the head's fields the ingress scan leaves free (`REQUEST_HEAD`), then `K0` prompt ids in its first cell after
    the head and `K` in each other cell (the grammar of `verity/core/protocols/certifier/io_request.py`, which
    `ingress_scan.py` scans). A bucket adds `lanes ⌈log₂(r + 2)⌉` bits, each lane's block absent or `d ≤ r` data cells
    then padding. A slot is held `D + 1` buckets from its admission, so a link-window of `T = window_buckets` buckets
    holds at most `lanes ⌈T / (D + 1)⌉` requests, and

        r_in ≤ windowBits / (T τ_b) = (lanes ⌈T / (D + 1)⌉ requestBits + T lanes ⌈log₂(r + 2)⌉) / (T τ_b)
        r_out ≤ lanes ⌈T / (D + 1)⌉ (KO IDB + ⌈log₂(KO + 1)⌉) / (T τ_b)

    (`requestBits`, `bucketBits`, `windowRequests` and `windowBits` of the Lean statement
    `Properties.NetworkCertifier.PerServerIngress`, `draft:per-server-ingress`; the egress at the same requests). The
    bound is stated per link-window, so the rate is too: this package never cites less than the bound states. This is
    its honest-δ form, at a δ where only the empty faulty set is possible. A faulty ingress unit adds
    `faulty_unit_bits` (`faultyIngressBits`), which this form excludes; it is 0 when every ingress unit is sampled. The
    bound is a draft until Lean proves it, so `declared_x` keeps the line rate as the ingress's upper end: the ingress
    is `declared_x · hosts · r_in`, at most the location's `links`. A cadence whose `r_in` is below the prompt rate of
    the serving credited at the location can't feed that serving, so `declared` is off by default until a deployment
    picks one that does."""

    #: the certified prompt ingress as a multiple of `α Φ` (`α = k/c`), the declared intake at full use: 1 when a
    #: server's ingress admits only declared inputs (without `declared`)
    intake: float = 1.0
    #: the certified prompt ingress in bit/s per location, given directly (None: `hosts · r_in` with `declared`, else
    #: `intake · α Φ`)
    ingress: float | None = None
    #: per host, bit/s: the certifier's timing channel, and side channels no certifier sees
    timing: float = 0.0
    side: float = 0.0
    #: each server's ingress at most its inference blocks' declared inputs, at the parameters below
    declared: bool = False
    #: `B`, bytes a cell; `r`, cells a lane a bucket (its `slots`, the warden's line rate); lanes a server; `C`, cells a
    #: request
    cell_bytes: float = math.nan
    cell_rate: float = math.nan
    lanes: float = math.nan
    cells: float = math.nan
    #: `IDB`, bits an id; `D`, a slot held `D + 1` buckets from its admission (the warden's `delay`); `KO`, outputs a
    #: request; `τ_b`, seconds a bucket
    id_bits: float = math.nan
    hold: float = math.nan
    outputs: float = math.nan
    bucket_s: float = math.nan
    #: `T`, buckets a link-window (the warden's `window`), the length the per-server bound is stated per
    window_buckets: float = math.nan
    #: the certified prompt ingress as a multiple of the declared bound, at most line rate (`Location.links`): 1 under
    #: the declared-ingress rule, and line rate the upper end until Lean proves the per-server bound
    declared_x: float = 1.0

    CITES = ("NetTiming.Guarantees.EncardIngressDecodableLeContents", "Accounting.Covered", "Accounting.TrustedUnits")
    INTAKE = ("draft:declared-ingress",)
    DECLARED = ("draft:declared-ingress", "draft:per-server-ingress")

    def __post_init__(self) -> None:
        unset = [k for k in DECLARED_IO if self.declared and not math.isfinite(getattr(self, k))]
        if unset:
            raise ValueError(f"the declared I/O bound needs {unset}")

    def cites(self) -> tuple[str, ...]:
        return self.CITES + (self.DECLARED if self.declared else self.INTAKE)

    @property
    def ids_per_cell(self) -> int:
        """`K = ⌊(B − 9) / 11⌋` prompt ids in a request's cell after its first (`idsPerCell`)."""
        return int(self.cell_bytes - CELL_HEADER) // ID_FIELD

    @property
    def first_cell_ids(self) -> int:
        """`K0 = ⌊(B − 9 − 367) / 11⌋` prompt ids in a request's first cell, after its head (`firstCellIds`)."""
        return int(self.cell_bytes - CELL_HEADER - PROMPT_START) // ID_FIELD

    @property
    def request_ids(self) -> int:
        """A request's most prompt ids, `K0 + (C − 1) K` (`requestIds`)."""
        return self.first_cell_ids + (int(self.cells) - 1) * self.ids_per_cell

    @property
    def request_in(self) -> float:
        """Bits a request's declared input holds: `Σ REQUEST_HEAD + (K0 + (C − 1) K) IDB` (`requestBits`)."""
        return sum(REQUEST_HEAD.values()) + self.request_ids * self.id_bits

    @property
    def bucket_in(self) -> int:
        """Bits a bucket adds to a server's ingress, each lane's block absent or `d ≤ r` data cells then padding:
        `lanes ⌈log₂(r + 2)⌉` (`bucketBits`)."""
        return int(self.lanes) * clog2(self.cell_rate + 2)

    @property
    def faulty_unit_bits(self) -> float:
        """Bits a faulty ingress unit adds, which the honest-δ form excludes and which are 0 when every ingress unit is
        sampled: its bucket's whole row, `8 B lanes r`, and two requests more a lane, its forged state's own and one
        admission its reset allows (`faultyIngressBits`)."""
        return 8 * self.cell_bytes * self.lanes * self.cell_rate + 2 * self.lanes * self.request_in

    @property
    def request_out(self) -> float:
        """Bits a request's declared output holds: `KO IDB + ⌈log₂(KO + 1)⌉`, its ids and their count."""
        return self.outputs * self.id_bits + math.ceil(math.log2(self.outputs + 1))

    def window_bits(self, buckets: float) -> tuple[float, float]:
        """`(ingress_bits, egress_bits)` of one server in a link-window of `buckets` buckets (`windowBits`), at the most
        requests it anchors, `windowRequests = lanes ⌈buckets / (D + 1)⌉`, and each bucket's `bucket_in` in. A period of
        `W` link-windows is `W` times it (`ingressBits`), not one ceiling over all its buckets."""
        n = self.lanes * math.ceil(buckets / (self.hold + 1))
        return n * self.request_in + buckets * self.bucket_in, n * self.request_out

    @property
    def r_in(self) -> float:
        """One server's declared ingress, bit/s: a link-window's `windowBits` over its `T τ_b` seconds."""
        return self.window_bits(self.window_buckets)[0] / (self.window_buckets * self.bucket_s)

    @property
    def r_out(self) -> float:
        """One server's declared egress, bit/s, at a link-window's requests."""
        return self.window_bits(self.window_buckets)[1] / (self.window_buckets * self.bucket_s)

    @property
    def warden_rate(self) -> float:
        """The warden's line rate a server, each way, bit/s: `lanes r 8B / τ_b`."""
        return self.lanes * self.cell_rate * 8 * self.cell_bytes / self.bucket_s

    def prompts(self, ctx: Context) -> float:
        """The certified prompt ingress into a location, bit/s."""
        loc = ctx.location
        if self.ingress is not None:
            return self.ingress
        if self.declared:
            return min(self.declared_x * loc.hosts * self.r_in, loc.links)
        return self.intake * ctx.policy.k / ctx.c * loc.phi

    def intake_of(self, ctx: Context) -> float:
        """The certified prompt ingress as a multiple of `α Φ`: `r_in / (α Φ_s)` with `declared`."""
        return self.prompts(ctx) / (ctx.policy.k / ctx.c * ctx.location.phi)

    def rate_in(self, ctx: Context) -> float:
        """`r_in + r_side`: bit/s of unregistered input into a location."""
        return self.prompts(ctx) + ctx.location.hosts * (self.timing + self.side)

    def delta(self, ctx: Context) -> tuple[float, float]:
        return 0.0, 0.0


class Regime(Protocol):
    name: str

    def partition(self, ctx: Context) -> Partition: ...

    def delta(self, ctx: Context) -> Delta: ...


def _dark(ctx: Context) -> dict[str, float]:
    return {"eps_dark": ctx.fleet.eps_dark, "dark_mfu": ctx.fleet.dark_mfu}


@dataclass(frozen=True)
class Unmonitored:
    """No protocol: every serving block is unconstrained (`g = 1`, overhead `μ`)."""

    name: str

    def partition(self, ctx: Context) -> Partition:
        f, loc, h = ctx.fleet, ctx.location, ctx.horizon
        return Partition((ServingBlock("server", loc.phi * h.seconds, f.count, (), mfu=f.mfu_train),), **_dark(ctx))

    def delta(self, ctx: Context) -> Delta:
        return Delta({})


@dataclass(frozen=True)
class Remote:
    """The remote audit: PoUW, detection, and the NCI check's registration. No certifier sees a server's links, so its
    serving block is a server over the horizon, and the inference cut is the only one."""

    name: str
    pouw: Pouw
    detection: Detection

    #: the inference cut: the remote composition, `Nci.Guarantees.DampenerRemote`, term for term, its envelopes'
    #: premises, and the registration
    CITES = ("Nci.Guarantees.DampenerRemote",) + Pouw.CITES + Detection.CITES + ("Nci.Assumptions.BaselineRegistered",)
    #: the routes a strategy that never cheats gets: no wrong certificate, faulty block or detection
    ABIDING = ("undecl.host", "undecl.gpu", "undecl.clock", "skip.pouw")

    def envelopes(self) -> dict[str, Any]:
        return {"pouw": self.pouw, "detection": self.detection}

    def slack(self, ctx: Context) -> dict[str, float]:
        """Shares of a location's capacity no inference block constrains, by route (`DampenerRemote`'s terms)."""
        return self.pouw.slack(ctx.location) | self.detection.slack(ctx)

    def uncertified(self, ctx: Context) -> float:
        """The share of capacity `DampenerRemote` leaves uncertified: `r̂ + ε_W W_cert/C + ε' + JΔ/H + θ`, and `χ`."""
        return math.fsum(self.slack(ctx).values())

    def blocks(self, ctx: Context, gates: float) -> Blocks:
        """The inference blocks in `gates` of a location: the certified work that ran, `(1 − ε_W) W_cert`, in blocks of
        `c`."""
        n = (1 - self.pouw.audited(ctx.location)) * (1 - self.pouw.eps_w) * gates / ctx.c
        return Blocks("inference", Budget(ctx.c, ctx.policy.k), n)

    def inference(self, ctx: Context, gates: float) -> Cut:
        """`n_S q_T(c, k) + q_U(U_S)`: the inference blocks and the uncertified gates by route."""
        return Cut("inference", (self.blocks(ctx, gates),), {k: v * gates for k, v in self.slack(ctx).items()},
                   self.CITES)

    def abiding(self, ctx: Context, gates: float) -> dict[str, float]:
        """The uncertified gates a strategy that never cheats trains with in `gates` of a location, at the training
        MFU."""
        used = gates * ctx.fleet.mfu_train
        return {k: v * used for k, v in self.slack(ctx).items() if k in self.ABIDING}

    def delta(self, ctx: Context) -> Delta:
        return Delta({name: e.delta(ctx) for name, e in self.envelopes().items()})

    def partition(self, ctx: Context) -> Partition:
        gates = ctx.location.phi * ctx.horizon.seconds
        layout = Cut("layout", (self.blocks(ctx, gates),), self.abiding(ctx, gates), self.CITES)
        block = ServingBlock("server", gates, ctx.fleet.count, (self.inference(ctx, gates),), layout,
                             ctx.fleet.mfu_train)
        return Partition((block,), **_dark(ctx))


FORMS = ("certified", "links")


@dataclass(frozen=True)
class Onsite(Remote):
    """The remote audit's envelopes plus the wipe and the certifier: each serving block, one server between two wipes,
    gets the serving cut `q_T(c_S, k_S)` as well, and contributes the lesser."""

    wipe: Wipe = Wipe()
    certifier: Certifier = Certifier()
    #: `certified` (`k_S = k_wipe + (r_in + r_side) Δ_wipe`) or `links` (`Windows`: the location's memory plus its
    #: line rate)
    form: str = "certified"

    #: `Windows` as proved: memory and line rate, on the physical annotation's four constraints
    LINKS = ("Accounting.Guarantees.Actual.Windows", "Accounting.Placed.Causal", "Accounting.Placed.ComputeRate",
             "Accounting.Placed.Memory", "Accounting.Placed.Links")
    #: `WindowsAt` at the wipes, its state term from the wipe and its link term from the certifier's ingress, which
    #: bounds information where `WindowsAt` reads width at line rate: the information form is a draft
    SERVING = ("Accounting.Guarantees.Actual.WindowsAt", "Accounting.Placed.Causal", "Accounting.Placed.ComputeRate",
               "Accounting.Placed.Links", "draft:windows-information-form")
    PHYSICAL = tuple(dict.fromkeys(LINKS + SERVING + Wipe.SERVER + Wipe.DEVICE + Certifier.CITES + Certifier.INTAKE
                                   + Certifier.DECLARED))

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ValueError(f"regime {self.name}: form {self.form!r} is not one of {FORMS}")

    def envelopes(self) -> dict[str, Any]:
        return super().envelopes() | {"wipe": self.wipe, "certifier": self.certifier}

    def cites(self) -> tuple[str, ...]:
        """The serving cut's citations."""
        if self.form == "links":
            return self.LINKS
        return tuple(dict.fromkeys(self.SERVING + self.wipe.cites() + self.certifier.cites()))

    def state(self, ctx: Context) -> float:
        """Bits a serving block carries in: all its servers' memory `M_s` (`links`), or what a wipe leaves
        (`certified`)."""
        loc = ctx.location
        return loc.hosts * loc.server_memory if self.form == "links" else self.wipe.state(loc)

    def rate_in(self, ctx: Context) -> float:
        """Bit/s into a serving block: its line rate (`links`), or `r_in + r_side` (`certified`)."""
        return ctx.location.links if self.form == "links" else self.certifier.rate_in(ctx)

    def budget(self, ctx: Context) -> Budget:
        """`(c_S, k_S)`: `⌊Φ_s Δ_wipe⌋` gates and `k_wipe + (r_in + r_side) Δ_wipe` bits."""
        w = self.wipe.wipe_s
        return Budget(float(math.floor(ctx.location.phi * w)), self.state(ctx) + self.rate_in(ctx) * w)

    def partition(self, ctx: Context) -> Partition:
        budget = self.budget(ctx)
        serving = Cut("serving", (Blocks("serving", budget, 1.0),), {}, self.cites())
        # The uncertified block may take the whole serving block's input: generous to the adversary, since the
        # inference blocks' prompts come out of it too.
        uncertified = Budget(math.fsum(self.abiding(ctx, budget.gates).values()), budget.inputs)
        layout = Cut("layout", (self.blocks(ctx, budget.gates), Blocks("uncertified", uncertified, 1.0)), {},
                     self.cites())
        count = ctx.fleet.count * ctx.horizon.seconds / self.wipe.wipe_s
        block = ServingBlock("serving-block", budget.gates, count, (self.inference(ctx, budget.gates), serving),
                             layout, ctx.fleet.mfu_train)
        return Partition((block,), **_dark(ctx))


KINDS: Mapping[str, type] = {"unmonitored": Unmonitored, "remote": Remote, "onsite": Onsite}
#: each envelope a regime holds, by its field name
PARTS: Mapping[str, type] = {"pouw": Pouw, "detection": Detection, "wipe": Wipe, "certifier": Certifier}


def policy(registry: Registry, name: str) -> Policy:
    return registry.fields(Policy, f"policy.{name}")


def table(registry: Registry, name: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Regime `name`'s table over its `base` chain: its own keys over its base's."""
    if name in seen:
        raise ValueError(f"regime bases loop: {' -> '.join(seen + (name,))}")
    own = registry.table(f"regime.{name}")
    base = own.pop("base", None)
    return (table(registry, base, seen + (name,)) if base else {}) | own


def regime(registry: Registry, name: str) -> Regime:
    """Regime `name`, each key of its flat table given to the envelope (or the regime) whose field it is."""
    values = table(registry, name)
    cls = KINDS[values.pop("kind")]
    names = {f.name for f in fields(cls)}
    args: dict[str, Any] = {"name": name}
    used = {"name"}
    for part in (p for p in PARTS if p in names):
        own = {f.name for f in fields(PARTS[part])}
        args[part] = PARTS[part](**{k: v for k, v in values.items() if k in own})
        used |= own
    args |= {k: v for k, v in values.items() if k in names - set(PARTS)}
    stray = set(values) - used - (names - set(PARTS))
    if stray:
        raise ValueError(f"regime.{name}: {sorted(stray)} are not fields of {cls.__name__} or its envelopes")
    try:
        return cls(**args)
    except ValueError as e:
        raise ValueError(f"regime {name}: {e}") from None
