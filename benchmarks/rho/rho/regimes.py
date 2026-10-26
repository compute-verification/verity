"""Monitoring regimes: how the actual datacenter maps onto the theorem's blocks.

The NCI verifier of record sees only the *declared* circuit: blocks of `Γ` FLOPs reading at most `k` novel bits (the
`Policy`). What the fleet actually computes is more: work the audits don't certify and whatever runs between and around
the declared blocks. A regime says, for one location over one window (a **region**), which cuts of it the protocols make
valid, and cites what makes each valid:

* **declared** (remote and onsite): the remote composition, `Nci.Guarantees.DampenerRemote`, term for term. The
  certified work `(1 − ε_W) W_cert` runs declared blocks `(Γ, k)`, and every other FLOP of the location's capacity is
  free at face value: the audited residual `r̂` (the GPUs' uncredited time, their clock above the credited one, the host
  CPUs at peak), PoUW's deficit `ε_W W_cert/C`, the certificate slack `ε'`, the verdict lag `JΔ/H`, the escape slack `θ`
  and the cheating share `χ` (with sampled proofs). The residual is a share of peak capacity, as Lean counts it
  (`M_r ≤ κ (C − (1 − ε_W) W_cert + …)/2`), so its free FLOPs are `share · Φ Δτ`: `g`'s free share is the Lean
  dampener's `share / MFU₀`. `benchmarks/nci/tiers.py` computes the same terms (`tests/test_regimes.py` holds them to it).
* **physical** (onsite only): the whole location-window is one block, its budget `(Φ ω, K_ω)`, by causality on the
  actual circuit's annotation (`WindowsAt` and `Windows` on `Accounting.Placed`; ASK-141). Its form says what `K_ω` is:
  `links`, what Lean proves today (`Windows`: the location's memory plus its line rate `β_X` over the window), or
  `certified`, what a wipe leaves plus what the certifier admits and the side channels carry, which needs `WindowsAt`'s
  information form (its reads are bounded by width at line rate, the certifier and the wipes bound information).

A region's *layout* is what a protocol-abiding attack gets: the declared blocks, and the location's time outside the
certified work that a strategy which never cheats can use (`ABIDING`), at the training MFU; remotely as free FLOPs
(nothing bounds what enters a location), onsite as one block with the window's input.

Adding a regime is one dataclass with `partition` and an entry in `KINDS`. A regime table may name a `base` table and
override only what differs (`regime`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

from rho.hardware import Fleet, Horizon, Location
from rho.params import Registry
from rho.theorem import Blocks, Budget, Cut, Partition, Region


@dataclass(frozen=True)
class Policy:
    """The NCI policy on declared work: blocks of `gamma` FLOPs, each reading at most `k` novel bits."""

    name: str
    gamma: float
    k: float


@dataclass(frozen=True)
class Context:
    fleet: Fleet
    location: Location
    policy: Policy
    horizon: Horizon

    @property
    def capacity(self) -> float:
        """`C`: the monitored fleet's peak FLOPs over the horizon."""
        return self.fleet.count * self.location.phi * self.horizon.seconds


class Regime(Protocol):
    name: str

    def partition(self, ctx: Context) -> Partition: ...


@dataclass(frozen=True)
class Unmonitored:
    """No protocol: every region is free (`g = 1`, `ρ = m`)."""

    name: str

    def partition(self, ctx: Context) -> Partition:
        f, loc, h = ctx.fleet, ctx.location, ctx.horizon
        return Partition((Region("location", loc.phi * h.seconds * f.mfu_train, f.count, ()),), f.dark_effective)


SAMPLING = ("proof_share", "overhead", "replay", "step_flop", "p_evade", "luck", "step_bits", "floor_steps", "chord",
            "chord_kappa")


@dataclass(frozen=True)
class Remote:
    """The remote audit: PoUW certifies declared work, sampled proofs check it, and NCI's verifier enforces the policy.
    No certifier sees a location's links, so the declared cut is the only one. Each field is `tiers.py`'s parameter of
    the same name (`proof_share` is its `share`)."""

    name: str
    #: the GPUs' share of capacity outside the floor credit (`tiers` `residual`)
    residual: float
    #: `ε_W`: PoUW's deficit, a share of certified work credited without its running
    eps_w: float
    #: `ε'`: the share of capacity whose certificates may be wrong (Markov over the windows)
    eps_agg: float
    #: `J`: identities a detection doesn't end; `Δ`: the verdict lag in seconds
    identities: float
    lag_s: float
    #: the GPUs' capacity above the clock PoUW credits, as a share
    clock_share: float = 0.0
    #: the host CPUs counted at `host` times their peak (`Location.host_ops`): 1 until something credits or bounds them
    host: float = 0.0
    #: whether sampled proofs check the declared work (tiers T5); off, every unit is proved (`q = 1`, `χ = θ = 0`)
    sampled: bool = False
    #: the sampled proofs' compute share, proving cost per unit of step work, replay cost per drawn step, work and
    #: output bits per step, failure probability `δ`, luck factor, and steps drawn at a floor of one
    proof_share: float = math.nan
    overhead: float = math.nan
    replay: float = math.nan
    step_flop: float = math.nan
    p_evade: float = math.nan
    luck: float = math.nan
    step_bits: float = math.nan
    floor_steps: float = math.nan
    #: counted multiplications per injected bit (`tiers.chord` at the injected bits), at `chord_kappa` products per
    #: FLOP: what turns `χ`'s products into the free FLOPs a cut carries
    chord: float = math.nan
    chord_kappa: float = math.nan
    #: `χ` given directly, in place of the sampled proofs' closed form (None: the closed form)
    cheat: float | None = None

    CITES = ("Nci.Guarantees.DampenerRemote", "Nci.Guarantees.CertificateSlack", "Nci.Guarantees.VerdictLag",
             "Nci.Assumptions.PouwSound", "Nci.Assumptions.TerminalDetection", "Nci.Assumptions.SampledDetection",
             "Nci.Assumptions.BaselineRegistered")
    #: the free sources a strategy that never cheats gets: no wrong certificate, faulty block or detection
    ABIDING = ("residual", "clock", "host", "pouw")

    def __post_init__(self) -> None:
        unset = [k for k in SAMPLING if self.sampled and not math.isfinite(getattr(self, k))]
        if unset:
            raise ValueError(f"regime {self.name}: sampled proofs need {unset}")

    def audited(self, loc: Location) -> float:
        """`r̂ = 1 − W_cert/C`: the GPUs' uncredited share, their clock above the credited one, and the host CPUs."""
        return min(1.0, self.residual + self.clock_share + self.host * loc.host_ops / loc.phi)

    def q(self) -> float:
        """The detection probability per faulty block: the proof share over the cost of proving a unit."""
        return self.proof_share / (self.overhead + self.replay) if self.sampled else 1.0

    def injected(self) -> float:
        """Bits the accepted faulty proof blocks inject: `J ln(1/δ) (1/q + floor) · step_bits`."""
        if not self.sampled:
            return 0.0
        return self.identities * math.log(1 / self.p_evade) * (1 / self.q() + self.floor_steps) * self.step_bits

    def free(self, ctx: Context) -> dict[str, float]:
        """Shares of the location's peak capacity that are free, by source (`DampenerRemote`'s terms)."""
        loc, c = ctx.location, ctx.capacity
        r = self.audited(loc)
        out = {"residual": self.residual, "clock": self.clock_share, "host": self.host * loc.host_ops / loc.phi,
               "pouw": self.eps_w * (1 - r), "slack": self.eps_agg,
               "lag": min(1.0, self.identities * self.lag_s / ctx.horizon.seconds)}
        if self.sampled:
            steps = self.luck * self.identities * math.log(1 / self.p_evade) / self.q()
            out["escape"] = steps * self.step_flop / c
        cheat = self.cheat if self.cheat is not None else (
            2 * self.chord * self.injected() / (self.chord_kappa * c) if self.sampled else 0.0)
        if cheat:
            out["cheat"] = min(1.0, cheat)
        return out

    def uncertified(self, ctx: Context) -> float:
        """The share of peak capacity `DampenerRemote` leaves free: `r̂ + ε_W W_cert/C + ε' + JΔ/H + θ`, and `χ`."""
        return math.fsum(self.free(ctx).values())

    def blocks(self, ctx: Context, seconds: float) -> Blocks:
        """The declared blocks run over `seconds`: the certified work that ran, `(1 − ε_W) W_cert`, in blocks of `Γ`."""
        loc, pol = ctx.location, ctx.policy
        n = (1 - self.audited(loc)) * (1 - self.eps_w) * loc.phi * seconds / pol.gamma
        return Blocks("declared", Budget(pol.gamma, pol.k), n)

    def declared(self, ctx: Context, seconds: float) -> Cut:
        peak = ctx.location.phi * seconds
        return Cut("declared", (self.blocks(ctx, seconds),), {k: v * peak for k, v in self.free(ctx).items()},
                   self.CITES)

    def abiding(self, ctx: Context, seconds: float) -> dict[str, float]:
        """The free FLOPs a strategy that never cheats trains with over `seconds`, at the training MFU."""
        work = ctx.location.phi * seconds * ctx.fleet.mfu_train
        return {k: v * work for k, v in self.free(ctx).items() if k in self.ABIDING}

    def partition(self, ctx: Context) -> Partition:
        s = ctx.horizon.seconds
        layout = Cut("layout", (self.blocks(ctx, s),), self.abiding(ctx, s), self.CITES)
        region = Region("location", ctx.location.phi * s * ctx.fleet.mfu_train, ctx.fleet.count,
                        (self.declared(ctx, s),), layout)
        return Partition((region,), ctx.fleet.dark_effective)


FORMS = ("certified", "links")


@dataclass(frozen=True)
class Onsite(Remote):
    """The remote audit's mechanisms, plus memory wipes every `wipe_s` seconds, a network certifier on each host's
    links and side channels: each location-window gets the physical cut as well, and contributes the lesser."""

    wipe_s: float = 86400.0
    #: what a wipe leaves per device, and per host (host memory the fill doesn't reach, DPU, BMC and flash), bits
    residual_bits: float = 0.0
    host_residual_bits: float = 0.0
    #: the certified prompt ingress as a multiple of `α Φ` (`α = k/Γ`), the declared intake at full use: 1 once
    #: NET-35's count check holds it there
    intake: float = 1.0
    #: the certified prompt ingress in bit/s per location, given directly (None: `intake · α Φ`)
    ingress: float | None = None
    #: per host, bit/s: the certifier's timing channel, and side channels no certifier sees
    timing: float = 0.0
    side: float = 0.0
    #: `certified` (the wipe residual plus certified ingress, timing and side channels) or `links` (`Windows`: the
    #: location's memory plus its line rate)
    form: str = "certified"

    #: `Windows` as proved: memory and line rate, on the physical annotation's four constraints
    LINKS = ("Accounting.Guarantees.Actual.Windows", "Accounting.Placed.Causal", "Accounting.Placed.ComputeRate",
             "Accounting.Placed.Memory", "Accounting.Placed.Links")
    #: `WindowsAt` at the wipes, its state term from the streamed wipe and its link term from the certifier's ingress,
    #: which bounds information where `WindowsAt` reads width at line rate: the information form is a draft
    CERTIFIED = ("Accounting.Guarantees.Actual.WindowsAt", "Accounting.Placed.Causal", "Accounting.Placed.ComputeRate",
                 "Accounting.Placed.Links", "Pous.Guarantees.StreamedWipeMeets2600GB",
                 "Accounting.Guarantees.Actual.WipeReadHeld", "Accounting.Guarantees.Actual.WipeLag",
                 "NetTiming.Guarantees.EncardIngressDecodableLeContents", "Accounting.Covered", "Accounting.TrustedUnits",
                 "draft:windows-information-form", "draft:NET-35")
    PHYSICAL = tuple(dict.fromkeys(LINKS + CERTIFIED))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.form not in FORMS:
            raise ValueError(f"regime {self.name}: form {self.form!r} is not one of {FORMS}")

    def cites(self) -> tuple[str, ...]:
        return self.LINKS if self.form == "links" else self.CERTIFIED

    def state(self, ctx: Context) -> float:
        """Bits a location carries into a window: all its memory (`links`), or what a wipe leaves (`certified`)."""
        loc = ctx.location
        if self.form == "links":
            return loc.memory + loc.host_memory
        return loc.devices * self.residual_bits + loc.hosts * self.host_residual_bits

    def rate_in(self, ctx: Context) -> float:
        """Bit/s into a location: its line rate (`links`), or the certified ingress, timing and side channels."""
        loc, pol = ctx.location, ctx.policy
        if self.form == "links":
            return loc.links
        prompts = self.ingress if self.ingress is not None else self.intake * pol.k / pol.gamma * loc.phi
        return prompts + loc.hosts * (self.timing + self.side)

    def budget(self, ctx: Context) -> Budget:
        w = self.wipe_s
        return Budget(ctx.location.phi * w, self.state(ctx) + self.rate_in(ctx) * w)

    def partition(self, ctx: Context) -> Partition:
        w = self.wipe_s
        budget = self.budget(ctx)
        declared = self.declared(ctx, w)
        physical = Cut("physical", (Blocks("physical", budget, 1.0),), {}, self.cites())
        # The uncertified block may take the whole window's input: generous to the adversary, since the declared
        # blocks' prompts come out of it too.
        uncertified = Budget(math.fsum(self.abiding(ctx, w).values()), budget.inputs)
        layout = Cut("layout", (self.blocks(ctx, w), Blocks("uncertified", uncertified, 1.0)), {}, self.cites())
        windows = ctx.fleet.count * ctx.horizon.seconds / w
        region = Region("location-window", ctx.location.phi * w * ctx.fleet.mfu_train, windows,
                        (declared, physical), layout)
        return Partition((region,), ctx.fleet.dark_effective)


KINDS: Mapping[str, type] = {"unmonitored": Unmonitored, "remote": Remote, "onsite": Onsite}


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
    values = table(registry, name)
    cls = KINDS[values.pop("kind")]
    names = {f.name for f in fields(cls)}
    stray = set(values) - names - {"name"}
    if stray:
        raise ValueError(f"regime.{name}: {sorted(stray)} are not fields of {cls.__name__}")
    return cls(**({"name": name} | values))
