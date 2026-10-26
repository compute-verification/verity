"""Monitoring regimes: how the actual datacenter maps onto the theorem's blocks.

The NCI verifier of record sees only the *declared* circuit: blocks of `Γ` FLOPs reading at most `k` novel bits (the
`Policy`). What the fleet actually computes is more: work the audits don't certify (PoUW's residual and deficit, the
certificate slack, work before a verdict lands, work a sampled proof let go undone) and whatever runs between and
around the declared blocks. A regime says, for one location over one window (a **region**), which cuts of it the
protocols make valid, and cites what makes each valid:

* **declared** (remote and onsite): the certified share of the region runs declared blocks `(Γ, k)`, and every other
  FLOP is free at face value. Valid under the remote composition (`DampenerRemote`: PoUW, terminal and sampled
  detection, the certificate slack and the verdict lag) and the verifier's enforcement of the policy.
* **physical** (onsite only): the whole region is one block, its budget `(Φ Δτ, R + I_cert Δτ + β_side Δτ)`: what a
  wipe leaves, what the certifier admits and what side channels carry over the window. Valid by causality on the actual
  circuit's annotation (`WindowsAt` on `Accounting.Placed`: `width_into_region_le`, `Links`, `Covered`; ASK-141).

A region's *layout* is what an attack gets: remotely, the declared blocks plus free FLOPs at face value (nothing bounds
what enters a location); onsite, the declared blocks plus the uncertified time as one block with the window's input.

Adding a regime is one dataclass with `partition` and an entry in `KINDS`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

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


class Regime(Protocol):
    name: str

    def partition(self, ctx: Context) -> Partition: ...


@dataclass(frozen=True)
class Unmonitored:
    """No protocol: every region is free (`g = 1`, `ρ = m`)."""

    name: str

    def partition(self, ctx: Context) -> Partition:
        f, loc, h = ctx.fleet, ctx.location, ctx.horizon
        return Partition((Region("location", loc.phi * h.seconds * f.mfu_train, f.count, ()),), f.dark)


@dataclass(frozen=True)
class Remote:
    """The remote audit: PoUW certifies declared work, sampled proofs check it, and NCI's verifier enforces the policy.
    No certifier sees a location's links, so the declared cut is the only one."""

    name: str
    #: `r̂`: the share of location time the audits leave uncertified
    residual: float
    #: `ε_W`: PoUW's deficit on certified work
    eps_w: float
    #: `ε'`: the aggregation and certificate slack
    eps_agg: float
    #: `J`, `Δ`: identities and the verdict lag in seconds (work before a verdict lands is free: `JΔ/H`)
    identities: float
    lag_s: float
    #: `χ`: the share of claimed declared work the sampled proofs let go undone
    cheat: float

    CITES = ("Nci.Guarantees.DampenerRemote", "Nci.Assumptions.PouwSound", "Nci.Assumptions.TerminalDetection",
             "Nci.Assumptions.SampledDetection", "Nci.Assumptions.BaselineRegistered")

    def free(self, horizon: float) -> dict[str, float]:
        """Shares of a region's time that are free, by source, over a horizon of `horizon` seconds."""
        r = self.residual
        return {"residual": r, "pouw": self.eps_w * (1 - r), "slack": self.eps_agg,
                "lag": self.identities * self.lag_s / horizon, "cheat": self.cheat}

    def declared(self, ctx: Context, seconds: float) -> Cut:
        f, loc, pol = ctx.fleet, ctx.location, ctx.policy
        free = self.free(ctx.horizon.seconds)
        certified = max(0.0, 1 - sum(free.values()))
        n = certified * loc.phi * seconds * f.mfu_serve / pol.gamma
        work = loc.phi * seconds * f.mfu_train
        return Cut("declared", (Blocks("declared", Budget(pol.gamma, pol.k), n),),
                   {k: v * work for k, v in free.items()}, self.CITES)

    def partition(self, ctx: Context) -> Partition:
        s = ctx.horizon.seconds
        work = ctx.location.phi * s * ctx.fleet.mfu_train
        declared = self.declared(ctx, s)
        return Partition((Region("location", work, ctx.fleet.count, (declared,), declared),), ctx.fleet.dark)


@dataclass(frozen=True)
class Onsite(Remote):
    """The remote audit's mechanisms, plus memory wipes every `wipe_s` seconds that leave at most `residual_bits` per
    device, a network certifier admitting `ingress` bit/s into each location, and side channels of `side` bit/s: each
    location-window gets the physical cut as well, and contributes the lesser."""

    wipe_s: float = 86400.0
    residual_bits: float = 0.0
    ingress: float = 0.0
    side: float = 0.0

    PHYSICAL = ("Accounting.Guarantees.Actual.WindowsAt",)

    def budget(self, ctx: Context) -> Budget:
        loc, w = ctx.location, self.wipe_s
        return Budget(loc.phi * w * ctx.fleet.mfu_train, loc.devices * self.residual_bits + (self.ingress + self.side) * w)

    def partition(self, ctx: Context) -> Partition:
        w = self.wipe_s
        budget = self.budget(ctx)
        declared = self.declared(ctx, w)
        physical = Cut("physical", (Blocks("physical", budget, 1.0),), {}, self.PHYSICAL)
        # A strategy runs the declared blocks as declared and the uncertified time as one block, which may take the
        # whole window's novel input (generous to the adversary: the declared blocks' prompts come out of it too).
        uncertified = Budget(sum(declared.free.values()), budget.inputs)
        layout = Cut("layout", declared.blocks + (Blocks("uncertified", uncertified, 1.0),), {}, self.PHYSICAL)
        windows = ctx.fleet.count * ctx.horizon.seconds / w
        region = Region("location-window", ctx.location.phi * w * ctx.fleet.mfu_train, windows,
                        (declared, physical), layout)
        return Partition((region,), ctx.fleet.dark)


KINDS: Mapping[str, type] = {"unmonitored": Unmonitored, "remote": Remote, "onsite": Onsite}


def policy(registry: Registry, name: str) -> Policy:
    return registry.fields(Policy, f"policy.{name}")


def regime(registry: Registry, name: str) -> Regime:
    prefix = f"regime.{name}"
    return registry.fields(KINDS[registry[f"{prefix}.kind"]], prefix)
