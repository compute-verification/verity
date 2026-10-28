"""The physical layer's reading of the claims: credit, headroom, where things were, and the covert budget, per block.

`account` reads only public things: the claims (logical facts about the prover system) and the profile (the
placement's public properties, the declaration with its named assumptions, the verifier's plan). For each block it
concludes:

- `credited`: the declared size in ANDs of the Program's part the claims prove ran there (`Profile.ands`), with its
  multiply-adds by format beside it, never credited (`macs`);
- `busy` and `headroom`: the share of the epoch that work needs on the block's devices, each kind at their declared
  rate for it (multiply-adds at a precision, word operations, Boolean gates), and the share the audits leave
  unaccounted;
- `outsourceable` and `local_at_least`: the work a helper off the block could have done for it, and so the credited
  work that must have run on the block. Crossing bytes bound nothing by themselves: a helper holding the registered
  weights needs only small inputs and outputs. Work is pinned to the block only by data it must consume that only the
  block receives, a trusted device's fresh data (a run input placed on a trusted device), which a helper could get only through the
  block's certified egress: outsourceable ≤ egress × work per forced bit. Or by the named premise `FLEET_CLOSED`. Or,
  under the premise `SALT_CUSTODY`, by a salt the block's own certifier releases only into it: the work that depends
  on it is pinned while `custody`, k, the bits that could carry the salt out before registration (the block's egress
  timing advice, the profile's advice, inside inputs and salt-dependent values crossing out), stays at most 384. With
  none, it is unbounded and nothing is attributed;
- `reads` and `reads_local_at_least`: the reads the bandwidth audits certify for stores on the block's devices, and the
  part no path into the registered device (a boundary link, or host DRAM over PCIe) can have carried in;
- `carried`: the bits that could have been carried into the epoch: each device's memory, less what a wipe's labels
  filled (at most its `fill_rate` over the wipe), and never less than its residual. For a block out for maintenance,
  only a wipe after it counts.

For the whole datacenter it gives where each held store could have been (`located`: the devices that could have
answered its sample by the deadline, and none when the prover could rebuild the sample in time), the covert budget (the profile's
advice, timing advice, inputs, every refused port-window charged at its link's full capacity, so it fails closed, and
everything a block out for maintenance held), and the failure probability (the sum of the claims' deltas). Beside the
covert budget, which is only what goes beyond the committed computation, it reports the export (λ_out, `export_bits`):
the committed bits that crossed a certified boundary out of a block, by port. They count as novel information read by
their receivers once the declaration's certifiers cover every link; until then the export is reported
as not applicable, with the uncovered links named.

Every claim's credit is also a `Credit` on a unit: compute in block-ticks, PoUW's proved bound (`Profile.ticks`, (1 − ε)
of the floor program's block-ticks, which no accepted window undercuts over any route) as `compute` on the whole block
where the scheme gives one, else the honest routing's
time on each facet (kinds on one facet add up, kinds on different facets run at once) as `compute:<facet>`, on the block as far as attribution places it there and graded by its anchor,
the rest on the whole datacenter, since PoUW credits the prover system; a store's device the bits read from it and the
bit-ticks it held over the interval the claim proves (a snapshot proves an instant, and carries none). `capacity` gives each unit's resources a tick by the
declaration, and an account whose credits exceed them anywhere (`overcredited`, by `double_credit`, Horn's interval
test) is unsound. A `Ledger` joins accounts of successive epochs or of overlapping audits, and applies the same rule
across them (`joint`): one device's time, bandwidth and memory are credited once. Its `custody` is the cumulative check
on the registered weights: the share of them that the epochs' covert budgets, charged exports and a side
channel of σ a server (`DEDICATED_RACKS`) could have carried out.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .binding import Profile
from .claims import Carried, Claim, Computed, Entered, Held, Read, Refused, Scheduled
from .memory import answer_time, fill_rate, into, regen_time
from .physical import Assumption, Datacenter, Port
from .time import Interval

#: the premise that no undeclared compute is reachable from a block
FLEET_CLOSED = Assumption("fleet-closed", 1, "premise", "declared")
#: salt custody (unproved): a salt the block's certifier releases only into it pins the work that depends on it, while
#: k, the bits that could carry it out before registration, stays at most this many
SALT_CUSTODY = Assumption("salt-custody", 384, "bits", "declared")
#: a block's proof blocks counted under the served query (Q_word v1 at X = 16) because no one-stage registration named the
#: Program's partition
SERVED_PARTITION = Assumption("served-partition", 1, "premise", "declared")
#: σ, the side channel from each server to machines off its block, under the premise that no untrusted
#: machine shares a block's rack
DEDICATED_RACKS = Assumption("dedicated-racks", 1e3, "bit/s per server", "declared")


def facet(kind: str) -> str:
    """The part of a device a kind of work runs on: a kind is `<facet>:<detail>`, so `mac:fp8` and `mac:bf16` share
    the tensor cores and add up, as `cuda:op` and `cuda:gate` share the CUDA cores, while kinds on different facets (the
    CUDA cores' roundings beside the tensor cores' products) run at once."""
    return kind.split(":")[0]


@dataclass
class BlockAccount:
    block: str
    credited: float
    #: the share of the epoch the credited work needs on the block's devices: each facet's kinds at their declared rates
    #: added up, and the busiest facet's share (`loads` has each)
    busy: float
    outsourceable: float
    carried: float
    reads: float = 0.0
    reads_local_at_least: float = 0.0
    forced: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    #: for a block with a salt in its certifier's custody, k: the bits that could carry it out before registration
    custody: float | None = None
    #: each facet's share of the epoch
    loads: dict[str, float] = field(default_factory=dict)
    #: what places the work on the block: "fleet-closed", "salt-custody" or "forced" (the tightest bound), or None
    anchor: str | None = None
    #: its multiply-adds by format (`mac:bf16`, ...), never credited
    macs: dict[str, float] = field(default_factory=dict)

    @property
    def headroom(self) -> float:
        """The share of the epoch the audits leave unaccounted."""
        return max(1.0 - self.busy, 0.0)

    @property
    def local_at_least(self) -> float:
        """Work that must have run on the block itself: the credited work less what a helper could have done."""
        return max(self.credited - self.outsourceable, 0.0)


@dataclass
class Account:
    blocks: list[BlockAccount]
    timing_bits: float
    input_bits: dict[str, int]
    refused: list[Refused]
    located: dict[str, list[str]] = field(default_factory=dict)
    advice_bits: float = 0.0
    refused_bits: float = 0.0
    delta: float = 0.0
    #: each block out for maintenance: the bits its devices hold, which may leave with whoever services it
    serviced: dict[str, float] = field(default_factory=dict)
    #: what the claims credit each unit, by resource and interval, and what each unit can give a tick (`capacity`)
    credits: list[Credit] = field(default_factory=list)
    capacity: dict[tuple[str, str], float] = field(default_factory=dict)
    #: the committed bits each egress port carried out of its block (`Profile.exported`)
    exported: dict[Port, float] = field(default_factory=dict)
    #: the links the declaration leaves uncovered (`Datacenter.uncovered_links`), by their ends
    uncovered: list[str] = field(default_factory=list)
    #: the epoch's real duration, in seconds, and its servers, counted as the blocks' GPU devices: what the side-channel
    #: premise multiplies (`Ledger.custody`)
    seconds: float = 0.0
    servers: int = 0
    #: why nothing is credited (`Profile.uncounted`)
    uncounted: str | None = None

    @property
    def export_bits(self) -> float:
        """The export, λ_out: the committed bits that crossed a certified boundary out of a block, novel information
        read by their receivers. Charged only when the declaration's certifiers cover every link (`export_charged`)."""
        return sum(self.exported.values())

    @property
    def export_charged(self) -> bool:
        """Whether the export is charged: only once network certifiers cover every link, and not
        applicable while `uncovered` names any."""
        return not self.uncovered

    @property
    def overcredited(self) -> list[tuple[str, str, Interval, float, float]]:
        """Every interval where the claims credit a unit more than the declaration says it can do (`double_credit`)."""
        return double_credit(self.credits, self.capacity)

    @property
    def sound(self) -> bool:
        """No refused port-window and nothing overcredited: the AND over this model's own record check
        (`claims.check_records`), with no ε or share credit. The network verdict of record is the certifier's
        `LinkWindows.network_verdict` (`verity/core/protocols/certifier/service.py`), the same AND over a link's outcomes
        of record; this model builds no proof-service streams, so it doesn't read it."""
        return not self.refused and not self.overcredited

    @property
    def covert_bits(self) -> float:
        """Every bit the run could have carried beyond the committed computation."""
        return (self.advice_bits + self.timing_bits + sum(self.input_bits.values()) + self.refused_bits
                + sum(self.serviced.values()))

    def __repr__(self) -> str:
        lines = [f"{'block':8} {'ANDs':>9} {'local ≥':>9} {'busy':>9} {'outsource':>9} {'reads':>8} {'local ≥':>8} "
                 f"{'carried (bits)':>14}  MACs by format (not credited)"]
        for b in self.blocks:
            lines.append(f"{b.block:8} {b.credited:9.3g} {b.local_at_least:9.3g} {100 * b.busy:8.3g}% {b.outsourceable:9.3g} "
                         f"{b.reads:8.3g} {b.reads_local_at_least:8.3g} {b.carried:14.3g}  "
                         + (", ".join(f"{k[4:]} {n:.3g}" for k, n in sorted(b.macs.items())) or "none"))
        if self.uncounted:
            lines.append(f"nothing credited: {self.uncounted}")
        inputs = ", ".join(f"{k}: {v}" for k, v in self.input_bits.items()) if len(self.input_bits) <= 4 else \
            f"{len(self.input_bits)} inputs"
        lines.append(f"covert budget: {self.covert_bits:g} bits = {self.advice_bits:g} profile advice + "
                     f"{self.timing_bits:g} timing advice + {sum(self.input_bits.values())} inputs ({inputs or 'none'})"
                     + (f" + {self.refused_bits:.3g} on refused windows" if self.refused_bits else "")
                     + "".join(f" + {v:.3g} held by {b} in maintenance" for b, v in self.serviced.items()))
        if self.exported:
            ports = ", ".join(f"{p!r}: {v:g}" for p, v in sorted(self.exported.items(), key=lambda kv: repr(kv[0])))
            lines.append(f"export: {self.export_bits:g} committed bits out of blocks ({ports}), " + (
                "charged as novel information read by their receivers" if self.export_charged else
                f"not applicable while no certifier covers {', '.join(self.uncovered)}"))
        for s, where in self.located.items():
            lines.append(f"store {s}: held on {where[0]}" if len(where) == 1 else
                         f"store {s}: held on one of {', '.join(where)}" if where else
                         f"store {s}: located nowhere (its deadline leaves time to rebuild it)")
        lines += [f"salt custody of {b.block}: k = {b.custody:g} bits could carry its salt out before registration"
                  for b in self.blocks if b.custody is not None]
        if self.delta:
            lines.append(f"fails with probability at most {self.delta:.3g}")
        lines += [f"REFUSED {r.on!r}: {r.why}" for r in self.refused]
        lines += [f"OVERCREDITED {u} {r} in {i}: {w:.3g} credited, {c:.3g} possible" for u, r, i, w, c in self.overcredited]
        return "\n".join(lines)


def _boundary_rate(dc: Datacenter, block: str) -> float:
    """Bits a second the block's certified boundary links carry, one direction (each link is full duplex)."""
    return sum(float(l.capacity) for l in dc.boundary if block in
               {getattr(dc.block_of(l.a), "name", None), getattr(dc.block_of(l.b), "name", None)})


def _sample(profile: Profile, c: Held) -> float:
    """The bits that answered the request behind `c`: the challenge's or the wipe's sample, else the whole store."""
    plan = profile.plan
    asks = [q.sample for q in plan.challenges if q.store == c.store and Interval(q.at, q.at + q.deadline) == c.during]
    asks += [w.sample for w in plan.wipes if w.store.name == c.store and w.during == c.during]
    return next((a for a in asks if a is not None), c.bits)


def account(claims: list[Claim], profile: Profile, assume: tuple[Assumption, ...] = ()) -> Account:
    """Read `claims` against `profile`: credit their ANDs, and the time the work takes at the devices' declared rates for
    each kind of it, under the named premises in `assume`."""
    dc, epoch = profile.datacenter, profile.plan.epoch
    seconds = dc.cadence.seconds(epoch.length)
    inside = lambda c: getattr(c, "during", epoch) in epoch  # noqa: E731
    refused = [c for c in claims if isinstance(c, Refused)]
    refused_parts = {r.on.name for r in refused if r.on.kind == "part"}
    refused_stores = {r.on.name for r in refused if r.on.kind == "store"}

    credited = defaultdict(float)
    for c in claims:
        if isinstance(c, Computed) and inside(c) and c.part not in refused_parts:
            credited[c.part] += c.work

    serviced = [m for m in profile.plan.maintenance if m.during.start < epoch.end and epoch.start < m.during.end]
    back = {m.block: max(n.during.end for n in serviced if n.block == m.block) for m in serviced}

    located: dict[str, list[str]] = {}
    filled: dict[str, float] = defaultdict(float)
    for c in claims:
        if isinstance(c, Held) and inside(c) and c.store not in refused_stores and c.store not in located:
            s, bits, deadline = profile.stores[c.store], _sample(profile, c), dc.cadence.seconds(c.during.length)
            located[c.store] = [] if regen_time(dc, s, bits) <= deadline else [
                d.name for d in dc.devices.values() if d.memory is not None and float(d.memory) >= c.bits
                and answer_time(dc, c.by, s.device, d.name, bits) <= deadline]
            rejoined = c.during.start >= back.get(getattr(dc.block_of(s.device), "name", None), epoch.start)
            if s.content == "labels" and located[c.store] == [s.device] and rejoined:
                filled[s.device] = max(filled[s.device], min(s.bits, fill_rate(dc, s.device) * deadline))

    reads, reads_local = defaultdict(float), defaultdict(float)
    for c in claims:
        if isinstance(c, Read) and inside(c) and c.store not in refused_stores:
            device = profile.stores[c.store].device
            if (b := dc.block_of(device)) is not None:
                reads[b.name] += c.bits
                reads_local[b.name] += max(c.bits - into(dc, device) * dc.cadence.seconds(c.during.length), 0.0)

    premises = {a.name for a in assume}
    ports = {repr(p): p for p in dc.ports}
    refused_blocks = {dc.watched(ports[r.on.name]).name for r in refused if r.on.kind == "port" and r.on.name in ports}
    blocks = []
    for b in dc.blocks:
        work = profile.work[b.name] if credited[b.name] else {}
        rates = {k: sum(float(d.rate[k]) for d in b.devices if k in d.rate) for k in work}
        loads: dict[str, float] = defaultdict(float)
        for k, n in work.items():
            loads[facet(k)] += (n / rates[k] if rates[k] else float("inf")) / seconds
        busy = profile.ticks[b.name] / epoch.length if b.name in profile.ticks and work else max(loads.values(), default=0.0)
        carried = sum(max(float(d.residual) if d.residual else 0.0, float(d.memory) - filled[d.name])
                      for d in b.devices if d.memory is not None)
        names = [a.name for d in b.devices for a in [*(d.rate.get(k) for k in work), d.memory, d.residual, d.bandwidth,
                                                     d.on_chip] if a is not None]
        if profile.partition == "assumed" and profile.units.get(b.name):
            names.append(SERVED_PARTITION.name)
        bounds: dict[str | None, float] = {None: float("inf")}
        if FLEET_CLOSED.name in premises:
            bounds[FLEET_CLOSED.name] = 0.0
            names.append(FLEET_CLOSED.name)
        if profile.forced[b.name]:
            bounds["forced"] = min(credited[b.name], _boundary_rate(dc, b.name) * seconds * credited[b.name] / profile.forced[b.name])
        k = None
        if profile.salt.get(b.name):
            k = profile.salt_out[b.name] + sum(c.bits for c in claims if isinstance(c, Scheduled)) + sum(
                c.bits for c in claims if isinstance(c, Carried) and c.port.direction == "egress"
                and dc.watched(c.port).name == b.name and c.during.start < profile.plan.registration)
            if SALT_CUSTODY.name in premises and k <= float(SALT_CUSTODY) and b.name not in refused_blocks:
                bounds[SALT_CUSTODY.name] = max(credited[b.name] - profile.pinned[b.name], 0.0)
                names.append(SALT_CUSTODY.name)
        anchor = min(bounds, key=lambda a: bounds[a])
        blocks.append(BlockAccount(b.name, credited[b.name], busy, bounds[anchor], carried, reads[b.name],
                                   reads_local[b.name], profile.forced[b.name], names, k, dict(loads), anchor,
                                   {kind: n for kind, n in profile.work.get(b.name, {}).items() if kind.startswith("mac:")}))

    links = {repr(port): dc.uplink(port.certifier) for port in dc.ports}
    window_seconds = dc.cadence.seconds(dc.cadence.window_ticks)
    voided = {(r.on.name, r.on.during) for r in refused if r.on.kind == "port"}
    refused_bits = sum(float(links[name].capacity) * window_seconds for name, _ in voided if name in links)
    return Account(blocks,
                   sum(c.bits for c in claims if isinstance(c, Carried) and inside(c)),
                   {c.input: c.bits for c in claims if isinstance(c, Entered)},
                   refused, located,
                   sum(c.bits for c in claims if isinstance(c, Scheduled)),
                   refused_bits,
                   sum(c.delta for c in claims if isinstance(c, (Computed, Held, Read))),
                   {b: sum(float(d.memory) for d in dc.block(b).devices if d.memory is not None) for b in back},
                   _credits(claims, profile, blocks, inside, refused_stores), capacity(dc),
                   dict(profile.exported), [f"{l.a}-{l.b}" for l in dc.uncovered_links()], seconds,
                   sum(d.kind == "gpu" for b in dc.blocks for d in b.devices), profile.uncounted)


def _credits(claims: list[Claim], profile: Profile, blocks: list[BlockAccount], inside, refused_stores) -> list[Credit]:
    """What the claims credit each unit. PoUW credits the prover system, so a block's busy share is the block's only as
    far as attribution places it there (`local_at_least`, graded by its anchor), and the rest is the whole datacenter's.
    A store's device is credited the bits read from it and the bit-ticks it held."""
    epoch, stores, fleet = profile.plan.epoch, profile.stores, profile.datacenter.name
    out = []
    for b in blocks:
        if not b.credited:
            continue
        local = b.local_at_least / b.credited
        proved = b.block in profile.ticks
        timed = {"compute": profile.ticks[b.block]} if proved else {
            f"compute:{f}": share * epoch.length for f, share in b.loads.items()}
        basis = "proved" if proved else "honest-routing"
        for resource, ticks in timed.items():
            if local:
                out.append(Credit(b.block, resource, epoch, ticks * local, b.anchor, basis))
            if local < 1:
                out.append(Credit(fleet, resource, epoch, ticks * (1 - local), None, basis))
    for c in claims:
        if isinstance(c, (Read, Held)) and inside(c) and c.store not in refused_stores:
            out.append(Credit(stores[c.store].device, "read", c.during, c.bits) if isinstance(c, Read) else
                       Credit(stores[c.store].device, "held", c.during, c.bits * c.during.length))
    return out


@dataclass
class Ledger:
    """The joint ledger: accounts of successive epochs (nights, say) or of audits that overlap, joined. Covert bits and
    failure probabilities add up, since an adversary with a long horizon can save them up and break out at the end, and
    so does what could have carried the registered weights out (`custody`); what each epoch could carry in is per epoch,
    and nightly wipes keep it down to the devices' residuals; and each unit's time, bandwidth and memory are credited
    once across all of them (`joint`)."""

    epochs: list[Account] = field(default_factory=list)

    def add(self, acc: Account) -> Ledger:
        self.epochs.append(acc)
        return self

    def joint(self) -> list[tuple[str, str, Interval, float, float]]:
        """The joint rule across the ledger's accounts: every interval where they credit a unit's resource more than the
        declaration says it can do. Two audits of one device in overlapping windows can't both have its whole time. And
        two PoUW windows (`compute` timed by its proved bound) that overlap on one block are refused outright: their
        bounds don't add, since an honest pair may have run on different facets at once, and the sound joint credit
        would be PoUW's bound over the union of their work."""
        cap = {k: v for a in self.epochs for k, v in a.capacity.items()}
        out = double_credit([c for a in self.epochs for c in a.credits], cap)
        timed = [(i, c) for i, a in enumerate(self.epochs) for c in a.credits if c.resource == "compute" and c.grade]
        for n, (i, x) in enumerate(timed):
            for j, y in timed[n + 1:]:
                lo, hi = max(x.during.start, y.during.start), min(x.during.end, y.during.end)
                if i != j and x.unit == y.unit and lo < hi:
                    out.append((x.unit, "compute", Interval(lo, hi), x.work + y.work, 0.0))
        return out

    @property
    def sound(self) -> bool:
        return all(a.sound for a in self.epochs) and not self.joint()

    @property
    def covert_bits(self) -> float:
        return sum(a.covert_bits for a in self.epochs)

    @property
    def delta(self) -> float:
        return sum(a.delta for a in self.epochs)

    @property
    def carried(self) -> list[float]:
        return [sum(b.carried for b in a.blocks) for a in self.epochs]

    def custody(self, registered_bits: float, sigma: Assumption = DEDICATED_RACKS) -> Custody:
        """The share of the registered weights, `registered_bits`, that could have left over the ledger's epochs: each
        epoch's covert budget, its export where it is charged, and σ (`sigma`) from each of its servers over its real
        duration. Weight custody over a horizon needs this share well below 1. The covert budget's inbound channels
        (inputs, ingress timing) count too, and the certifier protocol's free fields don't, since no record here has
        any."""
        return Custody(registered_bits, self.covert_bits, sum(a.export_bits for a in self.epochs if a.export_charged),
                       float(sigma) * sum(a.seconds * a.servers for a in self.epochs), sigma,
                       sum(a.export_bits for a in self.epochs if not a.export_charged))

    def __repr__(self) -> str:
        return (f"{len(self.epochs)} epochs, {'sound' if self.sound else 'NOT sound'}: {self.covert_bits:g} covert bits in "
                f"all, at most {max(self.carried, default=0):.3g} bits carried into any epoch, failure probability at "
                f"most {self.delta:.3g}")


@dataclass(frozen=True)
class Custody:
    """What of the registered weights could have left over a ledger's epochs (`Ledger.custody`): the covert budgets,
    the export where it is charged and the side channel, against the `registered` bits."""

    registered: float
    covert: float
    export: float
    side: float
    #: σ, the side channel's rate from each server
    sigma: Assumption
    #: the export of epochs whose declaration leaves a link uncovered: reported, not charged
    uncharged: float = 0.0

    @property
    def bits(self) -> float:
        """The bits that could have left."""
        return self.covert + self.export + self.side

    @property
    def share(self) -> float:
        """The share of the registered weights that could have left."""
        return self.bits / self.registered

    def __repr__(self) -> str:
        return (f"custody: {self.bits:g} bits could have left = {self.covert:g} covert + {self.export:g} export + "
                f"{self.side:.3g} side channel at {self.sigma!r}, {self.share:.3g} of the {self.registered:g} registered "
                f"bits" + (f"; {self.uncharged:g} exported bits not charged while a link is uncovered"
                           if self.uncharged else ""))


@dataclass(frozen=True)
class Credit:
    """Work `work` of `resource` an audit credits to `unit` within `during`: a block's `compute` (PoUW's proved bound) or
    `compute:<facet>` in block-ticks,
    a device's `read` in bits and its `held` in bit-ticks over the interval the claim proves (`capacity` gives each per
    tick; a unit with no capacity for a resource can give it nothing)."""

    unit: str
    resource: str
    during: Interval
    work: float
    #: what places it on the unit, for a block's compute (its attribution anchor)
    grade: str | None = None
    #: for compute, what the time rests on: "proved" (PoUW's bound, (1 − ε) of the floor program's time, over every
    #: route) or "honest-routing" (each kind on its named facet at the declared rates, which no audit proves)
    basis: str | None = None


def capacity(dc: Datacenter) -> dict[tuple[str, str], float]:
    """What each unit can give each resource in one tick, by the declaration: a block one tick of each facet of its
    devices (`compute:<facet>`, in block-ticks), and the datacenter its blocks' together, for the compute no anchor
    places on a block; a device its read bandwidth (`read`, in bits) and its memory (`held`, in bits, so bit-ticks a
    tick)."""
    tick = dc.cadence.seconds(1)
    out = {(b.name, f"compute:{f}"): 1.0 for b in dc.blocks for f in {facet(k) for d in b.devices for k in d.rate}}
    out.update({(b.name, "compute"): 1.0 for b in dc.blocks})
    for (b, r) in list(out):
        out[dc.name, r] = out.get((dc.name, r), 0.0) + 1.0
    for d in dc.devices.values():
        if d.bandwidth is not None:
            out[d.name, "read"] = float(d.bandwidth) * tick
        if d.memory is not None:
            out[d.name, "held"] = float(d.memory)
    return out


def double_credit(credits: list[Credit], capacity: dict[tuple[str, str], float]) -> list[tuple[str, str, Interval, float, float]]:
    """Every interval where credit from several audits exceeds what the unit could do: (unit, resource, [a, b),
    credited inside, capacity over b − a), wherever the work credited to windows inside [a, b) exceeds the unit's
    capacity per tick for that resource times b − a (Horn's preemptive test with release times and deadlines). Empty
    when each unit's time is credited once."""
    out = []
    groups = defaultdict(list)
    for c in credits:
        groups[c.unit, c.resource].append(c)
    for (u, r), cs in groups.items():
        starts, ends = sorted({c.during.start for c in cs}), sorted({c.during.end for c in cs})
        for a in starts:
            for b in ends:
                if b <= a:
                    continue
                inside = sum(c.work for c in cs if a <= c.during.start and c.during.end <= b)
                possible = capacity.get((u, r), 0.0) * (b - a)
                if inside > possible:
                    out.append((u, r, Interval(a, b), inside, possible))
    return out
