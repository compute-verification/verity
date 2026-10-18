"""The physical layer's reading of the claims: credit, headroom, where things were, and the covert budget, per block.

`account` reads only public things: the claims (logical facts about the prover system) and the profile (the
placement's public properties, the declaration with its named assumptions, the verifier's plan). For each block it
concludes:

- `credited`: the work the claims prove was spent on the declared circuit there, each kind at its weight (W_ref);
- `busy` and `headroom`: the share of the epoch that work needs on the block's devices, each kind at their declared
  rate for it (multiply-adds at a precision, Boolean gates), and the share the audits leave unaccounted;
- `outsourceable` and `local_at_least`: the work a helper off the block could have done for it, and so the credited
  work that must have run on the block. Crossing bytes bound nothing by themselves: a helper holding the registered
  weights needs only small inputs and outputs. Work is pinned to the block only by data it must consume that only the
  block receives, a trusted device's fresh data (`Input(source="trusted")`), which a helper could get only through the
  block's certified egress: outsourceable ≤ egress × work per forced bit. Or by the named premise `FLEET_CLOSED`. Or,
  under the premise `SALT_CUSTODY`, by a salt the block's own certifier releases only into it: the work that depends
  on it is pinned while `custody`, k, the bits that could carry the salt out before registration (the block's egress
  timing advice, the profile's advice, inside inputs and salt-dependent values crossing out), stays at most 384. With
  none, it is unbounded and nothing is attributed;
- `reads` and `reads_local_at_least`: the reads the bandwidth audits certify for stores on the block's devices, and the
  part no path into the registered device (a boundary link, or host DRAM over PCIe) can have carried in;
- `carried`: the bits that could have been carried into the epoch: each device's memory, less what a wipe's labels
  filled, and never less than its residual. For a block out for maintenance, only a wipe after it counts.

For the whole datacenter it gives where each held store could have been (`located`: the devices that could have
answered its sample by the deadline, and none when the prover could rebuild the sample in time), the covert budget (the profile's
advice, timing advice, inputs, every refused port-window charged at its link's full capacity, so it fails closed, and
everything a block out for maintenance held), and the failure probability (the sum of the claims' deltas).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .binding import Profile
from .claims import Carried, Claim, Computed, Entered, Held, Read, Refused, Scheduled
from .memory import answer_time, regen_time
from .physical import Assumption, Datacenter
from .time import Interval

#: the premise that no undeclared compute is reachable from a block (compute accounting's option (a))
FLEET_CLOSED = Assumption("fleet-closed", 1, "premise", "declared")
#: compute accounting's salt custody (note:compute-accounting/20261018T0730Z-draft-attrib, unproved): a salt the block's
#: certifier releases only into it pins the work that depends on it, while k, the bits that could carry it out before
#: registration, stays at most this many
SALT_CUSTODY = Assumption("salt-custody", 384, "bits", "declared")


@dataclass
class BlockAccount:
    block: str
    credited: float
    #: the share of the epoch the credited work needs on the block's devices, at their declared rates for each kind
    busy: float
    outsourceable: float
    carried: float
    reads: float = 0.0
    reads_local_at_least: float = 0.0
    forced: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    #: for a block with a salt in its certifier's custody, k: the bits that could carry it out before registration
    custody: float | None = None

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

    @property
    def sound(self) -> bool:
        return not self.refused

    @property
    def covert_bits(self) -> float:
        """Every bit the run could have carried beyond the committed computation."""
        return (self.advice_bits + self.timing_bits + sum(self.input_bits.values()) + self.refused_bits
                + sum(self.serviced.values()))

    def __repr__(self) -> str:
        lines = [f"{'block':8} {'credited':>9} {'local ≥':>9} {'busy':>9} {'outsource':>9} {'reads':>8} {'local ≥':>8} "
                 f"{'carried (bits)':>14}"]
        for b in self.blocks:
            lines.append(f"{b.block:8} {b.credited:9.3g} {b.local_at_least:9.3g} {100 * b.busy:8.3g}% {b.outsourceable:9.3g} "
                         f"{b.reads:8.3g} {b.reads_local_at_least:8.3g} {b.carried:14.3g}")
        inputs = ", ".join(f"{k}: {v}" for k, v in self.input_bits.items()) if len(self.input_bits) <= 4 else \
            f"{len(self.input_bits)} inputs"
        lines.append(f"covert budget: {self.covert_bits:g} bits = {self.advice_bits:g} profile advice + "
                     f"{self.timing_bits:g} timing advice + {sum(self.input_bits.values())} inputs ({inputs or 'none'})"
                     + (f" + {self.refused_bits:.3g} on refused windows" if self.refused_bits else "")
                     + "".join(f" + {v:.3g} held by {b} in maintenance" for b, v in self.serviced.items()))
        for s, where in self.located.items():
            lines.append(f"store {s}: held on {where[0]}" if len(where) == 1 else
                         f"store {s}: held on one of {', '.join(where)}" if where else
                         f"store {s}: located nowhere (its deadline leaves time to rebuild it)")
        lines += [f"salt custody of {b.block}: k = {b.custody:g} bits could carry its salt out before registration"
                  for b in self.blocks if b.custody is not None]
        if self.delta:
            lines.append(f"fails with probability at most {self.delta:.3g}")
        lines += [f"REFUSED {r.on!r}: {r.why}" for r in self.refused]
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


def _into(dc: Datacenter, device: str) -> float:
    """Bits a second every path into `device` from the rest of the datacenter carries: the links that end at it."""
    return sum(float(l.capacity) for l in dc.links if device in (l.a, l.b))


def account(claims: list[Claim], profile: Profile, assume: tuple[Assumption, ...] = ()) -> Account:
    """Read `claims` against `profile`: credit by the profile's weights, and the time it takes at the devices' declared
    rates for each kind of work, under the named premises in `assume`."""
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
                filled[s.device] = max(filled[s.device], s.bits)

    reads, reads_local = defaultdict(float), defaultdict(float)
    for c in claims:
        if isinstance(c, Read) and inside(c) and c.store not in refused_stores:
            device = profile.stores[c.store].device
            if (b := dc.block_of(device)) is not None:
                reads[b.name] += c.bits
                reads_local[b.name] += max(c.bits - _into(dc, device) * dc.cadence.seconds(c.during.length), 0.0)

    premises = {a.name for a in assume}
    ports = {repr(p): p for p in dc.ports}
    refused_blocks = {dc.watched(ports[r.on.name]).name for r in refused if r.on.kind == "port" and r.on.name in ports}
    blocks = []
    for b in dc.blocks:
        work = profile.work[b.name] if credited[b.name] else {}
        rates = {k: sum(float(d.rate[k]) for d in b.devices if k in d.rate) for k in work}
        busy = sum(n / rates[k] if rates[k] else float("inf") for k, n in work.items()) / seconds
        carried = sum(max(float(d.residual) if d.residual else 0.0, float(d.memory) - filled[d.name])
                      for d in b.devices if d.memory is not None)
        names = [a.name for d in b.devices for a in [*(d.rate.get(k) for k in work), d.memory, d.residual, d.bandwidth,
                                                     d.on_chip] if a is not None]
        bounds = [float("inf")]
        if FLEET_CLOSED.name in premises:
            bounds.append(0.0)
            names.append(FLEET_CLOSED.name)
        if profile.forced[b.name]:
            bounds.append(min(credited[b.name], _boundary_rate(dc, b.name) * seconds * credited[b.name] / profile.forced[b.name]))
        k = None
        if profile.salt.get(b.name):
            k = profile.salt_out[b.name] + sum(c.bits for c in claims if isinstance(c, Scheduled)) + sum(
                c.bits for c in claims if isinstance(c, Carried) and c.port.direction == "egress"
                and dc.watched(c.port).name == b.name and c.during.start < profile.plan.registration)
            if SALT_CUSTODY.name in premises and k <= float(SALT_CUSTODY) and b.name not in refused_blocks:
                bounds.append(max(credited[b.name] - profile.pinned[b.name], 0.0))
                names.append(SALT_CUSTODY.name)
        blocks.append(BlockAccount(b.name, credited[b.name], busy, min(bounds), carried, reads[b.name],
                                   reads_local[b.name], profile.forced[b.name], names, k))

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
                   {b: sum(float(d.memory) for d in dc.block(b).devices if d.memory is not None) for b in back})


@dataclass
class Ledger:
    """Accounts of successive epochs (nights, say), joined. Covert bits and failure probabilities add up across epochs,
    since an adversary with a long horizon can save them up and break out at the end; what each epoch could carry in is
    per epoch, and nightly wipes keep it down to the devices' residuals."""

    epochs: list[Account] = field(default_factory=list)

    def add(self, acc: Account) -> Ledger:
        self.epochs.append(acc)
        return self

    @property
    def sound(self) -> bool:
        return all(a.sound for a in self.epochs)

    @property
    def covert_bits(self) -> float:
        return sum(a.covert_bits for a in self.epochs)

    @property
    def delta(self) -> float:
        return sum(a.delta for a in self.epochs)

    @property
    def carried(self) -> list[float]:
        return [sum(b.carried for b in a.blocks) for a in self.epochs]

    def __repr__(self) -> str:
        return (f"{len(self.epochs)} epochs, {'sound' if self.sound else 'NOT sound'}: {self.covert_bits:g} covert bits in "
                f"all, at most {max(self.carried, default=0):.3g} bits carried into any epoch, failure probability at "
                f"most {self.delta:.3g}")


@dataclass(frozen=True)
class Credit:
    """Work `work` of `resource` (compute, held, read) an audit credits to `unit` (a device or a block) within `during`."""

    unit: str
    resource: str
    during: Interval
    work: float


def credits(acc: Account, epoch: Interval) -> list[Credit]:
    """The account's compute credit as `Credit`s, one per block, for the joint rule (`double_credit`)."""
    return [Credit(b.block, "compute", epoch, b.credited) for b in acc.blocks if b.credited]


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
                if inside > capacity[u, r] * (b - a):
                    out.append((u, r, Interval(a, b), inside, capacity[u, r] * (b - a)))
    return out
