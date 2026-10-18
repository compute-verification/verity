"""The physical layer's reading of the claims: credit, headroom, where things were, and the covert budget, per block.

`account` reads only public things: the claims (logical facts about the prover system) and the profile (the
placement's public properties, the declaration with its named assumptions, the verifier's plan). For each block it
concludes:

- `credited`: the work the claims prove was spent on the declared circuit there, in the audit's unit of work;
- `capacity` and `headroom`: what the block's devices could do in the epoch at their declared rates in that unit, and
  what the audit leaves unaccounted;
- `outsourceable` and `local_at_least`: the work a helper off the block could have done for it, and so the credited
  work that must have run on the block. Crossing bytes bound nothing by themselves: a helper holding the registered
  weights needs only small inputs and outputs. Work is pinned to the block only by data it must consume that only the
  block receives, a trusted device's fresh data (`Input(source="trusted")`), which a helper could get only through the
  block's certified egress: outsourceable ≤ egress × work per forced bit. Or by the named premise `FLEET_CLOSED`. With
  neither, it is unbounded and nothing is attributed;
- `reads` and `reads_local_at_least`: the reads the bandwidth audits certify for stores on the block's devices, and the
  part its boundary can't have carried in;
- `carried`: the bits that could have been carried into the epoch: each device's memory, less what a wipe's labels
  filled, and never less than its residual. For a block out for maintenance, only a wipe after it counts.

For the whole datacenter it gives where each held store could have been (`located`), the covert budget (the profile's
advice, timing advice, inputs, every refused port-window charged at its link's full capacity, so it fails closed, and
everything a block out for maintenance held), and the failure probability (the sum of the claims' deltas).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .binding import Profile
from .claims import Carried, Claim, Computed, Entered, Held, Read, Refused, Scheduled
from .memory import answer_time
from .physical import Assumption, Datacenter
from .time import Interval

#: the premise that no undeclared compute is reachable from a block (compute accounting's option (a))
FLEET_CLOSED = Assumption("fleet-closed", 1, "premise", "declared")


@dataclass
class BlockAccount:
    block: str
    credited: float
    capacity: float
    outsourceable: float
    carried: float
    reads: float = 0.0
    read_ingress: float = 0.0
    forced: float = 0.0
    assumptions: list[str] = field(default_factory=list)

    @property
    def headroom(self) -> float:
        return max(self.capacity - self.credited, 0.0)

    @property
    def local_at_least(self) -> float:
        """Work that must have run on the block itself: the credited work less what a helper could have done."""
        return max(self.credited - self.outsourceable, 0.0)

    @property
    def reads_local_at_least(self) -> float:
        """Certified reads that can't have been served from off the block."""
        return max(self.reads - self.read_ingress, 0.0)


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
        lines = [f"{'block':8} {'credited':>9} {'local ≥':>9} {'capacity':>9} {'headroom':>9} {'outsource':>9} "
                 f"{'reads':>8} {'carried (bits)':>14}"]
        for b in self.blocks:
            lines.append(f"{b.block:8} {b.credited:9.3g} {b.local_at_least:9.3g} {b.capacity:9.3g} {b.headroom:9.3g} "
                         f"{b.outsourceable:9.3g} {b.reads:8.3g} {b.carried:14.3g}")
        inputs = ", ".join(f"{k}: {v}" for k, v in self.input_bits.items()) if len(self.input_bits) <= 4 else \
            f"{len(self.input_bits)} inputs"
        lines.append(f"covert budget: {self.covert_bits:g} bits = {self.advice_bits:g} profile advice + "
                     f"{self.timing_bits:g} timing advice + {sum(self.input_bits.values())} inputs ({inputs or 'none'})"
                     + (f" + {self.refused_bits:.3g} on refused windows" if self.refused_bits else "")
                     + "".join(f" + {v:.3g} held by {b} in maintenance" for b, v in self.serviced.items()))
        for s, where in self.located.items():
            lines.append(f"store {s}: held on {where[0]}" if len(where) == 1 else
                         f"store {s}: held on one of {', '.join(where)}" if where else f"store {s}: on no declared device")
        if self.delta:
            lines.append(f"fails with probability at most {self.delta:.3g}")
        lines += [f"REFUSED {r.on!r}: {r.why}" for r in self.refused]
        return "\n".join(lines)


def _boundary_rate(dc: Datacenter, block: str) -> float:
    """Bits a second the block's certified boundary links carry, one direction (each link is full duplex)."""
    return sum(float(l.capacity) for l in dc.boundary if block in
               {getattr(dc.block_of(l.a), "name", None), getattr(dc.block_of(l.b), "name", None)})


def account(claims: list[Claim], profile: Profile, assume: tuple[Assumption, ...] = ()) -> Account:
    """Read `claims` against `profile`, crediting work in the profile's unit (Boolean gates for a traced circuit; PoUW's
    are MACs) at the devices' declared rates in it, under the named premises in `assume`."""
    dc, epoch, unit = profile.datacenter, profile.plan.epoch, profile.unit
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
            s = profile.stores[c.store]
            located[c.store] = [d.name for d in dc.devices.values() if d.memory is not None and float(d.memory) >= s.bits
                                and answer_time(dc, c.by, s.device, d.name, s.bits) <= dc.cadence.seconds(c.during.length)]
            rejoined = c.during.start >= back.get(getattr(dc.block_of(s.device), "name", None), epoch.start)
            if s.content == "labels" and located[c.store] == [s.device] and rejoined:
                filled[s.device] = max(filled[s.device], s.bits)

    reads, read_ingress = defaultdict(float), defaultdict(float)
    for c in claims:
        if isinstance(c, Read) and inside(c) and c.store not in refused_stores:
            b = dc.block_of(profile.stores[c.store].device)
            if b is not None:
                reads[b.name] += c.bits
                read_ingress[b.name] += _boundary_rate(dc, b.name) * dc.cadence.seconds(c.during.length)

    premises = {a.name for a in assume}
    blocks = []
    for b in dc.blocks:
        cap = sum(float(d.rate[unit]) for d in b.devices if unit in d.rate) * seconds
        carried = sum(max(float(d.residual) if d.residual else 0.0, float(d.memory) - filled[d.name])
                      for d in b.devices if d.memory is not None)
        names = [a.name for d in b.devices for a in [d.rate.get(unit), d.memory, d.residual, d.bandwidth, d.on_chip]
                 if a is not None]
        if FLEET_CLOSED.name in premises:
            outsourceable = 0.0
            names.append(FLEET_CLOSED.name)
        elif profile.forced[b.name]:
            outsourceable = min(credited[b.name], _boundary_rate(dc, b.name) * seconds * credited[b.name] / profile.forced[b.name])
        else:
            outsourceable = float("inf")
        blocks.append(BlockAccount(b.name, credited[b.name], cap, outsourceable, carried, reads[b.name],
                                   read_ingress[b.name], profile.forced[b.name], names))

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
