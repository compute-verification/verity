"""The physical layer's reading of the claims: credit, headroom, attribution and the covert budget, per block.

`account` takes the audit's claims (logical facts about the prover system), the placement (the binding), the declared
datacenter (named assumptions) and an epoch on the trusted clock. For each block it concludes:

- `credited`: the work the claims prove was spent on the declared circuit there;
- `capacity` and `headroom`: what the block's devices could do in the epoch, by their declared rates, and what the
  audit leaves unaccounted;
- `outsourceable` and `local_at_least`: the work a helper off the block could have done for it, and so the credited
  work that must have run on the block. A helper holding the registered weights needs only small inputs and outputs, so
  crossing bytes bound nothing by themselves (compute accounting, review 1). Work is pinned to the block only by data it
  must consume that only the block receives: a trusted device's fresh bulk data (`Input(source="trusted")`), which a
  helper could get only through the block's certified egress, so outsourceable ≤ egress × work per forced bit. Or by a
  named premise, `assume={"fleet-closed"}`: no undeclared compute is reachable from the block. With neither, it is
  unbounded and nothing is attributed;
- `reads` and `reads_local_at_least`: the reads the bandwidth audits certify for stores on the block's devices, and
  the part that can't have come in over its boundary;
- `state`: the bits that could survive the epoch: each device's memory less what a store it held filled (a verified
  wipe is such a store), and never less than its residual.

For the whole datacenter it gives the covert budget (timing advice, inputs, off-ramps), where each answered store is
(`located`: every device that could have held it and met the deadline), and every refused claim, which voids the
conclusion for its link, store or window.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .binding import Placement
from .claims import Carried, Claim, Computed, Emitted, Entered, Held, Read, Refused, Scheduled
from .memory import Store, answer_time, label_store
from .physical import Datacenter
from .time import Clock, Interval, Wipe


@dataclass
class BlockAccount:
    block: str
    credited: float
    capacity: float
    outsourceable: float
    state: float
    reads: float = 0.0
    read_ingress: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    forced: float = 0.0

    @property
    def headroom(self) -> float:
        return max(self.capacity - self.credited, 0.0)

    @property
    def local_at_least(self) -> float:
        """Work that must have been done on the block itself: what the claims credit, less what a helper off the block
        could have computed and shipped in through its certified ingress."""
        return max(self.credited - self.outsourceable, 0.0)

    @property
    def reads_local_at_least(self) -> float:
        """Certified reads that can't have been served from off the block: the certified reads less the bits its
        boundary could carry in during the read windows."""
        return max(self.reads - self.read_ingress, 0.0)


@dataclass
class Account:
    blocks: list[BlockAccount]
    timing_bits: float
    input_bits: dict[str, int]
    refused: list[Refused]
    offramp_bits: dict[str, int] = field(default_factory=dict)
    #: each held store, and every device that could have held it and still answered by its deadlines
    located: dict[str, list[str]] = field(default_factory=dict)
    #: bits of advice in the choice of occupancy profile (NET-5's declared scheduler), once per audit
    advice_bits: float = 0.0
    #: the probability that some claim the account relies on fails: the sum of their deltas
    delta: float = 0.0

    @property
    def sound(self) -> bool:
        return not self.refused

    @property
    def covert_bits(self) -> float:
        """Every bit the run could have carried beyond the committed computation: the profile's advice, timing advice,
        inputs and off-ramps."""
        return self.advice_bits + self.timing_bits + sum(self.input_bits.values()) + sum(self.offramp_bits.values())

    def __repr__(self) -> str:
        lines = [f"{'block':8} {'credited':>9} {'local ≥':>9} {'capacity':>9} {'headroom':>9} {'outsourceable':>13} "
                 f"{'reads':>9} {'state (bits)':>12}"]
        for b in self.blocks:
            lines.append(f"{b.block:8} {b.credited:9.3g} {b.local_at_least:9.3g} {b.capacity:9.3g} {b.headroom:9.3g} "
                         f"{b.outsourceable:13.3g} {b.reads:9.3g} {b.state:12.3g}")
        lines.append(f"covert budget: {self.covert_bits:g} bits = {self.advice_bits:g} profile advice + {self.timing_bits:g} timing advice + "
                     f"{sum(self.input_bits.values())} inputs ({', '.join(f'{k}: {v}' for k, v in self.input_bits.items()) or 'none'}) + "
                     f"{sum(self.offramp_bits.values())} off-ramp ({', '.join(f'{k}: {v}' for k, v in self.offramp_bits.items()) or 'none'})")
        for s, where in self.located.items():
            lines.append(f"store {s}: held on {where[0]}" if len(where) == 1 else
                         f"store {s}: held on one of {', '.join(where)} (not located to one device)")
        if self.delta:
            lines.append(f"fails with probability at most {self.delta:.3g} (the sum of the claims' deltas)")
        lines += [f"REFUSED {r.what}: {r.why}" for r in self.refused]
        return "\n".join(lines)


def _store(dc: Datacenter, p: Placement, name: str) -> Store:
    if name.startswith("wipe:"):
        return label_store(dc, Wipe(name.removeprefix("wipe:"), Interval(0, 0)))
    return next(s for s in p.stores if s.name == name)


def _ingress(dc: Datacenter, block: str, seconds: float) -> float:
    """Bits the block's certified boundary links can carry over `seconds`, in either direction (each link is full duplex
    at its capacity)."""
    return sum(float(l.capacity) * seconds for l in dc.boundary
               if block in {getattr(dc.block_of(l.a), "name", None), getattr(dc.block_of(l.b), "name", None)})


def account(claims: list[Claim], p: Placement, clock: Clock, epoch: Interval, op: str = "scalar",
            assume: frozenset[str] | set[str] = frozenset()) -> Account:
    """Read `claims` against `p`'s datacenter over `epoch`, crediting work priced in op class `op`, under the named
    premises in `assume` (so far only "fleet-closed")."""
    dc = p.datacenter
    seconds = clock.seconds(epoch.length)
    refused = [c for c in claims if isinstance(c, Refused)]
    work = defaultdict(float)
    if any(isinstance(c, Computed) for c in claims):
        for i, g in enumerate(p.circuit.gates):
            work[p.block(i)] += g.boolean_size
    forced = defaultdict(float)
    for i, g in enumerate(p.circuit.gates):
        if g.op == "input" and g.input.source == "trusted":
            readers = {p.block(j) for j, h in enumerate(p.circuit.gates) if i in h.operands} - {None}
            for b in readers:
                forced[b] += g.bits

    challengers = [d.name for d in dc.devices.values() if d.kind == "challenger"]
    located: dict[str, list[str]] = {}
    filled: dict[str, float] = defaultdict(float)
    for c in claims:
        if isinstance(c, Held) and c.at in epoch:
            s = _store(dc, p, c.store)
            if c.store.startswith("wipe:") or not challengers:
                where = [s.device]
            else:
                where = [d.name for d in dc.devices.values() if d.memory is not None and float(d.memory) >= s.bits
                         and answer_time(dc, clock, challengers[0], s.device, d.name, s.bits) <= clock.seconds(c.at.length)]
            located[c.store] = where
            if where == [s.device]:
                filled[s.device] += c.bits

    reads: dict[str, float] = defaultdict(float)
    read_ingress: dict[str, float] = defaultdict(float)
    for c in claims:
        if isinstance(c, Read):
            b = dc.block_of(_store(dc, p, c.store).device)
            if b is not None:
                reads[b.name] += c.bits
                read_ingress[b.name] += _ingress(dc, b.name, clock.seconds(c.window.length))

    blocks = []
    for b in dc.blocks:
        cap = sum(float(d.rate[op]) for d in b.devices if op in d.rate) * seconds
        state = 0.0
        for d in b.devices:
            if d.memory is not None:
                floor = float(d.residual) if d.residual else 0.0
                state += max(floor, float(d.memory) - filled[d.name])
        names = [a.name for d in b.devices for a in [*d.rate.values(), d.memory, d.residual, d.bandwidth, d.on_chip]
                 if a is not None]
        if "fleet-closed" in assume:
            outsourceable = 0.0
            names.append("fleet-closed")
        elif forced[b.name]:
            outsourceable = min(work[b.name], _ingress(dc, b.name, seconds) * work[b.name] / forced[b.name])
        else:
            outsourceable = float("inf")
        blocks.append(BlockAccount(b.name, work[b.name], cap, outsourceable, state, reads[b.name], read_ingress[b.name],
                                   names, forced[b.name]))

    per_window: dict[tuple[str, int], int] = {}
    for c in claims:
        if isinstance(c, Emitted):        # a stream crosses several certified links: count each window of it once
            per_window[c.stream, c.window] = max(per_window.get((c.stream, c.window), 0), c.bits)
    offramp: dict[str, int] = defaultdict(int)
    for (stream, _), bits in per_window.items():
        offramp[stream] += bits
    timing = sum(c.bits for c in claims if isinstance(c, Carried))
    inputs = {c.input: c.bits for c in claims if isinstance(c, Entered)}
    advice = sum(c.bits for c in claims if isinstance(c, Scheduled))
    delta = sum(c.delta for c in claims if isinstance(c, (Computed, Held, Read)))
    return Account(blocks, timing, inputs, refused, dict(offramp), located, advice, delta)


@dataclass
class Ledger:
    """Accounts of successive epochs (nights, say), joined. Covert bits and failure probabilities add up across epochs,
    since an adversary with a long horizon can save them up and break out at the end. State is per epoch: what could
    survive each epoch's end, which nightly wipes keep down to the devices' residuals."""

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
        """The bits that could survive each epoch's end, summed over blocks."""
        return [sum(b.state for b in a.blocks) for a in self.epochs]

    def __repr__(self) -> str:
        return (f"{len(self.epochs)} epochs, {'sound' if self.sound else 'NOT sound'}: {self.covert_bits:g} covert bits in "
                f"all, at most {max(self.carried, default=0):.3g} bits carried across any epoch boundary, "
                f"failure probability at most {self.delta:.3g}")


@dataclass(frozen=True)
class Credit:
    """Work `work` of `resource` (compute, bits held, bits read) an audit credits to `device` within `window`."""

    device: str
    resource: str
    window: Interval
    work: float


def schedulable(credits: list[Credit], capacity: dict[tuple[str, str], float]) -> list[tuple[str, str, Interval, float, float]]:
    """Every violation of the joint rule for credit from several audits: (device, resource, [a, b], credited inside,
    capacity over b − a) wherever the work credited to windows inside [a, b] exceeds the device's capacity for that
    resource per tick times b − a. Empty when the credits fit (Horn's preemptive test with release times and
    deadlines), so each device's time is credited once."""
    out = []
    groups = defaultdict(list)
    for c in credits:
        groups[c.device, c.resource].append(c)
    for (d, r), cs in groups.items():
        starts, ends = sorted({c.window.start for c in cs}), sorted({c.window.end for c in cs})
        for a in starts:
            for b in ends:
                if b <= a:
                    continue
                inside = sum(c.work for c in cs if a <= c.window.start and c.window.end <= b)
                if inside > capacity[d, r] * (b - a):
                    out.append((d, r, Interval(a, b), inside, capacity[d, r] * (b - a)))
    return out
