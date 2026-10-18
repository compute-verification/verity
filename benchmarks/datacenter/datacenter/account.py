"""The physical layer's reading of the claims: credit, headroom and the covert budget, per block and device.

`account` takes the audit's claims (logical facts about the prover system), the placement (the binding), the declared
datacenter (named assumptions) and an epoch on the trusted clock, and concludes, for each block:

- `credited`: the work the claims prove was spent on the declared circuit there;
- `capacity`: what the block's devices could do in the epoch, by their declared rates;
- `headroom`: the difference, the work the audit doesn't account for;
- `outsourceable`: an upper bound on work done off the block and shipped in, its certified ingress times the
  circuit's arithmetic intensity (work per bit);
- `state`: the bits of state that could survive the epoch, its devices' memory unless a verified wipe ran, then their
  residual.

and for the whole datacenter the covert budget: the bits that could leave through timing advice, the bits that entered
as inputs, and every refused window or wipe, which voids the conclusion for its link or device.

`feasible` is the joint rule for credit from several audits on one device (W5): per device and resource, for every
interval [a, b], the work credited to windows inside it fits its capacity over b − a.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .binding import Placement
from .claims import Carried, Claim, Computed, Emitted, Entered, Erased, Refused
from .time import Clock, Interval


@dataclass
class BlockAccount:
    block: str
    credited: float
    capacity: float
    outsourceable: float
    state: float
    assumptions: list[str] = field(default_factory=list)

    @property
    def headroom(self) -> float:
        return max(self.capacity - self.credited, 0.0)


@dataclass
class Account:
    blocks: list[BlockAccount]
    timing_bits: float
    input_bits: dict[str, int]
    refused: list[Refused]
    offramp_bits: dict[str, int] = field(default_factory=dict)

    @property
    def covert_bits(self) -> float:
        """Every bit the run could have carried beyond the committed computation: timing advice, inputs, off-ramps."""
        return self.timing_bits + sum(self.input_bits.values()) + sum(self.offramp_bits.values())

    @property
    def sound(self) -> bool:
        return not self.refused

    def __repr__(self) -> str:
        lines = [f"{'block':10} {'credited':>12} {'capacity':>12} {'headroom':>12} {'outsourceable':>14} {'state (bits)':>14}"]
        for b in self.blocks:
            lines.append(f"{b.block:10} {b.credited:12.3g} {b.capacity:12.3g} {b.headroom:12.3g} {b.outsourceable:14.3g} "
                         f"{b.state:14.3g}")
        lines.append(f"covert budget: {self.covert_bits:g} bits = {self.timing_bits:g} timing advice + "
                     f"{sum(self.input_bits.values())} inputs ({', '.join(f'{k}: {v}' for k, v in self.input_bits.items()) or 'none'}) + "
                     f"{sum(self.offramp_bits.values())} off-ramp ({', '.join(f'{k}: {v}' for k, v in self.offramp_bits.items()) or 'none'})")
        lines += [f"REFUSED {r.what}: {r.why}" for r in self.refused]
        return "\n".join(lines)


def account(claims: list[Claim], p: Placement, clock: Clock, epoch: Interval, op: str = "scalar") -> Account:
    """Read `claims` against `p`'s datacenter over `epoch`, crediting work priced in op class `op`."""
    dc = p.datacenter
    seconds = clock.seconds(epoch.length)
    refused = [c for c in claims if isinstance(c, Refused)]
    computed = any(isinstance(c, Computed) for c in claims)
    erased = {c.device for c in claims if isinstance(c, Erased) and c.at in epoch}
    work = defaultdict(float)
    if computed:
        for i, g in enumerate(p.circuit.gates):
            work[p.block(i)] += g.boolean_size
    total_bits = sum(g.bits for g in p.circuit.gates) or 1
    intensity = p.circuit.boolean_size / total_bits
    blocks = []
    for b in dc.blocks:
        cap = sum(float(d.rate[op]) for d in b.devices if op in d.rate) * seconds
        ingress = sum(float(l.capacity) * seconds for l in dc.boundary if b.name in {getattr(dc.block_of(l.a), "name", None),
                                                                                       getattr(dc.block_of(l.b), "name", None)})
        state = sum(float(d.residual if d.name in erased and d.residual else d.memory) for d in b.devices if d.memory)
        names = [a.name for d in b.devices for a in [*d.rate.values(), d.memory, d.residual] if a is not None]
        blocks.append(BlockAccount(b.name, work[b.name], cap, ingress * intensity, state, names))
    timing = sum(c.bits for c in claims if isinstance(c, Carried))
    inputs = {c.input: c.bits for c in claims if isinstance(c, Entered)}
    per_window: dict[tuple[str, int], int] = {}
    for c in claims:
        if isinstance(c, Emitted):        # a stream crosses several certified links: count each window of it once
            per_window[c.stream, c.window] = max(per_window.get((c.stream, c.window), 0), c.bits)
    offramp: dict[str, int] = defaultdict(int)
    for (stream, _), bits in per_window.items():
        offramp[stream] += bits
    return Account(blocks, timing, inputs, refused, dict(offramp))


@dataclass(frozen=True)
class Credit:
    """Work `work` an audit credits to `device` within `window`."""

    device: str
    window: Interval
    work: float


def feasible(credits: list[Credit], rate: dict[str, float]) -> list[tuple[str, Interval, float, float]]:
    """Every violation of the joint rule: (device, [a, b], credited inside, capacity over b − a) wherever the work
    credited to windows inside [a, b] exceeds the device's rate times b − a. Empty when the credits are jointly
    feasible (Horn's preemptive test with release times and deadlines)."""
    out = []
    by_device = defaultdict(list)
    for c in credits:
        by_device[c.device].append(c)
    for d, cs in by_device.items():
        starts, ends = sorted({c.window.start for c in cs}), sorted({c.window.end for c in cs})
        for a in starts:
            for b in ends:
                if b <= a:
                    continue
                inside = sum(c.work for c in cs if a <= c.window.start and c.window.end <= b)
                if inside > rate[d] * (b - a):
                    out.append((d, Interval(a, b), inside, rate[d] * (b - a)))
    return out
