"""The logical layer's verdicts: what an audit establishes, inside the proof, from the trace and the public profile.

`audit` reads the trace (witness: the placement, the values, the trusted devices' records and answers) and the profile
(public). It never reads a capacity, and its claims never say which device did something: that is the physical layer's
question. Each claim is a fact a proof would establish about the prover *system*, over an interval `during` of the
trusted clock:

- `Computed(part, work)`: the values committed for a part of the circuit are its evaluation (here by re-execution; in
  Verity, sampled proofs at ε and δ).
- `Carried(port, bits)`: a certifier port carried exactly the committed crossings in the declared occupancy, so its only
  freedom was `bits` of timing advice.
- `Entered(input, bits)`: an input's bits entered the computation (registered and trusted inputs don't count).
- `Scheduled(bits)`: the occupancy profile was one of several registered ones: advice, once.
- `Held(store, bits)`: the system held `bits` of a registered store on average over `during` (`bits` × the length in
  bit-ticks: PoUS's S_cert·T; a wipe's labels too).
- `Read(store, bits)`: the system read so many bits of a registered store within `during`: a read window's output,
  certified by its receipts (the bandwidth audits).
- `Refused(on, why)`: something didn't check, and `on` says what: a port's window, a store, a part, an input.

A record must exist for every certifier port and every window of the epoch: a missing one is refused, never silence.
The one exception is a block the verifier's plan has out for maintenance, whose ports owe no records then; the account
charges what that costs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .binding import Profile, traffic
from .logical import commit, digest as content_digest
from .memory import Challenge, ReadWindow, Wipe
from .observe import Record, Trace, digest
from .physical import Cadence, Port
from .time import Interval


@dataclass(frozen=True)
class Computed:
    part: str
    work: float
    during: Interval
    epsilon: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True)
class Carried:
    port: Port
    bits: float
    during: Interval


@dataclass(frozen=True)
class Entered:
    input: str
    bits: int
    source: str


@dataclass(frozen=True)
class Scheduled:
    bits: float


@dataclass(frozen=True)
class Held:
    store: str
    bits: float
    during: Interval
    by: str = "challenger"
    epsilon: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True)
class Read:
    store: str
    bits: float
    during: Interval
    by: str = "challenger"
    epsilon: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True)
class On:
    """What a refusal is about: a certifier port during a window, a store, a part of the circuit, or an input."""

    kind: Literal["port", "store", "part", "input"]
    name: str
    during: Interval | None = None

    def __repr__(self) -> str:
        return f"{self.kind} {self.name}" + (f" {self.during}" if self.during else "")


@dataclass(frozen=True)
class Refused:
    on: On
    why: str


Claim = Computed | Carried | Entered | Scheduled | Held | Read | Refused


def check_records(records: list[Record], declared: dict[tuple[Port, int], Record], cadence: Cadence) -> list[Carried | Refused]:
    """The certifiers' records against the ones an honest run publishes (`declared`, by port and window): each declared
    port-window needs a record, with the declared counts and content digest, and is then carried with log2 of the
    jitter classes' number bits of timing advice. Records of other windows are other epochs' to check. The digest is
    whatever the certifier signs (here `observe.digest` of the committed crossings; in the mock datacenter, the real
    certifier's)."""
    per = cadence.window_ticks
    have = {(r.port, r.window): r for r in records}
    out: list[Carried | Refused] = []
    for (port, w), d in declared.items():
        on = On("port", repr(port), Interval(w * per, (w + 1) * per))
        r = have.get((port, w))
        problems = (["no record: the certifier was down or its record was withheld"] if r is None else
                    ["occupancy differs from the declared profile"] * (r.counts != d.counts)
                    + ["a frame is not a committed crossing"] * (r.digest != d.digest))
        out += [Refused(on, why) for why in problems] or [Carried(port, math.log2(cadence.jitter_classes), on.during)]
    return out


def audit(trace: Trace, profile: Profile) -> list[Claim]:
    """The claims `trace` supports against the public `profile`."""
    p, out = trace.placement, []
    c, g, plan = p.circuit, profile.datacenter.cadence, profile.plan
    reference = c.evaluate(trace.inputs)
    if commit(c, reference, trace.salt)[0] != trace.commitment:
        out += [Refused(On("part", b), "the committed values are not the circuit's evaluation") for b in profile.work]
    else:
        out += [Computed(b, profile.credit(b), plan.epoch) for b in profile.work if profile.credit(b)]
    for v in c.inputs:
        spec = c.gates[v.id].input
        if spec.source == "registered" and spec.digest is not None and content_digest(trace.inputs[v.name], v.bits) != spec.digest:
            out.append(Refused(On("input", v.name), "its bytes differ from the registered digest"))
        elif spec.source in ("outside", "inside"):
            out.append(Entered(v.name, spec.bits, spec.source))
    if plan.profiles > 1:
        out.append(Scheduled(math.log2(plan.profiles)))

    content: dict[tuple[Port, int], list[tuple[int, int]]] = {}
    for port, row in traffic(p).items():
        for b, vs in row.items():
            content.setdefault((port, b // g.buckets), []).extend((v.id, reference[v.id]) for v in vs)
    dc, per = profile.datacenter, g.window_ticks
    declared = {(port, w): Record(port, w, tuple(profile.occupancy.get(port, {}).get(w * g.buckets + t, 0)
                                                 for t in range(g.buckets)), digest(content.get((port, w), [])))
                for port in dc.ports for w in profile.windows
                if not plan.out(dc.watched(port).name, Interval(w * per, (w + 1) * per))}
    out += check_records(trace.records, declared, g)

    for a in trace.answers:
        q = a.request
        if isinstance(q, Wipe):
            out.append(Held(q.store.name, q.bits, q.during, q.challenger) if a.correct
                       else Refused(On("store", q.store.name, q.during), "a spot check came back wrong"))
        elif isinstance(q, Challenge):
            s, during = profile.stores[q.store], Interval(q.at, q.at + q.deadline)
            if not a.correct:
                out.append(Refused(On("store", q.store, during), "the answer was wrong"))
            elif a.arrived - q.at > q.deadline:
                out.append(Refused(On("store", q.store, during),
                                   f"answered after {a.arrived - q.at} ticks, past its deadline of {q.deadline}"))
            else:
                out.append(Held(q.store, plan.held(s, during), during, q.challenger))
        elif isinstance(q, ReadWindow):
            out.append(Read(q.store, a.bits, q.during, q.challenger) if a.correct and a.arrived <= q.deadline
                       else Refused(On("store", q.store, q.during), "the roots arrived after the deadline"))
    return out
