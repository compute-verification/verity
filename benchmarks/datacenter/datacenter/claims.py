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
- `Held(store, bits)`: the system held a registered store throughout `during` (PoUS; a wipe's labels too).
- `Read(store, bits)`: the system read so many bits of a registered store within `during` (the bandwidth audits).
- `Refused(on, why)`: something didn't check, and `on` says what: a port's window, a store, a part, an input.

A record must exist for every certifier port and every window of the epoch: a missing one is refused, never silence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .binding import Profile, traffic
from .logical import commit, digest as content_digest
from .memory import Challenge, Reads, Wipe
from .observe import Trace, digest
from .physical import Port
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


def audit(trace: Trace, profile: Profile) -> list[Claim]:
    """The claims `trace` supports against the public `profile`."""
    p, out = trace.placement, []
    c, g, plan = p.circuit, profile.datacenter.grid, profile.plan
    reference = c.evaluate(trace.inputs)
    if commit(c, reference, trace.salt)[0] != trace.commitment:
        out += [Refused(On("part", b), "the committed values are not the circuit's evaluation") for b in profile.work]
    else:
        out += [Computed(b, w, plan.epoch) for b, w in profile.work.items() if w]
    for v in c.inputs:
        spec = c.gates[v.id].input
        if spec.source == "registered" and spec.digest is not None and content_digest(trace.inputs[v.name], v.bits) != spec.digest:
            out.append(Refused(On("input", v.name), "its bytes differ from the registered digest"))
        elif spec.source in ("outside", "inside"):
            out.append(Entered(v.name, spec.bits, spec.source))
    if plan.profiles > 1:
        out.append(Scheduled(math.log2(plan.profiles)))

    per = g.bucket * g.window
    expected: dict[tuple[Port, int], list[tuple[int, int]]] = {}
    for port, row in traffic(p).items():
        for b, vs in row.items():
            expected.setdefault((port, b // g.window), []).extend((v.id, reference[v.id]) for v in vs)
    have = {(r.port, r.window) for r in trace.records}
    for port in profile.datacenter.ports:
        for w in profile.windows:
            if (port, w) not in have:
                out.append(Refused(On("port", repr(port), Interval(w * per, (w + 1) * per)),
                                   "no record: the certifier was down or its record was withheld"))
    for r in trace.records:
        during = Interval(r.window * per, (r.window + 1) * per)
        declared = tuple(profile.occupancy.get(r.port, {}).get(r.window * g.window + t, 0) for t in range(g.window))
        problems = []
        if r.counts != declared:
            problems.append("occupancy differs from the declared profile")
        if r.digest != digest(expected.get((r.port, r.window), [])):
            problems.append("a frame is not a committed crossing")
        if problems:
            out += [Refused(On("port", repr(r.port), during), why) for why in problems]
        elif r.window in profile.windows:
            out.append(Carried(r.port, math.log2(g.syncs), during))

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
                out.append(Held(q.store, s.bits, during, q.challenger))
        elif isinstance(q, Reads):
            out.append(Read(q.store, q.bits, q.during, q.challenger) if a.correct and a.arrived <= q.during.end
                       else Refused(On("store", q.store, q.during), "the reads finished after the window"))
    return out
