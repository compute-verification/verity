"""The logical layer's verdicts: what an audit establishes from records and commitments alone.

`audit` reads only cryptographic objects (the commitment to every value, the trusted devices' records) and public
declarations (the circuit, the occupancy profile, the sync set, each challenge's deadline). It never reads a device's
capacity or names a device as the place something happened: that is the physical layer's job. Its output is a list of
claims about the prover *system*, each a fact a proof would establish:

- `Computed`: the committed values are the circuit's evaluation (here by re-execution; in Verity, sampled proofs at ε).
- `Carried`: a boundary link-window carried the committed crossings and nothing else, so its only freedom was timing
  advice, at most `bits`.
- `Entered`: an input's bits entered the computation.
- `Emitted`: a declared off-ramp (telemetry) carried bits the circuit doesn't explain.
- `Scheduled`: the occupancy profile was one of so many registered profiles, which is advice worth log2 of that.
- `Held`: the prover system held `bits` of a registered store throughout an interval (PoUS; a wipe is a store that
  fills a device's memory, so a verified wipe is a `Held`).
- `Read`: the prover system read at least `bits` of a registered store within an interval (the bandwidth audits).
- `Refused`: something didn't check, and which window, store or link it was.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .binding import crossings, occupancy
from .logical import commit
from .observe import Trace, _digest
from .time import Interval


@dataclass(frozen=True)
class Computed:
    gates: int
    epsilon: float = 0.0


@dataclass(frozen=True)
class Carried:
    link: tuple[str, str]
    window: int
    bits: float


@dataclass(frozen=True)
class Entered:
    input: str
    bits: int
    source: str


@dataclass(frozen=True)
class Emitted:
    """Off-ramp data (telemetry) left on a link in a window: `bits` bits the circuit doesn't explain."""

    stream: str
    link: tuple[str, str]
    window: int
    bits: int


@dataclass(frozen=True)
class Scheduled:
    """The occupancy profile was one of `profiles` registered profiles (a declared scheduler), so its choice is advice
    worth `bits` = log2 profiles, once per audit."""

    profiles: int
    bits: float


@dataclass(frozen=True)
class Held:
    """The prover system held `bits` of registered store `store` throughout `at`: every challenge in it came back right
    by its deadline. Where the bits were is the physical layer's reading of the deadlines."""

    store: str
    bits: float
    at: Interval
    epsilon: float = 0.0


@dataclass(frozen=True)
class Read:
    """The prover system read at least `bits` of registered store `store` within `window`, of which `credited` count as
    useful reads."""

    store: str
    window: Interval
    bits: float
    credited: float


@dataclass(frozen=True)
class Refused:
    what: str
    why: str


Claim = Computed | Carried | Entered | Emitted | Scheduled | Held | Read | Refused


def audit(trace: Trace, profile: dict[tuple[str, str], dict[int, int]], syncs: int = 1, profiles: int = 1) -> list[Claim]:
    """The claims `trace` supports against the public occupancy `profile` (declared before the run, one of `profiles`
    registered profiles) and a sync set of `syncs` values per window (the timing advice each window may carry: log2 syncs
    bits). Every boundary link must have a record for every window the profile spans: a missing one fails closed."""
    p, out = trace.placement, []
    c = p.circuit
    reference = c.evaluate(trace.inputs)
    recommitted, _ = commit([reference[i] for i in range(len(reference))], [g.bits for g in c.gates], trace.salt)
    if recommitted != trace.commitment:
        out.append(Refused("computation", "the committed values are not the circuit's evaluation"))
    else:
        out.append(Computed(c.boolean_size))
    expected: dict[tuple[tuple[str, str], int], list[tuple[int, int]]] = {}
    boundary = {(l.a, l.b) for l in p.datacenter.boundary}
    for x in crossings(p):
        for l in x.path:
            if (l.a, l.b) in boundary:
                w = p.tick(x.value) // trace.bucket // trace.window
                expected.setdefault(((l.a, l.b), w), []).append((x.value.id, reference[x.value.id]))
    if profiles > 1:
        out.append(Scheduled(profiles, math.log2(profiles)))
    horizon = max((b for row in profile.values() for b in row), default=0) // trace.window + 1
    have = {(r.link, r.window) for r in trace.records}
    for link in sorted(boundary):
        for w in range(horizon):
            if (link, w) not in have:
                out.append(Refused(f"link {link} window {w}", "no record: the certifier was down or its record was withheld"))
    declared_streams = {s.name: s for s in p.streams}
    for r in trace.records:
        where = f"link {r.link} window {r.window}"
        declared = tuple(profile.get(r.link, {}).get(r.window * trace.window + t, 0) for t in range(trace.window))
        problems = []
        if r.counts != declared:
            problems.append(Refused(where, "occupancy differs from the declared profile"))
        if r.digest != _digest(expected.get((r.link, r.window), [])):
            problems.append(Refused(where, "a frame is not a committed crossing"))
        problems += [Refused(f"stream {name} on {r.link} window {r.window}", "its bytes differ from the registered digest")
                     for name, _, ok in r.streams if declared_streams[name].registered and not ok]
        if problems:
            out += problems
            continue
        out.append(Carried(r.link, r.window, math.log2(syncs)))
        out += [Emitted(name, r.link, r.window, n * trace.frame_bits)
                for name, n, _ in r.streams if not declared_streams[name].registered]
    for v in c.inputs:
        spec = c.gates[v.id].input
        if spec.source != "registered":
            out.append(Entered(v.name, spec.bits, spec.source))
    stores = {s.name: s for s in trace.stores}
    for r in trace.challenges:
        ch, took = r.challenge, r.answered - r.challenge.at
        if not r.correct:
            out.append(Refused(f"store {ch.store} at {ch.at}", "the answer was wrong"))
        elif took > ch.deadline:
            out.append(Refused(f"store {ch.store} at {ch.at}", f"answered after {took} ticks, past its deadline of {ch.deadline}"))
        else:
            out.append(Held(ch.store, stores[ch.store].bits, Interval(ch.at, ch.at + ch.deadline)))
    for r in trace.reads:
        if r.served > r.audit.window.end:
            out.append(Refused(f"reads of {r.audit.store} in {r.audit.window}", "the reads finished after the window"))
        else:
            out.append(Read(r.audit.store, r.audit.window, r.audit.bits, r.audit.bits))
    return out


def declared_profile(trace: Trace) -> dict[tuple[str, str], dict[int, int]]:
    """The honest occupancy profile of `trace`'s placement: what the developer declares before the run."""
    return occupancy(trace.placement, trace.frame_bits, trace.bucket, trace.window)
