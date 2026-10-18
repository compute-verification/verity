"""The logical layer's verdicts: what an audit establishes from records and commitments alone.

`audit` reads only cryptographic objects (the commitment to every value, the trusted devices' records) and public
declarations (the circuit, the occupancy profile, the sync set). It never reads a device's capacity: that is the
physical layer's job. Its output is a list of claims about the prover *system*, each a fact a proof would establish:

- `Computed`: the committed values are the circuit's evaluation (here by re-execution; in Verity, sampled proofs at ε).
- `Carried`: a boundary link-window carried the committed crossings and nothing else, so its only freedom was timing
  advice, at most `bits`.
- `Entered`: an input's bits entered the computation.
- `Erased`: a device's memory was overwritten and checked during an interval.
- `Refused`: something didn't check, and which window or device it was.
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
class Erased:
    device: str
    at: Interval


@dataclass(frozen=True)
class Refused:
    what: str
    why: str


Claim = Computed | Carried | Entered | Erased | Refused


def audit(trace: Trace, profile: dict[tuple[str, str], dict[int, int]], syncs: int = 1) -> list[Claim]:
    """The claims `trace` supports against the public occupancy `profile` (declared before the run) and a sync set of
    `syncs` values per window (the timing advice each window may carry: log2 syncs bits)."""
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
    for r in trace.records:
        declared = tuple(profile.get(r.link, {}).get(r.window * trace.window + t, 0) for t in range(trace.window))
        if r.counts != declared:
            out.append(Refused(f"link {r.link} window {r.window}", "occupancy differs from the declared profile"))
        elif r.digest != _digest(expected.get((r.link, r.window), [])):
            out.append(Refused(f"link {r.link} window {r.window}", "a frame is not a committed crossing"))
        else:
            out.append(Carried(r.link, r.window, math.log2(syncs)))
    for v in c.inputs:
        spec = c.gates[v.id].input
        if spec.source != "registered":
            out.append(Entered(v.name, spec.bits, spec.source))
    for w in trace.wipes:
        out.append(Erased(w.wipe.device, w.wipe.at) if w.passed else Refused(f"wipe of {w.wipe.device}", "a spot check failed"))
    return out


def declared_profile(trace: Trace) -> dict[tuple[str, str], dict[int, int]]:
    """The honest occupancy profile of `trace`'s placement: what the developer declares before the run."""
    return occupancy(trace.placement, trace.frame_bits, trace.bucket)
