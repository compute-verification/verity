"""A run: a placed Program executing on a datacenter, and what the trusted devices record of it.

`run` sits outside both layers: it is the world. It binds the run's inputs and the registered ones, evaluates the
Program, commits every value, sends each crossing's frames through the certifier ports on its path, and has every
certifier record each window of each port: the frames in each bucket (the occupancy, the only thing timing can carry
once content is fixed) and a digest of their content. The challenger records each answer: to a challenge, to a read
window, to a wipe's spot checks. A `Deviation` plays the adversary.

This is the run's model, on simulated ticks. The same records come from the real certifier
(`verity.core.protocols.certifier`) when the mock datacenter runs it on real links (`benchmarks/network_sim`).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from verity.ml.commitments import advice

from .binding import Placed, frames, traffic
from .logical import RegisteredInputs, commit, evaluate
from .memory import Challenge, ReadWindow, Wipe, answer_time, read_rate
from .physical import Port
from .plan import Plan


@dataclass(frozen=True)
class PortRecord:
    """A certifier's signed record of one port for one window: frames per bucket, and the digest of their content."""

    port: Port
    window: int
    counts: tuple[int, ...]
    digest: bytes


@dataclass(frozen=True)
class Response:
    """The challenger's record of one request (a challenge, a read window, a wipe's spot checks): the tick the answer
    was complete, whether it was right, and for a read window the bits its receipts certify."""

    request: Challenge | ReadWindow | Wipe
    arrived: int
    correct: bool = True
    bits: float = 0.0


@dataclass
class Deviation:
    """How a dishonest run departs from the honest one:
    - `extra[(port, bucket)]`: covert frames added at a port;
    - `hold[(port, bucket)] = bucket'`: a bucket's frames sent later;
    - `down`: (port, window)s with no record (the certifier down, or its record withheld);
    - `moved[store] = device`: a store kept somewhere other than where it is registered;
    - `failed_wipes`: devices whose wipe spot checks come back wrong (state hidden through the wipe);
    - `swapped[parameter]`: other bytes run in place of a registered input's (a container image swapped, say)."""

    extra: dict[tuple[Port, int], int] = field(default_factory=dict)
    hold: dict[tuple[Port, int], int] = field(default_factory=dict)
    down: set[tuple[Port, int]] = field(default_factory=set)
    moved: dict[str, str] = field(default_factory=dict)
    failed_wipes: set[str] = field(default_factory=set)
    swapped: dict[str, int | Sequence[int]] = field(default_factory=dict)


@dataclass
class Trace:
    """What a run leaves behind, all of it witness: the placement, the inputs it bound, every value with the commitment
    to them, the registration it opened, and the trusted devices' records and answers. Only `audit`, inside the proof,
    reads it."""

    placement: Placed
    inputs: dict[str, int | Sequence[int]]
    values: dict[int, int]
    commitment: advice.Committed
    registration: RegisteredInputs | None
    records: list[PortRecord]
    answers: list[Response]


def digest(content: list[tuple[int, int]]) -> bytes:
    """The digest a certifier signs over a window's frames: their values, in order."""
    h = hashlib.sha512(b"datacenter/frames/v1\0")
    for vid, val in sorted(content):
        n = (val.bit_length() + 7) // 8
        h.update(vid.to_bytes(4, "big") + n.to_bytes(4, "big") + val.to_bytes(n, "big"))
    return h.digest()


def run(p: Placed, inputs: dict[str, int | Sequence[int]], plan: Plan, deviation: Deviation | None = None) -> Trace:
    """Execute placement `p` on the run's `inputs` (by parameter; the registered ones come from `p.registration`)
    through the epoch of `plan`, answering its challenges, read demands and wipes from wherever the bits really are."""
    dev, dc = deviation or Deviation(), p.datacenter
    g = dc.cadence
    bound = {**(p.registration.values if p.registration else {}), **dev.swapped, **inputs}
    values = evaluate(p.program, bound)
    commitment = commit(p.program, values)
    counts: dict[Port, dict[int, int]] = {port: {} for port in dc.ports}
    content: dict[tuple[Port, int], list[tuple[int, int]]] = {}
    for port, row in traffic(p).items():
        for b, vs in row.items():
            b2 = dev.hold.get((port, b), b)
            counts[port][b2] = counts[port].get(b2, 0) + frames(vs, g.frame_bits)
            content.setdefault((port, b2 // g.buckets), []).extend((x, values[x]) for v in vs for x in p.gates(v))
    for (port, b), n in dev.extra.items():
        counts[port][b] = counts[port].get(b, 0) + n
        content.setdefault((port, b // g.buckets), []).append((1 << 31, n))
    per = g.window_ticks
    windows = range(plan.epoch.start // per, -(-plan.epoch.end // per))
    records = [PortRecord(port, w, tuple(row.get(w * g.buckets + t, 0) for t in range(g.buckets)), digest(content.get((port, w), [])))
               for port, row in counts.items() for w in windows if (port, w) not in dev.down]
    stores = {s.name: s for s in p.stores}
    answers = []
    ticks = lambda seconds: math.ceil(seconds / float(g.tick) - 1e-9)  # noqa: E731
    for ch in plan.challenges:
        s = stores[ch.store]
        bits = plan.bits(s, ch.at) if ch.sample is None else ch.sample
        answers.append(Response(ch, ch.at + ticks(answer_time(dc, ch.challenger, s.device, dev.moved.get(s.name, s.device), bits))))
    for rw in plan.reads:
        s = stores[rw.store]
        rate = read_rate(dc, s.device, dev.moved.get(s.name, s.device))
        read = min(plan.bits(s, rw.start), rate * g.seconds(rw.boundary - rw.start))
        answers.append(Response(rw, rw.boundary + ticks(answer_time(dc, rw.challenger, s.device, s.device, 0)), bits=read))
    for w in plan.wipes:
        answers.append(Response(w, w.during.start, w.device not in dev.failed_wipes))
    return Trace(p, bound, values, commitment, p.registration, records, answers)
