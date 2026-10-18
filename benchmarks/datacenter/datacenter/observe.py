"""A run: a placed circuit executing on a datacenter, and what the trusted devices record of it.

`run` sits outside both layers: it is the world. It evaluates the circuit, sends each crossing's frames through the
certifier ports on its path, and has every certifier record each window of each port: the frames in each bucket (the
occupancy, the only thing timing can carry once content is fixed) and a digest of their content. The challenger records
each answer: to a challenge, to a read demand, to a wipe's spot checks. A `Deviation` plays the adversary.

This is the run's model, on simulated ticks. The same records come from the real certifier
(`verity.core.protocols.certifier`) when the mock datacenter runs it on real links (`benchmarks/network_sim`).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from .binding import Placement, frames, traffic
from .logical import Commitment, commit
from .memory import Challenge, Reads, Wipe, answer_time, serve_time
from .physical import Port
from .plan import Plan


@dataclass(frozen=True)
class Record:
    """A certifier's signed record of one port for one window: frames per bucket, and the digest of their content."""

    port: Port
    window: int
    counts: tuple[int, ...]
    digest: bytes


@dataclass(frozen=True)
class Answer:
    """The challenger's record of one request (a challenge, a read demand, a wipe's spot checks): the tick the answer
    was complete, and whether it was right."""

    request: Challenge | Reads | Wipe
    arrived: int
    correct: bool = True


@dataclass
class Deviation:
    """How a dishonest run departs from the honest one:
    - `extra[(port, bucket)]`: covert frames added at a port;
    - `hold[(port, bucket)] = bucket'`: a bucket's frames sent later;
    - `down`: (port, window)s with no record (the certifier down, or its record withheld);
    - `moved[store] = device`: a store kept somewhere other than where it is registered;
    - `failed_wipes`: devices whose wipe spot checks come back wrong (state hidden through the wipe)."""

    extra: dict[tuple[Port, int], int] = field(default_factory=dict)
    hold: dict[tuple[Port, int], int] = field(default_factory=dict)
    down: set[tuple[Port, int]] = field(default_factory=set)
    moved: dict[str, str] = field(default_factory=dict)
    failed_wipes: set[str] = field(default_factory=set)


@dataclass
class Trace:
    """What a run leaves behind, all of it witness: the placement, every value with the commitment to them, and the
    trusted devices' records and answers. Only `audit`, inside the proof, reads it."""

    placement: Placement
    inputs: dict[str, int]
    values: dict[int, int]
    commitment: Commitment
    salt: bytes
    records: list[Record]
    answers: list[Answer]


def digest(content: list[tuple[int, int]]) -> bytes:
    """The digest a certifier signs over a window's frames: their values, in order."""
    h = hashlib.sha512(b"datacenter/frames/v1\0")
    for vid, val in sorted(content):
        h.update(vid.to_bytes(4, "big") + val.to_bytes(16, "big", signed=False))
    return h.digest()


def run(p: Placement, inputs: dict[str, int], plan: Plan, deviation: Deviation | None = None) -> Trace:
    """Execute placement `p` on `inputs` through the epoch of `plan`, answering its challenges, read demands and
    wipes from wherever the bits really are."""
    dev, dc = deviation or Deviation(), p.datacenter
    g = dc.cadence
    values = p.circuit.evaluate(inputs)
    commitment, salt = commit(p.circuit, values)
    counts: dict[Port, dict[int, int]] = {port: {} for port in dc.ports}
    content: dict[tuple[Port, int], list[tuple[int, int]]] = {}
    for port, row in traffic(p).items():
        for b, vs in row.items():
            b2 = dev.hold.get((port, b), b)
            counts[port][b2] = counts[port].get(b2, 0) + frames(vs, g.frame_bits)
            content.setdefault((port, b2 // g.buckets), []).extend((v.id, values[v.id]) for v in vs)
    for (port, b), n in dev.extra.items():
        counts[port][b] = counts[port].get(b, 0) + n
        content.setdefault((port, b // g.buckets), []).append((1 << 31, n))
    per = g.window_ticks
    windows = range(plan.epoch.start // per, -(-plan.epoch.end // per))
    records = [Record(port, w, tuple(row.get(w * g.buckets + t, 0) for t in range(g.buckets)), digest(content.get((port, w), [])))
               for port, row in counts.items() for w in windows if (port, w) not in dev.down]
    stores = {s.name: s for s in p.stores}
    answers = []
    ticks = lambda seconds: math.ceil(seconds / float(g.tick) - 1e-9)  # noqa: E731
    for ch in plan.challenges:
        s = stores[ch.store]
        answers.append(Answer(ch, ch.at + ticks(answer_time(dc, ch.challenger, s.device, dev.moved.get(s.name, s.device), s.bits))))
    for rd in plan.reads:
        s = stores[rd.store]
        answers.append(Answer(rd, rd.during.start + ticks(serve_time(dc, s.device, dev.moved.get(s.name, s.device), rd.bits))))
    for w in plan.wipes:
        answers.append(Answer(w, w.during.start, w.device not in dev.failed_wipes))
    return Trace(p, dict(inputs), values, commitment, salt, records, answers)
