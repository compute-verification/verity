"""A run: a placed circuit executing on a datacenter, and what the trusted devices record of it.

`run` evaluates the circuit, moves each crossing's frames along its path at its value's tick, and has every certifier
record each window of its boundary link: the number of real frames in each bucket (the occupancy, the only thing timing
can carry once the content is fixed) and a digest of the frames' content. The challenger records each wipe it ran. A
`Deviation` lets a test play the adversary: extra frames, frames held to other buckets, or a wipe the device fails.

This is the run's model, on simulated ticks. The same records come from the real certifier (`verity.core.protocols
.certifier`) when the mock datacenter runs it on real links (`datacenter.lab`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import math

from .binding import Placement, traffic
from .logical import Commitment, commit
from .memory import Challenge, ReadAudit, Store, answer_time, label_store, read_time
from .time import Clock, Interval, Wipe


@dataclass(frozen=True)
class Record:
    """A certifier's signed record of one direction of one link, for one window: frames per bucket, and the digest of
    their content. `link` is the direction (from, to)."""

    link: tuple[str, str]
    window: int
    counts: tuple[int, ...]
    digest: bytes
    certifier: str
    #: frames of declared streams in the window, by stream (the certifier reads a frame's stream from its header), and
    #: whether a registered stream's frames matched its digest
    streams: tuple[tuple[str, int, bool], ...] = ()


@dataclass(frozen=True)
class ChallengeRecord:
    """The challenger's record of a challenge: the tick its answer arrived, and whether the answer was right."""

    challenge: Challenge
    answered: int
    correct: bool = True


@dataclass(frozen=True)
class ReadRecord:
    """The verifier's record of a bandwidth audit window: the tick the last read was served."""

    audit: ReadAudit
    served: int


@dataclass
class Deviation:
    """How a dishonest run departs from the honest one: `extra[(link, bucket)]` frames of covert content added on a
    boundary link, `hold[(link, bucket)] = bucket'` moving a bucket's frames later, wipes that fail, and stores kept on
    another device than the registered one (`moved[store] = device`), registered streams whose bytes differ from their
    digest (`tampered`), and link-windows with no record because the certifier was down or its record was suppressed
    (`down`)."""

    extra: dict[tuple[tuple[str, str], int], int] = field(default_factory=dict)
    hold: dict[tuple[tuple[str, str], int], int] = field(default_factory=dict)
    failed_wipes: set[str] = field(default_factory=set)
    moved: dict[str, str] = field(default_factory=dict)
    tampered: set[str] = field(default_factory=set)
    down: set[tuple[tuple[str, str], int]] = field(default_factory=set)


@dataclass
class Trace:
    """What a run leaves behind: the prover's commitment to every value (witness: values and salt), and the trusted
    devices' records (witness too: they reach the verifier only inside proofs)."""

    placement: Placement
    commitment: Commitment
    values: dict[int, int]
    salt: bytes
    records: list[Record]
    inputs: dict[str, int]
    frame_bits: int
    bucket: int
    window: int
    stores: tuple[Store, ...] = ()
    challenges: list[ChallengeRecord] = field(default_factory=list)
    reads: list[ReadRecord] = field(default_factory=list)
    clock: Clock = field(default_factory=Clock)


def _digest(frames: list[tuple[int, int]]) -> bytes:
    h = hashlib.sha512(b"datacenter/frames/v1\0")
    for vid, val in sorted(frames):
        h.update(vid.to_bytes(4, "big") + val.to_bytes(16, "big", signed=False))
    return h.digest()


def run(p: Placement, inputs: dict[str, int], *, frame_bits: int = 32, bucket: int = 1, window: int = 8,
        wipes: list[Wipe] | None = None, challenges: list[Challenge] | None = None,
        reads: list[ReadAudit] | None = None, clock: Clock | None = None, deviation: Deviation | None = None) -> Trace:
    """Execute placement `p` on `inputs`, recording every boundary link in windows of `window` buckets of `bucket`
    ticks, each challenge to a registered store and each bandwidth-audit window, served from wherever the bits really
    are. A wipe is a store of labels filling its device's memory (`label_store`), spot-checked within its interval."""
    clock = clock or Clock()
    dev = deviation or Deviation()
    values = p.circuit.evaluate(inputs)
    commitment, salt = commit([values[i] for i in range(len(values))], [g.bits for g in p.circuit.gates])
    links = {frozenset((l.a, l.b)): l for l in p.datacenter.boundary}
    counts: dict[tuple[str, str], dict[int, int]] = {}
    content: dict[tuple[tuple[str, str], int], list[tuple[int, int]]] = {}
    streamed: dict[tuple[tuple[str, str], int], dict[str, tuple[int, bool]]] = {}
    for hop, row in traffic(p, frame_bits, bucket, window).items():
        for b, slot in row.items():
            moved = dev.hold.get((hop, b), b)
            counts.setdefault(hop, {})[moved] = counts.get(hop, {}).get(moved, 0) + slot.frames(frame_bits)
            content.setdefault((hop, moved // window), []).extend((v.id, values[v.id]) for v in slot.values)
            for name, n in slot.streams.items():
                streamed.setdefault((hop, moved // window), {})[name] = (n, name not in dev.tampered)
    for (hop, b), n in dev.extra.items():
        counts.setdefault(hop, {})[b] = counts.get(hop, {}).get(b, 0) + n
        content.setdefault((hop, b // window), []).append((1 << 31, n))
    horizon = max((b for row in counts.values() for b in row), default=0) // window + 1
    records = []
    for hop, row in counts.items():
        cert = p.datacenter.certifier_of(links[frozenset(hop)])
        for w in range(horizon):
            if (hop, w) in dev.down:
                continue
            cs = tuple(row.get(w * window + t, 0) for t in range(window))
            records.append(Record(hop, w, cs, _digest(content.get((hop, w), [])), cert.name if cert else "?",
                                  tuple(sorted((n, f, ok) for n, (f, ok) in streamed.get((hop, w), {}).items()))))
    stores = {s.name: s for s in p.stores}
    answers = []
    for ch in challenges or []:
        s = stores[ch.store]
        seconds = answer_time(p.datacenter, clock, ch.challenger, s.device, dev.moved.get(s.name, s.device), s.bits)
        answers.append(ChallengeRecord(ch, ch.at + math.ceil(seconds / clock.tick - 1e-9)))
    for w in wipes or []:
        s = label_store(p.datacenter, w)
        stores[s.name] = s
        answers.append(ChallengeRecord(Challenge(s.name, w.at.start, w.at.length), w.at.start, w.device not in dev.failed_wipes))
    served = []
    for ra in reads or []:
        s = stores[ra.store]
        seconds = read_time(p.datacenter, s.device, dev.moved.get(s.name, s.device), ra.bits)
        served.append(ReadRecord(ra, ra.window.start + math.ceil(seconds / clock.tick - 1e-9)))
    return Trace(p, commitment, values, salt, records, dict(inputs), frame_bits, bucket, window, tuple(stores.values()),
                 answers, served, clock)
