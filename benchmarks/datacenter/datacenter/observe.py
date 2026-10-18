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

from .binding import Placement, crossings
from .logical import Commitment, commit
from .time import Interval, Wipe


@dataclass(frozen=True)
class Record:
    """A certifier's signed record of one link-window: frames per bucket, and the digest of their content."""

    link: tuple[str, str]
    window: int
    counts: tuple[int, ...]
    digest: bytes
    certifier: str


@dataclass(frozen=True)
class WipeRecord:
    """The challenger's record of a wipe: which device, when, and whether every spot check came back right in time."""

    wipe: Wipe
    passed: bool


@dataclass
class Deviation:
    """How a dishonest run departs from the honest one: `extra[(link, bucket)]` frames of covert content added on a
    boundary link, `hold[(link, bucket)] = bucket'` moving a bucket's frames later, and wipes that fail."""

    extra: dict[tuple[tuple[str, str], int], int] = field(default_factory=dict)
    hold: dict[tuple[tuple[str, str], int], int] = field(default_factory=dict)
    failed_wipes: set[str] = field(default_factory=set)


@dataclass
class Trace:
    """What a run leaves behind: the prover's commitment to every value (witness: values and salt), and the trusted
    devices' records (witness too: they reach the verifier only inside proofs)."""

    placement: Placement
    commitment: Commitment
    values: dict[int, int]
    salt: bytes
    records: list[Record]
    wipes: list[WipeRecord]
    inputs: dict[str, int]
    frame_bits: int
    bucket: int
    window: int


def _digest(frames: list[tuple[int, int]]) -> bytes:
    h = hashlib.sha512(b"datacenter/frames/v1\0")
    for vid, val in sorted(frames):
        h.update(vid.to_bytes(4, "big") + val.to_bytes(16, "big", signed=False))
    return h.digest()


def run(p: Placement, inputs: dict[str, int], *, frame_bits: int = 32, bucket: int = 1, window: int = 8,
        wipes: list[Wipe] | None = None, deviation: Deviation | None = None) -> Trace:
    """Execute placement `p` on `inputs`, recording every boundary link in windows of `window` buckets of `bucket`
    ticks."""
    dev = deviation or Deviation()
    values = p.circuit.evaluate(inputs)
    commitment, salt = commit([values[i] for i in range(len(values))], [g.bits for g in p.circuit.gates])
    boundary = {(l.a, l.b): l for l in p.datacenter.boundary}
    counts: dict[tuple[str, str], dict[int, int]] = {k: {} for k in boundary}
    content: dict[tuple[tuple[str, str], int], list[tuple[int, int]]] = {}
    for x in crossings(p):
        frames = -(-x.value.bits // frame_bits)
        for l in x.path:
            key = (l.a, l.b)
            if key not in boundary:
                continue
            b = dev.hold.get((key, p.tick(x.value) // bucket), p.tick(x.value) // bucket)
            counts[key][b] = counts[key].get(b, 0) + frames
            content.setdefault((key, b // window), []).append((x.value.id, values[x.value.id]))
    for (key, b), n in dev.extra.items():
        counts[key][b] = counts[key].get(b, 0) + n
        content.setdefault((key, b // window), []).append((1 << 31, n))
    horizon = max((b for row in counts.values() for b in row), default=0) // window + 1
    records = []
    for key, row in counts.items():
        cert = p.datacenter.certifier_of(boundary[key])
        for w in range(horizon):
            cs = tuple(row.get(w * window + t, 0) for t in range(window))
            records.append(Record(key, w, cs, _digest(content.get((key, w), [])), cert.name if cert else "?"))
    wipe_records = [WipeRecord(w, w.device not in dev.failed_wipes) for w in (wipes or [])]
    return Trace(p, commitment, values, salt, records, wipe_records, dict(inputs), frame_bits, bucket, window)
