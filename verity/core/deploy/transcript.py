"""A party's transcript, and the record (`note:memory-accounting/20261026T2305Z-draft-deployment-api-v3` §2).

A transcript is one party's view of one run, append-only: its start (its name and parameters), every frame it received
(`in`) and sent (`out`) in the order it saw them, with its channel and kind, when a peer's frame arrived on the party's
clock (`arrived`, private, just before its `in`), what the runtime told it (`event`: a channel closed), what it recorded
(`record`) and how it finished (`finish`); the handshakes' frames are `handshake-in` and `handshake-out`, of kind
`structure`. Each entry is itself a wire frame, its head `{"type": "Entry", "v": 1, "n", "dir",
"channel", "kind", "elided"}` and its body the frame or the canonical JSON it holds, so a transcript's bytes are
canonical and its digest names it. A frame past the transcript's bound (`keep`, 64 KiB unless given) is kept as its
SHA-512, `elided`: a fill of gigabytes is a digest a frame, which a replay compares and can't feed. The record is its
public projection (`public`): the entries of the four public kinds, private ones never. A run's record is its parties'
projections, in the order of their names (`record`).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from verity.core.wire import KINDS, NAME, PUBLIC, Body, Bool, Int, Message, Refused, Registry, Schema, Str, frames

DIRS = ("start", "in", "out", "arrived", "event", "record", "finish", "handshake-in", "handshake-out", "party")
ENTRY = Schema("Entry", 1, "structure", {"n": Int(), "dir": Str(16, "|".join(DIRS)), "channel": Str(64),
                                         "kind": Str(16, "|".join(KINDS)), "elided": Bool()}, Body(max=(1 << 32) - 1))
REGISTRY = Registry([ENTRY])
KEEP = 1 << 16
"""The largest frame a transcript keeps whole by default; past it, its SHA-512."""


@dataclass(frozen=True)
class Entry:
    n: int
    dir: str
    channel: str
    kind: str
    data: bytes
    elided: bool = False

    def frame(self) -> bytes:
        return REGISTRY.encode(Message("Entry", {"n": self.n, "dir": self.dir, "channel": self.channel, "kind": self.kind,
                                                 "elided": self.elided}, self.data))

    @classmethod
    def of(cls, frame: bytes) -> Entry:
        m = REGISTRY.decode(frame)
        return cls(m.fields["n"], m.fields["dir"], m.fields["channel"], m.fields["kind"], m.body, m.fields["elided"])

    def holds(self, data: bytes) -> bool:
        """Whether this entry is of `data`: the bytes, or their SHA-512 when elided."""
        return self.data == (hashlib.sha512(data).digest() if self.elided else bytes(data))


class Transcript:
    """One party's transcript: `append` only. With `path`, each entry is also appended to that file as it is made."""

    def __init__(self, party: str, path: str | os.PathLike | None = None, keep: int = KEEP) -> None:
        if not NAME.fullmatch(party):
            raise ValueError(f"a party's name is a name, not {party!r}")
        self.party, self.keep = party, keep
        self._entries: list[Entry] = []
        self._file: BinaryIO | None = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "xb")  # held open while the run appends

    def append(self, dir: str, channel: str, kind: str, data: bytes) -> Entry:
        big = len(data) > self.keep
        e = Entry(len(self._entries), dir, channel, kind, hashlib.sha512(data).digest() if big else bytes(data), big)
        f = e.frame()
        self._entries.append(e)
        if self._file is not None:
            self._file.write(f)
            self._file.flush()
        return e

    def close(self) -> None:
        if self._file is not None:
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(tuple(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def to_bytes(self) -> bytes:
        return b"".join(e.frame() for e in self._entries)

    def digest(self) -> bytes:
        return hashlib.sha512(self.to_bytes()).digest()

    def public(self) -> Transcript:
        """The record's part of this transcript: its entries of the public kinds, numbered as here."""
        out = Transcript(self.party)
        out._entries = [e for e in self._entries if e.kind in PUBLIC]
        return out

    def on(self, channel: str) -> list[Entry]:
        return [e for e in self._entries if e.channel == channel]

    @classmethod
    def of(cls, party: str, data: bytes) -> Transcript:
        """The transcript `data` holds, parsed as adversarial: entries numbered from 0 without a gap."""
        out = cls(party)
        out._entries = [Entry.of(f) for f in frames(data, REGISTRY.max_head, REGISTRY.max_body)]
        if [e.n for e in out._entries] != list(range(len(out._entries))):
            raise Refused("malformed", "a transcript's entries are numbered from 0, one after another")
        return out

    @classmethod
    def read(cls, path: str | os.PathLike, party: str | None = None) -> Transcript:
        p = Path(path)
        return cls.of(party or p.name.split(".")[0], p.read_bytes())


def record(transcripts: Mapping[str, Transcript]) -> bytes:
    """A run's record: for each party by name, a `party` entry naming it, then its transcript's public entries."""
    out = bytearray()
    for name in sorted(transcripts):
        out += Entry(0, "party", name, "structure", b"").frame()
        out += transcripts[name].public().to_bytes()
    return bytes(out)


__all__ = ["DIRS", "ENTRY", "REGISTRY", "Entry", "Transcript", "record"]
