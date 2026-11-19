"""A device's log: its attester's entries, oldest first, in a file of its own that fails closed.

An entry is Core's attester's entry (`Definitions/Core/Challenger.lean`, `Challenger.Attested`): the digests of what the
device was shown (`shown`), its data (`data`), and, where the device read its clock as it made the entry, that reading's
`ns` (`verity.clock.Reading`, whose `key` is the log's `device`). The log's file is its `kind` line, then one entry a
line: each of `shown` and `data` in lowercase hex at its registered width (`widths`, in bytes), separated by a space,
then the reading's ns when the entry carries one. `append` takes the file's lock, reads the file again, refuses an entry whose key (`key`) an entry already
holds, and appends it, fsynced, before it returns. Nothing here rewrites or truncates a log, and `create` refuses one
that exists.

What an entry says rests on the host keeping the file as written: it is never restored from a snapshot, a backup or
another older copy (`trusted-device`; Lean's `Verity.Assumptions.TrustedDevice`). The log is unsigned: a session's close
signs what the session read of it.
"""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class LogRefused(RuntimeError):
    """A device's log is missing, unreadable or unwritable, or refuses an entry: its key is already in it."""


@dataclass(frozen=True, slots=True)
class Attested:
    """An attester's entry: the digests of what the device was shown, its data, and its clock's reading in ns, if it read
    one."""

    shown: tuple[bytes, ...]
    data: bytes
    ns: int | None = None


class AttestedLog:
    """The log of the device registered under `device` at `path`, of `kind`, its entries' fields at `widths` (module
    docstring), one entry a `key`."""

    def __init__(self, path: str | os.PathLike, kind: str, widths: Sequence[int], key: Callable[[Attested], bytes],
                 device: str) -> None:
        self.path, self.kind, self.widths, self.key, self.device = Path(path), kind, tuple(widths), key, device
        self._line = re.compile(" ".join(f"([0-9a-f]{{{2 * w}}})" for w in self.widths).encode() + rb"(?: (\d+))?")
        self.entries()

    @staticmethod
    def create(path: str | os.PathLike, kind: str) -> None:
        """A new log of `kind`, with no entry; refused when the file exists."""
        p = Path(path)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise LogRefused(f"{p} exists: a {kind} log is never made again") from None
        try:
            os.write(fd, kind.encode() + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        d = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)

    def entries(self) -> tuple[Attested, ...]:
        """Every entry, oldest first; refused when the log is missing or unreadable: a first line that is not its kind, a
        line that is not an entry, a key twice, or a last line cut short."""
        try:
            lines = self.path.read_bytes().split(b"\n")
        except FileNotFoundError:
            raise LogRefused(f"{self.path}: missing") from None
        except OSError as e:
            raise LogRefused(f"{self.path}: unreadable ({type(e).__name__})") from None
        found = [self._line.fullmatch(x) for x in lines[1:-1]]
        if lines[0] != self.kind.encode() or lines[-1] != b"" or not all(found):
            raise LogRefused(f"{self.path}: unreadable (not a {self.kind} log of one entry a whole line)")
        out = tuple(self._entry(m) for m in found)
        if len({self.key(e) for e in out}) != len(out):
            raise LogRefused(f"{self.path}: unreadable (a {self.kind} log holds one entry a key)")
        return out

    def _entry(self, m: re.Match[bytes]) -> Attested:
        *fields, ns = m.groups()
        *shown, data = (bytes.fromhex(x.decode()) for x in fields)
        return Attested(tuple(shown), data, None if ns is None else int(ns))

    def find(self, key: bytes) -> Attested | None:
        """The entry under `key`, or None."""
        return next((e for e in self.entries() if self.key(e) == key), None)

    def append(self, entry: Attested, refused: str) -> None:
        """`entry` appended, fsynced, before this returns; refused (`refused` its reason) when its key is in the log."""
        fields = (*entry.shown, entry.data)
        if tuple(len(x) for x in fields) != self.widths:
            raise ValueError(f"a {self.kind} entry's fields are {self.widths} bytes, not {tuple(len(x) for x in fields)}")
        line = " ".join(x.hex() for x in fields) + ("" if entry.ns is None else f" {entry.ns}")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        except FileNotFoundError:
            raise LogRefused(f"{self.path}: missing") from None
        except OSError as e:
            raise LogRefused(f"{self.path}: unwritable ({type(e).__name__})") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if self.find(self.key(entry)) is not None:
                raise LogRefused(refused)
            data = line.encode() + b"\n"
            if os.write(fd, data) != len(data):
                raise LogRefused(f"{self.path}: a short write")
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["Attested", "AttestedLog", "LogRefused"]
