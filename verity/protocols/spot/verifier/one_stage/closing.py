"""A member's session's evidence, the bytes its closing round is opened after the SHA-512 of.

An audit of several members runs their sessions one after another, each on a coin tree of the window's challenger
session, and each session ends with its tree's closing round (`challenger.CLOSING_ROUND`, `challenger.closes`): a round
of no coins the challenger opens after the SHA-512 of the session's evidence, so its release log shows that evidence
fixed before the next session's tree is sampled. The evidence is what the session's verdict is read from, its record and
its proof files:

    evidence = be64(parts) ‖ (be64(len(part)) ‖ part)*       parts: session.json's bytes, then for each proof file,
                                                             by name, its name in UTF-8 and its bytes

`evidence_bytes` is Lean's `OneStageZk.evidenceBytes` and `evidence_parts` its decoder (`evidenceParts`), so the digest
fixes each part's bytes and not a digest of them; `session_evidence` is `OneStageZk.sessionEvidence` on the files'
bytes, with its refusal of a length past a word (`EVIDENCE_FITS`); and `closing_after` is its SHA-512 over a session's
directory as its server wrote it, read in chunks, which is what whoever asks the challenger for the closing round hands
it (`after`).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from .proofs import WORD

#: `OneStageZk.EVIDENCE_FITS`: the refusal of a session's evidence `evidence_bytes` can't hold
EVIDENCE_FITS = ("its evidence holds a file of 2^64 bytes or more, or as many files, past the 8-byte word its closing "
                 "round's digest holds each length in")
_CHUNK = 1 << 24


def _be64(n: int) -> bytes:
    return n.to_bytes(8, "big")


def evidence_bytes(parts: Sequence[bytes]) -> bytes:
    """The bytes a closing round's digest is of: the number of `parts` in an 8-byte big-endian word, then each part's
    length in such a word and its bytes."""
    return _be64(len(parts)) + b"".join(_be64(len(p)) + p for p in parts)


def evidence_parts(b: bytes) -> list[bytes] | None:
    """The parts `b` holds (`evidence_bytes`' decoder); None where `b` is not some parts' bytes."""
    if len(b) < 8:
        return None
    parts, at = [], 8
    for _ in range(int.from_bytes(b[:8], "big")):
        if at + 8 > len(b):
            return None
        stop = at + 8 + int.from_bytes(b[at:at + 8], "big")
        if stop > len(b):
            return None
        parts.append(b[at + 8:stop])
        at = stop
    return parts if at == len(b) else None


def session_evidence(record: bytes, reps: Sequence[tuple[str, bytes]]) -> bytes:
    """A session's evidence: its record's bytes (its ``session.json``), then each of its proof files `reps`, in the
    order given (the evidence's ``reps``: by name), its name in UTF-8 and its bytes. Raises `ValueError`
    (`EVIDENCE_FITS`) where a length is past a word."""
    parts = [record, *(x for name, b in reps for x in (name.encode(), b))]
    if len(parts) >= WORD or any(len(p) >= WORD for p in parts):
        raise ValueError(EVIDENCE_FITS)
    return evidence_bytes(parts)


def session_files(session: Path) -> tuple[Path, list[Path]]:
    """A session's directory's files as its evidence names them (`proofs.evidence_v2`): its ``session.json`` and its
    proof files (``*.bin``), by name."""
    return session / "session.json", sorted(session.glob("*.bin"), key=lambda f: f.name)


def closing_after(session: Path) -> bytes:
    """What the session in the directory `session` is closed after: the SHA-512 of its evidence (`session_evidence` of
    its `session_files`), the files read in chunks. Raises `ValueError` (`EVIDENCE_FITS`) as `session_evidence` does."""
    record, reps = session_files(session)
    sized = [(None, record), *((f.name.encode(), f) for f in reps)]
    count = 1 + 2 * len(reps)
    if count >= WORD or any(f.stat().st_size >= WORD or (name is not None and len(name) >= WORD) for name, f in sized):
        raise ValueError(EVIDENCE_FITS)
    h = hashlib.sha512(_be64(count))
    for name, f in sized:
        if name is not None:
            h.update(_be64(len(name)) + name)
        size = f.stat().st_size
        h.update(_be64(size))
        with open(f, "rb") as g:
            read = 0
            while chunk := g.read(_CHUNK):
                read += len(chunk)
                h.update(chunk)
        if read != size:
            raise ValueError(f"{f} changed while its evidence was read: {read} bytes, not the {size} it held")
    return h.digest()


__all__ = ["EVIDENCE_FITS", "closing_after", "evidence_bytes", "evidence_parts", "session_evidence", "session_files"]
