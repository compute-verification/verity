"""The docket's digest: SHA-512 over a registration record's canonical bytes, its format as the tag."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from verity.circuits.codec import canonical_json
from verity.commitments import domains as DM

#: the kinds a docket's window names (its record's ``window.kind``)
KINDS = ("served", "stand-in")


def record_digest(tag: str, canon: bytes) -> bytes:
    """The one docket digest, `SHA-512(tag || 0 || canon)`, over a record's canonical bytes: `digest`'s, and PoUW's
    over the bytes it was handed (`pouw.core_check.digest`)."""
    return hashlib.sha512(tag.encode() + b"\0" + canon).digest()


def digest(rec: Mapping[str, Any]) -> str:
    """`SHA-512(format || 0 || canon(record))`, the record's own format (v1 or v2) as the tag."""
    tag = DM.REGISTRATION if rec.get("format") == DM.REGISTRATION else DM.REGISTRATION_V1
    return record_digest(tag, canonical_json(dict(rec))).hex()


__all__ = ["KINDS", "digest", "record_digest"]
