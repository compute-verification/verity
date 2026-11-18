"""A registered value's formats: its commitment domain, its rows' digests and leaves, its entry's keys and its supplied salts."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from verity.commitments import hm96
from verity.commitments.identity import identity_digest_sha512
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import (ROLE_X, SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2,
                                        bit_row, row_leaf, sha512_row_digest, sha512_row_seg_digest, sha512_row_v2_digest)

REGISTERED_DOMAIN = "verity/one-stage/registered-domain/v1"
#: the `READS_V1` port key: proof block `reads_from + i` reads `reads[i]`, every other proof block nothing
READS_FROM = "reads_from"
#: a registered value's row formats, each a Flock port's (`Flock.Registered.SCHEMAS`)
SCHEMAS = (SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG)
#: the default leaf width, which the domain and the entry name only by leaving `word_bits` out
WORD_BITS = 16
WIDTHS = (16, 32)
#: u16 words per SHA-512 block: a statement's row port is whole blocks
BLOCK = 64
OWNER = -2
ENTRY_KEYS = ("value", "root", "leaves", "words", "domain")
#: the entry key a value of another width than `WORD_BITS` adds
WIDTH_KEY = "word_bits"
#: the domain binding's key a per-window value adds: the window's open root
WINDOW_KEY = "window"
#: the domain binding's key a produced value adds: the digest of the docket whose sample it follows
PRODUCED_KEY = "produced"
#: the domain binding's key a value committed inside its window's open root adds in the window's place: the session the
#: auditor's ledger opened for the window (PoUW's round-0 input-ids table, `credit.domain_of`)
SESSION_KEY = "session"
SUPPLIED_SALT = "verity/one-stage/supplied-salt/v1"
#: a row's hm96-sha512 commitment string ``b || c``
BC_BYTES = 128
_INST = hm96.SHA512
_HEX = re.compile(r"[0-9a-f]+")


def _width(word_bits: int) -> int:
    if word_bits not in WIDTHS:
        raise ValueError(f"a registered value's leaves are {' or '.join(map(str, WIDTHS))} bits, not {word_bits!r}")
    return int(word_bits)


def width_entry(word_bits: int) -> dict[str, int]:
    """`{"word_bits": word_bits}`, or nothing for the default 16: what a domain binding and an entry add."""
    return {} if _width(word_bits) == WORD_BITS else {WIDTH_KEY: int(word_bits)}


def known_schema(schema: str) -> str:
    """`schema`, one of `SCHEMAS`; ValueError otherwise."""
    if schema not in SCHEMAS:
        raise ValueError(f"a registered value's rows are {', '.join(SCHEMAS)} rows, not {schema!r}")
    return schema


def window_root(window: str | None) -> str | None:
    """The window's open root a per-window value's domain binds: None, or lowercase hex of 64 bytes; ValueError
    otherwise, which the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if window is not None and not (isinstance(window, str) and len(window) == 128 and _HEX.fullmatch(window)):
        raise ValueError(f"a window root is lowercase hex of 64 bytes, not {window!r}")
    return window


def domain(program: str, value: str, words: int, leaves: int, word_bits: int = WORD_BITS,
           window: str | None = None, produced: str | None = None, session: str | None = None) -> CommitmentDomain:
    """`window`: the window's open root, for a per-window value (none for a value held across windows). `produced`: the
    docket's digest, for a value produced after its sample (SPOT's `one_stage.produced`), which binds it in the window's
    place. `session`: the window's session, for a value whose root the window's open root contains, which so can't bind
    that root, in the window's place."""
    if sum(x is not None for x in (window, produced, session)) > 1:
        raise ValueError("a domain binds one of the window's open root, a registration and a session, not two")
    if session is not None and not (isinstance(session, str) and session):
        raise ValueError(f"a session is a nonempty string, not {session!r}")
    if produced is not None and not (isinstance(produced, str) and len(produced) == 128 and _HEX.fullmatch(produced)):
        raise ValueError(f"a registration digest is lowercase hex of 64 bytes, not {produced!r}")
    binding = identity_digest_sha512(REGISTERED_DOMAIN, {
        "program": program, "value": value, "schema": SCHEMA_HM96_SHA512_ROW, "words": int(words), "leaves": int(leaves),
        **width_entry(word_bits), **({} if window_root(window) is None else {WINDOW_KEY: window}),
        **({} if produced is None else {PRODUCED_KEY: produced}), **({} if session is None else {SESSION_KEY: session})})
    return CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


def row_words(row: Sequence[int], word_bits: int = WORD_BITS, schema: str = SCHEMA_HM96_SHA512_ROW) -> list[int]:
    """A registered row as the statement's port row: each leaf's `word_bits / 16` u16 words, low first, zero-padded to
    whole SHA-512 chunks (`BLOCK` words); a bit row's (`SCHEMA_HM96_SHA512_ROW_V2`, `SCHEMA_HM96_SHA512_ROW_SEG`) unpadded."""
    k, bound = _width(word_bits) // 16, 1 << word_bits
    out: list[int] = []
    for v in row:
        if not 0 <= int(v) < bound:
            raise ValueError(f"registered leaf {v} does not fit {word_bits} bits")
        out += [(int(v) >> (16 * i)) & 0xFFFF for i in range(k)]
    return out + [0] * (-len(out) % BLOCK if known_schema(schema) == SCHEMA_HM96_SHA512_ROW else 0)


def row_digest(row: Sequence[int], word_bits: int = WORD_BITS, schema: str = SCHEMA_HM96_SHA512_ROW) -> bytes:
    """The row's `x`, as the statement hashes the port's row: `sha512/row/v1` over its u16 words (`row_words`), role x;
    under `SCHEMA_HM96_SHA512_ROW_V2` or `SCHEMA_HM96_SHA512_ROW_SEG`, `sha512/row/v2` or `sha512/row-seg/v1` over its
    leaves' bit row."""
    if known_schema(schema) == SCHEMA_HM96_SHA512_ROW:
        return sha512_row_digest(row_words(row, word_bits), 16, ROLE_X)
    data, n = bit_row((row_words(row, word_bits, schema), 16))
    return sha512_row_v2_digest(data, n) if schema == SCHEMA_HM96_SHA512_ROW_V2 else sha512_row_seg_digest(data, n)


def leaf(dom: CommitmentDomain, position: int, bc: bytes, schema: str = SCHEMA_HM96_SHA512_ROW) -> bytes:
    return row_leaf(dom, position, _INST.tree_leaf(_INST.default_key, bc), known_schema(schema))


def supplied_salts(program: str, value: str, n: int) -> list[bytes]:
    """Row i's public salt of a supplied value: 192 bytes of SHA-512 in counter mode (`SHA-512(m ‖ u32be c)` for c < 3, each
    output whole) over `SUPPLIED_SALT`, the program's SHA-512 hex, the value's name and i (u32 little-endian), NUL between
    them. They hide nothing the record doesn't show: the reads need only the commitment's binding."""
    head = SUPPLIED_SALT.encode() + b"\0" + program.encode() + b"\0" + value.encode() + b"\0"
    return [b"".join(hashlib.sha512(head + i.to_bytes(4, "little") + c.to_bytes(4, "big")).digest()
                     for c in range(_INST.salt_bytes // 64)) for i in range(n)]


__all__ = ["BC_BYTES", "BLOCK", "ENTRY_KEYS", "OWNER", "PRODUCED_KEY", "READS_FROM", "REGISTERED_DOMAIN", "SCHEMAS",
           "SESSION_KEY", "SUPPLIED_SALT", "WIDTHS", "WIDTH_KEY", "WINDOW_KEY", "WORD_BITS", "domain", "known_schema", "leaf",
           "row_digest", "row_words", "supplied_salts", "width_entry", "window_root"]
