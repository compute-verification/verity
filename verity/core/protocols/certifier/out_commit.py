"""The network certifier's output value: the output it relayed, committed by the certifier as a registered value
(`one_stage.registered`) and its head signed with its key, never by the prover. The verifier pins the value's root from
that certificate alone (`check_out`), so every output unit's row is a registered read of it, opened against
the certifier's signed root by the Lean verifier's registered reads (`flock-verify --registered`), not compared in the
clear.

**The commitment** is the reading program's registered value `value` (``out`` by default), as the prover would commit
it, under the certifier's salts: `rows` rows of `words` leaves of `word_bits` bits in the row format `schema` (by
default `hm96-sha512/row/v2`, a unit's bit-row port), committed as `Registered.commit` commits them, under

    D = registered.domain(program, value, words, rows, word_bits)

the domain `registered_entries` derives for a value the verifier pins: `registration.check` holds its pinned values
across windows, so D leaves the window's open root out. `program` is the reading program's digest
(`partition.program_digest`), public. salt_r is 192 bytes the certifier draws from the operating system for row r and no
other row. The salts are witness, which the certifier hands to the prover alone (`CertifiedOut.salts`, `witness`); the
rows are what the server sent, which the prover holds.

**The signed head** (`OutHead`) is public, as a hiding root is:

    H = u16be len(value) ‖ value ‖ program (64 bytes) ‖ u32be words ‖ u32be rows ‖ u16be word_bits
      ‖ u16be len(schema) ‖ schema ‖ domain (64 bytes) ‖ root (64 bytes)
    out digest = tagged_sha512(OUT_TAG, H)

Its certificate is `Issuer.sign`'s of the out digest for the egress link-window, in the certifier's session
(`certificate.certificate_message`). D is the same in every window, so the certificate is what keeps one window's root
out of another's: `check_out` takes the window it expects. A record's, a clock's and the warden cells' digests are
under other tags, so none verifies as another.

**The record** (`CertifiedOut.public`) is the head and its certificate as one public JSON object, what the auditor
receives:

    {"format": RECORD, "link", "direction", "window", "session", "value", "program", "words", "rows", "word_bits",
     "row_schema", "domain", "root", "certificate": {"key", "digest", "signature"}}

`check_out` is the verifier's read of it: the root to pin (`registration.check`'s `pinned`), or `OutRefused` naming the
check that fails (`REFUSALS`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.program import Program
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.commitments.merkle import CommitmentDomain
from verity.core.protocols.certifier.certificate import (DIGEST_BYTES, Certificate, Direction, Issuer, PublicKey,
                                                         Signer, SignerRefused, _text, _u)
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

OUT_TAG = "verity-network-certifier/out/v1"
RECORD = "verity-network-certifier/out-record/v1"
VALUE = "out"
#: the link a certificate names when the caller names none
LINK = "out"
DIRECTION = Direction.EGRESS
SALT_BYTES = hm96.SHA512.salt_bytes
#: each refusal's name, and what it means
REFUSALS = {
    "out-record": "the record isn't an out record: not a JSON object of exactly its keys, each of its type",
    "out-key": "the certificate is another key's",
    "out-session": "the record is of another certifier session",
    "out-link": "the certificate is of another link or direction",
    "out-window": "the certificate is of another window",
    "out-domain": "the record's domain isn't the verifier's, or isn't its own fields'",
    "out-digest": "the certificate doesn't sign the record's head",
    "out-signature": "the signature doesn't verify under the key in the session for the link-window",
}
_HEX64 = re.compile("[0-9a-f]{128}")
_HEX = re.compile("(?:[0-9a-f]{2})+")


class OutRefused(ValueError):
    """`check_out`'s refusal: `name` is one of `REFUSALS`."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name = name


def _program(program: str | Program) -> str:
    if isinstance(program, Program):
        return P.program_digest(program)
    if not (isinstance(program, str) and _HEX64.fullmatch(program)):
        raise ValueError(f"a program digest is lowercase hex of 64 bytes, not {program!r}")
    return program


@dataclass(frozen=True, slots=True)
class OutHead:
    """What the certifier signs for the output value: public, since its root is hiding."""

    value: str
    program: bytes
    words: int
    rows: int
    word_bits: int
    schema: str
    domain: bytes
    root: bytes

    def __post_init__(self) -> None:
        if any(type(x) is not bytes or len(x) != DIGEST_BYTES for x in (self.program, self.domain, self.root)):
            raise ValueError("the program digest, the domain and the root are 64 bytes")

    def encode(self) -> bytes:
        return b"".join([_text(self.value, "value"), self.program, _u(self.words, 4, "words"), _u(self.rows, 4, "rows"),
                         _u(self.word_bits, 2, "word_bits"), _text(self.schema, "schema"), self.domain, self.root])

    def digest(self) -> bytes:
        return tagged_sha512(OUT_TAG, self.encode())


@dataclass(frozen=True)
class CertifiedOut:
    """The certifier's output value: the public part (`head`, `certificate` and `session`, read together as `public`,
    with `root` and `domain` its hex), and the witness (`committed`: the rows and their salts, `salts`), which goes to
    the prover alone."""

    head: OutHead
    certificate: Certificate
    session: bytes
    committed: G.Registered = field(repr=False, compare=False)

    @property
    def root(self) -> str:
        return self.head.root.hex()

    @property
    def domain(self) -> str:
        return self.head.domain.hex()

    @property
    def public(self) -> dict[str, Any]:
        """The record the auditor receives and `check_out` reads: the head and its certificate, none of the witness."""
        h, c = self.head, self.certificate
        return {"format": RECORD, "link": c.link, "direction": c.direction.value, "window": c.window,
                "session": self.session.hex(), "value": h.value, "program": h.program.hex(), "words": h.words,
                "rows": h.rows, "word_bits": h.word_bits, "row_schema": h.schema, "domain": h.domain.hex(),
                "root": h.root.hex(), "certificate": {"key": c.key_id.hex(), "digest": c.digest.hex(),
                                                      "signature": c.signature.hex()}}

    @property
    def salts(self) -> tuple[bytes, ...]:
        return self.committed.salts

    @property
    def witness(self) -> bytes:
        """What the certifier hands the prover beside the rows: their salts, in order (`read_witness`)."""
        return b"".join(self.salts)

    def statement(self, positions: Sequence[int]) -> tuple[list[tuple[int, ...]], list[bytes], dict[str, Any]]:
        """What a statement whose port reads the rows at `positions` commits: those rows, their salts (the prover's) and
        its header's `registered` entry, `{"value", "positions", "paths"}` (`Registered.statement`)."""
        return self.committed.statement(positions)


def read_witness(blob: bytes) -> list[bytes]:
    """`CertifiedOut.witness` read back: the rows' salts."""
    if not blob or len(blob) % SALT_BYTES:
        raise ValueError(f"an out value's witness is {SALT_BYTES} bytes a row")
    return [blob[i:i + SALT_BYTES] for i in range(0, len(blob), SALT_BYTES)]


def commit_out(rows: Sequence[Sequence[int]], *, program: str | Program, value: str = VALUE, words: int,
               word_bits: int = G.WORD_BITS, schema: str = G.SCHEMA_V2, key: Signer | Issuer, session: bytes,
               window: int, link: str = LINK, salts: Sequence[bytes] | None = None) -> CertifiedOut:
    """The output value the certifier relayed, committed under the domain the verifier derives for the reading program's
    pinned `value` and its head signed for the egress link-window `(link, window)` in certifier session `session`.

    `rows`: the value's rows, each `words` leaves of `word_bits` bits. `program`: the reading program, or its digest.
    `key`: the certifier's signer (which starts `session` at its first signature and refuses any other session), or
    its running `Issuer` in `session`. `salts`: one a row
    (default: fresh from the operating system; a test injects its own). Raises ValueError on rows of another width and
    `SignerRefused` when the signer refuses, so the window has no out certificate."""
    if not rows or any(len(r) != words for r in rows):
        raise ValueError(f"the out value is at least one row, each of {words} leaves")
    issuer = key if isinstance(key, Issuer) else Issuer(key, session)
    if issuer.session != session:
        raise ValueError("the issuer signs in another certifier session than `session`")
    committed = G.Registered.commit(_program(program), value, rows, salts, word_bits, schema)
    head = OutHead(value, bytes.fromhex(committed.program), committed.words, committed.leaves, committed.word_bits,
                   committed.schema, committed.domain.domain_id, committed.root)
    certificate = issuer.sign(link, DIRECTION, window, head.digest())
    if certificate is None:
        raise SignerRefused(f"the certifier's signer refused the out certificate of {link} window {window}")
    return CertifiedOut(head, certificate, session, committed)


_RECORD_TYPES = {"format": str, "link": str, "direction": str, "window": int, "session": str, "value": str,
                 "program": str, "words": int, "rows": int, "word_bits": int, "row_schema": str, "domain": str,
                 "root": str, "certificate": Mapping}
_CERT_KEYS = ("key", "digest", "signature")


def _read(doc: Any) -> tuple[OutHead, Certificate, bytes]:
    if not isinstance(doc, Mapping):
        raise OutRefused("out-record", f"an out record is a JSON object (`CertifiedOut.public`), not a "
                                       f"{type(doc).__name__}")
    if set(doc) != set(_RECORD_TYPES):
        raise OutRefused("out-record", f"an out record holds the keys {sorted(_RECORD_TYPES)} and no other")
    for k, t in _RECORD_TYPES.items():
        if not isinstance(doc[k], t) or isinstance(doc[k], bool):
            raise OutRefused("out-record", f"its {k} is {doc[k]!r}, not a {t.__name__}")
    if doc["format"] != RECORD:
        raise OutRefused("out-record", f"its format is {doc['format']!r}, not {RECORD!r}")
    c = doc["certificate"]
    if set(c) != set(_CERT_KEYS) or not all(isinstance(c[k], str) and _HEX.fullmatch(c[k]) for k in _CERT_KEYS):
        raise OutRefused("out-record", f"its certificate is an object of the hex strings {list(_CERT_KEYS)} alone")
    if not all(_HEX64.fullmatch(doc[k]) for k in ("program", "domain", "root")) or not _HEX.fullmatch(doc["session"]):
        raise OutRefused("out-record", "its program, domain and root are lowercase hex of 64 bytes, its session hex")
    if doc["row_schema"] not in G.SCHEMAS:
        raise OutRefused("out-record", f"its row_schema is {doc['row_schema']!r}, not one of {list(G.SCHEMAS)}")
    try:
        direction = Direction(doc["direction"])
        head = OutHead(doc["value"], bytes.fromhex(doc["program"]), doc["words"], doc["rows"], doc["word_bits"],
                       doc["row_schema"], bytes.fromhex(doc["domain"]), bytes.fromhex(doc["root"]))
        head.encode()
        cert = Certificate(bytes.fromhex(c["key"]), doc["link"], direction, doc["window"], bytes.fromhex(c["digest"]),
                           bytes.fromhex(c["signature"]))
    except ValueError as e:
        raise OutRefused("out-record", f"it doesn't read: {e}") from None
    return head, cert, bytes.fromhex(doc["session"])


def check_out(certificate: Any, key: PublicKey, session: bytes, expected_domain: str | CommitmentDomain, *,
              window: int, link: str = LINK) -> str:
    """The verifier's read of the out record `certificate` (`CertifiedOut.public`) for the egress link-window
    `(link, window)` it expects, under the network certifier's `key` in certifier session `session`: the root to pin
    (lowercase hex, `registration.check`'s `pinned` form), or `OutRefused` naming the first check that fails.
    `expected_domain` is the domain the verifier derives for the value (`registered_entries(...)[value]["domain"]`, with
    the value among `fixed`, as `registration.check` takes a pinned value's), as hex or a `CommitmentDomain`."""
    want = expected_domain.domain_id.hex() if isinstance(expected_domain, CommitmentDomain) else expected_domain
    if not (isinstance(want, str) and _HEX64.fullmatch(want)):
        raise ValueError(f"the expected domain is lowercase hex of 64 bytes or a CommitmentDomain, not "
                         f"{expected_domain!r}")
    head, cert, named = _read(certificate)
    if cert.key_id != key.key_id:
        raise OutRefused("out-key", f"the certificate is key {cert.key_id.hex()[:16]}…'s, not the certifier's "
                                    f"{key.key_id.hex()[:16]}…")
    if named != session:
        raise OutRefused("out-session", f"the record is of session {named.hex()[:16]}…, not {session.hex()[:16]}…")
    if (cert.link, cert.direction) != (link, DIRECTION):
        raise OutRefused("out-link", f"the certificate is of {cert.link!r} {cert.direction.value}, not {link!r} "
                                     f"{DIRECTION.value}")
    if cert.window != window:
        raise OutRefused("out-window", f"the certificate is of window {cert.window}, not {window}")
    if head.domain.hex() != want:
        raise OutRefused("out-domain", f"the record's domain is {head.domain.hex()[:16]}…, and the verifier's "
                                       f"{want[:16]}…")
    try:
        own = G.domain(head.program.hex(), head.value, head.words, head.rows, head.word_bits).domain_id
    except ValueError as e:
        raise OutRefused("out-domain", f"the record's fields name no domain: {e}") from None
    if own != head.domain:
        raise OutRefused("out-domain", "the record's domain isn't the one its program, value, words, rows and "
                                       "word_bits name")
    if cert.digest != head.digest():
        raise OutRefused("out-digest", "the certificate's digest isn't the record's head's")
    if not cert.verify(key, session):
        raise OutRefused("out-signature", f"the signature doesn't verify under the certifier's key in the session for "
                                          f"{link!r} window {window}")
    return head.root.hex()


__all__ = ["DIRECTION", "LINK", "OUT_TAG", "RECORD", "REFUSALS", "SALT_BYTES", "VALUE", "CertifiedOut", "OutHead",
           "OutRefused", "check_out", "commit_out", "read_witness"]
