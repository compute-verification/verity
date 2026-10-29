"""The network certifier's transfer rows: a host's load, each value of the weights it loads once a session, committed
by the network certifier on the host's uplink as it crosses, never by the prover, and signed with its key. The verifier
pins each load's root from that certificate alone (`blocks.pins`), so every unit's row of a load is a registered read
opened against the certifier's signed root by the Lean verifier's registered reads (`flock-verify --registered`,
`Flock.Registered.checkPort`), which don't change. Every other row into the host is the scan's state
(`ingress_scan.SCAN_STATE`), which the scan places inside the zero-knowledge proof, and its egress the scan's egress
side's (`egress`): no request, response or value one host sends another has a head of its own.

**The commitment** is a registered value's (`one_stage.registered`), as the prover would commit it, under the
certifier's own salts. The value is `rows` rows of `words` leaves of `word_bits` bits (16, or 32: the reading
parameter's width, `registered.width`), row r its leaves in the order the units read them, each row the bit row a unit's
port carries in its `schema` (`SCHEMAS`: ``hm96-sha512/row/v2``, or ``hm96-sha512/row-seg/v1`` for a row a port reads
in segments, as `EmbeddingRead_v1` reads a token's embedding):

    b‖c_r  = hm96.SHA512.commit_string(default_key, registered.row_digest(row r, word_bits, schema), salt_r)
    leaf_r = row_leaf(D, r, hm96.SHA512.tree_leaf(default_key, b‖c_r), schema)
    root   = the frame-v3-sha512 root of the rows' leaves under D = registered.domain(program, value, words, rows, word_bits)

D leaves the window out, as a value the verifier holds across windows does, so the root is the one `registration.check`
holds a pinned value's entry to (`pinned`) and the one the verifier's `--registered` port opens; the head and its
certificate bind the link and the execution phase. `program` is the circuit's program digest (``circuit.json``'s
``program``), public. salt_r is 192 bytes the certifier draws from the operating system for row r and no other row. The
salts are witness, which the certifier hands to the prover alone (`Transfer.witness`); the rows are what the host
received, which the prover holds.

**The signed head** (`TransferHead`) is public, as a hiding root is:

    H = u16be len(link) ‖ link ‖ u8 direction ‖ u64be window ‖ u16be len(kind) ‖ kind ‖ u16be len(value) ‖ value
      ‖ program (64 bytes) ‖ u32be words ‖ u32be rows ‖ u8 word_bits ‖ u16be len(schema) ‖ schema ‖ domain (64 bytes)
      ‖ root (64 bytes)
    transfer digest = tagged_sha512(TRANSFER_TAG, H)

and its certificate is `Issuer.sign`'s of the transfer digest: the certifier's message for the link and execution phase
in its session (`certificate.certificate_message`), so it verifies for that link, direction and execution phase alone.
The network certifier's cells digest and its record digest of an execution phase's transfer blocks are under other tags,
so none verifies as another.

**The kind** (`KINDS`) is ``load``, ingress: the registered parameters' leaves the load connection carries, once a
session. The head keeps its kind and direction fields, so a head signed for anything else never reads as a load's.

**The registration** (`registration`) is the head and its certificate as one public JSON object:

    {"format": REGISTRATION, "link", "direction", "window", "kind", "value", "program", "words", "rows", "word_bits",
     "row_schema", "domain", "root", "certificate": {"key", "digest", "signature"}}

`check_registration` is the verifier's read of it for the transfer it expects: every field the verifier's own, the
domain the one it derives, and the certificate over the head's digest verifying under the certifier's key in its session
for the link and execution phase it expects. `port` is the verifier's own `--registered` port of the value at the head's
root.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.commitments.merkle import CommitmentDomain
from verity.core.protocols.certifier.certificate import DIGEST_BYTES, Certificate, Direction, Issuer, PublicKey, _text, _u
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.core.protocols.sampled_proofs.one_stage import registered as G

TRANSFER_TAG = "verity-network-certifier/transfer/v1"
REGISTRATION = "verity-network-certifier/transfer-registration/v1"
#: each kind's directions
KINDS = {"load": (Direction.INGRESS,)}
WORD_BITS = 16
WIDTHS = G.WIDTHS
SCHEMA = G.SCHEMA_V2
#: the bit-row formats a unit's port carries
SCHEMAS = (G.SCHEMA_V2, G.SCHEMA_SEG)
_INST = hm96.SHA512
SALT_BYTES = _INST.salt_bytes
_HEX64 = re.compile("[0-9a-f]{128}")


def _kind(kind: str, direction: Direction) -> None:
    if direction not in KINDS.get(kind, ()):
        allowed = ", ".join(f"{k} {'/'.join(d.value for d in ds)}" for k, ds in KINDS.items())
        raise ValueError(f"a transfer is a {allowed}, not {kind!r} {getattr(direction, 'value', direction)!r}")


def _program(program: str) -> str:
    if not (isinstance(program, str) and _HEX64.fullmatch(program)):
        raise ValueError(f"a program digest is lowercase hex of 64 bytes, not {program!r}")
    return program


def _format(word_bits: int, schema: str) -> None:
    if type(word_bits) is not int or word_bits not in WIDTHS:
        raise ValueError(f"a transfer value's leaves are {' or '.join(map(str, WIDTHS))} bits, not {word_bits!r}")
    if schema not in SCHEMAS:
        raise ValueError(f"a transfer value's rows are {' or '.join(SCHEMAS)} rows, not {schema!r}")


def domain(program: str, value: str, words: int, rows: int, word_bits: int = WORD_BITS) -> CommitmentDomain:
    """The frame-v3-sha512 domain of a transfer value's rows: a registered value's held across windows
    (`registered.domain` with no window), the one a pinned value's entry is under."""
    _format(word_bits, SCHEMA)
    return G.domain(_program(program), value, words, rows, word_bits)


def bc(words: Sequence[int], salt: bytes, word_bits: int = WORD_BITS, schema: str = SCHEMA) -> bytes:
    """A row's hm96 commitment string `b ‖ c` under its salt: what a statement's table row of it is."""
    _format(word_bits, schema)
    return _INST.commit_string(_INST.default_key, G.row_digest(words, word_bits, schema), salt)


@dataclass(frozen=True, slots=True)
class TransferHead:
    """What the certifier signs for a transfer value: public, since its root is hiding."""

    link: str
    direction: Direction
    window: int
    kind: str
    value: str
    program: bytes
    words: int
    rows: int
    domain: bytes
    root: bytes
    word_bits: int = WORD_BITS
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _kind(self.kind, self.direction)
        _format(self.word_bits, self.schema)
        if any(len(x) != DIGEST_BYTES for x in (self.program, self.domain, self.root)):
            raise ValueError("the program digest, the domain and the root are 64 bytes")

    def encode(self) -> bytes:
        return b"".join([_text(self.link, "link"), _u(self.direction.code, 1, "direction"), _u(self.window, 8, "window"),
                         _text(self.kind, "kind"), _text(self.value, "value"), self.program, _u(self.words, 4, "words"),
                         _u(self.rows, 4, "rows"), _u(self.word_bits, 1, "word_bits"), _text(self.schema, "schema"),
                         self.domain, self.root])

    def digest(self) -> bytes:
        return tagged_sha512(TRANSFER_TAG, self.encode())


@dataclass(frozen=True)
class Transfer:
    """The certifier's opening of one transfer value, witness: the rows and their salts (`committed`), and the link,
    execution phase and kind it crossed in."""

    link: str
    direction: Direction
    window: int
    kind: str
    committed: G.Registered

    @classmethod
    def commit(cls, link: str, direction: Direction, window: int, kind: str, program: str, value: str,
               rows: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None, word_bits: int = WORD_BITS,
               schema: str = SCHEMA) -> Transfer:
        """`rows`: the value's rows of `word_bits`-bit leaves, all of one width. `salts`: one a row, drawn once (default:
        fresh from the operating system) and never for another row."""
        _kind(kind, direction)
        _format(word_bits, schema)
        return cls(link, direction, window, kind, G.Registered.commit(_program(program), value, rows, salts, word_bits,
                                                                      schema))

    @property
    def value(self) -> str:
        return self.committed.value

    @property
    def root(self) -> bytes:
        return self.committed.root

    @property
    def salts(self) -> tuple[bytes, ...]:
        return self.committed.salts

    @cached_property
    def head(self) -> TransferHead:
        c = self.committed
        return TransferHead(self.link, self.direction, self.window, self.kind, c.value, bytes.fromhex(c.program), c.words,
                            c.leaves, c.domain.domain_id, c.root, c.word_bits, c.schema)

    @property
    def witness(self) -> bytes:
        """What the certifier hands the prover beside the rows: their salts, in order (`read_witness`)."""
        return b"".join(self.salts)

    def statement(self, positions: Sequence[int]) -> tuple[list[tuple[int, ...]], list[bytes], dict[str, Any]]:
        """What a statement whose port reads the rows at `positions` commits: those rows, their salts (the prover's) and
        its header's `registered` entry, `{"value", "positions", "paths"}`."""
        return self.committed.statement(positions)


def read_witness(blob: bytes) -> list[bytes]:
    """`Transfer.witness` read back: the rows' salts."""
    if not blob or len(blob) % SALT_BYTES:
        raise ValueError(f"a transfer's witness is {SALT_BYTES} bytes a row")
    return [blob[i:i + SALT_BYTES] for i in range(0, len(blob), SALT_BYTES)]


def attest(issuer: Issuer, link: str, direction: Direction, window: int, kind: str, program: str, value: str,
           rows: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None, word_bits: int = WORD_BITS,
           schema: str = SCHEMA) -> tuple[Transfer, Certificate | None]:
    """The value's rows committed and their head signed by `issuer` for the link and execution phase (None when its
    signer refuses)."""
    t = Transfer.commit(link, direction, window, kind, program, value, rows, salts, word_bits, schema)
    return t, issuer.sign(link, direction, window, t.head.digest())


def registration(head: TransferHead, certificate: Certificate) -> dict[str, Any]:
    """The public record of a transfer value: its head, its rows' format, and the certificate."""
    return {"format": REGISTRATION, "link": head.link, "direction": head.direction.value, "window": head.window,
            "kind": head.kind, "value": head.value, "program": head.program.hex(), "words": head.words, "rows": head.rows,
            "word_bits": head.word_bits, "row_schema": head.schema, "domain": head.domain.hex(), "root": head.root.hex(),
            "certificate": {"key": certificate.key_id.hex(), "digest": certificate.digest.hex(),
                            "signature": certificate.signature.hex()}}


def check_registration(doc: Any, key: PublicKey, session: bytes, *, link: str, direction: Direction, window: int,
                       kind: str, value: str, program: str, words: int, rows: int, word_bits: int = WORD_BITS,
                       schema: str = SCHEMA) -> tuple[TransferHead, CommitmentDomain]:
    """The verifier's read of a transfer registration for the transfer it expects (its link and execution phase, kind,
    value, the circuit's program digest and the value's shape and row format, all from its own record), under the
    network certifier's key and session: the head and the domain the verifier derived, or ValueError naming what
    fails."""
    _kind(kind, direction)
    _format(word_bits, schema)
    dom = domain(program, value, words, rows, word_bits)
    want = {"format": REGISTRATION, "link": link, "direction": direction.value, "window": window, "kind": kind,
            "value": value, "program": program, "words": words, "rows": rows, "word_bits": word_bits, "row_schema": schema,
            "domain": dom.domain_id.hex()}
    if not isinstance(doc, Mapping):
        raise ValueError(f"a transfer registration is a JSON object, not {type(doc).__name__}")
    for k, v in want.items():
        if doc.get(k) != v or type(doc.get(k)) is not type(v):
            raise ValueError(f"the transfer registration's {k} is {doc.get(k)!r}, and the verifier's {v!r}")
    if set(doc) != set(want) | {"root", "certificate"}:
        raise ValueError(f"a transfer registration holds the keys {sorted(set(want) | {'root', 'certificate'})} and no other")
    try:
        if not (isinstance(doc["root"], str) and _HEX64.fullmatch(doc["root"])):
            raise ValueError(f"its root is lowercase hex of 64 bytes, not {doc['root']!r}")
        head = TransferHead(link, direction, window, kind, value, bytes.fromhex(program), words, rows, dom.domain_id,
                            bytes.fromhex(doc["root"]), word_bits, schema)
        c = doc["certificate"]
        if not isinstance(c, Mapping) or set(c) != {"key", "digest", "signature"}:
            raise ValueError("its certificate is {key, digest, signature}")
        cert = Certificate(bytes.fromhex(c["key"]), link, direction, window, bytes.fromhex(c["digest"]),
                           bytes.fromhex(c["signature"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the transfer registration's root or certificate doesn't read: {e}") from None
    if cert.digest != head.digest():
        raise ValueError("the transfer registration's certificate doesn't sign its head")
    if not cert.verify(key, session):
        raise ValueError(f"the transfer registration's certificate doesn't verify under the network certifier's key in its "
                         f"session for {link} {direction.value} window {window}")
    return head, dom


def port(head: TransferHead, reads: Sequence[int | None], reads_from: int = 0) -> dict[str, Any]:
    """The verifier's own `--registered` port (`verity/registered-reads/v1`) of a unit population that reads the head's
    value: unit `reads_from + i` opens row `reads[i]` (None: it reads none) at the certifier's signed root."""
    if any(r is not None and not (type(r) is int and 0 <= r < head.rows) for r in reads):
        raise ValueError(f"a read is a row of the value's {head.rows}, or None")
    e = {"leaves": head.rows, "words": head.words, "word_bits": head.word_bits, "domain": head.domain.hex()}
    return produced.port(head.value, e, head.root.hex(), list(reads), head.schema) | \
        ({G.READS_FROM: int(reads_from)} if reads_from else {})
