"""The warden's cells as the rows a proof unit reads (design v3: the transfer gates are the cells' bits). Each
link-window's cells are committed one `hm96-sha512/row/v2` row a cell, each under a hiding salt, in a frame-v3-sha512
tree whose root the warden signs with the certifier's signer (`certificate.Issuer.sign`, LMS n = 64), and the signed
head names the root. A unit opens a cell as its output port (the sender's) or its input port (the receiver's) by the
Lean verifier's registered reads (`flock-verify --registered`, `Flock.Registered.checkPort`): the port's table row is
the unit's row committed under the cell's salt, and it must climb its path to the signed root at the position the
verifier reads for the unit. No gate of the unit hashes the cell, and the verifier receives no row and no salt.

**The rows** (`window_rows`). A link-window of the warden's grid (`WardenParams.grid`: T buckets of S = lanes·r slots of
B-byte cells) is N = T·S rows: row b·S + j is slot j of the window's bucket b, in the grid's order (a bucket's cells in
ordinal order, a connection's r together), and a slot no connection holds is B zero bytes. Padding cells and empty slots
are rows, so N is the schedule's, whatever the load. A row's 8·B bits are the cell's bytes, bit 8k + i of the row bit i
of byte k: the `hm96-sha512/row/v2` bit row of its B/2 little-endian u16 words (`words`), which the unit's port carries.

**The commitment** is a registered value's (`one_stage.registered`), under the warden's own domain:

    x_k    = the sha512/row/v2 digest of row k's 8·B bits
    b‖c_k  = hm96.SHA512.commit_string(default_key, x_k, salt_k)
    leaf_k = row_leaf(domain, k, hm96.SHA512.tree_leaf(default_key, b‖c_k), "hm96-sha512/row/v2")
    root   = the frame-v3-sha512 root of the N leaves (`tree_levels`)
    domain = CommitmentDomain(identity_digest_sha512(DOMAIN_TAG, {parameters, session, link, direction, window,
             cell_bytes, rows, row_schema}), -1, RangeIndexedDomain(0, N), hash="sha512")

so a root opens under one link-window of one session's schedule alone. salt_k is 192 bytes the warden draws from the
operating system for each row, never for another. The opening (`Cells`: the rows and their salts) is witness, which
the warden hands to the prover alone.

**The signed head** (`CellsHead`) is public, as a hiding root is:

    H = u16be len(link) ‖ link ‖ u8 direction ‖ u64be window ‖ parameters (64 bytes) ‖ u32be B ‖ u32be N
      ‖ domain (64 bytes) ‖ root (64 bytes)
    cells digest = tagged_sha512(CELLS_TAG, H)

and its certificate is `Issuer.sign`'s of the cells digest: the certifier's message for the link-window in its session
(`certificate.certificate_message`). The wire record's certificate of the same link-window signs its salted record
digest instead, so neither verifies as the other (each would need a SHA-512 preimage).

**The registration** (`registration`) is the head and its certificate as one public JSON object:

    {"format": REGISTRATION, "value", "link", "direction", "window", "parameters", "cell_bytes", "rows", "words",
     "word_bits", "row_schema", "domain", "root", "layout": {"buckets", "slots", "first_bucket"},
     "certificate": {"key", "digest", "signature"}}

`value` is the registered value's name (`<link>/<direction>/<window>`), `words` and `word_bits` a row's B/2 leaves of 16
bits, and `layout` the rows' order: row (t − first_bucket)·slots + j is slot j of bucket t. `check_registration` is the
verifier's read of it: the head is the link-window's, its domain, row count, row format and layout are the ones the
schedule's parameters give, and its certificate verifies under the warden's key in the session over the head's digest.
`port` then gives the verifier's own `--registered` port (`verity/registered-reads/v1`): the value, root, leaves, words,
word bits, domain and row schema, and `reads`, the row each unit of the population opens.

**Which unit opens which row** is the wiring: connection q's rows (`connection_rows`, its cells in order) depend on which
connections were live in each bucket, so in the design they are private, proved by the wiring's permutation argument.
Tonight they are opened to the verifier as the units' wiring is (the named gap), and `reads` is that opening.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512, tagged_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest
from verity.core.protocols.certifier.active import HEADER
from verity.core.protocols.certifier.certificate import (DIGEST_BYTES, Certificate, Direction, Issuer, PublicKey, _text,
                                                         _u)
from verity.core.protocols.certifier.grid import GridParams, Record
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.core.protocols.sampled_proofs.one_stage import registered as G

DOMAIN_TAG = "verity-network-warden/cells-domain/v1"
CELLS_TAG = "verity-network-warden/cells/v1"
REGISTRATION = "verity-network-warden/registration/v1"
WORD_BITS = 16
OWNER = -1
_INST = hm96.SHA512
SALT_BYTES = _INST.salt_bytes


def rows_of(grid: GridParams) -> int:
    """N: a link-window's rows, every slot of its grid."""
    return grid.buckets * grid.allocation


def window_rows(record: Record, grid: GridParams) -> list[bytes]:
    """The link-window's rows: every slot of its grid, bucket by bucket, a slot no connection holds B zero bytes."""
    if len(record.grid) != grid.buckets or any(len(row) != grid.allocation for row in record.grid):
        raise ValueError(f"a window has {grid.buckets} buckets of {grid.allocation} slots")
    empty = bytes(grid.frame_bytes)
    return [empty if f is None else f for row in record.grid for f in row]


def connection_rows(record: Record, grid: GridParams) -> dict[int, list[int]]:
    """Each connection's rows in the link-window, its cells in order (its padding included), by the frames' ordinals:
    the wiring a unit's reads open. Read off the grid, not the rows: connection 0's padding cell is B zero bytes, as an
    empty slot is."""
    out: dict[int, list[int]] = {}
    for b, row in enumerate(record.grid):
        for j, f in enumerate(row):
            if f is not None:
                out.setdefault(HEADER.unpack_from(f)[1], []).append(b * grid.allocation + j)
    return out


def words(row: bytes) -> list[int]:
    """A row as its port's u16 words, little-endian, whose bit row is the cell's bytes."""
    if len(row) % 2:
        raise ValueError(f"a row of {len(row)} bytes is no whole number of u16 words")
    return [int.from_bytes(row[i:i + 2], "little") for i in range(0, len(row), 2)]


def domain(parameters: bytes, session: bytes, link: str, direction: Direction, window: int, cell_bytes: int,
           rows: int) -> CommitmentDomain:
    """The frame-v3-sha512 domain of one link-window's rows in one session of one schedule."""
    binding = identity_digest_sha512(DOMAIN_TAG, {
        "parameters": parameters.hex(), "session": session.hex(), "link": link, "direction": direction.value,
        "window": window, "cell_bytes": cell_bytes, "rows": rows, "row_schema": SCHEMA})
    return CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, rows), hash="sha512")


def value_name(link: str, direction: Direction, window: int) -> str:
    return f"{link}/{direction.value}/{window}"


def bc(row: bytes, salt: bytes) -> bytes:
    """A row's hm96 commitment string `b ‖ c` under its salt: what the statement's table row of it is."""
    return _INST.commit_string(_INST.default_key, sha512_row_v2_digest(row, 8 * len(row)), salt)


def opens(dom: CommitmentDomain, root: bytes, position: int, bc_: bytes, path: bytes) -> bool:
    """`bc_`, as the row at `position`, climbs `path` to `root` under the verifier's own `dom`: what the Lean verifier's
    registered read checks of each table row (`registered.verify_read`)."""
    return G.verify_read(dom, root, position, bc_, path, SCHEMA)


@dataclass(frozen=True, slots=True)
class CellsHead:
    """What the warden signs for a link-window's rows: public, since its root is hiding."""

    link: str
    direction: Direction
    window: int
    parameters: bytes
    cell_bytes: int
    rows: int
    domain: bytes
    root: bytes

    def __post_init__(self) -> None:
        if any(len(x) != DIGEST_BYTES for x in (self.parameters, self.domain, self.root)):
            raise ValueError("the parameters digest, the domain and the root are 64 bytes")

    @property
    def value(self) -> str:
        return value_name(self.link, self.direction, self.window)

    def encode(self) -> bytes:
        return b"".join([_text(self.link, "link"), _u(self.direction.code, 1, "direction"), _u(self.window, 8, "window"),
                         self.parameters, _u(self.cell_bytes, 4, "B"), _u(self.rows, 4, "N"), self.domain, self.root])

    def digest(self) -> bytes:
        return tagged_sha512(CELLS_TAG, self.encode())


@dataclass(frozen=True)
class Cells:
    """The warden's opening of one link-window's rows, witness: the rows and their salts, and what they commit to."""

    link: str
    direction: Direction
    window: int
    parameters: bytes
    session: bytes
    rows: tuple[bytes, ...]
    salts: tuple[bytes, ...]

    @classmethod
    def commit(cls, link: str, direction: Direction, window: int, rows: Sequence[bytes], parameters: bytes,
               session: bytes, salts: Sequence[bytes] | None = None) -> Cells:
        """`salts`: one a row, drawn once (default: fresh from the operating system) and never for another row."""
        rows = tuple(bytes(r) for r in rows)
        if not rows or len(rows[0]) < 2 or len(rows[0]) % 2 or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError("a link-window is at least one row, every row the same whole number of u16 words")
        salts = tuple(_INST.fresh_salt() for _ in rows) if salts is None else tuple(bytes(s) for s in salts)
        if len(salts) != len(rows) or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"one {_INST.salt_bytes}-byte salt a row")
        if len(set(salts)) != len(salts):
            raise ValueError("a salt is never reused: equal salts on two rows reveal the XOR of their digests")
        return cls(link, direction, window, bytes(parameters), bytes(session), rows, salts)

    @property
    def cell_bytes(self) -> int:
        return len(self.rows[0])

    @property
    def value(self) -> str:
        return value_name(self.link, self.direction, self.window)

    @cached_property
    def domain(self) -> CommitmentDomain:
        return domain(self.parameters, self.session, self.link, self.direction, self.window, self.cell_bytes,
                      len(self.rows))

    @cached_property
    def bcs(self) -> tuple[bytes, ...]:
        return tuple(bc(r, s) for r, s in zip(self.rows, self.salts))

    @cached_property
    def levels(self) -> tuple[tuple[bytes, ...], ...]:
        dom = self.domain
        return tree_levels(dom, [row_leaf(dom, k, _INST.tree_leaf(_INST.default_key, x), SCHEMA)
                                 for k, x in enumerate(self.bcs)])

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def head(self) -> CellsHead:
        return CellsHead(self.link, self.direction, self.window, self.parameters, self.cell_bytes, len(self.rows),
                         self.domain.domain_id, self.root)

    def path(self, position: int) -> bytes:
        """Row `position`'s siblings, leaf level first, 64 bytes each."""
        out, i = b"", position
        for level in self.levels[:-1]:
            out += level[i ^ 1]
            i >>= 1
        return out

    def statement(self, positions: Sequence[int]) -> tuple[list[list[int]], list[bytes], dict[str, Any]]:
        """What a statement whose port reads the rows at `positions` commits: those rows as u16 words, their salts (the
        prover's), and its header's `registered` entry, `{"value", "positions", "paths"}`."""
        if any(not 0 <= p < len(self.rows) for p in positions):
            raise ValueError(f"a position is outside the link-window's {len(self.rows)} rows")
        return ([words(self.rows[p]) for p in positions], [self.salts[p] for p in positions],
                {"value": self.value, "positions": [int(p) for p in positions],
                 "paths": [self.path(p).hex() for p in positions]})


def attest(issuer: Issuer, record: Record, grid: GridParams, parameters: bytes, direction: Direction,
           salts: Sequence[bytes] | None = None) -> tuple[Cells, Certificate | None]:
    """The link-window's rows committed and their head signed by `issuer` (None when its signer refuses)."""
    cells = Cells.commit(record.link, direction, record.window, window_rows(record, grid), parameters, issuer.session,
                         salts)
    return cells, issuer.sign(record.link, direction, record.window, cells.head.digest())


def registration(head: CellsHead, certificate: Certificate, grid: GridParams) -> dict[str, Any]:
    """The public record a unit's registered port reads: the head, the rows' format and order, and the certificate."""
    return {"format": REGISTRATION, "value": head.value, "link": head.link, "direction": head.direction.value,
            "window": head.window, "parameters": head.parameters.hex(), "cell_bytes": head.cell_bytes,
            "rows": head.rows, "words": head.cell_bytes // 2, "word_bits": WORD_BITS, "row_schema": SCHEMA,
            "domain": head.domain.hex(), "root": head.root.hex(),
            "layout": {"buckets": grid.buckets, "slots": grid.allocation, "first_bucket": head.window * grid.buckets},
            "certificate": {"key": certificate.key_id.hex(), "digest": certificate.digest.hex(),
                            "signature": certificate.signature.hex()}}


def check_registration(doc: Mapping[str, Any], key: PublicKey, session: bytes, grid: GridParams, parameters: bytes,
                       link: str, direction: Direction, window: int) -> tuple[CellsHead, CommitmentDomain]:
    """The verifier's read of a registration for the link-window it expects, under the warden's key and session and the
    schedule's grid and parameters digest: the head and the domain the verifier derived, or ValueError naming what
    fails."""
    n, B = rows_of(grid), grid.frame_bytes
    dom = domain(parameters, session, link, direction, window, B, n)
    want = {"format": REGISTRATION, "value": value_name(link, direction, window), "link": link,
            "direction": direction.value, "window": window, "parameters": parameters.hex(), "cell_bytes": B,
            "rows": n, "words": B // 2, "word_bits": WORD_BITS, "row_schema": SCHEMA, "domain": dom.domain_id.hex(),
            "layout": {"buckets": grid.buckets, "slots": grid.allocation, "first_bucket": window * grid.buckets}}
    for k, v in want.items():
        if doc.get(k) != v:
            raise ValueError(f"the registration's {k} is {doc.get(k)!r}, and the schedule's {v!r}")
    if set(doc) != set(want) | {"root", "certificate"}:
        raise ValueError(f"a registration holds the keys {sorted(set(want) | {'root', 'certificate'})} and no other")
    try:
        head = CellsHead(link, direction, window, parameters, B, n, dom.domain_id, bytes.fromhex(doc["root"]))
        c = doc["certificate"]
        cert = Certificate(bytes.fromhex(c["key"]), link, direction, window, bytes.fromhex(c["digest"]),
                           bytes.fromhex(c["signature"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the registration's root or certificate doesn't read: {e}") from None
    if cert.digest != head.digest():
        raise ValueError("the registration's certificate doesn't sign its head")
    if not cert.verify(key, session):
        raise ValueError("the registration's certificate doesn't verify under the warden's key in the session")
    return head, dom


def port(head: CellsHead, reads: Sequence[int | None], reads_from: int = 0) -> dict[str, Any]:
    """The verifier's own `--registered` port of a unit population that opens the head's rows: unit `reads_from + i`
    opens row `reads[i]` (None: no row)."""
    e = {"leaves": head.rows, "words": head.cell_bytes // 2, "word_bits": WORD_BITS, "domain": head.domain.hex()}
    return produced.port(head.value, e, head.root.hex(), [None if r is None else int(r) for r in reads], SCHEMA) | \
        ({G.READS_FROM: int(reads_from)} if reads_from else {})


def reads_file(ports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The verifier's own `--registered` file (`verity/registered-reads/v1`) of its ports, `{circuit port: port}`."""
    return {"format": G.READS_V1, "ports": {p: dict(e) for p, e in ports.items()}}
