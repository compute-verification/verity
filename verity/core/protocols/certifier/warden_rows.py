"""The warden's cells as the rows a proof block reads (design v3: the transfer gates are the cells' bits). Each
link-window's cells are committed one `hm96-sha512/row/v2` row a bucket, each under a hiding salt, in a frame-v3-sha512
tree. The warden signs a head that names the root and a salted digest of the link-window's wiring with the certifier's
signer (`certificate.Issuer.sign`, LMS n = 64). A unit opens a cell's row as its output port (the sender's) or its input
port (the receiver's) by the Lean verifier's registered reads (`flock-verify --registered`,
`Flock.Registered.checkPort`): the port's table row is the unit's row committed under the row's salt, and it must climb
its path to the signed root at the row the verifier reads for the unit. The verifier derives that row from the unit's
(connection, cell) and the warden's wiring (`port`). No gate of the unit hashes the row, and the verifier receives no
row and no salt.

**The rows** (`window_rows`). A link-window w of the warden's schedule (`WardenParams`: T buckets of S = lanes·r slots
of B-byte cells) is T rows of S·B bytes. Row t − w·T is bucket t, slot j at bytes j·B to (j + 1)·B (words j·B/2
onward), in the grid's order: a bucket's cells sorted by stream (the connection's ordinal), a connection's r cells
together and in order, and a slot no connection holds B zero bytes. So T and a row's size are the schedule's, whatever
the load. A row's 8·S·B bits are its bytes, bit 8k + i of the row bit i of byte k: the `hm96-sha512/row/v2` bit row of
its S·B/2 little-endian u16 words (`words`), which the unit's port carries.

**The wiring** (`Wiring`, read off the record by `wiring`) says which connection holds which slots. Connection q, live
in `cells`/r rows from row `first`, holds slots ρ·r … ρ·r + r − 1 of each, ρ its rank by ordinal among the row's live
connections; its cell k of the link-window (k = 0 … cells − 1) is in row first + ⌊k/r⌋ at slot ρ·r + (k mod r)
(`Wiring.place`). The warden commits it as (q, first, cells) by q under a 64-byte salt it samples for the link-window:

    wiring = tagged_sha512(WIRING_TAG, salt ‖ u32be count ‖ (u32be q ‖ u32be first ‖ u32be cells) by q)

**The commitment** is a registered value's (`one_stage.registered`), under the warden's own domain:

    x_k    = the sha512/row/v2 digest of row k's 8·S·B bits
    b‖c_k  = hm96.SHA512.commit_string(default_key, x_k, salt_k)
    leaf_k = row_leaf(domain, k, hm96.SHA512.tree_leaf(default_key, b‖c_k), "hm96-sha512/row/v2")
    root   = the frame-v3-sha512 root of the T leaves (`tree_levels`)
    domain = CommitmentDomain(identity_digest_sha512(DOMAIN_TAG, {parameters, session, link, direction, window,
             cell_bytes, slots, rows, row_schema}), -1, RangeIndexedDomain(0, T), hash="sha512")

so a root opens under one link-window of one session's schedule alone. salt_k is 192 bytes the warden draws from the
operating system for each row, never for another. The opening (`Cells`: the rows, their salts and the wiring with its
salt) is witness, which the warden hands to the prover alone (`Cells.witness`: the salts, since the rows and the
wiring are the record's).

**The signed head** (`CellsHead`) is public, as a hiding root is:

    H = u16be len(link) ‖ link ‖ u8 direction ‖ u64be window ‖ parameters (64 bytes) ‖ u32be B ‖ u32be S ‖ u32be T
      ‖ wiring (64 bytes) ‖ domain (64 bytes) ‖ root (64 bytes)
    cells digest = tagged_sha512(CELLS_TAG, H)

and its certificate is `Issuer.sign`'s of the cells digest: the certifier's message for the link-window in its session
(`certificate.certificate_message`). The wire record's certificate of the same link-window signs its salted record
digest instead, so neither verifies as the other (each would need a SHA-512 preimage).

**The registration** (`registration`) is the head and its certificate as one public JSON object:

    {"format": REGISTRATION, "value", "link", "direction", "window", "parameters", "cell_bytes", "rows", "words",
     "word_bits", "row_schema", "domain", "wiring", "root",
     "layout": {"buckets", "slots", "connection_slots", "first_bucket"}, "certificate": {"key", "digest", "signature"}}

`value` is the registered value's name (`<link>/<direction>/<window>`), `words` and `word_bits` a row's S·B/2 leaves of
16 bits, and `layout` the rows' order: row t − first_bucket is bucket t, of `slots` slots, `connection_slots` = r a
connection. `check_registration` is the verifier's read of it: the head is the link-window's, its domain, rows, row
format and layout are the ones the schedule's parameters give, and its certificate verifies under the warden's key in
the session over the head's digest.

**How a unit reads it** (`port`). The verifier's own `--registered` port (`verity/registered-reads/v1`) names the value,
root, leaves, words, word bits, domain and row schema, and `reads`: unit `reads_from + i` opens the row of its read
(q, k), connection q's cell k of the link-window. The verifier takes the wiring only when it is the one the head names
and the schedule's (ordinals and first rows in admission order, r cells a row, at most `lanes` connections a row), and
refuses a read of a connection it doesn't hold, of a cell past the connection's, or of a cell another read opens. So a
read opens its own connection's cell and no other: never an empty slot, a foreign connection's cell or a cell twice.
The unit's cell is at its slot's words of the row (`Wiring.place`). A population of one unit a bucket (the I/O tie's)
reads whole rows instead (`bucket_port`): unit `reads_from + t` opens row t, every slot held or not, and no wiring.

**The units' wiring is the Program's** (`Units`): which cell each unit reads is pinned with the Program, from public
structure alone (the schedule, the ordinals and the signed wiring, never the cells' bytes), so it is never the
prover's. The port's value names its digest, `<link>/<direction>/<window>/units/<digest>`, and the Lean verifier
refuses a statement whose header names another value (`Flock.Registered.entry`), so a statement made for any other
units' wiring is refused even where its rows are the same: a pad opened where the Program reads a data cell of the
same row, say. A bare sequence of reads binds nothing (the value is the head's alone): it is a5_warden's form until
it states `Units`. `bucket_port` names the head's value too, and needs no `Units`: its unit t reads row t whole, so a
statement's positions are all it can vary, and Lean's registered reads hold each to its unit's row.

**The wiring is opened to the verifier for now** (the named gap, whose plan is in PROTOCOL.md): it gives away each
connection's first row and cells. q is the cells' stream field, the warden's ordinal: the k-th connection to reach it,
from 0, is q = k + 1, as the active certifier's `Requests` streams it and the I/O tie's map puts advice request k
(`order[k] + 1`).
"""

from __future__ import annotations

import os
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
from verity.core.protocols.certifier.grid import Record
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.core.protocols.sampled_proofs.one_stage import registered as G

DOMAIN_TAG = "verity-network-warden/cells-domain/v2"
CELLS_TAG = "verity-network-warden/cells/v2"
WIRING_TAG = "verity-network-warden/wiring/v1"
UNITS_TAG = "verity-network-warden/units/v1"
REGISTRATION = "verity-network-warden/registration/v2"
WORD_BITS = 16
OWNER = -1
_INST = hm96.SHA512
SALT_BYTES = _INST.salt_bytes
WIRING_SALT_BYTES = DIGEST_BYTES


def rows_of(params: WardenParams) -> int:
    """A link-window's rows: one a bucket, T."""
    return params.window


def row_bytes(params: WardenParams) -> int:
    """A row's bytes: its bucket's S = lanes·r slots of B bytes."""
    return params.grid.allocation * params.cell_bytes


def window_rows(record: Record, params: WardenParams) -> list[bytes]:
    """The link-window's rows: each bucket's slots in order, a slot no connection holds B zero bytes."""
    grid = params.grid
    if len(record.grid) != grid.buckets or any(len(row) != grid.allocation for row in record.grid):
        raise ValueError(f"a window has {grid.buckets} buckets of {grid.allocation} slots")
    empty = bytes(grid.frame_bytes)
    return [b"".join(empty if f is None else f for f in row) for row in record.grid]


@dataclass(frozen=True)
class Wiring:
    """Which connection holds which slots of a link-window's rows (the module docstring's "The wiring"):
    `connections` is each connection's (ordinal q, first row, cells in the link-window), by q, and `salt` the 64 bytes
    that hide them. Witness; the signed head names its `digest`."""

    connections: tuple[tuple[int, int, int], ...]
    salt: bytes

    def encode(self) -> bytes:
        return _u(len(self.connections), 4, "connections") + b"".join(
            _u(q, 4, "ordinal") + _u(first, 4, "first") + _u(cells, 4, "cells") for q, first, cells in self.connections)

    def digest(self) -> bytes:
        return tagged_sha512(WIRING_TAG, self.salt + self.encode())

    def live(self, params: WardenParams) -> list[list[int]]:
        """Each row's live connections by ordinal, or ValueError naming how the wiring isn't the schedule's."""
        T, r = params.window, params.slots
        if len(self.salt) != WIRING_SALT_BYTES:
            raise ValueError(f"the wiring's salt is {WIRING_SALT_BYTES} bytes")
        rows: list[list[int]] = [[] for _ in range(T)]
        last_q = last_first = -1
        for c in self.connections:
            if not isinstance(c, tuple) or len(c) != 3 or any(type(x) is not int for x in c):
                raise ValueError(f"a connection is (ordinal, first row, cells), not {c!r}")
            q, first, cells = c
            if not last_q < q < 1 << 32:
                raise ValueError(f"ordinal {q} after {last_q}: the ordinals increase, each a u32")
            if first < last_first:
                raise ValueError(f"connection {q} is first live at row {first}, before connection {last_q}'s row "
                                 f"{last_first}: connections are admitted in ordinal order")
            if cells < r or cells % r or first < 0 or first + cells // r > T:
                raise ValueError(f"connection {q}'s {cells} cells from row {first} aren't r = {r} a row within the "
                                 f"link-window's {T} rows")
            for t in range(first, first + cells // r):
                rows[t].append(q)
            last_q, last_first = q, first
        for t, live in enumerate(rows):
            if len(live) > params.lanes:
                raise ValueError(f"row {t} has {len(live)} live connections, past the schedule's {params.lanes} lanes")
        return rows

    def place(self, params: WardenParams, reads: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """The (row, slot) of each read (q, k), connection q's cell k of the link-window: the row its bucket's, the cell
        at the slot's words j·B/2 … (j + 1)·B/2 − 1. ValueError for a wiring that isn't the schedule's (`live`), or a
        read of a connection the wiring doesn't hold, of a cell past its cells, or of a cell another read opens."""
        rows, r = self.live(params), params.slots
        held = {q: (first, cells) for q, first, cells in self.connections}
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for read in reads:
            if not isinstance(read, (tuple, list)) or len(read) != 2 or any(type(x) is not int for x in read):
                raise ValueError(f"a read is (connection, cell), not {read!r}")
            q, k = read
            if q not in held:
                raise ValueError(f"connection {q} holds no slot of the link-window: a read opens a connection's own "
                                 "cells, never an empty slot")
            first, cells = held[q]
            if not 0 <= k < cells:
                raise ValueError(f"connection {q} has {cells} cells in the link-window, not cell {k}")
            if (q, k) in seen:
                raise ValueError(f"connection {q}'s cell {k} is read twice")
            seen.add((q, k))
            t = first + k // r
            out.append((t, rows[t].index(q) * r + k % r))
        return out


def wiring(record: Record, params: WardenParams, salt: bytes | None = None) -> Wiring:
    """The link-window's wiring, read off its record (a cell's stream field is its connection's ordinal), under `salt`
    (default: 64 bytes fresh from the operating system); ValueError when the record's cells aren't where the schedule
    puts them."""
    held: dict[int, list[tuple[int, int]]] = {}
    for t, row in enumerate(record.grid):
        for j, f in enumerate(row):
            if f is not None:
                held.setdefault(HEADER.unpack_from(f)[1], []).append((t, j))
    out = Wiring(tuple((q, at[0][0], len(at)) for q, at in sorted(held.items())),
                 os.urandom(WIRING_SALT_BYTES) if salt is None else bytes(salt))
    if out.place(params, [(q, k) for q, at in sorted(held.items()) for k in range(len(at))]) != \
            [x for _, at in sorted(held.items()) for x in at]:
        raise ValueError(f"{record.link} window {record.window}'s cells aren't laid out as the schedule lays them")
    return out


@dataclass(frozen=True)
class Units:
    """The units' wiring, as the Program pins it (the module docstring's "The units' wiring is the Program's"): unit
    `reads_from + i` of the population reads `cells[i]` = (q, k), connection q's cell k of the link-window. Public, and
    fixed from public structure alone, never from the cells' bytes."""

    cells: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cells, tuple) or not self.cells:
            raise ValueError("a population reads at least one cell: a nonempty tuple of (connection, cell)")
        for c in self.cells:
            if not isinstance(c, tuple) or len(c) != 2 or any(type(x) is not int for x in c) or \
                    not (0 < c[0] < 1 << 32 and 0 <= c[1] < 1 << 32):
                raise ValueError(f"a unit reads (connection, cell), an ordinal from 1 and a cell, u32 each, not {c!r}")

    @classmethod
    def span(cls, q: int, first: int, n: int) -> Units:
        """n units reading connection q's cells first … first + n − 1 in order."""
        return cls(tuple((q, first + i) for i in range(n)))

    def encode(self) -> bytes:
        return _u(len(self.cells), 4, "units") + b"".join(_u(q, 4, "ordinal") + _u(k, 4, "cell") for q, k in self.cells)

    def digest(self) -> bytes:
        return tagged_sha512(UNITS_TAG, self.encode())

    def value(self, head: CellsHead) -> str:
        """The name the port and the statement give `head`'s rows read under this units' wiring."""
        return f"{head.value}/units/{self.digest().hex()}"


def words(row: bytes) -> list[int]:
    """A row as its port's u16 words, little-endian, whose bit row is the row's bytes."""
    if len(row) % 2:
        raise ValueError(f"a row of {len(row)} bytes is no whole number of u16 words")
    return [int.from_bytes(row[i:i + 2], "little") for i in range(0, len(row), 2)]


def domain(params: WardenParams, session: bytes, link: str, direction: Direction, window: int) -> CommitmentDomain:
    """The frame-v3-sha512 domain of one link-window's rows in one session of one schedule."""
    binding = identity_digest_sha512(DOMAIN_TAG, {
        "parameters": params.digest.hex(), "session": session.hex(), "link": link, "direction": direction.value,
        "window": window, "cell_bytes": params.cell_bytes, "slots": params.grid.allocation, "rows": rows_of(params),
        "row_schema": SCHEMA})
    return CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, rows_of(params)), hash="sha512")


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
    """What the warden signs for a link-window's rows: public, since its root and wiring digest are hiding."""

    link: str
    direction: Direction
    window: int
    parameters: bytes
    cell_bytes: int
    slots: int
    rows: int
    wiring: bytes
    domain: bytes
    root: bytes

    def __post_init__(self) -> None:
        if any(len(x) != DIGEST_BYTES for x in (self.parameters, self.wiring, self.domain, self.root)):
            raise ValueError("the parameters digest, the wiring digest, the domain and the root are 64 bytes")

    @property
    def value(self) -> str:
        return value_name(self.link, self.direction, self.window)

    def encode(self) -> bytes:
        return b"".join([_text(self.link, "link"), _u(self.direction.code, 1, "direction"), _u(self.window, 8, "window"),
                         self.parameters, _u(self.cell_bytes, 4, "B"), _u(self.slots, 4, "S"), _u(self.rows, 4, "T"),
                         self.wiring, self.domain, self.root])

    def digest(self) -> bytes:
        return tagged_sha512(CELLS_TAG, self.encode())


@dataclass(frozen=True)
class Cells:
    """The warden's opening of one link-window's rows, witness: the rows, their salts and the wiring, and what they
    commit to."""

    link: str
    direction: Direction
    window: int
    params: WardenParams
    session: bytes
    rows: tuple[bytes, ...]
    salts: tuple[bytes, ...]
    wiring: Wiring

    @classmethod
    def commit(cls, link: str, direction: Direction, window: int, rows: Sequence[bytes], wiring: Wiring,
               params: WardenParams, session: bytes, salts: Sequence[bytes] | None = None) -> Cells:
        """`rows` and `wiring` as the link-window's (`of` reads both off a record). `salts`: one a row, sampled once
        (default: fresh from the operating system) and never for another row."""
        rows = tuple(bytes(r) for r in rows)
        if len(rows) != rows_of(params) or any(len(r) != row_bytes(params) for r in rows):
            raise ValueError(f"a link-window is {rows_of(params)} rows of {row_bytes(params)} bytes, one a bucket")
        wiring.live(params)
        salts = tuple(_INST.fresh_salt() for _ in rows) if salts is None else tuple(bytes(s) for s in salts)
        if len(salts) != len(rows) or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"one {_INST.salt_bytes}-byte salt a row")
        if len(set(salts)) != len(salts):
            raise ValueError("a salt is never reused: equal salts on two rows reveal the XOR of their digests")
        return cls(link, direction, window, params, bytes(session), rows, salts, wiring)

    @classmethod
    def of(cls, link: str, direction: Direction, record: Record, params: WardenParams, session: bytes,
           salts: Sequence[bytes] | None = None, wiring_salt: bytes | None = None) -> Cells:
        """`record`'s rows and wiring committed as `link`'s link-window in `direction`."""
        return cls.commit(link, direction, record.window, window_rows(record, params),
                          wiring(record, params, wiring_salt), params, session, salts)

    @property
    def witness(self) -> bytes:
        """What the warden keeps for the prover beside the record: the rows' salts, then the wiring's
        (`read_witness`)."""
        return b"".join(self.salts) + self.wiring.salt

    @property
    def value(self) -> str:
        return value_name(self.link, self.direction, self.window)

    @cached_property
    def domain(self) -> CommitmentDomain:
        return domain(self.params, self.session, self.link, self.direction, self.window)

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
        p = self.params
        return CellsHead(self.link, self.direction, self.window, p.digest, p.cell_bytes, p.grid.allocation,
                         len(self.rows), self.wiring.digest(), self.domain.domain_id, self.root)

    def path(self, position: int) -> bytes:
        """Row `position`'s siblings, leaf level first, 64 bytes each."""
        out, i = b"", position
        for level in self.levels[:-1]:
            out += level[i ^ 1]
            i >>= 1
        return out

    def statement(self, positions: Sequence[int], units: Units | None = None) -> tuple[list[list[int]], list[bytes],
                                                                                       dict[str, Any]]:
        """What a statement whose port reads the rows at `positions` commits: those rows as u16 words, their salts (the
        prover's), and its header's `registered` entry, `{"value", "positions", "paths"}`. With `units`, the
        positions are the rows its cells are in, and the entry names its value (`Units.value`), as `port` does."""
        if any(not 0 <= p < len(self.rows) for p in positions):
            raise ValueError(f"a position is outside the link-window's {len(self.rows)} rows")
        if units is not None and [t for t, _ in self.wiring.place(self.params, units.cells)] != list(positions):
            raise ValueError("the positions aren't the rows the units' wiring reads")
        return ([words(self.rows[p]) for p in positions], [self.salts[p] for p in positions],
                {"value": self.value if units is None else units.value(self.head),
                 "positions": [int(p) for p in positions], "paths": [self.path(p).hex() for p in positions]})


def read_witness(blob: bytes) -> tuple[list[bytes], bytes]:
    """`Cells.witness` read back: the rows' salts and the wiring's."""
    rows = len(blob) - WIRING_SALT_BYTES
    if rows < 0 or rows % SALT_BYTES:
        raise ValueError(f"a link-window's witness is {SALT_BYTES} bytes a row and the wiring's {WIRING_SALT_BYTES}")
    return [blob[i:i + SALT_BYTES] for i in range(0, rows, SALT_BYTES)], blob[rows:]


def attest(issuer: Issuer, record: Record, params: WardenParams, direction: Direction,
           salts: Sequence[bytes] | None = None, wiring_salt: bytes | None = None) -> tuple[Cells, Certificate | None]:
    """The link-window's rows and wiring committed and their head signed by `issuer` (None when its signer refuses)."""
    cells = Cells.of(record.link, direction, record, params, issuer.session, salts, wiring_salt)
    return cells, issuer.sign(record.link, direction, record.window, cells.head.digest())


def _layout(params: WardenParams, window: int) -> dict[str, int]:
    return {"buckets": params.window, "slots": params.grid.allocation, "connection_slots": params.slots,
            "first_bucket": window * params.window}


def registration(head: CellsHead, certificate: Certificate, params: WardenParams) -> dict[str, Any]:
    """The public record a unit's registered port reads: the head, the rows' format and order, and the certificate."""
    return {"format": REGISTRATION, "value": head.value, "link": head.link, "direction": head.direction.value,
            "window": head.window, "parameters": head.parameters.hex(), "cell_bytes": head.cell_bytes,
            "rows": head.rows, "words": head.slots * head.cell_bytes // 2, "word_bits": WORD_BITS,
            "row_schema": SCHEMA, "domain": head.domain.hex(), "wiring": head.wiring.hex(), "root": head.root.hex(),
            "layout": _layout(params, head.window),
            "certificate": {"key": certificate.key_id.hex(), "digest": certificate.digest.hex(),
                            "signature": certificate.signature.hex()}}


def check_registration(doc: Mapping[str, Any], key: PublicKey, session: bytes, params: WardenParams, link: str,
                       direction: Direction, window: int) -> tuple[CellsHead, CommitmentDomain]:
    """The verifier's read of a registration for the link-window it expects, under the warden's key and session and the
    schedule's parameters: the head and the domain the verifier derived, or ValueError naming what fails."""
    B, S, T = params.cell_bytes, params.grid.allocation, rows_of(params)
    dom = domain(params, session, link, direction, window)
    want = {"format": REGISTRATION, "value": value_name(link, direction, window), "link": link,
            "direction": direction.value, "window": window, "parameters": params.digest.hex(), "cell_bytes": B,
            "rows": T, "words": S * B // 2, "word_bits": WORD_BITS, "row_schema": SCHEMA,
            "domain": dom.domain_id.hex(), "layout": _layout(params, window)}
    for k, v in want.items():
        if doc.get(k) != v:
            raise ValueError(f"the registration's {k} is {doc.get(k)!r}, and the schedule's {v!r}")
    read = {"wiring", "root", "certificate"}
    if set(doc) != set(want) | read:
        raise ValueError(f"a registration holds the keys {sorted(set(want) | read)} and no other")
    try:
        head = CellsHead(link, direction, window, params.digest, B, S, T, bytes.fromhex(doc["wiring"]), dom.domain_id,
                         bytes.fromhex(doc["root"]))
        c = doc["certificate"]
        cert = Certificate(bytes.fromhex(c["key"]), link, direction, window, bytes.fromhex(c["digest"]),
                           bytes.fromhex(c["signature"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the registration's wiring, root or certificate doesn't read: {e}") from None
    if cert.digest != head.digest():
        raise ValueError("the registration's certificate doesn't sign its head")
    if not cert.verify(key, session):
        raise ValueError("the registration's certificate doesn't verify under the warden's key in the session")
    return head, dom


def _schedules(head: CellsHead, params: WardenParams) -> None:
    if (head.parameters, head.cell_bytes, head.slots, head.rows) != \
            (params.digest, params.cell_bytes, params.grid.allocation, rows_of(params)):
        raise ValueError("the head isn't of the schedule's rows")


def _port(head: CellsHead, value: str, rows: Sequence[int], reads_from: int) -> dict[str, Any]:
    e = {"leaves": head.rows, "words": head.slots * head.cell_bytes // 2, "word_bits": WORD_BITS,
         "domain": head.domain.hex()}
    return produced.port(value, e, head.root.hex(), rows, SCHEMA) | \
        ({G.READS_FROM: int(reads_from)} if reads_from else {})


def port(head: CellsHead, wiring_: Wiring, params: WardenParams, reads: Units | Sequence[tuple[int, int]],
         reads_from: int = 0) -> dict[str, Any]:
    """The verifier's own `--registered` port of a unit population that opens the head's rows: unit `reads_from + i`
    opens the row of its read (q, k), connection q's cell k of the link-window, under the warden's wiring
    (`Wiring.place`, which refuses what no honest read is). `reads` is the Program's `Units`, whose value
    (`Units.value`) the port names, so a statement must name it too; a bare sequence of reads binds nothing, and the
    port names the head's value (a5_warden's form until it states `Units`). ValueError for a wiring the head doesn't
    name."""
    _schedules(head, params)
    if wiring_.digest() != head.wiring:
        raise ValueError("the wiring isn't the one the warden signed")
    units = reads if isinstance(reads, Units) else None
    rows = [t for t, _ in wiring_.place(params, reads if units is None else units.cells)]
    return _port(head, head.value if units is None else units.value(head), rows, reads_from)


def bucket_port(head: CellsHead, params: WardenParams, reads_from: int = 0) -> dict[str, Any]:
    """The verifier's own `--registered` port of a population of one unit a bucket (`IoTieBucket`): unit
    `reads_from + t` opens row t, its bucket's every slot, held or not. It names no connection, so it reads no
    wiring, and the head's rows alone fix its reads, so there is no units' wiring to name: the port names the head's
    value."""
    _schedules(head, params)
    return _port(head, head.value, list(range(head.rows)), reads_from)


def refusal(port_: Mapping[str, Any], dom: CommitmentDomain, entry: Mapping[str, Any],
            bcs: Sequence[bytes]) -> str | None:
    """The host's read of a statement's registered `entry` (its header's) against the verifier's `port_`, as the Lean
    verifier's registered reads make it, in its order (`Flock.Registered.checkPort`), when instance i is unit
    `reads_from + i` and reads table row i, whose `b ‖ c` is `bcs[i]`: None when they accept, else why not. The entry
    must name the port's value, each row must climb its path to the root at its position under `dom`, and row i must
    be at the position the port reads for its unit."""
    if (port_.get("domain"), port_.get("leaves")) != (dom.domain_id.hex(), dom.count):
        raise ValueError("the port names another domain or leaf count than `dom`, and the Lean verifier climbs the port's")
    if entry.get("value") != port_["value"]:
        return f"registered reads: the port reads {port_['value']}, not another value ({entry.get('value')})"
    positions, paths = list(entry.get("positions", ())), list(entry.get("paths", ()))
    if len(paths) != len(positions) or len(bcs) != len(positions):
        return "registered reads: one position and one path per table row"
    root = bytes.fromhex(port_["root"])
    for r, (p, b, path) in enumerate(zip(positions, bcs, paths)):
        if not opens(dom, root, p, b, bytes.fromhex(path)):
            return f"registered reads: table row {r} does not open the registered root at position {p}"
    reads, base = list(port_["reads"]), int(port_.get(G.READS_FROM, 0))
    if len(positions) != len(reads):
        return f"registered reads: {len(positions)} rows for the {len(reads)} units the verifier's program reads"
    for i, (p, want) in enumerate(zip(positions, reads)):
        if p != want:
            return f"registered reads: unit {base + i} reads position {p}, and the verifier's program reads {want}"
    return None


def reads_file(ports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The verifier's own `--registered` file (`verity/registered-reads/v1`) of its ports, `{circuit port: port}`."""
    return {"format": G.READS_V1, "ports": {p: dict(e) for p, e in ports.items()}}
