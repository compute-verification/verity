"""The network certifier's connection rows: what it signs of the connections it released in an execution phase of a
link, as tables of rows committed as its cells' rows are (`warden_rows`) and listed in one head for that execution
phase, which it signs once. A schedule has them only where its parameters say so (`WardenParams.lengths`): a schedule
without them has no such head, and its records, cells' heads and certificates are the ones it had. The head lists
the lengths rows and the fails rows and, where the schedule has content rows (`WardenParams.content`), three tables
a class after them. Every table of the same connections is under the one signature, and no proof block opens a row
of them under any other head.

**The lengths and fails tables** (`tables`) of execution phase w (`window` w in the code and the records) are made from
the releases the network certifier hands at execution phase w + 1's close, or under the lengths rows' form 2 at the
run's last close for its last execution phase (`warden.Released`, by slot stride: `warden`'s "The lengths rows").
The certifier writes them from the cells it cut and released itself, never from the
server's report, so they say what its records of the same execution phases say, in another order (`of_records` is
that reading of the records in the clear): that the two agree rests on the network certifier's own code
(`trusted-host`), and no proof ties them. Each table is T / `every` rows,
row j slot stride j's releases in release order in its first slots, K = `WardenParams.stride_releases` slots a row and
the rest zero, every field little-endian:

- `lengths`: a slot is u32 N + 1 ‖ u32 bytes ‖ u32 n, 12 bytes: N the request's place in the network certifier's
  release order (the field is 0 in an empty slot), bytes its request's bytes, and n its response's `DATA` cells before
  its `END` or `FAIL`, one id event each.
- `fails`: a slot is u16 class ‖ u16 at, 4 bytes, the slot of the lengths row at the same place: class 1 and at = n
  where the connection ended in a `FAIL`, both 0 where it ended in its `END` and in an empty slot. Only a released
  request has a slot, and a released connection's `FAIL` is the server's (late, limit, upstream or format), so the
  class says the server failed it or nobody did. These rows are not the `fails` record (`warden`'s "The records"),
  whose frames hold each `FAIL`'s reason in its bucket and stay witness.

So a table's size is the schedule's whatever the load, and which slots hold a request, each one's bytes, n and outcome
are values. `releases` reads the two tables back and refuses rows that aren't of this form.

**The content tables** of the same execution phase are made from the same hand, in which each release then carries
its request's bytes and its ids (`content_rows`' "The tables"): `request.i`, `response.i` and `end.i` for each class
i the schedule lists, each of the class's rows, a release's rows in the class `content_rows.file` gives it. `tables`
is that rule's one caller. They rest on `trusted-host` as the lengths and fails tables do.

**A row's slots** (`check_slots`): the lengths row's proof block returns the whole row after its two words, and a
proof block's outputs fit one public size class, so a row has at most (`UNIT_CLASS_MAX` − 2·16) / 96 slots. It is the
one statement of that bound: a schedule with lengths rows whose slot stride releases more is refused when its
parameters are built (`WardenParams`), and the proof block's shape refuses the same K.

**The commitment** of a table is its cells' rows' (`warden_rows`' "The commitment"), under the table's own domain:

    x_k    = the sha512/row/v2 digest of row k's bits
    b‖c_k  = hm96.SHA512.commit_string(default_key, x_k, salt_k)
    leaf_k = row_leaf(domain, k, hm96.SHA512.tree_leaf(default_key, b‖c_k), "hm96-sha512/row/v2")
    root   = the frame-v3-sha512 root of the table's leaves
    domain = CommitmentDomain(identity_digest_sha512(DOMAIN_TAG, {parameters, session, link, window, table, rows,
             row_bytes, row_schema}), -1, RangeIndexedDomain(0, rows), hash="sha512")

salt_k is fresh for each row of each table. The opening (`Rows`: each table's rows and their salts) is witness, the
network certifier's to hand to the prover alone.

**The signed head** (`Head`) is public, as a hiding root is, one for each link's execution phase:

    H     = u16be len(link) ‖ link ‖ u64be window ‖ parameters (64 bytes) ‖ u16be tables ‖ entry of each table
    entry = u16be len(name) ‖ name ‖ u32be rows ‖ u32be row bytes ‖ domain (64 bytes) ‖ root (64 bytes)
    digest = tagged_sha512(HEAD_TAG, H)

with the tables in the order the schedule's parameters list them (`tables_of`: `lengths`, then `fails`, then each
class's `request.i`, `response.i` and `end.i`). Its
certificate is `Issuer.sign`'s of the digest for the link's egress in execution phase w, the one whose releases it
holds, though it is signed an execution phase later. A record's, a clock's and a cells' head's certificates of the
same execution phase sign other digests, so none verifies as it (each would need a SHA-512 preimage). `parameters` is
the schedule's digest, which holds the connection rows' form.

**The word that no head comes** (`no_head_digest`, `no_head`, `check_no_head`; the lengths rows' form 2 only,
`WardenParams.closes`). Under form 2 the network certifier ends every execution phase of a run in exactly one of two
certificates of the link's egress in that execution phase: its head's, or this word's, which it signs where it
doesn't hand the execution phase (`warden`'s "The lengths rows": it, the one before it or the one after it wasn't
published complete, or the run ended early):

    digest = tagged_sha512(NO_HEAD_TAG, u16be len(link) ‖ link ‖ u64be window ‖ parameters (64 bytes))

a head's first three fields under a tag of its own. So the word is of one link's execution phase under one schedule,
whose digest holds the form, and, as every certificate is, of one session and one key
(`certificate.certificate_message`). Its public record is

    {"format": NO_HEAD, "link", "window", "parameters", "certificate": {"key", "digest", "signature"}}

A reader tells a head, the word and silence apart with public values alone. The word's digest is a function of the
link, the execution phase and the schedule's digest, so a certificate that signs it is the word, and no head's
certificate is (`check_registration` refuses the word's and `check_no_head` a head's: each would need a SHA-512
collision across the two tags). An execution phase with neither has not ended yet, or its signer refused the leaf:
it has no certificate, and fails closed as a record without one does. Which of the two an execution phase ends in
is a function of which execution phases' records were published complete, which their certificates already say, so
the word makes nothing else public. That an execution phase gets one of the two and never both rests on the network
certifier's own code (`trusted-host`): its signer signs the digest it is asked to. Under form 1 an execution phase
that isn't handed has no certificate of either kind.

**The registration** (`registration`) is the head and its certificate as one public JSON object:

    {"format": REGISTRATION, "link", "window", "parameters", "word_bits", "row_schema",
     "tables": [{"name", "value", "rows", "words", "domain", "root", "layout"}, ...],
     "certificate": {"key", "digest", "signature"}}

A table's `value` is its registered value's name (`<link>/<name>/<window>`), `words` a row's leaves of `word_bits` =
16 bits (a slot's fields low word first), and `layout` its rows' order: for the lengths and fails tables
`{"strides", "every", "first_bucket", "slots", "slot_bytes"}`, row j the releases of buckets first_bucket + j·every …
first_bucket + (j + 1)·every − 1 in `slots` slots of `slot_bytes`; for a content table
`{"class", "prompt_ids", "ids", "first_bucket", "buckets"}`, row r the r-th release of buckets first_bucket …
first_bucket + buckets − 1 filed in the schedule's class `class`. `check_registration` is the verifier's read of it:
the head is the execution phase's, it lists the schedule's tables and no other, each one's domain, rows and layout
are the ones the schedule's parameters give, and its one certificate verifies under the network certifier's key in the
session over the head's digest.

**The port** (`port`) is the verifier's own `--registered` port of the proof blocks that read a table, one a row: proof
block `reads_from + j` reads row j, so the head the verifier checked fixes every read. The lengths row's proof block
(`verity.protocols.certifier.prover.lengths_row`) reads the fails table's row j through one and commits the row it
returns as a read of the lengths table's row j through the other.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from verity.circuits.partition_v2 import UNIT_CLASS_MAX
from verity.commitments import hm96
from verity.commitments.identity import identity_digest_sha512, tagged_sha512
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, tree_levels
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.commitments.rowleaf import row_leaf
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier.active import DATA, PAD
from verity.protocols.certifier.verifier.certificate import (DIGEST_BYTES, Certificate, Direction, Issuer, PublicKey, _text,
                                                         _u)
from verity.protocols.certifier.verifier.grid import BlockGrid, row_count
from verity.protocols.certifier.verifier.io_tie import parse_cell
from verity.protocols.certifier.verifier.warden import END, FAIL, Released, WardenParams, read_cell
from verity.protocols.certifier.verifier.warden_rows import OWNER, SALT_BYTES, WORD_BITS, bc
from verity.protocols.spot.verifier.one_stage import produced
from verity.protocols.spot.verifier.one_stage import registered as G

DOMAIN_TAG = "verity-network-certifier/connection-rows-domain/v1"
HEAD_TAG = "verity-network-certifier/connection-rows/v1"
REGISTRATION = "verity-network-certifier/connection-rows-registration/v1"
NO_HEAD_TAG = "verity-network-certifier/connection-rows-no-head/v1"
NO_HEAD = "verity-network-certifier/connection-rows-no-head-word/v1"
LENGTHS, FAILS = "lengths", "fails"
#: a slot of each table: u32 N + 1 ‖ u32 bytes ‖ u32 n, and u16 class ‖ u16 at
SLOT = {LENGTHS: struct.Struct("<III"), FAILS: struct.Struct("<HH")}
#: the words the lengths row's proof block returns before the row: its verdict and its failures
BLOCK_WORDS = 2
_INST = hm96.SHA512


def check_slots(K: int, fix: Callable[[int], str] | None = None) -> None:
    """ValueError unless a lengths row of K slots is one its proof block returns (the module docstring's "A row's
    slots"). `fix(most)` is what the caller's refusal says to change, given the most slots a row has; without one it
    says what any schedule changes."""
    bits = 8 * SLOT[LENGTHS].size
    most = (UNIT_CLASS_MAX - BLOCK_WORDS * WORD_BITS) // bits
    if not 1 <= K <= most:
        raise ValueError(f"a lengths row is 1 <= K <= {most} slots, not K={K!r}: its proof block returns the whole "
                         f"row, {bits} bits a slot, after its {BLOCK_WORDS} words, in a public size class of at most "
                         f"{UNIT_CLASS_MAX} bits; "
                         + (fix(most) if fix else "a schedule's slot stride releases at most that many (fewer lanes "
                                                  "or a shorter stride)"))


def strides_of(params: WardenParams) -> int:
    """A lengths or fails table's rows: one a slot stride, T / every."""
    return params.window // params.every


def tables_of(params: WardenParams) -> tuple[tuple[str, int, int], ...]:
    """The tables a schedule's head lists, in its order: each one's (name, rows, a row's bytes). The lengths table,
    the fails table, then the content tables of a schedule that has them (`content_rows.tables_of`). ValueError for
    a schedule without connection rows."""
    if params.lengths is None:
        raise ValueError("the schedule has no connection rows (WardenParams.lengths)")
    return (*((name, strides_of(params), params.stride_releases * SLOT[name].size) for name in (LENGTHS, FAILS)),
            *CT.tables_of(params))


def tables(params: WardenParams, strides: Sequence[Sequence[Released]]) -> dict[str, list[bytes]]:
    """An execution phase's tables, every one its head lists and in the head's order, from its releases by slot
    stride, as the network certifier hands them: the lengths and fails tables (the module docstring's) and, in a
    schedule with content rows, each class's three, every release in the class the network certifier's rule files it
    in (`content_rows.file`, of which this is the one caller). ValueError for a schedule without connection rows, or
    releases that aren't one execution phase's of it."""
    (_, T, size), (_, _, fails_size) = tables_of(params)[:2]
    K = params.stride_releases
    if len(strides) != T or any(len(stride) > K for stride in strides):
        raise ValueError(f"an execution phase's releases are {T} slot strides of at most {K}")
    out: dict[str, list[bytes]] = {LENGTHS: [], FAILS: []}
    for stride in strides:
        for r in stride:
            if not (0 <= r.order < (1 << 32) - 1 and 0 < r.request_bytes < 1 << 32 and
                    0 <= r.outputs < params.egress_cells and type(r.failed) is bool):
                raise ValueError(f"{r} is no release of the schedule: a place in the release order, a request's "
                                 f"bytes, fewer than {params.egress_cells} outputs and whether it failed")
        out[LENGTHS].append(b"".join(SLOT[LENGTHS].pack(r.order + 1, r.request_bytes, r.outputs)
                                     for r in stride).ljust(size, b"\0"))
        out[FAILS].append(b"".join(SLOT[FAILS].pack(int(r.failed), r.outputs if r.failed else 0)
                                   for r in stride).ljust(fails_size, b"\0"))
    if params.content is not None:
        released = [r for stride in strides for r in stride]
        out |= CT.tables(params, released, [CT.file(params, r) for r in released])
    return out


def releases(params: WardenParams, lengths: Sequence[bytes], fails: Sequence[bytes]) -> list[list[Released]]:
    """The releases by slot stride that `tables` lays out as these two tables, or ValueError naming how they aren't
    the schedule's: a row's size, a slot after an empty one, a fails slot that isn't its lengths slot's (class 0 and
    at 0, or class 1 and at = n), or anything in an empty slot. A release reads back without its content."""
    (_, T, size), (_, _, fails_size) = tables_of(params)[:2]
    if len(lengths) != T or len(fails) != T or any(len(x) != size for x in lengths) or \
            any(len(x) != fails_size for x in fails):
        raise ValueError(f"an execution phase's tables are {T} rows of {params.stride_releases} slots each")
    out: list[list[Released]] = []
    for j, (a, b) in enumerate(zip(lengths, fails)):
        stride: list[Released] = []
        for k, ((order, request, n), (cls, at)) in enumerate(zip(SLOT[LENGTHS].iter_unpack(a),
                                                                 SLOT[FAILS].iter_unpack(b))):
            if not order:
                if request or n or cls or at:
                    raise ValueError(f"row {j} slot {k} is empty and not zero")
                continue
            if len(stride) != k:
                raise ValueError(f"row {j} slot {k} follows an empty slot")
            if (cls, at) not in ((0, 0), (1, n)):
                raise ValueError(f"row {j} slot {k}'s fails slot is ({cls}, {at}), not (0, 0) or (1, n = {n})")
            stride.append(Released(order - 1, request, n, bool(cls)))
        out.append(stride)
    return out


def of_records(params: WardenParams, ingress: Sequence[BlockGrid], egress: Sequence[BlockGrid],
               order: Mapping[int, int] | None = None) -> dict[int, tuple[tuple[Released, ...], ...]]:
    """The releases a run's records hold, read in the clear (a rehearsal's reading, as `warden.audit` is, and
    nothing a verifier runs): each execution phase's by slot stride, as the network certifier hands them and `tables`
    takes them. A release is an `END` in the ingress record, in its bucket's slot stride; N is its place among the
    run's, by bucket and then by connection; its request is its connection's ingress `DATA` and `END` payloads; its
    outputs are its connection's egress `DATA` cells, with content rows each one's id; and it failed where its
    egress ends in a `FAIL`. `ingress` and `egress` are each direction's records from the run's first execution
    phase, one each and in order, since N counts from the run's first release. The records count N only while every
    ingress record so far is complete: one that isn't has lost its cells from its fault on, and the releases among
    them. `order`, each released connection's place by its ordinal as the network certifier's log of the run has
    it (`Warden.log`'s `release`), gives N past such a record, and is held to the records' count before it; without
    it an execution phase past such a record has no entry. Nor has one with a release whose connection hasn't ended
    in the records. ValueError for a schedule without connection rows, records that aren't a run's from its first, a
    cell that isn't the schedule's, or an `order` that isn't the records'."""
    if params.lengths is None:
        raise ValueError("the schedule has no connection rows (WardenParams.lengths)")
    windows = list(range(len(ingress)))
    if [r.window for r in ingress] != windows or [r.window for r in egress] != windows:
        raise ValueError("a run's records are each direction's from its first execution phase, one each, in order")
    B, content, n = params.cell_bytes, params.content is not None, 0
    cut: dict[int, bytearray] = {}
    live: dict[int, list] = {}  # by connection, a release that hasn't ended: [N, its request, its ids, how it ended]
    out: dict[int, list[list[list]]] = {}
    counted = True  # whether the records still count N
    for w, (a, b) in enumerate(zip(ingress, egress)):
        if counted or order is not None:
            out[w] = [[] for _ in range(strides_of(params))]
        for t, (row_in, row_out) in enumerate(zip(a.grid, b.grid)):
            for f in row_in[:row_count(row_in)]:
                kind, o, payload = read_cell(f, B)
                if kind in (DATA, END):
                    cut.setdefault(o, bytearray()).extend(payload)
                if kind == END:
                    place = n if order is None else order.get(o)
                    if place is None or (counted and place != n):
                        raise ValueError(f"connection {o} is release {n} of the records, and {place} of the run's log")
                    live[o] = [place, bytes(cut.pop(o)), bytearray(), None]
                    if w in out:
                        out[w][t // params.every].append(live[o])
                    n += 1
            for f in row_out[:row_count(row_out)]:
                kind, o, _ = read_cell(f, B)
                if o not in live or kind == PAD:
                    continue
                read = parse_cell(f) if kind == DATA else None
                if kind not in (END, FAIL) and read is None:
                    raise ValueError(f"egress bucket {w * params.window + t}: connection {o}'s cell is no DATA cell "
                                     f"of the id egress form, END or FAIL")
                if read is not None:
                    live[o][2] += read.id.to_bytes(CT.U32_BYTES, "little")
                else:
                    live.pop(o)[3] = kind == FAIL
        counted = counted and a.status.complete
    return {w: tuple(tuple(Released(order, len(request), len(ids) // CT.U32_BYTES, failed,
                                    *((request, bytes(ids)) if content else ()))
                           for order, request, ids, failed in stride) for stride in strides)
            for w, strides in out.items() if all(e[3] is not None for stride in strides for e in stride)}


def domain(params: WardenParams, session: bytes, link: str, window: int, table: str, rows: int,
           row_bytes: int) -> CommitmentDomain:
    """The frame-v3-sha512 domain of one table of one execution phase of one link in one session of one schedule."""
    binding = identity_digest_sha512(DOMAIN_TAG, {
        "parameters": params.digest.hex(), "session": session.hex(), "link": link, "window": window, "table": table,
        "rows": rows, "row_bytes": row_bytes, "row_schema": SCHEMA})
    return CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, rows), hash="sha512")


def value_name(link: str, table: str, window: int) -> str:
    return f"{link}/{table}/{window}"


@dataclass(frozen=True, slots=True)
class Entry:
    """One table as the head lists it: its name, its rows and a row's bytes, its domain and its root."""

    name: str
    rows: int
    row_bytes: int
    domain: bytes
    root: bytes

    def __post_init__(self) -> None:
        if len(self.domain) != DIGEST_BYTES or len(self.root) != DIGEST_BYTES:
            raise ValueError("a table's domain and root are 64 bytes")

    def encode(self) -> bytes:
        return b"".join([_text(self.name, "table"), _u(self.rows, 4, "rows"), _u(self.row_bytes, 4, "row bytes"),
                         self.domain, self.root])


@dataclass(frozen=True, slots=True)
class Head:
    """What the network certifier signs, once, for an execution phase's connection rows: its tables. Public, since each
    root is hiding."""

    link: str
    window: int
    parameters: bytes
    tables: tuple[Entry, ...]

    def __post_init__(self) -> None:
        if len(self.parameters) != DIGEST_BYTES:
            raise ValueError("the parameters digest is 64 bytes")
        names = [e.name for e in self.tables]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"a head lists each of its tables once, and at least one: {names}")

    def table(self, name: str) -> Entry:
        return next(e for e in self.tables if e.name == name)

    def encode(self) -> bytes:
        return b"".join([_text(self.link, "link"), _u(self.window, 8, "window"), self.parameters,
                         _u(len(self.tables), 2, "tables"), *(e.encode() for e in self.tables)])

    def digest(self) -> bytes:
        return tagged_sha512(HEAD_TAG, self.encode())


@dataclass(frozen=True)
class Table:
    """The network certifier's opening of one table of an execution phase, witness: its rows and their salts, and
    what they commit to."""

    link: str
    window: int
    name: str
    params: WardenParams
    session: bytes
    rows: tuple[bytes, ...]
    salts: tuple[bytes, ...]

    @property
    def value(self) -> str:
        return value_name(self.link, self.name, self.window)

    @cached_property
    def domain(self) -> CommitmentDomain:
        return domain(self.params, self.session, self.link, self.window, self.name, len(self.rows), len(self.rows[0]))

    @cached_property
    def bcs(self) -> tuple[bytes, ...]:
        return tuple(bc(r, s) for r, s in zip(self.rows, self.salts))

    @cached_property
    def levels(self) -> tuple[tuple[bytes, ...], ...]:
        dom = self.domain
        return tree_levels(dom, [row_leaf(dom, k, _INST.tree_leaf(_INST.default_key, x), SCHEMA)
                                 for k, x in enumerate(self.bcs)])

    @property
    def entry(self) -> Entry:
        return Entry(self.name, len(self.rows), len(self.rows[0]), self.domain.domain_id, self.levels[-1][0])

    def path(self, position: int) -> bytes:
        """Row `position`'s siblings, leaf level first, 64 bytes each."""
        out, i = b"", position
        for level in self.levels[:-1]:
            out += level[i ^ 1]
            i >>= 1
        return out


@dataclass(frozen=True)
class Rows:
    """The network certifier's opening of an execution phase's connection rows, witness: each table the head lists."""

    link: str
    window: int
    params: WardenParams
    tables: tuple[Table, ...]

    @classmethod
    def commit(cls, link: str, window: int, tables_: Mapping[str, Sequence[bytes]], params: WardenParams,
               session: bytes, salts: Mapping[str, Sequence[bytes]] | None = None) -> Rows:
        """`tables_` (each table's rows by name) as the execution phase's connection rows. `salts`: one a row of each
        table, sampled once (default: fresh from the operating system) and never for another row."""
        want = tables_of(params)
        if list(tables_) != [name for name, _, _ in want]:
            raise ValueError(f"the schedule's tables are {[name for name, _, _ in want]}, in that order")
        out, seen = [], set()
        for name, count, size in want:
            rows = tuple(bytes(r) for r in tables_[name])
            if len(rows) != count or any(len(r) != size for r in rows):
                raise ValueError(f"the {name} table is {count} rows of {size} bytes")
            mine = tuple(_INST.fresh_salt() for _ in rows) if salts is None else tuple(bytes(s) for s in salts[name])
            if len(mine) != len(rows) or any(len(s) != SALT_BYTES for s in mine):
                raise ValueError(f"one {SALT_BYTES}-byte salt a row")
            if len(seen | set(mine)) != len(seen) + len(mine):
                raise ValueError("a salt is never reused: equal salts on two rows reveal the XOR of their digests")
            seen |= set(mine)
            out.append(Table(link, window, name, params, bytes(session), rows, mine))
        return cls(link, window, params, tuple(out))

    def table(self, name: str) -> Table:
        return next(t for t in self.tables if t.name == name)

    @property
    def witness(self) -> bytes:
        """What the network certifier keeps for the prover beside the rows: their salts, table by table."""
        return b"".join(s for t in self.tables for s in t.salts)

    @cached_property
    def head(self) -> Head:
        return Head(self.link, self.window, self.params.digest, tuple(t.entry for t in self.tables))


def attest(issuer: Issuer, link: str, window: int, strides: Sequence[Sequence[Released]], params: WardenParams,
           salts: Mapping[str, Sequence[bytes]] | None = None) -> tuple[Rows, Certificate | None]:
    """An execution phase's connection rows committed and their one head signed by `issuer` (None when its signer
    refuses)."""
    rows = Rows.commit(link, window, tables(params, strides), params, issuer.session, salts)
    return rows, issuer.sign(link, Direction.EGRESS, window, rows.head.digest())


def _layout(params: WardenParams, window: int, table: str) -> dict[str, int]:
    first = window * params.window
    if table not in SLOT:
        return CT.layout(params, table) | {"first_bucket": first, "buckets": params.window}
    return {"strides": strides_of(params), "every": params.every, "first_bucket": first,
            "slots": params.stride_releases, "slot_bytes": SLOT[table].size}


def registration(head: Head, certificate: Certificate, params: WardenParams) -> dict[str, Any]:
    """The public record of an execution phase's connection rows: the head, each table's format and order, and the one
    certificate."""
    return {"format": REGISTRATION, "link": head.link, "window": head.window, "parameters": head.parameters.hex(),
            "word_bits": WORD_BITS, "row_schema": SCHEMA,
            "tables": [{"name": e.name, "value": value_name(head.link, e.name, head.window), "rows": e.rows,
                        "words": e.row_bytes * 8 // WORD_BITS, "domain": e.domain.hex(), "root": e.root.hex(),
                        "layout": _layout(params, head.window, e.name)} for e in head.tables],
            "certificate": {"key": certificate.key_id.hex(), "digest": certificate.digest.hex(),
                            "signature": certificate.signature.hex()}}


def check_registration(doc: Mapping[str, Any], key: PublicKey, session: bytes, params: WardenParams, link: str,
                       window: int) -> tuple[Head, dict[str, CommitmentDomain]]:
    """The verifier's read of an execution phase's registration, under the network certifier's key and session and the
    schedule's parameters: the head and each table's domain as the verifier derived it, or ValueError naming what
    fails."""
    listed = tables_of(params)
    want = {"format": REGISTRATION, "link": link, "window": window, "parameters": params.digest.hex(),
            "word_bits": WORD_BITS, "row_schema": SCHEMA}
    for k, v in want.items():
        if doc.get(k) != v:
            raise ValueError(f"the registration's {k} is {doc.get(k)!r}, and the schedule's {v!r}")
    if set(doc) != set(want) | {"tables", "certificate"}:
        raise ValueError(f"a registration holds the keys {sorted(set(want) | {'tables', 'certificate'})} and no other")
    got = doc["tables"]
    if not isinstance(got, list) or len(got) != len(listed) or not all(isinstance(t, Mapping) for t in got):
        raise ValueError(f"the registration lists the schedule's {len(listed)} tables, "
                         f"{[name for name, _, _ in listed]}, and no other")
    domains, entries = {}, []
    for t, (name, rows, size) in zip(got, listed):
        dom = domain(params, session, link, window, name, rows, size)
        wanted = {"name": name, "value": value_name(link, name, window), "rows": rows, "words": size * 8 // WORD_BITS,
                  "domain": dom.domain_id.hex(), "layout": _layout(params, window, name)}
        for k, v in wanted.items():
            if t.get(k) != v:
                raise ValueError(f"the registration's {name} table has {k} {t.get(k)!r}, and the schedule's {v!r}")
        if set(t) != set(wanted) | {"root"}:
            raise ValueError(f"a table holds the keys {sorted(set(wanted) | {'root'})} and no other")
        try:
            entries.append(Entry(name, rows, size, dom.domain_id, bytes.fromhex(t["root"])))
        except (TypeError, ValueError) as e:
            raise ValueError(f"the registration's {name} table's root doesn't read: {e}") from None
        domains[name] = dom
    head = Head(link, window, params.digest, tuple(entries))
    try:
        c = doc["certificate"]
        cert = Certificate(bytes.fromhex(c["key"]), link, Direction.EGRESS, window, bytes.fromhex(c["digest"]),
                           bytes.fromhex(c["signature"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the registration's certificate doesn't read: {e}") from None
    if cert.digest != head.digest():
        raise ValueError("the registration's certificate doesn't sign its head")
    if not cert.verify(key, session):
        raise ValueError("the registration's certificate doesn't verify under the network certifier's key in the "
                         "session")
    return head, domains


def no_head_digest(link: str, window: int, params: WardenParams) -> bytes:
    """The 64 bytes the network certifier signs to say that an execution phase of a link has no head (the module
    docstring's "The word that no head comes"). ValueError for a schedule whose execution phases don't each end in
    one of the two (`WardenParams.closes`)."""
    if not params.closes:
        raise ValueError("the schedule's execution phases don't each end in a head or in the word that none comes "
                         "(WardenParams.lengths is 2 where they do)")
    return tagged_sha512(NO_HEAD_TAG, b"".join([_text(link, "link"), _u(window, 8, "window"), params.digest]))


def attest_no_head(issuer: Issuer, link: str, window: int, params: WardenParams) -> Certificate | None:
    """The word that execution phase `window` of `link` has no head, signed by `issuer` (None when its signer
    refuses)."""
    return issuer.sign(link, Direction.EGRESS, window, no_head_digest(link, window, params))


def no_head(certificate: Certificate, params: WardenParams) -> dict[str, Any]:
    """The public record of the word that an execution phase has no head: what it is of, and its certificate."""
    return {"format": NO_HEAD, "link": certificate.link, "window": certificate.window,
            "parameters": params.digest.hex(),
            "certificate": {"key": certificate.key_id.hex(), "digest": certificate.digest.hex(),
                            "signature": certificate.signature.hex()}}


def check_no_head(doc: Mapping[str, Any], key: PublicKey, session: bytes, params: WardenParams, link: str,
                  window: int) -> None:
    """The verifier's read of the word that an execution phase has no head, under the network certifier's key and
    session and the schedule's parameters: ValueError naming what fails."""
    want = {"format": NO_HEAD, "link": link, "window": window, "parameters": params.digest.hex()}
    for k, v in want.items():
        if doc.get(k) != v:
            raise ValueError(f"the word's {k} is {doc.get(k)!r}, and the schedule's {v!r}")
    if set(doc) != set(want) | {"certificate"}:
        raise ValueError(f"the word holds the keys {sorted(set(want) | {'certificate'})} and no other")
    try:
        c = doc["certificate"]
        cert = Certificate(bytes.fromhex(c["key"]), link, Direction.EGRESS, window, bytes.fromhex(c["digest"]),
                           bytes.fromhex(c["signature"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the word's certificate doesn't read: {e}") from None
    if cert.digest != no_head_digest(link, window, params):
        raise ValueError("the word's certificate doesn't sign that the execution phase has no head")
    if not cert.verify(key, session):
        raise ValueError("the word's certificate doesn't verify under the network certifier's key in the session")


def port(head: Head, table: str, reads_from: int = 0) -> dict[str, Any]:
    """The verifier's own `--registered` port of a population of one proof block a row of the head's table `table`:
    proof block `reads_from + j` reads row j, a slot stride's every slot or a class's every row, held or not, so the
    head alone fixes its reads. ValueError for a table the head doesn't list."""
    if table not in [e.name for e in head.tables]:
        raise ValueError(f"the head lists {[e.name for e in head.tables]}, not a table {table!r}")
    e = head.table(table)
    entry = {"leaves": e.rows, "words": e.row_bytes * 8 // WORD_BITS, "word_bits": WORD_BITS, "domain": e.domain.hex()}
    return produced.port(value_name(head.link, table, head.window), entry, e.root.hex(), list(range(e.rows)), SCHEMA) | \
        ({G.READS_FROM: int(reads_from)} if reads_from else {})
