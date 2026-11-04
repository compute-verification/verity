"""The ZK server's value service (live PROTOCOL.md §10.2): with ``FC_FIREWALL_VALUES`` set, every value commitment whose root or
leaves go public is the service's. verity_vllm sends each row's unsalted 64-byte digest and gets back each leaf's ``b ‖ c``, each
table's root and the server's tag, so the server, not verity_vllm, chooses every salt. Nothing here draws, derives, stores or passes a
salt, reads the server's key, or runs the service.

* `address`: the service's ``HOST:PORT`` (``FC_FIREWALL_VALUES``), or None: unset, every call site commits as it did before, under
  salts of its own.
* `Session`: one commit session, a window's or a registration's: one `Session.commit` of every table (`Table`: domain id, schema and
  the rows' digests, table-major in one request), then one `Session.register`. A session commits once and registers once, and only
  in the epoch that opened it; a retry is a new session, which the epoch's count ``m`` charges as a grinding choice, so a caller
  opens exactly the sessions its schedule names and never resumes one.
* `Committed`: a table as the service committed it: its session, index, domain id, leaf count, schema, root and tag (`ref`, the
  269-byte table ref) and every leaf's ``b ‖ c``, the tree's levels rebuilt here from them (leaf ``row_leaf(D, i, tree_leaf(K, bc),
  schema)``, frame-v3-sha512) and refused unless their root is the service's: verity_vllm never registers a root of its own.
  `Committed.of` reads one back from its ref and ``b ‖ c`` (a window's, on a sample).
* `Registration`: what the service appended to the ZK server's record for the session: its number ``n``, the epoch's session count
  ``m``, the table refs and the record itself (whose tag only the server can check).
* `refs_file`: what `flock-circuit --zk --salt-refs` reads on a sample: the table refs of the tables a statement reads, then one row
  ref per salt of its inputs file, in that file's order (`instance_rows`, `table_rows`).
* `refuse_own`: a call site that still draws its own salts refuses under ``FC_FIREWALL_VALUES``, naming what it waits on.
"""
from __future__ import annotations

import dataclasses
import os
import socket
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import frame_hash, levels, merkle_depth
from verity_vllm import settings

ENV = "FC_FIREWALL_VALUES"
#: the row schemas the service commits, by the index a table ref names (`Salts.schemaIndex`)
SCHEMAS = ("hm96-sha512/row/v1", "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1")
DIGEST, BC, REF = 64, 128, 269
#: a registration record's head: ``be64 n ‖ be64 m ‖ be32 T``
HEAD = 20
_INST = hm96.SHA512
_CHUNK = 1 << 20


class Refused(RuntimeError):
    """The service answered ``err WHY``, its answer is not what was asked, or a call site can't commit under its salts."""


def address() -> str | None:
    """The value service's ``HOST:PORT`` from ``FC_FIREWALL_VALUES``, or None when it is unset."""
    return settings.get(ENV) or None


def refuse_own(site: str, why: str) -> None:
    """Refuse ``site``'s commitment under its own salts when ``FC_FIREWALL_VALUES`` is set: the ZK server chooses every public
    salt, and ``why`` says what the site waits on before it commits through the service."""
    if address() is not None:
        raise Refused(f"{site} commits under salts of its own, and with {ENV} set the ZK server chooses every public value "
                      f"commitment's salts: {why}")


@dataclass(frozen=True)
class Table:
    """One table of a commit: its domain id (64 bytes), row schema (`SCHEMAS`) and its rows' unsalted digests, ``n`` × 64 bytes,
    as bytes or as a file holding exactly them (a registration's tables, streamed from disk)."""

    domain: bytes
    schema: str
    digests: bytes | Path

    @property
    def n(self) -> int:
        size = os.path.getsize(self.digests) if isinstance(self.digests, Path) else len(self.digests)
        return size // DIGEST

    def check(self) -> None:
        size = os.path.getsize(self.digests) if isinstance(self.digests, Path) else len(self.digests)
        if len(self.domain) != DIGEST or self.schema not in SCHEMAS or not size or size % DIGEST:
            raise ValueError(f"a table is a 64-byte domain id, one of {', '.join(SCHEMAS)} and at least one 64-byte digest, not "
                             f"{len(self.domain)} bytes, {self.schema!r} and {size} bytes")

    def chunks(self) -> Iterable[bytes]:
        if not isinstance(self.digests, Path):
            yield bytes(self.digests)
            return
        with open(self.digests, "rb") as f:
            while b := f.read(_CHUNK):
                yield b


def _u(i: int) -> bytes:
    return i.to_bytes(max(1, (i.bit_length() + 7) // 8), "big")


def tree(domain: bytes, schema: str, bcs: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    """The frame-v3-sha512 levels over the leaves ``row_leaf(D, i, tree_leaf(K, bc_i), schema)`` of ``bcs`` under domain id
    ``domain``, padded to a power of two: the leaves first, the root last."""
    tag = schema.encode()
    leaves = [frame_hash(b"leaf", domain, _u(i), _u(i), tag, _INST.tree_leaf(_INST.default_key, bc)) for i, bc in enumerate(bcs)]
    pads = [frame_hash(b"pad", domain, _u(r)) for r in range(len(leaves), 1 << merkle_depth(len(leaves)))]
    return levels(domain, leaves + pads)


@dataclass(frozen=True)
class Committed:
    """Table ``index`` of commit session ``session``, as the service committed it: ``root`` and ``tag`` its answer's, ``bcs`` every
    leaf's ``b ‖ c`` (public data) and ``levels`` the tree rebuilt from them, whose root is ``root``. A caller that streams a large
    commit to disk (`Session.commit`'s ``sink``) gets it back without ``bcs`` and ``levels``."""

    session: bytes
    index: int
    domain: bytes
    schema: str
    n: int
    root: bytes
    tag: bytes
    bcs: tuple[bytes, ...] = ()
    levels: tuple[tuple[bytes, ...], ...] = ()

    def ref(self) -> bytes:
        """The table ref: ``s ‖ be32 j ‖ D ‖ be64 N ‖ u8 schema ‖ root ‖ tag``."""
        return (self.session + struct.pack(">I", self.index) + self.domain + struct.pack(">Q", self.n)
                + bytes([SCHEMAS.index(self.schema)]) + self.root + self.tag)

    @classmethod
    def of(cls, ref: bytes, bcs: Sequence[bytes] = ()) -> Committed:
        """The table ``ref`` names, read back (a window's ``served_rows/w<k>.json`` on a sample): with ``bcs``, every leaf's
        ``b ‖ c``, its tree rebuilt and refused unless its root is the ref's."""
        if len(ref) != REF or ref[140] >= len(SCHEMAS):
            raise ValueError(f"a table ref is {REF} bytes naming one of {len(SCHEMAS)} schemas, not {len(ref)} bytes")
        (j,), (n,) = struct.unpack(">I", ref[64:68]), struct.unpack(">Q", ref[132:140])
        c = cls(ref[:64], j, ref[68:132], SCHEMAS[ref[140]], n, ref[141:205], ref[205:])
        return _held(c, bcs, "the table ref") if bcs else c

    def path(self, i: int) -> tuple[bytes, ...]:
        """Leaf ``i``'s frame-v3 path: its siblings, leaf level first."""
        if not 0 <= i < self.n or not self.levels:
            raise IndexError(f"leaf {i} of a table of {self.n} whose levels this side holds" if self.levels else "no levels held")
        out = []
        for level in self.levels[:-1]:
            out.append(level[i ^ 1])
            i >>= 1
        return tuple(out)


def _held(c: Committed, bcs: Sequence[bytes], what: str) -> Committed:
    """``c`` with its leaves' ``b ‖ c`` and the tree rebuilt from them, refused unless that tree's root is ``c``'s."""
    if len(bcs) != c.n or any(len(bc) != BC for bc in bcs):
        raise Refused(f"{what}: {len(bcs)} b ‖ c for a table of {c.n} leaves, {BC} bytes each")
    lv = tree(c.domain, c.schema, bcs)
    if lv[-1][0] != c.root:
        raise Refused(f"{what}: the tree over its b ‖ c has root {lv[-1][0].hex()[:16]}…, not the root {c.root.hex()[:16]}… named")
    return dataclasses.replace(c, bcs=tuple(bytes(bc) for bc in bcs), levels=lv)


@dataclass(frozen=True)
class Registration:
    """The session's registration on the ZK server's record: its number, the count of sessions its epoch opened (public: the
    stack's choice among them is a term of the guarantee), its table refs and the record ``be64 n ‖ be64 m ‖ be32 T ‖ refs ‖
    tag``."""

    number: int
    sessions: int
    refs: tuple[bytes, ...]
    record: bytes

    def entry(self) -> dict:
        return {"number": self.number, "sessions": self.sessions, "record": self.record.hex()}


def _split(host_port: str) -> tuple[str, int]:
    host, sep, port = host_port.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"{ENV}={host_port!r}: the value service's HOST:PORT (`core-firewall values`' VALUES line)")
    return host, int(port)


def _line(f: BinaryIO) -> str:
    raw = f.readline()
    if not raw.endswith(b"\n"):
        raise Refused(f"the value service's answer ended inside a line ({raw[:80]!r})")
    text = raw[:-1].decode()
    if text.startswith("err "):
        raise Refused(f"the value service refused: {text[4:]}")
    if not text.startswith("ok "):
        raise Refused(f"the value service answered {text[:120]!r}, neither ok nor err")
    return text[3:]


def _exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise Refused(f"the value service's answer ended {n - len(b)} bytes short")
    return b


def _hex(word: str, n: int, what: str) -> bytes:
    try:
        b = bytes.fromhex(word)
    except ValueError:
        b = b""
    if len(b) != n:
        raise Refused(f"the value service's {what} {word[:40]!r} is not {2 * n} hex digits")
    return b


class Session:
    """One commit session of the value service at ``address`` (default `address()`): `commit` once, then `register` once."""

    def __init__(self, at: str | None = None) -> None:
        at = at or address()
        if at is None:
            raise Refused(f"no value service: {ENV} names none")
        self.address = _split(at)
        self.id: bytes | None = None
        self.tables: tuple[Committed, ...] = ()
        self.registration: Registration | None = None

    def _request(self, line: str, body: Iterable[bytes], read: Callable[[BinaryIO], object]) -> object:
        """One request on a connection of its own: the line and body sent whole, the write side shut, the answer read to EOF."""
        with socket.create_connection(self.address) as s:
            s.sendall(line.encode() + b"\n")
            for chunk in body:
                s.sendall(chunk)
            s.shutdown(socket.SHUT_WR)
            with s.makefile("rb") as f:
                out = read(f)
                if rest := f.read(1):
                    raise Refused(f"the value service's answer runs past its end ({rest!r} …)")
        return out

    def commit(self, tables: Sequence[Table], sink: Callable[[Committed], None] | None = None) -> tuple[Committed, ...]:
        """Every table of the session in one ``commit`` request: each table's `Committed`, its tree rebuilt from the service's
        ``b ‖ c`` and refused unless its root is the service's. ``sink``: each table handed over as it is read and kept here without
        its ``b ‖ c`` and levels, so a registration of many tables holds one at a time."""
        if self.id is not None:
            raise Refused("a commit session commits once: a retry is a new session, which the epoch's count charges")
        if not tables:
            raise ValueError("a commit takes at least one table")
        for t in tables:
            t.check()
        line = f"commit {len(tables)}" + "".join(f" {t.domain.hex()} {t.n} {t.schema}" for t in tables)

        def read(f: BinaryIO) -> tuple[bytes, list[Committed]]:
            s = _hex(_line(f), DIGEST, "session")
            got = []
            for j, t in enumerate(tables):
                words = _line(f).split()
                if len(words) != 3 or not words[2].isdigit() or int(words[2]) != t.n:
                    raise Refused(f"table {j}: the value service answered {' '.join(words)[:80]!r}, not its root, tag and {t.n} leaves")
                root, tag = _hex(words[0], DIGEST, "root"), _hex(words[1], DIGEST, "tag")
                raw = _exact(f, BC * t.n)
                c = _held(Committed(s, j, t.domain, t.schema, t.n, root, tag), [raw[BC * i:BC * (i + 1)] for i in range(t.n)],
                          f"table {j} as the service answered it")
                if sink is not None:
                    sink(c)
                    c = dataclasses.replace(c, bcs=(), levels=())
                got.append(c)
            return s, got

        s, got = self._request(line, (b for t in tables for b in t.chunks()), read)
        self.id, self.tables = s, tuple(got)
        return self.tables

    def bcs(self, tables: Sequence[tuple[bytes, str, bytes]]) -> list[tuple[bytes, ...]]:
        """`commit` of ``(domain id, schema, digests)`` tables, answering each one's ``b ‖ c``: a staging's commit
        (`verity_flock.circuit.write`'s ``committed``)."""
        return [c.bcs for c in self.commit([Table(d, sc, xs) for d, sc, xs in tables])]

    def register(self) -> Registration:
        """The session's registration on the ZK server's record, which closes its epoch: refused for a session that never
        committed, has registered, or belongs to an epoch an earlier registration closed."""
        if self.id is None:
            raise Refused("a session registers what it committed: commit first")
        if self.registration is not None:
            raise Refused("a session registers once")

        def read(f: BinaryIO) -> tuple[int, bytes]:
            words = _line(f).split()
            if len(words) != 2 or not all(w.isdigit() for w in words):
                raise Refused(f"the value service answered {' '.join(words)[:80]!r} to register, not its session count and length")
            return int(words[0]), _exact(f, int(words[1]))

        m, record = self._request(f"register {self.id.hex()}", (), read)
        refs = tuple(c.ref() for c in self.tables)
        n, m2, T = struct.unpack(">QQI", record[:HEAD]) if len(record) >= HEAD else (None, None, None)
        if (len(record) != HEAD + REF * len(refs) + DIGEST or (m2, T) != (m, len(refs))
                or record[HEAD:HEAD + REF * len(refs)] != b"".join(refs)):
            raise Refused("the registration record is not this session's: its session count or table refs differ")
        self.registration = Registration(n, m, refs, record)
        return self.registration


def stored(tables: Sequence[Committed]) -> Callable[[Sequence[tuple[bytes, str, bytes]]], list[tuple[bytes, ...]]]:
    """A staging's ``committed`` on a sample: each table's ``b ‖ c`` as the service committed it before any coin, refused for a
    table (domain id, schema, leaf count) it did not commit; nothing is committed again."""
    by = {(c.domain, c.schema, c.n): c for c in tables}

    def answer(asked: Sequence[tuple[bytes, str, bytes]]) -> list[tuple[bytes, ...]]:
        out = []
        for d, sc, xs in asked:
            c = by.get((bytes(d), sc, len(xs) // DIGEST))
            if c is None or not c.bcs:
                raise Refused(f"the statement reads a table (domain {bytes(d).hex()[:16]}…, {sc}, {len(xs) // DIGEST} leaves) the "
                              "value service did not commit before the coins")
            out.append(c.bcs)
        return out
    return answer


def instance_rows(n: int, ports: int) -> list[tuple[int, int]]:
    """A per-instance inputs file's salts in its order: instance-major, ports in order (row ``i·np + p`` is leaf ``i`` of port
    ``p``'s table)."""
    return [(p, i) for i in range(n) for p in range(ports)]


def table_rows(counts: Sequence[int]) -> list[tuple[int, int]]:
    """A shared-row inputs file's salts in its order: its tables port-major (leaf ``r`` of port ``p``'s table)."""
    return [(p, r) for p, c in enumerate(counts) for r in range(c)]


def refs_file(tables: Sequence[Committed], rows: Sequence[tuple[int, int]]) -> bytes:
    """``be32 T ‖ be32 R ‖`` the ``T`` table refs ``‖`` the ``R`` row refs (``be32 t ‖ be64 i``, ``t`` an index into ``tables``)."""
    for t, i in rows:
        if not 0 <= t < len(tables) or not 0 <= i < tables[t].n:
            raise ValueError(f"row ref ({t}, {i}) names no leaf of the {len(tables)} tables")
    return (struct.pack(">II", len(tables), len(rows)) + b"".join(c.ref() for c in tables)
            + b"".join(struct.pack(">IQ", t, i) for t, i in rows))


__all__ = ["ENV", "SCHEMAS", "Committed", "Refused", "Registration", "Session", "Table", "address", "instance_rows", "refs_file",
           "refuse_own", "stored", "table_rows", "tree"]
