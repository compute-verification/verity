"""The ZK server's value service (`Flock/Firewall/Values.lean`, `flock-firewall values`), live PROTOCOL.md §10.2: every
value commitment's salt is the server's, derived from its own key, and no request takes one from the serving stack. Held
here to the Python reference: each leaf's salt `SHA-512(k ‖ L ‖ s ‖ be32 j ‖ be64 i ‖ ctr)` for `ctr` 0, 1 and 2, its
`b ‖ c` (`hm96.SHA512.commit_string`), its frame-v3-sha512 leaf and the table's root (`merkle.frame_hash`,
`merkle.root_from_subtrees`) and the table's tag; a commit body carrying a salt, a refs file of another length, a row past
its table and a forged tag, each refused; and the key file its owner's alone and kept. A registration is numbered on the
ZK server's record and admitted there (`core-firewall values` and `core-firewall session --salt-key`,
`verity/protocols/recursion/verifier/firewall/tests/test_firewall_registrations.py`), so this service, which has no record, registers
nothing and this loop admits no registration."""
from __future__ import annotations

import hashlib
import os
import socket
import stat
import struct
import subprocess
from pathlib import Path

import pytest
from test_lean_hello import BIN, built  # noqa: F401

from verity.commitments import hm96
from verity.commitments.merkle import frame_hash, root_from_subtrees

INST = hm96.SHA512
SCHEMAS = ["hm96-sha512/row/v1", "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1"]
SALT_LABEL = hashlib.sha512(b"verity/zk-server/value-salt/v1\0").digest()
TAG_LABEL = hashlib.sha512(b"verity/zk-server/value-commitment/v1\0").digest()
NOT_THE_SERVERS = ("its tag is not the ZK server's for its session, table, domain, leaf count, schema and root "
                   "(under its key, or an interior session's own while the service that drew it runs)")


def _u(i: int) -> bytes:
    return i.to_bytes(max(1, (i.bit_length() + 7) // 8), "big")


def salt(k: bytes, s: bytes, j: int, i: int) -> bytes:
    pre = k + SALT_LABEL + s + struct.pack(">IQ", j, i)
    return b"".join(hashlib.sha512(pre + bytes([c])).digest() for c in range(3))


def table_with(ys: list[bytes], tk: bytes, s: bytes, j: int, dom: bytes, schema: str,
               xs: list[bytes]) -> tuple[bytes, bytes, list[bytes]]:
    """The reference's root, tag under the key `tk` and leaves' `b ‖ c` of table `j` of the commit session `s`, leaf `i`'s
    salt `ys[i]`."""
    bcs = [INST.commit_string(INST.default_key, x, y) for x, y in zip(xs, ys, strict=True)]
    leaves = {(0, i): frame_hash(b"leaf", dom, _u(i), _u(i), schema.encode(), INST.tree_leaf(INST.default_key, bc))
              for i, bc in enumerate(bcs)}
    root = root_from_subtrees(dom, len(xs), leaves)
    tag = hashlib.sha512(tk + TAG_LABEL + s + struct.pack(">I", j) + dom + struct.pack(">Q", len(xs))
                         + bytes([SCHEMAS.index(schema)]) + root).digest()
    return root, tag, bcs


def table(k: bytes, s: bytes, j: int, dom: bytes, schema: str, xs: list[bytes]) -> tuple[bytes, bytes, list[bytes]]:
    """The reference's root, tag and leaves' `b ‖ c` of table `j` of the commit session `s`."""
    return table_with([salt(k, s, j, i) for i in range(len(xs))], k, s, j, dom, schema, xs)


def table_ref(s: bytes, j: int, dom: bytes, schema: str, n: int, root: bytes, tag: bytes) -> bytes:
    return s + struct.pack(">I", j) + dom + struct.pack(">Q", n) + bytes([SCHEMAS.index(schema)]) + root + tag


def refs(tables: list[bytes], rows: list[tuple[int, int]]) -> bytes:
    return struct.pack(">II", len(tables), len(rows)) + b"".join(tables) + b"".join(struct.pack(">IQ", t, i) for t, i in rows)


class Values:
    """A running `flock-firewall values` on the key file `key`."""

    def __init__(self, key: Path):
        self.p = subprocess.Popen([str(BIN), "values", "--salt-key", str(key)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        line = self.p.stdout.readline().decode()
        assert line.startswith("VALUES "), (line, self.p.stderr.read().decode() if self.p.poll() is not None else "")
        host, port = line.split()[1].split(":")
        self.addr = (host, int(port))

    def request(self, line: str, body: bytes = b"") -> bytes:
        """One request, sent whole before this side shuts down, and the whole answer."""
        with socket.create_connection(self.addr) as s:
            s.sendall(line.encode() + b"\n" + body)
            s.shutdown(socket.SHUT_WR)
            out = b""
            while b := s.recv(1 << 16):
                out += b
        return out

    def commit(self, tables: list[tuple[bytes, str, list[bytes]]],
               word: str = "commit") -> tuple[bytes, list[tuple[bytes, bytes, list[bytes]]]]:
        """Commits `tables` (domain id, schema, digests), an interior commit where `word` is `interior`: the session and
        each table's root, tag and `b ‖ c`."""
        line = f"{word} {len(tables)}" + "".join(f" {d.hex()} {len(xs)} {sc}" for d, sc, xs in tables)
        out = self.request(line, b"".join(x for _, _, xs in tables for x in xs))
        head, _, out = out.partition(b"\n")
        assert head.startswith(b"ok "), head
        s = bytes.fromhex(head[3:].decode())
        got = []
        for _ in tables:
            head, _, out = out.partition(b"\n")
            root, tag, n = head.decode().removeprefix("ok ").split()
            got.append((bytes.fromhex(root), bytes.fromhex(tag), [out[128 * i:128 * (i + 1)] for i in range(int(n))]))
            out = out[128 * int(n):]
        assert out == b""
        return s, got

    def salts(self, body: bytes) -> bytes | str:
        """The salts a refs file's rows take, or the refusal's reason."""
        out = self.request("salts", body)
        head, _, rest = out.partition(b"\n")
        if head.startswith(b"err "):
            return head[4:].decode()
        assert head == f"ok {len(rest)}".encode(), head
        return rest

    def register(self, s: bytes) -> tuple[int, bytes] | str:
        """The session count and record of the session `s`'s registration, or the refusal's reason."""
        head, _, rest = self.request(f"register {s.hex()}").partition(b"\n")
        if head.startswith(b"err "):
            return head[4:].decode()
        m, n = head.decode().removeprefix("ok ").split()
        assert int(n) == len(rest), head
        return int(m), rest

    def close(self):
        self.p.stdin.close()
        assert self.p.wait() == 0


@pytest.fixture
def values(tmp_path):
    v = Values(tmp_path / "salt.key")
    yield v, (tmp_path / "salt.key").read_bytes()
    v.close()


def _tables() -> list[tuple[bytes, str, list[bytes]]]:
    return [(os.urandom(64), SCHEMAS[0], [os.urandom(64) for _ in range(3)]),
            (os.urandom(64), SCHEMAS[2], [os.urandom(64)]),
            (os.urandom(64), SCHEMAS[1], [os.urandom(64) for _ in range(5)])]


def test_a_commit_is_the_references(values):
    """Each table's leaves' `b ‖ c`, root and tag are the reference's under the server's key at the session it drew, and the
    salts of the rows a refs file names are those leaves' salts, in its rows' order."""
    v, k = values
    ts = _tables()
    s, got = v.commit(ts)
    assert len(s) == 64
    for j, ((dom, sc, xs), g) in enumerate(zip(ts, got)):
        assert g == table(k, s, j, dom, sc, xs), j
    trefs = [table_ref(s, j, dom, sc, len(xs), got[j][0], got[j][1]) for j, (dom, sc, xs) in enumerate(ts)]
    rows = [(2, 4), (0, 0), (1, 0), (0, 2), (2, 0)]
    assert v.salts(refs(trefs, rows)) == b"".join(salt(k, s, t, i) for t, i in rows)
    assert v.salts(refs(trefs, [])) == b""


def test_each_commit_draws_its_own_session(values):
    """The same digests committed twice get two sessions, so two sets of salts: no commitment repeats another's."""
    v, _ = values
    ts = _tables()
    (s1, g1), (s2, g2) = v.commit(ts), v.commit(ts)
    assert s1 != s2 and all(a[0] != b[0] and set(a[2]).isdisjoint(b[2]) for a, b in zip(g1, g2))


def test_the_key_is_its_owners_and_kept(tmp_path):
    """The service creates its key file, 64 bytes only its owner reads and writes, and a service started again on it derives
    the same salts; a key file of another length is refused."""
    key = tmp_path / "salt.key"
    v = Values(key)
    ts = _tables()
    s, got = v.commit(ts)
    v.close()
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and len(key.read_bytes()) == 64
    trefs = [table_ref(s, j, dom, sc, len(xs), got[j][0], got[j][1]) for j, (dom, sc, xs) in enumerate(ts)]
    again = Values(key)
    assert again.salts(refs(trefs, [(0, 1)])) == salt(key.read_bytes(), s, 0, 1)
    again.close()
    (tmp_path / "short.key").write_bytes(os.urandom(63))
    p = subprocess.run([str(BIN), "values", "--salt-key", str(tmp_path / "short.key")], input=b"", capture_output=True)
    assert p.returncode == 2 and "63 bytes, where a salt key is 64" in p.stderr.decode()


@pytest.mark.parametrize("line, extra, why", [
    ("commit 1 {d} 3 hm96-sha512/row/v1", 192 * 3, "err a commit body of 768 bytes, where its tables' digests are 192"),
    ("commit 1 {d} 3 hm96-sha512/row/v1", -1, "err a commit body of 191 bytes, where its tables' digests are 192"),
    ("commit 0", 0, "err a commit request of 0 tables, where it takes 1 to 2^32 − 1"),
    ("commit 1 {d} 0 hm96-sha512/row/v1", 0, "err a table of 0 leaves, where it takes 1 to 2^64 − 1"),
    ("commit 1 {d} 3 sha512/row/v1", 0, "err a row schema `sha512/row/v1`, none of "),
    ("commit 1 00 3 hm96-sha512/row/v1", 0, "err a domain id `00` that is not 128 hex digits"),
    ("commit 2 {d} 3 hm96-sha512/row/v1", 0, "err a commit request with fewer words than its tables take"),
    ("commit 1 {d} 3 hm96-sha512/row/v1 x", 0, "err a commit request with more words than its tables take"),
    ("salt 1", 0, "err a request `salt 1`"),
])
def test_no_commit_takes_a_salt(values, line, extra, why):
    """A commit request's body is its digests and nothing else: a salt beside each digest, a byte short or a malformed line
    is refused, and nothing is committed for it."""
    v, _ = values
    xs = os.urandom(64 * 3)
    body = xs[:extra] if extra < 0 else xs + os.urandom(extra)
    out = v.request(line.format(d=os.urandom(64).hex()), body).decode(errors="replace")
    assert out.startswith(why) and out.count("\n") == 1, out


def test_salts_only_for_the_servers_tables(values, tmp_path):
    """A refs file of another length, a row past its table, a tag forged in any byte or made under another key: the whole
    request is refused, no salt answered."""
    v, _ = values
    ts = _tables()
    s, got = v.commit(ts)
    trefs = [table_ref(s, j, dom, sc, len(xs), got[j][0], got[j][1]) for j, (dom, sc, xs) in enumerate(ts)]
    good = refs(trefs, [(0, 2), (2, 4)])
    assert isinstance(v.salts(good), bytes)
    assert v.salts(good[:-1]).startswith(f"a refs file of {len(good) - 1} bytes, where 3 table refs and 2 row refs take")
    assert v.salts(refs(trefs, [(0, 3)])) == "a row ref to leaf 3 of table ref 0, past it"
    assert v.salts(refs(trefs, [(3, 0)])) == "a row ref to leaf 0 of table ref 3, past it"
    for at in (0, 64, 68, 132, 141, 205, 268):
        forged = bytearray(trefs[1])
        forged[at] ^= 1
        assert v.salts(refs([trefs[0], bytes(forged), trefs[2]], [(0, 0)])) == f"table ref 1: {NOT_THE_SERVERS}", at
    other = Values(tmp_path / "other.key")
    s2, got2 = other.commit(ts[:1])
    other.close()
    assert v.salts(refs([table_ref(s2, 0, ts[0][0], ts[0][1], 3, got2[0][0], got2[0][1])], [(0, 0)])) == f"table ref 0: {NOT_THE_SERVERS}"


def test_an_interior_commit_holds_its_own_salts(tmp_path):
    """An interior commit's leaves take salts the service drew from the operating system, not ones derived from its key:
    the salts `salts` answers for its rows make each leaf's `b ‖ c` and the table's root, and its tags are not under `k`
    (they are under a key the service drew for the session, which the ZK server's record holds:
    `test_firewall_registrations.py`). The same digests committed again take other salts, a refs file may name tables of
    both kinds, and a tag forged in any byte is refused. A service started again on the key holds no earlier interior
    session's key or salts, so it refuses that session's tables, and still answers a commit's."""
    key = tmp_path / "salt.key"
    v = Values(key)
    k = key.read_bytes()
    ts = _tables()
    s, got = v.commit(ts, "interior")
    trefs = [table_ref(s, j, dom, sc, len(xs), got[j][0], got[j][1]) for j, (dom, sc, xs) in enumerate(ts)]
    ys = [v.salts(refs(trefs, [(j, i) for i in range(len(xs))])) for j, (_, _, xs) in enumerate(ts)]
    for j, ((dom, sc, xs), g, y) in enumerate(zip(ts, got, ys)):
        per = [y[192 * i:192 * (i + 1)] for i in range(len(xs))]
        root, tag, bcs = table_with(per, k, s, j, dom, sc, xs)
        assert len(y) == 192 * len(xs) and (g[0], g[2]) == (root, bcs), j
        assert g[1] != tag and all(p != salt(k, s, j, i) for i, p in enumerate(per)), j
    s2, got2 = v.commit(ts, "interior")
    again2 = refs([table_ref(s2, 0, ts[0][0], ts[0][1], 3, got2[0][0], got2[0][1])], [(0, 0), (0, 1), (0, 2)])
    assert s2 != s and v.salts(again2) != ys[0]
    b, gotb = v.commit(ts[:1])
    bref = table_ref(b, 0, ts[0][0], ts[0][1], 3, gotb[0][0], gotb[0][1])
    assert v.salts(refs([trefs[1], bref], [(1, 2), (0, 0)])) == salt(k, b, 0, 2) + ys[1]
    for at in (0, 64, 141, 205, 268):
        forged = bytearray(trefs[2])
        forged[at] ^= 1
        assert v.salts(refs([bytes(forged)], [(0, 0)])) == f"table ref 0: {NOT_THE_SERVERS}", at
    v.close()
    again = Values(key)
    assert again.salts(refs([bref, trefs[0]], [(0, 0)])) == f"table ref 1: {NOT_THE_SERVERS}"
    assert again.salts(refs([bref], [(0, 1)])) == salt(k, b, 0, 1)
    again.close()


@pytest.mark.parametrize("line, body, why", [
    ("register {s}", b"x", "err a register request with a body of 1 bytes, where it takes none"),
    ("register 00", b"", "err a session `00` that is not 128 hex digits"),
    ("register", b"", "err a request `register`"),
    ("register {s} {s}", b"", "err a request `register "),
])
def test_a_malformed_registration_is_refused(values, line, body, why):
    """A register request with a body, a session of another length or more than one session is refused, and so is a
    well-formed one: this service has no record of the ZK server's to number it on."""
    v, _ = values
    s, _ = v.commit(_tables())
    out = v.request(line.format(s=s.hex()), body).decode(errors="replace")
    assert out.startswith(why) and out.count("\n") == 1, out
    assert v.register(s) == "a registration is numbered on the ZK server's record: `core-firewall values --state DIR`"


def test_the_loop_without_a_record_admits_no_registration(tmp_path):
    """`flock-firewall session` keeps no record of the ZK server's, so it refuses `--salt-key` before it listens (`refused
    WHY`, exit 2)."""
    for f in ("circuit", "pub"):
        (tmp_path / f).write_bytes(b"x")
    p = subprocess.run([str(BIN), "session", "--verifier", "127.0.0.1:9", "--circuit", str(tmp_path / "circuit"), "--public",
                        str(tmp_path / "pub"), "--salt-key", str(tmp_path / "salt.key")], input=b"", capture_output=True)
    assert p.returncode == 2 and p.stdout.decode() == (
        "refused --salt-key: a statement's value commitments are admitted on the ZK server's record (core-firewall session "
        "--state DIR --salt-key FILE)\n"), (p.stdout, p.stderr[-400:])
