"""The ZK server's value service (`Flock/Firewall/Values.lean`, `flock-firewall values`) and the session loop's check of a
statement's value commitments (`Flock/Firewall/Salts.lean`, `flock-firewall session --salt-key --commitments`), live
PROTOCOL.md §10.2: every value commitment's salt is the server's, derived from its own key, and no request takes one from
the serving stack. Held here to the Python reference: each leaf's salt `SHA-512(k ‖ L ‖ s ‖ be32 j ‖ be64 i ‖ ctr)` for
`ctr` 0, 1 and 2, its `b ‖ c` (`hm96.SHA512.commit_string`), its frame-v3-sha512 leaf and the table's root
(`merkle.frame_hash`, `merkle.root_from_subtrees`) and the table's tag; a commit body carrying a salt, a refs file of another
length, a row past its table and a forged tag, each refused; the key file its owner's alone and kept; a registration's
record counting the sessions its epoch opened, each session registering once and in its own epoch only; and the loop
serving a statement only when every hm96 tree of its public file is a table of a record of the server's
(`test_lean_hello.py`'s fixture)."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import struct
import subprocess
from pathlib import Path

import pytest
from test_lean_hello import BIN, FIXTURE, STATEMENT, built, zk  # noqa: F401
from test_lean_session import Auditor, Loop, plain

from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import frame_hash, root_from_subtrees

INST = hm96.SHA512
SCHEMAS = ["hm96-sha512/row/v1", "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1"]
SALT_LABEL = hashlib.sha512(b"verity/zk-server/value-salt/v1\0").digest()
TAG_LABEL = hashlib.sha512(b"verity/zk-server/value-commitment/v1\0").digest()
REG_LABEL = hashlib.sha512(b"verity/zk-server/registration/v1\0").digest()
NOT_THE_SERVERS = "its tag is not the ZK server's for its session, table, domain, leaf count, schema and root"
NOT_OPENED = "is none the current epoch opened: a session registers once, in the epoch that opened it"
NOT_A_RECORD = "its tag is not the ZK server's for its session count and table refs"


def _u(i: int) -> bytes:
    return i.to_bytes(max(1, (i.bit_length() + 7) // 8), "big")


def salt(k: bytes, s: bytes, j: int, i: int) -> bytes:
    pre = k + SALT_LABEL + s + struct.pack(">IQ", j, i)
    return b"".join(hashlib.sha512(pre + bytes([c])).digest() for c in range(3))


def table(k: bytes, s: bytes, j: int, dom: bytes, schema: str, xs: list[bytes]) -> tuple[bytes, bytes, list[bytes]]:
    """The reference's root, tag and leaves' `b ‖ c` of table `j` of the commit session `s`."""
    bcs = [INST.commit_string(INST.default_key, x, salt(k, s, j, i)) for i, x in enumerate(xs)]
    leaves = {(0, i): frame_hash(b"leaf", dom, _u(i), _u(i), schema.encode(), INST.tree_leaf(INST.default_key, bc))
              for i, bc in enumerate(bcs)}
    root = root_from_subtrees(dom, len(xs), leaves)
    tag = hashlib.sha512(k + TAG_LABEL + s + struct.pack(">I", j) + dom + struct.pack(">Q", len(xs))
                         + bytes([SCHEMAS.index(schema)]) + root).digest()
    return root, tag, bcs


def table_ref(s: bytes, j: int, dom: bytes, schema: str, n: int, root: bytes, tag: bytes) -> bytes:
    return s + struct.pack(">I", j) + dom + struct.pack(">Q", n) + bytes([SCHEMAS.index(schema)]) + root + tag


def refs(tables: list[bytes], rows: list[tuple[int, int]]) -> bytes:
    return struct.pack(">II", len(tables), len(rows)) + b"".join(tables) + b"".join(struct.pack(">IQ", t, i) for t, i in rows)


def record(k: bytes, m: int, trefs: list[bytes]) -> bytes:
    """The reference's registration record of a session whose table refs are `trefs`, of an epoch of `m` sessions."""
    head = struct.pack(">QI", m, len(trefs))
    return head + b"".join(trefs) + hashlib.sha512(k + REG_LABEL + head + b"".join(trefs)).digest()


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

    def commit(self, tables: list[tuple[bytes, str, list[bytes]]]) -> tuple[bytes, list[tuple[bytes, bytes, list[bytes]]]]:
        """Commits `tables` (domain id, schema, digests): the session and each table's root, tag and `b ‖ c`."""
        line = f"commit {len(tables)}" + "".join(f" {d.hex()} {len(xs)} {sc}" for d, sc, xs in tables)
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


def _trefs(s: bytes, ts, got) -> list[bytes]:
    return [table_ref(s, j, dom, sc, len(xs), got[j][0], got[j][1]) for j, (dom, sc, xs) in enumerate(ts)]


def test_a_registration_counts_its_epochs_sessions(values):
    """A registration's record is the reference's: the number of sessions its epoch opened, the registered session's table
    refs, each with its tag, and the record tag. It closes the epoch: the stack that committed its values three times and
    registered the last gets a record that says three, and none of the other two, nor the registered one again, registers
    after it; the next epoch counts from none."""
    v, k = values
    ts = _tables()
    s1, g1 = v.commit(ts)
    assert v.register(s1) == (1, record(k, 1, _trefs(s1, ts, g1)))
    assert v.register(s1) == f"the session {s1.hex()} {NOT_OPENED}"
    (s2, _), (s3, _), (s4, g4) = v.commit(ts), v.commit(ts[:1]), v.commit(ts)
    assert v.register(s4) == (3, record(k, 3, _trefs(s4, ts, g4)))
    for s in (s2, s3, s4, os.urandom(64)):
        assert v.register(s) == f"the session {s.hex()} {NOT_OPENED}"
    s5, g5 = v.commit(ts[1:])
    assert v.register(s5) == (1, record(k, 1, _trefs(s5, ts[1:], g5)))


def test_a_restarted_service_registers_no_earlier_session(tmp_path):
    """The epoch lives only in the service's process: started again on its key, it registers none of the sessions an earlier
    run opened, so a restart gives the stack no session the record doesn't count."""
    key = tmp_path / "salt.key"
    v = Values(key)
    s, _ = v.commit(_tables())
    v.close()
    again = Values(key)
    assert again.register(s) == f"the session {s.hex()} {NOT_OPENED}"
    again.close()


@pytest.mark.parametrize("line, body, why", [
    ("register {s}", b"x", "err a register request with a body of 1 bytes, where it takes none"),
    ("register 00", b"", "err a session `00` that is not 128 hex digits"),
    ("register", b"", "err a request `register`"),
    ("register {s} {s}", b"", "err a request `register "),
])
def test_a_malformed_registration_is_refused(values, line, body, why):
    """A register request with a body, a session of another length or more than one session is refused, and the epoch's
    session still registers after it."""
    v, _ = values
    s, _ = v.commit(_tables())
    out = v.request(line.format(s=s.hex()), body).decode(errors="replace")
    assert out.startswith(why) and out.count("\n") == 1, out
    assert v.register(s)[0] == 1


def _trees(public: bytes) -> list[tuple[str, bytes, str, int]]:
    """A public file's hm96 trees: each row's name, domain id, schema and leaf count."""
    h = json.loads(public[:public.index(b"\n")])
    fv, shared = h["frame_v3"], h.get("shared_rows")
    return [(name, bytes.fromhex(fv["domain_ids"][name]), fv["schemas"][name], shared[name] if shared else h["instances"])
            for name in sorted(fv["roots"]) if fv["schemas"][name] in SCHEMAS]


def _with_roots(public: bytes, roots: dict[str, bytes]) -> bytes:
    nl = public.index(b"\n")
    h = json.loads(public[:nl])
    h["frame_v3"]["roots"].update({k: r.hex() for k, r in roots.items()})
    return json.dumps(h, separators=(",", ":"), sort_keys=True).encode() + public[nl:]


def _session(public: Path, *flags: str) -> subprocess.CompletedProcess:
    args = ["--statement", STATEMENT, "--circuit", str(FIXTURE / "stage" / "circuit.txt"), "--public", str(public)]
    return subprocess.run([str(BIN), "session", "--verifier", "127.0.0.1:9", *args, *flags], input=b"", capture_output=True)


def test_the_loop_serves_only_the_servers_commitments(zk, tmp_path):
    """With the server's key and a records file of its registrations, the loop serves a statement every hm96 tree of whose
    public file is a table of one of them, at its domain, leaf count, schema and root, and refuses before any session
    (`refused WHY`, exit 2) the fixture's statement as the stack committed it, the statement on a session the server
    committed but never registered, a record forged in any byte or answered under another key, a records file cut short,
    and either flag without the other."""
    public = (zk / "stage" / "pub-16.bin").read_bytes()
    trees = _trees(public)
    assert trees
    key, other = tmp_path / "salt.key", tmp_path / "other.key"
    v = Values(key)
    s, got = v.commit([(dom, sc, [os.urandom(64) for _ in range(n)]) for _, dom, sc, n in trees])
    m, rec = v.register(s)
    s1, got1 = v.commit([(dom, sc, [os.urandom(64) for _ in range(n)]) for _, dom, sc, n in trees])
    v.close()
    w = Values(other)
    s2, got2 = w.commit([(dom, sc, [os.urandom(64) for _ in range(n)]) for _, dom, sc, n in trees])
    _, rec2 = w.register(s2)
    w.close()
    ours = tmp_path / "pub.bin"
    ours.write_bytes(_with_roots(public, {name: g[0] for (name, *_), g in zip(trees, got)}))
    assert m == 1
    (tmp_path / "records.bin").write_bytes(rec)
    flags = ["--salt-key", str(key), "--commitments", str(tmp_path / "records.bin")]

    lp = Loop(Auditor(plain), ["--statement", STATEMENT, "--circuit", str(zk / "stage" / "circuit.txt"), "--public", str(ours), *flags])
    lp.close()

    def refused(public: Path, *flags: str) -> str:
        p = _session(public, *flags)
        assert p.returncode == 2 and p.stdout.decode().startswith("refused "), (p.returncode, p.stdout, p.stderr[-400:])
        return p.stdout.decode().removeprefix("refused ").strip()

    def with_records(*recs: bytes) -> list[str]:
        (tmp_path / "given.bin").write_bytes(b"".join(recs))
        return ["--salt-key", str(key), "--commitments", str(tmp_path / "given.bin")]

    stack = refused(zk / "stage" / "pub-16.bin", *flags)
    assert stack.startswith("--commitments: row ") and "its root is no value commitment of the ZK server's" in stack
    unregistered = tmp_path / "pub1.bin"
    unregistered.write_bytes(_with_roots(public, {name: g[0] for (name, *_), g in zip(trees, got1)}))
    assert "its root is no value commitment of the ZK server's" in refused(unregistered, *flags)
    for at in (0, 7, 8, 11, 12, len(rec) - 65, len(rec) - 1):
        forged = bytearray(rec)
        forged[at] ^= 1
        why = refused(ours, *with_records(bytes(forged)))
        assert why == f"--commitments: record 0: {NOT_A_RECORD}" or why.startswith("--commitments: a records file"), (at, why)
    assert refused(ours, *with_records(rec, rec2)) == f"--commitments: record 1: {NOT_A_RECORD}"
    assert refused(ours, *with_records(rec[:-1])).startswith("--commitments: a records file whose record at byte 0")
    (tmp_path / "pub2.bin").write_bytes(_with_roots(public, {name: g[0] for (name, *_), g in zip(trees, got2)}))
    assert refused(tmp_path / "pub2.bin", *with_records(rec2)) == f"--commitments: record 0: {NOT_A_RECORD}"
    assert refused(ours, "--salt-key", str(key)).startswith("--salt-key: the loop serves only a statement whose value commitments")
    assert refused(ours, "--commitments", str(tmp_path / "records.bin")).startswith("--commitments: the loop checks them under")
    assert refused(ours, "--salt-key", str(tmp_path / "none.key"), "--commitments", str(tmp_path / "records.bin")).endswith(
        "no such file (`flock-firewall values` creates it)")
