"""Rule 5 through the Lean ZK server's processes: `core-firewall values --state DIR` numbers each registration on the
ZK server's record (`Firewall.Record.register`), its record `be64 n` first and its record tag over that number, and
`core-firewall session --salt-key FILE` admits the registrations a statement's value commitments read, each once and
oldest first (`admitReads`), at the auditor's answer to its outer session's `Register` and before the prover receives
it. A statement that reads one passed over, or whose roots no registration of the server's holds, or holds under
another key, is its recursive session's stop, and of it only `Register` reaches the auditor. An interior session's
registration is under a key the service drew for the session, which the record holds beside it with that key's tag
under the server's key. A tree the verifier lists after the draw in that answer is known content: it reads no
registration and is recorded, the loop cuts the list off before the prover receives the answer, and a list that is not
exactly the statement's own known trees is the session's stop. Each record of commit sessions' registrations is the one
`verity_flock.firewall` writes for the same ops, byte for byte.
The statements are public files' headers only (`Flock.Firewall.Salts.treesOf` reads no more), so no session sets up
(`test_firewall_session_loop`)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import stat
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_firewall_session_loop import REGISTER, Auditor, Loop, call, record, refusal, slot_left

import _release as R
import verity_difftest as D
from verity_flock import firewall as FW

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SCHEMAS = ["hm96-sha512/row/v1", "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1"]
REG_LABEL = hashlib.sha512(b"verity/zk-server/registration/v1\0").digest()
TAG_LABEL = hashlib.sha512(b"verity/zk-server/value-commitment/v1\0").digest()
KEY_TAG_LABEL = hashlib.sha512(b"verity/zk-server/interior-key-tag/v1\0").digest()
NOT_TAGGED = "its record tag is not the ZK server's for its number, session count and table refs"
NOT_OPENED = "is none the current epoch opened: a session registers once, in the epoch that opened it"
ONE_SESSION = ("the current epoch's session is not yet registered: an epoch opens one commit session (`register S` registers "
               "it, and the next commit opens the next epoch's)")
NO_HOLDER = ("its root is no value commitment of the ZK server's (no registration on its record holds a table of its "
             "domain, leaf count, schema and root)")
CUT = "the prover ended the session before its Finish"
LEAVES = 3
#: every statement's hm96 trees here: name, domain id and schema, each of `LEAVES` leaves
TREES = [("a", os.urandom(64), SCHEMAS[0]), ("b", os.urandom(64), SCHEMAS[2])]
#: a tree at known content: none of a registration's, at a domain of its own
K = ("k", os.urandom(64), SCHEMAS[1])
#: the auditor's answer to `Register` as the prover receives it: the draw of a sample of none
DRAW = b"\x06" + struct.pack("<I", 4) + b"null"
LIST = "the verifier's list of known trees"
OTHER = "the verifier lists its domain as known content with another leaf count, schema or root"


@pytest.fixture(scope="module")
def exe():
    return D.build(R.PKG, "core-firewall")


class Values:
    """A running `core-firewall values` on the state directory `state` and the key file `key`."""

    def __init__(self, exe: Path, state: Path, key: Path):
        self.p = subprocess.Popen([str(exe), "values", "--state", str(state), "--salt-key", str(key)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        line = self.p.stdout.readline().decode()
        assert line.startswith("VALUES "), (line, self.p.stderr.read().decode() if self.p.poll() is not None else "")
        host, port = line.split()[1].split(":")
        self.addr = (host, int(port))

    def request(self, line: str, body: bytes = b"") -> bytes:
        with socket.create_connection(self.addr) as s:
            s.sendall(line.encode() + b"\n" + body)
            s.shutdown(socket.SHUT_WR)
            out = b""
            while b := s.recv(1 << 16):
                out += b
        return out

    def commit(self, trees=TREES, word: str = "commit") -> tuple[bytes, list[tuple[bytes, bytes]]]:
        """Commits a table at each of `trees`, `LEAVES` random digests each, an interior commit where `word` is
        `interior`: the session and each table's root and tag."""
        out = self.request(f"{word} {len(trees)}" + "".join(f" {d.hex()} {LEAVES} {sc}" for _, d, sc in trees),
                           os.urandom(64 * LEAVES * len(trees)))
        head, _, out = out.partition(b"\n")
        assert head.startswith(b"ok "), head
        s, got = bytes.fromhex(head[3:].decode()), []
        for _ in trees:
            head, _, out = out.partition(b"\n")
            root, tag, n = head.decode().removeprefix("ok ").split()
            got.append((bytes.fromhex(root), bytes.fromhex(tag)))
            out = out[128 * int(n):]
        assert out == b""
        return s, got

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


def tag_of(k: bytes, s: bytes, j: int, dom: bytes, sc: str, root: bytes) -> bytes:
    """Table `j`'s tag, of the commit session `s`, its domain id `dom`, schema `sc` and root `root`, under the key
    `k`."""
    return hashlib.sha512(k + TAG_LABEL + s + struct.pack(">I", j) + dom + struct.pack(">Q", LEAVES)
                          + bytes([SCHEMAS.index(sc)]) + root).digest()


def reference(k: bytes, n: int, m: int, s: bytes, got: list[tuple[bytes, bytes]], trees=TREES) -> bytes:
    """Registration `n`'s record, of the session `s` (whose tables' roots and tags are `got`) of an epoch of `m` sessions:
    `be64 n ‖ be64 m ‖ be32 T ‖` its table refs `‖` the record tag, SHA-512 under the key `k` of everything before it."""
    trefs = b"".join(s + struct.pack(">I", j) + dom + struct.pack(">Q", LEAVES) + bytes([SCHEMAS.index(sc)]) + root + tag
                     for j, ((_, dom, sc), (root, tag)) in enumerate(zip(trees, got)))
    head = struct.pack(">QQI", n, m, len(trees))
    return head + trefs + hashlib.sha512(k + REG_LABEL + head + trefs).digest()


def public(path: Path, roots: list[bytes], trees=TREES) -> Path:
    """A public file whose header names `trees`, `LEAVES` instances, at `roots`."""
    fv = {"domain_ids": {name: d.hex() for name, d, _ in trees}, "roots": {name: r.hex() for (name, *_), r in zip(trees, roots)},
          "schemas": {name: sc for name, _, sc in trees}}
    path.write_bytes(json.dumps({"frame_v3": fv, "instances": LEAVES}).encode() + b"\n")
    return path


def test_values_numbers_each_registration_on_the_record(exe, tmp_path):
    """Each registration takes the next number on the record, with no gap, across a restart of the service, and is
    appended there whole as Python's `Firewall.register` appends it; its record is the reference's, its number first and
    its record tag over it. The epoch is the process's: a session registers once, in the epoch that opened it."""
    state, key = tmp_path / "state", tmp_path / "salt.key"
    v = Values(exe, state, key)
    k = key.read_bytes()
    s2, g2 = v.commit(TREES[:1])
    recs = [reference(k, 0, 1, s2, g2, TREES[:1])]
    assert v.register(s2) == (1, recs[0])
    assert v.register(s2) == f"the session {s2.hex()} {NOT_OPENED}"
    s3, g3 = v.commit()
    recs.append(reference(k, 1, 1, s3, g3))
    assert v.register(s3) == (1, recs[1])
    v.close()
    again = Values(exe, state, key)
    assert again.register(s3) == f"the session {s3.hex()} {NOT_OPENED}"
    s4, g4 = again.commit()
    recs.append(reference(k, 2, 1, s4, g4))
    assert again.register(s4) == (1, recs[2])
    py = FW.Firewall(tmp_path / "py")
    for rec in recs:
        py.register(lambda n, rec=rec: rec)
    assert record(state) == record(tmp_path / "py")
    with open(state / "record.jsonl", "a") as f:
        f.write('{"event": "register"')
    s5, _ = again.commit()
    why = again.register(s5)
    assert isinstance(why, str) and why.startswith("the ZK server's record took no registration: "), why
    assert again.register(s5) == f"the session {s5.hex()} {NOT_OPENED}"
    again.close()


def test_an_epoch_opens_one_session(exe, tmp_path):
    """The service of record fixes each record's session count at 1, so the stack chooses no count and no session: a
    commit while the epoch's session is not yet registered is refused, its answer the refusal's line alone, and of eight
    commits that run together into an epoch with none, one opens it. Once the session registers, a commit opens the next
    epoch's. (`flock-firewall values`, which registers nothing, opens any number: `test_lean_salts.py`.)"""
    state, key = tmp_path / "state", tmp_path / "salt.key"
    v = Values(exe, state, key)
    k = key.read_bytes()
    line = f"commit {len(TREES)}" + "".join(f" {d.hex()} {LEAVES} {sc}" for _, d, sc in TREES)

    def commit(_: int = 0) -> bytes:
        return v.request(line, os.urandom(64 * LEAVES * len(TREES)))

    s0, g0 = v.commit()
    assert commit() == f"err {ONE_SESSION}\n".encode()
    assert v.register(s0) == (1, reference(k, 0, 1, s0, g0))
    with ThreadPoolExecutor(8) as pool:
        outs = list(pool.map(commit, range(8)))
    took = [o for o in outs if o.startswith(b"ok ")]
    assert len(took) == 1 and outs.count(f"err {ONE_SESSION}\n".encode()) == 7, outs
    s1 = bytes.fromhex(took[0].partition(b"\n")[0][3:].decode())
    m, rec = v.register(s1)
    assert m == 1 and rec[:16] == struct.pack(">QQ", 1, 1)
    assert [e["n"] for e in FW.Firewall(state).record() if e["event"] == "register"] == [0, 1]
    v.close()


def test_values_without_its_state_is_refused(exe, tmp_path):
    p = subprocess.run([str(exe), "values", "--salt-key", str(tmp_path / "k")], input=b"", capture_output=True, check=False)
    assert p.returncode == 2 and p.stderr.decode().startswith("usage: core-firewall values --state DIR --salt-key FILE")


def _session(exe, state: Path, s: str, pub: Path, key: Path, tmp: Path, known: bytes = b"") -> tuple[str | None, list]:
    """One outer session of `s` at L0 through `core-firewall session --salt-key`, cut after its `Register`, which the
    auditor answers with the draw and then `known`: the loop's refusal of it (None if admitted, the prover then
    receiving the draw alone), and what the auditor received."""
    auditor = Auditor(known)
    lp = Loop(exe, state, s, "L0", auditor, tmp, public=pub, flags=("--salt-key", str(key)))
    c = lp.connect()
    resp = call(c, REGISTER)
    assert resp == DRAW or resp[:1] == b"\x03", resp[:40]
    why = None if resp == DRAW else refusal(resp).removeprefix(R.REFUSED)
    c.close()
    slot_left(state)
    lp.close()
    return why, auditor.got


def test_the_loop_admits_each_registration_once_oldest_first(exe, tmp_path):
    """Six registrations, each a session holding a table at each tree; a statement's trees read the registrations that
    hold their roots. A statement reading registration 0 is admitted, its `Register` reaching the auditor, and registration
    0 with it; one reading 0 and 2 admits 2 and skips 1, a failure of its stream; one reading 1 is refused, since 1 was
    passed over; one reading only 0 again is served and admits nothing; one whose roots no registration holds, or that
    reads 3 under another key (its record tag is not that key's), is refused. On a stream of budget 1, a statement reading
    5 skips 3 and 4, and the stream stops at its budget. Of each refused only `Register` reaches the auditor, each is its
    recursive session's stop, and the record is Python's for the same ops."""
    state, key, other = tmp_path / "state", tmp_path / "salt.key", tmp_path / "other.key"
    fw, py = FW.Firewall(state), FW.Firewall(tmp_path / "py")
    for f in (fw, py):
        for _ in range(6):
            f.finish(f.open("t", ["L0"]))
        f.finish(f.open("u", ["L0"], 1))
    v = Values(exe, state, key)
    roots = []
    for _ in range(6):
        s, got = v.commit()
        _, rec = v.register(s)
        roots.append([r for r, _ in got])
        py.register(lambda n, rec=rec: rec)
    v.close()
    Values(exe, tmp_path / "other-state", other).close()
    cases = [
        ("r0000", [roots[0][0], roots[0][1]], key, [0], None),
        ("r0001", [roots[0][0], roots[2][1]], key, [0, 2], None),
        ("r0002", [roots[1][0], roots[1][1]], key, [1], "registration 1 was skipped: one passed over is never admitted"),
        ("r0003", [roots[0][0], roots[0][1]], key, [0], None),
        ("r0004", [os.urandom(64), roots[0][1]], key, None, f"row a: {NO_HOLDER}"),
        ("r0005", [roots[3][0], roots[3][1]], other, None, f"registration 3: {NOT_TAGGED}"),
        ("r0006", [roots[5][0], roots[5][1]], key, [5], "stream u stopped at its failure budget (2 of 1)"),
    ]
    for s, rs, k, reads, why in cases:
        got, sent = _session(exe, state, s, public(tmp_path / f"{s}.pub", rs), k, tmp_path)
        stop = why is not None and not why.startswith("stream ")
        assert got == (None if why is None else f"recursive session {s} stopped at L0: {why}" if stop else why), (s, got)
        assert sent == [[REGISTER]], (s, sent)
        with py.slot(s, "L0"):
            py.admit(s, "L0", timed=True)
            try:
                py.admit_reads(s, "L0", reads if reads is not None else why)
            except FW.Refused as e:
                assert why is not None and why in str(e), (s, str(e))
            else:
                assert why is None, (s, why)
                py.stop(s, "L0", CUT)
        assert record(state) == record(tmp_path / "py"), s
    events = [(e["event"], e.get("n")) for e in FW.Firewall(state).record() if e["event"] in ("admit", "skip")]
    assert events == [("admit", 0), ("skip", 1), ("admit", 2), ("skip", 3), ("skip", 4), ("admit", 5)]
    assert fw.failures("u") == 2


def _registered(state: Path) -> list[tuple[int, dict]]:
    """The record's `register` events, each with its line's index."""
    lines = (state / "record.jsonl").read_text().splitlines()
    return [(i, e) for i, e in enumerate(map(json.loads, lines)) if e["event"] == "register"]


def _put(state: Path, at: int, ev: dict) -> None:
    """The record with its line `at` the event `ev`, as the ZK server writes it."""
    lines = (state / "record.jsonl").read_text().splitlines()
    lines[at] = json.dumps(ev, sort_keys=True)
    (state / "record.jsonl").write_text("\n".join(lines) + "\n")


def _flip(h: str) -> str:
    """The hex `h` with its first digit changed."""
    return ("1" if h[0] == "0" else "0") + h[1:]


def test_an_interior_registration_is_under_its_sessions_own_key(exe, tmp_path):
    """An interior commit's registration is numbered and appended as any other, its record the reference's with its table
    tags and record tag under a key the service drew for the session, not `k` and not another interior session's. The
    record's `register` line holds that key and its tag, SHA-512 under `k` of the key and the record, and the service
    answers neither. The loop, which holds `k`, checks the key's tag, then the record tag under the key: a statement
    reading the registration is admitted, and one reading it under another salt key, with the line's key or its tag
    changed, with a key one byte longer (its record and both tags made as the server would), or with neither on the
    line, is refused, only its `Register` having reached the auditor. The record's state directory is its user's
    alone."""
    state, key, other = tmp_path / "state", tmp_path / "salt.key", tmp_path / "other.key"
    fw = FW.Firewall(state)
    for _ in range(6):
        fw.finish(fw.open("t", ["L0"]))
    v = Values(exe, state, key)
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    k = key.read_bytes()
    s, got = v.commit(word="interior")
    m, rec = v.register(s)
    at, ev = _registered(state)[-1]
    tk, kt = bytes.fromhex(ev["key"]), bytes.fromhex(ev["key_tag"])
    assert (m, rec, ev["record"]) == (1, reference(tk, 0, 1, s, got), rec.hex())
    assert all(tag == tag_of(tk, s, j, dom, sc, root) != tag_of(k, s, j, dom, sc, root)
               for j, ((_, dom, sc), (root, tag)) in enumerate(zip(TREES, got)))
    assert kt == hashlib.sha512(k + KEY_TAG_LABEL + tk + rec).digest() and tk != k and tk not in rec and kt not in rec
    assert v.register(s) == f"the session {s.hex()} {NOT_OPENED}"
    s2, got2 = v.commit(word="interior")
    _, rec2 = v.register(s2)
    tk2 = bytes.fromhex(_registered(state)[-1][1]["key"])
    assert tk2 != tk and rec2 == reference(tk2, 1, 1, s2, got2)
    v.close()
    Values(exe, tmp_path / "other-state", other).close()
    roots = [r for r, _ in got]
    bare = {name: x for name, x in ev.items() if name not in ("key", "key_tag")}
    long = tk + b"\0"
    rec_long = reference(long, 0, 1, s, [(r, tag_of(long, s, j, dom, sc, r))
                                         for j, ((_, dom, sc), r) in enumerate(zip(TREES, roots))])
    longer = {**ev, "key": long.hex(), "key_tag": hashlib.sha512(k + KEY_TAG_LABEL + long + rec_long).hexdigest(),
              "record": rec_long.hex()}
    refused = f"registration 0: {NOT_TAGGED}"
    cases = [("r0000", other, ev, refused), ("r0001", key, {**ev, "key": _flip(ev["key"])}, refused),
             ("r0002", key, {**ev, "key_tag": _flip(ev["key_tag"])}, refused), ("r0003", key, bare, refused),
             ("r0004", key, longer, refused), ("r0005", key, ev, None)]
    for sid, kf, line, why in cases:
        _put(state, at, line)
        got_why, sent = _session(exe, state, sid, public(tmp_path / f"{sid}.pub", roots), kf, tmp_path)
        want = None if why is None else f"recursive session {sid} stopped at L0: {why}"
        assert got_why == want and sent == [[REGISTER]], (sid, got_why, sent)
    events = [(e["event"], e.get("n")) for e in FW.Firewall(state).record() if e["event"] in ("admit", "skip")]
    assert events == [("admit", 0)]


def known_entry(tree: tuple[str, bytes, str], root: bytes, n: int = LEAVES) -> bytes:
    """The entry of `tree` (its name, domain id and schema) at `root`, of `n` leaves, in the verifier's list of a
    statement's known trees: `D ‖ be64 N ‖ u8 schema ‖ ROOT`."""
    _, dom, sc = tree
    return dom + struct.pack(">Q", n) + bytes([SCHEMAS.index(sc)]) + root


def known_list(*entries: bytes) -> bytes:
    """The verifier's list of the known trees `entries`, as it follows the draw in its answer to `Register`."""
    return struct.pack(">I", len(entries)) + b"".join(entries)


class Server:
    """A ZK server's state with one registration, a table at each of `TREES` (`roots`), `sessions` recursive sessions
    of one statement open on it, and Python's record of the same ops."""

    def __init__(self, exe: Path, tmp: Path, sessions: int):
        self.exe, self.tmp, self.state, self.key = exe, tmp, tmp / "state", tmp / "salt.key"
        self.py = FW.Firewall(tmp / "py")
        for f in (FW.Firewall(self.state), self.py):
            for _ in range(sessions):
                f.finish(f.open("t", ["L0"]))
        v = Values(exe, self.state, self.key)
        s, got = v.commit()
        _, rec = v.register(s)
        self.py.register(lambda n: rec)
        v.close()
        self.roots = [r for r, _ in got]

    def session(self, s: str, trees: list, roots: list[bytes], known: bytes, reads: list[int] | str,
                entries: tuple[bytes, ...] = ()) -> None:
        """The outer session of `s` whose statement holds `trees` at `roots`, the auditor listing `known` after its
        draw. It reads the registrations `reads` and passes `entries` as known content, or, where `reads` is a
        refusal's reason, is its recursive session's stop for it; either way only its `Register` reaches the auditor
        and the record is Python's for the same ops."""
        got, sent = _session(self.exe, self.state, s, public(self.tmp / f"{s}.pub", roots, trees), self.key, self.tmp, known)
        assert sent == [[REGISTER]], (s, sent)
        with self.py.slot(s, "L0"):
            self.py.admit(s, "L0", timed=True)
            if isinstance(reads, str):
                assert got == f"recursive session {s} stopped at L0: {reads}", (s, got)
                with pytest.raises(FW.Refused, match=re.escape(reads)):
                    self.py.admit_reads(s, "L0", reads, entries)
            else:
                assert got is None, (s, got)
                self.py.admit_reads(s, "L0", reads, entries)
                self.py.stop(s, "L0", CUT)
        assert record(self.state) == record(self.tmp / "py"), s

    def events(self) -> list[tuple]:
        """The record's admissions, skips and known trees, in order."""
        return [(e["event"], e.get("n", e.get("tree"))) for e in FW.Firewall(self.state).record()
                if e["event"] in ("admit", "skip", "known")]


def test_a_statement_holding_a_known_row_is_admitted_under_the_key(exe, tmp_path):
    """A statement holds a tree at known content beside two value commitments of registration 0, and the verifier lists
    that tree after its draw. Under the key the loop admits the statement: it records the known tree, then admits the
    registration, and the prover receives the draw without the list. A second statement of the same trees records its
    known tree again and admits nothing, and one whose only tree the verifier lists reads no registration at all."""
    z = Server(exe, tmp_path, 3)
    root = os.urandom(64)
    entry = known_entry(K, root)
    z.session("r0000", [*TREES, K], [*z.roots, root], known_list(entry), [0], (entry,))
    z.session("r0001", [*TREES, K], [*z.roots, root], known_list(entry), [0], (entry,))
    z.session("r0002", [K], [root], known_list(entry), [], (entry,))
    assert z.events() == [("known", entry.hex()), ("admit", 0), ("known", entry.hex()), ("known", entry.hex())]


def test_a_statement_holding_a_row_whose_salt_the_stack_chose_is_refused(exe, tmp_path):
    """A row the stack committed under a salt of its own has a root that no registration holds and the verifier never
    computed. Whether the verifier lists nothing, or lists the tree it computed at that domain, the statement is its
    recursive session's stop under the key: nothing is admitted and no known tree recorded."""
    z = Server(exe, tmp_path, 2)
    own, root = os.urandom(64), os.urandom(64)
    z.session("r0000", [*TREES, K], [*z.roots, own], b"", f"row k: {NO_HOLDER}")
    z.session("r0001", [*TREES, K], [*z.roots, own], known_list(known_entry(K, root)), f"row k: {OTHER}")
    assert z.events() == []


def test_a_value_commitment_listed_as_known_content_is_refused(exe, tmp_path):
    """An entry whose root is the root of a table of a registration on the record, at the table's own domain or at
    another: a value commitment of the server's is admitted as one, never passed as known content, so the stack gets no
    second way to have a registration read."""
    z = Server(exe, tmp_path, 2)
    why = "its root is a table's of registration 0, and a value commitment of the ZK server's is no known content"
    z.session("r0000", TREES, z.roots, known_list(known_entry(TREES[0], z.roots[0])), f"known tree 0: {why}")
    z.session("r0001", [*TREES, K], [*z.roots, z.roots[1]], known_list(known_entry(K, os.urandom(64)), known_entry(K, z.roots[1])),
              f"known tree 1: {why}")
    assert z.events() == []


def test_a_list_with_two_entries_of_one_domain_is_refused(exe, tmp_path):
    """Two entries of one domain, the same tree twice or two roots: the list names a tree once."""
    z = Server(exe, tmp_path, 2)
    root = os.urandom(64)
    entry = known_entry(K, root)
    why = "known tree 1: a second entry of its domain"
    z.session("r0000", [*TREES, K], [*z.roots, root], known_list(entry, entry), why)
    z.session("r0001", [*TREES, K], [*z.roots, root], known_list(entry, known_entry(K, os.urandom(64))), why)
    assert z.events() == []


def test_a_tree_that_differs_from_its_entry_is_refused(exe, tmp_path):
    """A tree of the statement at a listed domain with another leaf count, another schema or another root than the
    entry's: at a domain the verifier lists, the statement holds the verifier's tree or is stopped."""
    z = Server(exe, tmp_path, 3)
    root = os.urandom(64)
    lists = [known_list(known_entry(K, root, LEAVES + 1)), known_list(known_entry(("k", K[1], SCHEMAS[0]), root)),
             known_list(known_entry(K, os.urandom(64)))]
    for i, known in enumerate(lists):
        z.session(f"r000{i}", [*TREES, K], [*z.roots, root], known, f"row k: {OTHER}")
    assert z.events() == []


def test_a_list_that_does_not_parse_is_refused(exe, tmp_path):
    """Bytes after the draw that are no list: fewer than a count, a count its entries don't end at (one byte short, one
    entry short, a byte past the last), a schema that is none of the three, an entry of no leaf. Each is the session's
    stop, as a statement the verifier never listed anything for would be."""
    z = Server(exe, tmp_path, 6)
    root = os.urandom(64)
    entry = known_entry(K, root)
    whole = known_list(entry)
    cases = [
        (b"\0", "1 bytes, where its count is 4"),
        (whole[:-1], "140 bytes, where its 1 entries end at 141"),
        (struct.pack(">I", 2) + entry, "141 bytes, where its 2 entries end at 278"),
        (whole + b"\0", "142 bytes, where its 1 entries end at 141"),
        (known_list(entry[:72] + b"\x03" + root), f"entry 0 of schema 3, none of [{', '.join(SCHEMAS)}]'s"),
        (known_list(known_entry(K, root, 0)), "entry 0 of no leaf"),
    ]
    for i, (known, why) in enumerate(cases):
        z.session(f"r000{i}", [*TREES, K], [*z.roots, root], known, f"{LIST}: {why}")
    assert z.events() == []


def test_an_entry_at_no_tree_of_the_statement_is_refused(exe, tmp_path):
    """An entry at whose domain the statement has no tree: the list is the statement's own known trees and nothing
    else, so a list can't differ between two statements that pass with the same trees."""
    z = Server(exe, tmp_path, 2)
    root = os.urandom(64)
    why = "the statement has no tree at its domain, and the verifier lists a statement's own trees"
    z.session("r0000", TREES, z.roots, known_list(known_entry(K, root)), f"known tree 0: {why}")
    z.session("r0001", [*TREES, K], [*z.roots, root],
              known_list(known_entry(K, root), known_entry(("x", os.urandom(64), SCHEMAS[0]), os.urandom(64))),
              f"known tree 1: {why}")
    assert z.events() == []


def test_a_loop_without_the_key_cuts_the_list_and_records_nothing(exe, tmp_path):
    """Without `--salt-key` the loop admits no registration and holds no tree to the list, and the prover still receives
    the draw alone: the list is the release side's to read, never the prover's."""
    state = tmp_path / "state"
    fw, py = FW.Firewall(state), FW.Firewall(tmp_path / "py")
    for f in (fw, py):
        f.finish(f.open("t", ["L0"]))
    auditor = Auditor(known_list(known_entry(K, os.urandom(64))))
    lp = Loop(exe, state, "r0000", "L0", auditor, tmp_path)
    c = lp.connect()
    assert call(c, REGISTER) == DRAW
    c.close()
    slot_left(state)
    lp.close()
    assert auditor.got == [[REGISTER]]
    with py.slot("r0000", "L0"):
        py.admit("r0000", "L0", timed=True)
        py.stop("r0000", "L0", CUT)
    assert record(state) == record(tmp_path / "py")


def test_the_loop_refuses_a_statement_it_cannot_read_under_its_key(exe, tmp_path):
    """A key file that isn't there, or a public file without a header of hm96 trees, is refused before the loop listens
    (`refused WHY`, exit 2)."""
    (tmp_path / "c").write_bytes(b"x")
    pub = public(tmp_path / "p.pub", [os.urandom(64), os.urandom(64)])
    (tmp_path / "bad.pub").write_bytes(b"x")

    def refused(pub: Path, key: Path) -> str:
        p = subprocess.run([str(exe), "session", "--state", str(tmp_path / "state"), "--session", "r0000", "--part", "L0",
                            "--salt-key", str(key), "--verifier", "127.0.0.1:9", "--circuit", str(tmp_path / "c"),
                            "--public", str(pub)], input=b"", capture_output=True, check=False)
        assert p.returncode == 2 and p.stdout.decode().startswith("refused "), (p.returncode, p.stdout, p.stderr[-400:])
        return p.stdout.decode().removeprefix("refused ").strip()

    assert refused(pub, tmp_path / "none.key") == f"--salt-key {tmp_path / 'none.key'}: no such file (`core-firewall values` creates it)"
    Values(exe, tmp_path / "state", tmp_path / "salt.key").close()
    assert refused(tmp_path / "bad.pub", tmp_path / "salt.key") == "--salt-key: public file: no header line"
