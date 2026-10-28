"""The challenger as a service on its own host (`remote`), over loopback: it signs and writes its record before its first
coin opens, answers each root with that family's next coin and logs the release before the coin leaves; it refuses
every malformed or out-of-order frame and ends the session there, and however the session ends it signs its release log
(the close record).  The client (`RemoteChallenger`) is `LiveRecord`'s challenger as it is, and refuses a coin that
doesn't open; it asks for several coins in one round trip (`open_many`), the release log the same as one at a time, and
a batch with any bad answer raises whole.  The verifier, reading the challenger's directory and the auditor's signed
ledger, trusts only the one session the ledger opens for the window under the pinned key, ended closed, its commitment
and the roots its signed log shows, so a prover that rewrites a coin's root, an entry of the log or the ledger, or
presents an unsigned log, an unclosed session, a second session, a ledger signed under its own key or one older than the
session's end is refused."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import re
import socket
import threading
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.core.protocols.sampled_proofs.coins import ChallengerCoins
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.crypto import lms

from test_challenger import HOST, entropy, lms_key, lms_verify

SCHEME = "verity/test/coins/v1"
FAMILIES = [("positions", 3), ("replay", 2)]
WINDOW = {"root": "11" * 64}
#: main's first-parent history, newest first, from c81c31e6d5 back to 6c2de12715: the first parent of the queue's merge
#: 1018753ee5, whose second parent is 882770e600
MAIN = R.main_lines(Path(__file__).parent / "fixtures" / "main-first-parent.txt")
PIN = "c81c31e6d51f9c481cdead96221d65ad6d7afd2c"
#: the host's pins log as its operator published it when it repinned vy-challenger-1 to `PIN`; every open seals it
#: unless a
#: test says otherwise
HOST_PINS = ['{"commit":"c81c31e6d51f9c481cdead96221d65ad6d7afd2c","main":"c81c31e6d51f9c481cdead96221d65ad6d7afd2c",'
             '"pinned_utc":"2026-10-22T11:07:28Z"}']
#: the challenger key the window's deployment pins: sixteen leaves, four a session (the ledger's open, the record, the
#: close and the ledger's end)
PINNED = lms.keygen(lms.Params(64, 4, 4), bytes([9]) * 64)


def pinned_key(private: lms.PrivateKey = PINNED) -> C.ChallengerKey:
    return C.ChallengerKey(private.params.W, private.params.H, lms.public_key(private))


def root(i: int) -> bytes:
    return bytes([i]) * 64


def frame(head: dict, body: bytes = b"") -> bytes:
    h = json.dumps(head).encode()
    return R.HEADER.pack(len(h), len(body)) + h + body


def reply(sock: socket.socket) -> dict:
    data = b""
    while len(data) < R.HEADER.size:
        data += sock.recv(R.HEADER.size - len(data))
    nh, _ = R.HEADER.unpack(data)
    head = b""
    while len(head) < nh:
        head += sock.recv(nh - len(head))
    return json.loads(head)


@pytest.fixture
def signer(tmp_path):
    """The challenger key's signer on its host, at its next leaf: one for every signature under the key."""
    path = tmp_path / "key.lms"
    lms.LeafState.create(path, lms.public_key(PINNED), PINNED.params.H)
    with lms.LeafState(path, lms.public_key(PINNED), PINNED.params.H) as state:
        yield lambda m: lms.sign(PINNED, state, m).encode()


def record_leaf(directory) -> int:
    return int(json.loads((directory / R.RECORD).read_text())["signature"][:8], 16)


class Ledger:
    """The auditor's ledger at `path` (its operator's ``windows.jsonl``): each entry signed with `sign` under key
    `private`, in
    `schema`."""

    def __init__(self, path, sign, private: lms.PrivateKey = PINNED, schema: str = C.LEDGER_SCHEMA) -> None:
        self.path, self.sign, self.I, self.schema = path, sign, lms.public_key(private)[:16], schema
        self.key_sha512 = hashlib.sha512(lms.public_key(private)).hexdigest()
        self.lines: list[str] = []
        self.reserves: dict[str, tuple[int, dict]] = {}

    def append(self, kind: str, session: str | None, body: dict) -> None:
        self.lines.append(C.ledger_line(self.lines, kind, session, body, self.I, self.sign, self.schema))
        self.path.write_text("".join(line + "\n" for line in self.lines))

    def reserve(self, session: str, **changes) -> None:
        """The operator's reserve of `session`: the window's request, its start and its expiry; the session's open
        binds it."""
        body = {k: v for k, v in {**RESERVE, **changes}.items() if v is not DROP}
        self.reserves[session] = (len(self.lines), body)
        self.append("reserve", session, body)

    def open(self, session: str, window: dict = WINDOW, **extra) -> None:
        if self.schema == C.LEDGER_SCHEMA_V1:
            window = {"commit": window["root"], "period": "p-0"}
        if session in self.reserves:
            seq, r = self.reserves[session]
            extra = {**{k: r[k] for k in C.RESERVED if k in r}, "reserve": seq, "opened_by": r.get("for") or r["by"],
                     "started_utc": "2026-10-20T12:05:00Z", **extra}
        body = {"window": window, "host": f"{HOST}.{session}", "pinned": PIN, "pins": C.pins_seal(HOST_PINS),
                "key_sha512": self.key_sha512, "port": 7811, "by": "proofs", **extra}
        self.append("open", session, {k: v for k, v in body.items() if v is not DROP})

    def end(self, session: str, directory, closed: bool, **changes) -> None:
        files = sealed(directory)
        record = json.loads((directory / R.RECORD).read_text())
        hellos = (directory / R.HELLOS).read_text().count("\n") if (directory / R.HELLOS).is_file() else 0
        body = {"closed": closed, "files": files, "rc": 0 if closed else 1, "record_leaf": record_leaf(directory),
                "commitment": record["commitment"]["root"], "refused_hellos": hellos, "complete": closed,
                "ended": "closed by the client" if closed else "the stream closed"}
        self.append("end", session, {k: v for k, v in {**body, **changes}.items() if v is not DROP})


DROP = object()
RESERVE = {"by": "proofs", "peer": "10.0.0.2", "scheme": SCHEME, "epoch": 0, "families": [list(f) for f in FAMILIES],
           "minutes": 30, "started_utc": "2026-10-20T12:00:00Z", "expires_utc": "2026-10-20T12:30:00Z"}
RESERVED_SESSION = "s20261020T120000Z-ab12"


def sealed(directory) -> dict[str, str]:
    """The SHA-512 of every file in `directory`, as the auditor's end seals them."""
    return {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


def at(*leaves: int):
    """The pinned key's signer at `leaves` in turn: a ledger signed beside a host's own, around the leaves its session
    signed (a fresh host's ledger opens at 0, its record is at 1, its close at 2 and the ledger's end at 3)."""
    it = iter(leaves)
    return lambda m: lms.sign_at(PINNED, next(it), m).encode()


class Host:
    """The challenger on its host: a challenger over `FAMILIES` whose session `session` the ledger opens for `window`
    (a new ledger in `tmp_path` unless `ledger`), whose record `open_session` signed with `sign` and wrote to
    `directory` naming the host the ledger opens it on (or `record_host`), serving one session on a loopback port;
    `reason` is how `serve` ended it, and the ledger's end says the session closed when its client closed it (or as
    `closed` says). With `withhold`, the session is a withheld one, and the ledger's open says so."""

    def __init__(self, tmp_path, sign, session: str = "s-1", seed: str = "host", ledger: Ledger | None = None,
                 window: dict = WINDOW, closed: bool | None = None, record_host: str | None = None,
                 secret: bytes | None = None, withhold: bool = False) -> None:
        self.secret = secret
        self.directory = tmp_path / session
        self.directory.mkdir(parents=True)
        self.ledger, self.closed = ledger or Ledger(tmp_path / "windows.jsonl", sign), closed
        self.ledger.open(session, window, **({"withheld": True} if withhold else {}))
        self.challenger = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy(seed))
        self.session = R.open_session(self.challenger, pinned_key(), record_host or f"{HOST}.{session}", sign,
                                      self.directory, session, withhold=withhold)
        (self.directory / R.HELLOS).write_text("")
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.address = "127.0.0.1:%d" % self.listener.getsockname()[1]
        self.reason: str | None = None
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self) -> None:
        conn, _ = self.listener.accept()
        self.listener.close()
        with conn:
            try:
                self.reason = R.serve(conn, self.session, itertools.count().__next__, self.secret)
            except Exception as e:
                self.error = e
            finally:
                closed = self.error is None and self.reason is None
                self.ledger.end(self.session.id, self.directory, closed if self.closed is None else self.closed)

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason

    def trusted(self, window: dict = WINDOW, key: C.ChallengerKey | None = None, ledger=None):
        """`remote.trusted_session` over this host's directory and the ledger's file (or `ledger`), its one head."""
        return R.trusted_session(self.directory, [ledger or self.ledger.path], window, key or pinned_key(), lms_verify)

    def trusted_of(self, window: dict = WINDOW):
        """`challenger.trusted_session_of` over this host's files as they are, past `remote`'s check of the sealed bytes."""
        return C.trusted_session_of(R.ledger_lines(self.ledger.path), window, pinned_key(), HOST_PINS, MAIN,
                                    R.session_files(self.directory), lms_verify)


def honest(host: Host) -> dict[str, C.LiveRecord]:
    """The honest client's session: a root and its coin per family, twice, a third positions root, then its close."""
    client = R.RemoteChallenger.connect(host.address)
    records = {label: C.LiveRecord(client, label, itertools.count().__next__) for label, _ in FAMILIES}
    for i in range(2):
        for record in records.values():
            record.record(root(i))
    records["positions"].record(root(2))
    client.close()
    assert host.ended() is None
    return records


def test_an_honest_session_checks_against_what_the_challenger_signed_and_logged(tmp_path, signer):
    host = Host(tmp_path, signer)
    records = honest(host)
    commitment, roots = host.trusted()
    assert commitment == host.challenger.commitment
    assert roots == {"positions": (root(0), root(1), root(2)), "replay": (root(0), root(1))}
    for label, record in records.items():
        assert record.reason() is None
        assert C.order_reason(commitment, label, roots[label], record.openings) is None
    close = json.loads((host.directory / R.CLOSE).read_text())
    assert (close["session"], close["ended"], close["lines"]) == ("s-1", None, 11)


def test_the_challenger_hangs_up_only_after_the_client_so_its_last_frame_arrives(tmp_path, signer):
    host = Host(tmp_path, signer)
    with socket.create_connection(host.listener.getsockname()) as s:
        s.sendall(frame({"type": "hello"}))
        assert reply(s)["type"] == "commitment"
        s.sendall(frame({"type": "close"}))
        assert reply(s) == {"type": "closed"}
        assert (host.directory / R.CLOSE).is_file()
        assert s.recv(1) == b""
        assert host._thread.is_alive()
    assert host.ended() is None


def test_the_challenger_serves_a_session_once_and_only_once_its_record_is_written(tmp_path, signer):
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    session = R.open_session(ch, pinned_key(), HOST, signer, tmp_path, "s-1")
    (tmp_path / R.RECORD).unlink()
    a, b = socket.socketpair()
    with a, b, pytest.raises(C.CoinError, match="open_session"):
        R.serve(a, session, itertools.count().__next__)
    R.close_session(session, None)
    with pytest.raises(C.CoinError, match="closed once"):
        R.close_session(session, None)
    with a, b, pytest.raises(C.CoinError, match="serves a session once"):
        R.serve(a, session, itertools.count().__next__)
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    ch.open("positions", 0, root(0))
    with pytest.raises(C.CoinError, match="before any coin opens"):
        R.open_session(ch, pinned_key(), HOST, lambda m: b"", tmp_path / "other", "s-2")
    for bad in ("", "a b", "x" * (C.MAX_SESSION_ID + 1)):
        with pytest.raises(C.CoinError, match="session id"):
            R.open_session(C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("h")), pinned_key(), HOST, signer, tmp_path,
                           bad)


OPEN = {"type": "open", "label": "positions", "index": 0, "after": root(0).hex()}
REPEATED = b'{"type":"close","type":"close"}'
# a refused frame's header is sent alone where the challenger refuses on it, so no unread bytes reset the stream
REFUSED = {
    "open before hello": [frame(OPEN)],
    "hello twice": [frame({"type": "hello"}), frame({"type": "hello"})],
    "an unknown type": [frame({"type": "hello"}), frame({"type": "draw"})],
    "an extra key": [frame({"type": "hello"}), frame({**OPEN, "retry": True})],
    "a repeated key": [frame({"type": "hello"}), R.HEADER.pack(len(REPEATED), 0) + REPEATED],
    "a coin ahead of its family's next": [frame({"type": "hello"}), frame({**OPEN, "index": 1})],
    "a coin opened twice": [frame({"type": "hello"}), frame(OPEN), frame(OPEN)],
    "an unknown family": [frame({"type": "hello"}), frame({**OPEN, "label": "other"})],
    "a family past its last coin": [frame({"type": "hello"})] + [frame({**OPEN, "label": "replay", "index": i})
                                                               for i in range(3)],
    "a short root": [frame({"type": "hello"}), frame({**OPEN, "after": root(0)[:32].hex()})],
    "an uppercase root": [frame({"type": "hello"}), frame({**OPEN, "after": root(10).hex().upper()})],
    "a head that is not JSON": [frame({"type": "hello"}), R.HEADER.pack(3, 0) + b"{x}"],
    "a frame with a body": [frame({"type": "hello"}), R.HEADER.pack(2, 1)],
    "a head over the limit": [frame({"type": "hello"}), R.HEADER.pack(R.MAX_HEAD + 1, 0)],
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_every_malformed_or_out_of_order_frame_is_refused_and_ends_the_session(tmp_path, signer, case):
    """A refused session is signed, with why it ended, and is no window's: the verifier refuses it, at the ledger's end
    and, were the ledger to say it closed, at the challenger's own close."""
    host = Host(tmp_path, signer, closed=True)
    frames, opened = REFUSED[case], 0
    with socket.create_connection(host.listener.getsockname()) as s:
        for f in frames[:-1]:
            s.sendall(f)
            kind = reply(s)["type"]
            assert kind in ("commitment", "opening")
            opened += kind == "opening"
        s.sendall(frames[-1])
        answer = reply(s)
        assert answer["type"] == "refused"
        assert s.recv(1) == b""
    assert host.ended() == answer["reason"]
    assert host.error is None
    assert json.loads((host.directory / R.CLOSE).read_text())["ended"] == answer["reason"]
    roots = R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)
    assert sum(len(r) for r in roots.values()) == opened
    with pytest.raises(C.CoinError, match="ended without its client closing it: " + re.escape(answer["reason"])):
        host.trusted()


def test_a_session_its_client_never_closed_is_signed_as_such_and_refused(tmp_path, signer):
    for closed, why in ((None, "the auditor's ledger ends session s-1 unclosed"),
                        (True, "ended without its client closing it: the stream closed")):
        host = Host(tmp_path / str(closed), signer, closed=closed)
        with socket.create_connection(host.listener.getsockname()) as s:
            s.sendall(frame({"type": "hello"}))
            assert reply(s)["type"] == "commitment"
            s.sendall(frame(OPEN))
            assert reply(s)["type"] == "opening"
        assert host.ended() == "the stream closed"
        with pytest.raises(C.CoinError, match=why):
            host.trusted()


SECRET = bytes(range(32))


def test_a_session_with_a_secret_serves_only_the_client_whose_hello_carries_it(tmp_path, signer):
    host = Host(tmp_path, signer, secret=SECRET)
    with R.RemoteChallenger.connect(host.address, SECRET) as client:
        C.LiveRecord(client, "positions", itertools.count().__next__).record(root(0))
    assert host.ended() is None
    assert host.trusted()[1]["positions"] == (root(0),)


@pytest.mark.parametrize("hello, why", [
    ({"type": "hello"}, "with the session's secret"),
    ({"type": "hello", "secret": bytes(32).hex()}, "not the session's"),
    ({"type": "hello", "secret": SECRET.hex().upper()}, "not the session's"),
    ({"type": "hello", "secret": SECRET.hex(), "extra": 1}, "with the session's secret"),
    ({"type": "hello", "secret": 7}, "not the session's"),
])
def test_a_session_with_a_secret_refuses_a_hello_without_it_before_any_coin(tmp_path, signer, hello, why):
    host = Host(tmp_path, signer, secret=SECRET)
    with socket.create_connection(host.listener.getsockname()) as s:
        s.sendall(frame(hello))
        answer = reply(s)
        assert answer["type"] == "refused" and why in answer["reason"]
    assert why in host.ended()
    assert R.released_roots(host.directory / R.RELEASES, host.challenger.commitment) == {"positions": (), "replay": ()}
    with pytest.raises(C.CoinError, match="unclosed"):
        host.trusted()


def test_greet_reads_the_hello_alone_so_frames_pipelined_behind_it_reach_serve(tmp_path, signer):
    """`greet` and `serve` each build their own `_Wire`; that's sound because `_Wire` reads exactly one frame and buffers
    nothing past it."""
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    session = R.open_session(ch, pinned_key(), HOST, signer, tmp_path, "s-1")
    a, b = socket.socketpair()
    with a, b:
        b.sendall(frame({"type": "hello", "secret": SECRET.hex()}) + frame(OPEN) + frame({"type": "close"}))
        b.shutdown(socket.SHUT_WR)
        assert R.greet(a, SECRET) is None
        reason = R.serve(a, session, itertools.count().__next__, SECRET, greeted=True)
        answers = []
        while (head := b.recv(R.HEADER.size, socket.MSG_WAITALL)):
            answers.append(json.loads(b.recv(R.HEADER.unpack(head)[0], socket.MSG_WAITALL))["type"])
    assert (reason, answers) == (None, ["commitment", "opening", "closed"])
    assert R.released_roots(tmp_path / R.RELEASES, ch.commitment)["positions"] == (root(0),)


def test_greet_refuses_a_bad_hello_alone_so_the_next_connection_is_served(tmp_path, signer):
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    session = R.open_session(ch, pinned_key(), HOST, signer, tmp_path, "s-1")
    reasons: list[str | None] = []
    listener = socket.create_server(("127.0.0.1", 0))
    whys: list[str | None] = []

    def run() -> None:
        while True:
            conn, _ = listener.accept()
            why = R.greet(conn, SECRET)
            whys.append(why)
            if why is None:
                with conn:
                    reasons.append(R.serve(conn, session, itertools.count().__next__, SECRET, greeted=True))
                return
            conn.close()

    t = threading.Thread(target=run)
    t.start()
    for hello in ({"type": "hello"}, {"type": "hello", "secret": bytes(32).hex()}):
        with socket.create_connection(listener.getsockname()) as s:
            s.sendall(frame(hello))
            assert reply(s)["type"] == "refused" and s.recv(1) == b""
    with R.RemoteChallenger.connect("127.0.0.1:%d" % listener.getsockname()[1], SECRET) as client:
        C.LiveRecord(client, "positions", itertools.count().__next__).record(root(0))
    t.join()
    listener.close()
    assert [w is None for w in whys] == [False, False, True] and reasons == [None]
    assert R.released_roots(tmp_path / R.RELEASES, ch.commitment)["positions"] == (root(0),)


class Liar:
    """A server that answers hello with the challenger's commitment and each open with `answer(opening)`."""

    def __init__(self, answer) -> None:
        self.challenger = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.address = "127.0.0.1:%d" % self.listener.getsockname()[1]
        self._answer = answer
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self) -> None:
        conn, _ = self.listener.accept()
        self.listener.close()
        with conn:
            reply(conn)
            conn.sendall(frame({"type": "commitment", "commitment": self.challenger.commitment.to_json()}))
            head = reply(conn)
            opening = self.challenger.open(head["label"], head["index"], bytes.fromhex(head["after"]))
            conn.sendall(self._answer(opening))

    def join(self) -> None:
        self._thread.join()


def _other_coin(o: C.LiveOpening) -> bytes:
    return frame({"type": "opening", "opening": dataclasses.replace(o, coin=bytes(64)).to_json()})


def _other_root(o: C.LiveOpening) -> bytes:
    return frame({"type": "opening", "opening": dataclasses.replace(o, after=root(9)).to_json()})


def _other_index(o: C.LiveOpening) -> bytes:
    return frame({"type": "opening", "opening": dataclasses.replace(o, index=1).to_json()})


ANSWERS = {
    "a coin that does not open": (_other_coin, "does not open"),
    "a coin after another root": (_other_root, "not opened after"),
    "another coin": (_other_index, "not its positions coin 0"),
    "a refusal": (lambda o: frame({"type": "refused", "reason": "no"}), "refused: no"),
    "an answer of another kind": (lambda o: frame({"type": "closed"}), "did not answer with 'opening'"),
    "a malformed opening": (lambda o: frame({"type": "opening", "opening": {"coin": "00"}}), "malformed"),
    "a closed stream": (lambda o: b"", "stream closed"),
}


@pytest.mark.parametrize("case", sorted(ANSWERS))
def test_the_client_refuses_a_bad_answer_and_every_coin_after(case):
    answer, why = ANSWERS[case]
    liar = Liar(answer)
    client = R.RemoteChallenger.connect(liar.address)
    with pytest.raises(C.CoinError, match=why):
        client.open("positions", 0, root(0))
    liar.join()
    with pytest.raises(C.CoinError, match="session ended"):
        client.open("positions", 1, root(1))


BATCH = [("positions", 0, root(0)), ("replay", 0, root(0)), ("positions", 1, root(1))]


def _served(directory, signer, ask) -> tuple[bytes, object, str | None]:
    """`ask(client)` against a challenger served on one end of a socketpair (`serve`, on a clock that counts), the
    client then closing the session: the release log's bytes, what `ask` returned, and how `serve` ended."""
    directory.mkdir()
    session = R.open_session(C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host")), pinned_key(), HOST, signer,
                             directory, "s-1")
    a, b = socket.socketpair()
    ended: list[str | None] = []
    t = threading.Thread(target=lambda: ended.append(R.serve(a, session, itertools.count().__next__)))
    t.start()
    client = R.RemoteChallenger(b)
    try:
        got = ask(client)
    finally:
        if client._ended is None:
            client.close()
        t.join()
        a.close()
    return (directory / R.RELEASES).read_bytes(), got, ended[0]


def test_open_many_releases_and_logs_the_coins_as_opening_them_one_at_a_time_does(tmp_path, signer):
    """One round trip for three coins (two of them after one root, as a paced window's openings coin and the challenge
    after it are): the same openings, the release log byte for byte, and the in-process challenger's `open_many` the
    same coins again."""
    one, sequential, why = _served(tmp_path / "one", signer, lambda c: [c.open(*r) for r in BATCH])
    many, batched, why_many = _served(tmp_path / "many", signer, lambda c: c.open_many(BATCH))
    assert (why, why_many) == (None, None)
    assert batched == sequential and many == one
    events = [json.loads(line)["event"] for line in many.decode().splitlines()]
    assert events == ["commitment"] + ["root", "release"] * 3
    assert C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host")).open_many(BATCH) == batched


def test_a_withheld_session_releases_only_raw_coins_until_its_signed_close(tmp_path, signer):
    """A withheld session (`open_session(..., withhold=True)`) answers the hello with the layout alone and each open with
    the coin's raw bytes alone: no frame before the close carries the root, a salt or a path. Its close, once signed,
    releases the commitment and every coin's opening, which open against it after the root each followed. The client
    (`RemoteChallenger`, `ChallengerCoins`) reads the raw coins as it runs and settles the full openings at the close; a
    session served without `withhold` answers the hello with the commitment, as before."""
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    session = R.open_session(ch, pinned_key(), HOST, signer, tmp_path, "s-1", withhold=True)
    a, b = socket.socketpair()
    t = threading.Thread(target=lambda: R.serve(a, session, itertools.count().__next__))
    t.start()
    b.sendall(frame({"type": "hello"}))
    before = [reply(b)]
    for label, index, after in BATCH:
        b.sendall(frame({"type": "open", "label": label, "index": index, "after": after.hex()}))
        before.append(reply(b))
    b.sendall(frame({"type": "close"}))
    released = reply(b)
    b.close()
    t.join()
    a.close()
    assert before[0] == {"type": "layout", "layout": {"epoch": 0, "families": [list(f) for f in FAMILIES], "scheme": SCHEME}}
    assert [sorted(x["coin"]) for x in before[1:]] == [["after", "coin", "epoch", "index", "label"]] * len(BATCH)
    assert not any(ch.commitment.root.hex() in json.dumps(x) for x in before)
    assert released["type"] == "released" and released["released"]["commitment"] == ch.commitment.to_json()
    openings = [C.LiveOpening.from_json(o) for o in released["released"]["openings"]]
    assert [o.coin.hex() for o in openings] == [x["coin"]["coin"] for x in before[1:]]
    assert all(C.live_reason(ch.commitment, o, label, index, after) is None
               for o, (label, index, after) in zip(openings, BATCH))
    assert (tmp_path / R.CLOSE).is_file()

    directory = tmp_path / "client"
    directory.mkdir()
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    session = R.open_session(ch, pinned_key(), HOST, signer, directory, "s-2", withhold=True)
    a, b = socket.socketpair()
    t = threading.Thread(target=lambda: R.serve(a, session, itertools.count().__next__))
    t.start()
    client = R.RemoteChallenger(b)
    assert client.withheld and client.commitment is None
    coins = ChallengerCoins(C.LiveRecord(client, "positions"))
    coins.commit(root(7))
    read = coins(100)
    assert all(o.salt == b"" and o.path == () for o in coins.segments[0].openings)
    client.close()
    t.join()
    a.close()
    coins.settle()
    assert client.commitment == ch.commitment
    assert read == b"".join(o.coin for o in coins.segments[0].openings)[:100]
    assert all(C.live_reason(ch.commitment, o, "positions", i, root(7)) is None
               for i, o in enumerate(coins.segments[0].openings))
    assert coins.record.reason() is None
    _, got, _ = _served(tmp_path / "plain", signer, lambda c: c.commitment)
    assert got is not None


def _scripted(n: int, answer):
    """A client over a socketpair to a server that sends the challenger's commitment, reads `n` open frames and only
    then answers them with the frames `answer(openings)` gives, the honest openings in the order asked: so the client
    wrote them back to back before reading a reply."""
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    a, b = socket.socketpair()

    def run() -> None:
        with a:
            reply(a)
            a.sendall(frame({"type": "commitment", "commitment": ch.commitment.to_json()}))
            heads = [reply(a) for _ in range(n)]
            a.sendall(b"".join(answer([ch.open(h["label"], h["index"], bytes.fromhex(h["after"])) for h in heads])))
            try:
                while a.recv(1 << 16):
                    pass
            except ConnectionResetError:
                pass  # the client hung up on a reply it refused, with the batch's later replies unread

    t = threading.Thread(target=run)
    t.start()
    return R.RemoteChallenger(b), t


def _opening(o: C.LiveOpening) -> bytes:
    return frame({"type": "opening", "opening": o.to_json()})


@pytest.mark.parametrize("bad, why", [(_other_coin, "replay coin 0: coin does not open"),
                                      (_other_root, "replay coin 0: coin was not opened after")])
def test_a_batch_one_of_whose_coins_fails_live_reason_raises_whole_and_ends_the_session(bad, why):
    """No partial credit: the coins either side of the bad one are good, and none is returned."""
    client, t = _scripted(3, lambda ops: [_opening(ops[0]), bad(ops[1]), _opening(ops[2])])
    with pytest.raises(C.CoinError, match=why):
        client.open_many(BATCH)
    t.join()
    for again in (lambda: client.open("positions", 2, root(2)), lambda: client.open_many(BATCH)):
        with pytest.raises(C.CoinError, match=r"session ended \(.*replay coin 0.*\): a retry never redraws"):
            again()


def test_a_batch_answered_out_of_order_is_refused():
    client, t = _scripted(2, lambda ops: [_opening(ops[1]), _opening(ops[0])])
    with pytest.raises(C.CoinError, match="positions coin 0: coin is the epoch's replay coin 0, not its positions coin"):
        client.open_many(BATCH[:2])
    t.join()
    with pytest.raises(C.CoinError, match="session ended"):
        client.open(*BATCH[0])


def test_a_batch_the_challenger_refuses_midway_raises_and_releases_only_the_coins_before_it(tmp_path, signer):
    """`serve` releases the batch's coins before the refused one, logs them and signs the session's close with the
    refusal; the client returns none of them, the one after is never opened, and every later coin is refused."""
    batch = [BATCH[0], ("positions", 5, root(1)), BATCH[1]]
    held: list[R.RemoteChallenger] = []

    def ask(client):
        held.append(client)
        with pytest.raises(C.CoinError, match="refused: the next positions coin is 1, not 5"):
            client.open_many(batch)
        with pytest.raises(C.CoinError, match="session ended"):
            client.open(*BATCH[2])

    log, _, why = _served(tmp_path / "s", signer, ask)
    assert why == "the next positions coin is 1, not 5"
    assert [json.loads(line).get("label") for line in log.decode().splitlines()] == [None, "positions", "positions"]
    close = json.loads((tmp_path / "s" / R.CLOSE).read_text())
    assert (close["ended"], close["lines"]) == (why, 3)


def test_a_batch_past_its_bound_is_refused_before_a_frame_is_written(tmp_path, signer):
    def ask(client):
        with pytest.raises(ValueError, match=f"at most {R.MAX_BATCH} coins"):
            client.open_many([BATCH[0]] * (R.MAX_BATCH + 1))
        return client.open_many(BATCH[:1])

    _, got, why = _served(tmp_path / "s", signer, ask)
    assert why is None and [(o.label, o.index) for o in got] == [("positions", 0)]


def test_a_root_the_prover_rewrites_is_refused_against_the_challengers_log(tmp_path, signer):
    host = Host(tmp_path, signer)
    with R.RemoteChallenger.connect(host.address) as client:
        record = C.LiveRecord(client, "positions", itertools.count().__next__)
        record.record(root(0))
    assert host.ended() is None
    commitment, roots = host.trusted()
    ground = dataclasses.replace(record.openings[0], after=root(5))
    assert C.live_reason(commitment, ground, "positions", 0, root(5)) is None
    assert "not opened after" in C.order_reason(commitment, "positions", roots["positions"], [ground])


def _edit(path, f) -> None:
    lines = path.read_text().splitlines()
    path.write_text("".join(line + "\n" for line in f(lines)))


def _set(i: int, key: str, value) -> object:
    def f(lines):
        doc = json.loads(lines[i])
        doc[key] = value
        return lines[:i] + [json.dumps(doc, sort_keys=True, separators=(",", ":"))] + lines[i + 1:]
    return f


def _one_coin(tmp_path, signer) -> Host:
    host = Host(tmp_path, signer)
    with R.RemoteChallenger.connect(host.address) as client:
        C.LiveRecord(client, "positions", itertools.count().__next__).record(root(0))
    assert host.ended() is None
    return host


TAMPERED = {
    "missing": (None, "no challenger release log"),
    "another commitment's": (_set(0, "root", "00" * 64), "not the one its close record signs"),
    "a root without its release": (lambda lines: lines[:-1], "has 2 lines, not the 3"),
    "an appended release": (lambda lines: lines + [lines[-1]], "has 4 lines, not the 3"),
    "a rewritten root": (_set(1, "root", root(7).hex()), "not the one its close record signs"),
    "a rewritten release": (_set(2, "after", root(7).hex()), "not the one its close record signs"),
    "a rewritten time": (_set(2, "released_at", 0.5), "not the one its close record signs"),
    "a skipped index": (_set(1, "index", 1), "not the one its close record signs"),
    "a torn line": (lambda lines: lines[:-1] + [lines[-1][:9]], "not the one its close record signs"),
}


@pytest.mark.parametrize("case", sorted(TAMPERED))
def test_the_verifier_refuses_a_missing_or_tampered_release_log(tmp_path, signer, case):
    """Any edit to a signed log's entries, however well-formed, is refused against the chain its close signs."""
    host = _one_coin(tmp_path, signer)
    host.trusted()
    edit, why = TAMPERED[case]
    log = host.directory / R.RELEASES
    if edit is None:
        log.unlink()
    else:
        _edit(log, edit)
    with pytest.raises(C.CoinError, match=why):
        host.trusted()


def _rechain(host: Host, edit) -> None:
    """Rewrite the log with `edit` and the close's chain and line count to match it, under the old signature."""
    _edit(host.directory / R.RELEASES, edit)
    lines = R.log_lines(host.directory / R.RELEASES)
    close = json.loads((host.directory / R.CLOSE).read_text())
    close.update(chain=C.log_chain(lines).hex(), lines=len(lines))
    (host.directory / R.CLOSE).write_text(json.dumps(close))


CLOSES = {
    "an unsigned log": (lambda h: (h.directory / R.CLOSE).unlink(), "^no challenger close record"),
    "a log re-chained under the old signature": (lambda h: _rechain(h, _set(1, "root", root(7).hex())),
                                                 "close record's signature does not verify"),
    "a close renamed to another session": (lambda h: _close(h, session="s-2"), "close record's signature does not verify"),
    "how the session ended, rewritten": (lambda h: _close(h, ended="the stream closed"),
                                         "close record's signature does not verify"),
    "a close of another form": (lambda h: _close(h, extra=1), "close record is malformed"),
    "a close of a torn signature": (lambda h: _close(h, signature="00"), "close record's signature does not verify"),
}


def _close(host: Host, **changes) -> None:
    close = json.loads((host.directory / R.CLOSE).read_text())
    (host.directory / R.CLOSE).write_text(json.dumps({**close, **changes}))


@pytest.mark.parametrize("case", sorted(CLOSES))
def test_the_verifier_refuses_an_unsigned_log_or_a_close_the_challenger_did_not_sign(tmp_path, signer, case):
    """Refused at the bytes the ledger's end sealed, and past them at the close's own signature."""
    host = _one_coin(tmp_path, signer)
    edit, why = CLOSES[case]
    edit(host)
    with pytest.raises(C.CoinError, match="is not the one the auditor's ledger sealed"):
        host.trusted()
    with pytest.raises(C.CoinError, match=why):
        host.trusted_of()


STRUCTURE = {
    "another commitment's": (_set(0, "root", "00" * 64), "not coin commitment"),
    "a root without its release": (lambda lines: lines[:-1], "without its release"),
    "a rewritten root": (_set(1, "root", root(7).hex()), "malformed"),
    "a skipped index": (_set(1, "index", 1), "malformed"),
    "a torn line": (lambda lines: lines[:-1] + [lines[-1][:9]], "malformed"),
    "a line not in canonical form": (lambda lines: lines[:1] + [json.dumps(json.loads(lines[1]))] + lines[2:],
                                     "malformed"),
    "a time not an integer": (_set(2, "released_at", 0.5), "malformed"),
    "an index of another form": (lambda lines: lines[:1] + [lines[1].replace('"index":0', '"index":0e0')] + lines[2:],
                                 "malformed"),
}


@pytest.mark.parametrize("case", sorted(STRUCTURE))
def test_a_log_that_is_not_a_release_order_is_refused_even_unsigned(tmp_path, signer, case):
    host = _one_coin(tmp_path, signer)
    edit, why = STRUCTURE[case]
    _edit(host.directory / R.RELEASES, edit)
    with pytest.raises(C.CoinError, match=why):
        R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)


def test_the_verifier_refuses_a_session_whose_record_is_missing_or_another_keys(tmp_path, signer):
    host = Host(tmp_path, signer)
    with R.RemoteChallenger.connect(host.address):
        pass
    assert host.ended() is None
    other = C.ChallengerKey(4, 2, lms.public_key(lms_key(1)))
    with pytest.raises(C.CoinError, match="entry 0 is signed under challenger key .*, not the window's pinned key"):
        host.trusted(key=other)
    (host.directory / R.RECORD).unlink()
    with pytest.raises(C.CoinError, match="record.json is not the one the auditor's ledger sealed"):
        host.trusted()
    with pytest.raises(C.CoinError, match="^no challenger record"):
        host.trusted_of()


def test_a_ledger_signed_under_a_key_the_deployment_does_not_pin_is_refused(tmp_path, signer):
    """A forger who holds a key of its own can sign a whole ledger, chained and well-formed, naming its key's hash in
    every open or the pinned key's: the first is refused before a signature is read, the second at its signature."""
    host = Host(tmp_path, signer)
    honest(host)
    forger = lms_key(1)
    leaves = iter([0, 3, 0, 3])
    sign = lambda m: lms.sign_at(forger, next(leaves), m).encode()
    own = Ledger(tmp_path / "own.jsonl", sign, forger)
    own.open("s-1")
    own.end("s-1", host.directory, True)
    with pytest.raises(C.CoinError, match="entry 0 is signed under challenger key .*, not the window's pinned"):
        host.trusted(ledger=own.path)
    with pytest.raises(C.CoinError, match="challenger record's signature does not verify"):
        host.trusted(key=pinned_key(forger), ledger=own.path)
    copied = Ledger(tmp_path / "copied.jsonl", sign, forger)
    copied.key_sha512 = host.ledger.key_sha512
    copied.open("s-1")
    copied.end("s-1", host.directory, True)
    with pytest.raises(C.CoinError, match="entry 0: its signature doesn't verify under the challenger key"):
        host.trusted(ledger=copied.path)


def _ledger_edit(host: Host, edit) -> None:
    lines = R.ledger_lines(host.ledger.path)
    host.ledger.path.write_text("".join(line + "\n" for line in edit(lines)))


def _entry_set(i: int, change) -> object:
    def f(lines):
        doc = json.loads(lines[i])
        change(doc)
        return lines[:i] + [C.ledger_text(doc)] + lines[i + 1:]
    return f


LEDGERS = {
    "no ledger": (lambda h: h.ledger.path.unlink(), "^no auditor's ledger: .*none at"),
    "an empty ledger": (lambda h: _ledger_edit(h, lambda lines: []), "opens no challenger session for this window"),
    "a ledger truncated before the session's end": (lambda h: _ledger_edit(h, lambda lines: lines[:-1]),
                                                    "has no end for session s-1: .* older than its end"),
    "a ledger without its first entry": (lambda h: _ledger_edit(h, lambda lines: lines[1:]),
                                         "entry 0: not entry 0 of a .* ledger chained to its predecessor"),
    "an entry re-chained to another predecessor": (lambda h: _ledger_edit(h, _entry_set(1, lambda d: d.update(
        prev="00" * 64))), "entry 1: not entry 1 of a .* chained"),
    "an open moved to another host": (lambda h: _ledger_edit(h, _entry_set(0, lambda d: d["body"].update(
        host="elsewhere"))), "entry 0: its signature doesn't verify"),
    "an end rewritten to unclosed": (lambda h: _ledger_edit(h, _entry_set(1, lambda d: d["body"].update(closed=False))),
                                   "entry 1: its signature doesn't verify"),
    "an end's hello count rewritten": (lambda h: _ledger_edit(h, _entry_set(1, lambda d: d["body"].update(
        refused_hellos=0 if d["body"]["refused_hellos"] else 1))), "entry 1: its signature doesn't verify"),
    "an entry not in canonical form": (lambda h: _ledger_edit(h, lambda lines: [json.dumps(json.loads(lines[0]))]
                                                              + lines[1:]), "ledger entry 0 is malformed"),
    "an entry of another kind": (lambda h: _ledger_edit(h, _entry_set(1, lambda d: d.update(kind="close"))),
                                 "ledger entry 1 is malformed"),
    "an open whose root ends in a newline": (lambda h: _ledger_edit(h, _entry_set(0, lambda d: d["body"]["window"].update(
        root=WINDOW["root"] + "\n"))), "ledger entry 0 is malformed"),
    "an open in v1's window form under v2": (lambda h: _ledger_edit(h, _entry_set(0, lambda d: d["body"].update(
        window={"commit": WINDOW["root"], "period": "p-0"}))), "ledger entry 0 is malformed"),
    "an open naming no prover": (lambda h: _ledger_edit(h, _entry_set(0, lambda d: d["body"].pop("by"))),
                                 "ledger entry 0 is malformed"),
    "an entry past ASCII unescaped": (lambda h: _ledger_edit(h, lambda lines: lines[:1] + [
        json.dumps({**json.loads(lines[1]), "body": {**json.loads(lines[1])["body"], "ended": "closed\u2026"}},
                   sort_keys=True, separators=(",", ":"), ensure_ascii=False)]), "ledger entry 1 is malformed"),
}


def test_the_verifier_reads_an_entry_whose_text_past_ascii_the_auditor_escaped(tmp_path, signer):
    """An auditor's string past ASCII is in its line escaped (`ledger_text`), while its signature is over the entry's
    canonical JSON in UTF-8, as the challenger signs it: such an entry reads and verifies, and is refused only unescaped
    (`LEDGERS`)."""
    class Escaping(Ledger):
        def end(self, session, directory, closed, **changes):
            super().end(session, directory, closed, ended="closed by the client \u2026", **changes)

    host = Host(tmp_path, signer, ledger=Escaping(tmp_path / "windows.jsonl", signer))
    honest(host)
    assert '"ended":"closed by the client \\u2026"' in R.ledger_lines(host.ledger.path)[-1]
    host.trusted()
    entry = {"body": {"ended": "\u2026"}, "kind": "end", "prev": "0" * 128, "schema": C.LEDGER_SCHEMA, "seq": 1,
             "session": "s"}
    utf8 = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert C.ledger_message(bytes(16), entry) == hashlib.sha512(
        C._field(C.LEDGER_ROLE.encode()) + bytes(16)
        + C._field(tagged_sha512(C.LEDGER_TAGS[C.LEDGER_SCHEMA], utf8))).digest()


@pytest.mark.parametrize("case", sorted(LEDGERS))
def test_the_verifier_refuses_a_missing_truncated_or_tampered_ledger(tmp_path, signer, case):
    """The window's session is only one the latest ledger opens and ends, every entry chained and signed under the
    pinned key: a ledger cut before the session's end, or any entry edited, is refused."""
    host = Host(tmp_path, signer)
    honest(host)
    host.trusted()
    edit, why = LEDGERS[case]
    edit(host)
    with pytest.raises(C.CoinError, match=why):
        host.trusted()


def test_the_verifier_refuses_a_window_the_ledger_does_not_open_or_a_malformed_one(tmp_path, signer):
    host = Host(tmp_path, signer)
    honest(host)
    with pytest.raises(C.CoinError, match="opens no challenger session for this window"):
        host.trusted({"root": "44" * 64})
    for bad in ({"root": 1}, {"root": "11"}, {"root": WINDOW["root"] + "\n"}, {"root": WINDOW["root"] + " "},
                {"root": "AB" * 64}, {**WINDOW, "period": "p-0"}, {"commit": WINDOW["root"], "period": "p-0"},
                None):
        with pytest.raises(C.CoinError, match="window's identity is malformed"):
            host.trusted(bad)


def test_the_session_must_be_the_one_the_ledger_opened_on_its_host_and_sealed(tmp_path, signer):
    """The record names the host the ledger's open names, and the record, close and release log are the bytes its end
    sealed."""
    host = Host(tmp_path, signer, record_host=HOST)
    honest(host)
    with pytest.raises(C.CoinError, match=f"names host {HOST}, not {HOST}.s-1, the one the auditor's ledger opened"):
        host.trusted()
    host = Host(tmp_path, signer, "s-2", ledger=Ledger(tmp_path / "two.jsonl", signer))
    honest(host)
    record = host.directory / R.RECORD
    record.write_text(record.read_text() + "\n")
    with pytest.raises(C.CoinError, match="record.json is not the one the auditor's ledger sealed"):
        host.trusted()
    assert host.trusted_of()[0] == host.challenger.commitment


def _resealed(tmp_path, host: Host, name: str = "resealed", **changes):
    """A ledger opening `host`'s session and ending it at the leaves around its record, its end as `changes` says."""
    ledger = Ledger(tmp_path / f"{name}.jsonl", at(0, 3))
    ledger.open(host.session.id)
    ledger.end(host.session.id, host.directory, True, **changes)
    return ledger.path


def test_a_release_log_the_ledgers_end_did_not_seal_is_refused(tmp_path, signer):
    host = Host(tmp_path, signer)
    honest(host)
    log = host.directory / R.RELEASES
    log.write_text(log.read_text() + "\n")
    resealed = _resealed(tmp_path, host)
    log.write_text(log.read_text()[:-1])
    with pytest.raises(C.CoinError, match="release log is not the one the auditor's ledger sealed"):
        host.trusted(ledger=resealed)


def test_two_sessions_under_one_key_cant_both_pass_for_one_window(tmp_path, signer):
    """Two honest sessions under the one key, each signed and closed: the window takes only the session the auditor's
    ledger opens for it, and a ledger opening a second session for a window is refused."""
    first = Host(tmp_path, signer, "s-1", "one")
    honest(first)
    later = {"root": "22" * 64}
    second = Host(tmp_path, signer, "s-2", "two", ledger=first.ledger, window=later)
    honest(second)
    assert first.trusted()[0] == first.challenger.commitment
    assert second.trusted(later)[0] == second.challenger.commitment
    with pytest.raises(C.CoinError, match="record.json is not the one the auditor's ledger sealed"):
        second.trusted()
    with pytest.raises(C.CoinError, match="the challenger's session is s-2, not s-1, the one the auditor's ledger opens"):
        second.trusted_of()
    first.ledger.open("s-3")
    for host, window in ((first, WINDOW), (second, later)):
        with pytest.raises(C.CoinError, match="entry 4: a second open for root 1{16}… or session s-3"):
            host.trusted(window)


def test_one_root_gets_one_session_whichever_schema_opens_it(tmp_path, signer):
    """RT4-INT-24: a v1 open names its window by Commit and a period label; a v2 open of the same root later would give
    it a second draw of coins, so the ledger is refused for both. A v1 open alone is refused too: it names no prover and
    seals no refused hellos. And no v1 entry follows a v2 one."""
    ledger = Ledger(tmp_path / "windows.jsonl", signer, schema=C.LEDGER_SCHEMA_V1)
    first = Host(tmp_path, signer, "s-1", "one", ledger=ledger)
    honest(first)
    with pytest.raises(C.CoinError, match=f"opens this window's session s-1 under {C.LEDGER_SCHEMA_V1}, which names no"):
        first.trusted()
    ledger.schema = C.LEDGER_SCHEMA
    second = Host(tmp_path, signer, "s-2", "two", ledger=ledger)
    honest(second)
    for host in (first, second):
        with pytest.raises(C.CoinError, match="entry 2: a second open for root 1{16}… or session s-2"):
            host.trusted()
    ledger.lines = ledger.lines[:2]
    ledger.schema = C.LEDGER_SCHEMA
    ledger.open("s-3", {"root": "33" * 64})
    ledger.schema = C.LEDGER_SCHEMA_V1
    ledger.open("s-4", {"root": "44" * 64})
    with pytest.raises(C.CoinError, match="entry 3: not entry 3 of a .* ledger chained to its predecessor"):
        first.trusted()


@pytest.mark.parametrize("leaves", [[3, 3], [5, 2], [4, 6, 6, 7]])
def test_a_ledger_that_reuses_or_goes_back_on_a_leaf_is_refused(leaves):
    """RT4-INT-27: a host whose key state and ledger are restored signs its next entry at a leaf it has signed at
    already; the key signs at increasing leaves, so a ledger whose leaves don't increase was forked."""
    key_sha512 = hashlib.sha512(lms.public_key(PINNED)).hexdigest()

    def ledger(qs):
        sign, lines, entries = at(*qs), [], []
        for n, r in ((1, "11"), (2, "22")):
            entries += [("open", f"s-{n}", {"window": {"root": r * 64}, "host": f"{HOST}.s-{n}", "pinned": "ab" * 20,
                                            "key_sha512": key_sha512, "by": "proofs"}),
                        ("end", f"s-{n}", {"closed": True, "files": {f: "00" * 64 for f in C.SEALED_FILES},
                                           "record_leaf": (qs + [0, 0])[2 * n - 2] + 1, "commitment": "00" * 64,
                                           "refused_hellos": 0, "complete": True})]
        for kind, session, body in entries[:len(qs)]:
            lines.append(C.ledger_line(lines, kind, session, body, lms.public_key(PINNED)[:16], sign))
        return lines

    assert C.audited_session(ledger([2 * q + 1 for q in range(len(leaves))]), WINDOW, pinned_key(),
                             lms_verify).session == "s-1"
    i = next(i for i in range(1, len(leaves)) if leaves[i] <= leaves[i - 1])
    with pytest.raises(C.CoinError, match=f"entry {i}: signed at leaf {leaves[i]}, not after the last entry's "
                                          f"{leaves[i - 1]}"):
        C.audited_session(ledger(leaves), WINDOW, pinned_key(), lms_verify)


def test_a_record_leaf_outside_its_session_or_at_another_entry_is_refused(tmp_path, signer):
    """A v2 end names the leaf its record was signed at: strictly between its open's and its end's, and no entry's, so a
    record signed for one session can't be named by another's end."""
    host = Host(tmp_path, signer)
    honest(host)
    late = Ledger(tmp_path / "late.jsonl", at(2, 3))
    late.open("s-1")
    late.end("s-1", host.directory, True)
    with pytest.raises(C.CoinError, match="entry 1: record leaf 1 isn't between its open's and its end's, once"):
        host.trusted(ledger=late.path)
    shared = Ledger(tmp_path / "shared.jsonl", at(0, 1, 3))
    shared.open("s-1")
    shared.open("s-2", {"root": "22" * 64})
    shared.end("s-1", host.directory, True)
    with pytest.raises(C.CoinError, match="entry 2: record leaf 1 is also an entry's"):
        host.trusted(ledger=shared.path)


def test_a_ledger_whose_key_changes_without_a_signed_move_is_refused(tmp_path, signer):
    """A ledger handed to a new key (an auditor host moved to another project, its key rotated) with no move entry signed
    under the old key: whichever key the verifier pins, the entries under the other one are refused, so neither key's
    windows pass on that ledger."""
    host = Host(tmp_path, signer)
    honest(host)
    host.trusted()
    new = lms_key(2)
    leaves = itertools.count()
    moved = Ledger(host.ledger.path, lambda m: lms.sign_at(new, next(leaves), m).encode(), new)
    moved.lines = list(host.ledger.lines)
    moved.open("s-2", {"root": "22" * 64})
    with pytest.raises(C.CoinError, match="entry 2 is signed under challenger key .*, not the window's pinned key"):
        host.trusted()
    with pytest.raises(C.CoinError, match="entry 0 is signed under challenger key .*, not the window's pinned key"):
        host.trusted(key=pinned_key(new))
    moved.lines = list(host.ledger.lines)
    moved.key_sha512 = host.ledger.key_sha512
    moved.open("s-2", {"root": "22" * 64})
    with pytest.raises(C.CoinError, match="entry 2: its signature doesn't verify under the challenger key"):
        host.trusted()


ENDS = {
    "no record leaf": lambda h: {"record_leaf": DROP},
    "no commitment": lambda h: {"commitment": DROP},
    "a commitment of another form": lambda h: {"commitment": "11"},
    "no refused hellos": lambda h: {"refused_hellos": DROP},
    "a hello count of true": lambda h: {"refused_hellos": True},
    "files sealing no refused hellos": lambda h: {"files": {k: v for k, v in sealed(h.directory).items() if k != R.HELLOS}},
    "files sealing no close": lambda h: {"files": {k: v for k, v in sealed(h.directory).items() if k != R.CLOSE}},
}


@pytest.mark.parametrize("case", sorted(ENDS))
def test_an_end_that_names_no_record_leaf_commitment_or_hellos_is_refused(tmp_path, signer, case):
    """A v2 end names what the verifier reads (the record's leaf, the commitment, the hellos it refused) and seals every
    file of the session, `hellos.jsonl` included; one that doesn't is refused, not read as naming nothing."""
    host = Host(tmp_path, signer)
    honest(host)
    host.trusted(ledger=_resealed(tmp_path, host, "honest"))
    with pytest.raises(C.CoinError, match="the auditor's ledger's end of session s-1 is malformed"):
        host.trusted(ledger=_resealed(tmp_path, host, **ENDS[case](host)))


def test_the_record_is_the_one_signed_at_the_leaf_and_over_the_commitment_the_end_names(tmp_path, signer):
    host = Host(tmp_path, signer)
    honest(host)
    for i, changes in enumerate(({"record_leaf": 2}, {"commitment": "00" * 64})):
        with pytest.raises(C.CoinError, match="record is signed at leaf 1 over commitment .*, not at leaf "
                                              f"{changes.get('record_leaf', 1)} over"):
            host.trusted(ledger=_resealed(tmp_path, host, f"r{i}", **changes))


def test_the_refused_hellos_are_the_ones_the_end_sealed_and_counted(tmp_path, signer):
    """The hellos the challenger refused (each a wrong secret or a malformed hello) are sealed and counted by the
    ledger's end: a log of them edited, removed or miscounted is refused."""
    host = Host(tmp_path, signer)
    honest(host)
    hellos = host.directory / R.HELLOS
    with pytest.raises(C.CoinError, match="refused hellos are not the 1 the auditor's ledger sealed"):
        host.trusted(ledger=_resealed(tmp_path, host, "one", refused_hellos=1))
    hellos.write_text('{"why":"not the session\'s secret"}\n')
    with pytest.raises(C.CoinError, match="hellos.jsonl is not the one the auditor's ledger sealed"):
        host.trusted()
    with pytest.raises(C.CoinError, match="refused hellos are not the 0 the auditor's ledger sealed"):
        host.trusted_of()
    with pytest.raises(C.CoinError, match="refused hellos are not the 0 the auditor's ledger sealed"):
        host.trusted(ledger=_resealed(tmp_path, host, "zero", refused_hellos=0))
    host.trusted(ledger=_resealed(tmp_path, host, "counted"))
    hellos.unlink()
    with pytest.raises(C.CoinError, match="hellos.jsonl is not the one the auditor's ledger sealed"):
        host.trusted()
    with pytest.raises(C.CoinError, match=C.HELLOS_FORM):
        host.trusted_of()


def test_a_windows_prover_is_the_lane_its_open_names_it_for_else_its_opener(tmp_path, signer):
    host = Host(tmp_path, signer)
    honest(host)
    assert C.audited_session(R.ledger_lines(host.ledger.path), WINDOW, pinned_key(), lms_verify).prover == "proofs"
    named = Ledger(tmp_path / "for.jsonl", at(0, 3))
    named.open("s-1", **{"for": "vllm"})
    named.end("s-1", host.directory, True)
    assert C.audited_session(named.lines, WINDOW, pinned_key(), lms_verify).prover == "vllm"
    host.trusted(ledger=named.path)
    unnamed = Ledger(tmp_path / "unnamed.jsonl", at(0))
    unnamed.open("s-1", **{"for": 7})
    with pytest.raises(C.CoinError, match="ledger entry 0 is malformed"):
        C.audited_session(unnamed.lines, WINDOW, pinned_key(), lms_verify)


def _period_open(ledger: Ledger, name: str = "p-0", lane: str = "proofs", **changes) -> None:
    body = {"period": name, "lane": lane, "by": lane, "cap": 4, "circuit": "cd" * 64, "pinned": "ab" * 20,
            "key_sha512": ledger.key_sha512, "opened_utc": "2026-10-20T12:00:00Z", **changes}
    ledger.append("period-open", None, {k: v for k, v in body.items() if v is not DROP})


def _period_close(ledger: Ledger, windows: list, complete: bool, name: str = "p-0", lane: str = "proofs") -> None:
    ledger.append("period-close", None, {"period": name, "lane": lane, "by": lane, "windows": windows,
                                         "complete": complete, "closed_utc": "2026-10-20T13:00:00Z"})


def test_a_ledger_with_periods_is_read_and_a_close_that_misstates_its_windows_is_refused(tmp_path, signer):
    """The operator's signed `period-open` and `period-close` entries pass through the ledger's check
    (`verify_periods`): a
    period's windows are the opens of its lane between them, and a close that lists other windows or outcomes is
    refused, as is a period entry under another key or under v1."""
    ledger = Ledger(tmp_path / "windows.jsonl", signer)
    _period_open(ledger)
    host = Host(tmp_path, signer, ledger=ledger)
    honest(host)
    before = list(ledger.lines)
    _period_close(ledger, [{"root": WINDOW["root"], "session": "s-1", "complete": True}], True)
    assert host.trusted()[0] == host.challenger.commitment
    verify = lambda m, sig: lms_verify(pinned_key(), m, sig)
    periods = C.verify_periods([line.encode() for line in ledger.lines], ledger.I, verify)
    assert periods["p-0"]["windows"] == ["s-1"] and periods["p-0"]["close"] is not None
    audited = C.audited_session(ledger.lines, WINDOW, pinned_key(), lms_verify)
    assert (audited.period, audited.cap, audited.circuit) == ("p-0", 4, "cd" * 64)
    for windows, complete in (([{"root": WINDOW["root"], "session": "s-1", "complete": True}], False), ([], False),
                              ([{"root": WINDOW["root"], "session": "s-1", "complete": False}], False)):
        ledger.lines = list(before)
        _period_close(ledger, windows, complete)
        with pytest.raises(C.CoinError, match="entry 3: period p-0's close doesn't list exactly its windows"):
            host.trusted()
    other = Ledger(tmp_path / "other.jsonl", at(0))
    _period_open(other, key_sha512="00" * 64)
    with pytest.raises(C.CoinError, match="ledger entry 0 is signed under challenger key 0{16}..., not the window's"):
        C.audited_session(other.lines, WINDOW, pinned_key(), lms_verify)
    v1 = Ledger(tmp_path / "v1.jsonl", at(0), schema=C.LEDGER_SCHEMA_V1)
    _period_open(v1)
    with pytest.raises(C.CoinError, match="entry 0: a period-open is v2 and names its period and its lane"):
        C.audited_session(v1.lines, WINDOW, pinned_key(), lms_verify)


def test_a_reserved_window_is_read_when_its_open_binds_its_reserve(tmp_path, signer):
    """The operator's two-step window: a reserve, then an open binding it (its seq), carrying its request, by its
    prover and
    inside its minutes, is read, its period counting it at the reserve and closing an unopened reserve with root null.
    An open that misstates any of these, a malformed reserve, and a one-step open after the first reserve are refused."""
    ledger = Ledger(tmp_path / "windows.jsonl", signer)
    _period_open(ledger)
    ledger.reserve(RESERVED_SESSION)
    host = Host(tmp_path, signer, RESERVED_SESSION, ledger=ledger)
    honest(host)
    assert host.trusted()[0] == host.challenger.commitment
    audited = C.audited_session(ledger.lines, WINDOW, pinned_key(), lms_verify)
    assert (audited.session, audited.prover, audited.period, audited.cap) == (RESERVED_SESSION, "proofs", "p-0", 4)
    unopened = "s20261020T121000Z-cd34"
    ledger.reserve(unopened)
    _period_close(ledger, [{"root": WINDOW["root"], "session": RESERVED_SESSION, "complete": True},
                           {"root": None, "session": unopened, "complete": False}], False)
    assert host.trusted()[0] == host.challenger.commitment
    periods = C.verify_periods([line.encode() for line in ledger.lines], ledger.I,
                               lambda m, sig: lms_verify(pinned_key(), m, sig))
    assert periods["p-0"]["windows"] == [RESERVED_SESSION, unopened]
    for i, (reserve, opened, refusal) in enumerate((
            ({}, {"opened_by": "vllm"}, "was opened by 'vllm', not its reserve's prover"),
            ({}, {"opened_by": ["proofs"]}, "was opened by no lane, not its reserve's prover"),
            ({}, {"epoch": True}, "open doesn't carry its reserve's request"),
            ({}, {"minutes": 31}, "open doesn't carry its reserve's request"),
            ({}, {"reserve": True}, "binds no reserve of it"),
            ({}, {"reserve": 1}, "binds no reserve of it"),
            ({}, {"started_utc": "2026-10-20T12:30:01Z"}, "was opened outside its reserve's minutes"),
            ({}, {"started_utc": "2026-10-20T11:59:59Z"}, "was opened outside its reserve's minutes"),
            ({}, {"started_utc": "2026-10-20T12:05:60Z"}, "was opened outside its reserve's minutes"),
            ({"expires_utc": "2026-10-20T12:31:00Z"}, {}, "a reserve names its prover, its minutes, its start"),
            ({"minutes": 721, "expires_utc": "2026-10-21T00:01:00Z"}, {}, "a reserve names its prover, its minutes"),
            ({"started_utc": "2026-02-30T12:00:00Z"}, {}, "a reserve names its prover, its minutes, its start"),
            ({"for": 7}, {"for": "proofs"}, "a reserve names its prover"))):
        bad = Ledger(tmp_path / f"reserved{i}.jsonl", at(0, 1))
        bad.reserve(RESERVED_SESSION, **reserve)
        bad.open(RESERVED_SESSION, **opened)
        with pytest.raises(C.CoinError, match=re.escape(refusal)):
            C.audited_session(bad.lines, WINDOW, pinned_key(), lms_verify)
    unbound = Ledger(tmp_path / "unbound.jsonl", at(0, 1))
    unbound.reserve(RESERVED_SESSION)
    unbound.open("s-1")
    with pytest.raises(C.CoinError, match="entry 1: a v2 open without a reserve after the ledger's first reserve"):
        C.audited_session(unbound.lines, WINDOW, pinned_key(), lms_verify)
    loose = Ledger(tmp_path / "loose.jsonl", at(0))
    loose.reserve("s-1")
    with pytest.raises(C.CoinError, match="entry 0: a reserve is v2 and the first entry of its session s-1"):
        C.audited_session(loose.lines, WINDOW, pinned_key(), lms_verify)


def test_a_period_commits_its_cap_and_circuit_and_holds_no_window_past_its_cap(tmp_path, signer):
    """A period's open commits the most windows it may hold and its circuit (compute-accounting's CircuitWorkPeriod): a
    window past the cap is refused, as is a cap or circuit of another form, and a window in a period that committed
    neither (opened before they were required) isn't read, since its cap is unknown."""
    one = Ledger(tmp_path / "one.jsonl", signer)
    _period_open(one, cap=1)
    host = Host(tmp_path, signer, ledger=one)
    honest(host)
    assert host.trusted()[0] == host.challenger.commitment
    one.open("s-2", {"root": "22" * 64})
    with pytest.raises(C.CoinError, match="entry 3: period p-0's window 1 is past its cap 1"):
        host.trusted()
    for i, changes in enumerate(({"cap": 0}, {"cap": True}, {"cap": 2 ** 24 + 1}, {"cap": DROP},
                                 {"circuit": "CD" * 64}, {"circuit": "-x"}, {"circuit": DROP})):
        bad = Ledger(tmp_path / f"bad{i}.jsonl", at(0))
        _period_open(bad, **changes)
        with pytest.raises(C.CoinError, match="entry 0: period p-0's cap is a positive integer and its circuit"):
            C.audited_session(bad.lines, WINDOW, pinned_key(), lms_verify)
    old = Ledger(tmp_path / "old.jsonl", at(0, 1, 4))
    _period_open(old, cap=DROP, circuit=DROP)
    old.open("s-1")
    old.end("s-1", host.directory, True)
    with pytest.raises(C.CoinError, match="period p-0, which holds session s-1, commits no cap or circuit"):
        C.audited_session(old.lines, WINDOW, pinned_key(), lms_verify)
    named = Ledger(tmp_path / "named.jsonl", at(0))
    _period_open(named, circuit="verity/pouw/h1@v1")
    with pytest.raises(C.CoinError, match="entry 0: period p-0's cap is a positive integer and its circuit a SHA-512"):
        C.audited_session(named.lines, WINDOW, pinned_key(), lms_verify)


XOR8 = PrimitiveDefinition("RemoteTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)


def _declared(steps: int, annotate: bool = False) -> tuple[bytes, str]:
    """A static window's declared Program of `steps` calls, as the H1 rehearsal writes it before it serves
    (`pouw_h1_rehearsal.build_program`): its descriptor's canonical JSON without annotations, and its `program_digest`,
    the period's ``--circuit`` (with `annotate`, a descriptor that names its window in its annotations)."""
    def sig(S):
        return (("x", Array(steps, Value(8))), ("w", Array(steps, Value(8)))), Array(steps, Value(8))

    prog = Program(bind(CompositeDefinition(
        "RemoteTestWindow", 0, (), sig,
        lambda B, S, x, w: coll_concat(sig(S)[1], [B.call(XOR8, x[i], w[i]) for i in range(steps)]), register=False)))
    desc = encode_program(prog)
    if annotate:
        desc["annotations"] = {"window": WINDOW["root"][:32]}
    return canonical_json({k: v for k, v in desc.items() if k != "annotations"}), program_digest(desc)


def _rehearsal(tmp_path, signer, circuit: str, period: bool = True) -> Host:
    """A served H1 rehearsal's ledger: a prover's period opened with cap 1 and `circuit`, its one window reserved, opened,
    served and ended complete, then the period closed (no period with `period` false)."""
    tmp_path.mkdir()
    ledger = Ledger(tmp_path / "windows.jsonl", signer)
    if period:
        _period_open(ledger, lane="vllm", by="vllm", cap=1, circuit=circuit)
    ledger.reserve(RESERVED_SESSION, by="vllm")
    host = Host(tmp_path, signer, RESERVED_SESSION, ledger=ledger)
    honest(host)
    if period:
        _period_close(ledger, [{"root": WINDOW["root"], "session": RESERVED_SESSION, "complete": True}], True, lane="vllm")
    return host


def test_a_window_counts_toward_its_period_only_when_it_proves_the_circuit_the_period_committed(tmp_path, signer):
    """The period's open commits its circuit (the declared circuit's SHA-512), and `credited_session` counts the
    window only when the descriptor of the circuit its sessions prove hashes to it. The rehearsal's shape is credited;
    a window proving another circuit (another declaration, or the same one's bytes written otherwise), a descriptor that
    isn't bytes, and a window in no period count toward none, each naming the fix."""
    descriptor, digest = _declared(4)
    assert hashlib.sha512(descriptor).hexdigest() == digest == _declared(4, annotate=True)[1]
    host = _rehearsal(tmp_path / "honest", signer, digest)
    lines = R.ledger_lines(host.ledger.path)
    audited = C.credited_session(lines, WINDOW, pinned_key(), HOST_PINS, MAIN, descriptor, lms_verify)
    assert (audited.session, audited.prover, audited.period, audited.cap, audited.circuit) == (
        RESERVED_SESSION, "vllm", "p-0", 1, digest)
    assert C.circuit_reason("p-0", digest, descriptor) is None
    other, other_digest = _declared(5)
    assert other_digest != digest
    for wrong in (other, json.dumps(json.loads(descriptor), indent=1).encode(), descriptor + b"\n"):
        got = hashlib.sha512(wrong).hexdigest()
        with pytest.raises(C.CoinError, match=re.escape(
                f"the circuit the window's sessions prove (descriptor SHA-512 {got[:16]}...) is not {digest[:16]}..., "
                "the one period p-0 committed: prove the period's declared circuit, or open a period whose --circuit "
                "is this descriptor's SHA-512")):
            C.credited_session(lines, WINDOW, pinned_key(), HOST_PINS, MAIN, wrong, lms_verify)
    for bad in (descriptor.decode(), json.loads(descriptor), bytearray(descriptor), None):
        with pytest.raises(C.CoinError, match=re.escape(C.DESCRIPTOR_FORM)):
            C.credited_session(lines, WINDOW, pinned_key(), HOST_PINS, MAIN, bad, lms_verify)
    assert C.circuit_reason("p-0", None, descriptor) == (
        "period p-0 commits no circuit, so no window counts toward it: open a period whose --circuit is the declared "
        "circuit's SHA-512")
    loose = _rehearsal(tmp_path / "loose", signer, digest, period=False)
    assert C.audited_session(R.ledger_lines(loose.ledger.path), WINDOW, pinned_key(), lms_verify).period is None
    with pytest.raises(C.CoinError, match=re.escape("holds the window's session in no period, so it counts toward none")):
        C.credited_session(R.ledger_lines(loose.ledger.path), WINDOW, pinned_key(), HOST_PINS, MAIN, descriptor,
                           lms_verify)
    with pytest.raises(C.CoinError, match="the auditor's ledger opens no challenger session for this window"):
        C.credited_session(lines, {"root": "22" * 64}, pinned_key(), HOST_PINS, MAIN, descriptor, lms_verify)


MAIN_PIN, OLD_PIN, LATER_PIN, BRANCH_PIN = "c0" * 20, "c1" * 20, "c2" * 20, "dd" * 20
PINS = [C.pin_line(OLD_PIN, "e0" * 20, "2026-10-20T08:00:00Z"), C.pin_line(MAIN_PIN, "e1" * 20, "2026-10-21T13:00:00Z")]
FIRST_PARENT = [LATER_PIN, MAIN_PIN, OLD_PIN]


def _pinned_host(tmp_path, pinned: str, **seal) -> Host:
    """A served window, on a host of its own key's state, whose open names the tree its host ran (`pinned`) and seals
    the host's pins log as `seal` says (``pins=``; `PINS`, both lines, unless set; DROP: an open from a gate that seals
    none)."""
    tmp_path.mkdir()
    signer = at(*range(16))
    ledger = Ledger(tmp_path / "windows.jsonl", signer)
    opened = ledger.open
    ledger.open = lambda session, window=WINDOW: opened(session, window, pinned=pinned,
                                                        pins=seal.get("pins", C.pins_seal(PINS)))
    ledger.reserve(RESERVED_SESSION)
    host = Host(tmp_path, signer, RESERVED_SESSION, ledger=ledger)
    honest(host)
    return host


def test_a_session_counts_only_from_a_tree_the_hosts_pins_log_records_as_mains(tmp_path):
    """RT4-INT-40: the open names the commit its host ran and seals the host's pins log as it stood, and
    `pinned_session` takes the session only when one of the sealed lines pins that commit. An honest session at a main
    commit is read under any later extension of the log; a commit pinned off main, or only after the open, is refused
    naming the fix; so are a missing, malformed, cut or edited pins log and an open that seals none."""
    lines = R.ledger_lines(_pinned_host(tmp_path / "honest", MAIN_PIN).ledger.path)
    later = PINS + [C.pin_line(LATER_PIN, LATER_PIN, "2026-10-21T14:00:00Z")]
    for log in (PINS, later):
        assert C.pinned_session(lines, WINDOW, pinned_key(), log, FIRST_PARENT, lms_verify) == C.audited_session(
            lines, WINDOW, pinned_key(), lms_verify)

    def refused(lines, pins, why):
        with pytest.raises(C.CoinError, match=re.escape(why)):
            C.pinned_session(lines, WINDOW, pinned_key(), pins, FIRST_PARENT, lms_verify)

    for pin in (BRANCH_PIN, LATER_PIN):
        off = R.ledger_lines(_pinned_host(tmp_path / pin, pin).ledger.path)
        refused(off, later, f"session {RESERVED_SESSION} ran commit {pin}, which the host's pins log doesn't record "
                            "as a commit of main's history: repin the challenger host to a commit origin/main "
                            "contains (vy-challenger-pin) and rerun the window there")
    refused(lines, None, C.NO_PINS)
    refused(lines, "\n".join(PINS), f"the host's pins log is malformed: {C.PINS_FORM}")
    for bad in (PINS[1].replace("c0", "C0"), PINS[1].replace(":", ": "), PINS[1][:-1] + ',"x":1}',
                C.ledger_text({"commit": MAIN_PIN, "main": "e1" * 20, "pinned_utc": "2026-02-30T13:00:00Z"}), 5):
        refused(lines, [PINS[0], bad], f"the host's pins log line 1 is malformed: {C.PINS_FORM}")
    refused(lines, PINS[:1], f"the host's pins log has 1 lines, fewer than the 2 the open of session "
                             f"{RESERVED_SESSION} sealed: take infra's newest published pins log")
    edited = [PINS[0], C.pin_line(MAIN_PIN, "e2" * 20, "2026-10-21T13:00:00Z")]
    for log in (edited, PINS[::-1]):
        refused(lines, log, f"the host's pins log is not the one the open of session {RESERVED_SESSION} sealed: its "
                            "first 2 lines were edited or rewritten")
    seals = (DROP, None, {"lines": 0, "sha512": C.log_digest([])}, {"lines": True, "sha512": C.log_digest(PINS[:1])},
             {**C.pins_seal(PINS), "at": 1}, {"lines": 2, "sha512": C.log_digest(PINS).upper()})
    for i, seal in enumerate(seals):
        unsealed = R.ledger_lines(_pinned_host(tmp_path / f"seal-{i}", MAIN_PIN, pins=seal).ledger.path)
        refused(unsealed, PINS, f"the auditor's ledger's open of session {RESERVED_SESSION} seals no pins log "
                                "(pins: {lines, sha512}): open the window on a host whose gate seals its pins log in "
                                "each open")
    with pytest.raises(C.CoinError, match="the auditor's ledger opens no challenger session for this window"):
        C.pinned_session(lines, {"root": "22" * 64}, pinned_key(), PINS, FIRST_PARENT, lms_verify)
    for bad in (("C0" * 20, MAIN_PIN, "2026-10-21T13:00:00Z"), (MAIN_PIN, "e1" * 19, "2026-10-21T13:00:00Z"),
                (MAIN_PIN, MAIN_PIN, "2026-10-21 13:00:00Z")):
        with pytest.raises(ValueError, match=re.escape(C.PINS_FORM)):
            C.pin_line(*bad)


#: seven commits the challenger host was pinned to: a branch's commits, which landed through the queue's merge `MERGE`,
#: so each is in main's history and none is on its first-parent line
BRANCH_PINS = ("5902f9e85d36b0de284dd4de819cad464a801739", "63179aad879e4a38c025746abde7d747b6e73fe3",
               "a655828a4fc022fd6527f64f1c20d30b610f0c18", "aa782259fb3b6d7ffee53fafa63bc571403f6d49",
               "c33bf091a3f12522fd6e6be04046fdb05d06a6f6", "d2ad7bdea69504f6b4ed848207f65a4d37f31a8b",
               "882770e6009e73bdf015f4069fa77ef698d7f326")
MERGE = "1018753ee5ff8c659fa8e8c5ccbc31b0d2b1a366"


def test_a_session_is_credited_only_from_a_logged_commit_on_mains_first_parent_line(tmp_path, signer):
    """`credited_session` reads the window's session through `pinned_session`, the pin first, then the circuit.
    A session whose open names `PIN`, main's tip when vy-challenger-1 was repinned and the one line of the pins log it
    published, sealed in the open, is credited against the auditor's own first-parent list of main. Refused, each with
    its reason and no fallback to `audited_session`: a pin on main's first-parent line the log doesn't record, an absent
    log, an open that seals none, no copy of main or a malformed one, and a logged pin that is in main's history but not
    on its first-parent line (`BRANCH_PINS[-1]`, the second parent of `MERGE`, which landed it, as the other six)."""
    assert MAIN[0] == PIN and MAIN[-1] == "6c2de12715770b0cb1662e5036e5ee239ff191c1" and MERGE in MAIN
    assert not set(BRANCH_PINS) & set(MAIN)
    descriptor, digest = _declared(4)
    host = _rehearsal(tmp_path / "honest", signer, digest)
    lines = R.ledger_lines(host.ledger.path)
    audited = C.credited_session(lines, WINDOW, pinned_key(), HOST_PINS, MAIN, descriptor, lms_verify)
    assert audited == C.pinned_session(lines, WINDOW, pinned_key(), HOST_PINS, MAIN, lms_verify) == C.audited_session(
        lines, WINDOW, pinned_key(), lms_verify)
    assert audited.circuit == digest
    assert host.trusted_of()[0] == host.challenger.commitment

    def refused(lines, pins, main, why):
        for read in (lambda: C.credited_session(lines, WINDOW, pinned_key(), pins, main, descriptor, lms_verify),
                     lambda: C.pinned_session(lines, WINDOW, pinned_key(), pins, main, lms_verify)):
            with pytest.raises(C.CoinError, match=re.escape(why)):
                read()

    branch = BRANCH_PINS[-1]
    logged = HOST_PINS + [C.pin_line(branch, MERGE, "2026-10-21T03:05:00Z")]
    off = R.ledger_lines(_pinned_host(tmp_path / "branch", branch, pins=C.pins_seal(logged)).ledger.path)
    refused(off, logged, MAIN, f"session {RESERVED_SESSION} ran commit {branch}, which the host's pins log records but "
                               "which is not on main's first-parent line in the verifier's copy (`git rev-list "
                               "--first-parent origin/main`): a branch commit merged into main is in main's history but "
                               "no tree main's check ran on; repin the challenger host to a commit on main's "
                               "first-parent line (vy-challenger-pin) and rerun the window there")
    for pin in BRANCH_PINS:
        assert "not on main's first-parent line" in C.first_parent_reason(RESERVED_SESSION, pin, MAIN)
    assert C.first_parent_reason(RESERVED_SESSION, PIN, MAIN) is None
    unlogged = R.ledger_lines(_pinned_host(tmp_path / "unlogged", MAIN[1], pins=C.pins_seal(HOST_PINS)).ledger.path)
    refused(unlogged, HOST_PINS, MAIN, f"session {RESERVED_SESSION} ran commit {MAIN[1]}, which the host's pins log "
                                       "doesn't record as a commit of main's history")
    refused(lines, None, MAIN, C.NO_PINS)
    unsealed = R.ledger_lines(_pinned_host(tmp_path / "unsealed", PIN, pins=DROP).ledger.path)
    refused(unsealed, HOST_PINS, MAIN, f"the auditor's ledger's open of session {RESERVED_SESSION} seals no pins log")
    refused(lines, HOST_PINS, None, C.NO_MAIN)
    for bad in ("\n".join(MAIN), [PIN.upper()], [PIN[:10]], [PIN, None], tuple(MAIN)):
        refused(lines, HOST_PINS, bad, f"the verifier's copy of main's history is malformed: {C.MAIN_FORM}")
    with pytest.raises(C.CoinError, match=re.escape(f"{C.NO_MAIN} (none at {tmp_path / 'absent'})")):
        R.main_lines(tmp_path / "absent")
    wrong, _ = _declared(5)
    with pytest.raises(C.CoinError, match="not on main's first-parent line"):
        C.credited_session(off, WINDOW, pinned_key(), logged, MAIN, wrong, lms_verify)
    opened = json.loads(next(line for line in lines if json.loads(line)["kind"] == "open"))["body"]
    assert C.pins_reason(RESERVED_SESSION, opened, HOST_PINS) is None
    assert C.pins_reason(RESERVED_SESSION, {k: v for k, v in opened.items() if k != "pinned"}, HOST_PINS) == (
        f"session {RESERVED_SESSION} ran commit None, which the host's pins log doesn't record as a commit of main's "
        "history: repin the challenger host to a commit origin/main contains (vy-challenger-pin) and rerun the window "
        "there")


def test_the_ledger_is_the_longest_of_the_operators_heads_when_each_extends_the_others(tmp_path, signer):
    """Labels aren't authenticated, so the verifier checks every head the auditor host's operator published under the
    pin and takes the
    longest only when any two extend one another: a head forked, cut and re-extended, or malformed is refused."""
    first = Host(tmp_path, signer, "s-1", "one")
    honest(first)
    early = tmp_path / "early.jsonl"
    early.write_text(first.ledger.path.read_text())
    later = {"root": "22" * 64}
    second = Host(tmp_path, signer, "s-2", "two", ledger=first.ledger, window=later)
    honest(second)
    for heads in ([early, first.ledger.path], [first.ledger.path, early, first.ledger.path]):
        assert R.trusted_session(second.directory, heads, later, pinned_key(), lms_verify)[0] == second.challenger.commitment
    with pytest.raises(C.CoinError, match="opens no challenger session for this window"):
        R.trusted_session(second.directory, [early], later, pinned_key(), lms_verify)
    fork = Ledger(tmp_path / "fork.jsonl", at(8))
    fork.lines = R.ledger_lines(early)
    fork.open("s-9", {"root": "99" * 64})
    with pytest.raises(C.CoinError, match="published heads 1 and 2 don't extend one another: its ledger was forked"):
        R.trusted_session(second.directory, [early, first.ledger.path, fork.path], later, pinned_key(), lms_verify)
    torn = tmp_path / "torn.jsonl"
    torn.write_text(first.ledger.path.read_text()[:-9] + "\n")
    with pytest.raises(C.CoinError, match="published head 1 is refused: the auditor's ledger entry 3 is malformed"):
        R.trusted_session(second.directory, [early, torn], later, pinned_key(), lms_verify)
    for heads, why in ((None, C.NO_AUDIT), ([], C.NO_AUDIT), ("lines", C.HEADS_FORM)):
        with pytest.raises(C.CoinError, match=re.escape(why)):
            C.ledger_head(heads, pinned_key(), lms_verify)


def test_an_end_whose_reason_is_past_ascii_is_read_escaped(tmp_path, signer):
    host = Host(tmp_path, signer)
    honest(host)
    path = _resealed(tmp_path, host, ended="the client closed it \u2026")
    assert "\\u2026" in path.read_text()
    assert host.trusted(ledger=path)[0] == host.challenger.commitment


def test_the_pin_is_read_from_the_operators_key_record():
    """`key_of_record` reads a ``challenger-key/v1`` record (the store artifact the auditor host's operator labels
    ``challenger-key current``)
    and refuses one whose public key isn't the one its SHA-512 names, or of another form."""
    public = lms.public_key(PINNED)
    record = {"schema": "challenger-key/v1", "W": 4, "H": 4, "pubkey": public.hex(),
              "pubkey_sha512": hashlib.sha512(public).hexdigest(), "retired": [], "valid_from_utc": "2026-10-20T09:45:24Z"}
    assert C.key_of_record(record) == pinned_key()
    with pytest.raises(C.CoinError, match="is not the one whose SHA-512 it names"):
        C.key_of_record({**record, "pubkey": public[::-1].hex()})
    for bad in ({**record, "schema": "challenger-key/v2"}, {**record, "pubkey": public.hex().upper()},
                {**record, "pubkey_sha512": "00"}, {**record, "W": 3}, None):
        with pytest.raises(C.CoinError, match="challenger key"):
            C.key_of_record(bad)


def test_a_release_log_rewritten_after_its_end_is_refused(tmp_path, signer):
    """RT4-INT-26: the round roots in the release log rewritten after the session's end, to roots chosen after every
    coin: the log is no longer the one the ledger's end sealed (nor the one the close signs)."""
    host = Host(tmp_path, signer)
    honest(host)
    log = host.directory / R.RELEASES
    early = [json.loads(line)["root"] for line in log.read_text().splitlines() if json.loads(line)["event"] == "root"]
    text = log.read_text()
    for r in early:
        text = text.replace(r, hashlib.sha512(b"chosen after every coin " + r.encode()).hexdigest())
    log.write_text(text)
    with pytest.raises(C.CoinError, match="release log is not the one"):
        host.trusted()


@pytest.mark.parametrize("address", ["7791", ":7791", "host:port", "host:"])
def test_an_address_is_host_and_port(address):
    with pytest.raises(ValueError, match="HOST:PORT"):
        R.RemoteChallenger.connect(address)


def _clock_and_sleep():
    now, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s

    return (lambda: now[0]), sleep, slept


def test_a_session_whose_tunnel_listens_late_is_connected_once_it_listens():
    clock, sleep, slept = _clock_and_sleep()
    tries = []

    def connect(address, secret):
        tries.append((address, secret))
        if len(tries) < 3:
            raise ConnectionRefusedError(111, "Connection refused")
        return "client"

    assert R.connect_when_listening("127.0.0.1:7811", b"s", connect, 120, clock, sleep) == "client"
    assert tries == [("127.0.0.1:7811", b"s")] * 3 and slept == [0.5, 0.5]


def test_a_session_that_never_listens_raises_once_its_wait_is_over():
    clock, sleep, slept = _clock_and_sleep()

    def refused(address, secret):
        raise ConnectionRefusedError(111, "Connection refused")

    with pytest.raises(C.CoinError, match="nothing listened"):
        R.connect_when_listening("127.0.0.1:7811", None, refused, 2, clock, sleep)
    assert slept == [0.5] * 4


@pytest.mark.parametrize("error", [C.CoinError("the challenger refused the hello"), ConnectionResetError(104, "reset"),
                                   ValueError("a challenger's address is HOST:PORT")])
def test_any_other_failure_to_connect_raises_at_once(error):
    clock, sleep, slept = _clock_and_sleep()
    tries = []

    def failing(address, secret):
        tries.append(address)
        raise error

    with pytest.raises(type(error)):
        R.connect_when_listening("127.0.0.1:7811", None, failing, 120, clock, sleep)
    assert len(tries) == 1 and slept == []
