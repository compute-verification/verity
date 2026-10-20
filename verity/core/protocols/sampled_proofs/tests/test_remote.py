"""The challenger as a service on its own host (`remote`), over loopback: it signs and writes its record before its first
coin opens, answers each root with that family's next coin and logs the release before the coin leaves; it refuses
every malformed or out-of-order frame and ends the session there.  The client (`RemoteChallenger`) is `LiveRecord`'s
challenger as it is, and refuses a coin that doesn't open.  The verifier, reading the challenger's directory, trusts only
the commitment it signed and checks the openings against the roots it logged, so a prover that rewrites a coin's root is
refused."""

from __future__ import annotations

import dataclasses
import itertools
import json
import socket
import threading

import pytest

from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

from test_challenger import HOST, entropy, lms_key, lms_verify, verifier_key

SCHEME = "verity/test/coins/v1"
FAMILIES = [("positions", 3), ("replay", 2)]


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


class Host:
    """The challenger on its host: a challenger over `FAMILIES` whose record `open_session` signed with the test key and
    wrote to `directory`, serving one session on a loopback port; `reason` is how `serve` ended it."""

    def __init__(self, tmp_path, signed: bool = True) -> None:
        self.directory = tmp_path / "challenger"
        self.directory.mkdir()
        self.challenger = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
        if signed:
            k = lms_key(0)
            state_path = tmp_path / "key.lms"
            lms.LeafState.create(state_path, lms.public_key(k), k.params.H)
            state = lms.LeafState(state_path, lms.public_key(k), k.params.H)
            R.open_session(self.challenger, verifier_key(), HOST, lambda m: lms.sign(k, state, m).encode(),
                           self.directory)
            state.close()
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
                self.reason = R.serve(conn, self.challenger, self.directory, itertools.count().__next__)
            except Exception as e:
                self.error = e

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason


def test_an_honest_session_checks_against_what_the_challenger_signed_and_logged(tmp_path):
    host = Host(tmp_path)
    client = R.RemoteChallenger.connect(host.address)
    assert client.commitment == host.challenger.commitment
    records = {label: C.LiveRecord(client, label, itertools.count().__next__) for label, _ in FAMILIES}
    for i in range(2):
        for record in records.values():
            record.record(root(i))
    records["positions"].record(root(2))
    client.close()
    assert host.ended() is None

    commitment, roots = R.trusted_session(host.directory, verifier_key(), lms_verify)
    assert commitment == client.commitment
    assert roots == {"positions": (root(0), root(1), root(2)), "replay": (root(0), root(1))}
    for label, record in records.items():
        assert record.reason() is None
        assert C.order_reason(commitment, label, roots[label], record.openings) is None


def test_the_challenger_hangs_up_only_after_the_client_so_its_last_frame_arrives(tmp_path):
    host = Host(tmp_path)
    with socket.create_connection(host.listener.getsockname()) as s:
        s.sendall(frame({"type": "hello"}))
        assert reply(s)["type"] == "commitment"
        s.sendall(frame({"type": "close"}))
        assert reply(s) == {"type": "closed"}
        assert s.recv(1) == b""
        assert host._thread.is_alive()
    assert host.ended() is None


def test_the_challenger_serves_only_once_its_record_is_written(tmp_path):
    ch = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    a, b = socket.socketpair()
    with a, b, pytest.raises(C.CoinError, match="open_session"):
        R.serve(a, ch, tmp_path)
    ch.open("positions", 0, root(0))
    with pytest.raises(C.CoinError, match="before any coin opens"):
        R.open_session(ch, verifier_key(), HOST, lambda m: b"", tmp_path)
    assert not (tmp_path / R.RECORD).exists()


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
def test_every_malformed_or_out_of_order_frame_is_refused_and_ends_the_session(tmp_path, case):
    host = Host(tmp_path)
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
    commitment, roots = R.trusted_session(host.directory, verifier_key(), lms_verify)
    assert sum(len(r) for r in roots.values()) == opened


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


def test_a_root_the_prover_rewrites_is_refused_against_the_challengers_log(tmp_path):
    host = Host(tmp_path)
    with R.RemoteChallenger.connect(host.address) as client:
        record = C.LiveRecord(client, "positions", itertools.count().__next__)
        record.record(root(0))
    assert host.ended() is None
    commitment, roots = R.trusted_session(host.directory, verifier_key(), lms_verify)
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
        return lines[:i] + [json.dumps(doc)] + lines[i + 1:]
    return f


TAMPERED = {
    "missing": (None, "no challenger release log"),
    "another commitment's": (_set(0, "root", "00" * 64), "not coin commitment"),
    "a root without its release": (lambda lines: lines[:-1], "without its release"),
    "a rewritten root": (_set(1, "root", root(7).hex()), "malformed"),
    "a skipped index": (_set(1, "index", 1), "malformed"),
    "a torn line": (lambda lines: lines[:-1] + [lines[-1][:9]], "unreadable"),
}


@pytest.mark.parametrize("case", sorted(TAMPERED))
def test_the_verifier_refuses_a_missing_or_tampered_release_log(tmp_path, case):
    host = Host(tmp_path)
    with R.RemoteChallenger.connect(host.address) as client:
        C.LiveRecord(client, "positions", itertools.count().__next__).record(root(0))
    assert host.ended() is None
    edit, why = TAMPERED[case]
    log = host.directory / R.RELEASES
    if edit is None:
        log.unlink()
    else:
        _edit(log, edit)
    with pytest.raises(C.CoinError, match=why):
        R.trusted_session(host.directory, verifier_key(), lms_verify)


def test_the_verifier_refuses_a_session_whose_record_is_missing_or_another_keys(tmp_path):
    host = Host(tmp_path)
    with R.RemoteChallenger.connect(host.address):
        pass
    assert host.ended() is None
    other = C.ChallengerKey(4, 2, lms.public_key(lms_key(1)))
    with pytest.raises(C.CoinError, match="does not verify"):
        R.trusted_session(host.directory, other, lms_verify)
    (host.directory / R.RECORD).unlink()
    with pytest.raises(C.CoinError, match="no challenger record"):
        R.trusted_session(host.directory, verifier_key(), lms_verify)


@pytest.mark.parametrize("address", ["7791", ":7791", "host:port", "host:"])
def test_an_address_is_host_and_port(address):
    with pytest.raises(ValueError, match="HOST:PORT"):
        R.RemoteChallenger.connect(address)
