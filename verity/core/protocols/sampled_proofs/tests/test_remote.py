"""The challenger as a service on its own host (`remote`), over loopback: it signs and writes its record before its first
coin opens, answers each root with that family's next coin and logs the release before the coin leaves; it refuses
every malformed or out-of-order frame and ends the session there, and however the session ends it signs its release log
(the close record).  The client (`RemoteChallenger`) is `LiveRecord`'s challenger as it is, and refuses a coin that
doesn't open.  The verifier, reading the challenger's directory and the auditor's session record, trusts only the one
session the record names for the window, its commitment and the roots its signed log shows, so a prover that rewrites a
coin's root, an entry of the log, or presents an unsigned log, an unclosed session or a second session is refused."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import re
import socket
import threading

import pytest

from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

from test_challenger import HOST, entropy, lms_key, lms_verify, verifier_key

SCHEME = "verity/test/coins/v1"
FAMILIES = [("positions", 3), ("replay", 2)]
WINDOW = {"commit": "11" * 64, "period": 0, "pouw_salt": "22" * 64, "stream": "33" * 64}


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
    """The challenger key's signer on its host, at its next leaf: one for every session under the key."""
    k = lms_key(0)
    path = tmp_path / "key.lms"
    lms.LeafState.create(path, lms.public_key(k), k.params.H)
    with lms.LeafState(path, lms.public_key(k), k.params.H) as state:
        yield lambda m: lms.sign(k, state, m).encode()


def entry(session: str, window: dict = WINDOW, key: C.ChallengerKey | None = None) -> dict:
    """The auditor's entry naming `session` for `window`, under `key` (the test key by default)."""
    return {**window, "code": "ab" * 20, "key": hashlib.sha512((key or verifier_key()).public).hexdigest(),
            "session": session}


def audit(tmp_path, *entries: dict):
    path = tmp_path / "auditor.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


class Host:
    """The challenger on its host: a challenger over `FAMILIES` whose record `open_session` signed with `sign` and wrote to
    `directory` for session `session`, serving one session on a loopback port; `reason` is how `serve` ended it."""

    def __init__(self, tmp_path, sign, session: str = "s-1", seed: str = "host") -> None:
        self.directory = tmp_path / session
        self.directory.mkdir()
        self.challenger = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy(seed))
        self.session = R.open_session(self.challenger, verifier_key(), HOST, sign, self.directory, session)
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
                self.reason = R.serve(conn, self.session, itertools.count().__next__)
            except Exception as e:
                self.error = e

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason

    def trusted(self, tmp_path, auditor=None, window: dict = WINDOW, key: C.ChallengerKey | None = None):
        auditor = audit(tmp_path, entry(self.session.id)) if auditor is None else auditor
        return R.trusted_session(self.directory, auditor, window, key or verifier_key(), lms_verify)


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
    commitment, roots = host.trusted(tmp_path)
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
    session = R.open_session(ch, verifier_key(), HOST, signer, tmp_path, "s-1")
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
        R.open_session(ch, verifier_key(), HOST, lambda m: b"", tmp_path / "other", "s-2")
    for bad in ("", "a b", "x" * (C.MAX_SESSION_ID + 1)):
        with pytest.raises(C.CoinError, match="session id"):
            R.open_session(C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("h")), verifier_key(), HOST, signer, tmp_path,
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
    """A refused session is signed, with why it ended, and is no window's: the verifier refuses it."""
    host = Host(tmp_path, signer)
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
        host.trusted(tmp_path)


def test_a_session_its_client_never_closed_is_signed_as_such_and_refused(tmp_path, signer):
    host = Host(tmp_path, signer)
    with socket.create_connection(host.listener.getsockname()) as s:
        s.sendall(frame({"type": "hello"}))
        assert reply(s)["type"] == "commitment"
        s.sendall(frame(OPEN))
        assert reply(s)["type"] == "opening"
    assert host.ended() == "the stream closed"
    with pytest.raises(C.CoinError, match="ended without its client closing it: the stream closed"):
        host.trusted(tmp_path)


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


def test_a_root_the_prover_rewrites_is_refused_against_the_challengers_log(tmp_path, signer):
    host = Host(tmp_path, signer)
    with R.RemoteChallenger.connect(host.address) as client:
        record = C.LiveRecord(client, "positions", itertools.count().__next__)
        record.record(root(0))
    assert host.ended() is None
    commitment, roots = host.trusted(tmp_path)
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
    host.trusted(tmp_path)
    edit, why = TAMPERED[case]
    log = host.directory / R.RELEASES
    if edit is None:
        log.unlink()
    else:
        _edit(log, edit)
    with pytest.raises(C.CoinError, match=why):
        host.trusted(tmp_path)


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
    host = _one_coin(tmp_path, signer)
    edit, why = CLOSES[case]
    edit(host)
    with pytest.raises(C.CoinError, match=why):
        host.trusted(tmp_path)


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
    with pytest.raises(C.CoinError, match="does not verify"):
        host.trusted(tmp_path, audit(tmp_path, entry("s-1", key=other)), key=other)
    with pytest.raises(C.CoinError, match="not the window's pinned key"):
        host.trusted(tmp_path, key=other)
    (host.directory / R.RECORD).unlink()
    with pytest.raises(C.CoinError, match="no challenger record"):
        host.trusted(tmp_path)


def test_two_sessions_under_one_key_cant_both_pass_for_one_window(tmp_path, signer):
    """Two honest sessions under the one key, each signed and closed: the window takes only the session the auditor's
    record names for it, and a record naming both, or naming one for two windows, is refused."""
    first = Host(tmp_path, signer, "s-1", "one")
    honest(first)
    second = Host(tmp_path, signer, "s-2", "two")
    honest(second)
    auditor = audit(tmp_path, entry("s-1"))
    assert first.trusted(tmp_path, auditor)[0] == first.challenger.commitment
    with pytest.raises(C.CoinError, match="the challenger's session is s-2, not s-1, the one the auditor's record names"):
        second.trusted(tmp_path, auditor)
    with pytest.raises(C.CoinError, match="names 2 challenger sessions for this window, and a window has one"):
        first.trusted(tmp_path, audit(tmp_path, entry("s-1"), entry("s-2")))
    later = {**WINDOW, "period": 1}
    both = audit(tmp_path, entry("s-1"), entry("s-1", later))
    for window in (WINDOW, later):
        with pytest.raises(C.CoinError, match="names challenger session s-1 for more than one window"):
            first.trusted(tmp_path, both, window)
    assert second.trusted(tmp_path, audit(tmp_path, entry("s-1"), entry("s-2", later)), later)[0] == \
        second.challenger.commitment


AUDITS = {
    "no record": (None, "^no auditor's session record: .*none at"),
    "a record naming another window": ([entry("s-1", {**WINDOW, "stream": "44" * 64})], "names no challenger session"),
    "an entry without its key": ([{k: v for k, v in entry("s-1").items() if k != "key"}], "record is malformed"),
    "an entry with a short commit": ([{**entry("s-1"), "commit": "11"}], "record is malformed"),
    "an entry with an uppercase key": ([{**entry("s-1"), "key": entry("s-1")["key"].upper()}], "record is malformed"),
}


@pytest.mark.parametrize("case", sorted(AUDITS))
def test_the_verifier_refuses_a_window_the_auditors_record_does_not_name_one_session_for(tmp_path, signer, case):
    host = Host(tmp_path, signer)
    honest(host)
    entries, why = AUDITS[case]
    path = tmp_path / "auditor.jsonl" if entries is None else audit(tmp_path, *entries)
    with pytest.raises(C.CoinError, match=why):
        host.trusted(tmp_path, path)
    with pytest.raises(C.CoinError, match="window's identity is malformed"):
        host.trusted(tmp_path, window={**WINDOW, "period": -1})


@pytest.mark.parametrize("address", ["7791", ":7791", "host:port", "host:"])
def test_an_address_is_host_and_port(address):
    with pytest.raises(ValueError, match="HOST:PORT"):
        R.RemoteChallenger.connect(address)
