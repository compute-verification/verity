"""The challenger as a service on its own host (`remote`), over loopback: it signs and writes its record before its first
coin opens, answers each root with that family's next coin and logs the release before the coin leaves; it refuses
every malformed or out-of-order frame and ends the session there, and however the session ends it signs its release log
(the close record).  The client (`RemoteChallenger`) is `LiveRecord`'s challenger as it is, and refuses a coin that
doesn't open.  The verifier, reading the challenger's directory and the auditor's signed ledger, trusts only the one
session the ledger opens for the window under the pinned key, ended closed, its commitment and the roots its signed log
shows, so a prover that rewrites a coin's root, an entry of the log or the ledger, or presents an unsigned log, an
unclosed session, a second session, a ledger signed under its own key or one older than the session's end is refused."""

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
from verity.ml.commitments.identity import canonical_json_text
from verity.ml.crypto import lms

from test_challenger import HOST, entropy, lms_key, lms_verify

SCHEME = "verity/test/coins/v1"
FAMILIES = [("positions", 3), ("replay", 2)]
WINDOW = {"commit": "11" * 64, "period": "p-0"}
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


class Ledger:
    """The auditor's ledger at `path` (infra's launcher's ``windows.jsonl``): each entry signed with `sign` under key
    `private`."""

    def __init__(self, path, sign, private: lms.PrivateKey = PINNED) -> None:
        self.path, self.sign, self.I = path, sign, lms.public_key(private)[:16]
        self.key_sha512 = hashlib.sha512(lms.public_key(private)).hexdigest()
        self.lines: list[str] = []

    def append(self, kind: str, session: str, body: dict) -> None:
        self.lines.append(C.ledger_line(self.lines, kind, session, body, self.I, self.sign))
        self.path.write_text("".join(line + "\n" for line in self.lines))

    def open(self, session: str, window: dict = WINDOW) -> None:
        self.append("open", session, {"window": window, "host": f"{HOST}.{session}", "pinned": "ab" * 20,
                                      "key_sha512": self.key_sha512, "port": 7811})

    def end(self, session: str, directory, closed: bool) -> None:
        files = {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
        self.append("end", session, {"closed": closed, "files": files, "rc": 0 if closed else 1})


class Host:
    """The challenger on its host: a challenger over `FAMILIES` whose session `session` the ledger opens for `window`
    (a new ledger in `tmp_path` unless `ledger`), whose record `open_session` signed with `sign` and wrote to
    `directory` naming the host the ledger opens it on (or `record_host`), serving one session on a loopback port;
    `reason` is how `serve` ended it, and the ledger's end says the session closed when its client closed it (or as
    `closed` says)."""

    def __init__(self, tmp_path, sign, session: str = "s-1", seed: str = "host", ledger: Ledger | None = None,
                 window: dict = WINDOW, closed: bool | None = None, record_host: str | None = None) -> None:
        self.directory = tmp_path / session
        self.directory.mkdir(parents=True)
        self.ledger, self.closed = ledger or Ledger(tmp_path / "windows.jsonl", sign), closed
        self.ledger.open(session, window)
        self.challenger = C.LiveChallenger(SCHEME, 0, FAMILIES, entropy(seed))
        self.session = R.open_session(self.challenger, pinned_key(), record_host or f"{HOST}.{session}", sign,
                                      self.directory, session)
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
            finally:
                closed = self.error is None and self.reason is None
                self.ledger.end(self.session.id, self.directory, closed if self.closed is None else self.closed)

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason

    def trusted(self, window: dict = WINDOW, key: C.ChallengerKey | None = None, ledger=None):
        """`remote.trusted_session` over this host's directory and the ledger's file (or `ledger`)."""
        return R.trusted_session(self.directory, ledger or self.ledger.path, window, key or pinned_key(), lms_verify)

    def trusted_of(self, window: dict = WINDOW):
        """`challenger.trusted_session_of` over this host's files as they are, past `remote`'s check of the sealed bytes."""
        return C.trusted_session_of(R.ledger_lines(self.ledger.path), window, pinned_key(), R.session_files(self.directory),
                                    lms_verify)


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
    with pytest.raises(C.CoinError, match="opens session s-1 under challenger key .*, not the window's pinned key"):
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
    leaves = itertools.count()
    sign = lambda m: lms.sign_at(forger, next(leaves), m).encode()
    own = Ledger(tmp_path / "own.jsonl", sign, forger)
    own.open("s-1")
    own.end("s-1", host.directory, True)
    with pytest.raises(C.CoinError, match="entry 0 opens session s-1 under challenger key .*, not the window's pinned"):
        host.trusted(ledger=own.path)
    with pytest.raises(C.CoinError, match="challenger record's signature does not verify"):
        host.trusted(key=pinned_key(forger), ledger=own.path)
    copied = Ledger(tmp_path / "copied.jsonl", sign, forger)
    copied.key_sha512 = host.ledger.key_sha512
    copied.open("s-1")
    copied.end("s-1", host.directory, True)
    with pytest.raises(C.CoinError, match="entry 0's signature does not verify under the verifier's challenger key"):
        host.trusted(ledger=copied.path)


def _ledger_edit(host: Host, edit) -> None:
    lines = R.ledger_lines(host.ledger.path)
    host.ledger.path.write_text("".join(line + "\n" for line in edit(lines)))


def _entry_set(i: int, change) -> object:
    def f(lines):
        doc = json.loads(lines[i])
        change(doc)
        return lines[:i] + [canonical_json_text(doc)] + lines[i + 1:]
    return f


LEDGERS = {
    "no ledger": (lambda h: h.ledger.path.unlink(), "^no auditor's ledger: .*none at"),
    "an empty ledger": (lambda h: _ledger_edit(h, lambda lines: []), "opens no challenger session for this window"),
    "a ledger truncated before the session's end": (lambda h: _ledger_edit(h, lambda lines: lines[:-1]),
                                                    "has no end for session s-1: .* older than its end"),
    "a ledger without its first entry": (lambda h: _ledger_edit(h, lambda lines: lines[1:]),
                                         "entry 0 is not entry 0 of a .* ledger chained to its predecessor"),
    "an entry re-chained to another predecessor": (lambda h: _ledger_edit(h, _entry_set(1, lambda d: d.update(
        prev="00" * 64))), "entry 1 is not entry 1 of a .* chained"),
    "an open moved to another host": (lambda h: _ledger_edit(h, _entry_set(0, lambda d: d["body"].update(
        host="elsewhere"))), "entry 0's signature does not verify"),
    "an end rewritten to unclosed": (lambda h: _ledger_edit(h, _entry_set(1, lambda d: d["body"].update(closed=False))),
                                   "entry 1's signature does not verify"),
    "an entry not in canonical form": (lambda h: _ledger_edit(h, lambda lines: [json.dumps(json.loads(lines[0]))]
                                                              + lines[1:]), "entry 0 is malformed"),
    "an entry of another kind": (lambda h: _ledger_edit(h, _entry_set(1, lambda d: d.update(kind="close"))),
                                 "entry 1 is malformed"),
    "an open whose Commit ends in a newline": (lambda h: _ledger_edit(h, _entry_set(0, lambda d: d["body"]["window"].update(
        commit=WINDOW["commit"] + "\n"))), "entry 0 is malformed"),
}


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
        host.trusted({**WINDOW, "period": "p-1"})
    with pytest.raises(C.CoinError, match="opens no challenger session for this window"):
        host.trusted({**WINDOW, "commit": "44" * 64})
    for bad in ({**WINDOW, "period": 1}, {**WINDOW, "period": ".p"}, {**WINDOW, "stream": "44" * 64},
                {**WINDOW, "commit": "11"}, {**WINDOW, "commit": WINDOW["commit"] + "\n"},
                {**WINDOW, "period": "p-0\n"}, {**WINDOW, "commit": WINDOW["commit"] + " "}):
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


def test_a_release_log_the_ledgers_end_did_not_seal_is_refused(tmp_path, signer):
    host = Host(tmp_path, signer)
    honest(host)
    resealed = Ledger(tmp_path / "resealed.jsonl", signer)
    resealed.open("s-1")
    (host.directory / R.RELEASES).write_text((host.directory / R.RELEASES).read_text() + "\n")
    resealed.end("s-1", host.directory, True)
    (host.directory / R.RELEASES).write_text((host.directory / R.RELEASES).read_text()[:-1])
    with pytest.raises(C.CoinError, match="release log is not the one the auditor's ledger sealed"):
        host.trusted(ledger=resealed.path)


def test_two_sessions_under_one_key_cant_both_pass_for_one_window(tmp_path, signer):
    """Two honest sessions under the one key, each signed and closed: the window takes only the session the auditor's
    ledger opens for it, and a ledger opening a second session for a window is refused."""
    first = Host(tmp_path, signer, "s-1", "one")
    honest(first)
    later = {"commit": "22" * 64, "period": "p-1"}
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
        with pytest.raises(C.CoinError, match="entry 4 opens a second session for its window's Commit"):
            host.trusted(window)


def test_one_commit_gets_one_session_whatever_period_label_its_windows_name(tmp_path, signer):
    """RT4-INT-24: two sessions for one Commit under two period labels, both closed, would give the Commit two draws of
    coins; the ledger that opens the second is refused for both windows."""
    first = Host(tmp_path, signer, "s-1", "one")
    honest(first)
    relabelled = {**WINDOW, "period": "p-0-again"}
    second = Host(tmp_path, signer, "s-2", "two", ledger=first.ledger, window=relabelled)
    honest(second)
    for host, window in ((first, WINDOW), (second, relabelled)):
        with pytest.raises(C.CoinError, match="entry 2 opens a second session for its window's Commit, or session s-2"):
            host.trusted(window)


@pytest.mark.parametrize("leaves", [[3, 3], [5, 2], [4, 6, 6, 7]])
def test_a_ledger_that_reuses_or_goes_back_on_a_leaf_is_refused(leaves):
    """RT4-INT-27: a host whose key state and ledger are restored signs its next entry at a leaf it has signed at
    already; the key signs at increasing leaves, so a ledger whose leaves don't increase was forked."""
    def ledger(qs):
        at = iter(qs)
        lines: list[str] = []
        for kind, session, body in [("open", "s-1", {"window": WINDOW, "host": f"{HOST}.s-1", "pinned": "ab" * 20,
                                                     "key_sha512": hashlib.sha512(lms.public_key(PINNED)).hexdigest()}),
                                    ("end", "s-1", {"closed": True, "files": {}}),
                                    ("open", "s-2", {"window": {**WINDOW, "commit": "22" * 64}, "host": f"{HOST}.s-2",
                                                     "pinned": "ab" * 20,
                                                     "key_sha512": hashlib.sha512(lms.public_key(PINNED)).hexdigest()}),
                                    ("end", "s-2", {"closed": True, "files": {}})][:len(qs)]:
            lines.append(C.ledger_line(lines, kind, session, body, lms.public_key(PINNED)[:16],
                                       lambda m: lms.sign_at(PINNED, next(at), m).encode()))
        return lines

    assert C.audited_session(ledger([2 * q + 1 for q in range(len(leaves))]), WINDOW, pinned_key(),
                             lms_verify).session == "s-1"
    i = next(i for i in range(1, len(leaves)) if leaves[i] <= leaves[i - 1])
    with pytest.raises(C.CoinError, match=f"entry {i} is signed at leaf {leaves[i]}, not after leaf {leaves[i - 1]}"):
        C.audited_session(ledger(leaves), WINDOW, pinned_key(), lms_verify)


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
    moved.open("s-2", {"commit": "22" * 64, "period": "p-1"})
    with pytest.raises(C.CoinError, match="entry 2 opens session s-2 under challenger key .*, not the window's pinned key"):
        host.trusted()
    with pytest.raises(C.CoinError, match="entry 0 opens session s-1 under challenger key .*, not the window's pinned key"):
        host.trusted(key=pinned_key(new))
    moved.lines = list(host.ledger.lines)
    moved.key_sha512 = host.ledger.key_sha512
    moved.open("s-2", {"commit": "22" * 64, "period": "p-1"})
    with pytest.raises(C.CoinError, match="entry 2's signature does not verify under the verifier's challenger key"):
        host.trusted()


def test_the_pin_is_read_from_infras_key_record():
    """`key_of_record` reads a ``challenger-key/v1`` record (the store artifact infra labels ``challenger-key current``)
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
