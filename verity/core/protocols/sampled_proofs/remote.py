"""The proof service's challenger as a service on its own host, and its client: `LiveChallenger` over one TCP stream.

The challenger is auditor-trusted (`trusted-host`): it runs on a machine the prover doesn't control, and the
prover's process reaches it over the stream at its own 127.0.0.1:PORT.  For a session of the auditor host,
vy-challenger-1, that port is one of the prover machine's certifier coin link (`benchmarks/live_challenger/auditor.py`,
the coins' path; the premises `CoinsConfidential` and `OneClock` are this package's README's); in development
(`benchmarks/live_challenger/run.sh`) it is `ssh -R PORT` to the prover's machine, as `benchmarks/remote_session` places
its verifier.  On its host, before the stream opens, `open_session` signs the challenger's record and writes it, the key
and the head of the release log to the challenger's directory (`record.json`, `key.json`, `releases.jsonl`), for the
session id the auditor launched it under; `serve` refuses a session whose record isn't written there, so no coin opens
before it.  When the session ends, however it ends, the challenger signs its release log and writes the close record,
`challenger.json` (`close_session`; `challenger`'s "The session's close").

~~~text
frame       u32be len(head) ‖ u32be len(body) ‖ head ‖ body: head a JSON object with its "type", body empty
client      {"type": "hello"} (with "secret" when the session has one), then {"type": "open", "label", "index", "after"}
            per coin, then {"type": "close"}
challenger  {"type": "commitment", "commitment"}, {"type": "opening", "opening"} and {"type": "closed"}, each answering
            the client's frame, or {"type": "refused", "reason"}, which ends the session
~~~

Every message is fail-closed.  The challenger refuses a frame that isn't the next of that sequence in exactly that form,
a coin other than its family's next (coin i after the family's i-th root, once: `LiveRecord`) and a root that isn't 64
bytes; a refusal or a broken stream ends the session, and no coin opens twice (a retry never redraws).  It appends each
release to `releases.jsonl` (`LiveRecord.log`'s two events, a line each in canonical JSON, integer nanoseconds on its
monotonic clock), fsynced, before the opening leaves its host, and answers the client's close only once its signed close
is written.  A session served with a `secret` (the auditor host's gate gives one to the window's opener, RT4-INT-29)
refuses a hello that doesn't carry it, so no other process that reaches the port takes the session's coins first.
`greet` reads the hello alone, so a server can refuse a bad one and take the next connection: a refused hello releases
nothing.

`RemoteChallenger` is the client's `LiveChallenger`, its `commitment`, `open(label, index, after)` and `open_many`, so
`LiveRecord` (and a client's seam over it) runs on it unchanged.  `open_many` asks for several coins in one round trip:
their frames back to back, then the replies in order, the wire and the release log as the same opens one at a time.  It
raises `CoinError` on a refusal, a broken stream, or an opening that isn't the coin asked for opened against the
commitment after `after` (`live_reason`), and refuses every coin after.  A client whose session the gate has just opened
connects with `connect_when_listening`: until the session's tunnel listens, the connection is refused before any hello,
so it is tried again, up to `TUNNEL_WAIT_S`, then refused; nothing else is retried.

The verifier reads the challenger's directory from the auditor's side (the challenger's own run), never from the prover's
bundle (`trusted_session`), and the auditor's signed ledger of which session is a window's, every head the auditor
host's operator published
(`ledger_lines`, `challenger.ledger_head`): the commitment the challenger signed and the roots its signed release log
shows it received (`challenger.session_named`), its record, close and refused hellos the files the ledger's end sealed,
against which a family's openings are `order_reason`'s.  An
opening's `after` is the prover's to write, so only the challenger's log shows which root a coin followed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.challenger import (ChallengerClose, ChallengerKey, CoinError, LiveChallenger,
                                                             LiveCommitment, LiveOpening, LiveRecord, Verify, live_reason,
                                                             write_record)
from verity.ml.commitments.identity import canonical_json_text

RECORD, KEY, RELEASES, CLOSE, HELLOS = Ch.RECORD_FILE, "key.json", Ch.LOG_FILE, Ch.CLOSE_FILE, Ch.HELLOS_FILE
ROOT_BYTES = 64
HEADER = struct.Struct(">II")
MAX_HEAD = 1 << 20
MAX_BATCH = 64
"""The most coins `RemoteChallenger.open_many` asks for at once: its frames (about 200 bytes each) are written before
any reply is read, so they must fit the stream's send buffer while the challenger writes its replies back."""
SEQUENCE = "a client's frames are hello, then open {label, index, after} per coin, then close"
SESSION_ID = f"a session id is at most {Ch.MAX_SESSION_ID} ASCII letters, digits, '-', '.' and '_'"


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({k for k, _ in pairs}) != len(pairs):
        raise CoinError("a frame's head repeats a key")
    return dict(pairs)


class _Wire:
    def __init__(self, sock: socket.socket) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock

    def send(self, *heads: dict) -> None:
        """Each of `heads` as a frame, back to back in one write."""
        out = bytearray()
        for head in heads:
            h = json.dumps(head, sort_keys=True, separators=(",", ":")).encode()
            out += HEADER.pack(len(h), 0) + h
        self.sock.sendall(out)

    def _exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.sock.recv(min(n - len(out), 1 << 16))
            if not chunk:
                raise CoinError("the stream closed")
            out += chunk
        return bytes(out)

    def recv(self) -> dict:
        nh, nb = HEADER.unpack(self._exact(HEADER.size))
        if nh > MAX_HEAD or nb:
            raise CoinError(f"a frame is a head of at most {MAX_HEAD} bytes and no body, not {nh} + {nb}")
        try:
            head = json.loads(self._exact(nh), object_pairs_hook=_unique)
        except CoinError:
            raise
        except ValueError as e:
            raise CoinError(f"a frame's head is not JSON ({e})") from e
        if type(head) is not dict or type(head.get("type")) is not str:
            raise CoinError("a frame's head is an object with a type")
        return head


def _line(doc: dict) -> str:
    return canonical_json_text(doc)


def _write_new(path: Path, doc: dict) -> None:
    with open(path, "x", encoding="utf-8") as f:
        f.write(_line(doc) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _root(s: Any) -> bytes:
    if type(s) is not str or len(s) != 2 * ROOT_BYTES or s != s.lower():
        raise CoinError(f"a root is {ROOT_BYTES} bytes in lowercase hex")
    try:
        return bytes.fromhex(s)
    except ValueError:
        raise CoinError(f"a root is {ROOT_BYTES} bytes in lowercase hex") from None


class Session:
    """The challenger's one session on its host (`open_session`): its challenger, the key and host its record names, the
    signer of its key, its directory, the session id the auditor launched it under, and the release log's lines as it
    wrote them, which its close signs."""

    def __init__(self, challenger: LiveChallenger, key: ChallengerKey, host: str, sign: Callable[[bytes], bytes],
                 directory: Path, session: str, head: str) -> None:
        self.challenger, self.key, self.host, self._sign = challenger, key, host, sign
        self.directory, self.id, self.lines = directory, session, [head]
        self.close: ChallengerClose | None = None


def open_session(challenger: LiveChallenger, key: ChallengerKey, host: str, sign: Callable[[bytes], bytes],
                 directory: str | os.PathLike, session: str) -> Session:
    """On the challenger's host `host`, before any coin opens, for session `session` (the id the auditor launched it
    under): sign its record with `sign` (LMS at `key`'s parameters, `LiveChallenger.sign_record`, which signs its close
    too, at its next leaf) and write the record, `key` and the release log's head to `directory`, each once."""
    if not Ch._session_id(session):
        raise CoinError(SESSION_ID)
    d = Path(directory)
    record = challenger.sign_record(host, key.public, sign)
    _write_new(d / KEY, key.to_json())
    write_record(d / RECORD, record)
    head = {"event": "commitment", "root": challenger.commitment.root.hex()}
    _write_new(d / RELEASES, head)
    return Session(challenger, key, host, sign, d, session, _line(head))


def close_session(session: Session, ended: str | None) -> ChallengerClose:
    """The session's close, once: sign the release log's lines as the challenger wrote them and how the session `ended`
    (None: its client closed it) at the key's next leaf, and write it to the directory (`CLOSE`)."""
    if session.close is not None:
        raise CoinError("a challenger's session is closed once")
    lines, chain = len(session.lines), Ch.log_chain(session.lines)
    message = Ch.close_message(session.key.public[:16], session.host, session.challenger.commitment, session.id, lines,
                               chain, ended)
    close = ChallengerClose(session.id, lines, chain, ended, bytes(session._sign(message)))
    _write_new(session.directory / CLOSE, close.to_json())
    session.close = close
    return close


def _release(head: dict, records: dict[str, LiveRecord], log: Any, lines: list[str]) -> LiveOpening:
    if head.get("type") != "open" or sorted(head) != ["after", "index", "label", "type"]:
        raise CoinError(SEQUENCE)
    label, index, after = head["label"], head["index"], _root(head["after"])
    record = records.get(label) if type(label) is str else None
    if record is None:
        raise CoinError(f"the epoch's coins have no {label!r} family")
    if type(index) is not int or index != len(record.roots):
        raise CoinError(f"the next {label} coin is {len(record.roots)}, not {index!r}")
    n = len(record.log)
    opening = record.record(after)
    try:
        new = [_line(e) for e in record.log[n:]]
    except TypeError as e:
        raise CoinError(f"the release log's clock reads other than integers, so {label} coin {index} stays on the host "
                        f"({e})") from e
    try:
        log.write("".join(line + "\n" for line in new))
        log.flush()
        os.fsync(log.fileno())
    except OSError as e:
        raise CoinError(f"the release log is unwritable, so {label} coin {index} stays on the host ({e})") from e
    lines.extend(new)
    return opening


def _hang_up(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_WR)
        while sock.recv(1 << 16):
            pass
    except OSError:
        pass


def _closed(session: Session, ended: str | None) -> str | None:
    """`close_session`, and how the session ended: `ended`, or why its close went unsigned."""
    try:
        close_session(session, ended)
    except Exception as e:  # noqa: BLE001 - an unsigned close is a session no verifier takes, so it is reported, not raised
        return f"{ended or 'the client closed the session'}, and the challenger's close went unwritten ({e})"
    return ended


def _hello_reason(head: dict, secret: bytes | None) -> str | None:
    if secret is None:
        return None if head == {"type": "hello"} else SEQUENCE
    if head.get("type") != "hello" or sorted(head) != ["secret", "type"]:
        return "a client's first frame is hello with the session's secret"
    given = head["secret"]
    if type(given) is not str or not hmac.compare_digest(given.encode(), secret.hex().encode()):
        return "the hello's secret is not the session's"
    return None


def greet(sock: socket.socket, secret: bytes | None = None) -> str | None:
    """Read the client's hello on `sock`: None when it is the session's (`serve(..., greeted=True)` then serves the
    stream), else why it is refused, which is sent before the stream is hung up.  It releases nothing, so a server may
    take another connection after a refused hello without redrawing anything."""
    try:
        why = _hello_reason(_Wire(sock).recv(), secret)
    except CoinError as e:
        why = str(e)
    except OSError as e:
        return f"the stream broke ({e})"
    if why is not None:
        try:
            _Wire(sock).send({"type": "refused", "reason": why})
        except OSError:
            pass
        _hang_up(sock)
    return why


def serve(sock: socket.socket, session: Session, clock: Callable[[], int] = time.monotonic_ns,
          secret: bytes | None = None, greeted: bool = False) -> str | None:
    """Serve one client's session on the connected stream `sock` (`open_session`), on the release log's clock `clock`
    (integers, the log being canonical JSON): None when the client closed it, else why it ended.  With `secret`, the
    hello must carry it (lowercase hex); `greeted` when `greet` has already accepted it on `sock`.  However it ends, the
    session's close is signed and written (`close_session`), before the client's close is answered.  After its last frame
    it ends its side of the stream and returns only once the client ends its own, so a tunnel torn down after `serve`
    returns never cuts that frame."""
    d, challenger = session.directory, session.challenger
    if (session.close is not None or challenger.record is None or not (d / RECORD).is_file()
            or not (d / RELEASES).is_file()):
        raise CoinError("the challenger serves a session once, after its record is signed and written (open_session)")
    wire = _Wire(sock)
    records = {label: LiveRecord(challenger, label, clock) for label, _ in challenger.commitment.families}
    try:
        with open(d / RELEASES, "a", encoding="utf-8") as log:
            why = None if greeted else _hello_reason(wire.recv(), secret)
            if why is not None:
                raise CoinError(why)
            wire.send({"type": "commitment", "commitment": challenger.commitment.to_json()})
            while True:
                head = wire.recv()
                if head == {"type": "close"}:
                    reason = _closed(session, None)
                    if reason is not None:
                        break
                    wire.send({"type": "closed"})
                    _hang_up(sock)
                    return None
                wire.send({"type": "opening", "opening": _release(head, records, log, session.lines).to_json()})
    except CoinError as e:
        reason = _closed(session, str(e))
    except OSError as e:
        return _closed(session, f"the stream broke ({e})")
    try:
        wire.send({"type": "refused", "reason": reason})
    except OSError:
        pass
    _hang_up(sock)
    return reason


class RemoteChallenger:
    """The client of a challenger on its own host over the connected stream `sock`: `LiveChallenger`'s `commitment` (the
    challenger's answer to hello) and `open`.  After a refusal, a broken stream or a bad opening it refuses every coin."""

    def __init__(self, sock: socket.socket, secret: bytes | None = None) -> None:
        self._wire, self._ended = _Wire(sock), None
        hello = {"type": "hello"} if secret is None else {"type": "hello", "secret": bytes(secret).hex()}
        self.commitment = self._ask(hello, "commitment", LiveCommitment.from_json)

    @classmethod
    def connect(cls, address: str, secret: bytes | None = None) -> RemoteChallenger:
        """The challenger at ``HOST:PORT``, its session's `secret` in the hello when it has one (`connect_when_listening`
        waits for a session's tunnel to listen)."""
        host, sep, port = address.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError(f"a challenger's address is HOST:PORT, not {address!r}")
        return cls(socket.create_connection((host, int(port))), secret)

    def _end(self, why: str) -> CoinError:
        self._ended = why
        self._wire.sock.close()
        return CoinError(why)

    def _send(self, *heads: dict) -> None:
        if self._ended is not None:
            raise CoinError(f"the challenger's session ended ({self._ended}): a retry never redraws")
        try:
            self._wire.send(*heads)
        except OSError as e:
            raise self._end(f"the stream broke ({e})") from e

    def _reply(self, kind: str, read: Callable[[Any], Any]) -> Any:
        try:
            reply = self._wire.recv()
        except CoinError as e:
            raise self._end(str(e)) from e
        except OSError as e:
            raise self._end(f"the stream broke ({e})") from e
        if reply.get("type") == "refused" and sorted(reply) == ["reason", "type"] and type(reply["reason"]) is str:
            raise self._end(f"the challenger refused: {reply['reason']}")
        if reply.get("type") != kind or sorted(reply) != (["type"] if kind == "closed" else sorted(["type", kind])):
            raise self._end(f"the challenger did not answer with {kind!r}")
        if kind == "closed":
            return None
        try:
            return read(reply[kind])
        except CoinError as e:
            raise self._end(f"the challenger's {kind} is malformed: {e}") from e

    def _ask(self, head: dict, kind: str, read: Callable[[Any], Any]) -> Any:
        self._send(head)
        return self._reply(kind, read)

    def open(self, label: str, index: int, after: bytes) -> LiveOpening:
        return self.open_many([(label, index, after)])[0]

    def open_many(self, requests: Sequence[tuple[str, int, bytes]]) -> list[LiveOpening]:
        """`open` of each (label, index, after) of `requests` in one round trip: every open frame written back to back,
        then the replies read in order, each checked as `open` checks its own (`live_reason` against its request).  A
        refusal, a reply of another form, or a coin that isn't its request's, opened after its `after`, ends the session
        and raises: none of the batch's openings is returned, and a retry never redraws.  The challenger answers frames
        in order (`serve`), so its release log is the one the same coins opened one at a time would have written.  At
        most `MAX_BATCH` coins: their frames are written before any reply is read."""
        asked = [(label, index, bytes(after)) for label, index, after in requests]
        if len(asked) > MAX_BATCH:
            raise ValueError(f"a batch is at most {MAX_BATCH} coins, not {len(asked)}")
        self._send(*({"type": "open", "label": label, "index": index, "after": after.hex()}
                     for label, index, after in asked))
        openings = []
        for label, index, after in asked:
            opening = self._reply("opening", LiveOpening.from_json)
            try:
                why = live_reason(self.commitment, opening, label, index, after)
            except CoinError as e:
                why = str(e)
            if why is not None:
                raise self._end(f"the challenger's {label} coin {index}: {why}")
            openings.append(opening)
        return openings

    def close(self) -> None:
        """End the session: the challenger closes it, and the stream closes."""
        self._ask({"type": "close"}, "closed", lambda _: None)
        self._end("the client closed the session")

    def __enter__(self) -> RemoteChallenger:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._ended is None:
            self.close()


#: how long a client waits for an opened session's address to listen: the gate's open answers once the session's challenger
#: has signed its commitment, before the peer's certifier coin link carries its 127.0.0.1:PORT (live_challenger/auditor.py's
#: coins' path)
TUNNEL_WAIT_S = 120


def connect_when_listening(address: str, secret: bytes | None, connect: Callable[..., Any] | None = None,
                           wait_s: float = TUNNEL_WAIT_S, clock: Callable[[], float] = time.monotonic,
                           sleep: Callable[[float], None] = time.sleep) -> Any:
    """The session's client at `address` once something listens there. A refused connection sends no hello and opens no
    coin, so it is tried again every half second for up to `wait_s`, then refused (`CoinError`); any other failure raises
    at once.  `connect` is `RemoteChallenger.connect` by default, read at the call."""
    connect = connect or RemoteChallenger.connect
    deadline = clock() + wait_s
    while True:
        try:
            return connect(address, secret)
        except ConnectionRefusedError as e:
            if clock() >= deadline:
                raise CoinError(f"nothing listened at {address} within {wait_s} s of the session's open (its tunnel never "
                                "came up); no hello was sent and no coin opened") from e
            sleep(0.5)


def _read(path: Path, what: str) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise CoinError(f"the {what} at {path} is unreadable ({e})") from e


def _doc(path: Path, what: str) -> Any:
    text = _read(path, what)
    try:
        return None if text is None else json.loads(text)
    except ValueError as e:
        raise CoinError(f"the {what} at {path} is unreadable ({e})") from e


def log_lines(path: str | os.PathLike) -> list[str]:
    """The release log's lines at `path`, each without its newline; a missing or unreadable log raises `CoinError`."""
    text = _read(Path(path), "challenger's release log")
    if text is None:
        raise CoinError(f"no challenger release log at {path}")
    return (text[:-1] if text.endswith("\n") else text).split("\n")


def released_roots(path: str | os.PathLike, commitment: LiveCommitment) -> dict[str, tuple[bytes, ...]]:
    """The roots the release log at `path` shows the challenger received, by family in order (`released_roots_of`),
    unsigned: a verifier reads them only through `trusted_session`."""
    return Ch.released_roots_of(commitment, log_lines(path))


def hello_lines(path: str | os.PathLike) -> list[str] | None:
    """The refused hellos' lines at `path`, each without its newline (None when there is no such file)."""
    text = _read(Path(path), "challenger's refused hellos")
    return None if text is None else (text[:-1] if text.endswith("\n") else text).split("\n") if text else []


def session_files(directory: str | os.PathLike) -> dict[str, Any]:
    """The challenger's session as its directory holds it, ``{"close", "hellos", "log", "record"}``
    (`trusted_session_of`'s `session`): its close record, refused hellos' lines and record (None where missing) and its
    release log's lines."""
    d = Path(directory)
    return {"close": _doc(d / CLOSE, "challenger's close record"), "hellos": hello_lines(d / HELLOS),
            "log": log_lines(d / RELEASES), "record": _doc(d / RECORD, "challenger record")}


def ledger_lines(path: str | os.PathLike) -> list[str]:
    """One published head of the auditor's ledger at `path` (its operator's ``windows.jsonl``), its lines without their
    newlines; a missing or unreadable head raises `CoinError`."""
    text = _read(Path(path), "auditor's ledger")
    if text is None:
        raise CoinError(f"{Ch.NO_AUDIT} (none at {path})")
    return [line for line in text.split("\n") if line]


def main_lines(path: str | os.PathLike) -> list[str]:
    """The verifier's copy of main's first-parent history at `path`, one commit a line, newest first, without their
    newlines; a missing or unreadable file raises `CoinError`.  The auditor host's operator writes the file at each
    re-pin from
    ``git rev-list --first-parent origin/main`` (vy-auditor-1 reads it from /opt/vy-verifier/main-first-parent.txt), so
    the line is main's first-parent commits.  A deployment that wants `pouw.credit`'s freshness floor measured against the
    checked train heads instead (the commits whose own ``check`` passed and landed them, less a reverted train's) writes
    ``research queue checked-heads --out`` to the same file (`merge.main_line`)."""
    text = _read(Path(path), "copy of main's history")
    if text is None:
        raise CoinError(f"{Ch.NO_MAIN} (none at {path})")
    return [line for line in text.split("\n") if line]


def trusted_session(directory: str | os.PathLike, heads: Sequence[str | os.PathLike], window: Any, key: ChallengerKey,
                    verify: Verify) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the verifier takes from the challenger's directory for window `window`, read on the auditor's side: the
    auditor's ledger as the auditor host's operator published it, every head at `heads` checked and the longest taken
    (`challenger.ledger_head`);
    the one session it opens for the window, under the challenger key the window's deployment pins, `key`
    (`challenger.audited_session`); its record, close and refused hellos the files the ledger's end sealed; and from that
    session's directory the coin commitment the challenger signed and the roots its signed release log shows it received
    (`challenger.session_named`).  Each fails closed, raising `CoinError`.  It checks no pin: a path that credits the
    session also reads it through `challenger.pinned_session`."""
    d = Path(directory)
    ledger = Ch.ledger_head([ledger_lines(p) for p in heads], key, verify)
    audited = Ch.audited_session(ledger, window, key, verify)
    for name in (RECORD, CLOSE, HELLOS):
        try:
            digest = hashlib.sha512((d / name).read_bytes()).hexdigest()
        except OSError:
            digest = None
        if digest is None or audited.files.get(name) != digest:
            raise CoinError(f"the challenger's {name} is not the one the auditor's ledger sealed at its session's end")
    return Ch.session_named(audited, key, session_files(d), verify)


__all__ = ["CLOSE", "HEADER", "HELLOS", "KEY", "MAX_BATCH", "MAX_HEAD", "RECORD", "RELEASES", "ROOT_BYTES",
           "RemoteChallenger", "Session", "close_session", "greet", "hello_lines", "ledger_lines", "log_lines",
           "main_lines", "open_session", "released_roots", "serve", "session_files", "trusted_session"]
