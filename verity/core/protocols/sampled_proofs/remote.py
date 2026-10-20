"""The proof service's challenger as a service on its own host, and its client: `LiveChallenger` over one TCP stream.

The challenger is auditor-trusted (`trusted-nebius-host`): it runs on a machine the prover doesn't control, and the
prover's process reaches it over the stream (`benchmarks/live_challenger/run.sh` places it as `benchmarks/remote_session`
places its verifier: `ssh -R PORT` to the prover's machine, whose own 127.0.0.1:PORT the prover connects to).  On its
host, before the stream opens, `open_session` signs the challenger's record and writes it, the key and the head of the
release log to the challenger's directory (`record.json`, `key.json`, `releases.jsonl`), for the session id the auditor
launched it under; `serve` refuses a session whose record isn't written there, so no coin opens before it.  When the
session ends, however it ends, the challenger signs its release log and writes the close record, `challenger.json`
(`close_session`; `challenger`'s "The session's close").

~~~text
frame       u32be len(head) ‖ u32be len(body) ‖ head ‖ body: head a JSON object with its "type", body empty
client      {"type": "hello"}, then {"type": "open", "label", "index", "after"} per coin, then {"type": "close"}
challenger  {"type": "commitment", "commitment"}, {"type": "opening", "opening"} and {"type": "closed"}, each answering
            the client's frame, or {"type": "refused", "reason"}, which ends the session
~~~

Every message is fail-closed.  The challenger refuses a frame that isn't the next of that sequence in exactly that form,
a coin other than its family's next (coin i after the family's i-th root, once: `LiveRecord`) and a root that isn't 64
bytes; a refusal or a broken stream ends the session, and no coin opens twice (a retry never redraws).  It appends each
release to `releases.jsonl` (`LiveRecord.log`'s two events), fsynced, before the opening leaves its host, and answers
the client's close only once its signed close is written.

`RemoteChallenger` is the client's `LiveChallenger`, its `commitment` and `open(label, index, after)`, so `LiveRecord`
(and a client's seam over it) runs on it unchanged.  It raises `CoinError` on a refusal, a broken stream, or an opening
that isn't the coin asked for opened against the commitment after `after` (`live_reason`), and refuses every coin after.

The verifier reads the challenger's directory from the auditor's side (the challenger's own run), never from the prover's
bundle (`trusted_session`), and the auditor's record of which session is a window's (`auditor_record`): the commitment the
challenger signed and the roots its signed release log shows it received (`challenger.trusted_session_of`), against
which a family's openings are `order_reason`'s.  An opening's `after` is the prover's to write, so only the challenger's
log shows which root a coin followed.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.challenger import (ChallengerClose, ChallengerKey, CoinError, LiveChallenger,
                                                             LiveCommitment, LiveOpening, LiveRecord, Verify, live_reason,
                                                             write_record)

RECORD, KEY, RELEASES, CLOSE = "record.json", "key.json", "releases.jsonl", "challenger.json"
ROOT_BYTES = 64
HEADER = struct.Struct(">II")
MAX_HEAD = 1 << 20
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

    def send(self, head: dict) -> None:
        h = json.dumps(head, sort_keys=True, separators=(",", ":")).encode()
        self.sock.sendall(HEADER.pack(len(h), 0) + h)

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
    return json.dumps(doc, sort_keys=True)


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
    new = [_line(e) for e in record.log[n:]]
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


def serve(sock: socket.socket, session: Session, clock: Callable[[], float] = time.monotonic) -> str | None:
    """Serve one client's session on the connected stream `sock` (`open_session`), on the release log's clock `clock`:
    None when the client closed it, else why it ended.  However it ends, the session's close is signed and written
    (`close_session`), before the client's close is answered.  After its last frame it ends its side of the stream and
    returns only once the client ends its own, so a tunnel torn down after `serve` returns never cuts that frame."""
    d, challenger = session.directory, session.challenger
    if (session.close is not None or challenger.record is None or not (d / RECORD).is_file()
            or not (d / RELEASES).is_file()):
        raise CoinError("the challenger serves a session once, after its record is signed and written (open_session)")
    wire = _Wire(sock)
    records = {label: LiveRecord(challenger, label, clock) for label, _ in challenger.commitment.families}
    try:
        with open(d / RELEASES, "a", encoding="utf-8") as log:
            if wire.recv() != {"type": "hello"}:
                raise CoinError(SEQUENCE)
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

    def __init__(self, sock: socket.socket) -> None:
        self._wire, self._ended = _Wire(sock), None
        self.commitment = self._ask({"type": "hello"}, "commitment", LiveCommitment.from_json)

    @classmethod
    def connect(cls, address: str) -> RemoteChallenger:
        """The challenger at ``HOST:PORT``."""
        host, sep, port = address.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError(f"a challenger's address is HOST:PORT, not {address!r}")
        return cls(socket.create_connection((host, int(port))))

    def _end(self, why: str) -> CoinError:
        self._ended = why
        self._wire.sock.close()
        return CoinError(why)

    def _ask(self, head: dict, kind: str, read: Callable[[Any], Any]) -> Any:
        if self._ended is not None:
            raise CoinError(f"the challenger's session ended ({self._ended}): a retry never redraws")
        try:
            self._wire.send(head)
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

    def open(self, label: str, index: int, after: bytes) -> LiveOpening:
        after = bytes(after)
        opening = self._ask({"type": "open", "label": label, "index": index, "after": after.hex()}, "opening",
                            LiveOpening.from_json)
        try:
            why = live_reason(self.commitment, opening, label, index, after)
        except CoinError as e:
            why = str(e)
        if why is not None:
            raise self._end(f"the challenger's {label} coin {index}: {why}")
        return opening

    def close(self) -> None:
        """End the session: the challenger closes it, and the stream closes."""
        self._ask({"type": "close"}, "closed", lambda _: None)
        self._end("the client closed the session")

    def __enter__(self) -> RemoteChallenger:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._ended is None:
            self.close()


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


def session_files(directory: str | os.PathLike) -> dict[str, Any]:
    """The challenger's session as its directory holds it, ``{"close", "log", "record"}`` (`trusted_session_of`'s
    `session`): its close record and record (None where missing) and its release log's lines."""
    d = Path(directory)
    return {"close": _doc(d / CLOSE, "challenger's close record"), "log": log_lines(d / RELEASES),
            "record": _doc(d / RECORD, "challenger record")}


def auditor_record(path: str | os.PathLike) -> list[Any]:
    """The auditor's session record at `path`, one entry a line (`challenger.audited_session`, `AUDIT_FORM`), as infra's
    launcher appends it before each session's first coin.  A missing or unreadable record raises `CoinError`."""
    text = _read(Path(path), "auditor's session record")
    if text is None:
        raise CoinError(f"{Ch.NO_AUDIT} (none at {path})")
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError as e:
        raise CoinError(f"the auditor's session record at {path} is unreadable ({e})") from e


def trusted_session(directory: str | os.PathLike, auditor: str | os.PathLike, window: Any, key: ChallengerKey,
                    verify: Verify) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the verifier takes from the challenger's directory for window `window`, read on the auditor's side: the one
    session the auditor's record at `auditor` names for the window, under the challenger key the window's deployment
    pins, `key`, and from that session's directory the coin commitment the challenger signed and the roots its signed
    release log shows it received (`challenger.trusted_session_of`).  Each fails closed, raising `CoinError`."""
    return Ch.trusted_session_of(auditor_record(auditor), window, key, session_files(directory), verify)


__all__ = ["CLOSE", "HEADER", "KEY", "MAX_HEAD", "RECORD", "RELEASES", "ROOT_BYTES", "RemoteChallenger", "Session",
           "auditor_record", "close_session", "log_lines", "open_session", "released_roots", "serve", "session_files",
           "trusted_session"]
