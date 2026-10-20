"""The proof service's challenger as a service on its own host, and its client: `LiveChallenger` over one TCP stream.

The challenger is auditor-trusted (`trusted-nebius-host`): it runs on a machine the prover doesn't control, and the
prover's process reaches it over the stream (`benchmarks/live_challenger/run.sh` places it as `benchmarks/remote_session`
places its verifier: `ssh -R PORT` to the prover's machine, whose own 127.0.0.1:PORT the prover connects to).  On its
host, before the stream opens, `open_session` signs the challenger's record and writes it, the key and the head of the
release log to the challenger's directory (`record.json`, `key.json`, `releases.jsonl`); `serve` refuses a challenger
whose record isn't written there, so no coin opens before it.

~~~text
frame       u32be len(head) ‖ u32be len(body) ‖ head ‖ body: head a JSON object with its "type", body empty
client      {"type": "hello"}, then {"type": "open", "label", "index", "after"} per coin, then {"type": "close"}
challenger  {"type": "commitment", "commitment"}, {"type": "opening", "opening"} and {"type": "closed"}, each answering
            the client's frame, or {"type": "refused", "reason"}, which ends the session
~~~

Every message is fail-closed.  The challenger refuses a frame that isn't the next of that sequence in exactly that form,
a coin other than its family's next (coin i after the family's i-th root, once: `LiveRecord`) and a root that isn't 64
bytes; a refusal or a broken stream ends the session, and no coin opens twice (a retry never redraws).  It appends each
release to `releases.jsonl` (`LiveRecord.log`'s two events), fsynced, before the opening leaves its host.

`RemoteChallenger` is the client's `LiveChallenger`, its `commitment` and `open(label, index, after)`, so `LiveRecord`
(and a client's seam over it) runs on it unchanged.  It raises `CoinError` on a refusal, a broken stream, or an opening
that isn't the coin asked for opened against the commitment after `after` (`live_reason`), and refuses every coin after.

The verifier reads the challenger's directory from the auditor's side (the challenger's own run), never from the prover's
bundle (`trusted_session`): the commitment the challenger signed (`challenger.trusted_commitment` on `record.json`, under
the verifier's challenger key) and the roots it received (`released_roots`), against which a family's openings are
`order_reason`'s.  An opening's `after` is the prover's to write, so only the challenger's log shows which root a coin
followed.
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

from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, ChallengerRecord, CoinError, LiveChallenger,
                                                             LiveCommitment, LiveOpening, LiveRecord, Verify, live_reason,
                                                             trusted_commitment, write_record)

RECORD, KEY, RELEASES = "record.json", "key.json", "releases.jsonl"
ROOT_BYTES = 64
HEADER = struct.Struct(">II")
MAX_HEAD = 1 << 20
SEQUENCE = "a client's frames are hello, then open {label, index, after} per coin, then close"


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


def _write_new(path: Path, doc: dict) -> None:
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(doc, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _root(s: Any) -> bytes:
    if type(s) is not str or len(s) != 2 * ROOT_BYTES or s != s.lower():
        raise CoinError(f"a root is {ROOT_BYTES} bytes in lowercase hex")
    try:
        return bytes.fromhex(s)
    except ValueError:
        raise CoinError(f"a root is {ROOT_BYTES} bytes in lowercase hex") from None


def open_session(challenger: LiveChallenger, key: ChallengerKey, host: str, sign: Callable[[bytes], bytes],
                 directory: str | os.PathLike) -> ChallengerRecord:
    """On the challenger's host `host`, before any coin opens: sign its record with `sign` (LMS at `key`'s parameters,
    `LiveChallenger.sign_record`) and write the record, `key` and the release log's head to `directory`, each once."""
    d = Path(directory)
    record = challenger.sign_record(host, key.public, sign)
    _write_new(d / KEY, key.to_json())
    write_record(d / RECORD, record)
    _write_new(d / RELEASES, {"event": "commitment", "root": challenger.commitment.root.hex()})
    return record


def _release(head: dict, records: dict[str, LiveRecord], log: Any) -> LiveOpening:
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
        log.write("".join(json.dumps(e, sort_keys=True) + "\n" for e in record.log[n:]))
        log.flush()
        os.fsync(log.fileno())
    except OSError as e:
        raise CoinError(f"the release log is unwritable, so {label} coin {index} stays on the host ({e})") from e
    return opening


def _hang_up(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_WR)
        while sock.recv(1 << 16):
            pass
    except OSError:
        pass


def serve(sock: socket.socket, challenger: LiveChallenger, directory: str | os.PathLike,
          clock: Callable[[], float] = time.monotonic) -> str | None:
    """Serve one client's session on the connected stream `sock`, the challenger's record and release log in `directory`
    (`open_session`), on the release log's clock `clock`: None when the client closed it, else why it ended.  After its
    last frame it ends its side of the stream and returns only once the client ends its own, so a tunnel torn down after
    `serve` returns never cuts that frame."""
    d = Path(directory)
    if challenger.record is None or not (d / RECORD).is_file() or not (d / RELEASES).is_file():
        raise CoinError("the challenger serves only once its record is signed and written (open_session)")
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
                    wire.send({"type": "closed"})
                    _hang_up(sock)
                    return None
                wire.send({"type": "opening", "opening": _release(head, records, log).to_json()})
    except CoinError as e:
        reason = str(e)
    except OSError as e:
        return f"the stream broke ({e})"
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


_ROOT_EVENT = ["event", "index", "label", "received_at", "root"]
_RELEASE_EVENT = ["after", "event", "index", "label", "released_at"]


def released_roots(path: str | os.PathLike, commitment: LiveCommitment) -> dict[str, tuple[bytes, ...]]:
    """The roots the challenger received, by family in order: its release log at `path`, whose head names
    `commitment`, each root followed by its coin's release.  A missing, unreadable or malformed log, or another
    commitment's, raises `CoinError`."""
    try:
        events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    except FileNotFoundError:
        raise CoinError(f"no challenger release log at {path}") from None
    except (OSError, ValueError) as e:
        raise CoinError(f"the challenger release log at {path} is unreadable ({e})") from e
    if not events or events[0] != {"event": "commitment", "root": commitment.root.hex()}:
        raise CoinError(f"the challenger release log at {path} is not coin commitment {commitment.root.hex()[:16]}'s")
    roots: dict[str, list[bytes]] = {label: [] for label, _ in commitment.families}
    pairs = events[1:]
    if len(pairs) % 2:
        raise CoinError(f"the challenger release log at {path} ends on a root without its release")
    for root, release in zip(pairs[::2], pairs[1::2]):
        if (not isinstance(root, dict) or not isinstance(release, dict) or sorted(root) != _ROOT_EVENT
                or sorted(release) != _RELEASE_EVENT or root["event"] != "root" or release["event"] != "release"
                or type(root["label"]) is not str or root["label"] not in roots or type(root["index"]) is not int
                or root["index"] != len(roots[root["label"]]) or type(release["index"]) is not int
                or (release["label"], release["index"]) != (root["label"], root["index"])
                or release["after"] != root["root"]):
            raise CoinError(f"the challenger release log at {path} is malformed")
        roots[root["label"]].append(_root(root["root"]))
    return {label: tuple(r) for label, r in roots.items()}


def trusted_session(directory: str | os.PathLike, key: ChallengerKey, verify: Verify
                    ) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the verifier takes from the challenger's directory, read on the auditor's side: the coin commitment the
    challenger signed under `key` (`trusted_commitment`) and the roots it received (`released_roots`).  Each fails
    closed, raising `CoinError`."""
    d = Path(directory)
    commitment = trusted_commitment(d / RECORD, key, verify)
    return commitment, released_roots(d / RELEASES, commitment)


__all__ = ["HEADER", "KEY", "MAX_HEAD", "RECORD", "RELEASES", "ROOT_BYTES", "RemoteChallenger", "open_session",
           "released_roots", "serve", "trusted_session"]
