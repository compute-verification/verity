"""``firewall/v0``: the ZK server's release side (live PROTOCOL.md §10.5). It sits between the workers' outer prover
(``flock-circuit prove --zk`` with its send check, ``FC_FIREWALL=1``, §10.4) and the challenger, beside the inner phase's
proxy (`verity_flock.rec_live`, §10.3).

    python -m verity_flock.firewall relay --state DIR --session ID --part NAME --to HOST:PORT [--listen 127.0.0.1:0] [--sessions N] [--proofs 2]
    python -m verity_flock.firewall record --state DIR

The release side of record is Lean's: ``flock-circuit prove --firewall-state DIR --session ID --part NAME`` runs each outer
session through ``core-firewall session`` (``Core.Firewall.Loop`` around the Lean session loop), which keeps the same record,
byte for byte, and the rules below (``live/src/firewall_record_vectors.json``, which both sides' tests run). This module is the inner side's (`rec_live`'s ``serve`` opens and ends recursive sessions
here) and the tests' relay.

Every process of one ZK server shares its state directory: ``record.jsonl``, its events in order (each with its ``seq``),
appended under ``record.lock``, and ``slot.lock``, the outer slot. Four rules hold through it:

* **A recursive session's statements go in their order, each once, and it ends at its first stop.** The inner phase opens
  a recursive session (`Firewall.open`: `rec_live`'s ``serve`` opens one for each session it serves) with its V*
  statements in their order (``parts``, public, from K: L0–L6 and the algebra at K = 4096). Its first stop, inner or
  outer, is its one failure (``stop``). An outer session is admitted only once its recursive session's inner session has
  finished, and only for its next statement not yet released: a request for any other (a later one, or a released one
  again) is a stop at that next statement, never opened, so a statement the worker skips ends the recursive session where
  it stands. After a stop, and after the last statement's release, a relay refuses each outer session of it before
  anything reaches the challenger (``refused``, not a failure: the challenger sees nothing of it).
* **This relay takes outer proofs one at a time**, stricter than AGENTS.md's independent sessions (each statement fixed
  first, one session each with its own live coins, every verdict counted). It has one slot, across streams and protocols.
  An outer session takes it at its first request and holds it until the session is released (the challenger's verdict
  accepts it) or fails. A second
  session that asks meanwhile waits (``wait``) and takes the slot after (``slot``), so each ``slot`` event is followed by its
  session's outcome before the next one. A holder that ended before its outcome (killed, say) is recorded as its session's
  stop by the next session to take the slot.
* **A stream stops at its failure budget.** A stream opened with ``budget`` b (the challenger's stream's ``budget``, P1)
  refuses every outer session and every open after its b-th failure: a recursive session's stop, or a refusal of a recursive
  session that had neither stopped nor released its last statement, on a stream that had not stopped.
* **Proofs leave at public times.** A recursive session opened with a ``release`` schedule (``start_ms``, and one window a
  statement, ``windows_ms``) releases statement j's proofs at ``start_ms + windows_ms[0] + ... + windows_ms[j]``: proofs
  ready early wait, and proofs not ready by then are a stop. Only the Lean ZK server keeps the times; this module's relay
  refuses a scheduled session rather than release it early.

An outer session stops when the worker cuts it, when the worker sends ``Finish`` before all of the session's proofs (the
send check refused: ``flock-circuit`` then sends ``Finish`` at once), when the challenger's verdict rejects it, or on any
fault of the relay's. Except for a rejecting verdict, which the challenger reached itself, the relay then cuts the
challenger's connection without sending ``Finish``. So every stop the ZK server makes looks the same to the challenger: a
session cut after some message, never why. A refused or stopped session's worker is answered outside the slot, so it
can't hold up another session's.

What the rules read of the record is kept beside it in ``index-py/`` (`_Index`: each recursive session's open, stop,
finish and releases, each stream's first open and failures, the last ``slot``, the counts), and a call reads only what
was appended to the record since the index last read it, by whichever process. The Lean ZK server keeps none and reads
the record whole. The index is rebuilt from the record's start when
the record was replaced, cut shorter or no longer holds the last bytes the index read, when the machine has booted since
the index was written, when a writer of it ended mid-write, or when a file of it won't read; an edit further back that
leaves the record no shorter isn't seen, nor is a file of the index edited by hand that still reads. Since the ZK server
only appends, the verdicts are the ones a read of the whole record gives.
A record the index can't answer is read whole at every call, as
before, and each rule's read is a scan of it (`_Scan`): one whose last line has no newline yet, or one holding a line
the ZK server doesn't write or a name over `KEY_MAX` bytes. A torn last line (a writer killed inside its append) fails
every call, so the ZK server serves nothing until the developer repairs the record: it fails closed, never by skipping an
event it can't read.
"""
from __future__ import annotations

import fcntl
import json
import locale
import os
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .rec_live import (REQ_FINISH, REQ_PING, REQ_PROOF, RESP_REFUSED, RESP_VERDICT, decode_resp, read_frame, refused,
                       resp_ok, resp_str, write_frame)

KIND = "firewall/v0"
INNER = "inner"
#: an outer session's proofs, one a rep (2 at K = 4096)
PROOFS = 2


class Refused(Exception):
    """The ZK server serves nothing of this outer session (its message is the refusal)."""


class Firewall:
    """The state every process of one ZK server shares (``state``, a directory)."""

    def __init__(self, state: Path):
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        self._log = self.state / "record.jsonl"

    @contextmanager
    def _locked(self, sessions: tuple[str, ...] = (), streams: tuple[str, ...] = ()) -> Iterator[_Index | _Scan]:
        """The record as the rules read it (`_read`), under ``record.lock``; then what changed of the index is written
        back."""
        with open(self.state / "record.lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                rec = self._read(sessions, streams)
                try:
                    yield rec
                finally:
                    if (ix := rec if isinstance(rec, _Index) else rec.index) is not None:
                        ix.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _events(self) -> list[dict]:
        if not self._log.exists():
            return []
        return [json.loads(line) for line in self._log.read_text().splitlines() if line]

    def _read(self, sessions: tuple[str, ...] = (), streams: tuple[str, ...] = ()) -> _Index | _Scan:
        """The index, given the lines appended since it last read the record; or the record read whole (`_Scan`, with the
        index when it read as far as it could), when the index can't answer it or a line won't read (the whole read then
        fails as it always has). ``sessions`` and ``streams`` are the ones the call reads and appends for (`_Index.load`)."""
        try:
            f = open(self._log, "rb")
        except FileNotFoundError:
            f = None
        if f is None:
            ix, data = _Index.of(self.state / _INDEX, None), b""
        else:
            with f:
                st = os.fstat(f.fileno())
                ix = _Index.of(self.state / _INDEX, ([st.st_dev, st.st_ino], st.st_size))
                f.seek(ix.offset - len(ix.last))
                data = f.read()
                if data.startswith(ix.last):
                    data = data[len(ix.last):]
                else:
                    ix = _Index(ix.dir, ix.boot, ix.file)
                    f.seek(0)
                    data = f.read()
        end = data.rfind(b"\n") + 1
        try:
            lines = [json.loads(line) for line in data[:end].decode(_ENCODING).splitlines() if line]
        except (ValueError, RecursionError):
            return _Scan(self._events())
        try:
            for e in lines:
                ix.add(e)
            ix.read(data[:end])
            ix.load(sessions, streams)
        except _BAD:
            ix.broken = True
            ix.flush()
            return _Scan(self._events())
        if not ix.regular or end < len(data):
            return _Scan(self._events(), ix)
        return ix

    def _append(self, rec: _Index | _Scan, ev: dict) -> dict:
        ev = {"seq": rec.count, **ev}
        line = (json.dumps(ev, sort_keys=True) + "\n").encode(_ENCODING)
        with open(self._log, "ab") as f:
            f.write(line)
            if isinstance(rec, _Index) and rec.file is None:
                st = os.fstat(f.fileno())
                rec.file = [st.st_dev, st.st_ino]
        if isinstance(rec, _Index):
            try:
                rec.add(ev)
            except BaseException:
                rec.broken = True
                raise
            rec.read(line)
        else:
            rec.add(ev)
        return ev

    def record(self) -> list[dict]:
        """Every event so far, in order."""
        with self._locked():
            return self._events()

    def open(self, stream: str, parts: list[str], budget: int | None = None, release: dict | None = None) -> str:
        """A new recursive session on ``stream``, opened by its inner session, with its V* statements in their order
        (``parts``, public: the ZK server's rule 1): its id. ``budget`` is the stream's failure budget (P1's ``budget``,
        b), the same at every open of the stream; once the stream has ``budget`` failures, `Refused`. ``release`` is the
        recursive session's release schedule, public and fixed here, before the session: statement ``j`` is released at the
        end of its window, ``start_ms + sum(windows_ms[:j + 1])`` (rule 3)."""
        _parts_ok(parts)
        if budget is not None and not (type(budget) is int and budget >= 1):
            raise ValueError(f"a failure budget is a positive integer, not {budget!r}")
        if release is not None:
            _schedule_ok(release, len(parts))
        with self._locked(streams=(stream,)) as rec:
            if rec.known(stream) and (b := rec.budget(stream)) != budget:
                raise ValueError(f"stream {stream}'s budget is {b}, not {budget}")
            if budget is not None and (n := rec.failures(stream)) >= budget:
                raise Refused(refused(f"stream {stream} stopped at its failure budget ({n} of {budget})"))
            sid = f"r{rec.opens:04d}"
            e = {"event": "open", "session": sid, "stream": stream, "parts": list(parts)}
            if budget is not None:
                e["budget"] = budget
            if release is not None:
                e["release"] = release
            self._append(rec, e)
            return sid

    def finish(self, session: str) -> None:
        """The recursive session's inner session finished: its outer sessions may be served."""
        with self._locked((session,)) as rec:
            self._append(rec, {"event": "finish", "session": session, "part": INNER})

    def stop(self, session: str, part: str, why: str) -> bool:
        """The recursive session stops at ``part``: its one failure, recorded once (False if it had already stopped)."""
        with self._locked((session,)) as rec:
            if rec.stopped(session):
                return False
            self._append(rec, {"event": "stop", "session": session, "part": part, "why": why})
            return True

    def release(self, session: str, part: str) -> None:
        """An outer proof of the recursive session was released (the challenger's verdict accepted it)."""
        with self._locked((session,)) as rec:
            self._append(rec, {"event": "release", "session": session, "part": part})

    def admit(self, session: str, part: str, *, timed: bool = False) -> int | None:
        """Raise `Refused` unless an outer session of ``session`` may be served now: the recursive session is open, has not
        stopped, has a statement not yet released, its stream has not stopped at its budget, its inner session finished,
        and ``part`` is its next statement. A request for any other statement is the recursive session's stop at the next
        one (never opened), then refused. A refusal is recorded (``refused``). Returns ``part``'s release time (rule 3), or
        None. Only a ``timed`` caller (one that holds the proofs to that time: the Lean ZK server, or a test of its rules)
        is admitted to a scheduled session; this module's relay is not."""
        with self._locked((session,)) as rec:
            why, at = None, None
            open_ = rec.opened(session)
            if open_ is None:
                why = f"recursive session {session} was never opened"
            elif (s := rec.stopped(session)) is not None:
                why = f"recursive session {session} stopped at {s['part']}: the firewall serves nothing more of it"
            elif (j := rec.next(open_)) is None:
                why = f"recursive session {session} released all its statements: the firewall serves nothing more of it"
            elif (b := rec.budget(open_["stream"])) is not None and (n := rec.failures(open_["stream"])) >= b:
                why = f"stream {open_['stream']} stopped at its failure budget ({n} of {b})"
            elif not rec.finished(session):
                why = f"recursive session {session}: its inner session has not finished"
            elif (nxt := open_["parts"][j]) != part:
                self._append(rec, {"event": "stop", "session": session, "part": nxt, "why": f"never opened: {part} was asked in its place"})
                why = f"recursive session {session} stopped at {nxt}: {part} is not its next statement"
            elif (r := open_.get("release")) is not None:
                if not timed:
                    why = (f"recursive session {session}: its release schedule needs the Lean firewall "
                           "(core-firewall session: flock-circuit prove --firewall-state)")
                else:
                    at = r["start_ms"] + sum(r["windows_ms"][:j + 1])
            if why is not None:
                self._append(rec, {"event": "refused", "session": session, "part": part})
                raise Refused(refused(why))
            return at

    def stopped(self, session: str) -> dict | None:
        with self._locked((session,)) as rec:
            return rec.stopped(session)

    def failures(self, stream: str) -> int:
        """The failures on ``stream`` so far (rule 4): its recursive sessions' stops, and each refusal of one that had not
        stopped while the stream had not."""
        with self._locked(streams=(stream,)) as rec:
            return rec.failures(stream)

    @contextmanager
    def slot(self, session: str, part: str, waiting: Callable[[], None] | None = None) -> Iterator[None]:
        """The ZK server's one outer slot, held for the block. A session that finds it taken records ``wait``, calls
        ``waiting``, and blocks until the holder leaves it. Once it holds the slot, a ``slot`` event whose session has no
        outcome after it (its holder died) is recorded as that session's stop, and then ``slot`` is recorded."""
        with open(self.state / "slot.lock", "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                with self._locked((session,)) as rec:
                    self._append(rec, {"event": "wait", "session": session, "part": part})
                if waiting:
                    waiting()
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                with self._locked((session,)) as rec:
                    if (o := rec.orphan()) is not None and rec.stopped(o["session"]) is None:
                        self._append(rec, {"event": "stop", "session": o["session"], "part": o["part"],
                                           "why": f"the slot's holder ended before its outcome (slot {o['seq']})"})
                    self._append(rec, {"event": "slot", "session": session, "part": part})
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def _stopped(ev: list[dict], session: str) -> dict | None:
    return next((e for e in ev if e["event"] == "stop" and e["session"] == session), None)


def _budget(ev: list[dict], stream: str) -> int | None:
    return next((e.get("budget") for e in ev if e["event"] == "open" and e["stream"] == stream), None)


def _next(ev: list[dict], open_: dict) -> int | None:
    """The position of the recursive session's next statement, the first of its ``parts`` not released; None once all are."""
    released = {e["part"] for e in ev if e["event"] == "release" and e["session"] == open_["session"]}
    return next((j for j, p in enumerate(open_["parts"]) if p not in released), None)


def _failures(ev: list[dict], stream: str) -> int:
    budget, mine, released, stopped, n = _budget(ev, stream), {}, set(), set(), 0
    for e in ev:
        s = e["session"]
        if e["event"] == "open" and e["stream"] == stream:
            mine[s] = e["parts"]
        elif e["event"] == "release" and s in mine:
            released.add((s, e["part"]))
        elif e["event"] == "stop" and s in mine:
            stopped.add(s)
            n += 1
        elif (e["event"] == "refused" and s in mine and not all((s, p) in released for p in mine[s]) and s not in stopped
              and (budget is None or n < budget)):
            n += 1
    return n


def _orphan(ev: list[dict]) -> dict | None:
    """The last ``slot`` event, if no outcome of its session at its part (``release``, ``stop`` or ``refused``) follows it."""
    k = next((k for k in range(len(ev) - 1, -1, -1) if ev[k]["event"] == "slot"), None)
    if k is None:
        return None
    s = ev[k]
    done = any(e["session"] == s["session"] and (e["event"] in ("release", "refused") and e["part"] == s["part"] or e["event"] == "stop")
               for e in ev[k + 1:])
    return None if done else s


#: the encoding `Path.read_text` reads the record in
_ENCODING = locale.getpreferredencoding(False)
#: the index's directory in the state directory
_INDEX = "index-py"
#: the longest session or stream name the index keeps, in bytes (its files are named by it); a record holding a longer
#: one is read whole
KEY_MAX = 100
#: how many of the last bytes it read the index keeps, to tell that the record still holds them
_LAST = 256
_KINDS = ("finish", "stop", "release", "refused", "wait", "slot")
#: what a file of the index that won't read raises
_BAD = (OSError, ValueError, KeyError, TypeError, RecursionError)
#: the head's fields of one type (besides ``file``, ``last`` and ``slot``)
_HEAD = {"v": int, "boot": str, "dirty": bool, "offset": int, "regular": bool, "count": int, "opens": int, "slot_done": bool}


class _Scan:
    """The record read whole, each rule's read a scan of it; and the index, if it read as far as it could, to be written
    back."""

    def __init__(self, events: list[dict], index: _Index | None = None) -> None:
        self.events, self.index = events, index

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def opens(self) -> int:
        return sum(e["event"] == "open" for e in self.events)

    def add(self, e: dict) -> None:
        self.events.append(e)

    def opened(self, session: str) -> dict | None:
        return next((e for e in self.events if e["event"] == "open" and e["session"] == session), None)

    def known(self, stream: str) -> bool:
        return any(e["event"] == "open" and e["stream"] == stream for e in self.events)

    def stopped(self, session: str) -> dict | None:
        return _stopped(self.events, session)

    def finished(self, session: str) -> bool:
        return any(e["event"] == "finish" and e["session"] == session for e in self.events)

    def next(self, open_: dict) -> int | None:
        return _next(self.events, open_)

    def budget(self, stream: str) -> int | None:
        return _budget(self.events, stream)

    def failures(self, stream: str) -> int:
        return _failures(self.events, stream)

    def orphan(self) -> dict | None:
        return _orphan(self.events)


def _key(name: str) -> str | None:
    """The hex of ``name``'s bytes, which names its files in the index; None for a name over `KEY_MAX` bytes."""
    b = name.encode("utf-8", "surrogatepass")
    return b.hex() if len(b) <= KEY_MAX else None


def _regular(e: Any) -> bool:
    """Whether ``e`` is an event as the ZK server writes it, one whose reads `_Index` answers as `_Scan` does: an object with
    a ``seq`` and a string ``session``, and either an ``open`` with a string ``stream``, a list of string ``parts`` and an
    integer ``budget`` or none, or one of the other events with a string ``part``; every name at most `KEY_MAX` bytes."""
    if not (type(e) is dict and "seq" in e and type(e.get("session")) is str and _key(e["session"]) is not None):
        return False
    if e.get("event") == "open":
        return (type(e.get("stream")) is str and _key(e["stream"]) is not None and type(e.get("parts")) is list
                and all(type(p) is str for p in e["parts"]) and (e.get("budget") is None or type(e["budget"]) is int))
    return e.get("event") in _KINDS and type(e.get("part")) is str


def _strings(v: Any) -> bool:
    return type(v) is list and all(type(x) is str for x in v)


class _Session:
    """What the index keeps of a recursive session: its first open and first stop, whether its inner session finished, the
    statements released, and for each stream it was opened on (``member``), what that stream's failures read of it: its
    statements as its last open there gave them, and what was released and whether it stopped after its first open
    there."""

    def __init__(self, v: Any = None) -> None:
        if v is None:
            v = {"open": None, "stop": None, "finished": False, "released": [], "member": []}
        if not (type(v) is dict and set(v) == {"open", "stop", "finished", "released", "member"}
                and all(v[k] is None or type(v[k]) is dict for k in ("open", "stop")) and type(v["finished"]) is bool
                and _strings(v["released"]) and type(v["member"]) is list
                and all(type(m) is dict and set(m) == {"stream", "parts", "released", "stopped"} and type(m["stream"]) is str
                        and _strings(m["parts"]) and _strings(m["released"]) and type(m["stopped"]) is bool
                        for m in v["member"])):
            raise ValueError("the firewall's index: not a session")
        self.open, self.stop, self.finished = v["open"], v["stop"], v["finished"]
        self.released, self.member = v["released"], v["member"]

    def json(self) -> dict:
        return {"open": self.open, "stop": self.stop, "finished": self.finished, "released": self.released,
                "member": self.member}


def _boot() -> str:
    """The machine's boot: an index written before it may have lost writes the page cache held."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def _put(dir: Path, name: str, v: Any) -> None:
    tmp = dir / f".{name}"
    tmp.write_text(json.dumps(v))
    os.replace(tmp, dir / name)


class _Index:
    """What the rules read of the record, kept as it grows, an event at a time, in ``index-py/`` beside it: `_Scan`'s
    answers. Its files are ``head.json`` (where the index stands in the record: the record's device and inode
    (``file``), how much of it was read (``offset``) and its last bytes (``last``), whether every line so far is one the
    index answers for (``regular``), the counts, the last ``slot`` (its seq, session and part) and whether an outcome of
    it followed), and one for each recursive session and each opened stream (``s`` or ``t``, then `_key` of its name: a
    `_Session`, and the stream's first open's budget and its failures so far). Each call loads the files it reads and
    writes back the ones that changed, under ``record.lock``: first the head marked dirty, then the files, then the head,
    each written whole and renamed into place, so a writer that ends mid-write leaves a dirty head, and the next call
    rebuilds the index from the record."""

    def __init__(self, dir: Path, boot: str, file: list[int] | None, head: dict | None = None) -> None:
        self.dir, self.boot, self.file = dir, boot, file
        self.offset, self.last, self.regular, self.count, self.opens = 0, b"", True, 0, 0
        self.slot: list | None = None
        self.slot_done = False
        #: rebuilt from the record's start: no file of the index is read, and every one is written anew
        self.fresh = head is None
        if head is not None:
            self.offset, self.last, self.regular = head["offset"], head["last"], head["regular"]
            self.count, self.opens, self.slot, self.slot_done = head["count"], head["opens"], head["slot"], head["slot_done"]
        #: a file of the index wouldn't read while an event was applied: the next call rebuilds it
        self.broken = False
        #: the head changed, so it is written back
        self.moved = self.fresh
        self._sessions: dict[str, _Session] = {}
        self._streams: dict[str, dict | None] = {}
        self._changed: tuple[set[str], set[str]] = (set(), set())

    @classmethod
    def of(cls, dir: Path, at: tuple[list[int], int] | None) -> _Index:
        """The index in ``dir`` of the record as it stands (``at``: its device and inode, and its length; None if there is
        none yet), or a new one to be built from the record's start if the one there isn't of this record."""
        boot, file = _boot(), None if at is None else at[0]
        try:
            with open(dir / "head.json", "rb") as f:
                h = json.load(f)
            h["last"] = bytes.fromhex(h["last"])
            if not (set(h) == {*_HEAD, "file", "last", "slot"} and all(type(h[k]) is t for k, t in _HEAD.items())
                    and h["v"] == 1 and h["boot"] == boot and h["dirty"] is False and h["file"] == file
                    and min(h["offset"], h["count"], h["opens"]) >= 0 and len(h["last"]) <= h["offset"]
                    and (0 if at is None else at[1]) >= h["offset"]
                    and (h["slot"] is None or type(h["slot"]) is list and len(h["slot"]) == 3
                         and all(type(x) is str for x in h["slot"][1:]))):
                h = None
        except _BAD:
            h = None
        return cls(dir, boot, file, h)

    def _get(self, name: str) -> Any:
        try:
            with open(self.dir / name, "rb") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def session(self, s: str) -> _Session:
        if s not in self._sessions:
            key = _key(s)
            self._sessions[s] = _Session(None if self.fresh or key is None else self._get(f"s{key}.json"))
        return self._sessions[s]

    def stream(self, t: str) -> dict | None:
        if t not in self._streams:
            key = _key(t)
            v = None if self.fresh or key is None else self._get(f"t{key}.json")
            if not (v is None or type(v) is dict and set(v) == {"budget", "n"} and type(v["n"]) is int
                    and (v["budget"] is None or type(v["budget"]) is int)):
                raise ValueError(f"the firewall's index: t{key}.json is not a stream")
            self._streams[t] = v
        return self._streams[t]

    def load(self, sessions: tuple[str, ...], streams: tuple[str, ...]) -> None:
        """Load what a call may read and append for: ``sessions`` and ``streams``, the session the next open would name,
        the slot's session if no outcome of it followed, and each of those sessions' streams. A file of the index that
        won't read raises here, before the call reads anything."""
        orphan = () if self.slot is None or self.slot_done else (self.slot[1],)
        for s in (*sessions, f"r{self.opens:04d}", *orphan):
            self.members(s)
        for t in streams:
            self.stream(t)

    def members(self, s: str) -> list[dict]:
        """The session's ``member``, each stream of it loaded."""
        member = self.session(s).member
        for m in member:
            if self.stream(m["stream"]) is None:
                raise ValueError(f"the firewall's index: session {s}'s stream {m['stream']} is missing")
        return member

    def add(self, e: Any) -> None:
        """The record's next event, ``e``, is in the index. Everything it changes is loaded first, so a load that fails
        changes nothing."""
        if not (self.regular and _regular(e)):
            self.count, self.regular, self.moved = self.count + 1, False, True
            return
        kind, s = e["event"], e["session"]
        if kind not in ("wait", "slot"):
            self.members(s)
            if kind == "open":
                self.stream(e["stream"])
        self.count, self.moved = self.count + 1, True
        if kind == "slot":
            self.slot, self.slot_done = [e["seq"], s, e["part"]], False
            return
        if kind != "open" and self.slot is not None and self.slot[1] == s and (
                kind == "stop" or kind in ("release", "refused") and e["part"] == self.slot[2]):
            self.slot_done = True
        if kind == "wait":
            return
        sess, tallies = self._sessions[s], self._streams
        if kind == "open":
            self.opens += 1
            if sess.open is None:
                sess.open = e
            m = next((m for m in sess.member if m["stream"] == e["stream"]), None)
            if m is None:
                sess.member.append({"stream": e["stream"], "parts": list(e["parts"]), "released": [], "stopped": False})
            else:
                m["parts"] = list(e["parts"])
            if tallies[e["stream"]] is None:
                tallies[e["stream"]] = {"budget": e.get("budget"), "n": 0}
            self._changed[1].add(e["stream"])
        elif kind == "finish":
            sess.finished = True
        elif kind == "stop":
            if sess.stop is None:
                sess.stop = e
            for m in sess.member:
                m["stopped"] = True
                tallies[m["stream"]]["n"] += 1
                self._changed[1].add(m["stream"])
        elif kind == "release":
            for released in [sess.released] + [m["released"] for m in sess.member]:
                if e["part"] not in released:
                    released.append(e["part"])
        else:
            for m in sess.member:
                t = tallies[m["stream"]]
                if (not all(p in m["released"] for p in m["parts"]) and not m["stopped"]
                        and (t["budget"] is None or t["n"] < t["budget"])):
                    t["n"] += 1
                    self._changed[1].add(m["stream"])
        self._changed[0].add(s)

    def read(self, chunk: bytes) -> None:
        """The record's next bytes, whole lines, are in the index."""
        if chunk:
            self.offset += len(chunk)
            self.last = (self.last + chunk[-_LAST:])[-_LAST:]
            self.moved = True

    def flush(self) -> None:
        """Write back what changed. A write that fails leaves the old head, whose record the next call reads again from
        where it stood, or a dirty one, and the next call rebuilds the index: either way nothing of the index outruns its
        head."""
        try:
            if self.broken:
                (self.dir / "head.json").unlink(missing_ok=True)
            elif self.moved:
                self._write()
        except OSError:
            pass

    def _head(self, dirty: bool) -> dict:
        return {"v": 1, "boot": self.boot, "dirty": dirty, "file": self.file, "offset": self.offset,
                "last": self.last.hex(), "regular": self.regular, "count": self.count, "opens": self.opens,
                "slot": self.slot, "slot_done": self.slot_done}

    def _write(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _put(self.dir, "head.json", self._head(True))
        if self.fresh:
            for p in self.dir.iterdir():
                if p.name != "head.json":
                    p.unlink()
        for s in self._changed[0]:
            _put(self.dir, f"s{_key(s)}.json", self._sessions[s].json())
        for t in self._changed[1]:
            if (x := self._streams.get(t)) is not None:
                _put(self.dir, f"t{_key(t)}.json", x)
        _put(self.dir, "head.json", self._head(False))

    def opened(self, session: str) -> dict | None:
        return self.session(session).open

    def known(self, stream: str) -> bool:
        return self.stream(stream) is not None

    def stopped(self, session: str) -> dict | None:
        return self.session(session).stop

    def finished(self, session: str) -> bool:
        return self.session(session).finished

    def next(self, open_: dict) -> int | None:
        released = self.session(open_["session"]).released
        return next((j for j, p in enumerate(open_["parts"]) if p not in released), None)

    def budget(self, stream: str) -> int | None:
        t = self.stream(stream)
        return None if t is None else t["budget"]

    def failures(self, stream: str) -> int:
        t = self.stream(stream)
        return 0 if t is None else t["n"]

    def orphan(self) -> dict | None:
        if self.slot is None or self.slot_done:
            return None
        seq, session, part = self.slot
        return {"seq": seq, "session": session, "part": part}


def _parts_ok(parts) -> None:
    if not (isinstance(parts, (list, tuple)) and parts and all(isinstance(p, str) and p and p != INNER for p in parts)
            and len(set(parts)) == len(parts)):
        raise ValueError(f"a recursive session's statements are distinct names, at least one, none of them {INNER}, "
                         f"not {json.dumps(parts, sort_keys=True, default=repr)}")


def _schedule_ok(r: dict, n: int) -> None:
    windows = r.get("windows_ms") if isinstance(r, dict) else None
    if not (isinstance(r, dict) and set(r) == {"start_ms", "windows_ms"} and type(r["start_ms"]) is int and r["start_ms"] >= 0
            and isinstance(windows, list) and len(windows) == n and all(type(w) is int and w > 0 for w in windows)):
        raise ValueError(f'a release schedule is {{"start_ms": S, "windows_ms": [one width > 0 for each of the {n} statements]}}, '
                         f'not {json.dumps(r, sort_keys=True, default=repr)}')


Conn = tuple  # (reader, writer, close): a connection's two file objects and how to close it


def relay_session(fw: Firewall, session: str, part: str, worker_r, worker_w, connect: Callable[[], Conn],
                  waiting: Callable[[], None] | None = None, proofs: int = PROOFS) -> dict:
    """One outer session, from a worker (``worker_r``, ``worker_w``) to the challenger (``connect()`` opens its connection):
    held to the module docstring's rules, ``Finish`` forwarded only after the session's ``proofs``. Returns its outcome:
    ``release``, ``stop``, ``refused``, or ``none`` (the worker sent nothing but pings)."""
    while True:
        req = read_frame(worker_r)
        if req is None:
            return {"outcome": "none"}
        if req[:1] != bytes([REQ_PING]):
            break
        write_frame(worker_w, resp_ok())
    with fw.slot(session, part, waiting):
        try:
            fw.admit(session, part)
        except Refused as e:
            out, rest = {"outcome": "refused", "why": str(e)}, (req, str(e))
        else:
            out, rest = _relay_admitted(fw, session, part, req, worker_r, worker_w, connect, proofs)
    if rest is not None:
        try:
            _refuse_all(worker_r, worker_w, *rest)
        except (OSError, ValueError):
            pass
    return out


def _relay_admitted(fw: Firewall, session: str, part: str, req: bytes, worker_r, worker_w, connect: Callable[[], Conn],
                    need: int) -> tuple[dict, tuple | None]:
    """An admitted outer session, relayed while it holds the slot: its outcome, and the request to answer from and the
    refusal to answer with (`_refuse_all`, once the slot is left) when the relay's fault stopped it."""
    up, proofs = None, 0
    try:
        up = connect()
        while True:
            if req[:1] == bytes([REQ_FINISH]):
                if proofs < need:
                    why = f"the worker sent Finish after {proofs} of the session's {need} proofs"
                    fw.stop(session, part, why)
                    _verdict(worker_w, refused(f"{why}: recursive session {session} stopped at {part}"))
                    return {"outcome": "stop", "why": why}, None
                write_frame(up[1], req)
                resp = read_frame(up[0])
                if resp is None:
                    raise ValueError("the challenger ended the session before its verdict")
                v = decode_resp(resp)
                accepted = v[0] == RESP_VERDICT and json.loads(v[1]).get("accepted") is True
                if accepted:
                    fw.release(session, part)
                else:
                    fw.stop(session, part, "the challenger's verdict rejects it")
                write_frame(worker_w, resp)
                return {"outcome": "release" if accepted else "stop", "proofs": proofs}, None
            proofs += req[:1] == bytes([REQ_PROOF])
            write_frame(up[1], req)
            resp = read_frame(up[0])
            if resp is None:
                raise ValueError("the challenger ended the session")
            write_frame(worker_w, resp)
            req = read_frame(worker_r)
            if req is None:
                why = "the worker ended the session before Finish"
                fw.stop(session, part, why)
                return {"outcome": "stop", "why": why}, None
    except Exception as e:  # noqa: BLE001 - any fault of the relay's stops the session, never the firewall
        why = f"the relay's fault ({type(e).__name__}: {e})"
        fw.stop(session, part, why)
        return {"outcome": "stop", "why": why}, (req, refused(f"{why}: recursive session {session} stopped at {part}"))
    finally:
        if up is not None:
            up[2]()


def _refuse_all(worker_r, worker_w, req: bytes | None, why: str) -> None:
    """Answer every request of a refused session with its refusal (``Finish`` with a verdict that names it) until the worker
    stops; nothing reaches the challenger."""
    while req is not None:
        if req[:1] == bytes([REQ_FINISH]):
            _verdict(worker_w, why)
            return
        write_frame(worker_w, resp_str(RESP_REFUSED, why))
        req = read_frame(worker_r)


def _verdict(worker_w, why: str) -> None:
    write_frame(worker_w, resp_str(RESP_VERDICT, json.dumps({"accepted": False, "why": why})))


def tcp(addr: str) -> Callable[[], Conn]:
    """``connect`` for `relay_session`: a TCP connection to ``addr`` (``HOST:PORT``)."""
    host, port = addr.rsplit(":", 1)

    def connect() -> Conn:
        s = socket.create_connection((host, int(port)))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        r, w = s.makefile("rb"), s.makefile("wb")

        def close() -> None:
            for x in (w, r, s):
                try:
                    x.close()
                except OSError:
                    pass
        return r, w, close
    return connect


def relay(fw: Firewall, session: str, part: str, listen: str, to: str, sessions: int = 1,
          ready: Callable[[str], None] | None = None, done: Callable[[dict], None] | None = None, proofs: int = PROOFS) -> list[dict]:
    """Accept ``sessions`` connections from workers on ``listen``, each an outer session of ``session`` at ``part``, relayed to
    the challenger at ``to`` (`relay_session`), each on its own thread, and the slot orders them. The ZK server serves
    ``part`` once: a connection cut before its first request leaves it to the next, and any after its release is refused.
    Returns their outcomes in the order accepted; a fault of the ZK server's own (its record unreadable, say) is the outcome
    ``error``, with nothing served."""
    host, port = listen.rsplit(":", 1)
    res: list[dict | None] = [None] * sessions
    threads = []
    with socket.create_server((host, int(port))) as srv:
        h, p = srv.getsockname()[:2]
        if ready:
            ready(f"{h}:{p}")
        for k in range(sessions):
            conn, _ = srv.accept()

            def one(k=k, conn=conn) -> None:
                with conn, conn.makefile("rb") as rf, conn.makefile("wb") as wf:
                    try:
                        res[k] = {"k": k, **relay_session(fw, session, part, rf, wf, tcp(to), proofs=proofs)}
                    except OSError as e:
                        res[k] = {"k": k, "outcome": "none", "why": f"the worker's connection: {e}"}
                    except Exception as e:  # noqa: BLE001 - reported, so that done is called for every connection
                        res[k] = {"k": k, "outcome": "error", "why": f"the firewall's fault ({type(e).__name__}: {e})"}
                if done:
                    done(res[k])
            t = threading.Thread(target=one)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    return res


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relay", help="relay outer sessions of one recursive session from workers to the challenger")
    r.add_argument("--state", required=True)
    r.add_argument("--session", required=True, help="the recursive session (rec_live's firewall.json)")
    r.add_argument("--part", required=True, help="the V* statement these outer sessions prove (e.g. L3, alg)")
    r.add_argument("--to", required=True, help="the challenger's address, HOST:PORT")
    r.add_argument("--listen", default="127.0.0.1:0")
    r.add_argument("--sessions", type=int, default=1)
    r.add_argument("--proofs", type=int, default=PROOFS, help="the session's proofs, one a rep: Finish is forwarded only after them")
    rc = sub.add_parser("record", help="print the ZK server's record")
    rc.add_argument("--state", required=True)
    a = ap.parse_args(argv)
    fw = Firewall(Path(a.state))
    if a.cmd == "record":
        for e in fw.record():
            print(json.dumps(e, sort_keys=True))
        return 0
    relay(fw, a.session, a.part, a.listen, a.to, a.sessions, ready=lambda addr: print(f"SERVING {addr}", flush=True),
          done=lambda res: print("FIREWALL\t" + json.dumps({"session": a.session, "part": a.part, **res}), flush=True),
          proofs=a.proofs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
