"""``firewall/v0``: the firewall's release side (live PROTOCOL.md §10.5). It sits between the workers' outer prover
(``flock-circuit prove --zk`` with its send check, ``FC_FIREWALL=1``, §10.4) and the challenger, beside the inner phase's
proxy (`verity_flock.rec_live`, §10.3).

    python -m verity_flock.firewall relay --state DIR --session ID --part NAME --to HOST:PORT [--listen 127.0.0.1:0] [--sessions N]
    python -m verity_flock.firewall record --state DIR

The release side of record is Rust's: ``flock-circuit prove --firewall-state DIR --session ID --part NAME``
(``flock_live::release``) keeps the same record, byte for byte (``live/src/firewall_record_vectors.json``, which both sides'
tests run), and the rules below. This module is the inner side's (`rec_live`'s ``serve`` opens and ends recursive sessions
here) and the tests' relay.

Every process of one firewall shares its state directory: ``record.jsonl``, its events in order (each with its ``seq``),
appended under ``record.lock``, and ``slot.lock``, the outer slot. Four rules hold through it:

* **A recursive session ends at its first stop.** The inner phase opens a recursive session (`Firewall.open`: `rec_live`'s
  ``serve`` opens one for each session it serves). Its first stop, inner or outer, is its one failure (``stop``). After it
  the firewall serves and releases nothing more of that recursive session: a relay refuses each later outer session of it
  before anything reaches the challenger (``refused``, not a failure: the challenger sees nothing of it). An outer session
  is admitted only once its recursive session's inner session has finished.
* **Outer proofs go one at a time.** The firewall has one slot, across streams and protocols. An outer session takes it at
  its first request and holds it until the session is released (the challenger's verdict accepts it) or fails. A second
  session that asks meanwhile waits (``wait``) and takes the slot after (``slot``), so each ``slot`` event is followed by its
  session's outcome before the next one. A holder that ended before its outcome (killed, say) is recorded as its session's
  stop by the next session to take the slot.
* **A stream stops at its failure budget.** A stream opened with ``budget`` b (the challenger's stream's ``budget``, P1)
  refuses every outer session and every open after its b-th failure: a recursive session's stop, or a refusal of a recursive
  session that had not stopped, on a stream that had not.
* **Proofs leave at public times.** A recursive session opened with a ``release`` schedule (``start_ms``, its ``parts`` in
  order and one window a part, ``windows_ms``) releases part j's proofs at ``start_ms + windows_ms[0] + ... +
  windows_ms[j]``: proofs ready early wait, and proofs not ready by then are a stop. Only the Rust firewall keeps the
  times; this module's relay refuses a scheduled session rather than release it early.

An outer session stops when the worker cuts it, when the worker sends ``Finish`` before its proofs (the send check refused:
``flock-circuit`` then sends ``Finish`` at once), when the challenger's verdict rejects it, or on any fault of the relay's.
Except for a rejecting verdict, which the challenger reached itself, the relay then cuts the challenger's connection without
sending ``Finish``. So every stop the firewall makes looks the same to the challenger: a session cut after some message,
never why.
"""
from __future__ import annotations

import fcntl
import json
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .rec_live import (REQ_FINISH, REQ_PING, REQ_PROOF, RESP_REFUSED, RESP_VERDICT, decode_resp, read_frame, refused,
                       resp_ok, resp_str, write_frame)

KIND = "firewall/v0"
INNER = "inner"


class Refused(Exception):
    """The firewall serves nothing of this outer session (its message is the refusal)."""


class Firewall:
    """The state every process of one firewall shares (``state``, a directory)."""

    def __init__(self, state: Path):
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        self._log = self.state / "record.jsonl"

    @contextmanager
    def _locked(self) -> Iterator[list[dict]]:
        with open(self.state / "record.lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield self._events()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _events(self) -> list[dict]:
        if not self._log.exists():
            return []
        return [json.loads(line) for line in self._log.read_text().splitlines() if line]

    def _append(self, events: list[dict], ev: dict) -> dict:
        ev = {"seq": len(events), **ev}
        with open(self._log, "a") as f:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
        events.append(ev)
        return ev

    def record(self) -> list[dict]:
        """Every event so far, in order."""
        with self._locked() as ev:
            return ev

    def open(self, stream: str, budget: int | None = None, release: dict | None = None) -> str:
        """A new recursive session on ``stream``, opened by its inner session: its id. ``budget`` is the stream's failure
        budget (P1's ``budget``, b), the same at every open of the stream; once the stream has ``budget`` failures, `Refused`.
        ``release`` is the recursive session's release schedule, public and fixed here, before the session: part ``j`` of
        ``release["parts"]`` is released at the end of its window, ``start_ms + sum(windows_ms[:j + 1])`` (the Rust
        firewall's rule 3)."""
        if budget is not None and not (type(budget) is int and budget >= 1):
            raise ValueError(f"a failure budget is a positive integer, not {budget!r}")
        if release is not None:
            _schedule_ok(release)
        with self._locked() as ev:
            first = next((e for e in ev if e["event"] == "open" and e["stream"] == stream), None)
            if first is not None and first.get("budget") != budget:
                raise ValueError(f"stream {stream}'s budget is {first.get('budget')}, not {budget}")
            if budget is not None and (n := _failures(ev, stream)) >= budget:
                raise Refused(refused(f"stream {stream} stopped at its failure budget ({n} of {budget})"))
            sid = f"r{sum(e['event'] == 'open' for e in ev):04d}"
            e = {"event": "open", "session": sid, "stream": stream}
            if budget is not None:
                e["budget"] = budget
            if release is not None:
                e["release"] = release
            self._append(ev, e)
            return sid

    def finish(self, session: str) -> None:
        """The recursive session's inner session finished: its outer sessions may be served."""
        with self._locked() as ev:
            self._append(ev, {"event": "finish", "session": session, "part": INNER})

    def stop(self, session: str, part: str, why: str) -> bool:
        """The recursive session stops at ``part``: its one failure, recorded once (False if it had already stopped)."""
        with self._locked() as ev:
            if _stopped(ev, session):
                return False
            self._append(ev, {"event": "stop", "session": session, "part": part, "why": why})
            return True

    def release(self, session: str, part: str) -> None:
        """An outer proof of the recursive session was released (the challenger's verdict accepted it)."""
        with self._locked() as ev:
            self._append(ev, {"event": "release", "session": session, "part": part})

    def admit(self, session: str, part: str, *, timed: bool = False) -> int | None:
        """Raise `Refused` unless an outer session of ``session`` may be served now: the recursive session is open, has not
        stopped, its stream has not stopped at its budget, its inner session finished, and ``part`` has a place in its
        release schedule if it has one. A refusal is recorded (``refused``). Returns ``part``'s release time (rule 3), or
        None. Only a ``timed`` caller (one that holds the proofs to that time: the Rust firewall, or a test of its rules)
        is admitted to a scheduled session; this module's relay is not."""
        with self._locked() as ev:
            why, at = None, None
            open_ = next((e for e in ev if e["event"] == "open" and e["session"] == session), None)
            if open_ is None:
                why = f"recursive session {session} was never opened"
            elif (s := _stopped(ev, session)) is not None:
                why = f"recursive session {session} stopped at {s['part']}: the firewall serves nothing more of it"
            elif (b := _budget(ev, open_["stream"])) is not None and (n := _failures(ev, open_["stream"])) >= b:
                why = f"stream {open_['stream']} stopped at its failure budget ({n} of {b})"
            elif not any(e["event"] == "finish" and e["session"] == session for e in ev):
                why = f"recursive session {session}: its inner session has not finished"
            elif (r := open_.get("release")) is not None:
                if part not in r["parts"]:
                    why = f"recursive session {session}: {part} has no place in its release schedule"
                elif not timed:
                    why = (f"recursive session {session}: its release schedule needs the Rust firewall "
                           "(flock-circuit prove --firewall-state)")
                else:
                    at = r["start_ms"] + sum(r["windows_ms"][:r["parts"].index(part) + 1])
            if why is not None:
                self._append(ev, {"event": "refused", "session": session, "part": part})
                raise Refused(refused(why))
            return at

    def stopped(self, session: str) -> dict | None:
        with self._locked() as ev:
            return _stopped(ev, session)

    def failures(self, stream: str) -> int:
        """The failures on ``stream`` so far (rule 4): its recursive sessions' stops, and each refusal of one that had not
        stopped while the stream had not."""
        with self._locked() as ev:
            return _failures(ev, stream)

    @contextmanager
    def slot(self, session: str, part: str, waiting: Callable[[], None] | None = None) -> Iterator[None]:
        """The firewall's one outer slot, held for the block. A session that finds it taken records ``wait``, calls
        ``waiting``, and blocks until the holder leaves it. Once it holds the slot, a ``slot`` event whose session has no
        outcome after it (its holder died) is recorded as that session's stop, and then ``slot`` is recorded."""
        with open(self.state / "slot.lock", "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                with self._locked() as ev:
                    self._append(ev, {"event": "wait", "session": session, "part": part})
                if waiting:
                    waiting()
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                with self._locked() as ev:
                    if (o := _orphan(ev)) is not None and _stopped(ev, o["session"]) is None:
                        self._append(ev, {"event": "stop", "session": o["session"], "part": o["part"],
                                          "why": f"the slot's holder ended before its outcome (slot {o['seq']})"})
                    self._append(ev, {"event": "slot", "session": session, "part": part})
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def _stopped(ev: list[dict], session: str) -> dict | None:
    return next((e for e in ev if e["event"] == "stop" and e["session"] == session), None)


def _budget(ev: list[dict], stream: str) -> int | None:
    return next((e.get("budget") for e in ev if e["event"] == "open" and e["stream"] == stream), None)


def _failures(ev: list[dict], stream: str) -> int:
    budget, mine, stopped, n = _budget(ev, stream), set(), set(), 0
    for e in ev:
        if e["event"] == "open" and e["stream"] == stream:
            mine.add(e["session"])
        elif e["event"] == "stop" and e["session"] in mine:
            stopped.add(e["session"])
            n += 1
        elif e["event"] == "refused" and e["session"] in mine and e["session"] not in stopped and (budget is None or n < budget):
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


def _schedule_ok(r: dict) -> None:
    parts, windows = (r.get("parts"), r.get("windows_ms")) if isinstance(r, dict) else (None, None)
    if not (isinstance(r, dict) and set(r) == {"start_ms", "parts", "windows_ms"} and type(r["start_ms"]) is int and r["start_ms"] >= 0
            and isinstance(parts, list) and parts and all(isinstance(p, str) for p in parts) and len(set(parts)) == len(parts)
            and isinstance(windows, list) and len(windows) == len(parts) and all(type(w) is int and w > 0 for w in windows)):
        raise ValueError(f'a release schedule is {{"start_ms": S, "parts": [distinct names], "windows_ms": [one width > 0 a part]}}, '
                         f'not {r!r}')


Conn = tuple  # (reader, writer, close): a connection's two file objects and how to close it


def relay_session(fw: Firewall, session: str, part: str, worker_r, worker_w, connect: Callable[[], Conn],
                  waiting: Callable[[], None] | None = None) -> dict:
    """One outer session, from a worker (``worker_r``, ``worker_w``) to the challenger (``connect()`` opens its connection):
    held to the module docstring's rules. Returns its outcome: ``release``, ``stop``, ``refused``, or ``none`` (the worker
    sent nothing but pings)."""
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
            _refuse_all(worker_r, worker_w, req, str(e))
            return {"outcome": "refused", "why": str(e)}
        up, proofs = None, 0
        try:
            up = connect()
            while True:
                if req[:1] == bytes([REQ_FINISH]):
                    if proofs == 0:
                        why = "the worker sent Finish before its proofs"
                        fw.stop(session, part, why)
                        _verdict(worker_w, refused(f"{why}: recursive session {session} stopped at {part}"))
                        return {"outcome": "stop", "why": why}
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
                    return {"outcome": "release" if accepted else "stop", "proofs": proofs}
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
                    return {"outcome": "stop", "why": why}
        except Exception as e:  # noqa: BLE001 - any fault of the relay's stops the session, never the firewall
            why = f"the relay's fault ({type(e).__name__}: {e})"
            fw.stop(session, part, why)
            if up is not None:
                up[2]()
                up = None
            try:
                _refuse_all(worker_r, worker_w, req, refused(f"{why}: recursive session {session} stopped at {part}"))
            except (OSError, ValueError):
                pass
            return {"outcome": "stop", "why": why}
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
          ready: Callable[[str], None] | None = None, done: Callable[[dict], None] | None = None) -> list[dict]:
    """Accept ``sessions`` connections from workers on ``listen``, each an outer session of ``session`` at ``part``, relayed to
    the challenger at ``to`` (`relay_session`), each on its own thread: a worker may open its next session while the last
    one awaits its verdict (``FC_VERIFY_AHEAD``), and the slot orders them. Returns their outcomes in the order accepted."""
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
                        res[k] = {"k": k, **relay_session(fw, session, part, rf, wf, tcp(to))}
                    except OSError as e:
                        res[k] = {"k": k, "outcome": "none", "why": f"the worker's connection: {e}"}
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
    rc = sub.add_parser("record", help="print the firewall's record")
    rc.add_argument("--state", required=True)
    a = ap.parse_args(argv)
    fw = Firewall(Path(a.state))
    if a.cmd == "record":
        for e in fw.record():
            print(json.dumps(e, sort_keys=True))
        return 0
    relay(fw, a.session, a.part, a.listen, a.to, a.sessions, ready=lambda addr: print(f"SERVING {addr}", flush=True),
          done=lambda res: print("FIREWALL\t" + json.dumps({"session": a.session, "part": a.part, **res}), flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
