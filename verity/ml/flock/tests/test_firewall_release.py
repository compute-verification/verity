"""The firewall's release side (`verity_flock.firewall`, live PROTOCOL.md §10.5), one rule a test. A recursive session's V*
statements go in their order, each once, and it ends at its first stop, inner or outer: its stop is its one failure, and
the firewall serves nothing more of it, so a later V* statement's outer session is refused before anything reaches the
challenger, and so is a statement asked out of its order (the one it skipped, never opened, is the stop). Outer proofs go
one at a time across streams: a second requester waits for the first's release or failure, and the record shows the order.
The worker and the challenger here are the test's, on socket pairs; the clock is never read."""
from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest
from verity_flock import firewall as FW
from verity_flock import rec_live as RL

import test_rec_live as T
import test_rec_live_firewall as G

HONEST = [RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, "{}"), RL.encode_req(RL.REQ_ROUND, "circuit/rep0", b"m", 1),
          RL.encode_req(RL.REQ_PROOF, "circuit", 0, b"p0"), RL.encode_req(RL.REQ_PROOF, "circuit", 1, b"p1"),
          RL.encode_req(RL.REQ_FINISH)]
#: what flock-circuit sends after its send check refuses: Finish at once, no proofs
REFUSED = HONEST[:3] + [RL.encode_req(RL.REQ_FINISH)]
#: a recursive session's V* statements in their order
PARTS = ["L0", "L1", "L2", "L3", "alg"]


class Challenger:
    """The challenger's side of one or more outer sessions: answers every request (``Finish`` with a verdict that accepts,
    unless ``accept`` is False) and logs ``(name, tag)`` for each request it receives and ``(name, "cut")`` when a
    connection ends without ``Finish``. The first request sets ``got_first``, then waits for ``hold`` if given."""

    def __init__(self, log: list, name: str, accept: bool = True, hold: threading.Event | None = None):
        self.log, self.name, self.accept, self.hold = log, name, accept, hold
        self.got_first, self.threads = threading.Event(), []

    def connect(self) -> FW.Conn:
        a, b = socket.socketpair()
        t = threading.Thread(target=self._serve, args=(b,))
        t.start()
        self.threads.append(t)
        r, w = a.makefile("rb"), a.makefile("wb")
        return r, w, lambda: (w.close(), r.close(), a.close())

    def _serve(self, s: socket.socket) -> None:
        try:
            with s, s.makefile("rb") as r, s.makefile("wb") as w:
                first = True
                while (req := RL.read_frame(r)) is not None:
                    self.log.append((self.name, req[0]))
                    if first:
                        self.got_first.set()
                        if self.hold is not None:
                            self.hold.wait()
                    first = False
                    if req[0] == RL.REQ_FINISH:
                        RL.write_frame(w, RL.resp_str(RL.RESP_VERDICT, json.dumps({"accepted": self.accept, "why": ""})))
                        return
                    RL.write_frame(w, RL.resp_ok())
        except ConnectionError:
            pass  # the relay closed before reading our last response: a cut like any other
        self.log.append((self.name, "cut"))

    def join(self) -> None:
        for t in self.threads:
            t.join()


def _outer(fw, session, part, reqs, chal: Challenger, waiting=None, cut_after: int | None = None) -> tuple[dict, list]:
    """One outer session through `relay_session`, the worker sending ``reqs`` (or only the first ``cut_after`` of them, then
    closing): its outcome and the worker's responses."""
    a, b = socket.socketpair()
    got: dict = {}

    def relay():
        with b, b.makefile("rb") as r, b.makefile("wb") as w:
            got["out"] = FW.relay_session(fw, session, part, r, w, chal.connect, waiting)

    t = threading.Thread(target=relay)
    t.start()
    resp = []
    with a, a.makefile("rb") as r, a.makefile("wb") as w:
        for req in reqs[:cut_after]:
            RL.write_frame(w, req)
            resp.append(RL.decode_resp(RL.read_frame(r)))
            if resp[-1][0] == RL.RESP_VERDICT:
                break
    t.join()
    chal.join()
    return got["out"], resp


def _opened(tmp_path, stream="pouw"):
    fw = FW.Firewall(tmp_path / "fw")
    sid = fw.open(stream, PARTS)
    fw.finish(sid)
    return fw, sid


def _events(fw):
    return [(e["event"], e.get("part")) for e in fw.record()]


def test_an_honest_recursive_session_is_released_statement_by_statement(tmp_path):
    """Every statement in its order is released, and then the recursive session is done: a statement asked after its last
    release is refused, which is no failure."""
    fw, sid = _opened(tmp_path)
    log: list = []
    for part in PARTS:
        out, resp = _outer(fw, sid, part, HONEST, Challenger(log, part))
        assert out["outcome"] == "release" and json.loads(resp[-1][1])["accepted"] is True
    assert _events(fw) == [("open", None), ("finish", "inner")] + [(e, p) for p in PARTS for e in ("slot", "release")]
    assert [t for _, t in log] == [r[0] for r in HONEST] * len(PARTS) and fw.failures("pouw") == 0
    out, _ = _outer(fw, sid, "L0", HONEST, Challenger(log, "again"))
    assert out["why"] == f"FIREWALL-REFUSED: recursive session {sid} released all its statements: the firewall serves nothing more of it"
    assert fw.failures("pouw") == 0 and fw.stopped(sid) is None and "again" not in [n for n, _ in log]


def test_l0_cut_before_its_first_request_then_l1_is_refused(tmp_path):
    """L0's session cut before its first request reaches nothing (the relay connects to the
    challenger only at that request) and leaves no record. Asking L1 next is refused, and L0, never opened, is the recursive
    session's stop where it stands; L0 is then refused too."""
    fw, sid = _opened(tmp_path)
    log: list = []
    out, _ = _outer(fw, sid, "L0", HONEST, Challenger(log, "L0"), cut_after=0)
    assert out == {"outcome": "none"} and _events(fw) == [("open", None), ("finish", "inner")]
    out, resp = _outer(fw, sid, "L1", HONEST, Challenger(log, "L1"))
    assert out == {"outcome": "refused", "why": f"FIREWALL-REFUSED: recursive session {sid} stopped at L0: L1 is not its next statement"}
    assert resp[0] == (RL.RESP_REFUSED, out["why"])
    assert fw.stopped(sid) == {"seq": 3, "event": "stop", "session": sid, "part": "L0", "why": "never opened: L1 was asked in its place"}
    out, _ = _outer(fw, sid, "L0", HONEST, Challenger(log, "L0"))
    assert out["why"] == f"FIREWALL-REFUSED: recursive session {sid} stopped at L0: the firewall serves nothing more of it"
    assert log == [] and fw.failures("pouw") == 1


def test_l1_asked_before_l0_is_refused(tmp_path):
    """No statement is served before the ones ahead of it, so the worker can't choose which
    of a recursive session's statements the challenger sees, nor in what order."""
    fw, sid = _opened(tmp_path)
    log: list = []
    out, _ = _outer(fw, sid, "L1", HONEST, Challenger(log, "L1"))
    assert out["why"] == f"FIREWALL-REFUSED: recursive session {sid} stopped at L0: L1 is not its next statement"
    assert _events(fw)[2:] == [("slot", "L1"), ("stop", "L0"), ("refused", "L1")]
    assert log == [] and fw.failures("pouw") == 1


def test_l0_asked_again_after_its_release_is_refused(tmp_path):
    """A released statement is not served again. Its repeat is refused, and the recursive
    session stops at the next statement, never opened."""
    fw, sid = _opened(tmp_path)
    log: list = []
    assert _outer(fw, sid, "L0", HONEST, Challenger(log, "L0"))[0]["outcome"] == "release"
    out, _ = _outer(fw, sid, "L0", HONEST, Challenger(log, "L0 again"))
    assert out["why"] == f"FIREWALL-REFUSED: recursive session {sid} stopped at L1: L0 is not its next statement"
    assert fw.stopped(sid)["part"] == "L1" and fw.stopped(sid)["why"] == "never opened: L0 was asked in its place"
    assert _outer(fw, sid, "L1", HONEST, Challenger(log, "L1"))[0]["outcome"] == "refused"
    assert [n for n, _ in log] == ["L0"] * len(HONEST) and fw.failures("pouw") == 1


def test_finish_is_forwarded_only_after_all_of_the_sessions_proofs(tmp_path):
    """``Finish`` after one of a session's two proofs is a stop, and the challenger gets the
    cut, never ``Finish``."""
    fw, sid = _opened(tmp_path)
    log: list = []
    out, resp = _outer(fw, sid, "L0", HONEST[:4] + HONEST[-1:], Challenger(log, "L0"))
    assert out == {"outcome": "stop", "why": "the worker sent Finish after 1 of the session's 2 proofs"}
    assert json.loads(resp[-1][1])["accepted"] is False
    assert [t for _, t in log] == [r[0] for r in HONEST[:4]] + ["cut"] and fw.stopped(sid)["part"] == "L0"


def test_a_refused_sessions_worker_does_not_hold_the_slot(tmp_path):
    """A refused session's worker that keeps its connection open is answered outside the slot,
    so another stream's session is served meanwhile without waiting for it."""
    fw, sid = _opened(tmp_path)
    other = fw.open("pous", PARTS)
    fw.finish(other)
    fw.stop(sid, "inner", "the test's")
    a, b = socket.socketpair()
    got: dict = {}

    def relay():
        with b, b.makefile("rb") as r, b.makefile("wb") as w:
            got["out"] = FW.relay_session(fw, sid, "L0", r, w, Challenger([], "unreached").connect)

    t = threading.Thread(target=relay)
    t.start()
    log: list = []
    with a, a.makefile("rb") as r, a.makefile("wb") as w:
        RL.write_frame(w, HONEST[0])
        assert RL.decode_resp(RL.read_frame(r))[0] == RL.RESP_REFUSED

        def waited():
            raise AssertionError("the other stream's session waited for the slot a refused session's worker holds")
        out, _ = _outer(fw, other, "L0", HONEST, Challenger(log, "other"), waiting=waited)
        assert out["outcome"] == "release"
        RL.write_frame(w, HONEST[-1])
        assert json.loads(RL.decode_resp(RL.read_frame(r))[1])["accepted"] is False
    t.join()
    assert got["out"]["outcome"] == "refused" and "wait" not in [e for e, _ in _events(fw)]


def test_a_torn_record_fails_closed_and_the_relay_reports_it(tmp_path):
    """A torn last line (a writer killed inside its append) is never skipped. Every call fails,
    so nothing is served, and the relay reports the fault for the connection instead of losing it."""
    fw, sid = _opened(tmp_path)
    rec = tmp_path / "fw" / "record.jsonl"
    torn = rec.read_text() + '{"event": "sto'
    rec.write_text(torn)
    with pytest.raises(json.JSONDecodeError):
        fw.admit(sid, "L0")
    bound, got, outs = threading.Event(), {}, []

    def run():
        got["res"] = FW.relay(fw, sid, "L0", "127.0.0.1:0", "127.0.0.1:9", ready=lambda a: (got.update(addr=a), bound.set()),
                              done=outs.append)

    t = threading.Thread(target=run)
    t.start()
    bound.wait()
    w = G._Wire(got["addr"])
    try:
        try:
            answer = w.handle(HONEST[0])
        except OSError:
            answer = None
        assert answer is None
    finally:
        w.close()
    t.join()
    assert [r["outcome"] for r in got["res"]] == ["error"] and got["res"][0]["why"].startswith("the firewall's fault (JSONDecodeError")
    assert outs == got["res"] and rec.read_text() == torn


@pytest.mark.parametrize("how", ["refused", "cut", "rejected", "fault"])
def test_an_outer_stop_ends_the_recursive_session_and_the_next_statement_is_refused(tmp_path, how):
    """Each way an outer session stops (the send check refused, so the worker sends Finish before its proofs; the worker
    cuts it; the challenger's verdict rejects it; the relay's own fault, here a challenger that cannot be reached) is the
    recursive session's one failure. The next V* statement's session is then refused before the challenger hears of it, and
    so is every later one, none of them a failure."""
    fw, sid = _opened(tmp_path)
    log: list = []
    assert _outer(fw, sid, "L0", HONEST, Challenger(log, "L0"))[0]["outcome"] == "release"
    chal = Challenger(log, "L1", accept=how != "rejected")
    if how == "fault":
        chal.connect = lambda: (_ for _ in ()).throw(ConnectionRefusedError("no challenger"))
    out, resp = _outer(fw, sid, "L1", REFUSED if how == "refused" else HONEST, chal, cut_after=3 if how == "cut" else None)
    assert out["outcome"] == "stop"
    for part in ("L2", "L3"):
        nxt, nresp = _outer(fw, sid, part, HONEST, Challenger(log, part))
        assert nxt["outcome"] == "refused" and nxt["why"] == f"FIREWALL-REFUSED: recursive session {sid} stopped at L1: the firewall serves nothing more of it"
        assert nresp[0] == (RL.RESP_REFUSED, nxt["why"]) and json.loads(nresp[-1][1]) == {"accepted": False, "why": nxt["why"]}
    assert not any(name in ("L2", "L3") for name, _ in log), "nothing of a later statement reached the challenger"
    assert _events(fw)[-6:] == [("slot", "L1"), ("stop", "L1"), ("slot", "L2"), ("refused", "L2"), ("slot", "L3"), ("refused", "L3")]
    assert fw.failures("pouw") == 1 and fw.stopped(sid)["part"] == "L1"


@pytest.mark.parametrize("how", ["refused", "cut", "fault"])
def test_a_firewall_stop_reaches_the_challenger_as_a_cut_without_finish(tmp_path, how):
    """The challenger sees the same thing however the firewall stopped a session: the messages before the stop, then the
    connection cut, never ``Finish`` (so no bit of how it ended)."""
    fw, sid = _opened(tmp_path)
    log: list = []
    chal = Challenger(log, "L0")
    if how == "fault":
        real = chal.connect

        def failing():
            r, w, close = real()
            r.close()
            return r, w, close
        chal.connect = failing
    _outer(fw, sid, "L0", REFUSED if how == "refused" else HONEST, chal, cut_after=3 if how == "cut" else None)
    assert RL.REQ_FINISH not in [t for _, t in log] and log[-1] == ("L0", "cut")
    assert fw.stopped(sid)["part"] == "L0"


def test_an_outer_session_is_served_only_after_its_inner_session_finished(tmp_path):
    """A refusal of a recursive session that had not stopped is one of its stream's failures (rule 4); one that was never
    opened is on no stream."""
    fw = FW.Firewall(tmp_path / "fw")
    log: list = []
    out, _ = _outer(fw, "r0000", "L0", HONEST, Challenger(log, "L0"))
    assert out == {"outcome": "refused", "why": "FIREWALL-REFUSED: recursive session r0000 was never opened"}
    sid = fw.open("pouw", PARTS)
    out, _ = _outer(fw, sid, "L0", HONEST, Challenger(log, "L0"))
    assert out["why"] == f"FIREWALL-REFUSED: recursive session {sid}: its inner session has not finished" and log == []
    assert fw.failures("pouw") == 1


VECTORS = Path(__file__).resolve().parents[1] / "live" / "src" / "firewall_record_vectors.json"


def _vector_op(fw: FW.Firewall, op: list):
    def why(e: FW.Refused) -> dict:
        return {"refused": str(e).removeprefix("FIREWALL-REFUSED: ")}
    k = op[0]
    if k == "open":
        try:
            return fw.open(op[1], op[2], op[3], op[4])
        except FW.Refused as e:
            return why(e)
        except ValueError as e:
            return {"error": str(e)}
    if k == "admit":
        try:
            return {"release_ms": fw.admit(op[1], op[2], timed=True)}
        except FW.Refused as e:
            return why(e)
    if k == "slot":
        with fw.slot(op[1], op[2]):
            return None
    return getattr(fw, k)(*op[1:])


def test_the_record_is_the_rust_firewalls_byte_for_byte(tmp_path):
    """Each case of ``firewall_record_vectors.json``, run on an empty state directory, gives its results and its
    ``record.jsonl`` exactly: rules 1 and 2 (with a slot holder that died), the statements' order (a skip, a repeat, a
    recursive session released whole, statements that can't be one's), the budget, the schedule, and strings Python
    escapes. ``flock_live::release``'s ``the_shared_vectors_hold`` runs the same file, so the two firewalls read and write
    one record."""
    v = json.loads(VECTORS.read_text())
    assert v["kind"] == "firewall/v0/vectors" and {c["name"] for c in v["cases"]} == {"rules-1-2", "order", "budget", "schedule", "escapes"}
    for c in v["cases"]:
        fw = FW.Firewall(tmp_path / c["name"])
        assert [_vector_op(fw, op) for op in c["ops"]] == c["results"], c["name"]
        assert (tmp_path / c["name"] / "record.jsonl").read_text() == c["record"], c["name"]


def test_a_stream_stops_at_its_budget_and_the_python_relay_keeps_no_schedule(tmp_path):
    """At its ``budget``-th failure a stream refuses every outer session and every open; the budget is the same at every
    open. A scheduled recursive session is the Rust firewall's: this module's relay refuses it, rather than release its
    proofs before their time."""
    fw = FW.Firewall(tmp_path / "fw")
    log: list = []
    a, b = fw.open("pouw", PARTS, budget=1), fw.open("pouw", PARTS, budget=1)
    for sid in (a, b):
        fw.finish(sid)
    with pytest.raises(ValueError, match="stream pouw's budget is 1, not 2"):
        fw.open("pouw", PARTS, budget=2)
    assert _outer(fw, a, "L0", REFUSED, Challenger(log, "L0"))[0]["outcome"] == "stop"
    out, _ = _outer(fw, b, "L0", HONEST, Challenger(log, "b"))
    assert out["why"] == "FIREWALL-REFUSED: stream pouw stopped at its failure budget (1 of 1)" and fw.failures("pouw") == 1
    with pytest.raises(FW.Refused, match=r"stream pouw stopped at its failure budget \(1 of 1\)"):
        fw.open("pouw", PARTS, budget=1)
    for bad in ({"start_ms": 0, "windows_ms": [0]}, {"start_ms": 0, "windows_ms": [1, 1]}, {"start_ms": 0, "parts": ["L0"], "windows_ms": [1]}):
        with pytest.raises(ValueError, match="a release schedule is"):
            fw.open("other", ["L0"], release=bad)
    for bad in ([], ["L0", "L0"], ["L0", "inner"], [""], "L0"):
        with pytest.raises(ValueError, match="a recursive session's statements are"):
            fw.open("other", bad)
    s = fw.open("other", ["L0"], release={"start_ms": 0, "windows_ms": [1000]})
    fw.finish(s)
    out, _ = _outer(fw, s, "L0", HONEST, Challenger(log, "s"))
    assert out["why"].endswith("its release schedule needs the Rust firewall (flock-circuit prove --firewall-state)")
    assert not any(name in ("b", "s") for name, _ in log)


def _serve(tmp_path, sessions, fw, **kw):
    bound, got = threading.Event(), {}

    def run():
        try:
            got["res"] = RL.serve("127.0.0.1:0", tmp_path / "inner", T._counter_rand(), sessions=sessions,
                                  schedule=G._schedule(), ready=lambda a: (got.update(addr=a), bound.set()), firewall=fw,
                                  stream="pouw", **{"parts": PARTS, **kw})
        except Exception as e:  # noqa: BLE001 - kept as text, so that no frame keeps serve's connection open
            got["error"] = f"{type(e).__name__}: {e}"
        finally:
            bound.set()

    t = threading.Thread(target=run)
    t.start()
    bound.wait()
    return t, got


def test_an_inner_stop_ends_the_recursive_session(tmp_path):
    """`rec_live.serve` opens a recursive session for each session it serves: a refused inner session (here Link's y of the
    wrong length) is that recursive session's stop, and no V* statement of it is then served; a finished one lets them be.
    The ``firewall.json`` beside each inner session's record names its recursive session and nothing else: the firewall's
    state stays with the developer."""
    fw = FW.Firewall(tmp_path / "fw")
    t, got = _serve(tmp_path, 2, fw)
    w = G._Wire(got["addr"])
    assert not G._held(G._run(w, y=bytes(48)))
    w.close()
    w = G._Wire(got["addr"])
    assert G._held(G._run(w))
    w.close()
    t.join()
    named = [json.loads((tmp_path / "inner" / f"s{k}" / "firewall.json").read_text()) for k in range(2)]
    assert named == [{"kind": "firewall/v0/session", "session": r["firewall"]} for r in got["res"]]
    assert [r["firewall"] for r in got["res"]] == ["r0000", "r0001"]
    assert fw.stopped("r0000")["part"] == "inner" and fw.stopped("r0001") is None
    log: list = []
    out, _ = _outer(fw, "r0000", "L0", HONEST, Challenger(log, "L0"))
    assert out["why"] == "FIREWALL-REFUSED: recursive session r0000 stopped at inner: the firewall serves nothing more of it" and log == []
    assert _outer(fw, "r0001", "L0", HONEST, Challenger(log, "L0"))[0]["outcome"] == "release"
    assert fw.failures("pouw") == 1


def test_serve_opens_each_recursive_session_with_the_streams_budget_and_its_public_schedule(tmp_path):
    """`rec_live.serve` opens each recursive session with its statements in their order, the stream's ``budget`` and the
    public windows, the schedule starting when the connection was accepted (``now_ms``, injected); once the stream has
    stopped at its budget, the next connection is closed before anything is served, and ``serve`` returns."""
    fw = FW.Firewall(tmp_path / "fw")
    windows = {"windows_ms": [12000, 28000]}
    with pytest.raises(ValueError, match=r"statements in their order \(parts\)"):
        RL.serve("127.0.0.1:0", tmp_path / "x", schedule=G._schedule(), firewall=fw)
    t, got = _serve(tmp_path, 2, fw, parts=["L0", "alg"], budget=1, release=windows, now_ms=lambda: 1234)
    w = G._Wire(got["addr"])
    assert not G._held(G._run(w, y=bytes(48)))
    w.close()
    w = G._Wire(got["addr"])
    try:
        try:
            answer = w.handle(RL.encode_req(RL.REQ_REGISTER))
        except OSError:
            answer = None
        assert answer is None, "a connection on a stopped stream was served"
    finally:
        w.close()
    t.join()
    assert [r["firewall"] for r in got["res"]] == ["r0000", None] and fw.failures("pouw") == 1
    assert got["res"][1]["refused"] == "FIREWALL-REFUSED: stream pouw stopped at its failure budget (1 of 1)"
    opened = [e for e in fw.record() if e["event"] == "open"]
    assert opened == [{"seq": 0, "event": "open", "session": "r0000", "stream": "pouw", "parts": ["L0", "alg"], "budget": 1,
                       "release": {"start_ms": 1234, **windows}}]


def test_a_recursive_session_the_firewall_cannot_open_is_closed_unserved(tmp_path):
    """A recursive session the firewall refuses to open as given (here a release schedule with one window for two
    statements) is the developer's fault: `serve` closes the connection before anything is served and raises, rather than
    leave the prover waiting on a connection nobody answers, and nothing is opened."""
    fw = FW.Firewall(tmp_path / "fw")
    t, got = _serve(tmp_path, 1, fw, parts=["L0", "alg"], release={"windows_ms": [12000]}, now_ms=lambda: 1234)
    w = G._Wire(got["addr"])
    try:
        try:
            answer = w.handle(RL.encode_req(RL.REQ_REGISTER))
        except OSError:
            answer = None
        assert answer is None, "a connection the firewall could not open was served"
    finally:
        w.close()
    t.join()
    assert got["error"].startswith("ValueError: a release schedule is") and "res" not in got
    assert fw.record() == []


def test_a_stop_is_recorded_once():
    """A recursive session has one failure however many of its parts report a stop after the first."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        fw = FW.Firewall(Path(d))
        sid = fw.open("pous", PARTS)
        assert fw.stop(sid, "inner", "a") and not fw.stop(sid, "L0", "b")
        assert fw.stopped(sid) == {"seq": 1, "event": "stop", "session": sid, "part": "inner", "why": "a"} and fw.failures("pous") == 1


@pytest.mark.parametrize("second", ["release", "stop"])
def test_outer_proofs_go_one_at_a_time_across_streams(tmp_path, second):
    """Two requesters on two streams (each with its own `Firewall` on the shared state, as two processes have): the second
    asks while the first holds the slot, waits, and is served only after the first's outcome; the record shows wait, then
    the first's outcome, then the second's slot. Nothing of the second reaches its challenger before the first's verdict."""
    fa, fb = FW.Firewall(tmp_path / "fw"), FW.Firewall(tmp_path / "fw")
    sa, sb = fa.open("pouw", PARTS), fb.open("pous", PARTS)
    fa.finish(sa)
    fb.finish(sb)
    log: list = []
    hold, b_moved = threading.Event(), threading.Event()
    chal_a = Challenger(log, "A", hold=hold, accept=True)
    chal_b = Challenger(log, "B", accept=second == "release")
    chal_b.got_first = b_moved
    got: dict = {}
    ta = threading.Thread(target=lambda: got.update(a=_outer(fa, sa, "L0", HONEST, chal_a)[0]))
    ta.start()
    chal_a.got_first.wait()
    tb = threading.Thread(target=lambda: got.update(b=_outer(fb, sb, "L0", HONEST, chal_b, waiting=b_moved.set)[0]))
    tb.start()
    try:
        b_moved.wait()  # B waits for the slot, or (were there no slot) its first request reaches its challenger
        assert [n for n, _ in log] == ["A"], "B reached its challenger while A held the slot"
    finally:
        hold.set()
        ta.join()
        tb.join()
    assert got["a"]["outcome"] == "release" and got["b"]["outcome"] == second
    ev = [(e["event"], e["session"]) for e in fa.record() if e["event"] in ("wait", "slot", "release", "stop")]
    assert ev == [("slot", sa), ("wait", sb), ("release", sa), ("slot", sb), (second, sb)]
    names = [n for n, _ in log]
    assert names == ["A"] * len(HONEST) + ["B"] * len(HONEST), "B's first request reached its challenger after A's verdict"


def test_the_relay_serves_workers_over_tcp_and_the_record_prints(tmp_path, capsys):
    """`relay` and the CLI's ``record``: three sessions of one statement through a TCP relay to a TCP challenger (the first
    cut before its first request, so the second is served; the third a repeat, refused, and the recursive session's stop at
    the next statement), each outcome passed to ``done``, and every event printed in order."""
    fw, sid = _opened(tmp_path)
    log: list = []
    chal = Challenger(log, "L0")
    srv = socket.create_server(("127.0.0.1", 0))
    addr = "127.0.0.1:%d" % srv.getsockname()[1]

    def challenger():
        c, _ = srv.accept()
        chal._serve(c)
        srv.close()

    tc = threading.Thread(target=challenger)
    tc.start()
    bound, outs = threading.Event(), []
    got: dict = {}

    def run():
        got["res"] = FW.relay(fw, sid, "L0", "127.0.0.1:0", addr, sessions=3, ready=lambda a: (got.update(addr=a), bound.set()),
                              done=outs.append)

    tr = threading.Thread(target=run)
    tr.start()
    bound.wait()
    for reqs in ([], HONEST, HONEST):
        w = G._Wire(got["addr"])
        for r in reqs:
            if RL.decode_resp(w.handle(r))[0] == RL.RESP_VERDICT:
                break
        w.close()
    tr.join()
    tc.join()
    assert [r["outcome"] for r in got["res"]] == ["none", "release", "refused"] and len(outs) == 3
    assert [t for _, t in log] == [r[0] for r in HONEST]
    assert FW.main(["record", "--state", str(tmp_path / "fw")]) == 0
    printed = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    # the third session may ask before the second's thread has left the slot (its verdict is sent first): then it waits
    assert [e for e in printed if e != "wait"] == ["open", "finish", "slot", "release", "slot", "stop", "refused"]
