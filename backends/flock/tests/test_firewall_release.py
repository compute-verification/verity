"""The firewall's release side (`verity_flock.firewall`, live PROTOCOL.md §10.5), one rule a test. A recursive session ends
at its first stop, inner or outer: its stop is its one failure, and the firewall serves nothing more of it, so a later V*
statement's outer session is refused before anything reaches the challenger. Outer proofs go one at a time across streams:
a second requester waits for the first's release or failure, and the record shows the order. The worker and the challenger
here are the test's, on socket pairs; the clock is never read."""
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
    sid = fw.open(stream)
    fw.finish(sid)
    return fw, sid


def _events(fw):
    return [(e["event"], e.get("part")) for e in fw.record()]


def test_an_honest_recursive_session_is_released_statement_by_statement(tmp_path):
    fw, sid = _opened(tmp_path)
    log: list = []
    for part in ("L0", "L1", "alg"):
        out, resp = _outer(fw, sid, part, HONEST, Challenger(log, part))
        assert out["outcome"] == "release" and json.loads(resp[-1][1])["accepted"] is True
    assert _events(fw) == [("open", None), ("finish", "inner")] + [(e, p) for p in ("L0", "L1", "alg") for e in ("slot", "release")]
    assert [t for _, t in log] == [r[0] for r in HONEST] * 3 and fw.failures("pouw") == 0


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
    chal = Challenger(log, "L1")
    if how == "fault":
        real = chal.connect

        def failing():
            r, w, close = real()
            r.close()
            return r, w, close
        chal.connect = failing
    _outer(fw, sid, "L1", REFUSED if how == "refused" else HONEST, chal, cut_after=3 if how == "cut" else None)
    assert RL.REQ_FINISH not in [t for _, t in log] and log[-1] == ("L1", "cut")
    assert fw.stopped(sid)["part"] == "L1"


def test_an_outer_session_is_served_only_after_its_inner_session_finished(tmp_path):
    """A refusal of a recursive session that had not stopped is one of its stream's failures (rule 4); one that was never
    opened is on no stream."""
    fw = FW.Firewall(tmp_path / "fw")
    log: list = []
    out, _ = _outer(fw, "r0000", "L0", HONEST, Challenger(log, "L0"))
    assert out == {"outcome": "refused", "why": "FIREWALL-REFUSED: recursive session r0000 was never opened"}
    sid = fw.open("pouw")
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
            return fw.open(op[1], op[2], op[3])
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
    ``record.jsonl`` exactly: rules 1 and 2 (with a slot holder that died), the budget, the schedule, and strings Python
    escapes. ``flock_live::release``'s ``the_shared_vectors_hold`` runs the same file, so the two firewalls read and write
    one record."""
    v = json.loads(VECTORS.read_text())
    assert v["kind"] == "firewall/v0/vectors" and {c["name"] for c in v["cases"]} == {"rules-1-2", "budget", "schedule", "escapes"}
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
    a, b = fw.open("pouw", budget=1), fw.open("pouw", budget=1)
    for sid in (a, b):
        fw.finish(sid)
    with pytest.raises(ValueError, match="stream pouw's budget is 1, not 2"):
        fw.open("pouw", budget=2)
    assert _outer(fw, a, "L0", REFUSED, Challenger(log, "L0"))[0]["outcome"] == "stop"
    out, _ = _outer(fw, b, "L0", HONEST, Challenger(log, "b"))
    assert out["why"] == "FIREWALL-REFUSED: stream pouw stopped at its failure budget (1 of 1)" and fw.failures("pouw") == 1
    with pytest.raises(FW.Refused, match=r"stream pouw stopped at its failure budget \(1 of 1\)"):
        fw.open("pouw", budget=1)
    s = fw.open("other", release={"start_ms": 0, "period_ms": 1000, "parts": ["L0"]})
    fw.finish(s)
    out, _ = _outer(fw, s, "L0", HONEST, Challenger(log, "s"))
    assert out["why"].endswith("its release schedule needs the Rust firewall (flock-circuit prove --firewall-state)")
    assert not any(name in ("b", "s") for name, _ in log)


def _serve(tmp_path, sessions, fw):
    bound, got = threading.Event(), {}

    def run():
        got["res"] = RL.serve("127.0.0.1:0", tmp_path / "inner", T._counter_rand(), sessions=sessions, schedule=G._schedule(),
                              ready=lambda a: (got.update(addr=a), bound.set()), firewall=fw, stream="pouw")

    t = threading.Thread(target=run)
    t.start()
    bound.wait()
    return t, got


def test_an_inner_stop_ends_the_recursive_session(tmp_path):
    """`rec_live.serve` opens a recursive session for each session it serves: a refused inner session (here Link's y of the
    wrong length) is that recursive session's stop, and no V* statement of it is then served; a finished one lets them be."""
    fw = FW.Firewall(tmp_path / "fw")
    t, got = _serve(tmp_path, 2, fw)
    w = G._Wire(got["addr"])
    assert not G._held(G._run(w, y=bytes(48)))
    w.close()
    w = G._Wire(got["addr"])
    assert G._held(G._run(w))
    w.close()
    t.join()
    ids = [json.loads((tmp_path / "inner" / f"s{k}" / "firewall.json").read_text())["session"] for k in range(2)]
    assert ids == [r["firewall"] for r in got["res"]] == ["r0000", "r0001"]
    assert fw.stopped("r0000")["part"] == "inner" and fw.stopped("r0001") is None
    log: list = []
    out, _ = _outer(fw, "r0000", "L0", HONEST, Challenger(log, "L0"))
    assert out["why"] == "FIREWALL-REFUSED: recursive session r0000 stopped at inner: the firewall serves nothing more of it" and log == []
    assert _outer(fw, "r0001", "L0", HONEST, Challenger(log, "L0"))[0]["outcome"] == "release"
    assert fw.failures("pouw") == 1


def test_a_stop_is_recorded_once():
    """A recursive session has one failure however many of its parts report a stop after the first."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        fw = FW.Firewall(Path(d))
        sid = fw.open("pous")
        assert fw.stop(sid, "inner", "a") and not fw.stop(sid, "L0", "b")
        assert fw.stopped(sid) == {"seq": 1, "event": "stop", "session": sid, "part": "inner", "why": "a"} and fw.failures("pous") == 1


@pytest.mark.parametrize("second", ["release", "stop"])
def test_outer_proofs_go_one_at_a_time_across_streams(tmp_path, second):
    """Two requesters on two streams (each with its own `Firewall` on the shared state, as two processes have): the second
    asks while the first holds the slot, waits, and is served only after the first's outcome; the record shows wait, then
    the first's outcome, then the second's slot. Nothing of the second reaches its challenger before the first's verdict."""
    fa, fb = FW.Firewall(tmp_path / "fw"), FW.Firewall(tmp_path / "fw")
    sa, sb = fa.open("pouw"), fb.open("pous")
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
    """`relay` and the CLI's ``record``: two sessions of one statement through a TCP relay to a TCP challenger (the second
    session a stop), each outcome passed to ``done``, and every event printed in order."""
    fw, sid = _opened(tmp_path)
    log: list = []
    chal = Challenger(log, "L0")
    srv = socket.create_server(("127.0.0.1", 0))
    addr = "127.0.0.1:%d" % srv.getsockname()[1]

    def challenger():
        for _ in range(2):
            c, _ = srv.accept()
            chal._serve(c)
        srv.close()

    tc = threading.Thread(target=challenger)
    tc.start()
    bound, outs = threading.Event(), []
    got: dict = {}

    def run():
        got["res"] = FW.relay(fw, sid, "L0", "127.0.0.1:0", addr, sessions=2, ready=lambda a: (got.update(addr=a), bound.set()),
                              done=outs.append)

    tr = threading.Thread(target=run)
    tr.start()
    bound.wait()
    for reqs in (HONEST, REFUSED):
        w = G._Wire(got["addr"])
        for r in reqs:
            if RL.decode_resp(w.handle(r))[0] == RL.RESP_VERDICT:
                break
        w.close()
    tr.join()
    tc.join()
    assert [r["outcome"] for r in got["res"]] == ["release", "stop"] and len(outs) == 2
    assert FW.main(["record", "--state", str(tmp_path / "fw")]) == 0
    printed = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    # the second session may ask before the first's thread has left the slot (its verdict is sent first): then it waits
    assert [e for e in printed if e != "wait"] == ["open", "finish", "slot", "release", "slot", "stop"]
