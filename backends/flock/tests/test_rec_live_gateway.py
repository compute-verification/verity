"""The inner phase's gateway (`verity_flock.rec_live`) held to what live PROTOCOL.md §10.3 says it does, one rule a test:
``Link``'s ``y`` is the schedule's number of zero elements, and the coin server is shown the schedule's ``Hello``, once the
prover's is the same canonical JSON, and the record lists streams by name, whatever order the prover opened them in."""
from __future__ import annotations

import json
import socket
import threading

import pytest
from verity_flock import rec_live as RL

import test_rec_live as T

HELLO = {"profile": "fast100", "reps": 2, "flavor": "rs", "tables": ["circuit"],
         "link": {"sigma": "ab" * 64, "m": 4, "points": 2}, "rounds": "bytes retained", "digests": "sha512"}


def _schedule(streams=("circuit/rep0",)):
    """The schedule of `_run`'s honest session: one binding round and one message round per stream."""
    return {"kind": RL.KIND + "/schedule", "hello": {**HELLO, "link": {"m": 4, "points": 2}}, "roots": {"circuit": 2},
            "link": {"y": 2}, "streams": {s: {"parent": "", "at": 0, "rounds": 2} for s in streams},
            "rounds": [r for s in streams for r in ({"stream": s, "k": 0, "n": 6, "caps": [2], "commits": 0},
                                                    {"stream": s, "k": 1, "n": 1, "caps": [], "commits": 1})]}


def _proxy(schedule):
    return RL.Proxy(RL.CoinServer(T._counter_rand()), T._salts(), schedule=schedule)


def _run(proxy, hello=None, y=bytes(32), opens=("circuit/rep0",), streams=("circuit/rep0",)):
    """`test_rec_live._session`'s requests, with the Hello (an object, or its text), Link's y and the Open order the test
    picks; one binding round and one message round per stream in `streams`."""
    hello = hello if isinstance(hello, str) else json.dumps(HELLO if hello is None else hello, separators=(",", ":"))
    reqs = [RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, hello),
            RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", T.CAP0)], []), RL.encode_req(RL.REQ_LINK, y)]
    reqs += [RL.encode_req(RL.REQ_OPEN, s, "", s.replace("circuit", "flock-circuit/fast100").encode(), 0) for s in opens]
    for s in streams:
        reqs += [RL.encode_req(RL.REQ_ROUND, s, T._binding_round(), 6),
                 RL.encode_req(RL.REQ_ROUND, s, T._obs([5, 6, 7]) + T._sq(1, RL.KIND_SCALAR), 1)]
    reqs += [RL.encode_req(RL.REQ_PROOF, "circuit", 0, b"the proof"), RL.encode_req(RL.REQ_FINISH)]
    return [RL.decode_resp(proxy.handle(r)) for r in reqs]


def _held(resp):
    return all(r[0] != RL.RESP_REFUSED for r in resp) and resp[-1][0] == RL.RESP_VERDICT


def _refusal(resp):
    return next(r[1] for r in resp if r[0] == RL.RESP_REFUSED)


def test_an_honest_session_is_held_to_its_schedule():
    sched = _schedule()
    proxy = _proxy(sched)
    assert _held(_run(proxy)) and RL.schedule_of(proxy.coins.record()) == sched


@pytest.mark.parametrize("n", [0, 16, 31, 33, 48, 16000])
def test_link_y_of_another_length_is_refused_before_the_coin_server_sees_it(n):
    """``y`` of zeros passes ``any(y)`` whatever its length, so the length is the schedule's (2 elements here), and a ``y``
    that is not whole elements is refused as well."""
    proxy = _proxy(_schedule())
    assert _refusal(_run(proxy, y=bytes(n))).startswith("rec proxy: GATE-REFUSED: Link's y: ")
    assert proxy.coins.link["y"] is None and proxy.coins.rounds == []


def test_the_coin_server_records_the_pinned_hello_never_the_provers_spelling():
    """A Hello in another key order (or other whitespace) is held, and the coin server records the schedule's, in canonical
    form; key order alone was log2(7! 2!) free bits a session."""
    sched = _schedule()
    spelled = {k: HELLO[k] for k in reversed(list(HELLO))}
    spelled["link"] = {"points": 2, "m": 4, "sigma": "ab" * 64}
    for hello in (spelled, json.dumps(spelled, indent=3)):
        proxy = _proxy(sched)
        assert _held(_run(proxy, hello=hello))
        rec = proxy.coins.record()["hello"]
        assert json.dumps(rec, separators=(",", ":")) == RL.canonical(sched["hello"]) and rec == sched["hello"]


@pytest.mark.parametrize("edit", [{"reps": 2.0}, {"link": {"sigma": "ab" * 64, "m": 4, "points": 2.0}},
                                  {"link": {"sigma": "ab" * 64, "m": 4.0, "points": 2}},
                                  {"tables": ["circuit", "circuit"]}, {"note": "5e" * 64}, {"link": {"m": 4, "points": 2}},
                                  {"link": {"sigma": 7, "m": 4, "points": 2}}])
def test_a_hello_whose_canonical_json_is_not_the_pinned_ones_is_refused(edit):
    """``==`` holds ``2.0`` equal to ``2``; the canonical JSON does not. A Hello without Σ as a string is refused too (the
    session record needs it, the coin server never sees it)."""
    proxy = _proxy(_schedule())
    assert _refusal(_run(proxy, hello={**HELLO, **edit})).startswith("rec proxy: GATE-REFUSED: Hello: ")
    assert proxy.coins.hello is None and proxy.coins.link is None


def test_the_record_lists_streams_in_the_schedules_order_whatever_order_they_were_opened_in():
    """Each stream is pinned by its parent and position, not by when it is opened, so the prover may open rep1 first; the
    coin server's record is the same either way (log2(s!) free bits with s streams, otherwise)."""
    two = ("circuit/rep0", "circuit/rep1")
    recs = []
    for opens in (two, two[::-1]):
        proxy = _proxy(_schedule(two))
        assert _held(_run(proxy, opens=opens, streams=two))
        recs.append(json.dumps(proxy.coins.record()))
    assert recs[0] == recs[1] and list(json.loads(recs[0])["streams"]) == list(two)


def _serve(tmp_path, sessions, schedule):
    """`rec_live.serve` on a thread; its result (or what ended it) in ``got``, the address it bound in ``got["addr"]``."""
    bound, got = threading.Event(), {}

    def run():
        try:
            got["res"] = RL.serve("127.0.0.1:0", tmp_path, T._counter_rand(), sessions=sessions, schedule=schedule,
                                  ready=lambda a: (got.update(addr=a), bound.set()))
        except Exception as e:  # noqa: BLE001 - the test reports what ended serve
            got["crash"] = f"{type(e).__name__}: {e}"
        finally:
            bound.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    bound.wait()
    return t, got


class _Wire:
    """A prover's connection to `serve`: `_run` drives it as it drives a `Proxy`."""

    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.rf, self.wf = self.sock.makefile("rb"), self.sock.makefile("wb")

    def handle(self, req):
        RL.write_frame(self.wf, req)
        return RL.read_frame(self.rf)

    def close(self):
        self.rf.close()
        self.wf.close()
        self.sock.close()


def test_a_session_that_ends_without_finish_is_written_and_counted_as_a_stop(tmp_path):
    """A connection that closes after drawing coins (here after its binding round), and one cut inside a frame, are each a
    session that stopped: the coin server's record of it is written and says so, and it counts. Otherwise the prover could
    draw coins, look at them and drop the connection as often as it likes, and the record would never show it."""
    t, got = _serve(tmp_path, 3, _schedule())
    w = _Wire(got["addr"])
    drew = [RL.decode_resp(w.handle(r))[0] for r in (
        RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, json.dumps(HELLO)),
        RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", T.CAP0)], []), RL.encode_req(RL.REQ_LINK, bytes(32)),
        RL.encode_req(RL.REQ_OPEN, "circuit/rep0", "", b"flock-circuit/fast100/rep0", 0),
        RL.encode_req(RL.REQ_ROUND, "circuit/rep0", T._binding_round(), 6))]
    w.close()
    w = _Wire(got["addr"])
    w.handle(RL.encode_req(RL.REQ_REGISTER))
    w.wf.write(b"\x40\x00")
    w.close()
    w = _Wire(got["addr"])
    assert _held(_run(w))
    w.close()
    t.join()
    assert drew == [RL.RESP_DRAW, RL.RESP_OK, RL.RESP_COINS, RL.RESP_OK, RL.RESP_OK, RL.RESP_COINS] and "crash" not in got
    res = got["res"]
    assert [r["session"] for r in res] == [None, None, str(tmp_path / "s2" / "session")]
    assert [(r["refused"], r["rounds"]) for r in res[:2]] == [("rec proxy: the session ended without Finish", 1),
                                                              ("rec proxy: the session ended without Finish", 0)]
    pubs = [json.loads((tmp_path / f"s{k}" / "public.json").read_text()) for k in range(3)]
    assert [p["end"] for p in pubs] == ["stop", "stop", "finish"] and [len(p["rounds"]) for p in pubs] == [1, 0, 2]
    assert not (tmp_path / "s0" / "session").exists() and not (tmp_path / "s1" / "session").exists()


def test_a_refused_session_is_recorded_as_a_stop():
    proxy = _proxy(_schedule())
    resp = _run(proxy, y=bytes(48))
    assert resp[-1][0] == RL.RESP_VERDICT and json.loads(resp[-1][1])["accepted"] is False
    assert proxy.coins.record()["end"] == "stop"
    proxy = _proxy(_schedule())
    assert _held(_run(proxy)) and proxy.coins.record()["end"] == "finish"
