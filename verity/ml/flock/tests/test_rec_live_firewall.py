"""The inner phase's ZK server (`verity_flock.rec_live`) held to what live PROTOCOL.md §10.3 says it does, one rule a test:
``Link``'s ``y`` is the schedule's number of zero elements; the coin server is shown the schedule's ``Hello``, once the
prover's is the same canonical JSON; the record lists streams by name, whatever order the prover opened them in; a session
that ends without ``Finish`` is a stop, written and counted; any exception while handling a request is a refusal; and the
schedule is required, derived from the statement's public shape (`schedule_for`), never from a session the prover ran."""
from __future__ import annotations

import json
import re
import socket
import threading
from pathlib import Path

import pytest
from verity_flock import rec_live as RL

import test_rec_live as T

HELLO = {"profile": "fast100", "reps": 2, "flavor": "rs", "tables": ["circuit"],
         "link": {"sigma": "ab" * 64, "m": 4, "points": 2}, "rounds": "bytes retained", "digests": "sha512"}


def _schedule(streams=("circuit/rep0",)):
    """The schedule of `_run`'s honest session: one binding round and one message round per stream."""
    return {"kind": RL.KIND + "/schedule", "hello": {**HELLO, "link": {"m": 4, "points": 2}}, "roots": {"circuit": 2},
            "link": {"y": 2}, "streams": {s: {"parent": "", "at": 0, "rounds": 2} for s in streams},
            "rounds": [r for s in streams for r in ({"stream": s, "k": 0, "n": 6, "tops": [], "level0": 1, "commits": 0},
                                                    {"stream": s, "k": 1, "n": 1, "tops": [], "level0": 0, "commits": 1})]}


def _proxy(schedule, coins=None):
    """A proxy held to ``schedule``, on live coins."""
    return RL.Proxy(coins or RL.CoinServer(T._counter_rand()), T._salts(), schedule=schedule)


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


def _extra(rec):
    rec["rounds"].append({**rec["rounds"][-1], "k": 2})
    rec["streams"]["circuit/rep0"]["rounds"] = 3


def _coins(rec):
    rec["rounds"][1]["n"], rec["rounds"][1]["coins"] = 2, rec["rounds"][1]["coins"] * 2


@pytest.mark.parametrize("change,why", [
    (_extra, 'the record declares the streams {"circuit/rep0":{"at":0,"parent":"","rounds":3}}'),
    (lambda rec: rec["rounds"].pop(), "1 rounds, and the schedule's 2"),
    (_coins, 'round 1 is {"commits":1,"k":1,"level0":0,"n":2,"stream":"circuit/rep0","tops":[]} of 2 coins'),
    (lambda rec: rec["rounds"][1]["coins"].append(rec["rounds"][1]["coins"][0]), "round 1 is "),
    (lambda rec: rec["rounds"][0].__setitem__("level0", 0), 'round 0 is {"commits":0,"k":0,"level0":0,'),
], ids=["extra-round", "missing-round", "coin-count", "coins-not-n", "level0"])
def test_a_finished_record_off_its_schedule_is_refused(change, why):
    """`on_schedule`, the verifier's `RecVStar.checkSchedule`: an honest session's record is on its schedule. A record
    that ends ``finish`` and is off it, as a session that bypassed the proxy leaves it (an extra round, a missing round,
    a round of another coin count, coins other than its ``n``, a round of another ``level0``), is refused."""
    sched = _schedule()
    proxy = _proxy(sched)
    assert _held(_run(proxy))
    rec = proxy.coins.record()
    assert rec["end"] == "finish"
    RL.on_schedule(rec, sched)
    change(rec)
    with pytest.raises(ValueError, match=re.escape(why)):
        RL.on_schedule(rec, sched)


@pytest.mark.parametrize("n", [0, 16, 31, 33, 48, 16000])
def test_link_y_of_another_length_is_refused_before_the_coin_server_sees_it(n):
    """``y`` of zeros passes ``any(y)`` whatever its length, so the length is the schedule's (2 elements here), and a ``y``
    that is not whole elements is refused as well."""
    proxy = _proxy(_schedule())
    assert _refusal(_run(proxy, y=bytes(n))).startswith("rec proxy: FIREWALL-REFUSED: Link's y: ")
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
    assert _refusal(_run(proxy, hello={**HELLO, **edit})).startswith("rec proxy: FIREWALL-REFUSED: Hello: ")
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


def _stopped_record(reqs, then_stop=b"not a request"):
    """The coin server's record of a session sent `reqs` and then a request the ZK server refuses."""
    proxy = _proxy(_schedule())
    for r in reqs + [then_stop]:
        proxy.handle(r)
    return proxy.coins.record()


def test_a_stopped_record_does_not_show_whether_or_when_an_open_was_sent():
    """An Open is accepted any time before its stream's first round, so a stream enters the record only at that round;
    otherwise two sessions stopped at the same point differ in whether the Open came (the review's probe, r2)."""
    reg, hello = RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, json.dumps(HELLO, separators=(",", ":")))
    commit = RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", T.CAP0)], [])
    link = RL.encode_req(RL.REQ_LINK, bytes(32))
    opn = RL.encode_req(RL.REQ_OPEN, "circuit/rep0", "", b"flock-circuit/fast100/rep0", 0)
    for prefix in ([reg, hello], [reg, hello, commit, link]):
        recs = [_stopped_record(reqs) for reqs in (prefix, [opn] + prefix, prefix + [opn])]
        assert recs[0]["end"] == "stop" and recs[0]["streams"] == {}
        assert recs[1] == recs[0] and recs[2] == recs[0]
    proxy = _proxy(_schedule())
    assert _held(_run(proxy)) and list(proxy.coins.record()["streams"]) == ["circuit/rep0"]


def _coin_calls(reqs, schedule):
    """Every call the ZK server makes on the coin server while it handles `reqs`, in order, the ones it refuses included."""
    coins, calls = RL.CoinServer(T._counter_rand()), []
    for name in ("start", "commit", "y", "open", "round", "record"):
        f = getattr(coins, name)
        setattr(coins, name, lambda *a, _f=f, _n=name, **kw: (calls.append((_n, a, kw)), _f(*a, **kw))[1])
    proxy = RL.Proxy(coins, T._salts(), schedule=schedule)
    for r in reqs:
        proxy.handle(r)
    return calls


def test_the_coin_server_never_sees_where_an_open_was_sent():
    """The coin server is the auditor's, so its calls, not only its record, must not show where the prover sent each Open
    (ZK server-contract's gap 3): every placement of the two Opens before their first rounds, finished or cut anywhere,
    gives the coin server the honest session's calls or a prefix of them."""
    two = ("circuit/rep0", "circuit/rep1")
    sched = _schedule(two)
    opn = {s: RL.encode_req(RL.REQ_OPEN, s, "", s.replace("circuit", "flock-circuit/fast100").encode(), 0) for s in two}
    base = [RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, json.dumps(HELLO, separators=(",", ":"))),
            RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", T.CAP0)], []), RL.encode_req(RL.REQ_LINK, bytes(32))]
    first = {}
    for s in two:
        first[s] = len(base)
        base += [RL.encode_req(RL.REQ_ROUND, s, T._binding_round(), 6),
                 RL.encode_req(RL.REQ_ROUND, s, T._obs([5, 6, 7]) + T._sq(1, RL.KIND_SCALAR), 1)]
    base += [RL.encode_req(RL.REQ_PROOF, "circuit", 0, b"the proof"), RL.encode_req(RL.REQ_FINISH)]
    honest = _coin_calls(base[:2] + [opn[two[0]]] + base[2:first[two[1]]] + [opn[two[1]]] + base[first[two[1]]:], sched)
    assert [c[0] for c in honest].count("open") == 2
    sessions = 0
    for i in range(first[two[0]] + 1):
        for j in range(first[two[1]] + 1):
            for order in (two, two[::-1]):
                at = {two[0]: i, two[1]: j}
                reqs = []
                for p, r in enumerate(base + [None]):
                    reqs += [opn[s] for s in order if at[s] == p]
                    if r is not None:
                        reqs.append(r)
                assert _coin_calls(reqs, sched) == honest
                for cut in range(len(reqs)):
                    calls = _coin_calls(reqs[:cut] + [b"not a request"], sched)
                    assert calls == honest[:len(calls)]
                sessions += 1
    assert sessions == 2 * (first[two[0]] + 1) * (first[two[1]] + 1)


def test_a_stray_commit_link_or_finish_reaches_the_coin_server_as_a_plain_stop():
    """A Commit before Hello or after the first, a Link before Commit or after the first, and a Finish before the last
    round are refused by the ZK server itself, before any call on the coin server, so wherever one comes the coin server's
    calls are a prefix of the honest session's: they show where the session stopped, never which request stopped it
    (review-zk-gateway's r4)."""
    two = ("circuit/rep0", "circuit/rep1")
    sched = _schedule(two)
    reg, hello = RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, json.dumps(HELLO, separators=(",", ":")))
    commit = RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", T.CAP0)], [])
    link, finish = RL.encode_req(RL.REQ_LINK, bytes(32)), RL.encode_req(RL.REQ_FINISH)
    base = [reg, hello, commit, link]
    for s in two:
        base += [RL.encode_req(RL.REQ_OPEN, s, "", s.replace("circuit", "flock-circuit/fast100").encode(), 0),
                 RL.encode_req(RL.REQ_ROUND, s, T._binding_round(), 6),
                 RL.encode_req(RL.REQ_ROUND, s, T._obs([5, 6, 7]) + T._sq(1, RL.KIND_SCALAR), 1)]
    base += [RL.encode_req(RL.REQ_PROOF, "circuit", 0, b"the proof"), finish]
    honest = _coin_calls(base, sched)
    assert [c[0] for c in honest] == ["start", "commit", "y"] + ["open", "round", "round"] * 2
    for stray in (commit, link, finish):
        for p in range(len(base)):
            calls = _coin_calls(base[:p] + [stray] + base[p:], sched)
            assert calls == honest[:len(calls)], (stray[:1], p)


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
    """A connection that closes after flipping coins (here after its binding round), and one cut inside a frame, are each a
    session that stopped: the coin server's record of it is written and says so, and it counts. Otherwise the prover could
    flip coins, look at them and drop the connection as often as it likes, and the record would never show it."""
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


@pytest.mark.parametrize("hello", ["[]", "7", '"x"', "{", "[" * 100000, '{"link": []}'],
                         ids=["list", "number", "string", "not-json", "nested", "link-list"])
def test_a_hello_the_firewall_cannot_read_is_a_refusal(hello):
    """Not an object, not JSON, or nested past what the parser takes (RecursionError, not a ValueError)."""
    proxy = _proxy(_schedule())
    resp = _run(proxy, hello=hello)
    assert _refusal(resp).startswith("rec proxy: ") and resp[-1][0] == RL.RESP_VERDICT
    assert proxy.coins.hello is None and proxy.coins.record()["end"] == "stop"


def test_any_exception_while_handling_a_request_is_a_refusal_and_serve_goes_on(tmp_path):
    """The ZK server's own fault (here the coin server's, a KeyError) stops the session it happened in, as a refusal; and a
    request that raises something other than ValueError ends neither `Proxy.handle` nor `serve`."""

    class Broken(RL.CoinServer):
        def round(self, *a, **kw):
            raise KeyError("no such coin")

    proxy = _proxy(_schedule(), Broken(T._counter_rand()))
    why = _refusal(_run(proxy))
    assert why == "rec proxy: FIREWALL-REFUSED: a request the firewall cannot read (KeyError: 'no such coin')"
    assert proxy.coins.record()["end"] == "stop"
    t, got = _serve(tmp_path, 2, _schedule())
    w = _Wire(got["addr"])
    first = _run(w, hello="[" * 100000)
    w.close()
    w = _Wire(got["addr"])
    assert _held(_run(w))
    w.close()
    t.join()
    assert "RecursionError" in _refusal(first) and "crash" not in got
    assert [r["session"] for r in got["res"]] == [None, str(tmp_path / "s1" / "session")]
    assert json.loads((tmp_path / "s0" / "public.json").read_text())["end"] == "stop"


def test_a_refused_session_is_recorded_as_a_stop():
    proxy = _proxy(_schedule())
    resp = _run(proxy, y=bytes(48))
    assert resp[-1][0] == RL.RESP_VERDICT and json.loads(resp[-1][1])["accepted"] is False
    assert proxy.coins.record()["end"] == "stop"
    proxy = _proxy(_schedule())
    assert _held(_run(proxy)) and proxy.coins.record()["end"] == "finish"


def _edited(edit):
    s = json.loads(json.dumps(_schedule()))
    edit(s)
    return s


@pytest.mark.parametrize("schedule", [
    None, {}, [], _edited(lambda s: s.update(kind=RL.KIND)), _edited(lambda s: s["hello"]["link"].update(sigma="ab" * 64)),
    _edited(lambda s: s["link"].update(y=2.0)), _edited(lambda s: s["link"].update(y=True)), _edited(lambda s: s["roots"].update(circuit=-1)),
    _edited(lambda s: s["rounds"][0].update(stream="circuit/rep1")), _edited(lambda s: s["rounds"][0].update(tops=["2"])),
    _edited(lambda s: s["rounds"][0].update(level0=True)), _edited(lambda s: s.pop("rounds"))],
    ids=["none", "empty", "list", "kind", "sigma", "y-float", "y-bool", "root-negative", "stream-unknown", "top-string", "level0-bool",
         "no-rounds"])
def test_a_schedule_that_is_not_one_is_refused_before_anything_is_served(tmp_path, schedule):
    with pytest.raises(ValueError, match=f"not a {RL.KIND}/schedule"):
        _proxy(schedule)
    with pytest.raises(ValueError, match=f"not a {RL.KIND}/schedule"):
        RL.serve("127.0.0.1:0", tmp_path, schedule=schedule)
    assert list(tmp_path.iterdir()) == []


def test_the_firewall_serves_only_with_a_schedule(tmp_path, capsys):
    """No first session pins one: `Proxy` and `serve` take the schedule as a required keyword, and `main` will not serve
    without ``--schedule``."""
    with pytest.raises(TypeError):
        RL.Proxy(RL.CoinServer(T._counter_rand()), T._salts())
    with pytest.raises(TypeError):
        RL.serve("127.0.0.1:0", tmp_path)
    with pytest.raises(SystemExit) as e:
        RL.main(["--out", str(tmp_path)])
    assert e.value.code == 2 and "--schedule are required" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


def test_the_first_session_is_held_to_the_schedule_serve_was_given(tmp_path):
    """`serve` writes its schedule before it accepts a connection, so the first session pins nothing: a Hello field of the
    prover's choosing (64 bytes of hex) is refused, and reaches neither a record nor ``schedule.json``."""
    sched = _schedule()
    t, got = _serve(tmp_path, 2, sched)
    assert json.loads((tmp_path / "schedule.json").read_text()) == sched
    secret = "5e" * 64
    w = _Wire(got["addr"])
    first = _run(w, hello={**HELLO, "note": secret})
    w.close()
    w = _Wire(got["addr"])
    assert _held(_run(w))
    w.close()
    t.join()
    assert _refusal(first).startswith("rec proxy: FIREWALL-REFUSED: Hello: ") and "crash" not in got
    assert [r["session"] for r in got["res"]] == [None, str(tmp_path / "s1" / "session")]
    assert json.loads((tmp_path / "schedule.json").read_text()) == sched
    assert not any(secret in p.read_text() for p in tmp_path.rglob("*.json"))


FLAT = Path(__file__).resolve().parent / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"


def test_the_schedule_of_a_statements_shape_is_the_one_a_real_session_keeps():
    """`schedule_for` from the shape alone (m = 25, k_log = 22, nbl = 3, three regions) is the flat GEMM selftest's honest
    session (the fetched fixture `test_rec_algebra` reads), round for round: coins, tops' node counts, level-0 copies, message rows. That
    session sampled committed coins, so its Hello also names them; the inner session's, on live OS coins, does not."""
    assert (FLAT / "records-cpu" / "honest.json").is_file(), (
        f"{FLAT} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/hidden-outputs`")
    rec = json.loads((FLAT / "records-cpu" / "honest.json").read_text())["record"]
    want = RL.schedule_for({"dense_m": 25, "k_log": 22, "nbl": 3, "regions": 3})
    hello = json.loads(rec["hello"])
    assert hello.pop("coins") == "coin-commit/sha512" and isinstance(hello["link"].pop("sigma"), str)
    assert RL.canonical(hello) == RL.canonical(want["hello"])
    assert {t: len(v) // 2 // RL.CAP_NODE_BYTES for t, v in rec["link"]["root_b"].items()} == want["roots"]
    assert len(rec["link"]["y"]) == want["link"]["y"]
    assert {st["stream"]: {"parent": st["parent"], "at": st["at"], "rounds": len(st["rounds"])} for st in rec["streams"]} == want["streams"]
    level0 = {bytes.fromhex(v) for v in rec["link"]["root_b"].values()}
    rounds = []
    for st in rec["streams"]:
        for k, rd in enumerate(st["rounds"]):
            s = RL.split_round(bytes.fromhex(rd["msg"]), k == 0 and not st["parent"])
            rounds.append({"stream": st["stream"], "k": k, "n": rd["n"], "tops": [len(RL.top_nodes(c)) for c in s.caps if c not in level0],
                           "level0": sum(c in level0 for c in s.caps), "commits": len(RL.message_rows(s.free))})
    assert rounds == want["rounds"] and len(rounds) == 2 * 94


def test_the_schedule_of_the_firewalled_k4096_statement():
    """The K = 4096 statement (``flock-circuit statement``: dense_m 35, k_log 24, nbl 11, three regions) gives the schedule
    its inner sessions keep: link points of 35 coordinates, a 128-node level-0 cap, 139 rounds a rep, 109 of them with a
    message row."""
    text = "statement circuit ...\nSTATEMENT\t" + json.dumps({"statement": "verity/flock-circuit", "k_log": 24, "nbl": 11,
                                                             "dense_m": 35, "regions": 3, "statement_digest": "00" * 64})
    s = RL.schedule_for(RL.statement_of(text))
    assert RL.check_schedule(s) is s and "00" * 64 not in json.dumps(s)
    assert s["hello"]["link"] == {"m": 35, "points": 2} and s["roots"] == {"circuit": 128} and s["link"] == {"y": 2}
    assert s["streams"] == {f"circuit/rep{r}": {"parent": "", "at": 0, "rounds": 139} for r in range(2)}
    assert (sum(r["n"] for r in s["rounds"]), sum(r["commits"] for r in s["rounds"])) == (2082, 218)


@pytest.mark.parametrize("statement", [{}, {"dense_m": 35, "k_log": 24, "nbl": 11}, {"dense_m": 35.0, "k_log": 24, "nbl": 11, "regions": 3},
                                       {"dense_m": 35, "k_log": True, "nbl": 11, "regions": 3}, {"dense_m": 35, "k_log": 24, "nbl": -1, "regions": 3}],
                         ids=["empty", "no-regions", "m-float", "k-bool", "nbl-negative"])
def test_a_statement_without_its_shape_has_no_schedule(statement):
    with pytest.raises(ValueError, match="names dense_m, k_log, nbl and regions"):
        RL.schedule_for(statement)
