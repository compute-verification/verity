"""The inner phase's gateway (`verity_flock.rec_live`) held to what live PROTOCOL.md §10.3 says it does, one rule a test:
``Link``'s ``y`` is the schedule's number of zero elements."""
from __future__ import annotations

import json

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
