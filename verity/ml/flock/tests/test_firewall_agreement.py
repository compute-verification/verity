"""The firewall's contract, inner phase: `rec_live`'s proxy sends the coin server exactly what `Flock.Firewall.run` (the Lean
program `flock-firewall`) sends, byte for byte, on an honest session and on every session a worker can make of it here (each
schedule edit of `test_rec_live`, every prefix, drop, duplicate and swap of a request, requests after ``Finish``, forged
requests, and seeded random byte flips). The Lean program's type is the contract (live PROTOCOL.md §10.3): each message is the
schedule's own value or a commitment under one of the firewall's salts, so agreement means nothing else leaves the proxy."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_live as RL

from test_rec_live import CAP0, CAP1, HELLO, SCHEDULE, _binding_round, _bytes, _content, _domain, _hdr, _label, _obs, _sq

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-firewall"
SALTS = [hashlib.shake_256(b"firewall-salt" + i.to_bytes(8, "little")).digest(192) for i in range(64)]


def _coins(n: int) -> bytes:
    return hashlib.shake_256(b"firewall-coins" + n.to_bytes(8, "little")).digest(n)


def honest() -> list[bytes]:
    """`test_rec_live`'s session: the binding round (level 0's top again), a free message with a fresh top, and Ligerito's
    copy of level 0's top."""
    return [RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, HELLO),
            RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", CAP0)], []),
            RL.encode_req(RL.REQ_LINK, bytes(32)),
            RL.encode_req(RL.REQ_OPEN, "circuit/rep0", "", b"flock-circuit/fast100/rep0", 0),
            RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _binding_round(), 6),
            RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _content()[0], 1),
            RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _content()[1], 2),
            RL.encode_req(RL.REQ_PROOF, "circuit", 0, b"the proof"),
            RL.encode_req(RL.REQ_FINISH)]


#: a schedule with a child stream, so `Open`'s parent and position are pinned too, and a child's round with a fresh top
#: and a message row comes between its parent's
SCHEDULE2 = {**json.loads(json.dumps(SCHEDULE)),
             "streams": {"circuit/rep0": {"parent": "", "at": 0, "rounds": 3}, "circuit/rep0/x": {"parent": "circuit/rep0", "at": 1, "rounds": 1}},
             "rounds": SCHEDULE["rounds"][:2] + [{"stream": "circuit/rep0/x", "k": 0, "n": 3, "tops": [1], "level0": 0, "commits": 1}]
             + SCHEDULE["rounds"][2:]}


def honest2() -> list[bytes]:
    r = honest()
    child = RL.encode_req(RL.REQ_ROUND, "circuit/rep0/x", _bytes(b"\x00" * 8) + _bytes(CAP1) + _obs(list(range(1, 10))) + _sq(3), 3)
    return r[:5] + [RL.encode_req(RL.REQ_OPEN, "circuit/rep0/x", "circuit/rep0", b"x", 1)] + r[5:7] + [child] + r[7:]


#: how `Flock.Firewall.calls` writes each call the proxy makes on the coin server
CALL = {"start": lambda a: ["start"], "commit": lambda a: ["commit"], "y": lambda a: ["y"],
        "open": lambda a: ["open", a[0], a[1], a[2]], "round": lambda a: ["round", a[0], a[3]]}


def proxy_session(schedule: dict, reqs: list[bytes]) -> tuple[dict, list]:
    """The coin server's record and every call made on it, in order, refused or not, while the proxy handles ``reqs``."""
    salts, coins, calls = iter(SALTS), RL.CoinServer(_coins), []
    for name, show in CALL.items():
        f = getattr(coins, name)
        setattr(coins, name, lambda *a, _f=f, _show=show, **kw: (calls.append(_show(a)), _f(*a, **kw))[1])
    proxy = RL.Proxy(coins, lambda: next(salts), schedule=schedule)
    for b in reqs:
        proxy.handle(b)
    if proxy.verdict is None and proxy.stopped is None:
        proxy.stop("rec proxy: the session ended without Finish")
    rec = proxy.coins.record()
    if rec["link"] is not None:
        rec["link"] = {k: v for k, v in rec["link"].items() if k != "points"}
    rec["rounds"] = [{k: v for k, v in r.items() if k != "coins"} for r in rec["rounds"]]
    return rec, calls


def placements() -> list[tuple[str, dict, list[bytes]]]:
    """`honest2`'s session with its two `Open`s moved to every place before their streams' first rounds, in either order,
    finished and cut after every request."""
    h = honest2()
    opn = {"p": h[4], "x": h[5]}
    base = h[:4] + h[6:]
    first = {"p": 4, "x": 6}
    out = []
    for i in range(first["p"] + 1):
        for j in range(first["x"] + 1):
            for order in ("px", "xp"):
                at = {"p": i, "x": j}
                reqs = []
                for q, r in enumerate(base + [None]):
                    reqs += [opn[s] for s in order if at[s] == q]
                    reqs += [r] if r is not None else []
                name = f"Opens at {i} and {j}, {order}"
                out.append((name, SCHEDULE2, reqs))
                out += [(f"{name}, cut at {c}", SCHEDULE2, reqs[:c] + [b"not a request"]) for c in range(len(reqs))]
    return out


def cases() -> list[tuple[str, dict, list[bytes]]]:
    out = [("honest", SCHEDULE, honest()), ("honest, two streams", SCHEDULE2, honest2())]
    for name, sched, reqs in list(out):
        n = len(reqs)
        out += [(f"{name}: the first {i}", sched, reqs[:i]) for i in range(n)]
        out += [(f"{name}: without {i}", sched, reqs[:i] + reqs[i + 1:]) for i in range(n)]
        out += [(f"{name}: {i} twice", sched, reqs[:i + 1] + reqs[i:]) for i in range(n)]
        out += [(f"{name}: {i} and {i + 1} swapped", sched, reqs[:i] + [reqs[i + 1], reqs[i]] + reqs[i + 2:]) for i in range(n - 1)]
        for tag, req in [("Ping", RL.encode_req(RL.REQ_PING, b"")), ("Register", RL.encode_req(RL.REQ_REGISTER)),
                         ("Finish", RL.encode_req(RL.REQ_FINISH)), ("Hello", reqs[1])]:
            out.append((f"{name}: then {tag}", sched, reqs + [req]))
    edits = {"Hello": lambda s: s["hello"].update(reps=1), "roots": lambda s: s["roots"].update(circuit=1),
             "Open": lambda s: s["streams"]["circuit/rep0"].update(at=1), "round 0": lambda s: s["rounds"][0].update(n=5),
             "round 1's tops": lambda s: s["rounds"][1].update(tops=[2]),
             "round 1's commits": lambda s: s["rounds"][1].update(commits=0),
             "round 2's level0": lambda s: s["rounds"][2].update(level0=0),
             "round 2's level-0 copy as a top": lambda s: s["rounds"][2].update(tops=[2], level0=0),
             "rounds": lambda s: s["rounds"].append({**s["rounds"][1], "k": 2}), "y": lambda s: s["link"].update(y=1),
             "an extra key in a round": lambda s: s["rounds"][0].update(extra=0)}
    for what, edit in edits.items():
        s = json.loads(json.dumps(SCHEDULE))
        edit(s)
        out.append((f"schedule's {what} edited", s, honest()))
    h = json.loads(HELLO)
    hellos = {"reps 2.0": HELLO.replace('"reps":2', '"reps":2.0'), "reps 2e0": HELLO.replace('"reps":2', '"reps":2e0'),
              "an extra key": json.dumps({**h, "extra": 1}), "coins": json.dumps({**h, "coins": "seed"}),
              "no sigma": json.dumps({**h, "link": {"m": 4, "points": 2}}), "not JSON": HELLO[:-1],
              "keys reordered": json.dumps(dict(reversed(list(h.items())))), "a list": "[]",
              "reps true": HELLO.replace('"reps":2', '"reps":true'), "spaces": json.dumps(h, indent=1)}
    for what, text in hellos.items():
        r = honest()
        r[1] = RL.encode_req(RL.REQ_HELLO, text)
        out.append((f"Hello: {what}", SCHEDULE, r))
    forged = {
        "Link's y not zero": (3, RL.encode_req(RL.REQ_LINK, bytes(17) + b"\x01" + bytes(14))),
        "Link's y short": (3, RL.encode_req(RL.REQ_LINK, bytes(16))),
        "Commit names a table twice": (2, RL.encode_req(RL.REQ_COMMIT, b"rf", [("circuit", CAP0), ("circuit", CAP0)], [])),
        "Commit with public words": (2, RL.encode_req(RL.REQ_COMMIT, b"rf", [("circuit", CAP0)], [("circuit", b"\x01")])),
        "Commit with an empty public": (2, RL.encode_req(RL.REQ_COMMIT, b"rf", [("circuit", CAP0)], [("circuit", b"")])),
        "Commit's top of 65 bytes": (2, RL.encode_req(RL.REQ_COMMIT, b"rf", [("circuit", CAP0[:65])], [])),
        "Commit's other table": (2, RL.encode_req(RL.REQ_COMMIT, b"rf", [("other", CAP0)], [])),
        "Open off the schedule": (4, RL.encode_req(RL.REQ_OPEN, "circuit/rep1", "", b"l", 0)),
        "round asks for 5": (5, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _binding_round(), 5)),
        "round without a squeeze": (5, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _binding_round()[:-16], 6)),
        "round binds no cap": (5, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _label(b"flock-r1cs-v0") + _bytes(bytes(64)) + _sq(6), 6)),
        "round's cap of 63 bytes": (5, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _label(b"flock-r1cs-v0") + _bytes(bytes(64)) + _bytes(bytes(63)) + _sq(6), 6)),
        "round with an unknown op": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _hdr(9, 0, 0) + _sq(1), 1)),
        "round with a truncated header": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _sq(1) + bytes(8), 1)),
        "round with a scalar of two": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _obs([1, 2], RL.KIND_SCALAR) + _sq(1), 1)),
        "round with a BYTES frame of 9": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _bytes(bytes(9)) + _sq(1, RL.KIND_SCALAR), 1)),
        "round with a nonce": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _bytes(bytes(8)) + _obs([5, 6, 7]) + _sq(1, RL.KIND_SCALAR), 1)),
        "round with a domain": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _domain(b"d") + _obs([5, 6, 7]) + _sq(1, RL.KIND_SCALAR), 1)),
        "round with nine elements": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _obs(list(range(9))) + _sq(1, RL.KIND_SCALAR), 1)),
        "round on an unopened stream": (6, RL.encode_req(RL.REQ_ROUND, "circuit/rep9", _sq(1), 1)),
        "Prime": (7, RL.REQ_PRIME.to_bytes(1, "little") + (0).to_bytes(4, "little") + (0).to_bytes(4, "little")),
        "an unknown tag": (7, bytes([42])),
        "an empty frame": (7, b""),
        "a short frame": (1, RL.encode_req(RL.REQ_HELLO, HELLO)[:-1]),
        "not UTF-8": (1, bytes([RL.REQ_HELLO]) + (2).to_bytes(4, "little") + b"\xff\xfe"),
    }
    for what, (i, req) in forged.items():
        r = honest()
        r[i] = req
        out.append((f"forged: {what}", SCHEDULE, r))
    stop = b"not a request"
    out += [("stopped before Open", SCHEDULE, honest()[:4] + [stop]), ("stopped after Open", SCHEDULE, honest()[:5] + [stop]),
            ("stopped after a round", SCHEDULE, honest()[:6] + [stop]),
            ("two streams: stopped before Open", SCHEDULE2, honest2()[:4] + [stop]),
            ("two streams: stopped after both Opens", SCHEDULE2, honest2()[:6] + [stop]),
            ("two streams: stopped after the parent's round", SCHEDULE2, honest2()[:7] + [stop])]
    rng = random.Random(20261006)
    for k in range(300):
        name, sched, reqs = out[rng.randrange(2)]
        r = list(reqs)
        i = rng.randrange(len(r))
        b = bytearray(r[i])
        for _ in range(rng.randrange(1, 4)):
            if b:
                b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
        r[i] = bytes(b)
        out.append((f"{name}: flip {k} in {i}", sched, r))
    return out


@pytest.fixture(scope="module")
def lean_runs():
    """Every case and every placement through `flock-firewall`: each one's record and the coin server's calls."""
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)
    cs = cases() + placements()
    lines = "".join(json.dumps({"schedule": s, "requests": [b.hex() for b in r], "salts": [x.hex() for x in SALTS],
                                "calls": True}) + "\n" for _, s, r in cs)
    out = subprocess.run([str(BIN)], input=lines, capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(cs)
    return cs, [json.loads(line) for line in out]


@pytest.fixture(scope="module")
def lean_records(lean_runs):
    cs, out = lean_runs
    return cs, [o["record"] for o in out]


def test_the_proxy_sends_what_the_lean_firewall_sends(lean_runs):
    """The same record and the same calls on the coin server, in the same order."""
    cs, lean = lean_runs
    differ = [(name, mine, got) for (name, s, r), got in zip(cs, lean)
              if (mine := dict(zip(("record", "calls"), proxy_session(s, r)))) != got]
    assert not differ, differ[:3]


def test_the_coin_server_never_sees_where_an_open_was_sent(lean_runs):
    """The coin server is the auditor's, so its calls, not only its record, must not show where the worker sent each Open
    (firewall-contract's gap 3): `Flock.Firewall.run` opens a stream there just before its first round. Every placement of
    the two Opens gives the honest session's calls, and every cut of one a prefix of them."""
    cs, lean = lean_runs
    got = {name: o["calls"] for (name, _, _), o in zip(cs, lean) if name.startswith("Opens at ")}
    honest = got["Opens at 4 and 4, px"]
    assert [c[0] for c in honest].count("open") == 2
    full = [n for n in got if ", cut at " not in n]
    assert len(full) == 2 * 5 * 7
    assert all(got[n] == honest for n in full)
    assert all(c == honest[:len(c)] for n, c in got.items() if ", cut at " in n)


def test_a_stopped_record_does_not_show_an_open(lean_records):
    """A stream enters the record at its first round, so a session stopped before then records the same with or without
    its ``Open`` and leaves nothing that says whether, or when, the worker opened it."""
    cs, lean = lean_records
    got = {name: rec for (name, _, _), rec in zip(cs, lean)}
    for before, after in [("stopped before Open", "stopped after Open"),
                          ("two streams: stopped before Open", "two streams: stopped after both Opens")]:
        assert got[before]["end"] == "stop" and got[before]["streams"] == {}
        assert got[after] == got[before]
    assert list(got["stopped after a round"]["streams"]) == ["circuit/rep0"]
    assert list(got["two streams: stopped after the parent's round"]["streams"]) == ["circuit/rep0"]


def test_the_cases_reach_every_ending(lean_records):
    """The cases are not all refused at the first request: some finish, and many stop after the coin server saw rounds."""
    _, lean = lean_records
    ends = [r["end"] for r in lean]
    assert ends.count("finish") >= 2 and ends.count("stop") >= 50
    assert sum(1 for r in lean if r["end"] == "stop" and r["rounds"]) >= 20


def test_a_schedule_of_another_form_is_refused(lean_runs):
    """A schedule not of rec-step3's form (`rec_live.check_schedule`) is refused by both before any request: v0's rounds,
    whose ``caps`` count level 0's copies as tops, a round without ``level0``, and v0's kind."""
    v0 = json.loads(json.dumps(SCHEDULE))
    for r in v0["rounds"]:
        r["caps"] = r.pop("tops") + [2] * r.pop("level0")
    no_level0 = json.loads(json.dumps(SCHEDULE))
    del no_level0["rounds"][1]["level0"]
    bad = [v0, no_level0, {**SCHEDULE, "kind": "rec/inner-transcript/v0/schedule"}]
    lines = "".join(json.dumps({"schedule": s, "requests": [b.hex() for b in honest()], "salts": [x.hex() for x in SALTS]}) + "\n"
                    for s in bad)
    out = subprocess.run([str(BIN)], input=lines, capture_output=True, text=True, check=True).stdout.splitlines()
    assert [json.loads(x) for x in out] == [{"error": "not a rec/inner-transcript/v1/schedule"}] * len(bad)
    for s in bad:
        with pytest.raises(ValueError, match="not a rec/inner-transcript/v1/schedule"):
            RL.Proxy(RL.CoinServer(_coins), lambda: SALTS[0], schedule=s)
