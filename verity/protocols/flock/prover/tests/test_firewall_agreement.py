"""The ZK server's contract, inner phase: `rec_live`'s proxy sends the challenger exactly what `Flock.Firewall.run` (the Lean
program `flock-firewall`) sends, byte for byte, on an honest session and on every session a worker can make of it here (each
schedule edit of `test_rec_live`, every prefix, drop, duplicate and swap of a request, requests after ``Finish``, forged
requests, a Commit of two tables in and out of the schedule's order, and seeded random byte flips). The Lean program's
type is the contract (live PROTOCOL.md §10.3): each message is the schedule's own value or a commitment under one of the
ZK server's salts, so agreement means nothing else leaves the proxy.

Under ``rec_live --lean`` the send check is Lean's own (`rec_live.LeanFirewall`, ``flock-firewall live``): on every case
and every whole placement the challenger's live-coin part's record and calls are `Flock.Firewall.run`'s at the salts the live session drew,
so the live mode's steps are `run`'s loop and the proxy relays what Lean sent and nothing else. On the challenger's coin
tree (``rec_live --coin-tree``) the same holds with the tree's spec given to Lean (``coin_tree``, `Flock.Firewall.treeHello`),
on the honest session and on every Hello a worker can name the tree with."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_live as RL

from test_rec_live import (CAP0, CAP1, HELLO, PROVER_NONCE, SCHEDULE, TREE, _binding_round, _bytes, _content, _domain, _hdr,
                           _label, _obs, _sq, _tree_hello)
from verity.commitments import coin_tree as CT

LEAN = Path(__file__).resolve().parents[2] / "verifier"
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


#: a schedule of two tables with roots, its ``roots`` listing them out of the schedule's order (their names, sorted)
SCHEDULE3 = {**json.loads(json.dumps(SCHEDULE)), "hello": {**SCHEDULE["hello"], "tables": ["aux", "circuit"]},
             "roots": {"circuit": 2, "aux": 2}}
HELLO3 = json.dumps({**json.loads(HELLO), "tables": ["aux", "circuit"]}, separators=(",", ":"))


def honest3(tables=("aux", "circuit")) -> list[bytes]:
    """`honest`'s session under `SCHEDULE3`, its Commit naming ``tables`` in order."""
    r = honest()
    r[1] = RL.encode_req(RL.REQ_HELLO, HELLO3)
    r[2] = RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [(t, CAP0) for t in tables], [])
    return r


#: how `Flock.Firewall.calls` writes each call the proxy makes on the challenger's live-coin part
CALL = {"start": lambda a: ["start"], "commit": lambda a: ["commit"], "y": lambda a: ["y"],
        "open": lambda a: ["open", a[0], a[1], a[2]], "round": lambda a: ["round", a[0], a[3]]}


def proxy_session(schedule: dict, reqs: list[bytes], lean: RL.LeanFirewall | None = None,
                  tree: CT.Spec | None = None) -> tuple[dict, list]:
    """The challenger's live-coin part's record and every call made on it, in order, refused or not, while the proxy handles ``reqs``
    (relaying them through ``lean`` when it is given; on a coin tree of spec ``tree`` this host samples, when it is given)."""
    coins = RL.CoinServer(_coins) if tree is None else RL.CoinServer(_coins, tree, RL.OsTree(_coins))
    salts, calls = iter(SALTS), []
    for name, show in CALL.items():
        f = getattr(coins, name)
        setattr(coins, name, lambda *a, _f=f, _show=show, **kw: (calls.append(_show(a)), _f(*a, **kw))[1])
    proxy = RL.Proxy(coins, lambda: next(salts), schedule=schedule) if lean is None else RL.Proxy(coins, schedule=schedule, lean=lean)
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
             "y": lambda s: s["link"].update(y=1)}
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
        "a copy that is not level 0's top": (7, RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _label(b"flock-ligerito") + _bytes(CAP1 + CAP1) + _sq(2), 2)),
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
    out += [("honest, two tables", SCHEDULE3, honest3()),
            ("two tables: Commit names them out of order", SCHEDULE3, honest3(("circuit", "aux")))]
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
def built():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)
    return BIN


def lean_run(cs: list[tuple[str, dict, list[bytes]]], salts: list[list[bytes]], tree: CT.Spec | None = None) -> list[dict]:
    """`flock-firewall` on each case at its salts (on the coin tree of spec ``tree``, when it is given): its record and the
    challenger's live-coin part's calls."""
    on = {} if tree is None else {"coin_tree": tree.to_json()}
    lines = "".join(json.dumps({"schedule": s, "requests": [b.hex() for b in r], "salts": [x.hex() for x in y],
                                "calls": True, **on}) + "\n" for (_, s, r), y in zip(cs, salts))
    out = subprocess.run([str(BIN)], input=lines, capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(cs)
    return [json.loads(line) for line in out]


@pytest.fixture(scope="module")
def lean_runs(built):
    """Every case and every placement through `flock-firewall`: each one's record and the challenger's live-coin part's calls."""
    cs = cases() + placements()
    return cs, lean_run(cs, [SALTS] * len(cs))


class Drawn(RL.LeanFirewall):
    """`rec_live.LeanFirewall`, keeping every salt it drew, in order."""

    def __init__(self, *a):
        self.drawn: list[bytes] = []
        super().__init__(*a)

    def step(self, req):
        sent, salts = super().step(req)
        self.drawn += salts
        return sent, salts


def relay_session(schedule: dict, reqs: list[bytes], tree: CT.Spec | None = None) -> tuple[dict, list, list[bytes]]:
    """`proxy_session` relayed through ``flock-firewall live``, and the salts it drew."""
    lean = Drawn(BIN, schedule, None if tree is None else tree.to_json())
    try:
        rec, calls = proxy_session(schedule, reqs, lean, tree)
    finally:
        lean.close()
    return rec, calls, lean.drawn


@pytest.fixture(scope="module")
def relayed(built):
    """Every case and every whole placement (a cut one is a case's prefix) relayed through ``flock-firewall live``, one
    process a session, and `flock-firewall` on its requests at the salts the live session drew."""
    cs = cases() + [c for c in placements() if ", cut at " not in c[0]]
    got = [relay_session(s, r) for _, s, r in cs]
    return cs, got, lean_run(cs, [drawn for _, _, drawn in got])


@pytest.fixture(scope="module")
def lean_records(lean_runs):
    cs, out = lean_runs
    return cs, [o["record"] for o in out]


def test_the_proxy_sends_what_the_lean_firewall_sends(lean_runs):
    """The same record and the same calls on the challenger, in the same order."""
    cs, lean = lean_runs
    differ = [(name, mine, got) for (name, s, r), got in zip(cs, lean)
              if (mine := dict(zip(("record", "calls"), proxy_session(s, r)))) != got]
    assert not differ, differ[:3]


def test_the_coin_server_never_sees_where_an_open_was_sent(lean_runs):
    """The challenger's live-coin part is the auditor's, so its calls, not only its record, must not show where the worker sent each Open:
    `Flock.Firewall.run` opens a stream there just before its first round. Every placement of
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
    """The cases are not all refused at the first request: some finish, and many stop after the challenger saw rounds."""
    _, lean = lean_records
    ends = [r["end"] for r in lean]
    assert ends.count("finish") >= 2 and ends.count("stop") >= 50
    assert sum(1 for r in lean if r["end"] == "stop" and r["rounds"]) >= 20


def test_copies_are_by_position_never_by_value(lean_records):
    """A cap is a copy of level 0's top only where the schedule's round names one, whatever its value: under a schedule
    whose round 2 has a fresh top of two nodes and no copy, the honest session's copy of level 0's top there is committed
    as a fresh salted top and the session finishes, and a copy that is not level 0's top stops it at that round."""
    cs, lean = lean_records
    got = {name: rec for (name, _, _), rec in zip(cs, lean)}
    assert (got["honest"]["rounds"][2]["level0"], got["honest"]["rounds"][2]["tops"]) == (1, [])
    fresh = got["schedule's round 2's level-0 copy as a top edited"]
    assert fresh["end"] == "finish" and fresh["rounds"][2]["level0"] == 0 and [len(t) for t in fresh["rounds"][2]["tops"]] == [2]
    forged = got["forged: a copy that is not level 0's top"]
    assert forged["end"] == "stop" and len(forged["rounds"]) == 2


def test_commit_takes_the_tables_in_the_schedules_order_only(built, lean_records):
    """Commit's table order is the schedule's (its roots' names, sorted), never the worker's, so the order carries
    nothing: under a schedule of two tables whose ``roots`` lists them in another order, the session naming them in the
    schedule's order finishes, and the same session naming them out of it stops at Commit, the challenger's live-coin part shown no
    link, in the proxy, in `Flock.Firewall.run` and in the relay through ``flock-firewall live``."""
    cs, lean = lean_records
    got = {name: rec for (name, _, _), rec in zip(cs, lean)}
    assert got["honest, two tables"]["end"] == "finish"
    out = got["two tables: Commit names them out of order"]
    assert out["end"] == "stop" and out["link"] is None and out["rounds"] == []
    why = RL.refused("Commit's roots: the tables are not in the pinned schedule's order")
    proxy = RL.Proxy(RL.CoinServer(_coins), iter(SALTS).__next__, schedule=SCHEDULE3)
    for b in honest3(("circuit", "aux"))[:3]:
        proxy.handle(b)
    assert proxy.stopped == f"rec proxy: {why}"
    lean_fw = RL.LeanFirewall(BIN, SCHEDULE3)
    try:
        proxy = RL.Proxy(RL.CoinServer(_coins), schedule=SCHEDULE3, lean=lean_fw)
        for b in honest3(("circuit", "aux"))[:3]:
            proxy.handle(b)
    finally:
        lean_fw.close()
    assert proxy.stopped == f"rec proxy: the Lean firewall: {why}"


def _bad_schedules() -> list[dict]:
    def edited(edit):
        s = json.loads(json.dumps(SCHEDULE))
        edit(s)
        return s

    def v0(s):
        for r in s["rounds"]:
            r["caps"] = r.pop("tops") + [2] * r.pop("level0")

    return [edited(v0), edited(lambda s: s["rounds"][1].pop("level0")), {**SCHEDULE, "kind": "rec/inner-transcript/v0/schedule"},
            edited(lambda s: s["rounds"][0].update(extra=0)), edited(lambda s: s["rounds"].append({**s["rounds"][1], "k": 2})),
            edited(lambda s: s["rounds"].insert(0, s["rounds"].pop(1))), edited(lambda s: s["streams"]["circuit/rep0"].update(rounds=4)),
            edited(lambda s: s["streams"].update({"circuit/rep1": {"parent": "", "at": 0, "rounds": 0}})),
            edited(lambda s: s["hello"].update(coins="seed")), edited(lambda s: s["hello"].update(reps=2.5)),
            edited(lambda s: s["hello"].update(reps=-2)), edited(lambda s: s["hello"].update(flavor='r"s'))]


def test_a_schedule_of_another_form_is_refused(lean_runs):
    """A schedule not of v1's form, or one no honest session can finish (`rec_live.check_schedule`,
    `Flock.Firewall.Schedule.canonical`), is refused by the proxy, `flock-firewall` and its live mode before any request:
    v0's rounds, whose ``caps`` count level 0's copies as tops, a round without ``level0``, v0's kind, a round with another
    key, a round its stream does not number next, a stream with a round count not its rounds' or with none, and a Hello
    naming coins or with a number that is not a natural or a string holding a ``"``. Lean's firewall, which holds no
    registered top, also refuses a stream whose table has no root."""
    bad = _bad_schedules()
    rootless = {**SCHEDULE, "roots": {}}
    lines = "".join(json.dumps({"schedule": s, "requests": [b.hex() for b in honest()], "salts": [x.hex() for x in SALTS]}) + "\n"
                    for s in bad + [rootless])
    out = subprocess.run([str(BIN)], input=lines, capture_output=True, text=True, check=True).stdout.splitlines()
    assert [json.loads(x) for x in out] == [{"error": "not a rec/inner-transcript/v1/schedule"}] * (len(bad) + 1)
    for s in bad:
        with pytest.raises(ValueError, match="not a rec/inner-transcript/v1/schedule"):
            RL.Proxy(RL.CoinServer(_coins), lambda: SALTS[0], schedule=s)
    for s in bad + [rootless]:
        with pytest.raises(ValueError, match="the schedule is refused"):
            RL.LeanFirewall(BIN, s)
    RL.Proxy(RL.CoinServer(_coins), lambda: SALTS[0], schedule=rootless)


def test_the_relay_sends_what_run_sends_at_the_salts_it_drew(relayed):
    """``rec_live --lean`` on every case and whole placement: the challenger's live-coin part's record and calls, in order, are
    `Flock.Firewall.run`'s on the same requests at the salts ``flock-firewall live`` drew."""
    cs, got, lean = relayed
    differ = [name for (name, _, _), (rec, calls, _), want in zip(cs, got, lean) if {"record": rec, "calls": calls} != want]
    assert not differ, differ[:5]


def test_the_relay_reaches_every_ending_under_fresh_salts(relayed):
    """The honest sessions finish under the relay and the others stop as they do without it, and every salt it drew is
    a fresh one of 192 bytes: none repeats across the cases."""
    cs, got, _ = relayed
    ends = {name: rec["end"] for (name, _, _), (rec, _, _) in zip(cs, got)}
    assert ends["honest"] == ends["honest, two streams"] == "finish"
    plain = {name: proxy_session(s, r)[0]["end"] for name, s, r in cs}
    assert ends == plain
    salts = [y for _, _, drawn in got for y in drawn]
    assert len(salts) > len(cs) and len(set(salts)) == len(salts) and {len(y) for y in salts} == {192}


class More(RL.LeanFirewall):
    """`rec_live.LeanFirewall` that answers ``Commit`` with one more salt or one more message than Lean's."""

    def __init__(self, more, *a):
        self.more = more
        super().__init__(*a)

    def step(self, req):
        sent, salts = super().step(req)
        if req[:1] == bytes([RL.REQ_COMMIT]):
            return (sent + [{"link": 2}], salts) if self.more == "message" else (sent, salts + [bytes(192)])
        return sent, salts


@pytest.mark.parametrize("edit, more, why", [
    ({"commit_top": lambda cap, salts: [bytes(96)] * len(salts)}, None, "commit: not the message the Lean firewall sent"),
    ({"top_rows": lambda cap, _f=RL.top_rows: _f(cap) + _f(cap)[:1]}, None, "a salt the Lean firewall did not draw"),
    ({}, "salt", "the Lean firewall sent or drew more for the request than the proxy"),
    ({}, "message", "the Lean firewall sent or drew more for the request than the proxy"),
], ids=["another commitment", "a salt more", "a salt fewer", "a message fewer"])
def test_a_proxy_that_parts_from_the_lean_firewall_stops_the_session(built, monkeypatch, edit, more, why):
    """The relay delivers only what Lean sent, in its order, with its salts: a proxy whose Commit would send another
    commitment, sample a salt Lean did not, or leave one of Lean's salts or messages unused stops the session at Commit, and
    the challenger's live-coin part is never shown the link exchange."""
    for name, f in edit.items():
        monkeypatch.setattr(RL, name, f)
    lean = RL.LeanFirewall(BIN, SCHEDULE) if more is None else More(more, BIN, SCHEDULE)
    coins = RL.CoinServer(_coins)
    proxy = RL.Proxy(coins, schedule=SCHEDULE, lean=lean)
    try:
        for b in honest()[:4]:
            proxy.handle(b)
    finally:
        lean.close()
    assert proxy.stopped == f"rec proxy: {RL.refused(why)}"
    assert coins.hello is not None and (coins.link is None or coins.link["y"] is None) and coins.end == "stop"


def test_a_lean_firewall_that_exits_stops_the_session(built):
    lean = RL.LeanFirewall(BIN, SCHEDULE)
    coins = RL.CoinServer(_coins)
    proxy = RL.Proxy(coins, schedule=SCHEDULE, lean=lean)
    for b in honest()[:2]:
        proxy.handle(b)
    lean.proc.kill()
    lean.proc.wait()
    proxy.handle(honest()[2])
    lean.close()
    assert proxy.stopped.startswith(f"rec proxy: {BIN} live: ") and coins.link is None and coins.end == "stop"


def test_the_lean_firewall_draws_its_own_salts(built):
    sch = json.loads(json.dumps(SCHEDULE))
    lean = RL.LeanFirewall(BIN, sch)
    try:
        for coins in (RL.CoinServer(_coins, RL.INNER_COIN_SPEC, RL.OsTree()), RL.CoinServer(_coins)):
            with pytest.raises(ValueError, match="the salts are its own"):
                RL.Proxy(coins, lambda: SALTS[0], schedule=sch, lean=lean)
    finally:
        lean.close()


def tree_cases() -> list[tuple[str, dict, list[bytes]]]:
    """The honest session on the coin tree ``TREE``, its prefixes, and every Hello a worker can name the tree with: no
    nonce, another nonce, another spec, no coins, a key more in coins or beside them, a number that is not an integer, keys
    reordered and spaced out, not JSON."""
    v, spec = json.loads(HELLO), TREE.to_json()
    hellos = {"honest": _tree_hello(), "no nonce": json.dumps({**v, "coins": spec}),
              "another nonce": json.dumps({**v, "coins": {**spec, "nonce": "00" * CT.NONCE_BYTES}}),
              "another spec": _tree_hello(CT.Spec((CT.SESSION, "circuit/rep0"), 2, 8)), "no coins": HELLO,
              "coins live": json.dumps({**v, "coins": "seed"}),
              "a key more in coins": json.dumps({**v, "coins": {**spec, "nonce": PROVER_NONCE, "x": 1}}),
              "a key more beside coins": json.dumps({**v, "x": 1, "coins": {**spec, "nonce": PROVER_NONCE}}),
              "block 8.0": _tree_hello().replace('"block":8', '"block":8.0'),
              "rounds_log 2e0": _tree_hello().replace('"rounds_log":2', '"rounds_log":2e0'),
              "keys reordered and spaced": json.dumps({"coins": {**spec, "nonce": PROVER_NONCE}, **dict(reversed(list(v.items())))}, indent=1),
              "not JSON": _tree_hello()[:-1]}
    out = []
    for what, text in hellos.items():
        r = honest()
        r[1] = RL.encode_req(RL.REQ_HELLO, text)
        out.append((f"tree, Hello: {what}", SCHEDULE, r))
    out += [(f"tree: the first {i}", SCHEDULE, out[0][2][:i]) for i in range(len(out[0][2]))]
    out.append(("tree: Hello twice", SCHEDULE, out[0][2][:2] + out[0][2][1:]))
    return out


def test_on_the_coin_tree_the_proxy_and_the_relay_send_what_the_lean_firewall_sends(built):
    """On the challenger's coin tree, the proxy alone (its Python send check), the relay through ``flock-firewall live``
    and `Flock.Firewall.run` with the tree's spec (`treeHello`) give the challenger's live-coin part the same record and calls, case by
    case; the honest session finishes, and each edited Hello other than its nonce and its spelling stops it before the
    challenger's live-coin part sees anything."""
    cs = tree_cases()
    plain = [dict(zip(("record", "calls"), proxy_session(s, r, tree=TREE))) for _, s, r in cs]
    assert lean_run(cs, [SALTS] * len(cs), TREE) == plain
    relayed = [relay_session(s, r, TREE) for _, s, r in cs]
    assert [{"record": rec, "calls": calls} for rec, calls, _ in relayed] == lean_run(cs, [d for _, _, d in relayed], TREE)
    ends = {name: rec["end"] for (name, _, _), (rec, _, _) in zip(cs, relayed)}
    held = {"honest", "no nonce", "another nonce", "keys reordered and spaced"}
    assert {n for n, e in ends.items() if e == "finish"} == {f"tree, Hello: {w}" for w in held}
    for (name, _, _), (rec, calls, _) in zip(cs, relayed):
        if name.startswith("tree, Hello: ") and name.removeprefix("tree, Hello: ") not in held:
            assert rec["end"] == "stop" and calls == [], name


def test_serve_relays_each_session_through_its_own_lean_firewall(built, tmp_path):
    """``serve(lean=…)``: a session Lean refuses (a Hello field of the prover's choosing) stops with Lean's reason and is
    written as a stop, and the next, honest one is held and written whole, on a ZK server of its own."""
    from test_rec_live_firewall import HELLO as FW_HELLO
    from test_rec_live_firewall import _held, _refusal, _run, _schedule, _serve, _Wire

    t, got = _serve(tmp_path, 2, _schedule(), lean=BIN)
    w = _Wire(got["addr"])
    first = _run(w, hello={**FW_HELLO, "note": "5e" * 64})
    w.close()
    w = _Wire(got["addr"])
    second = _run(w)
    w.close()
    t.join()
    assert "crash" not in got and _held(second)
    assert _refusal(first).startswith("rec proxy: the Lean firewall: FIREWALL-REFUSED: Hello: ")
    assert [r["session"] for r in got["res"]] == [None, str(tmp_path / "s1" / "session")]
    assert json.loads((tmp_path / "s0" / "public.json").read_text())["end"] == "stop"
    assert json.loads((tmp_path / "s1" / "public.json").read_text())["end"] == "finish"


FLAT = Path(__file__).resolve().parent / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"


def test_a_real_provers_session_passes_the_lean_firewall_as_run_does(built):
    """The flat GEMM selftest's honest session (the fetched fixture `test_rec_vstage` replays), at `schedule_for`'s schedule
    of its shape, relayed through ``flock-firewall live``: it finishes, and its live-coin part record is `Flock.Firewall.run`'s
    at the salts the live session drew. The cases above are synthetic; this is the unmodified prover's own rounds."""
    assert (FLAT / "records-cpu" / "honest.json").is_file(), (
        f"{FLAT} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/hidden-outputs`")
    rec = json.loads((FLAT / "records-cpu" / "honest.json").read_text())["record"]
    hello = json.loads(rec["hello"])
    del hello["coins"]
    lk = rec["link"]
    reqs = [RL.encode_req(RL.REQ_HELLO, json.dumps(hello)),
            RL.encode_req(RL.REQ_COMMIT, bytes.fromhex(lk["root_f"]), [(t, bytes.fromhex(v)) for t, v in lk["root_b"].items()], []),
            RL.encode_req(RL.REQ_LINK, bytes(16 * len(lk["y"])))]
    for st in rec["streams"]:
        reqs.append(RL.encode_req(RL.REQ_OPEN, st["stream"], st["parent"] or "", st["label"].encode(), st["at"]))
        reqs += [RL.encode_req(RL.REQ_ROUND, st["stream"], bytes.fromhex(rd["msg"]), rd["n"]) for rd in st["rounds"]]
    reqs += [RL.encode_req(RL.REQ_PROOF, "circuit", r, (FLAT / "records-cpu" / f"honest.rep{r}.proof").read_bytes()) for r in (0, 1)]
    reqs.append(RL.encode_req(RL.REQ_FINISH))
    schedule = RL.schedule_for({"dense_m": 25, "k_log": 22, "nbl": 3, "regions": 3})
    got, calls, drawn = relay_session(schedule, reqs)
    assert got["end"] == "finish" and len(got["rounds"]) == sum(len(st["rounds"]) for st in rec["streams"])
    assert lean_run([("real", schedule, reqs)], [drawn]) == [{"record": got, "calls": calls}]


def test_main_serves_only_through_the_lean_firewall(tmp_path, capsys):
    (tmp_path / "s.json").write_text(json.dumps(SCHEDULE))
    base = ["--listen", "127.0.0.1:0", "--out", str(tmp_path / "o"), "--schedule", str(tmp_path / "s.json")]
    for argv, why in [(base, "--lean BIN: Lean's flock-firewall"),
                      (base + ["--coin-tree", "--tree-test-os"], "--lean BIN: Lean's flock-firewall")]:
        with pytest.raises(SystemExit) as e:
            RL.main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
