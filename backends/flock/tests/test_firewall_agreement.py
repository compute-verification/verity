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

from test_rec_live import CAP0, CAP1, HELLO, SCHEDULE, _binding_round, _bytes, _domain, _hdr, _label, _obs, _sq

LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-firewall"
SALTS = [hashlib.shake_256(b"firewall-salt" + i.to_bytes(8, "little")).digest(192) for i in range(64)]


def _coins(n: int) -> bytes:
    return hashlib.shake_256(b"firewall-coins" + n.to_bytes(8, "little")).digest(n)


def honest() -> list[bytes]:
    return [RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, HELLO),
            RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", CAP0)], []),
            RL.encode_req(RL.REQ_LINK, bytes(32)),
            RL.encode_req(RL.REQ_OPEN, "circuit/rep0", "", b"flock-circuit/fast100/rep0", 0),
            RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _binding_round(), 6),
            RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _obs([5, 6, 7]) + _sq(1, RL.KIND_SCALAR), 1),
            RL.encode_req(RL.REQ_PROOF, "circuit", 0, b"the proof"),
            RL.encode_req(RL.REQ_FINISH)]


#: a schedule with a child stream, so `Open`'s parent and position and a round's caps after the binding round are pinned too
SCHEDULE2 = {**json.loads(json.dumps(SCHEDULE)),
             "streams": {"circuit/rep0": {"parent": "", "at": 0, "rounds": 2}, "circuit/rep0/x": {"parent": "circuit/rep0", "at": 1, "rounds": 1}},
             "rounds": SCHEDULE["rounds"] + [{"stream": "circuit/rep0/x", "k": 0, "n": 3, "caps": [1], "commits": 2}]}


def honest2() -> list[bytes]:
    r = honest()
    child = RL.encode_req(RL.REQ_ROUND, "circuit/rep0/x", _bytes(b"\x00" * 8) + _bytes(CAP1) + _obs(list(range(1, 10))) + _sq(3), 3)
    return r[:5] + [RL.encode_req(RL.REQ_OPEN, "circuit/rep0/x", "circuit/rep0", b"x", 1)] + r[5:7] + [child] + r[7:]


def proxy_record(schedule: dict, reqs: list[bytes]) -> dict:
    salts = iter(SALTS)
    proxy = RL.Proxy(RL.CoinServer(_coins), lambda: next(salts), schedule=schedule)
    for b in reqs:
        proxy.handle(b)
    if proxy.verdict is None and proxy.stopped is None:
        proxy.stop("rec proxy: the session ended without Finish")
    rec = proxy.coins.record()
    if rec["link"] is not None:
        rec["link"] = {k: v for k, v in rec["link"].items() if k != "points"}
    rec["rounds"] = [{k: v for k, v in r.items() if k != "coins"} for r in rec["rounds"]]
    return rec


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
             "round 1": lambda s: s["rounds"][1].update(commits=2),
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
def lean_records():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)
    cs = cases()
    lines = "".join(json.dumps({"schedule": s, "requests": [b.hex() for b in r], "salts": [x.hex() for x in SALTS]}) + "\n"
                    for _, s, r in cs)
    out = subprocess.run([str(BIN)], input=lines, capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(cs)
    return cs, [json.loads(line) for line in out]


def test_the_proxy_sends_what_the_lean_firewall_sends(lean_records):
    cs, lean = lean_records
    differ = [(name, proxy_record(s, r), got) for (name, s, r), got in zip(cs, lean) if proxy_record(s, r) != got]
    assert not differ, differ[:3]


def test_the_cases_reach_every_ending(lean_records):
    """The cases are not all refused at the first request: some finish, and many stop after the coin server saw rounds."""
    _, lean = lean_records
    ends = [r["end"] for r in lean]
    assert ends.count("finish") >= 2 and ends.count("stop") >= 50
    assert sum(1 for r in lean if r["end"] == "stop" and r["rounds"]) >= 20
