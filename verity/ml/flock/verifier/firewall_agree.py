"""The firewall's contract check: the inner phase's firewall (`verity_flock.rec_live.Proxy`) and its contract
(`flock-firewall contract`, `Flock/Firewall/Contract.lean`) on the same sessions. Whether the proxy sends what the Lean
program `Flock.Firewall.run` sends is `tests/test_firewall_agreement.py`'s question; this one is whether what it sends has
the contract's property.

    python firewall_agree.py --lean BIN --out DIR [--cases] [--sessions DIR... [--schedule FILE]]

Each session runs through the firewall with `verity_flock.firewall_transcript` watching what it sends the coin server; the
firewall's decision is whether the session ended with Finish, the contract's is `Flock.Firewall.Contract.run` on the
transcript under the verifier's public input, the session's own schedule (`verdicts`, `decide`). `--cases` runs the built-in sessions (`cases`): honest ones (on live coins,
as `rec_live serve`'s, and on coins the challenger committed to), the red team's six on #1270 (review-zk-gateway,
c00b85b0c), challengers whose coins the firewall must refuse (an opening of live coins; of committed ones, an opening that
does not open, or none), and tamper controls, firewalls that let out what the contract forbids, which the contract must refuse. `leaks` are review r6's transcripts on #1323 that leak a bit, each with its public input and the item the
contract must cut. `--sessions` replays
recorded inner sessions (`rec_live serve`'s output directories: ``session/session.json``, ``public.json``, ``private.json``,
``coins.json`` when the challenger committed, and ``schedule.json`` beside them or in their parent, or ``--schedule``) with
their own coins and salts, checks the replay gives the recorded coin server record and commitments byte for byte, and
expects both to accept. Each case carries its expectation (``firewall``, ``lean``,
and a prefix of the contract's reason); one where the two may differ says why (``divergence``). Writes DIR/<case>.json,
DIR/agreement.tsv and DIR/result.json; exits 1 if any case misses its expectation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

from verity_flock import firewall_transcript as FT
from verity_flock import rec_live as RL


def _rand(seed: bytes):
    st = {"i": 0}

    def rand(n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(seed + st["i"].to_bytes(8, "little")).digest()
            st["i"] += 1
        return out[:n]

    return rand


def _salts(seed: bytes):
    st = {"i": 0}

    def salt() -> bytes:
        st["i"] += 1
        return hashlib.shake_256(seed + st["i"].to_bytes(8, "little")).digest(FT.INST.salt_bytes)

    return salt


def _hdr(op, kind, n):
    return bytes([op, kind]) + bytes(6) + struct.pack("<Q", n)


def _framed(op, b):
    return _hdr(op, 0, len(b)) + b + bytes(-len(b) % 16)


def _obs(vs):
    return _hdr(RL.OP_OBSERVE, RL.KIND_SLICE, len(vs)) + b"".join(RL.f128_bytes(v) for v in vs)


STREAMS = ("circuit/rep0", "circuit/rep1")
CAP0 = hashlib.sha512(b"cap0").digest() + hashlib.sha512(b"cap0'").digest()
CAP1 = hashlib.sha512(b"cap1").digest()
DIGEST = bytes(range(64))
HELLO = {"profile": "fast100", "reps": 2, "flavor": "rs", "tables": ["circuit"], "link": {"sigma": "ab" * 64, "m": 4, "points": 2},
         "rounds": "bytes retained", "digests": "sha512"}
SCHEDULE = {"kind": RL.KIND + "/schedule",
            "hello": {"digests": "sha512", "flavor": "rs", "link": {"m": 4, "points": 2}, "profile": "fast100", "reps": 2,
                      "rounds": "bytes retained", "tables": ["circuit"]},
            "roots": {"circuit": 2}, "link": {"y": 2}, "streams": {s: {"parent": "", "at": 0, "rounds": 2} for s in STREAMS},
            "rounds": [r for s in STREAMS for r in ({"stream": s, "k": 0, "n": 6, "tops": [], "level0": 1, "commits": 0},
                                                      {"stream": s, "k": 1, "n": 1, "tops": [1], "level0": 0, "commits": 1})]}


def _rounds(s):
    """A rep's two rounds: the binding round (statement digest, level 0's top again) and one of 9 free elements (one row)
    and a fresh top."""
    dom = s.replace("circuit", "flock-circuit/fast100").encode()
    return [(_framed(RL.OP_DOMAIN, dom) + _framed(RL.OP_LABEL, RL.LABEL_R1CS) + _framed(RL.OP_BYTES, DIGEST) + _framed(RL.OP_BYTES, CAP0)
             + _framed(RL.OP_LABEL, b"flock-zerocheck-v0") + _hdr(RL.OP_SQUEEZE, RL.KIND_SLICE, 6), 6),
            (_obs([5 + i + 100 * len(s) for i in range(9)]) + _framed(RL.OP_BYTES, CAP1) + _hdr(RL.OP_SQUEEZE, RL.KIND_SCALAR, 1), 1)]


def requests(hello=None, y=bytes(32), order="prover", finish=True, ping=False):
    """An inner session's requests in the prover's order (`flock-circuit`'s live session: Hello, rep 0's Open, Commit, Link,
    rep 0's rounds, rep 1's Open, its rounds, the proofs, Finish), or another ``order`` of the Opens: ``reversed`` (rep 1's
    first), ``after-link`` (both after Link, `test_rec_live`'s driver) or ``early`` (rep 1's among rep 0's rounds)."""
    text = hello if isinstance(hello, str) else json.dumps(HELLO if hello is None else hello, separators=(",", ":"))
    op = lambda s: RL.encode_req(RL.REQ_OPEN, s, "", s.replace("circuit", "flock-circuit/fast100").encode(), 0)
    rd = lambda s: [RL.encode_req(RL.REQ_ROUND, s, c, n) for c, n in _rounds(s)]
    commit = [RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", CAP0)], []), RL.encode_req(RL.REQ_LINK, y)]
    head = [RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, text)]
    if order == "prover":
        body = [op(STREAMS[0])] + commit + rd(STREAMS[0]) + [op(STREAMS[1])] + rd(STREAMS[1])
    elif order == "reversed":
        body = [op(STREAMS[1]), op(STREAMS[0])] + commit + rd(STREAMS[0]) + rd(STREAMS[1])
    elif order == "after-link":
        body = commit + [op(STREAMS[0]), op(STREAMS[1])] + rd(STREAMS[0]) + rd(STREAMS[1])
    elif order == "early":
        r0 = rd(STREAMS[0])
        body = [op(STREAMS[0])] + commit + r0[:1] + [op(STREAMS[1])] + r0[1:] + rd(STREAMS[1])
    else:
        raise ValueError(order)
    if ping:
        body = [x for b in body for x in (b, RL.encode_req(RL.REQ_PING, b"ping"))]
    tail = [RL.encode_req(RL.REQ_PROOF, "circuit", r, b"the proof") for r in range(2)] + [RL.encode_req(RL.REQ_FINISH)]
    return head + body + (tail if finish else [])


class ForwardsHello(RL.Proxy):
    """#1270 before d828d3ea9 (red team's gap 2): the coin server records the prover's Hello, not the schedule's."""

    def _handle(self, r):
        if r[0] == RL.REQ_HELLO:
            v = json.loads(r[1])
            pub = {**v, "link": {k: x for k, x in v["link"].items() if k != "sigma"}}
            self.hello, self.hello_v = r[1], v
            self.coins.start(pub)
            return RL.resp_ok()
        return super()._handle(r)


class AnyY(RL.Proxy):
    """Gap 1: Link's y checked to be zero, not its length."""

    def _handle(self, r):
        if r[0] == RL.REQ_LINK and not any(r[1]):
            self.y = r[1]
            self.coins.y([int.from_bytes(r[1][i:i + 16], "little") for i in range(0, len(r[1]), 16)])
            return RL.resp_ok()
        return super()._handle(r)


class ClearTop(RL.Proxy):
    """A firewall that sends each top's nodes as they are, unsalted."""

    def _salted(self, top):
        salts = [self.fresh_salt() for _ in RL.top_nodes(top)]
        nodes = RL.top_nodes(top)
        return [n + bytes(len(n)) for n in nodes], {"nodes": [n.hex() for n in nodes], "salts": [x.hex() for x in salts]}


class ShowsSalt(RL.Proxy):
    """A firewall that answers a Proof with the last salt it drew."""

    def _handle(self, r):
        if r[0] == RL.REQ_PROOF:
            super()._handle(r)
            return RL.resp_str(RL.RESP_DRAW, json.dumps(self.private[-1]["salts"][-1]))
        return super()._handle(r)


class SwapsCoins(RL.Proxy):
    """A firewall that hands the workers other coins than the challenger's at a round."""

    def _handle(self, r):
        out = super()._handle(r)
        if r[0] == RL.REQ_ROUND and out[:1] == bytes([RL.RESP_COINS]):
            return out[:5] + bytes([out[5] ^ 1]) + out[6:]
        return out


class IgnoresOpenings(RL.Proxy):
    """A firewall that relays the challenger's coins without checking their opening."""

    def _opened(self, stream, k, coins):
        pass


class WrongOpening(FT.RecordingCoins):
    """A challenger that sends its link points with a salt that opens no commitment of its to them: a wrong one when it
    committed to its coins, a stray one when they are live."""

    def _draw(self, stream, k, n):
        coins = super()._draw(stream, k, n)
        if (stream, k) == (RL.LINK_COINS, 0):
            self.openings[(stream, k)] = bytes(FT.INST.salt_bytes)
        return coins


class NoOpening(FT.RecordingCoins):
    """A challenger that sends round 1 of rep 0's coins without the salt that opens its commitment to them."""

    def _draw(self, stream, k, n):
        coins = super()._draw(stream, k, n)
        if (stream, k) == (STREAMS[0], 1):
            del self.openings[(stream, k)]
        return coins


def cases():
    """(name, requests, firewall, salt source, expectation, challenger): ``firewall`` and ``lean`` are the verdicts expected
    of each, ``why`` a prefix of the contract's reason; ``challenger`` is the session's challenger (`run_case`'s), when it
    is not the honest one."""
    same = lambda ok, why=None: {"firewall": ok, "lean": ok, "why": why}
    stop = "the session ended without Finish"
    reused = lambda: (lambda: b"\x5a" * FT.INST.salt_bytes)
    out = [("honest", requests(), RL.Proxy, None, same(True)),
           ("honest-pings", requests(ping=True), RL.Proxy, None, same(True)),
           ("hello-keys-reordered", requests(hello={k: HELLO[k] for k in reversed(list(HELLO))}), RL.Proxy, None, same(True))]
    # the red team's six on #1270 (c00b85b0c), on today's firewall: each a stop, or the schedule's value sent in its place
    out += [(f"gap1-y-{k}", requests(y=bytes(16 * k)), RL.Proxy, None, same(False, stop)) for k in (0, 1, 3, 1000)]
    out += [("gap2-hello-reps-float", requests(hello={**HELLO, "reps": 2.0}), RL.Proxy, None, same(False, stop))]
    out += [(f"gap3-open-{o}", requests(order=o), RL.Proxy, None, same(True)) for o in ("reversed", "after-link", "early")]
    out += [("gap4-no-finish", requests(finish=False), RL.Proxy, None, same(False, stop))]
    out += [(f"gap5-unreadable-hello-{i}", requests(hello=t), RL.Proxy, None, same(False, stop)) for i, t in enumerate(("[]", "7", '"x"'))]
    out += [("gap6-unpinned-field", requests(hello={**HELLO, "note": "5e" * 64}), RL.Proxy, None, same(False, stop))]
    # tamper controls: the firewall lets out what the contract forbids, and only the contract refuses
    tamper = lambda why: {"firewall": True, "lean": False, "why": why}
    out += [("tamper-forwards-hello", requests(hello={**HELLO, "reps": 2.0}), ForwardsHello, None, tamper("hello: not the value")),
            ("tamper-any-y", requests(y=bytes(48)), AnyY, None, tamper("link.y: not the value")),
            ("tamper-reused-salt", requests(), RL.Proxy, reused(), tamper("the tape's salts are not pairwise distinct")),
            ("tamper-clear-top", requests(), ClearTop, None, tamper("a commitment under salt 0 that is not the message's")),
            ("tamper-shows-salt", requests(), ShowsSalt, None, tamper("a commitment under salt")),
            ("tamper-swaps-coins", requests(), SwapsCoins, None, tamper("circuit/rep0 round 0: the workers' coins"))]
    # the challenger's coins: committed ones accepted; an opening of live coins, committed ones that do not open their
    # commitment or come without an opening, and firewalls that serve such a session anyway
    committed = lambda **kw: {"commit": True, **kw}
    out += [("honest-committed", requests(), RL.Proxy, None, same(True), committed()),
            ("challenger-stray-opening", requests(), RL.Proxy, None, same(False, "link round 0: the workers' coins"),
             {"coins": WrongOpening}),
            ("challenger-wrong-opening", requests(), RL.Proxy, None, same(False, "link round 0: the workers' coins"),
             committed(coins=WrongOpening)),
            ("challenger-no-opening", requests(), RL.Proxy, None, same(False, "circuit/rep0 round 1: the workers' coins"),
             committed(coins=NoOpening)),
            ("tamper-stray-opening", requests(), IgnoresOpenings, None,
             tamper("link round 0: an opening of coins the challenger never committed to"), {"coins": WrongOpening}),
            ("tamper-wrong-opening", requests(), IgnoresOpenings, None,
             tamper("link round 0: coins that do not open the challenger's commitment"), committed(coins=WrongOpening)),
            ("tamper-no-opening", requests(), IgnoresOpenings, None,
             tamper("circuit/rep0 round 1: coins without the opening of the challenger's commitment"),
             committed(coins=NoOpening))]
    return [c if len(c) == 6 else (*c, None) for c in out]


def run_case(name, reqs, proxy, fresh, schedule=SCHEDULE, rand=None, challenger=None, coin_salt=None):
    s = FT.Session(schedule, rand or _rand(name.encode()), fresh or _salts(name.encode()), proxy=proxy,
                   coin_salt=coin_salt or _salts(b"challenger:" + name.encode()), **(challenger or {}))
    for r in reqs:
        s.handle(r)
    return s


def replay(d: Path, schedule: dict | None = None) -> tuple[FT.Session, str | None]:
    """A recorded inner session through today's firewall again, with its own coins and salts and its requests in the prover's
    order; and why the replay differs from the record, if it does."""
    rec = json.loads((d / "session" / "session.json").read_text())
    pub = json.loads((d / "public.json").read_text())
    priv = json.loads((d / "private.json").read_text())
    if schedule is None:
        sched_path = next((p for p in (d / "schedule.json", d.parent / "schedule.json") if p.exists()), None)
        schedule = json.loads(sched_path.read_text()) if sched_path else RL.schedule_of(pub)
    coins = b"".join(RL.f128_bytes(int(h, 16)) for h in pub["link"]["points"] + [c for r in pub["rounds"] for c in r["coins"]])
    at = {"i": 0}

    def rand(n):
        out = coins[at["i"]:at["i"] + n]
        at["i"] += n
        return out

    link = rec["link"]
    salts = [bytes.fromhex(x) for t in sorted(link["root_b"]) for x in priv["roots"][t]["salts"]]
    salts += [bytes.fromhex(x) for r in priv["rounds"] for x in [y for o in r["tops"] for y in o["salts"]] + r["salts"]]
    it = iter(salts)
    streams = sorted(rec["streams"], key=lambda s: s["stream"])
    op = lambda s: RL.encode_req(RL.REQ_OPEN, s["stream"], s["parent"], s["label"].encode(), s["at"])
    rd = lambda s: [RL.encode_req(RL.REQ_ROUND, s["stream"], bytes.fromhex(r["msg"]), r["n"]) for r in s["rounds"]]
    reqs = [RL.encode_req(RL.REQ_REGISTER), RL.encode_req(RL.REQ_HELLO, rec["hello"]), op(streams[0]),
            RL.encode_req(RL.REQ_COMMIT, bytes.fromhex(link["root_f"]), [(t, bytes.fromhex(v)) for t, v in sorted(link["root_b"].items())],
                          [(t, bytes.fromhex(v)) for t, v in sorted(link["publics"].items())]),
            RL.encode_req(RL.REQ_LINK, b"".join(RL.f128_bytes(int(h, 16)) for h in link["y"]))]
    reqs += rd(streams[0])
    for s in streams[1:]:
        reqs += [op(s)] + rd(s)
    for p in rec["proof_sha512"]:
        t, r = p["stream"].rsplit("/rep", 1)
        f = d / "session" / f"{t}.rep{r}.bin"
        reqs.append(RL.encode_req(RL.REQ_PROOF, t, int(r), f.read_bytes() if f.exists() else b""))
    reqs.append(RL.encode_req(RL.REQ_FINISH))
    cf = d / "coins.json"
    rec_coins = json.loads(cf.read_text()) if cf.exists() else None
    coin_salt = FT.INST.fresh_salt
    if rec_coins is not None:
        op = rec_coins["openings"]
        ys = iter([op[RL.LINK_COINS][0]] + [op[r["stream"]][r["k"]] for r in schedule["rounds"]])
        coin_salt = lambda: bytes.fromhex(next(ys))
    s = FT.Session(schedule, rand, lambda: next(it), coin_salt=coin_salt, commit=rec_coins is not None)
    for r in reqs:
        s.handle(r)
    got = s.coins.record()
    diff = next((k for k in got if k in pub and json.dumps(got[k], sort_keys=True) != json.dumps(pub[k], sort_keys=True)), None)
    if diff is None and rec_coins is not None and s.coins.commitment_record() != rec_coins:
        return s, "the replay's commitments to the coins differ from coins.json"
    return s, (f"the replay's coin server record differs from public.json at {diff}" if diff else None)


def decide(lean: str, out: Path, named: list[tuple[str, dict, dict]]) -> list[dict]:
    """Each ``(name, transcript, public input)`` written to ``out/<name>.json`` and decided by `flock-firewall contract
    --public` under its public input (one call per distinct public input, written to ``out/public-<k>.json``): its verdict
    (``accepted``, ``released``, ``why``), or ``{"accepted": None, "why": …}`` when the contract could not read it."""
    groups: dict[str, list[str]] = {}
    for name, t, pub in named:
        f = out / f"{name}.json"
        f.write_text(json.dumps(t))
        groups.setdefault(json.dumps(pub, sort_keys=True), []).append(str(f))
    got: dict[str, dict] = {}
    for k, (pub, files) in enumerate(groups.items()):
        pf = out / f"public-{k}.json"
        pf.write_text(pub)
        p = subprocess.run([lean, "contract", "--public", str(pf), *files], capture_output=True, text=True)
        lines = [json.loads(x) for x in p.stdout.splitlines()]
        bad = next((v["error"] for v in lines if "public" in v), None)
        for f in files:
            v = next((v for v in lines if v.get("file") == f), None)
            got[f] = ({"accepted": None, "why": bad or p.stderr.strip()[-200:]} if v is None
                      else {"accepted": None, "why": v["error"]} if "error" in v else v)
    return [got[str(out / f"{name}.json")] for name, _, _ in named]


def verdicts(lean: str, out: Path, sessions: list[tuple[str, FT.Session]]) -> list[dict]:
    """Each session's transcript decided under its public input (`FT.Session.public`: the schedule the session was run
    with, never the transcript's), as `decide` does."""
    return decide(lean, out, [(name, s.transcript(), s.public()) for name, s in sessions])


def leaks() -> list[tuple[str, dict, dict, dict]]:
    """Review r6's transcripts on #1323 (``note:review-zk-gateway/20261006T0934Z-finding-review-zk-gateway-r6``), each a
    bit the firewall could choose, built from the honest case's transcript: ``(name, transcript, public input,
    expectation)``, the expectation the contract's verdict (``accepted``), the items it releases (``released``: the cut is
    the next one) and a part of why. B1, the commitments' count and place: one more, one fewer, one moved. B2, a tape
    number naming another's value: a salt's bytes under two numbers, the workers seeing one under another number, the same
    for a pad; and two commitments under one number. B3, the public input from the transcript: a masked value no relation
    is checked on, or one checked only on a relation the transcript names, a statement value nothing fixes or a wrong one,
    a Hello from the transcript's own schedule, and no public input at all. N1, a value of the firewall's own under a name
    it chose. Committed coins, on a session whose challenger committed to them: coins without the opening of the challenger's
    commitment, under another round's opening, other than the ones committed, with a short opening, relayed twice, and
    their openings under a public input with no commitments. Accepted: the honest transcript (live coins), the committed
    one under its public input, and a masked value checked against the verifier's relation before a value of the
    firewall's own that depends on it leaves."""
    s = run_case("honest", requests(), RL.Proxy, None)
    honest, pub = s.transcript(), s.public()
    n = len(honest["items"])
    at = {it["fixed"]["what"]: i for i, it in enumerate(honest["items"]) if "fixed" in it}
    commits = [i for i, it in enumerate(honest["items"]) if "commit" in it]
    r1 = max(i for i in commits if at["round:1"] < i < at["open:circuit/rep1"])
    c0 = honest["items"][commits[0]]["commit"]
    r2 = at["round:2"]

    def commit(msg: bytes, salt: str, sid: int) -> dict:
        sent = FT.INST.commit_string(FT.INST.default_key, hashlib.sha512(msg).digest(), bytes.fromhex(salt))
        return {"commit": {"msg": msg.hex(), "salt": sid, "sent": sent.hex()}}

    def refused(released: int, why: str) -> dict:
        return {"accepted": False, "released": released, "why": why}

    def masked(relations=(), check=None, computes=()):
        t = copy.deepcopy(honest)
        t["tape"]["pads"] = ["%032x" % 7]
        mid = [{"masked": {"value": "%032x" % 1, "pad": 0, "sent": "%032x" % (1 ^ 7)}}]
        if check is not None:
            mid.append({"check": {"relation": check, "pads": [0], "ok": True}})
        mid.append({"fixed": {"what": "final_c'", "by": "own", "want": "01", "sent": "01", "pads": [0]}})
        t["items"][r2:r2] = mid
        return t, FT.public(pub["schedule"], relations, computes=computes)

    out = [("honest", honest, pub, {"accepted": True, "released": n, "why": ""})]
    t = copy.deepcopy(honest)
    t["tape"]["salts"].append(hashlib.shake_256(b"r6-extra1").digest(FT.INST.salt_bytes).hex())
    sid = len(t["tape"]["salts"]) - 1
    t["items"].insert(r2, commit(b"the worker's bit", t["tape"]["salts"][sid], sid))
    out.append(("extra-commitment", t, pub, refused(r2, f"a commitment under salt {sid} that the schedule does not owe here")))
    t = copy.deepcopy(honest)
    del t["items"][r1]
    out.append(("dropped-commitment", t, pub, refused(r1, "before the commitments the schedule owes")))
    t = copy.deepcopy(honest)
    t["items"].insert(at["link.y"] + 1, t["items"].pop(r1))
    out.append(("moved-commitment", t, pub, refused(at["link.y"] + 1, "that the schedule does not owe here")))
    t = copy.deepcopy(honest)
    t["tape"]["salts"].append(t["tape"]["salts"][c0["salt"]])
    sid = len(t["tape"]["salts"]) - 1
    t["items"].insert(r2, commit(bytes.fromhex(c0["msg"]), t["tape"]["salts"][sid], sid))
    out.append(("salt-bytes-reused", t, pub, refused(0, "the tape's salts are not pairwise distinct")))
    t = copy.deepcopy(honest)
    t["tape"]["salts"].append(t["tape"]["salts"][c0["salt"]])
    t["exposed"] = {"salts": [len(t["tape"]["salts"]) - 1], "pads": []}
    out.append(("salt-shown-under-another-id", t, pub, refused(0, "the tape's salts are not pairwise distinct")))
    t = copy.deepcopy(honest)
    c1 = t["items"][commits[1]]["commit"]
    t["items"][commits[1]] = commit(bytes.fromhex(c1["msg"]), t["tape"]["salts"][c0["salt"]], c0["salt"])
    out.append(("salt-number-reused", t, pub, refused(commits[1], f"a commitment under salt {c0['salt']}, used before")))
    t, p = masked()
    out.append(("masked-no-relations", t, p, refused(r2, "but the verifier checks no relation on it")))
    t, p = masked(check="anything the firewall likes")
    t["relations"] = ["anything the firewall likes"]
    out.append(("masked-own-relation", t, p, refused(r2, "but the verifier checks no relation on it")))
    t, p = masked(["the verifier's relation"], "anything the firewall likes", ["final_c'"])
    out.append(("masked-unlisted-relation", t, p, refused(r2 + 1, "which is not a relation the verifier checks")))
    t, p = masked(["the verifier's relation"], "the verifier's relation", ["final_c'"])
    out.append(("masked-checked", t, p, {"accepted": True, "released": n + 3, "why": ""}))
    t, p = masked()
    t["tape"]["pads"].append(t["tape"]["pads"][0])
    t["exposed"] = {"salts": [], "pads": [1]}
    out.append(("pad-shown-under-another-id", t, p, refused(0, "the tape's pads are not pairwise distinct")))
    t = copy.deepcopy(honest)
    t["items"].insert(r2, {"fixed": {"what": "anything", "by": "statement", "want": "5e" * 8, "sent": "5e" * 8}})
    out.append(("statement-value-unchecked", t, pub, refused(r2, "anything: a value the public statement does not fix")))
    out.append(("statement-value-wrong", t, FT.public(pub["schedule"], statement={"anything": bytes(8)}),
                refused(r2, "anything: not the value the statement fixes")))
    t = copy.deepcopy(honest)
    t["schedule"] = copy.deepcopy(pub["schedule"])
    t["schedule"]["hello"]["note"] = "5e"
    t["items"][at["hello"]]["fixed"]["sent"] = FT.canon(t["schedule"]["hello"]).hex()
    out.append(("schedule-from-the-transcript", t, pub, refused(at["hello"], "hello: not the value the schedule fixes")))
    out.append(("no-public-input", honest, FT.public(None), refused(at["hello"], "a schedule the public input does not carry")))
    t = copy.deepcopy(honest)
    t["items"].insert(r2, {"fixed": {"what": "own:bit=1", "by": "own", "want": "", "sent": ""}})
    out.append(("own-item-chosen", t, pub,
                refused(r2, "own:bit=1: not a value the public input lets the firewall fix by its own computation")))
    sc = run_case("honest-committed", requests(), RL.Proxy, None, challenger={"commit": True})
    committed, cpub = sc.transcript(), sc.public()
    out.append(("honest-committed", committed, cpub, {"accepted": True, "released": len(committed["items"]), "why": ""}))
    coin_at = {(it["coins"]["stream"], it["coins"]["round"]): i for i, it in enumerate(committed["items"]) if "coins" in it}
    l0, r01 = coin_at[(RL.LINK_COINS, 0)], coin_at[(STREAMS[0], 1)]
    t = copy.deepcopy(committed)
    del t["items"][l0]["coins"]["opening"]
    out.append(("coins-without-opening", t, cpub, refused(l0, "link round 0: coins without the opening of the challenger's commitment")))
    t = copy.deepcopy(committed)
    t["items"][l0]["coins"]["opening"] = t["items"][r01]["coins"]["opening"]
    out.append(("coins-under-another-rounds-opening", t, cpub, refused(l0, "link round 0: coins that do not open the challenger's commitment")))
    t = copy.deepcopy(committed)
    c = t["items"][r01]["coins"]
    c["coins"] = c["relayed"] = [RL.f128_hex(3)]
    out.append(("coins-not-the-committed-ones", t, cpub,
                refused(r01, f"{STREAMS[0]} round 1: coins that do not open the challenger's commitment")))
    t = copy.deepcopy(committed)
    t["items"][l0]["coins"]["opening"] = t["items"][l0]["coins"]["opening"][:-2]
    out.append(("coins-opening-short", t, cpub, refused(l0, "link round 0: an opening of 191 bytes, not a 192-byte salt")))
    t = copy.deepcopy(committed)
    t["items"].insert(r01 + 1, copy.deepcopy(t["items"][r01]))
    out.append(("coins-twice", t, cpub, refused(r01 + 1, f"{STREAMS[0]} round 1: coins a second time")))
    out.append(("coins-uncommitted", committed, FT.public(cpub["schedule"]),
                refused(l0, "link round 0: an opening of coins the challenger never committed to")))
    return out


def meets(v: dict, expect: dict) -> bool:
    """Whether a verdict is the expected one: accepted or not, as many items released, and its reason containing the
    expected part."""
    return (v.get("accepted") == expect["accepted"] and v.get("released") == expect["released"]
            and expect["why"] in (v.get("why") or ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", action="store_true")
    ap.add_argument("--sessions", nargs="*", default=[])
    ap.add_argument("--schedule", type=Path, help="the schedule every --sessions session is held to (default: its schedule.json)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[str, FT.Session, dict, str | None]] = []
    if a.cases:
        for name, reqs, proxy, fresh, expect, challenger in cases():
            todo.append((name, run_case(name, reqs, proxy, fresh, challenger=challenger), expect, None))
    for d in map(Path, a.sessions):
        s, bad = replay(d, json.loads(a.schedule.read_text()) if a.schedule else None)
        todo.append((f"session-{d.name}", s, {"firewall": True, "lean": True, "why": None}, bad))
    lean = verdicts(a.lean, a.out, [(name, s) for name, s, _, _ in todo])
    rows, bad = ["case\tfirewall\tlean\treleased\titems\twhy\tok"], []
    for (name, s, expect, replay_bad), L in zip(todo, lean):
        fw = s.accepted()
        n = len(json.loads((a.out / f"{name}.json").read_text())["items"])
        why = L.get("why") or ""
        ok = fw == expect["firewall"] and L["accepted"] == expect["lean"] and why.startswith(expect["why"] or "") and replay_bad is None
        if not ok:
            bad.append({"case": name, "firewall": fw, "lean": L, "expect": expect, "replay": replay_bad})
        rows.append(f"{name}\t{fw}\t{L['accepted']}\t{L.get('released')}\t{n}\t{why}\t{'ok' if ok else 'MISS'}")
    (a.out / "agreement.tsv").write_text("\n".join(rows) + "\n")
    result = {"cases": len(todo), "divergences": [n for n, _, e, _ in todo if "divergence" in e], "missed": bad}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print("\n".join(rows))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
