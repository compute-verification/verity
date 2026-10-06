"""The firewall's agreement check: the inner phase's firewall (`verity_flock.rec_live.Proxy`) and its contract
(`flock-firewall check`, `Flock/Firewall/Contract.lean`) on the same sessions.

    python firewall_agree.py --lean BIN --out DIR [--cases] [--sessions DIR... [--schedule FILE]]

Each session runs through the firewall with `verity_flock.firewall_transcript` watching what it sends the coin server; the
firewall's decision is whether the session ended with Finish, the contract's is `Flock.Firewall.run` on the transcript.
`--cases` runs the built-in sessions (`cases`): honest ones, the red team's six on #1270 (review-zk-gateway, c00b85b0c), and
tamper controls, firewalls that let out what the contract forbids, which the contract must refuse. `--sessions` replays
recorded inner sessions (`rec_live serve`'s output directories: ``session/session.json``, ``public.json``, ``private.json``,
and ``schedule.json`` beside them or in their parent, or ``--schedule``) with their own coins and salts, checks the replay gives the recorded
coin server record byte for byte, and expects both to accept. Each case carries its expectation (``firewall``, ``lean``,
and a prefix of the contract's reason); one where the two may differ says why (``divergence``). Writes DIR/<case>.json,
DIR/agreement.tsv and DIR/result.json; exits 1 if any case misses its expectation.
"""
from __future__ import annotations

import argparse
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
DIGEST = bytes(range(64))
HELLO = {"profile": "fast100", "reps": 2, "flavor": "rs", "tables": ["circuit"], "link": {"sigma": "ab" * 64, "m": 4, "points": 2},
         "rounds": "bytes retained", "digests": "sha512"}
SCHEDULE = {"kind": RL.KIND + "/schedule",
            "hello": {"digests": "sha512", "flavor": "rs", "link": {"m": 4, "points": 2}, "profile": "fast100", "reps": 2,
                      "rounds": "bytes retained", "tables": ["circuit"]},
            "roots": {"circuit": 2}, "link": {"y": 2}, "streams": {s: {"parent": "", "at": 0, "rounds": 2} for s in STREAMS},
            "rounds": [r for s in STREAMS for r in ({"stream": s, "k": 0, "n": 6, "caps": [2], "commits": 0},
                                                      {"stream": s, "k": 1, "n": 1, "caps": [], "commits": 2})]}


def _rounds(s):
    """A rep's two rounds: the binding round (statement digest, level-0 cap) and one of 9 free elements (two rows)."""
    dom = s.replace("circuit", "flock-circuit/fast100").encode()
    return [(_framed(RL.OP_DOMAIN, dom) + _framed(RL.OP_LABEL, RL.LABEL_R1CS) + _framed(RL.OP_BYTES, DIGEST) + _framed(RL.OP_BYTES, CAP0)
             + _framed(RL.OP_LABEL, b"flock-zerocheck-v0") + _hdr(RL.OP_SQUEEZE, RL.KIND_SLICE, 6), 6),
            (_obs([5 + i + 100 * len(s) for i in range(9)]) + _hdr(RL.OP_SQUEEZE, RL.KIND_SCALAR, 1), 1)]


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
    """A firewall that sends the level-0 cap's nodes as they are, unsalted."""

    def _salted(self, top):
        salts = [self.fresh_salt() for _ in RL.top_nodes(top)]
        return top + bytes(len(top)), salts


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


def cases():
    """(name, requests, firewall, salt source, expectation): ``firewall`` and ``lean`` are the verdicts expected of each,
    ``why`` a prefix of the contract's reason."""
    same = lambda ok, why=None: {"firewall": ok, "lean": ok, "why": why}
    stop = "the session ended without Finish"
    reused = lambda: (lambda: b"\x5a" * FT.INST.salt_bytes)
    out = [("honest", requests(), RL.Proxy, None, same(True)),
           ("honest-pings", requests(ping=True), RL.Proxy, None, same(True)),
           ("hello-keys-reordered", requests(hello={k: HELLO[k] for k in reversed(list(HELLO))}), RL.Proxy, None, same(True))]
    # the red team's six on #1270 (c00b85b0c), on today's firewall: each a stop, or the schedule's value sent
    out += [(f"gap1-y-{k}", requests(y=bytes(16 * k)), RL.Proxy, None, same(False, stop)) for k in (0, 1, 3, 1000)]
    out += [("gap2-hello-reps-float", requests(hello={**HELLO, "reps": 2.0}), RL.Proxy, None, same(False, stop))]
    out += [(f"gap3-open-{o}", requests(order=o), RL.Proxy, None,
             {"firewall": True, "lean": False, "why": why,
              "divergence": "the firewall holds each Open's parent and position in its stream, not where it comes in the "
                            "session; the contract gives each Open one place (the prover's order), so the order carries nothing"})
            for o, why in (("reversed", "Open circuit/rep1: not the schedule's next stream"),
                           ("after-link", "Commit's roots: not after Hello and the first stream's Open"),
                           ("early", "Open circuit/rep1: a later stream opens just before its first round"))]
    out += [("gap4-no-finish", requests(finish=False), RL.Proxy, None, same(False, stop))]
    out += [(f"gap5-unreadable-hello-{i}", requests(hello=t), RL.Proxy, None, same(False, stop)) for i, t in enumerate(("[]", "7", '"x"'))]
    out += [("gap6-unpinned-field", requests(hello={**HELLO, "note": "5e" * 64}), RL.Proxy, None, same(False, stop))]
    # tamper controls: the firewall lets out what the contract forbids, and only the contract refuses
    tamper = lambda why: {"firewall": True, "lean": False, "why": why}
    out += [("tamper-forwards-hello", requests(hello={**HELLO, "reps": 2.0}), ForwardsHello, None, tamper("hello: not the value")),
            ("tamper-any-y", requests(y=bytes(48)), AnyY, None, tamper("link.y: not the value")),
            ("tamper-reused-salt", requests(), RL.Proxy, reused(), tamper("a commitment under salt 0, used before")),
            ("tamper-clear-top", requests(), ClearTop, None, tamper("a commitment under salt 0 that is not the message's")),
            ("tamper-shows-salt", requests(), ShowsSalt, None, tamper("a commitment under salt")),
            ("tamper-swaps-coins", requests(), SwapsCoins, None, tamper("circuit/rep0 round 0: the workers' coins"))]
    return out


def run_case(name, reqs, proxy, fresh, schedule=SCHEDULE, rand=None):
    s = FT.Session(schedule, rand or _rand(name.encode()), fresh or _salts(name.encode()), proxy=proxy)
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
    salts += [bytes.fromhex(x) for r in priv["rounds"] for x in r["salts"] + [y for cs in r["cap_salts"] for y in cs]]
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
    s = FT.Session(schedule, rand, lambda: next(it))
    for r in reqs:
        s.handle(r)
    got = s.coins.record()
    want = {k: pub[k] for k in got if k in pub}
    diff = next((k for k in got if json.dumps(got[k], sort_keys=True) != json.dumps(want.get(k), sort_keys=True)), None)
    return s, (f"the replay's coin server record differs from public.json at {diff}" if diff else None)


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
        for name, reqs, proxy, fresh, expect in cases():
            todo.append((name, run_case(name, reqs, proxy, fresh), expect, None))
    for d in map(Path, a.sessions):
        s, bad = replay(d, json.loads(a.schedule.read_text()) if a.schedule else None)
        todo.append((f"session-{d.name}", s, {"firewall": True, "lean": True, "why": None}, bad))
    files = []
    for name, s, expect, _ in todo:
        f = a.out / f"{name}.json"
        f.write_text(json.dumps(s.transcript()))
        files.append(str(f))
    p = subprocess.run([a.lean, "check", *files], capture_output=True, text=True)
    lean = {v["file"]: v for v in map(json.loads, p.stdout.splitlines())}
    rows, bad = ["case\tfirewall\tlean\treleased\titems\twhy\tok"], []
    for (name, s, expect, replay_bad), f in zip(todo, files):
        L = lean.get(f) or {"accepted": None, "why": p.stderr.strip()[-200:]}
        if "error" in L:
            L = {"accepted": None, "why": L["error"]}
        fw = s.accepted()
        n = len(json.loads(Path(f).read_text())["items"])
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
