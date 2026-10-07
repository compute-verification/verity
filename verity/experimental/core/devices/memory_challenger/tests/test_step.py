"""The challenger itself: `MemoryChallenger.step` (`memory-challenger-difftest`'s `step`) and the driver over an in-memory
world (`memory-challenger-drive`) against `memory_challenger.MemoryChallenger` on an injected clock and an in-memory link.

`Challenger` is the Python's challenger behind `step`'s events, and what it adds is the Lean's own, which the Python has no
counterpart for (`MemoryChallenger.Step`'s docstring): it frames the answer from the link's bytes (the Python's `receive`
takes a whole block), and a tick past the open round's deadline or the end of the link ends the session `late` with the
open round unrecorded (the Python's `respond` always returns). Everything else is the Python's: the coins, the challenge,
the receipt time read before the hash, the first late round ending the session, the signature (`close`) and the
openings."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
from hypothesis import strategies as st

import mc_cases as M
import verity_difftest as D
from verity.experimental.protocols.pous import memory_challenger as mc

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

REFUSALS = (D.Reject, ValueError, TypeError, OverflowError)

#: how many traces each reference ran recorded a round answered before its challenge was sent, by reference
BACKWARDS = {"step": 0, "drive": 0}


def _backwards(ch: Challenger) -> bool:
    return any(r.elapsed_ns < 0 for r in ch.c.rounds)


class Challenger:
    """`memory_challenger.MemoryChallenger` behind `step`'s events, signed at the leaf a `reserved` event names."""

    def __init__(self, c) -> None:
        coins = mc.CoinTree.draw(M.hex_(c, "source"), M.hex_(c, "session"), M.nat(c, "blocks"), M.nat(c, "k"),
                                 key=M.hex_(c, "key"), salts=M.hexes(c, "salts"))
        height = M.nat(c, "height")
        if height not in M.ts.HEIGHTS:
            raise D.Reject(f"a test signer's height is in {M.ts.HEIGHTS}, not {height}")
        if coins.k >= 2 ** 32 or coins.blocks > 2 ** 64:
            raise D.Reject("a challenge is a u32 round and a u64 index")
        self.key, self.scheme, self.clock = M.test_key(M.hex_(c, "secret"), height), M.ts.scheme(height), M.Clock(0)
        self.c = mc.MemoryChallenger(coins, M.AtLeaf(self.key, 0, self.scheme), limit_ns=M.nat(c, "limit_ns"),
                                     block_bytes=M.nat(c, "block_bytes"), binding=M.hex_(c, "binding"), clock=self.clock)
        self.phase, self.buffer, self.refused = "fresh", b"", False

    def _end(self, how: str) -> list:
        self.c.ended, self.c._open, self.buffer, self.phase = how, None, b"", "signing"
        return [{"reserve": None}]

    def event(self, e: tuple) -> list:
        c = self.c
        if e[0] == "tick":
            if self.phase == "fresh":
                self.phase = "running"
                return [{"publish": {"commitment": c.commitment.to_json()}}]
            if self.phase == "running" and c._open is None:
                self.clock.now = e[1]
                j, i = len(c.rounds), c.challenge()
                return [{"send": (j.to_bytes(4, "big") + i.to_bytes(8, "big")).hex()}]
            if self.phase == "running" and e[1] > c._open[2] + c.commitment.limit_ns:
                return self._end(mc.LATE)
        elif e[0] == "received" and self.phase == "running" and c._open is not None:
            self.buffer += e[2]
            if len(self.buffer) >= c.commitment.block_bytes:
                self.clock.now = e[1]
                c.receive(self.buffer[:c.commitment.block_bytes])
                self.buffer = b""
                if c.done:
                    self.phase = "signing"
                    return [{"reserve": None}]
        elif e[0] == "closed":
            if self.phase == "fresh":
                self.phase = "done"
            elif self.phase == "running":
                return self._end(mc.LATE)
        elif e[0] in ("reserved", "refused") and self.phase == "signing":
            self.phase = "done"
            if e[0] == "refused":
                self.refused = True
                return []
            c.signer = M.AtLeaf(self.key, e[1], self.scheme)
            try:
                signed = c.close()
            except (ValueError, OverflowError):
                self.refused = True
                return []
            return [{"publish": {"openings": [o.to_json() for o in c.openings()], "session": signed.to_json()}}]
        return []


def _nat(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def event_of(e) -> tuple:
    """`MemoryChallenger.Difftest`'s reading of an event: the first of its forms that `e` is."""
    if isinstance(e, dict):
        if _nat(e.get("tick")):
            return ("tick", e["tick"])
        if "received" in e:
            return ("received", M.nat(e["received"], "now"), M.hex_(e["received"], "bytes"))
        if "closed" in e:
            return ("closed",)
        if _nat(e.get("reserved")):
            return ("reserved", e["reserved"])
        if isinstance(e.get("refused"), str):
            return ("refused", e["refused"])
    raise D.Reject("not an event")


def step(v) -> dict:
    ch = Challenger(M._get(v, "config"))
    events = [event_of(e) for e in M.list_(v, "events")]
    out = {"actions": [ch.event(e) for e in events], "refused": ch.refused}
    BACKWARDS["step"] += _backwards(ch)
    return out


@st.composite
def configs(draw, valid: bool = True) -> dict:
    k = draw(st.integers(1, 4))
    out = {"source": b"source".hex(), "session": hashlib.sha512(b"session %d" % draw(st.integers(0, 3))).digest()[:32].hex(),
           "blocks": draw(st.integers(1, 16)), "k": k, "key": draw(st.sampled_from(M.COIN_KEYS)).hex(),
           "salts": [hashlib.sha512(b"salt %d" % j).digest().hex() * 3 for j in range(k)],
           "limit_ns": draw(st.integers(1, 1000)), "block_bytes": draw(st.integers(1, 24)),
           "binding": draw(st.binary(max_size=4)).hex(), "secret": draw(st.sampled_from(M.SECRETS)).hex(),
           "height": draw(st.integers(1, 2))}
    if not valid:
        out |= draw(st.sampled_from([{"limit_ns": 0}, {"block_bytes": 0}, {"height": 0}, {"height": 11}, {"k": 0},
                                     {"salts": out["salts"][1:]}, {"session": "00"}, {"source": ""}]))
    return out


def _answer(ch: Challenger, draw, now: int, pace: int = 2) -> tuple[list[tuple], int]:
    """The open round's block (sometimes tampered, short or with bytes past it) in chunks, each later than the last by
    up to `limit_ns / pace`."""
    n = ch.c.commitment.block_bytes
    b = M.block(ch.c._open[1] if ch.c._open else 0, n)
    b = draw(st.sampled_from([b, b, b, b[:-1] + bytes([b[-1] ^ 1]), b[:n // 2], b + b"extra"]))
    cuts = sorted(draw(st.sets(st.integers(1, max(1, len(b) - 1)), max_size=3)))
    pieces = [b[i:j] for i, j in zip([0, *cuts], [*cuts, len(b)]) if b[i:j]]
    out = []
    for p in pieces:
        now += draw(st.integers(0, ch.c.commitment.limit_ns // pace + 1))
        out.append(("received", now, p))
    return out, now


def _early(ch: Challenger, draw) -> list[tuple]:
    """The open round's whole block, received up to 100 ns before its challenge was sent: a clock that ran backwards."""
    j, i, send = ch.c._open
    return [("received", max(0, send - draw(st.integers(1, 100))), M.block(i, ch.c.commitment.block_bytes))]


def _json(e: tuple) -> dict:
    if e[0] == "tick":
        return {"tick": e[1]}
    if e[0] == "received":
        return {"received": {"now": e[1], "bytes": e[2].hex()}}
    return {"closed": None} if e[0] == "closed" else {e[0]: e[1]}


@st.composite
def step_inputs(draw) -> dict:
    """A config and events drawn against the Python's own state as they go: ticks, the open round's answer in chunks,
    stray bytes, the link's end, and the leaf state's answers, the clock mostly moving forward; an honest run sometimes
    has one round answered before its send."""
    if draw(st.integers(0, 9)) == 9:
        return {"config": draw(configs(valid=False)), "events": [{"tick": 0}]}
    cfg = draw(configs())
    ch = Challenger(cfg)
    now, events = draw(st.integers(0, 2 ** 40)), []
    if draw(st.booleans()):
        early = draw(st.integers(0, 2 * cfg["k"] - 1))
        while ch.phase in ("fresh", "running"):
            if ch.c._open is None:
                new = [("tick", now)]
            elif ch.c._open[0] == early:
                new = _early(ch, draw)
            else:
                new, now = _answer(ch, draw, now, pace=16)
            for e in new:
                ch.event(e)
                events.append(_json(e))
    for _ in range(draw(st.integers(0, 24))):
        kind = draw(st.sampled_from(["tick"] * 4 + ["answer"] * 4 + ["stray", "closed", "reserved", "reserved", "refused",
                                                                      "late", "back", "early"]))
        if kind == "answer":
            new, now = _answer(ch, draw, now)
        elif kind == "early":
            new = _early(ch, draw) if ch.c._open is not None else [("tick", now)]
        elif kind == "stray":
            new = [("received", now, draw(st.binary(min_size=1, max_size=8)))]
        elif kind == "late":
            now += ch.c.commitment.limit_ns + draw(st.integers(1, 3))
            new = [("tick", now)]
        elif kind == "back":
            new = [("tick", max(0, now - draw(st.integers(1, 100))))]
        elif kind == "closed":
            new = [("closed",)]
        elif kind == "reserved":
            new = [("reserved", draw(st.integers(0, 2 ** int(cfg["height"]))))]
        elif kind == "refused":
            new = [("refused", "the leaf state is behind its journal")]
        else:
            now += draw(st.integers(0, 3))
            new = [("tick", now)]
        for e in new:
            ch.event(e)
            events.append(_json(e))
    return {"config": cfg, "events": events}


def port(fn, reference, inputs, exe="memory-challenger-difftest", **kw) -> D.Port:
    return D.Port(exe, fn, reference, inputs, package=M.PACKAGE, rejects=REFUSALS, **kw)


server = D.server_fixture(port("step", step, st.none()))
drive_server = D.server_fixture(port("drive", lambda v: v, st.none(), exe="memory-challenger-drive"))


MALFORMED_EVENTS = st.sampled_from([{"tick": -1}, {"tick": "1"}, {"received": {"now": 1}}, {"received": 1}, {"reserved": -1},
                                    {"refused": 1}, {}, [], None])


@pytest.mark.slow
def test_step(server):
    """Sessions run to completion, ended late by a slow answer, an answer received before its send, a tick past the
    deadline or the link's end, signed at a leaf or refused one, and events no session expects."""
    BACKWARDS["step"] = 0
    inputs = st.one_of(step_inputs(), step_inputs(), step_inputs(), step_inputs(),
                       st.builds(lambda v, e, i: v | {"events": v["events"][:i] + [e] + v["events"][i:]}, step_inputs(),
                                 MALFORMED_EVENTS, st.integers(0, 30)))
    stats = D.run(port("step", step, inputs), examples=300, server=server)
    assert stats.outputs > 200, stats
    assert BACKWARDS["step"] > 20, BACKWARDS


def test_a_session_signs_once_after_it_ends(server):
    """Honest rounds to the end, then a reserved leaf: one commitment before any challenge, one challenge per round, one
    reservation, one signed session, which `verdict` accepts; a second reservation does nothing."""
    cfg = {"source": b"source".hex(), "session": bytes(32).hex(), "blocks": 8, "k": 3, "key": M.COIN_KEYS[0].hex(),
           "salts": [bytes([j]).hex() * 192 for j in range(3)], "limit_ns": 100, "block_bytes": 16, "binding": "",
           "secret": M.SECRETS[0].hex(), "height": 1}
    ch, events = Challenger(cfg), []
    for e in [("tick", 10), ("tick", 10)]:
        ch.event(e)
        events.append(e)
    for t in (20, 30, 40):
        new = [("received", t, M.block(ch.c._open[1], 16)), ("tick", t)]
        for e in new:
            ch.event(e)
            events.append(e)
    events += [("reserved", 1), ("reserved", 0)]
    v = {"config": cfg, "events": [_json(e) for e in events]}
    lean = D.replay(port("step", step, st.none()), v, server)
    assert lean == port("step", step, st.none()).python(v), lean
    actions = json.loads(lean[1])["actions"]
    kinds = [next(iter(a)) for acts in actions for a in acts]
    assert kinds == ["publish", "send", "send", "send", "reserve", "publish"], kinds
    published = actions[-2][0]["publish"]
    signed, openings = M.signed_of(published["session"]), [M.opening_of(o) for o in published["openings"]]
    assert signed.ended == mc.COMPLETE and [r.elapsed_ns for r in signed.rounds] == [10, 10, 10]
    expected = [mc.block_digest(M.block(i, 16)) for i in range(8)]
    v = mc.verdict(signed, signed.commitment, openings, expected.__getitem__, trusted={signed.public_key},
                   schemes=M.ts.SCHEMES)
    assert v.accepted, v


# --- the driver ----------------------------------------------------------------------------------------------------------

def _chunks(r) -> list[tuple[int, bytes]]:
    if not isinstance(r, list):
        raise D.Reject("a round is an array of chunks")
    return [(M.nat(c, "at"), M.hex_(c, "bytes")) for c in r]


def drive(v) -> dict:
    """What `drive` does over `MemoryWorld.scripted`: the events it feeds `step`, in its order, given that the peer answers
    a round only once its challenge is sent and closes the link when its script for the round runs out short."""
    ch = Challenger(M._get(v, "config"))
    now = M.nat(v, "start")
    rounds = [_chunks(r) for r in M.list_(v, "rounds")]
    leaf = M._get(v, "leaf")
    leaf = ("refused", leaf["refused"]) if isinstance(leaf, dict) and isinstance(leaf.get("refused"), str) else (
        "reserved", M.nat(v, "leaf"))
    sent, published, revealed, delivered = [], [], 0, 0

    def feed(e):
        nonlocal revealed, delivered
        for a in ch.event(e):
            if "send" in a:
                sent.append(a["send"])
                revealed, delivered = revealed + 1, 0
            elif "publish" in a:
                published.append(a["publish"])
            else:
                feed(leaf)

    while ch.phase in ("fresh", "running"):
        if ch.phase == "fresh" or ch.c._open is None:
            feed(("tick", now))
            continue
        chunks = rounds[revealed - 1] if revealed - 1 < len(rounds) else []
        if delivered < len(chunks):
            now, b = chunks[delivered]
            delivered += 1
            feed(("received", now, b))
        else:
            feed(("closed",))
    BACKWARDS["drive"] += _backwards(ch)
    return {"sent": sent, "published": published, "refused": ch.refused}


@st.composite
def drive_inputs(draw) -> dict:
    """A config, and a peer that answers each round in chunks, the last completing the block (sometimes late, tampered,
    or before the round's challenge was sent, whose send time is the clock's reading when the last round's answer came
    in), or a last round cut short; then a leaf, past the key's last sometimes, or a refusal."""
    cfg = draw(configs())
    n, limit, start = cfg["block_bytes"], cfg["limit_ns"], draw(st.integers(0, 2 ** 40))
    now, rounds = start, []
    for j in range(draw(st.integers(0, cfg["k"] + 1))):
        b = draw(st.sampled_from([M.block(draw(st.integers(0, cfg["blocks"] - 1)), n), bytes(n), bytes(n) + b"extra"]))
        short = draw(st.integers(0, 9)) == 9
        if short:
            b = b[:n - 1]
        cuts = sorted(draw(st.sets(st.integers(1, max(1, n - 1)), max_size=2))) if n > 1 else []
        pieces = [b[i:k] for i, k in zip([0, *cuts], [*cuts, len(b)]) if b[i:k]]
        chunks, early = [], draw(st.integers(0, 2)) == 2
        if early:
            now = max(0, now - draw(st.integers(1, 100)))
        for p in pieces:
            if not early:
                now += draw(st.one_of(st.integers(0, limit // 3 + 1), st.just(limit + 1)))
            chunks.append({"at": now, "bytes": p.hex()})
        rounds.append(chunks)
        if short:
            break
    leaf = draw(st.one_of(st.integers(0, 2 ** cfg["height"]), st.just({"refused": "held by another signer"})))
    return {"config": cfg, "start": start, "rounds": rounds, "leaf": leaf}


def test_drive(drive_server):
    """The driver's order of events over a scripted link and clock: the commitment published before the first challenge, a
    challenge's send time read before it is sent, an answer's receipt time after its bytes, the session signed once, and
    a round whose receipt reads before its send ending the session late."""
    BACKWARDS["drive"] = 0
    stats = D.run(port("drive", drive, drive_inputs(), exe="memory-challenger-drive", need_rejects=False), examples=100,
                  server=drive_server)
    assert stats.outputs == 100, stats
    assert BACKWARDS["drive"] > 5, BACKWARDS
