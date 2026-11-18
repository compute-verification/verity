"""The reference prover under refresh against PoUB's Lean verifier (`Poub.Session`, served by `poub-verifier`), driven
event by event as the session verifier drives it and with no Python books beside it: every append in its slot of the
public schedule, each re-key of its origin whole, every plan the schedule's, every opening naming its origin's position,
and the Lean accepts each window; an opening at another origin is rejected; and with refresh off the same prover's
session, which names no slot, is accepted as before."""

from __future__ import annotations

import dataclasses
import hashlib
import queue
import shutil

import pytest

from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.verifier.rota import Schedule
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, coins, receipts, wire
from verity.protocols.resource_accounting.memory.poub.verifier.session import lean_verifier as lv
from verity.protocols.resource_accounting.memory.poub.verifier.session import wire as session_wire

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

N, LIFETIME, WINDOWS, PASSES, FRESH = 8, 2, 7, 2, 2
NODE, SOURCE = b"refresh-node", b"the verifier's secret"
P512 = codec.Codec(codec.P512, 2, False)
WINDOW_NS, DEADLINE_NS, CLOSE_NS = 10, 10**6, 10**9


class Entropy:
    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def payload(label: bytes) -> bytes:
    return b"".join(hashlib.sha512(label + b"/%d" % u).digest()[:63] + b"\0" for u in range(codec.BLOCK_BYTES // 64))


def salts(label: bytes, n: int) -> list[bytes]:
    e = Entropy(label)
    return [e(receipts.SALT_BYTES) for _ in range(n)]


def runs_of(indices) -> list[list[int]]:
    out: list[list[int]] = []
    for j in sorted(indices):
        if out and out[-1][0] + out[-1][1] == j:
            out[-1][1] += 1
        else:
            out.append([j, 1])
    return out


@pytest.fixture(scope="module")
def exe():
    return lv.build()


class Lean:
    """One session of the Lean verifier's process, its coins from a challenger in this process."""

    def __init__(self, exe, store: audit_prover.Store, refresh: bool) -> None:
        reg = store.registration
        self.done: queue.Queue = queue.Queue()
        self.lean = lv.Lean(exe, self.done.put)
        self.lean.call({"op": "init", "source": SOURCE.hex(), "registry": audit.Registry().to_json(), "chunk": 64})
        got = self.lean.call(lv.register_request(NODE, N, reg.payload_root, P512, reg.receipt, reg.rows, bytes(32)))
        assert got == {"codec_seed": reg.codec_seed.hex(), "digest": reg.digest.hex()}
        params = audit.Params(blocks=N, deadline_ns=DEADLINE_NS, windows=WINDOWS, samples=10**6, delta=1e-6,
                              window_ns=WINDOW_NS, codec=P512, lifetime=LIFETIME)
        named = {"refresh": True, "fresh_slots": FRESH} if refresh else {}
        assert self.lean.call({"op": "session", "node": NODE.hex(), "params": lv.params_json(params) | named,
                               "coins": None, "append_coins": [], "not_before_ns": 0, "index": 0}) == {"session": 0}
        self.coins = coins.local(reg.digest, 0, WINDOWS, Entropy(b"coins"))
        self.now, self.latest, self.checks = 0, reg.digest, 0

    def event(self, ev: dict, coin: bytes | None = None, *, at: int | None = None) -> dict:
        if at is not None:
            self.now = max(self.now + 1, at)
            ev["now"] = self.now
        return self.lean.call({"op": "event", "node": NODE.hex(), "session": 0, "event": ev,
                               "coin": None if coin is None else coin.hex()})

    def verdict(self, t: int, openings) -> dict:
        body = wire.encode_openings(openings)
        self.lean.call({"op": "openings", "node": NODE.hex(), "session": 0, "t": t, "bytes": len(body)}, body)
        self.checks += 1
        assert self.lean.call({"op": "verify", "node": NODE.hex(), "session": 0, "t": t, "id": self.checks}) == {
            "spawned": True}
        self.done.get()
        return self.lean.call({"op": "settle", "id": self.checks})


def session(exe, refresh: bool = True, freed: dict[int, list[int]] | None = None, tamper=None) -> tuple[list, dict, list, list]:
    """W windows of L = 2 over N registered blocks, each opened by its due re-keys, pages in windows 0, 1 (two appends,
    F = 2) and 2, `PASSES` passes a window over the refresh plan (every block without refresh); `freed` names the chunks
    freed at the start of a window.  Every receipt is sampled and opened, `tamper` rewriting each opening.  The windows'
    settled verdicts, the session's outcome, the appends sent as (slot, origin, count) and the blocks withdrawn.
    Without refresh the session appends nothing."""
    payloads = [payload(b"weights %d" % j) for j in range(N)]
    ys = salts(b"payload salts", N)
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    s = audit_prover.Store(audit.register(SOURCE, NODE, N, root, P512, nonce=bytes(32)), payloads, ys)
    lean = Lean(exe, s, refresh)
    r = audit_prover.Refresh(s, Schedule(N, LIFETIME, WINDOWS, FRESH)) if refresh else None
    prover = audit_prover.Prover(s, None, entropy=Entropy(b"salts"))
    challenge = lean.coins.challenge(0, lean.latest)
    assert lean.event({"kind": "start"}, coins.coin(challenge), at=0) == {"released": 0}
    challenges, head, step, passes, sent, plan_sent, withdrawn = [challenge], None, -1, {}, [], receipts.Plan.whole(N), []

    def send(a: audit_prover.Append, t: int) -> None:
        nonlocal head
        if r is not None:
            a = r.sent(a, t)
        ev = {"kind": "append", "root": a.root.hex(), "count": a.count}
        ev |= {} if a.slot is None else {"slot": a.slot}
        ev |= {} if a.origin is None else {"origin": session_wire.origin_json(a.origin)}
        coin = coins.coin(lean.coins.append(a.root))
        assert lean.event(ev, coin) == {"coin": coin.hex(), "first": a.first}, (t, ev)
        head = audit.append_head(head, a.first, a.count, a.root)
        if a.origin is not None:
            s.encode(a.first, coin)
        sent.append((a.slot, a.origin, a.count))

    def release(old, gone=()) -> None:
        withdrawn.extend(gone)
        if old or gone:
            assert lean.event(lv.release_event(runs_of([*old, *gone]), runs_of(gone))) == {"ok": True}

    for t in range(WINDOWS):
        if r is not None:
            for q in (freed or {}).get(t, []):
                release([r.freed(q)])
            for k, origin, count in r.due(t):
                a = s.rekey(origin, count, k)
                old, gone = r.appended(a)
                send(a, t)
                release(old, gone)
        for count in {0: (3,), 1: (2, 1), 2: (2,)}.get(t, ()) if r is not None else ():
            a = s.append([payload(b"page %d" % (s.extent + u)) for u in range(count)],
                         salts(b"page salts %d" % s.extent, count))
            r.appended(a)
            send(a, t)
        for i in range(PASSES):
            plan = r.plan(t) if r is not None else receipts.Plan.whole(N)
            ps = prover.run(challenges[t], i, lean.latest, plan, head=head)
            step += 1
            change = [runs_of(set(plan) - set(plan_sent)), runs_of(set(plan_sent) - set(plan))]
            assert lean.event(lv.receive_event(t, i, ps.root, change, step), at=lean.now) == {"ok": True}, (t, i)
            passes[t, i], lean.latest, plan_sent = ps, ps.root, plan
        if t + 1 < WINDOWS:
            challenge = lean.coins.challenge(t + 1, lean.latest)
            challenges.append(challenge)
        coin = coins.coin(challenge) if t + 1 < WINDOWS else None
        assert lean.event({"kind": "boundary"}, coin, at=lean.now + WINDOW_NS) == {
            "released": t + 1 if t + 1 < WINDOWS else None}
    verdicts = []
    for t in range(WINDOWS):
        coin = coins.coin(lean.coins.openings(t, passes[t, PASSES - 1].root))
        drew = lean.event({"kind": "close", "t": t}, coin, at=CLOSE_NS + t)
        opened = [prover.open(passes[t, i], q) for i, q in drew["draws"]]
        verdicts.append(lean.verdict(t, [tamper(s, o) if tamper else o for o in opened]))
        if "accepted" not in verdicts[-1]:
            break
    outcome = lean.lean.call({"op": "outcome", "node": NODE.hex(), "session": 0})
    lean.lean.close()
    return verdicts, outcome, sent, withdrawn


def test_the_lean_accepts_a_refresh_session_of_the_reference_prover(exe):
    verdicts, outcome, sent, withdrawn = session(exe)
    assert [v.get("accepted") for v in verdicts] == list(range(WINDOWS)), verdicts
    assert outcome["outcome"] == "accepted" and outcome["record"]["refresh"] is True
    slots = [k for k, _, _ in sent]
    assert slots == sorted(set(slots))
    kinds = {o[0] if o else "fresh" for _, o, _ in sent}
    assert kinds == {"fresh", "registered", "append"} and not withdrawn


def test_the_lean_accepts_a_refresh_session_whose_rekeys_withdraw_freed_chunks(exe):
    """A page freed before its re-key and registered blocks freed before theirs leave each re-key whole, with the
    freed chunks' blocks withdrawn."""
    verdicts, outcome, _, withdrawn = session(exe, freed={2: [N + 1], 3: [5, 6]})
    assert outcome["outcome"] == "accepted", (verdicts, outcome)
    assert len(withdrawn) == len(range(2, WINDOWS, LIFETIME)) + 2 * len(range(3, WINDOWS, LIFETIME))


def test_the_lean_rejects_a_rekey_s_opening_at_another_origin(exe):
    def rewrite(_s, o: audit.Opening) -> audit.Opening:
        return o if o.origin_position is None else dataclasses.replace(o, origin_position=o.origin_position ^ 1)
    verdicts, outcome, *_ = session(exe, tamper=rewrite)
    assert [("accepted" in v) for v in verdicts] == [True, False] and outcome["outcome"] != "accepted"


def test_with_refresh_off_the_reference_prover_s_session_is_accepted_and_names_no_slot(exe):
    verdicts, outcome, sent, _ = session(exe, refresh=False)
    assert outcome["outcome"] == "accepted" and "refresh" not in outcome["record"], (verdicts, outcome)
    assert all(k is None and o is None for k, o, _ in sent)
