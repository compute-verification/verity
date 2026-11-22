"""The Python's books of a P2 session, the reference `test_p2.py` holds `Poub.P2Session` to: `audit.AuditRun` over the
P2 codec, each event's result as the Lean writes it (`Difftest.resultJ`), and the session's answer (`answer`,
`P2Session.answer`). The verifier's party (`p2_parties.py`) runs on the Lean alone."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verity.protocols.accounting.memory.poub.verifier import audit, p2, pooled
from verity.protocols.accounting.memory.poub.verifier import coins as C
from verity.protocols.accounting.memory.poub.verifier import period as P
from verity.protocols.accounting.memory.poub.verifier.p2_parties import _HEX, PROTOCOL, message, opening_of
from verity.protocols.accounting.memory.poub.verifier.receipts import Plan

REFUSED = {"refused": True}


def held_json(r: p2.Registration, plan: Plan) -> dict[str, Any]:
    """What the Lean answers on holding the store (`P2Session.Held.json`): its digest, units and plan."""
    return {"digest": r.digest.hex(), "plan": [list(run) for run in plan.runs], "plan_units": plan.length,
            "units": r.blocks}


def verdict_json(v: audit.Verdict) -> dict[str, Any]:
    """A verdict as the Lean writes a result (`Difftest.resultJ`)."""
    return {"accepted": v.window} if v.accepted else {
        "at": None if v.at is None else list(v.at), "failure": v.failure, "window": v.window}


def _done(v: audit.Verdict | None) -> dict[str, Any]:
    return {"ok": True} if v is None else verdict_json(v)


def answer(s: audit.AuditRun, reason: str | None = None, plan: Plan | None = None) -> dict[str, Any]:
    """The session's answer (`P2Session.answer`, with `reason`): `accept` with the credited reads, `reject` with the
    failure and the sample it failed at, or `undecided`; `units` are the store's and `plan_units` those each pass
    reads."""
    reg = s.registration
    out: dict[str, Any] = {"protocol": PROTOCOL, "registration": reg.digest.hex(), "unit_bytes": p2.UNIT_BYTES,
                           "units": reg.blocks, "plan_units": (plan or Plan.whole(reg.blocks)).length, "windows": [],
                           "reason": reason}
    for w in s.windows:
        out["windows"].append({"passes": len(w.roots), "draws": None if w.draws is None else len(w.draws),
                               "verdict": None if w.verdict is None else
                               {"accepted": w.verdict.accepted, "failure": w.verdict.failure, "at": w.verdict.at}})
    outcome = s.outcome
    if outcome is None:
        return out | {"verdict": "undecided"}
    if outcome != "accepted":
        f = s.failed
        return out | {"verdict": "reject", "failure": f.failure, "at": f.at, "window": f.window}
    record = s.record()
    ws, delta = audit.pooled_windows(record)
    us = sum(u for u, _ in ws)
    try:
        b = pooled.bound(ws, delta)
        wrong = -(-b.wrong.numerator // b.wrong.denominator)
        bound: dict[str, Any] = b.to_json()
    except pooled.Refused as e:
        wrong, bound = us, {"refused": str(e)}
    q_bytes = (us - wrong) * p2.UNIT_BYTES
    span = record.span_ns
    return out | {"verdict": "accept", "receipts": us, "wrong_at_most": wrong, "delta": str(delta), "bound": bound,
                  "draws": [s for _, s in ws], "q_bytes": q_bytes, "span_ns": span,
                  "credited_bytes_per_s": q_bytes * 10**9 // span if span > 0 else None}


class _Out:
    """A line out of form."""


_OUT = _Out()


class Books:
    """The Python's books of one P2 session: `audit.AuditRun` over the P2 codec on the clock each event sets
    (`period._Held`), and each event's result as the Lean writes it (`Difftest.resultJ`), a call the session refuses
    `REFUSED`. A root or openings line is read as `P2Session` reads it: a line out of form rejects the session
    `malformed` at the window filling (`fail`), and `why` says what was out of form. `released` is the coin the last
    event released: a start's or a boundary's challenge, or a close's sample's coins."""

    def __init__(self, params: audit.Params, registration: p2.Registration, coins_: C.SessionCoins, plan: Plan,
                 not_before_ns: int) -> None:
        self.clock, self.plan = P._Held(), plan
        self.s = audit.AuditRun(params, registration, None, coins_, not_before_ns=not_before_ns, clock=self.clock)
        self.released: C.LiveOpening | None = None
        self.sample: bytes | None = None
        """A close's: its sample's coins, whole, when the close released them."""
        self.why: str | None = None

    def _at(self, now: int, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        self.clock.now, self.released, self.sample = now, None, None
        try:
            return call()
        except audit.SessionError:
            return dict(REFUSED)

    def _line(self, line: bytes, key: str) -> Any:
        try:
            return message(line, key)
        except (ConnectionError, ValueError) as e:
            self.why = f"the prover's side failed: {type(e).__name__}: {e}"
            return _OUT

    def start(self, now: int) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            self.released = self.s.start()
            return {"released": self.released.index}
        return self._at(now, call)

    def boundary(self, now: int) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            self.released = c = self.s.boundary()
            return {"released": None if c is None else c.index}
        return self._at(now, call)

    def root(self, t: int, i: int, line: bytes, now: int) -> dict[str, Any]:
        got = self._line(line, "root")
        if got is _OUT:
            return self.fail()
        root = bytes.fromhex(got) if type(got) is str and _HEX.fullmatch(got) else b""
        return self._at(now, lambda: _done(self.s.receive(t, i, root, self.plan.runs, step=self.s._step + 1)))

    def close(self, t: int, now: int) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            c = t // self.s.params.groups_per_coin
            had = self.s.coins.sample_bytes(c)
            out = self.s.close(t)
            got = self.s.coins.sample_bytes(c)
            self.sample = got if had is None else None
            return verdict_json(out) if isinstance(out, audit.Verdict) else {"draws": [list(d) for d in out]}
        return self._at(now, call)

    def openings(self, t: int, line: bytes) -> dict[str, Any]:
        got = self._line(line, "openings")
        if got is _OUT:
            return self.fail()
        openings = [opening_of(x) for x in got] if isinstance(got, list) else []
        if any(o is None for o in openings):
            openings = []
        return self._at(self.clock.now, lambda: verdict_json(self.s.verify(t, openings)))

    def fail(self) -> dict[str, Any]:
        """The session rejected `malformed` at the window filling (the prover's line out of form, or the prover gone),
        or `REFUSED` once it has its outcome."""
        self.released = None
        if self.s.outcome is not None:
            return dict(REFUSED)
        return verdict_json(self.s._fail(self.s.filling, "malformed"))
