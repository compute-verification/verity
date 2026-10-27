"""The verifier and its challenger of a remote bandwidth session over PoUS's P2 store (`p2.py`) as a party of their own,
so they run on a host the prover doesn't control (`trusted-host`) while the prover serves on its node.

The channel is a pair of byte streams carrying one JSON object of one key a line: the ssh session that starts this side
(`--stdio`, as a gate's verb starts it) or a TCP connection (`--listen`). The session has no round-trip floor, so ssh's
latency changes nothing in its verdict. The verifier holds its own registration (`--registration FILE`, the auditor's
record of PoUS's setup, `registration_json`), never the prover's word for it, and draws its coins from its own host's OS.
The record's `plan` is the units every pass reads (`plan_of`): the units holding a tensor's block, so not the store's
padding past the last tensor, which is PoUS's and which the decode never reads; without one, every unit.

- The prover opens with the request `{"request": {"deadline_ns", "node", "passes", "samples", "window_ns",
  "windows"}}`, whose node must be the registration's.
- The verifier answers `{"hold": {"registration"}}`: the registration with its plan, which the prover checks against
  its store and reads each pass in.
- It then drives the session on its clock (`session`): `{"run": {"challenge", "i", "previous", "t"}}` asks for pass i
  of window t under the window's challenge, its value and the root it was opened after (`coins.Released`), after the
  root `previous`, answered `{"root": hex}` when the step that read it is done. Once a window's roots are final (the
  next window's first root, or its deadline for the last) it sends `{"open": {"draws", "t"}}`, answered `{"openings":
  [opening_json, ...]}`, and verifies them while the prover serves on.
- Once the session has its outcome, `{"end": {"commitment", "openings"}}`: the session's coin commitment and every
  released coin's opening (`audit.Session.disclose`), against which the prover checks each challenge it was sent
  (`audit.Prover.end`). The commitment is a function of every coin, so the prover gets it only after its last root.
- `{"answer": {...}}` ends the session (`answer`): the verdict, the failure and where, and on acceptance the credited
  reads `q_bytes`, the receipts less the wrong ones the pooled bound leaves (`pooled.bound` at δ = 2^-40), a unit's
  8,224 bytes each, over the session's span.

The verifier is the Lean, since the trusted base is the compiled Lean verifier, which parses bytes of
untrusted provenance, and Python is never trusted: the process `remote-audit-verifier` (`p2_lean.py`,
`RemoteAudit.P2Session`) holds the registration file's bytes, recomputes the registration's digest from its fields, and
reads the prover's request and every root and openings line as the prover sent it, each opening's four blocks checked
on the registered vk root, and its answer is the session's. The Python's books (`Books`, `p2.Session`) take every
event beside it and can only veto (`Twin`): an event, a held store or an answer on which the two differ, or one the
Lean answers with an error, rejects the session `disagreement` at that window, and no event follows it. What the
Lean takes as given is the clock's readings and the coins the challenger (`coins.local`, on this host) releases.

`python -m verity.core.protocols.remote_bandwidth.p2_party --registration FILE (--stdio | --listen HOST:PORT [--sessions
N]) [--answer FILE | --device DIR] [--lean-exe EXE]` is the verifier's side (`main`), each session with fresh coins, the
Lean built in its package (`lake build remote-audit-verifier`) unless `--lean-exe` names a built one, or, without `lake`
on the PATH, the one already built there (`p2_lean.exe`). Listening, it prints `{"listening": port}` and then each
session's `{"answer"}` on stdout, the shape of a party the auditor's gate starts over ssh with its `--device` directory
(`benchmarks/live_verifier/host.py`'s party verbs). `prove(reader, writer, prover)` is the prover's: it answers each
message with `prover`'s `hold`, `run`, `open` and `end` until the answer, which it returns.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from verity.core.protocols.pous.schemes.p2 import P2Params

from . import audit, p2, p2_lean, pooled
from . import coins as C
from . import period as P
from .p2_lean import Lean, LeanError
from .receipts import Plan

MODULE = "verity.core.protocols.remote_bandwidth.p2_party"
PROTOCOL = "verity/remote-bandwidth/p2/v0"
REQUEST_KEYS = ("deadline_ns", "node", "passes", "samples", "window_ns", "windows")
REFUSED = {"refused": True}
DISAGREEMENT = "disagreement"
#: The draws a thread of the Lean checks at once (`P2Session.checkOpeningsChunked`).
CHUNK = 64
_HEX = re.compile(r"(?:[0-9a-f]{2})*")
_ANSWER_ERRORS = (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, audit.SessionError)


def send(f: BinaryIO, doc: Mapping[str, Any]) -> None:
    f.write(json.dumps(dict(doc), sort_keys=True).encode() + b"\n")
    f.flush()


def message(line: bytes, key: str | None = None) -> Any:
    """The message `line` carries (its one key `key`'s value, if given; `P2Session.messageOf`): ConnectionError for a
    line its sender didn't end, ValueError for one that isn't UTF-8 JSON, an object of one key (`key`)."""
    if not line.endswith(b"\n"):
        raise ConnectionError("the peer closed mid-session")
    try:
        doc = json.loads(line.decode())
    except RecursionError:
        raise ValueError("a message nests too deep") from None
    if not isinstance(doc, dict) or len(doc) != 1:
        raise ValueError("a message is a JSON object of one key")
    if key is None:
        return doc
    if key not in doc:
        raise ValueError(f"expected {key!r}, got {next(iter(doc))!r}")
    return doc[key]


def recv(f: BinaryIO, key: str | None = None) -> Any:
    """The next message's value (its one key `key`, if given)."""
    return message(f.readline(), key)


def _unhex(x: Any) -> bytes:
    """Lowercase hex's bytes (`Flock.unhex`), else ValueError."""
    if type(x) is not str or not _HEX.fullmatch(x):
        raise ValueError("not lowercase hex")
    return bytes.fromhex(x)


def _hexes(xs: Any) -> tuple[bytes, ...]:
    if type(xs) is not list:
        raise ValueError("not a list")
    return tuple(_unhex(x) for x in xs)


def _nat(doc: Mapping[str, Any], k: str) -> int:
    v = doc[k]
    if type(v) is not int or v < 0:
        raise ValueError(f"{k} is a natural number")
    return v


def registration_json(r: p2.Registration, plan: Plan | None = None) -> dict[str, Any]:
    c = r.codec
    out = {"blocks_per_segment": c.blocks_per_segment, "c0": c.c0, "commitment": r.commitment.hex(),
           "digest": r.digest.hex(), "expander": c.expander, "node": r.node.hex(), "root": r.root.hex(),
           "rows": r.rows, "salt": r.salt.hex(), "segments": c.segments, "w": c.w}
    return out if plan is None else out | {"plan": [list(run) for run in plan.runs]}


def plan_of(doc: Mapping[str, Any], reg: p2.Registration) -> Plan:
    """The plan of `registration_json`'s record `doc`, its `plan` runs `[first, count]` over `reg`'s units, or every
    unit without one; refused (ValueError) unless it is a plan of them."""
    runs = doc.get("plan")
    if runs is None:
        return Plan.whole(reg.blocks)
    if not isinstance(runs, list) or not all(isinstance(r, list) and len(r) == 2 and all(type(x) is int for x in r)
                                             for r in runs):
        raise ValueError("a registration's plan is a list of runs [first, count]")
    return Plan(tuple(tuple(r) for r in runs), reg.blocks)


def registration_of(doc: Mapping[str, Any]) -> p2.Registration:
    """`registration_json`'s inverse (`P2Session.heldOf`), refused if its `digest` (when given) isn't the fields'."""
    if type(doc["expander"]) is not str:
        raise ValueError("expander is a string")
    params = P2Params(_nat(doc, "w"), _nat(doc, "c0"), _nat(doc, "blocks_per_segment"), _nat(doc, "segments"),
                      doc["expander"])
    reg = p2.Registration(_unhex(doc["node"]), p2.codec_of(params), _unhex(doc["commitment"]), _unhex(doc["salt"]),
                          _unhex(doc["root"]), _nat(doc, "rows"))
    if "digest" in doc and _unhex(doc["digest"]) != reg.digest:
        raise ValueError("the registration's digest is not its fields'")
    return reg


def opening_json(o: p2.Opening) -> dict[str, Any]:
    return {"blocks": [b.hex() for b in o.blocks], "leaf": o.leaf.hex(), "leaf_salt": o.leaf_salt.hex(),
            "pass": o.pass_index, "paths": [[x.hex() for x in path] for path in o.paths], "position": o.position,
            "receipt_path": [x.hex() for x in o.receipt_path]}


def opening_of(doc: Any) -> p2.Opening | None:
    """`opening_json`'s inverse (`P2Session.openingOf`), None for anything else (the session rejects it
    `malformed`)."""
    try:
        if (not isinstance(doc, dict) or type(doc["pass"]) is not int or type(doc["position"]) is not int
                or type(doc["paths"]) is not list):
            return None
        return p2.Opening(doc["pass"], doc["position"], _hexes(doc["blocks"]),
                          tuple(_hexes(path) for path in doc["paths"]), _unhex(doc["leaf"]),
                          _unhex(doc["leaf_salt"]), _hexes(doc["receipt_path"]))
    except (KeyError, ValueError):
        return None


def request_of(node: bytes, *, windows: int, passes: int, samples: int, window_ns: int,
               deadline_ns: int) -> dict[str, Any]:
    return {"deadline_ns": deadline_ns, "node": node.hex(), "passes": passes, "samples": samples,
            "window_ns": window_ns, "windows": windows}


def params_of(reg: p2.Registration, request: Mapping[str, Any]) -> audit.Params:
    """The session's parameters from the prover's request (`P2Session.paramsOf`), refused unless its node is the
    registration's and the rest are positive integers."""
    if not isinstance(request, dict) or sorted(request) != sorted(REQUEST_KEYS):
        raise ValueError(f"a request is {list(REQUEST_KEYS)}")
    if _unhex(request["node"]) != reg.node:
        raise ValueError("the request's node is not the registration's")
    if any(type(request[k]) is not int or request[k] < 1 for k in REQUEST_KEYS if k != "node"):
        raise ValueError("a request's deadline_ns, passes, samples, window_ns and windows are positive integers")
    return audit.Params(blocks=reg.blocks, deadline_ns=request["deadline_ns"], windows=request["windows"],
                        samples=request["samples"], min_passes=request["passes"], window_ns=request["window_ns"],
                        codec=reg.codec, receipt=reg.receipt, layout=reg.layout, rows=reg.rows)


def verdict_json(v: audit.Verdict) -> dict[str, Any]:
    """A verdict as the Lean writes a result (`Difftest.resultJ`)."""
    return {"accepted": v.window} if v.accepted else {
        "at": None if v.at is None else list(v.at), "failure": v.failure, "window": v.window}


def _done(v: audit.Verdict | None) -> dict[str, Any]:
    return {"ok": True} if v is None else verdict_json(v)


def answer(s: p2.Session, reason: str | None = None, plan: Plan | None = None) -> dict[str, Any]:
    """The session's answer (`P2Session.answer`, with `reason`): `accept` with the credited reads, `reject` with the
    failure and the draw it failed at, or `undecided`; `units` are the store's and `plan_units` those each pass
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
    """The Python's books of one P2 session, the veto beside the Lean (`Twin`): `p2.Session` on the clock each event
    sets (`period._Held`), and each event's result as the Lean writes it (`Difftest.resultJ`), a call `p2.Session`
    refuses `REFUSED`. A root or openings line is read as `P2Session` reads it: a line out of form rejects the session
    `malformed` at the window filling (`fail`), and `why` says what was out of form. `released` is the coin the last
    event released: a start's or a boundary's challenge, or a close's openings coin."""

    def __init__(self, params: audit.Params, registration: p2.Registration, coins_: C.Coins, plan: Plan,
                 not_before_ns: int) -> None:
        self.clock, self.plan = P._Held(), plan
        self.s = p2.Session(params, registration, coins_, not_before_ns=not_before_ns, clock=self.clock)
        self.released: C.LiveOpening | None = None
        self.why: str | None = None

    def _at(self, now: int, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        self.clock.now, self.released = now, None
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
            n = len(self.s.coins.opened(C.OPENINGS))
            out = self.s.close(t)
            opened = self.s.coins.opened(C.OPENINGS)
            self.released = opened[-1] if len(opened) > n else None
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


class _Refused(Exception):
    """The session is refused before its first coin."""


class _Ended(Exception):
    """The session has its outcome: a root or a window rejected, or the verifiers disagree."""

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason


class Twin:
    """A P2 session through the Lean verifier (session `k` of `lean`, `RemoteAudit.P2Session`) with the
    Python's books beside it (`books`), which can only veto: each event goes to both and its result is the Lean's
    where the two agree; where they differ, or the Lean answers an error, the session is rejected `disagreement` at
    the event's window (`disagreement` says where and what each answered) and the event raises `_Ended`."""

    def __init__(self, lean: Lean, k: int, books: Books) -> None:
        self.lean, self.k, self.books = lean, k, books
        self.disagreement: dict[str, Any] | None = None

    @classmethod
    def open(cls, lean: Lean, registration: p2.Registration, plan: Plan, line: bytes, secret: bytes,
             not_before_ns: int) -> Twin:
        """The session the prover's request `line` asks for, on coins from `secret`: `_Refused` where the Lean refuses
        it, or the Python does where the Lean doesn't (the veto)."""
        try:
            got = lean.call({"op": "p2_session", "bytes": len(line), "not_before_ns": not_before_ns}, line)
        except LeanError as e:
            raise _Refused(f"the Lean verifier failed: {e}") from None
        if not isinstance(got, dict) or type(got.get("session")) is not int:
            why = got.get("refused") if isinstance(got, dict) else None
            raise _Refused(why if type(why) is str else f"the Lean verifier answered {got}")
        try:
            params = params_of(registration, message(line, "request"))
        except (ConnectionError, ValueError, KeyError, TypeError) as e:
            raise _Refused(f"{DISAGREEMENT}: the Lean verifier takes the request, the Python refuses it: "
                           f"{type(e).__name__}: {e}") from None
        coins_ = C.local(registration.digest, 0, params.windows, P._Stream(secret, registration.node))
        return cls(lean, got["session"], Books(params, registration, coins_, plan, not_before_ns))

    def _ask(self, request: dict[str, Any], body: bytes = b"") -> Any:
        try:
            return self.lean.call(request, body)
        except LeanError as e:
            return {"error": str(e)}

    def _veto(self, t: int, event: str, got: Any, mine: Any) -> None:
        self.disagreement = {"event": event, "lean": got, "python": mine, "t": t}
        self.books.s._fail(t, DISAGREEMENT)

    def _settle(self, t: int, event: str, got: Any, mine: dict[str, Any]) -> dict[str, Any]:
        if got != mine:
            self._veto(t, event, got, mine)
            raise _Ended(f"{DISAGREEMENT}: the Lean verifier and the Python differ on window {t}'s {event}")
        return got

    def _event(self, t: int, event: dict[str, Any], mine: dict[str, Any]) -> dict[str, Any]:
        c = self.books.released
        got = self._ask({"op": "p2_event", "session": self.k, "event": event,
                         "coin": None if c is None else C.coin(c).hex()})
        return self._settle(t, event["kind"], got, mine)

    def start(self, now: int) -> tuple[dict[str, Any], C.LiveOpening | None]:
        """The session's first coin at `now`: the result, and the challenge to send."""
        mine = self.books.start(now)
        return self._event(0, {"kind": "start", "now": now}, mine), self.books.released

    def boundary(self, now: int) -> tuple[dict[str, Any], C.LiveOpening | None]:
        """The next unended window's boundary at `now`: the result, and the next window's challenge, or None."""
        t = max(0, len(self.books.s.windows) - 1)
        mine = self.books.boundary(now)
        return self._event(t, {"kind": "boundary", "now": now}, mine), self.books.released

    def root(self, t: int, i: int, line: bytes, now: int) -> dict[str, Any]:
        """Root i of window t from the prover's `line`, arriving at `now`."""
        mine = self.books.root(t, i, line, now)
        got = self._ask({"op": "p2_root", "session": self.k, "t": t, "i": i, "now": now, "bytes": len(line)}, line)
        return self._settle(t, "root", got, mine)

    def close(self, t: int, now: int) -> dict[str, Any]:
        """Window t's draws at `now`."""
        mine = self.books.close(t, now)
        return self._event(t, {"kind": "close", "now": now, "t": t}, mine)

    def openings(self, t: int, line: bytes) -> dict[str, Any]:
        """Window t's openings from the prover's `line`: the Lean checks them while the books do."""
        got = None
        try:
            self.lean.send({"op": "p2_openings", "session": self.k, "t": t, "bytes": len(line)}, line)
        except LeanError as e:
            got = {"error": str(e)}
        mine = self.books.openings(t, line)
        if got is None:
            try:
                got = self.lean.read()
            except LeanError as e:
                got = {"error": str(e)}
        return self._settle(t, "openings", got, mine)

    def fail(self) -> dict[str, Any]:
        """The session rejected `malformed` at the window filling: the prover is gone."""
        t = self.books.s.filling
        mine = self.books.fail()
        got = self._ask({"op": "p2_fail", "session": self.k})
        if got != mine:
            self._veto(t, "fail", got, mine)
        return got

    def answer(self, reason: str | None) -> dict[str, Any]:
        """The session's answer, the Lean's (`P2Session.answer`) with `reason`, where the Python's is the same;
        otherwise, or after a disagreement, the session rejected `disagreement` (the Python's `answer` of its books,
        which a disagreement rejected), with the `disagreement`."""
        got = self._ask({"op": "p2_answer", "session": self.k})
        if self.disagreement is None:
            try:
                mine = json.loads(json.dumps(answer(self.books.s, plan=self.books.plan)))
                mine.pop("reason")
            except _ANSWER_ERRORS as e:
                mine = {"error": f"{type(e).__name__}: {e}"}
            if got == mine:
                return got | {"reason": reason}
            self._veto(self.books.s.filling, "answer", got, mine)
        d = self.disagreement
        why = reason or f"{DISAGREEMENT}: the Lean verifier and the Python differ on window {d['t']}'s {d['event']}"
        return answer(self.books.s, why, self.books.plan) | {"disagreement": d}


def held_json(r: p2.Registration, plan: Plan) -> dict[str, Any]:
    """What the Lean answers on holding the store (`P2Session.Held.json`): its digest, units and plan."""
    return {"digest": r.digest.hex(), "plan": [list(run) for run in plan.runs], "plan_units": plan.length,
            "units": r.blocks}


def hold(lean: Lean, data: bytes, chunk: int = CHUNK) -> tuple[p2.Registration, Plan]:
    """The registration file's bytes `data`, held by both verifiers: the Python's registration and plan
    (`registration_of`, `plan_of`), refused (ValueError) unless the Lean holds the same store, every `chunk` draws
    checked on a thread (`P2Session.heldOf`)."""
    doc = json.loads(data.decode())
    reg = registration_of(doc)
    plan = plan_of(doc, reg)
    try:
        got = lean.call({"op": "p2_hold", "bytes": len(data), "chunk": chunk}, data)
    except LeanError as e:
        raise ValueError(f"the Lean verifier refuses the registration: {e}") from None
    mine = held_json(reg, plan)
    if got != mine:
        raise ValueError(f"{DISAGREEMENT}: the Lean verifier holds {got}, the Python {mine}")
    return reg, plan


def session(f_in: BinaryIO, f_out: BinaryIO, registration: p2.Registration, *, coins: bytes, lean: Lean,
            plan: Plan | None = None, clock: Callable[[], int] = time.monotonic_ns,
            sleep_until: Callable[[Callable[[], int], int], None] = P._sleep_until,
            log: Callable[[str], None] = lambda _m: None, host: str | None = None) -> dict[str, Any]:
    """The verifier's side of one session over the channel (`f_in`, `f_out`), decided by `lean`, which holds
    `registration` and `plan` (`hold`; every unit by default), with the books beside it (`Twin`): the prover's
    request, the registration, each pass asked for in order under `plan`, each window's openings drawn once its roots
    are final and verified, the session's coins disclosed, and the answer, sent last and returned, naming the
    verifier's `host` (this one's name by default). A line out of form or a closed channel rejects the session (its
    answer says why)."""
    host = host or socket.gethostname()
    plan = Plan.whole(registration.blocks) if plan is None else plan
    line = f_in.readline()
    try:
        twin = Twin.open(lean, registration, plan, line, coins, clock())
    except _Refused as e:
        out = {"protocol": PROTOCOL, "registration": registration.digest.hex(), "verdict": "refused",
               "reason": str(e), "host": host}
        return _last(f_out, out)
    W, passes = twin.books.s.params.windows, twin.books.s.params.min_passes

    def opened(t: int) -> None:
        got = twin.close(t, clock())
        if "draws" not in got:
            raise _Ended
        send(f_out, {"open": {"draws": got["draws"], "t": t}})
        v = twin.openings(t, f_in.readline())
        log(f"window {t}: {len(got['draws'])} openings, "
            f"{'accepted' if 'accepted' in v else 'rejected ' + str(v.get('failure'))}")
        if v != {"accepted": t}:
            raise _Ended(twin.books.why)

    reason = None
    try:
        send(f_out, {"hold": {"registration": registration_json(registration, plan)}})
        got, challenge = twin.start(clock())
        if got != {"released": 0}:
            raise _Ended("the session's first coin was refused")
        previous = registration.digest
        for t in range(W):
            for i in range(passes):
                send(f_out, {"run": {"challenge": C.Released.of(challenge).to_json(), "i": i, "previous": previous.hex(),
                                     "t": t}})
                if twin.root(t, i, f_in.readline(), clock()) != {"ok": True}:
                    raise _Ended(twin.books.why)
                previous = twin.books.s.windows[t].roots[i]
                log(f"window {t} pass {i}: root {previous.hex()[:16]}")
                if i == 0 and t > 0:
                    opened(t - 1)
            sleep_until(clock, twin.books.s.next_boundary_ns())
            got, challenge = twin.boundary(clock())
            if "released" not in got:
                raise _Ended(f"window {t}'s boundary was refused")
        sleep_until(clock, twin.books.s.deadline(W - 1) + 1)
        opened(W - 1)
    except _Ended as e:
        reason = e.reason
    except (OSError, ValueError, KeyError, TypeError, audit.SessionError) as e:
        reason = f"the prover's side failed: {type(e).__name__}: {e}"
    if twin.books.s.outcome is None:
        twin.fail()
    commitment, openings = twin.books.s.disclose()
    return _last(f_out, twin.answer(reason) | {"host": host},
                 {"commitment": commitment.to_json(), "openings": [o.to_json() for o in openings]})


def _last(f_out: BinaryIO, out: dict[str, Any], end: dict[str, Any] | None = None) -> dict[str, Any]:
    """The session's coins disclosed (`end`, when it has any) and the answer, sent if the prover is still there to read
    them, and the answer returned."""
    try:
        if end is not None:
            send(f_out, {"end": end})
        send(f_out, {"answer": out})
    except OSError:
        pass
    return out


class Prover(Protocol):
    """The prover's side as `prove` drives it."""

    def hold(self, registration: Mapping[str, Any]) -> None:
        """The verifier's registration, before any coin: refuse a registration that isn't the store's."""

    def run(self, challenge: C.Released, t: int, i: int, previous: bytes) -> bytes:
        """Pass i of window t: one step's reads under the pass's seed, its receipts committed; the root."""

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[p2.Opening]:
        """Window t's drawn receipts `(pass, position)` opened."""

    def end(self, commitment: C.LiveCommitment, openings: Sequence[C.LiveOpening]) -> None:
        """The session's coin commitment and every released coin's opening, after its last root: refuse them unless
        every challenge this prover was sent opens against them (`coins.received_reason`)."""


def prove(f_in: BinaryIO, f_out: BinaryIO, prover: Prover, request: Mapping[str, Any],
          log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """The prover's side of one session over the channel: `request` sent, then each message answered by `prover`
    until the verifier's answer, which it returns."""
    send(f_out, {"request": dict(request)})
    while True:
        (kind, m), = recv(f_in).items()
        if kind == "hold":
            prover.hold(m["registration"])
        elif kind == "run":
            root = prover.run(C.Released.from_json(m["challenge"]), m["t"], m["i"], bytes.fromhex(m["previous"]))
            send(f_out, {"root": root.hex()})
        elif kind == "open":
            send(f_out, {"openings": [opening_json(o) for o in prover.open(m["t"], [tuple(d) for d in m["draws"]])]})
        elif kind == "end":
            prover.end(C.LiveCommitment.from_json(m["commitment"]), [C.LiveOpening.from_json(o) for o in m["openings"]])
        elif kind == "answer":
            return m
        else:
            raise ValueError(f"no message {kind!r}")
        log(f"{kind} answered")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n\n")[0])
    ap.add_argument("--registration", type=Path, required=True, help="the auditor's registration with its plan (registration_json)")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--stdio", action="store_true", help="the channel is this process's stdin and stdout")
    where.add_argument("--listen", help="HOST:PORT to accept provers on (port 0: any, printed as {\"listening\"})")
    ap.add_argument("--sessions", type=int, default=1, help="with --listen: the sessions to serve, one after another")
    record = ap.add_mutually_exclusive_group()
    record.add_argument("--answer", type=Path, help="a file each session's answer is appended to, a JSON line each: "
                                                    "the verifier's own record of them")
    record.add_argument("--device", type=Path, help="the party's directory, as the auditor's gate gives it: --answer "
                                                    "is its answers.jsonl")
    ap.add_argument("--lean-exe", type=Path, help=f"the built {p2_lean.EXE}, the Lean verifier (default: built "
                                                  "in its Lake package, or without lake on the PATH, the one built "
                                                  "there)")
    a = ap.parse_args(argv)
    data = a.registration.read_bytes()
    if a.device is not None:
        a.device.mkdir(parents=True, exist_ok=True)
        a.answer = a.device / "answers.jsonl"

    def log(m: str) -> None:
        print(f"p2_party {time.strftime('%H:%M:%SZ', time.gmtime())}: {m}", file=sys.stderr, flush=True)

    def done(out: dict[str, Any]) -> None:
        if a.answer is not None:
            with a.answer.open("a") as f:
                f.write(json.dumps(out, sort_keys=True) + "\n")
        log(f"answer {out['verdict']} {out.get('failure') or ''} q_bytes={out.get('q_bytes')} "
            f"host={socket.gethostname()}")

    exe = p2_lean.exe(a.lean_exe)
    with Lean(exe) as lean:
        reg, plan = hold(lean, data)
    log(f"registration {reg.digest.hex()[:16]} ({reg.blocks} units, {plan.length} a pass) on "
        f"{socket.gethostname()}, of record in {p2_lean.EXE}")

    def served(r: BinaryIO, w: BinaryIO) -> dict[str, Any]:
        """One session on a Lean process of its own, so that one the prover crashes ends only its session."""
        with Lean(exe) as lean:
            hold(lean, data)
            return session(r, w, reg, coins=secrets.token_bytes(64), lean=lean, plan=plan, log=log)

    if a.stdio:
        done(served(sys.stdin.buffer, sys.stdout.buffer))
        return 0
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        for _ in range(a.sessions):
            conn, _ = srv.accept()
            with conn, conn.makefile("rb") as r, conn.makefile("wb") as w:
                out = served(r, w)
            done(out)
            print(json.dumps({"answer": out}, sort_keys=True), flush=True)
    return 0


__all__ = ["Books", "PROTOCOL", "Prover", "Twin", "answer", "held_json", "hold", "main", "message", "opening_json",
           "opening_of", "params_of", "plan_of", "prove", "registration_json", "registration_of", "request_of",
           "session", "verdict_json"]

if __name__ == "__main__":
    sys.exit(main())
