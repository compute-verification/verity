"""The verifier and its challenger of a remote bandwidth session over PoUS's P2 store (`p2.py`) as a party of their own,
so they run on a host the prover doesn't control (`trusted-host`) while the prover serves on its node.

The channel is a pair of byte streams carrying one JSON object of one key a line: the ssh session that starts this side
(`--stdio`, as a gate's verb starts it) or a TCP connection (`--listen`). The session has no round-trip floor, so ssh's
latency changes nothing in its verdict. The verifier holds its own registration (`--registration FILE`, the auditor's
record of PoUS's setup, `registration_json`), never the prover's word for it, and flips its coins from its own host's OS.
The record's `plan` is the units every pass reads (`plan_of`): the units holding a tensor's block, so not the store's
padding past the last tensor, which is PoUS's and which the decode never reads; without one, every unit.

- The prover opens with the request `{"request": {"deadline_ns", "node", "passes", "samples", "window_ns",
  "windows"}}`, whose node must be the registration's.
- The verifier answers `{"hold": {"commitment", "registration"}}`: the session's coin commitment and the registration
  with its plan, which the prover checks against its store and reads each pass in.
- It then drives the session on its clock (`session`): `{"run": {"challenge", "i", "previous", "t"}}` asks for pass i
  of window t under the window's challenge after the root `previous`, answered `{"root": hex}` when the step that read
  it is done. Once a window's roots are final (the next window's first root, or its deadline for the last) it sends
  `{"open": {"draws", "t"}}`, answered `{"openings": [opening_json, ...]}`, and verifies them while the prover serves
  on.
- `{"answer": {...}}` ends the session (`answer`): the verdict, the failure and where, and on acceptance the credited
  reads `q_bytes`, the receipts less the wrong ones the pooled bound leaves (`pooled.bound` at δ = 2^-40), a unit's
  8,224 bytes each, over the session's span.

`python -m verity.core.protocols.remote_bandwidth.p2_party --registration FILE (--stdio | --listen HOST:PORT
[--sessions N]) [--answer FILE | --device DIR]` is the verifier's side (`main`), each session with fresh coins.
Listening, it prints `{"listening": port}` and then each session's `{"answer"}` on stdout, the shape of a party the
auditor's gate starts over ssh with its `--device` directory (`benchmarks/live_verifier/host.py`'s party verbs).
`prove(reader, writer, prover)` is the prover's: it answers each message with `prover`'s `hold`, `run` and `open` until
the answer, which it returns.

No Lean replays a P2 session yet (`RemoteAudit.P2` checks one opening); the verdict here is the Python session's.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from verity.core.protocols.pous.schemes.p2 import P2Params

from . import audit, p2, pooled
from . import coins as C
from . import period as P
from .receipts import Plan

MODULE = "verity.core.protocols.remote_bandwidth.p2_party"
PROTOCOL = "verity/remote-bandwidth/p2/v0"
REQUEST_KEYS = ("deadline_ns", "node", "passes", "samples", "window_ns", "windows")


def send(f: BinaryIO, doc: Mapping[str, Any]) -> None:
    f.write(json.dumps(dict(doc), sort_keys=True).encode() + b"\n")
    f.flush()


def recv(f: BinaryIO, key: str | None = None) -> Any:
    """The next message's value (its one key `key`, if given)."""
    line = f.readline()
    if not line.endswith(b"\n"):
        raise ConnectionError("the peer closed mid-session")
    doc = json.loads(line)
    if not isinstance(doc, dict) or len(doc) != 1:
        raise ValueError("a message is a JSON object of one key")
    if key is None:
        return doc
    if key not in doc:
        raise ValueError(f"expected {key!r}, got {next(iter(doc))!r}")
    return doc[key]


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
    if not isinstance(runs, list) or not all(isinstance(r, list) for r in runs):
        raise ValueError("a registration's plan is a list of runs [first, count]")
    return Plan(tuple(tuple(r) for r in runs), reg.blocks)


def registration_of(doc: Mapping[str, Any]) -> p2.Registration:
    """`registration_json`'s inverse, refused if its `digest` (when given) isn't the fields'."""
    params = P2Params(doc["w"], doc["c0"], doc["blocks_per_segment"], doc["segments"], doc["expander"])
    reg = p2.Registration(bytes.fromhex(doc["node"]), p2.codec_of(params), bytes.fromhex(doc["commitment"]),
                          bytes.fromhex(doc["salt"]), bytes.fromhex(doc["root"]), doc["rows"])
    if "digest" in doc and bytes.fromhex(doc["digest"]) != reg.digest:
        raise ValueError("the registration's digest is not its fields'")
    return reg


def opening_json(o: p2.Opening) -> dict[str, Any]:
    return {"blocks": [b.hex() for b in o.blocks], "leaf": o.leaf.hex(), "leaf_salt": o.leaf_salt.hex(),
            "pass": o.pass_index, "paths": [[x.hex() for x in path] for path in o.paths], "position": o.position,
            "receipt_path": [x.hex() for x in o.receipt_path]}


def opening_of(doc: Any) -> p2.Opening | None:
    """`opening_json`'s inverse, None for anything else (the session rejects it `malformed`)."""
    try:
        return p2.Opening(doc["pass"], doc["position"], tuple(bytes.fromhex(b) for b in doc["blocks"]),
                          tuple(tuple(bytes.fromhex(x) for x in path) for path in doc["paths"]),
                          bytes.fromhex(doc["leaf"]), bytes.fromhex(doc["leaf_salt"]),
                          tuple(bytes.fromhex(x) for x in doc["receipt_path"]))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def request_of(node: bytes, *, windows: int, passes: int, samples: int, window_ns: int,
               deadline_ns: int) -> dict[str, Any]:
    return {"deadline_ns": deadline_ns, "node": node.hex(), "passes": passes, "samples": samples,
            "window_ns": window_ns, "windows": windows}


def params_of(reg: p2.Registration, request: Mapping[str, Any]) -> audit.Params:
    """The session's parameters from the prover's request, refused unless its node is the registration's."""
    if sorted(request) != sorted(REQUEST_KEYS):
        raise ValueError(f"a request is {list(REQUEST_KEYS)}")
    if bytes.fromhex(request["node"]) != reg.node:
        raise ValueError("the request's node is not the registration's")
    return audit.Params(blocks=reg.blocks, deadline_ns=request["deadline_ns"], windows=request["windows"],
                        samples=request["samples"], min_passes=request["passes"], window_ns=request["window_ns"],
                        codec=reg.codec, receipt=reg.receipt, layout=reg.layout, rows=reg.rows)


def answer(s: p2.Session, reason: str | None = None, plan: Plan | None = None) -> dict[str, Any]:
    """The session's answer: `accept` with the credited reads, `reject` with the failure and the sample it failed at, or
    `undecided`; `units` are the store's and `plan_units` those each pass reads."""
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
                  "credited_bytes_per_s": math.floor(q_bytes * 10**9 / span) if span > 0 else None}


def session(f_in: BinaryIO, f_out: BinaryIO, registration: p2.Registration, *, coins: bytes,
            plan: Plan | None = None, clock: Callable[[], int] = time.monotonic_ns,
            sleep_until: Callable[[Callable[[], int], int], None] = P._sleep_until,
            log: Callable[[str], None] = lambda _m: None, host: str | None = None) -> dict[str, Any]:
    """The verifier's side of one session over the channel (`f_in`, `f_out`): the prover's request, the coin commitment,
    each pass asked for in order under `plan` (every unit by default), each window's openings sampled once its roots are
    final and verified, and the answer, sent last and returned, naming the verifier's `host` (this one's name by
    default). A message out of form or a closed channel rejects the session (its answer says why)."""
    host = host or socket.gethostname()
    plan = Plan.whole(registration.blocks) if plan is None else plan
    try:
        params = params_of(registration, recv(f_in, "request"))
    except (ConnectionError, ValueError, KeyError, TypeError) as e:
        out = {"protocol": PROTOCOL, "registration": registration.digest.hex(), "verdict": "refused",
               "reason": f"{type(e).__name__}: {e}", "host": host}
        return _last(f_out, out)
    W, passes = params.windows, params.min_passes
    s = p2.Session(params, registration, C.local(registration.digest, 0, W, P._Stream(coins, registration.node)),
                   not_before_ns=clock(), clock=clock)
    send(f_out, {"hold": {"commitment": s.commitment.to_json(), "registration": registration_json(registration, plan)}})

    def opened(t: int) -> audit.Verdict:
        draws = s.close(t)
        if isinstance(draws, audit.Verdict):
            return draws
        mine = s.windows[t].draws
        send(f_out, {"open": {"draws": [list(d) for d in mine], "t": t}})
        got = recv(f_in, "openings")
        openings = [opening_of(x) for x in got] if isinstance(got, list) else []
        if any(o is None for o in openings):
            openings = []
        v = s.verify(t, openings)
        log(f"window {t}: {len(mine)} openings, {'accepted' if v.accepted else 'rejected ' + str(v.failure)}")
        return v

    reason = None
    try:
        challenge, previous, step = s.start(), registration.digest, 0
        for t in range(W):
            for i in range(passes):
                send(f_out, {"run": {"challenge": challenge.to_json(), "i": i, "previous": previous.hex(), "t": t}})
                got = recv(f_in, "root")
                root = bytes.fromhex(got) if isinstance(got, str) else b""
                v = s.receive(t, i, root, plan.runs, step=step)
                log(f"window {t} pass {i}: root {root.hex()[:16]}")
                if v is not None:
                    raise _Ended
                previous, step = root, step + 1
                if i == 0 and t > 0 and not opened(t - 1).accepted:
                    raise _Ended
            due = s.next_boundary_ns()
            sleep_until(clock, due)
            challenge = s.boundary()
        sleep_until(clock, s.deadline(W - 1) + 1)
        opened(W - 1)
    except _Ended:
        pass
    except (ConnectionError, ValueError, KeyError, TypeError) as e:
        reason = f"the prover's side failed: {type(e).__name__}: {e}"
        if s.outcome is None:
            s._fail(s.filling, "malformed")
    return _last(f_out, answer(s, reason, plan) | {"host": host})


def _last(f_out: BinaryIO, out: dict[str, Any]) -> dict[str, Any]:
    """The answer sent, if the prover is still there to read it, and returned."""
    try:
        send(f_out, {"answer": out})
    except OSError:
        pass
    return out


class _Ended(Exception):
    """The session rejected a root or a window."""


class Prover(Protocol):
    """The prover's side as `prove` drives it."""

    def hold(self, registration: Mapping[str, Any], commitment: C.LiveCommitment) -> None:
        """The verifier's registration and the session's coin commitment, before any coin: refuse a registration that
        isn't the store's."""

    def run(self, challenge: C.LiveOpening, t: int, i: int, previous: bytes) -> bytes:
        """Pass i of window t: one step's reads under the pass's seed, its receipts committed; the root."""

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[p2.Opening]:
        """Window t's drawn receipts `(pass, position)` opened."""


def prove(f_in: BinaryIO, f_out: BinaryIO, prover: Prover, request: Mapping[str, Any],
          log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """The prover's side of one session over the channel: `request` sent, then each message answered by `prover`
    until the verifier's answer, which it returns."""
    send(f_out, {"request": dict(request)})
    while True:
        (kind, m), = recv(f_in).items()
        if kind == "hold":
            prover.hold(m["registration"], C.LiveCommitment.from_json(m["commitment"]))
        elif kind == "run":
            root = prover.run(C.LiveOpening.from_json(m["challenge"]), m["t"], m["i"], bytes.fromhex(m["previous"]))
            send(f_out, {"root": root.hex()})
        elif kind == "open":
            send(f_out, {"openings": [opening_json(o) for o in prover.open(m["t"], [tuple(d) for d in m["draws"]])]})
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
    a = ap.parse_args(argv)
    doc = json.loads(a.registration.read_text())
    reg = registration_of(doc)
    plan = plan_of(doc, reg)
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

    log(f"registration {reg.digest.hex()[:16]} ({reg.blocks} units, {plan.length} a pass) on {socket.gethostname()}")
    if a.stdio:
        done(session(sys.stdin.buffer, sys.stdout.buffer, reg, coins=secrets.token_bytes(64), plan=plan, log=log))
        return 0
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        for _ in range(a.sessions):
            conn, _ = srv.accept()
            with conn, conn.makefile("rb") as r, conn.makefile("wb") as w:
                out = session(r, w, reg, coins=secrets.token_bytes(64), plan=plan, log=log)
            done(out)
            print(json.dumps({"answer": out}, sort_keys=True), flush=True)
    return 0


__all__ = ["PROTOCOL", "Prover", "answer", "main", "opening_json", "opening_of", "params_of", "plan_of", "prove",
           "registration_json", "registration_of", "request_of", "session"]

if __name__ == "__main__":
    sys.exit(main())
