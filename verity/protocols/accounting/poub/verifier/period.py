"""One PoUB session a period on one node, at a small size: the registration, the verifier's session against the
prover's passes over its store, and the answer of the bandwidth component of onsite Verity (`verity/remote-bandwidth/v0`,
`RemoteAudit.Protocol` in `lean/`).

`session` is the verifier's side, against a prover in its process or across a socket (`Peer`), in the protocol's
order; `verity/protocols/accounting/poub/prover/period_prover.py`'s `run` plays every party in one process:

1. the prover packs `store` into blocks (`payloads`), commits to them under fresh salts (`audit.payload_tree`), and the
   verifier samples the codec seed after that root from its live `coins` (`audit.register`); the prover encodes its store
   (`audit.Store`);
2. the verifier opens a session (`audit.AuditRun`) on coins its challenger commits to before the first one, sampled from
   `coins` (`coins.local`), and the prover runs `passes` passes a window over the whole store
   (`audit_prover.Prover`) under each window's challenge, sent as its value and the root it was opened after
   (`coins.Released`), each root sent before the window's boundary; the verifier waits each window out on its clock
   (`window_ns`) and each last deadline (`deadline_ns`) before it closes the window;
3. the verifier draws each window's openings, the prover opens them, and the verifier verifies them;
4. once the session has its outcome, the verifier sends the prover the coin commitment and every released coin's
   opening (`audit.AuditRun.disclose`), and the prover checks each challenge it was sent against them (`Peer.end`).

Every call to the session is an event of the verifier's log with the clock reading it took, and the log, its coins and
the registration are the core case (`case`) that `RemoteAudit.Protocol` replays. The answer (`answer`) is the session's
own: accepted, its failure, or undecided while a window is unverified; on acceptance `certified`'s reads `q_bytes`, the
receipts less the wrong ones the openings leave (`pooled`'s W, rounded up), a `BLOCK_BYTES` block each, at
`eps = (2^24 + ⌈W/g⌉·s)/2^64`, the session's openings coins at its `samples` s each, as the Lean's `certify` computes
them: the bound `Properties.RemoteAudit.Component.LineHeldAtComponent` proves (`PublishedEps`), not the draws a session
made, min(s, U) of a draw's U receipts, which can be fewer.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, Protocol

from verity.protocols.accounting.pous.verifier.period import Period, Run
from verity.protocols.proofs.spot.verifier.profile import WINDOW_DELTA
from verity.commitments.identity import canonical_json_bytes

from . import audit, codec, pooled
from . import coins as C

PROTOCOL = "verity/remote-bandwidth/v0"
DELTA = WINDOW_DELTA
"""`audit.DELTA`'s exact value, the pooled bound's δ (`RemoteAudit.Protocol.DELTA`).  An accepted session's `eps` is this
δ and its samples' rejected first words, and leaves out Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed): it
reads δ + Σ_t adv_t (`ledger.DELTA_FORM`)."""
WORD_BYTES = 64
DATA_BYTES = WORD_BYTES - 1
"""Each 64-byte little-endian piece of a block carries 63 bytes of the store under a zero top byte, so every word is below
the field's prime at either width."""


def payloads(store: bytes) -> list[bytes]:
    """`store` as blocks of `codec.BLOCK_BYTES`, 63 bytes of it a 64-byte piece and the last block padded with zeros."""
    per = codec.BLOCK_BYTES // WORD_BYTES * DATA_BYTES
    n = max(1, -(-len(store) // per))
    padded = bytes(store) + bytes(n * per - len(store))
    return [b"".join(padded[b * per + u * DATA_BYTES:b * per + (u + 1) * DATA_BYTES] + b"\0"
                     for u in range(codec.BLOCK_BYTES // WORD_BYTES)) for b in range(n)]


def eps(windows: int, groups_per_coin: int, samples: int) -> Fraction:
    """`RemoteAudit.Protocol.eps`: δ + ⌈W/g⌉·s/2^64 for a session of W `windows`, g `groups_per_coin` and s `samples`."""
    return DELTA + Fraction(C.openings_coins(windows, groups_per_coin) * samples, 2 ** 64)


def certified(record: audit.Record) -> tuple[dict[str, Any], dict[str, Any]]:
    """`RemoteAudit.Protocol.certify`: an accepted session's public outputs `{"eps", "q_bytes"}` and its profile."""
    if record.samples is None:
        raise ValueError("the record states no samples, at which eps charges each draw's rejection: certify a "
                         "session's record (`Session.record`)")
    windows, _ = audit.pooled_windows(record)
    us = sum(u for u, _ in windows)
    base = {"delta": str(DELTA), "receipts": us, "windows": [[u, s] for u, s in windows]}
    try:
        b = pooled.bound(windows, DELTA)
    except pooled.Refused as e:
        return {"eps": "0", "q_bytes": 0}, base | {"gaps": [str(e)], "wrong": None}
    wrong = -(-b.wrong.numerator // b.wrong.denominator)
    return ({"eps": str(eps(len(record.windows), record.groups_per_coin, record.samples)),
             "q_bytes": (us - wrong) * codec.BLOCK_BYTES},
            base | {"gaps": [], "wrong": str(b.wrong)})


def answer(session: audit.AuditRun) -> dict[str, Any]:
    """The session's answer in the core interface's canonical form, as `RemoteAudit.Protocol.check` gives it. Like the
    check (`RemoteAudit.Protocol.sessionOf`), it refuses the one form no property of the Lean covers, a session paced at
    one window in flight."""
    out = {"commitment": session.registration.digest.hex(), "profile": None, "protocol": PROTOCOL, "public": None,
           "reason": None}
    if session.params.pace is not None and session.params.pace.in_flight == 1:
        return out | {"reason": "in_flight is 2: no property covers one window in flight", "verdict": "reject"}
    outcome = session.outcome
    if outcome == "accepted":
        public, profile = certified(session.record())
        return out | {"profile": profile, "public": public, "verdict": "accept"}
    if outcome is not None:
        return out | {"reason": f"the session was rejected: {outcome}", "verdict": "reject"}
    return out | {"reason": "the session has a window not yet verified", "verdict": "undecided"}


def registration_json(r: audit.Registration) -> dict[str, Any]:
    return {"bind": r.bind, "blocks": r.blocks, "codec": {"field": r.codec.field.name, "masked": r.codec.masked,
                                                          "rounds": r.codec.rounds},
            "codec_seed": r.codec_seed.hex(), "layout": r.layout, "node": r.node.hex(), "payload_root": r.payload_root.hex(),
            "receipt": r.receipt, "rows": r.rows}


def registration_of(doc: Any) -> audit.Registration:
    """`registration_json`'s inverse."""
    c = doc["codec"]
    return audit.Registration(bytes.fromhex(doc["node"]), doc["blocks"], bytes.fromhex(doc["payload_root"]),
                              bytes.fromhex(doc["codec_seed"]), codec.Codec(codec.FIELDS[c["field"]], c["rounds"], c["masked"]),
                              doc["receipt"], doc["bind"], doc["layout"], doc["rows"])


def opening_json(o: audit.Opening) -> dict[str, Any]:
    return {"block": o.block.hex(), "digest": o.digest.hex(), "leaf": o.leaf.hex(), "leaf_salt": o.leaf_salt.hex(),
            "pass": o.pass_index, "payload_path": [x.hex() for x in o.payload_path], "payload_salt": o.payload_salt.hex(),
            "position": o.position, "receipt_path": [x.hex() for x in o.receipt_path], "tile": o.tile.hex()}


class _Stream:
    """The challenger's entropy: `coins` in SHA-512 counter mode, so a period's coins are its live secret's."""

    def __init__(self, coins: bytes, node: bytes) -> None:
        self.base, self.count = hashlib.sha512(b"verity/remote-bandwidth-period/challenger/v1\0" + node + coins).digest(), 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.base + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


class _Held:
    """The session's clock: the one reading the period took for the event in progress."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def _sleep_until(clock: Callable[[], int], t: int) -> None:
    while (now := clock()) < t:
        time.sleep((t - now) / 1e9)


class Peer(Protocol):
    """The prover as the verifier's side of a session sees it: in its process (`period_prover.Local`) or across a socket
    (`verifier_party`)."""

    def register(self) -> tuple[int, bytes]:
        """The store's block count and its payload root, for the verifier to register."""

    def hold(self, registration: audit.Registration) -> None:
        """The registration, before any challenge."""

    def run(self, challenge: C.Released, t: int, i: int, previous: bytes) -> bytes:
        """Pass i of window t under `challenge`, after the root `previous`: its root."""

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[audit.Opening]:
        """Window t's sampled receipts `(pass, position)` opened."""

    def end(self, commitment: C.LiveCommitment, openings: Sequence[C.LiveOpening]) -> None:
        """The session's coin commitment and every released coin's opening, once it has its outcome: refused unless
        every challenge the prover was sent opens against them (`audit_prover.Prover.end`)."""


def opened(reg: audit.Registration, coins: bytes, clock: Callable[[], int], *, not_before: int, windows: int, passes: int,
           samples: int, window_ns: int, deadline_ns: int) -> audit.AuditRun:
    """The verifier's session over the store `reg` registers, on `clock`, its challenger's coins flipped from `coins`."""
    params = audit.Params(reg.blocks, deadline_ns, windows, samples=samples, min_passes=passes, window_ns=window_ns,
                          codec=reg.codec, receipt=reg.receipt, rows=reg.rows)
    challenger = C.local(reg.digest, 0, windows, _Stream(coins, reg.node), clock=lambda: clock() / 1e9)
    return audit.AuditRun(params, reg, None, challenger, not_before_ns=not_before, clock=clock)


def case_of(s: audit.AuditRun, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The core case `RemoteAudit.Protocol` replays: session `s`'s registration, every coin its challenger committed to
    (each one it didn't release opened now), its log `events` and its parameters."""
    params, challenger = s.params, s.coins
    committed = []
    for label, n in ((C.CHALLENGE, params.windows), (C.OPENINGS, C.openings_coins(params.windows))):
        got = {o.index: o for o in challenger.opened(label)}
        committed += [C.coin(got[j] if j in got else challenger.challenger.open(label, j, b"")) for j in range(n)]
    p = {"deadline_ns": params.deadline_ns, "deferred": params.deferred, "lifetime": params.lifetime,
         "min_passes": params.min_passes, "pace": None, "samples": params.samples, "window_ns": params.window_ns,
         "windows": params.windows}
    evidence = {"append_coins": [], "coins": [c.hex() for c in committed], "events": list(events)}
    return {"commitment": canonical_json_bytes(registration_json(s.registration)).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": {"not_before_ns": s.not_before_ns, "params": p}}


def session(node: str, prover: Peer, coins: bytes, *, windows: int = 2, passes: int = 1, samples: int = audit.SAMPLES,
            window_ns: int = 100_000_000, deadline_ns: int = 500_000_000, codec_: codec.Codec = codec.DEFAULT,
            verify: int | None = None, clock: Callable[[], int] = time.monotonic_ns,
            sleep_until: Callable[[Callable[[], int], int], None] = _sleep_until) -> Period:
    """The verifier's side of the period's session on `node` against `prover` (`run`'s, wherever the prover is): the
    registration it draws from `coins`, its challenger's coins, its clock and its log, which are the case."""
    n, root = prover.register()
    reg = audit.register(coins, node.encode(), n, root, codec_)
    held = _Held()
    s = opened(reg, coins, held, not_before=clock(), windows=windows, passes=passes, samples=samples,
               window_ns=window_ns, deadline_ns=deadline_ns)
    prover.hold(reg)
    events: list[dict[str, Any]] = []

    def call(event: dict[str, Any], fn: Callable[[], Any], at: int | None = None) -> Any:
        if at is not None:
            sleep_until(clock, at)
        held.now = clock()
        events.append(event | ({} if event["kind"] == "verify" else {"now": held.now}))
        return fn()

    challenge = call({"kind": "start"}, s.start)
    previous, step = reg.digest, -1
    for t in range(windows):
        for i in range(passes):
            root = prover.run(C.Released.of(challenge), t, i, previous)
            previous, step = root, step + 1
            call({"kind": "receive", "t": t, "i": i, "root": root.hex(), "runs": None, "change": None, "step": step},
                 lambda: s.receive(t, i, root, None, change=None, step=step))
        challenge = call({"kind": "boundary"}, s.boundary, s.next_boundary_ns())
    for t in range(windows if verify is None else verify):
        d = s.deadline(t)
        draws = call({"kind": "close", "t": t}, lambda: s.close(t), None if s.windows[t].final else d + 1)
        if isinstance(draws, audit.Verdict):
            break
        got = prover.open(t, s.windows[t].draws)
        v = call({"kind": "verify", "t": t, "openings": [opening_json(o) for o in got]}, lambda: s.verify(t, got))
        if not v.accepted:
            break
    if s.outcome is not None:
        prover.end(*s.disclose())
    return Period(answer(s), case_of(s, events))


def serve(exe: str | os.PathLike, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`EXE serve` on `cases`, one canonical JSON line each: the Lean's answers (`Core.Protocol` CLI)."""
    lines = "".join(canonical_json_bytes(c).decode() + "\n" for c in cases)
    p = subprocess.run([str(exe), "serve"], input=lines, capture_output=True, text=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]


__all__ = ["DELTA", "PROTOCOL", "Peer", "Period", "Run", "answer", "case_of", "certified", "opened", "payloads",
           "registration_json", "registration_of", "serve", "session"]
