"""One remote bandwidth session a period on one node, at a small size: the registration, the verifier's session against the
prover's passes over its store, and the answer of the bandwidth component of onsite Verity (`verity/remote-bandwidth/v0`,
`RemoteAudit.Protocol` in `lean/`).

`run(node, store, coins)` plays every party in one process (the verifier's side, `session`, against the prover in its
process, `Local`), in the protocol's order:

1. the prover packs `store` into blocks (`payloads`), commits to them under fresh salts (`audit.payload_tree`), and the
   verifier draws the codec seed after that root from its live `coins` (`audit.register`); the prover encodes its store
   (`audit.Store`);
2. the verifier opens a session (`audit.Session`) on coins its challenger commits to before the first one, drawn from
   `coins` (`coins.local`), and the prover runs `passes` passes a window over the whole store (`audit.Prover`) under
   each window's challenge, sent as its value and the root it was opened after (`coins.Released`), each root sent
   before the window's boundary; the verifier waits each window out on its clock (`window_ns`) and each last deadline
   (`deadline_ns`) before it closes the window;
3. the verifier draws each window's openings, the prover opens them, and the verifier verifies them;
4. once the session has its outcome, the verifier sends the prover the coin commitment and every released coin's
   opening (`audit.Session.disclose`), and the prover checks each challenge it was sent against them (`Prover.end`).

Every call to the session is an event of the verifier's log with the clock reading it took, and the log, its coins and
the registration are the core case (`case`) that `RemoteAudit.Protocol` replays. The answer (`answer`) is the session's
own: accepted, its failure, or undecided while a window is unverified; on acceptance `certified`'s reads `q_bytes`, the
receipts less the wrong ones the openings leave (`pooled`'s W, rounded up), a `BLOCK_BYTES` block each, at
`eps = (2^24 + ⌈W/g⌉·s)/2^64`, the session's openings coins at its `samples` s each, as the Lean's `certify` computes
them: the bound `Properties.RemoteAudit.Component.LineHeldAtComponent` proves (`PublishedEps`), not the draws a session
made, min(s, U) of a draw's U receipts, which can be fewer.

`leg(period_dir, node, out)` is `benchmarks/all_protocols`' leg for bandwidth: the store is the period's records
(`pous.period.records`, every copy of the auditor's ledger in them cut to the entries their window cites). The
verifier and its challenger are a party of their own (`verifier_party`) on `period.json`'s `challenger_host`, with coins
fresh from that host's OS, and the prover runs on the node. `trusted-host`'s premise, that they run on a machine
the prover doesn't control, holds only when the challenger host isn't the node, and the leg says which
(`verifier_party.placement`). The prover reads a store in its process's memory, not a remote node's.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from verity.core.protocols.pous.period import records
from verity.core.protocols.profile import WINDOW_DELTA
from verity.ml.commitments.identity import canonical_json_bytes

from . import audit, codec, pooled, receipts
from . import coins as C

PROTOCOL = "verity/remote-bandwidth/v0"
DELTA = WINDOW_DELTA
"""`audit.DELTA`'s exact value, the pooled bound's δ (`RemoteAudit.Protocol.DELTA`)."""
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


def answer(session: audit.Session) -> dict[str, Any]:
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


@dataclass(frozen=True)
class Period:
    """A period's bandwidth: the session's answer and its core case."""

    answer: dict[str, Any]
    case: dict[str, Any]

    @property
    def verdict(self) -> str:
        return self.answer["verdict"]

    @property
    def q_bytes(self) -> int:
        """The reads certified, in bytes: 0 unless accepted under a pooled bound."""
        return self.answer["public"]["q_bytes"] if self.verdict == "accept" else 0

    @property
    def eps(self) -> Fraction:
        return Fraction(self.answer["public"]["eps"]) if self.verdict == "accept" else Fraction(0)


class Prover(Protocol):
    """The prover as the verifier's side of a session sees it: in its process (`Local`) or across a socket
    (`verifier_party`)."""

    def register(self) -> tuple[int, bytes]:
        """The store's block count and its payload root, for the verifier to register."""

    def hold(self, registration: audit.Registration) -> None:
        """The registration, before any challenge."""

    def run(self, challenge: C.Released, t: int, i: int, previous: bytes) -> bytes:
        """Pass i of window t under `challenge`, after the root `previous`: its root."""

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[audit.Opening]:
        """Window t's drawn receipts `(pass, position)` opened."""

    def end(self, commitment: C.LiveCommitment, openings: Sequence[C.LiveOpening]) -> None:
        """The session's coin commitment and every released coin's opening, once it has its outcome: refused unless
        every challenge the prover was sent opens against them (`audit.Prover.end`)."""


class Local:
    """The prover in the verifier's process: `store` packed into blocks (`payloads`), committed under salts from
    `entropy`, and read through `cheat` (the honest store when None)."""

    def __init__(self, node: str, store: bytes, *, cheat: Callable[[audit.Store], audit.Store] | None = None,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.name, self.blocks, self.cheat, self.entropy = node.encode(), payloads(store), cheat, entropy
        self.salts = [entropy(receipts.SALT_BYTES) for _ in self.blocks]
        self.made: dict[tuple[int, int], audit.Pass] = {}

    def register(self) -> tuple[int, bytes]:
        return len(self.blocks), audit.payload_tree(self.name, self.blocks, self.salts).commitment.root

    def hold(self, registration: audit.Registration) -> None:
        honest = audit.Store(registration, self.blocks, self.salts)
        self.prover = audit.Prover(self.cheat(honest) if self.cheat else honest, None, entropy=self.entropy)

    def run(self, challenge: C.Released, t: int, i: int, previous: bytes) -> bytes:
        ps = self.prover.run(challenge, i, previous)
        self.made[t, i] = ps
        return ps.root

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[audit.Opening]:
        return [self.prover.open(self.made[t, i], q) for i, q in draws]

    def end(self, commitment: C.LiveCommitment, openings: Sequence[C.LiveOpening]) -> None:
        self.prover.end(commitment, openings)


def run(node: str, store: bytes, coins: bytes, *, windows: int = 2, passes: int = 1, samples: int = audit.SAMPLES,
        window_ns: int = 100_000_000, deadline_ns: int = 500_000_000, codec_: codec.Codec = codec.DEFAULT,
        cheat: Callable[[audit.Store], audit.Store] | None = None, verify: int | None = None,
        clock: Callable[[], int] = time.monotonic_ns, sleep_until: Callable[[Callable[[], int], int], None] = _sleep_until,
        entropy: Callable[[int], bytes] = os.urandom) -> Period:
    """The period's session on `node` (module docstring): `store` is what the node keeps, `coins` the verifier's live
    secret for the period. `cheat` turns the honest prover's store into the one it reads, and `verify` stops the log
    after that many windows are verified (None: all), for a period that ends undecided. `clock` is the verifier's in
    ns, and `sleep_until(clock, t)` waits for it to reach t."""
    return session(node, Local(node, store, cheat=cheat, entropy=entropy), coins, windows=windows, passes=passes,
                   samples=samples, window_ns=window_ns, deadline_ns=deadline_ns, codec_=codec_, verify=verify,
                   clock=clock, sleep_until=sleep_until)


def opened(reg: audit.Registration, coins: bytes, clock: Callable[[], int], *, not_before: int, windows: int, passes: int,
           samples: int, window_ns: int, deadline_ns: int) -> audit.Session:
    """The verifier's session over the store `reg` registers, on `clock`, its challenger's coins drawn from `coins`."""
    params = audit.Params(reg.blocks, deadline_ns, windows, samples=samples, min_passes=passes, window_ns=window_ns,
                          codec=reg.codec, receipt=reg.receipt, rows=reg.rows)
    challenger = C.local(reg.digest, 0, windows, _Stream(coins, reg.node), clock=lambda: clock() / 1e9)
    return audit.Session(params, reg, None, challenger, not_before_ns=not_before, clock=clock)


def case_of(s: audit.Session, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
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


def session(node: str, prover: Prover, coins: bytes, *, windows: int = 2, passes: int = 1, samples: int = audit.SAMPLES,
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


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """Bandwidth's leg of `benchmarks/all_protocols` (`leg(period_dir, node, out) -> answer`): the session on the
    period's records, the verifier and its challenger a party of their own on `period.json`'s `challenger_host`
    (`verifier_party.side_of`; on this node by default, as a process of its own, with its own coins) and the prover
    here. With `remote-audit-protocol` among the period's tools the Lean verifier answers and must agree
    with the Python; it writes `out/bandwidth.case.json` (onsite's bandwidth case) and `out/bandwidth.json`. Which host
    ran each party, and whether `trusted-host`'s premise holds (`verifier_party.placement`), go in
    `bandwidth.json` and in an accepted answer's profile (`placement`)."""
    from . import verifier_party  # noqa: PLC0415  (it imports this module)

    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools = doc.get("tools") or {}
    party = verifier_party.side_of(doc, node)
    store = records(period_dir)
    t = time.monotonic()
    period, side = verifier_party.period_with(party, verifier_party.request_of(node), store)
    seconds = round(time.monotonic() - t, 2)
    where = verifier_party.placement(node, party.host)
    answer_, lean = period.answer, None
    if "remote-audit-protocol" in tools:
        lean = serve(tools["remote-audit-protocol"], [period.case])[0]
        if {k: v for k, v in lean.items() if k != "reason"} != {k: v for k, v in answer_.items() if k != "reason"}:
            raise RuntimeError(f"remote-audit-protocol answered {lean}, the Python {answer_}")
        answer_ = lean
    if answer_["profile"] is not None:
        answer_ = answer_ | {"profile": answer_["profile"] | {"placement": where}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "bandwidth.case.json").write_text(json.dumps(period.case, sort_keys=True))
    (out / "bandwidth.json").write_text(json.dumps({"answer": answer_, "eps": str(period.eps), "lean": lean,
                                                    "placement": where | {"reported_host": side["host"]},
                                                    "q_bytes": period.q_bytes, "seconds": seconds,
                                                    "store_bytes": len(store)}, sort_keys=True, indent=1))
    return answer_


__all__ = ["DELTA", "PROTOCOL", "Local", "Period", "Prover", "answer", "case_of", "certified", "leg", "opened", "payloads",
           "records", "registration_json", "registration_of", "run", "serve", "session"]
