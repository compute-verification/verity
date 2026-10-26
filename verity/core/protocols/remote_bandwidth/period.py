"""One remote bandwidth session a period on one node, at a small size: the registration, the verifier's session against the
prover's passes over its store, and the answer of the bandwidth component of onsite Verity (`verity/remote-bandwidth/v0`,
`RemoteAudit.Protocol` in `lean/`).

`run(node, store, coins)` plays every party in one process, in the protocol's order:

1. the prover packs `store` into blocks (`payloads`), commits to them under fresh salts (`audit.payload_tree`), and the
   verifier draws the codec seed after that root from its live `coins` (`audit.register`); the prover encodes its store
   (`audit.Store`);
2. the verifier opens a session (`audit.Session`) on coins its challenger commits to before the first one, drawn from
   `coins` (`coins.local`), and the prover runs `passes` passes a window over the whole store (`audit.Prover`), each root
   sent before the window's boundary; the verifier waits each window out on its clock (`window_ns`) and each last deadline
   (`deadline_ns`) before it closes the window;
3. the verifier draws each window's openings, the prover opens them, and the verifier verifies them.

Every call to the session is an event of the verifier's log with the clock reading it took, and the log, its coins and
the registration are the core case (`case`) that `RemoteAudit.Protocol` replays. The answer (`answer`) is the session's
own: accepted, its failure, or undecided while a window is unverified; on acceptance `certified`'s reads `q_bytes`, the
receipts less the wrong ones the openings leave (`pooled`'s W, rounded up), a `BLOCK_BYTES` block each, at
`eps = (2^24 + S)/2^64` for S the draws, as the Lean's `certify` computes them.

`leg(period_dir, node, out)` is `benchmarks/all_protocols`' leg for bandwidth: the store is the period's records, the
coins fresh from the OS. The prover and the verifier run on the one node, so the run exercises the code and the reads of
a store in one process's memory, not a remote node's.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.ml.commitments.identity import canonical_json_bytes

from . import audit, codec, pooled, receipts
from . import coins as C

PROTOCOL = "verity/remote-bandwidth/v0"
DELTA = Fraction(1, 2 ** 40)
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


def certified(record: audit.Record) -> tuple[dict[str, Any], dict[str, Any]]:
    """`RemoteAudit.Protocol.certify`: an accepted session's public outputs `{"eps", "q_bytes"}` and its profile."""
    windows, _ = audit.pooled_windows(record)
    us, ss = sum(u for u, _ in windows), sum(s for _, s in windows)
    base = {"delta": str(DELTA), "receipts": us, "windows": [[u, s] for u, s in windows]}
    try:
        b = pooled.bound(windows, DELTA)
    except pooled.Refused as e:
        return {"eps": "0", "q_bytes": 0}, base | {"gaps": [str(e)], "wrong": None}
    wrong = -(-b.wrong.numerator // b.wrong.denominator)
    return ({"eps": str(Fraction(2 ** 24 + ss, 2 ** 64)), "q_bytes": (us - wrong) * codec.BLOCK_BYTES},
            base | {"gaps": [], "wrong": str(b.wrong)})


def answer(session: audit.Session) -> dict[str, Any]:
    """The session's answer in the core interface's canonical form, as `RemoteAudit.Protocol.check` gives it. Like the
    check (`RemoteAudit.Protocol.sessionOf`), it refuses the sessions no property of the Lean covers: one drawn more than
    one window an openings coin (`groups_per_coin` above 1) and one paced at one window in flight."""
    out = {"commitment": session.registration.digest.hex(), "profile": None, "protocol": PROTOCOL, "public": None,
           "reason": None}
    if session.params.groups_per_coin != 1:
        return out | {"reason": "groups_per_coin is 1: no property covers an openings coin over more than one window",
                      "verdict": "reject"}
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


def run(node: str, store: bytes, coins: bytes, *, windows: int = 2, passes: int = 1, samples: int = audit.SAMPLES,
        window_ns: int = 100_000_000, deadline_ns: int = 500_000_000, codec_: codec.Codec = codec.DEFAULT,
        cheat: Callable[[audit.Store], audit.Store] | None = None, verify: int | None = None,
        clock: Callable[[], int] = time.monotonic_ns, sleep_until: Callable[[Callable[[], int], int], None] = _sleep_until,
        entropy: Callable[[int], bytes] = os.urandom) -> Period:
    """The period's session on `node` (module docstring): `store` is what the node keeps, `coins` the verifier's live
    secret for the period. `cheat` turns the honest prover's store into the one it reads, and `verify` stops the log
    after that many windows are verified (None: all), for a period that ends undecided. `clock` is the verifier's in
    ns, and `sleep_until(clock, t)` waits for it to reach t."""
    name = node.encode()
    blocks = payloads(store)
    salts = [entropy(receipts.SALT_BYTES) for _ in blocks]
    reg = audit.register(coins, name, len(blocks), audit.payload_tree(name, blocks, salts).commitment.root, codec_)
    honest = audit.Store(reg, blocks, salts)
    params = audit.Params(reg.blocks, deadline_ns, windows, samples=samples, min_passes=passes, window_ns=window_ns,
                          codec=reg.codec, receipt=reg.receipt, rows=reg.rows)
    held, not_before = _Held(), clock()
    challenger = C.local(reg.digest, 0, windows, _Stream(coins, name), clock=lambda: held.now / 1e9)
    s = audit.Session(params, reg, None, challenger, not_before_ns=not_before, clock=held)
    prover = audit.Prover(cheat(honest) if cheat else honest, None, s.commitment, entropy=entropy)
    events: list[dict[str, Any]] = []

    def call(event: dict[str, Any], fn: Callable[[], Any], at: int | None = None) -> Any:
        if at is not None:
            sleep_until(clock, at)
        held.now = clock()
        events.append(event | ({} if event["kind"] == "verify" else {"now": held.now}))
        return fn()

    challenge = call({"kind": "start"}, s.start)
    previous, step, made = reg.digest, -1, {}
    for t in range(windows):
        for i in range(passes):
            ps = prover.run(challenge, i, previous)
            made[t, i], previous, step = ps, ps.root, step + 1
            call({"kind": "receive", "t": t, "i": i, "root": ps.root.hex(), "runs": None, "change": None, "step": step},
                 lambda: s.receive(t, i, ps.root, None, change=None, step=step))
        challenge = call({"kind": "boundary"}, s.boundary, s.next_boundary_ns())
    for t in range(windows if verify is None else verify):
        d = s.deadline(t)
        draws = call({"kind": "close", "t": t}, lambda: s.close(t), None if s.windows[t].final else d + 1)
        if isinstance(draws, audit.Verdict):
            break
        opened = [prover.open(made[t, i], q) for i, q in s.windows[t].draws]
        v = call({"kind": "verify", "t": t, "openings": [opening_json(o) for o in opened]}, lambda: s.verify(t, opened))
        if not v.accepted:
            break
    committed = []
    for label, n in ((C.CHALLENGE, params.windows), (C.OPENINGS, C.openings_coins(params.windows))):
        got = {o.index: o for o in challenger.opened(label)}
        committed += [C.coin(got[j] if j in got else challenger.challenger.open(label, j, b"")) for j in range(n)]
    p = {"deadline_ns": params.deadline_ns, "deferred": params.deferred, "lifetime": params.lifetime,
         "min_passes": params.min_passes, "pace": None, "samples": params.samples, "window_ns": params.window_ns,
         "windows": params.windows}
    evidence = {"append_coins": [], "coins": [c.hex() for c in committed], "events": events}
    case = {"commitment": canonical_json_bytes(registration_json(reg)).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": {"not_before_ns": not_before, "params": p}}
    return Period(answer(s), case)


def serve(exe: str | os.PathLike, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`EXE serve` on `cases`, one canonical JSON line each: the Lean's answers (`Core.Protocol` CLI)."""
    lines = "".join(canonical_json_bytes(c).decode() + "\n" for c in cases)
    p = subprocess.run([str(exe), "serve"], input=lines, capture_output=True, text=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]


def records(period_dir: str | os.PathLike, skip: tuple[str, ...] = ("answers",)) -> bytes:
    """The period's records as one byte string (`pous.period.records`'s framing)."""
    root = Path(period_dir)
    out = bytearray()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.relative_to(root).parts[0] not in skip):
        name, data = f.relative_to(root).as_posix().encode(), f.read_bytes()
        out += len(name).to_bytes(4, "big") + name + len(data).to_bytes(8, "big") + data
    return bytes(out)


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """Bandwidth's leg of `benchmarks/all_protocols` (`leg(period_dir, node, out) -> answer`): `run` on the period's
    records with fresh coins. With `remote-audit-protocol` among the period's tools the Lean answers (the verifier of
    record) and must agree with the Python; it writes `out/bandwidth.case.json` (onsite's bandwidth case) and
    `out/bandwidth.json`."""
    tools = json.loads((Path(period_dir) / "period.json").read_text()).get("tools") or {}
    store = records(period_dir)
    t = time.monotonic()
    period = run(node, store, secrets.token_bytes(64))
    seconds = round(time.monotonic() - t, 2)
    answer_, lean = period.answer, None
    if "remote-audit-protocol" in tools:
        lean = serve(tools["remote-audit-protocol"], [period.case])[0]
        if {k: v for k, v in lean.items() if k != "reason"} != {k: v for k, v in answer_.items() if k != "reason"}:
            raise RuntimeError(f"remote-audit-protocol answered {lean}, the Python {answer_}")
        answer_ = lean
    out.mkdir(parents=True, exist_ok=True)
    (out / "bandwidth.case.json").write_text(json.dumps(period.case, sort_keys=True))
    (out / "bandwidth.json").write_text(json.dumps({"answer": answer_, "eps": str(period.eps), "lean": lean,
                                                    "q_bytes": period.q_bytes, "seconds": seconds,
                                                    "store_bytes": len(store)}, sort_keys=True, indent=1))
    return answer_


__all__ = ["DELTA", "PROTOCOL", "Period", "answer", "certified", "leg", "payloads", "records", "run", "serve"]
