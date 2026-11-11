"""PoUS's memory challenger as a party of its own, so it runs on a host the developer doesn't control (the challenger host,
`trusted-host`) while the prover stays on its node.

The challenger listens and the prover connects. Each round is one frame each way: the challenger sends `u64be index`
(reading the send time) and the prover answers with the whole block, `block_bytes` of it (the challenger reads the receipt
time); `u64be 2^64 − 1` ends the session, after its last round or a late one. Before round 0 the challenger sends
`WARMUPS` frames on `WARMUP_INDEX`, which no coin chose, and discards their answers: nothing times, records or signs them,
so the first timed round doesn't pay for waking both processes and the prover's memory path after the session's setup.
The challenger flips its coins on its own host, designates the session id from them, the node and the setup's root
(`period.SESSION_DOMAIN`), signs under its own key and claims the id in its own session log. It reports `{"host",
"public_key", "session", "session_id", "warmups"}`.

That id is a request's that names no audit (`period.audit`'s), whose salt is sampled where setup ran. An audit's two coins
are the challenger's, the setup's salt and the session's:

- **The salt.** Before setup encodes `C*`, the prover's side sends `{"salt": {"audit", "commitment"}}`: the audit's
  64-byte binding and the commitment to `W` that setup will publish. The challenger samples the salt's source from its
  host's OS after it (`give_salt`), logs the audit, the commitment and the salt (`setup.draw_salt` of that source) in its
  salt log (`memory_challenger.SaltLog`, which holds one salt an audit), and prints `{"source": HEX}`, which setup takes
  as its salt source (`challenger_session.salt`).
- **The session.** A request naming its audit (`audit`, and `published`, setup's published outputs) runs under
  `session_id(audit, commitment, salt)`, derived from what the challenger logged and never sampled. The challenger
  refuses it (`designate`) unless the published outputs carry exactly the logged commitment and salt and the session
  binds the root they publish; its session log refuses a second session under the id. So an audit has one salt and one
  session, and a retry is a visible failure, never a resample. The verifier computes the id from the audit's binding and
  what setup published, never from the report, so a session under any other id is refused.

A refusal prints `{"refused": REASON}` and exits 1 before the side listens.

The two sides are parties of the deployment layer (`verity.configuration.deploy`), the challenger (`Challenger`) and the prover
(`challenger_session.Prover`). Their messages and codec, the request, the ids and the prover's side are PoUS's
(`challenger_session`), which this device imports and which never imports it. The challenger takes its coin tree's hiding
key and each round's salt from its host's coins too. `serve` is its party, run on the deployment layer's byte-stream
adapter (`verity.configuration.deploy.stream`) on the clock and coins it is given: it reads its clock once before each index it
sends and once after each block it takes, those readings alone, so on the same coins and readings it signs the same
bytes wherever it runs.

`python -m verity.protocols.pous.device.challenger_party --device DIR [--listen HOST:PORT]` is the challenger's
side (`main`): it reads one JSON line on stdin, a salt request or the session's `{"binding", "block_bytes", "blocks",
"k", "limit_ns", "node"}` (with `audit` and `published` when it names its audit, and P2's parameters, which
`challenger_session.setup_of` checks against them), and answers the salt request with its source, or prints
`{"listening": PORT}`, serves one prover (`serve`) and prints its report. Its key, session log (`sessions`) and salt log
(`salts`) are in `DIR`. Each session it completes is kept on its own host, `DIR/setups/<session id>.json` (0700, the
file 0600: `keep`), the request and the id it designated: the auditor's record of that window's setup, from which the
remote bandwidth audit's verifier on the same host builds its registration (`remote_bandwidth.p2_party --pous-record`),
never from a file the prover sends.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from verity.configuration.deploy import stream
from verity.configuration.deploy.party import Finish, Identity, Received, Send
from verity.configuration.deploy.runtime import ProtocolDef
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import period
from verity.protocols.pous.verifier import run as R
from verity.protocols.pous.verifier import setup as pous_setup
from verity.protocols.pous.device.challenger_session import (CHALLENGER, MANIFEST, MODULE, PROVER, SETUPS, WARMUP_INDEX,
                                                           WARMUPS, Frames, Polled, Prover, coin_session_id, session_id,
                                                           setup_of)
from verity.configuration.wire import Message, Phase, Refused

COIN_BYTES = 64
"""The session's coins, which the challenger draws first (`main`'s `secrets.token_bytes(64)`), before its coin tree's
hiding key and salts."""
SOURCE_BYTES = 64
"""The setup salt's source, which the challenger draws from its OS for an audit (`give_salt`)."""


class Refusal(RuntimeError):
    """A request the challenger's side refuses; the message is why."""


def give_salt(salts: mc.SaltLog, request: Mapping[str, Any], draw: Callable[[int], bytes] = os.urandom) -> bytes:
    """The challenger's answer to a salt request `{"audit", "commitment"}`: a source sampled from `draw` after the
    commitment, its salt (`setup.draw_salt`) logged under the audit in `salts` before the source is returned. Refused
    (`Refusal`) when the audit has a salt already."""
    audit, commitment = bytes.fromhex(request["audit"]), bytes.fromhex(request["commitment"])
    source = draw(SOURCE_BYTES)
    try:
        salts.record(audit, commitment, pous_setup.draw_salt(source, commitment))
    except (mc.SessionLogError, ValueError) as e:
        raise Refusal(str(e)) from None
    return source


def designate(request: Mapping[str, Any], salts: mc.SaltLog, sessions: mc.SessionLog) -> bytes:
    """The id of the session `request` names its audit for (`session_id`), from the commitment and salt `salts` logged
    under the audit; `Refusal` unless setup's published outputs (`published`) carry exactly those and the session audits
    the `C*` they publish, or when a session has run under the id (`sessions`)."""
    try:
        audit, published = bytes.fromhex(request["audit"]), R.Setup.from_json(request["published"])
    except (KeyError, TypeError, ValueError) as e:
        raise Refusal(f"the request names no audit and setup's published outputs ({type(e).__name__}: {e})") from None
    logged = salts.get(audit)
    if logged is None:
        raise Refusal(f"the memory challenger drew no salt for audit {audit.hex()}: a session runs on a setup salted by it")
    if (published.commitment, published.salt) != logged:
        raise Refusal(f"setup's published outputs do not carry the commitment and salt the memory challenger logged for "
                      f"audit {audit.hex()}")
    if published.root.hex() != request.get("binding") or published.blocks != request.get("blocks"):
        raise Refusal("the session does not audit the C* setup published: its binding and blocks are the published root's "
                      "and count")
    sid = session_id(audit, *logged)
    if sid in sessions:
        raise Refusal(f"a session has run for audit {audit.hex()}: the memory challenger runs one an audit, and a retry "
                      "is refused, never redrawn")
    return sid


class Challenger:
    """The memory challenger: one session of `request_of`'s parameters on its clock and coins (module docstring), under
    its audit's id (`designate`, from `capabilities["salts"]`, its salt log) when the request names one, else the id
    designated from its coins under `domain` (`period.SESSION_DOMAIN`). Its machine gives it `signing-key`
    (`capabilities["signer"]`, a `memory_challenger.Signer`, with `capabilities["log"]`, its session log);
    `resolution_ns` is how finely its clock ticks (`memory_challenger.resolution_of`). It finishes with the session and
    its id (private: what it signs is witness), or refused: with `designate`'s reason before any frame, or by name when
    the prover leaves (`closed`), sends a frame the wire refuses, or one out of phase."""

    def __init__(self, domain: str, warmups: int = WARMUPS, resolution_ns: float = 1) -> None:
        self.domain, self.warmups, self.resolution_ns = domain, warmups, resolution_ns
        self.c: mc.MemoryChallenger | None = None
        self.sid: bytes | None = None
        self._drawn: list[bytes] = []
        self.session: R.ChallengePhaseRecord | None = None
        self._reading: int | None = None
        self._block: bytes | None = None
        self._warm = 0

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.request, caps = dict(params), identity.capabilities
        self.signer, self.log = caps["signer"], caps["log"]
        if "audit" in self.request:
            try:
                if caps.get("salts") is None:
                    raise Refusal("the memory challenger keeps no salt log, so it runs no audit's session")
                self.sid = designate(self.request, caps["salts"], self.log)
            except (Refusal, mc.SessionLogError) as e:
                return [Finish(None, str(e), kind="private")]
        self.phase = Phase("Coins")
        return [Send("coins", Message("CoinsDraw", {"n": COIN_BYTES}))]

    def _clock(self) -> int:
        """The reading the challenger asked its clock for, used once: `memory_challenger.MemoryChallenger` reads it here."""
        if self._reading is None:
            raise RuntimeError("the memory challenger read its clock without a fresh reading")
        out, self._reading = self._reading, None
        return out

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Refused", "Closed"):
            return [Finish(None, m.fields["name"] if m.type == "Refused" else "closed", kind="private")]
        if m.type == "ClockReading":
            self._reading = m.fields["ns"]
            return self._challenge() if self._block is None else self._receive()
        try:
            self.phase.check(m)
        except Refused as e:
            return [Finish(None, e.name, kind="private")]
        self.phase.expect()
        if m.type == "Coins":
            return self._coins(m.body)
        if self._warm <= self.warmups:
            return self._warmup()
        self._block = m.body
        return [_read_clock()]

    def _coins(self, body: bytes) -> list[Any]:
        """The session's coins, then the coin tree's hiding key, then each round's salt, each sampled in turn; once all
        are in, the session's challenger."""
        q, self._drawn = self.request, [*self._drawn, body]
        if len(self._drawn) < q["k"] + 2:
            self.phase.expect("Coins")
            n = mc.HM96.key_bytes if len(self._drawn) == 1 else mc.HM96.salt_bytes
            return [Send("coins", Message("CoinsDraw", {"n": n}))]
        coins, key, *salts = self._drawn
        binding = bytes.fromhex(q["binding"])
        if self.sid is None:
            self.sid = coin_session_id(coins, self.domain, q["node"], binding)
        tree = mc.CoinTree.draw(coins, self.sid, q["blocks"], q["k"], key=mc.coin_key(key), salts=salts)
        self.c = mc.MemoryChallenger(tree, self.signer, log=self.log, limit_ns=q["limit_ns"],
                                     block_bytes=q["block_bytes"], binding=binding, clock=self._clock,
                                     resolution_ns=self.resolution_ns)
        return self._warmup()

    def _warmup(self) -> list[Any]:
        """The next warm-up, or, after the last one's answer, the reading round 0 is sent at."""
        self._warm += 1
        if self._warm > self.warmups:
            return [_read_clock()]
        self.phase.expect("Block")
        return [Send(PROVER, Message("Warmup", {"index": WARMUP_INDEX}))]

    def _challenge(self) -> list[Any]:
        i = self.c.challenge()
        self.phase.expect("Block")
        return [Send(PROVER, Message("Index", {"index": i}))]

    def _receive(self) -> list[Any]:
        block, self._block = self._block, None
        self.c.receive(block)
        if not self.c.done:
            return [_read_clock()]
        self.session = R.ChallengePhaseRecord(self.c.close(), self.c.commitment, self.c.openings())
        return [Send(PROVER, Message("End")),
                Finish({"session": self.session.to_json(), "session_id": self.sid.hex()}, kind="private")]


def _challenger() -> Challenger:
    return Challenger(period.SESSION_DOMAIN)


PROTOCOL = ProtocolDef(MANIFEST, {CHALLENGER: _challenger, PROVER: Prover})


def serve(f: BinaryIO, request: Mapping[str, Any], *, signer: mc.Signer, log: mc.SessionLog, coins: bytes,
          domain: str, salts: mc.SaltLog | None = None, clock: Callable[[], int] = time.perf_counter_ns,
          warmups: int = WARMUPS, draw: Callable[[int], bytes] = os.urandom) -> tuple[bytes, R.ChallengePhaseRecord]:
    """The challenger's side of one session over the connection `f`: `Challenger` on `coins` and `clock`, under its
    audit's id from `salts` when `request` names one, else the id designated under `domain`, its coin tree's hiding key
    and salts from `draw`, `warmups` untimed exchanges first, the session signed by `signer`, with the collector paused
    (`memory_challenger.collector_paused`); its id and record. `Refusal` when the party refuses the session."""
    party, given = Challenger(domain, warmups, mc.resolution_of(clock)), [coins]
    with mc.collector_paused():
        done, _ = stream.run(party, request,
                             Identity(CHALLENGER, capabilities={"signer": signer, "log": log, "salts": salts}), f,
                             Frames(request["block_bytes"]), peer=PROVER, clock=clock,
                             coins=lambda n: given.pop() if given else draw(n))
    if done.refused is not None:
        raise Refusal(done.refused)
    return party.sid, party.session


def _read_clock() -> Send:
    return Send("clock", Message("ClockRead"))


def report(sid: bytes, session: R.ChallengePhaseRecord, public_key: bytes, warmups: int = WARMUPS) -> dict[str, Any]:
    return {"host": socket.gethostname(), "public_key": public_key.hex(), "session": session.to_json(),
            "session_id": sid.hex(), "warmups": warmups}


def keep(device: Path, sid: bytes, request: Mapping[str, Any]) -> Path:
    """The completed session's request and id, `device/setups/<id>.json` (0700, the file 0600, written once)."""
    d = device / SETUPS
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    path = d / f"{sid.hex()}.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"request": dict(request), "session_id": sid.hex()}, sort_keys=True) + "\n")
    return path


def _logs(device: Path) -> tuple[mc.SaltLog, mc.SessionLog]:
    """The salt log and session log in `device`, made there the first time under its lock (`period.challenger`'s)."""
    device.mkdir(parents=True, exist_ok=True)
    with open(device / "device.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, log in (("salts", mc.SaltLog), ("sessions", mc.SessionLog)):
            if not (device / name).exists():
                log.create(device / name)
    return mc.SaltLog(device / "salts"), mc.SessionLog(device / "sessions")


def _refused(e: BaseException) -> int:
    print(json.dumps({"refused": str(e)}), flush=True)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", required=True, help="the challenger's key, session log and salt log (`period.challenger`)")
    ap.add_argument("--listen", default="127.0.0.1:0", help="HOST:PORT to listen on (port 0: any)")
    a = ap.parse_args(argv)
    request = json.loads(sys.stdin.readline())
    device = Path(a.device)
    try:
        salts, sessions = _logs(device)
        if "salt" in request:
            source = give_salt(salts, request["salt"])
            print(json.dumps({"source": source.hex()}), flush=True)
            return 0
        if "audit" in request:
            designate(request, salts, sessions)
        setup_of(request)
    except (Refusal, mc.SessionLogError, KeyError, TypeError, ValueError) as e:
        return _refused(e)
    signer = period.challenger(device, workers=min(period.SETUP_WORKERS, len(os.sched_getaffinity(0))))
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        conn, _ = srv.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sid, s = serve(Polled(conn), request, signer=signer, log=sessions, salts=salts,
                           coins=secrets.token_bytes(COIN_BYTES), domain=period.SESSION_DOMAIN)
    keep(device, sid, request)
    print(json.dumps(report(sid, s, signer.public_key), sort_keys=True), flush=True)
    return 0


__all__ = ["COIN_BYTES", "PROTOCOL", "SOURCE_BYTES", "Challenger", "Refusal", "designate", "give_salt", "keep",
           "main", "report", "serve"]

if __name__ == "__main__":
    sys.exit(main())
