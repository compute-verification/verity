"""PoUB's period (`verity.protocols.resource_accounting.memory.poub.verifier.period`) from the prover's side: `Local`, the prover of a
store in this process; `run`, which plays every party of the period in one process (the verifier's side,
`period.session`, against `Local`); `prove` and `period_with`, the prover's end of one session with the verifier's
side as a party of its own (`verifier_party`), over its JSON lines; and `subprotocol`, bandwidth's subprotocol of
`benchmarks/all_protocols` on a period's records.

`subprotocol`'s store is the period's records (`pous.period_prover.records`, every copy of the auditor's ledger in them cut to the
entries their window cites). The verifier and its challenger are a party of their own (`verifier_party`) on
`period.json`'s `challenger_host`, with coins fresh from that host's OS, and the prover runs on the node.
`trusted-device`'s premise, that they run on a machine the developer doesn't control, holds only when the challenger
host isn't the node, and the subprotocol says which (`verifier_party.placement`). The prover reads a store in its process's
memory, not a remote node's.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol, TypeVar

from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, receipts, wire
from verity.protocols.resource_accounting.memory.pous.prover.period_prover import records
from verity.protocols.resource_accounting.memory.poub.verifier import coins as C
from verity.protocols.resource_accounting.memory.poub.verifier import period as P
from verity.protocols.resource_accounting.memory.poub.verifier.verifier_party import _recv, _send


class Local:
    """The prover in the verifier's process: `store` packed into blocks (`period.payloads`), committed under salts from
    `entropy`, and read through `cheat` (the honest store when None)."""

    def __init__(self, node: str, store: bytes, *, cheat: Callable[[audit_prover.Store], audit_prover.Store] | None = None,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        self.name, self.blocks, self.cheat, self.entropy = node.encode(), P.payloads(store), cheat, entropy
        self.salts = [entropy(receipts.SALT_BYTES) for _ in self.blocks]
        self.made: dict[tuple[int, int], audit_prover.Pass] = {}

    def register(self) -> tuple[int, bytes]:
        return len(self.blocks), audit.payload_tree(self.name, self.blocks, self.salts).commitment.root

    def hold(self, registration: audit.Registration) -> None:
        honest = audit_prover.Store(registration, self.blocks, self.salts)
        self.prover = audit_prover.Prover(self.cheat(honest) if self.cheat else honest, None, entropy=self.entropy)

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
        cheat: Callable[[audit_prover.Store], audit_prover.Store] | None = None, verify: int | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep_until: Callable[[Callable[[], int], int], None] = P._sleep_until,
        entropy: Callable[[int], bytes] = os.urandom) -> P.Run:
    """The period's session on `node` (`period`'s docstring): `store` is what the node keeps, `coins` the verifier's
    live secret for the period. `cheat` turns the honest prover's store into the one it reads, and `verify` stops the
    log after that many windows are verified (None: all), for a period that ends undecided. `clock` is the verifier's in
    ns, and `sleep_until(clock, t)` waits for it to reach t."""
    return P.session(node, Local(node, store, cheat=cheat, entropy=entropy), coins, windows=windows, passes=passes,
                     samples=samples, window_ns=window_ns, deadline_ns=deadline_ns, codec_=codec_, verify=verify,
                     clock=clock, sleep_until=sleep_until)


def prove(f: BinaryIO, prover: Local) -> int:
    """The prover's side over the connection `f`, `prover` holding the store: each message answered until the verifier
    ends the session, whose coins it checks then if the end discloses them; the passes run."""
    blocks, root = prover.register()
    _send(f, {"register": {"blocks": blocks, "payload_root": root.hex()}})
    passes = 0
    while True:
        (kind, m), = _recv(f).items()
        if kind == "hold":
            prover.hold(P.registration_of(m["registration"]))
        elif kind == "run":
            root = prover.run(C.Released.from_json(m["challenge"]), m["t"], m["i"], bytes.fromhex(m["previous"]))
            _send(f, {"root": root.hex()})
            passes += 1
        elif kind == "open":
            _send(f, {"openings": wire.encode_openings(prover.open(m["t"], [tuple(d) for d in m["draws"]])).hex()})
        elif kind == "end":
            if m:
                prover.end(C.LiveCommitment.from_json(m["commitment"]),
                           [C.LiveOpening.from_json(o) for o in m["openings"]])
            return passes
        else:
            raise ValueError(f"no message {kind!r}")


T = TypeVar("T")


class Side(Protocol):
    """The verifier's side as what starts a run hands it over (the runtime's `Side` is one): the machine it runs on,
    and `talk(request, drive)`, one session with it: `request` sent, `drive` handed the connected socket, the side's
    report back with what `drive` returned."""

    host: str

    def talk(self, request: Mapping[str, Any], drive: Callable[[socket.socket], T]) -> tuple[dict[str, Any], T]: ...


def period_with(side: Side, request: Mapping[str, Any], store: bytes, *,
                entropy: Callable[[int], bytes] = os.urandom) -> tuple[P.Run, dict[str, Any]]:
    """One session with the verifier's side `side` starts and this process the prover of `store`: the period, and the
    side's report (`host`, `passes`)."""

    def drive(s: socket.socket) -> int:
        with s.makefile("rwb") as f:
            return prove(f, Local(request["node"], store, entropy=entropy))

    got, passes = side.talk(request, drive)
    return P.Run(got["answer"], got["case"]), {"host": got["host"], "passes": passes}


def subprotocol(period_dir: Path, node: str, out: Path, *, side_of: Callable[..., Any]) -> dict[str, Any]:
    """Bandwidth's subprotocol of `benchmarks/all_protocols` (`subprotocol(period_dir, node, out, side_of=...) -> answer`): the session
    on the period's records, the verifier and its challenger a party of their own on `period.json`'s `challenger_host`
    (`side_of(period, node, key, local)`, which the caller that starts the run hands in, the runtime's `Side.of`; on
    this node by default, as a process of its own, with its own coins) and the prover here. With `poub-protocol`
    among the period's tools the Lean verifier answers and must agree with the Python; it writes
    `out/bandwidth.case.json` (onsite's bandwidth case) and `out/bandwidth.json`. Which host ran each party, and whether
    `trusted-device`'s premise holds (`verifier_party.placement`), go in `bandwidth.json` and in an accepted answer's
    profile (`placement`); `bandwidth.json` names the term its `eps` leaves out beside it (`ledger.NAMED_TERMS`)."""
    from verity.protocols.resource_accounting.memory.poub.verifier import ledger, verifier_party

    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools = doc.get("tools") or {}
    party = side_of(doc, node, "bandwidth", verifier_party.command())
    store = records(period_dir)
    t = time.monotonic()
    period, side = period_with(party, verifier_party.request_of(node), store)
    seconds = round(time.monotonic() - t, 2)
    where = verifier_party.placement(node, party.host)
    answer_, lean = period.answer, None
    if "poub-protocol" in tools:
        lean = P.serve(tools["poub-protocol"], [period.case])[0]
        if {k: v for k, v in lean.items() if k != "reason"} != {k: v for k, v in answer_.items() if k != "reason"}:
            raise RuntimeError(f"poub-protocol answered {lean}, the Python {answer_}")
        answer_ = lean
    if answer_["profile"] is not None:
        answer_ = answer_ | {"profile": answer_["profile"] | {"placement": where}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "bandwidth.case.json").write_text(json.dumps(period.case, sort_keys=True))
    (out / "bandwidth.json").write_text(json.dumps({"answer": answer_, "eps": str(period.eps), "lean": lean,
                                                    "named": ledger.NAMED_TERMS,
                                                    "placement": where | {"reported_host": side["host"]},
                                                    "q_bytes": period.public("q_bytes"), "seconds": seconds,
                                                    "store_bytes": len(store)}, sort_keys=True, indent=1))
    return answer_


__all__ = ["Local", "Side", "subprotocol", "period_with", "prove", "run"]
