"""The composed check of a recorded PoUS run: what setup published and what the memory challenger signed, joined.

A run is the setup's public outputs (`verity/pous/p2/v4/setup` on acceptance, `{"blocks", "commitment", "root", "salt"}`,
`root` the root of `C*`'s leaves under `salt`), the memory challenger's table of `C*` (every block's `sha512/row/v2` digest
in block order, `memory_challenger.block_digest`: the harness's `DIGESTS.bin`) and one signed session (`Session`: the signed
session, the commitment published before it, its coin openings and the trusted memory challengers' keys).  `check` accepts
it at P2's deployed point (`DEPLOYED`: `2^19` blocks of 2,056 bytes in 8 segments, `k = 105`, `Δ + RTT ≤ 854,000 ns`)
under S1's LMS (`memory_challenger.SCHEMES`) when, in order:

1. the setup published the point's block count and a 24-byte salt;
2. the table holds one 64-byte digest a block, and its leaves under that salt (`verifier.row_leaf`) have the root the setup
   published, so the table is the one of the `C*` setup checked;
3. the session audits that setup at the point: every block, the point's `k`, a limit within the point's `Δ + RTT`, answers
   of the point's block size, and its `binding` the published root;
4. `memory_challenger.verdict` accepts the session against the table.

The first refusal is the reason (`Refused`).  `check_at` takes another point and other schemes, for tests.
`PousCheck.Run` (`pous-run`, in `check/`) is the check in Lean, held to this by `check/tests/test_run.py`; what its
acceptance establishes is `RunCheckAccepts` (`Security/Properties/Pous/EndToEnd.lean`), proved."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import memory_challenger as mc
from .schemes.p2 import CERT_BLOCKS, CERT_K, DELTA_NS, POINT, RTT_ALLOWANCE_CAP_NS, P2Params
from .verifier import DIGEST_BYTES, VerifierKey

SALT_BYTES = 24
"""The setup salt's bytes (`labels.SALT_BITS / 8`)."""


@dataclass(frozen=True, slots=True)
class Point:
    """What a run is checked at: the codec's parameters, the rounds `k` and the largest limit `Δ + RTT`, in ns."""

    params: P2Params
    k: int
    limit_ns: int


DEPLOYED = Point(P2Params(*POINT, segments=CERT_BLOCKS // POINT[2]), CERT_K, DELTA_NS + RTT_ALLOWANCE_CAP_NS)
"""P2's deployed point: `p2-p16448-c21065`, `2^19` blocks of 2,056 bytes, `k = 105`, `Δ + RTT ≤ 854,000 ns`."""


@dataclass(frozen=True, slots=True)
class Setup:
    """The setup's public outputs (`PousCheck.SetupProtocol.Public`)."""

    blocks: int
    commitment: bytes
    root: bytes
    salt: bytes

    def to_json(self) -> dict[str, Any]:
        return {"blocks": self.blocks, "commitment": self.commitment.hex(), "root": self.root.hex(), "salt": self.salt.hex()}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Setup:
        return cls(int(doc["blocks"]), bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["root"]),
                   bytes.fromhex(doc["salt"]))


@dataclass(frozen=True, slots=True)
class Session:
    """One session as the memory challenger records it, and the keys it is judged under."""

    signed: mc.SignedSession
    commitment: mc.SessionCommitment
    openings: tuple[mc.CoinOpening, ...]
    trusted: tuple[bytes, ...]

    def to_json(self) -> dict[str, Any]:
        return {"commitment": self.commitment.to_json(), "openings": [o.to_json() for o in self.openings],
                "signed": self.signed.to_json(), "trusted": [k.hex() for k in self.trusted]}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Session:
        return cls(mc.SignedSession.from_json(doc["signed"]), mc.SessionCommitment.from_json(doc["commitment"]),
                   tuple(mc.CoinOpening.from_json(o) for o in doc["openings"]), tuple(bytes.fromhex(k) for k in doc["trusted"]))


class Refused(ValueError):
    """The run is refused; the message is the first reason."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise Refused(reason)


def check_at(point: Point, schemes: Mapping[str, mc.Verify], setup: Setup, digests: bytes, session: Session) -> None:
    """`check` at `point` under `schemes`."""
    q, sc = point.params, session.commitment
    _need(setup.blocks == q.blocks, f"the setup published {setup.blocks} blocks, not the point's {q.blocks}")
    _need(len(setup.salt) == SALT_BYTES, f"the setup's salt is {len(setup.salt)} bytes, not {SALT_BYTES}")
    _need(len(digests) == q.blocks * DIGEST_BYTES,
          f"the table holds {len(digests)} bytes, not a {DIGEST_BYTES}-byte digest for each of {q.blocks} blocks")
    rows = [digests[DIGEST_BYTES * i:DIGEST_BYTES * (i + 1)] for i in range(q.blocks)]
    _need(VerifierKey.from_row_digests(q, setup.salt, rows).root == setup.root,
          "the table's leaves do not have the root the setup published")
    _need(sc.blocks == q.blocks, f"the session audits {sc.blocks} blocks, not the setup's {q.blocks}")
    _need(sc.k == point.k, f"the session runs {sc.k} rounds, not the point's {point.k}")
    _need(sc.limit_ns <= point.limit_ns,
          f"the session's limit of {sc.limit_ns} ns is past the point's Δ + RTT = {point.limit_ns} ns")
    _need(sc.block_bytes == q.block_bytes, f"the session's answers are {sc.block_bytes} bytes, not the point's {q.block_bytes}")
    _need(sc.binding == setup.root, "the session is not bound to the root the setup published")
    v = mc.verdict(session.signed, sc, session.openings, lambda i: rows[i] if 0 <= i < q.blocks else None,
                   trusted=frozenset(session.trusted), schemes=schemes)
    _need(v.accepted, "; ".join(v.reasons))


def check(setup: Setup, digests: bytes, session: Session) -> None:
    """Accept the run, or raise `Refused` with the first reason (module docstring)."""
    check_at(DEPLOYED, mc.SCHEMES, setup, digests, session)


__all__ = ["DEPLOYED", "SALT_BYTES", "Point", "Refused", "Session", "Setup", "check", "check_at"]
