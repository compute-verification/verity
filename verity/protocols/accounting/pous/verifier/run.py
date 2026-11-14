"""The composed check of a recorded PoUS run: what setup published and what the memory challenger signed, joined.

A run is the setup's public outputs (`verity/pous/p2/v4/setup` on acceptance, `{"blocks", "commitment", "root", "salt"}`,
`root` the root of `C*`'s leaves under `salt`), the memory challenger's table of `C*` (every block's `sha512/row/v2` digest
in block order, `memory_challenger.block_digest`: the harness's `DIGESTS.bin`) and its record of one signed session
(`ChallengePhaseRecord`: the signed session, the commitment it released with it and its coin openings).  `check` judges
it with two inputs of the verifier's own: `trusted`, the verifier's list of memory-challenger keys, which never comes
from the run's record (whoever writes the record could otherwise list its own key), and the 32-byte id the audit fixes
for its session, which the verifier computes itself and never reads from the record: on the served piece
`challenger_party.session_id`, derived from the audit's binding and the commitment and salt the memory challenger logged
for it.  It accepts the run at P2's deployed point (`DEPLOYED`: `2^19` blocks of 2,056 bytes in 8 segments, `k = 105`,
`Δ + RTT ≤ 854,000 ns`) under S1's LMS (`memory_challenger.SCHEMES`) when, in order:

1. the setup published the point's block count and a 24-byte salt;
2. the table holds one 64-byte digest a block, and its leaves under that salt (`verifier.row_leaf`) have the root the setup
   published, so the table is the one of the `C*` setup checked;
3. the session audits that setup at the point: every block, the point's `k`, a limit within the point's `Δ + RTT`, answers
   of the point's block size, and its `binding` the published root;
4. the session runs under the audit's id (`session`, in the signed header), which the memory challenger runs at most one
   session under (`memory_challenger.SessionLog`), so an audit has at most one session `check` accepts, and a retry
   under another id is refused here;
5. `memory_challenger.verdict` accepts the session against the table, signed under a key in `trusted`.

The first refusal is the reason (`Refused`), word for word the Lean's.  `check_at` takes another point and other schemes, for tests.
`PousCheck.Run` (`pous-run`, in `check/`) is the check in Lean, held to this by `check/tests/test_run.py`; what its
acceptance establishes is `RunCheckAccepts` (`security/Properties/Pous/EndToEnd.lean`), proved.

`check_signed` (`check_signed_at` at another point) is the run on the memory challenger's signed setup
(`PousCheck.SignedRun`): in place of the table it reads the setup record the session's key signed under the session's id
(`memory_challenger.SetupRecord`) and one `Opening` per round, the row of the block the round asks for and its path to
the published root.  It checks step 1 and steps 3 to 4 above, then that each opening's row is under the root, then the
verdict against the rows the openings opened (an answer that is not its block is wrong), then that the setup record is
signed by the session's key and scheme under the session's id and holds the published blocks, commitment, salt and
root.  Setup's acceptance it does not check: it is the open premise `period.SETUP_ACCEPTED`.  Its guarantee is not
proved."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.protocols.accounting.pous.device import memory_challenger as mc

from .schemes.p2 import (
    CERT_BLOCKS,
    CERT_K,
    DELTA_NS,
    POINT,
    RTT_ALLOWANCE_CAP_NS,
    P2Params,
)
from .verifier import DIGEST_BYTES, VerifierKey, row_leaf, verify_path

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
class ChallengePhaseRecord:
    """One session as the memory challenger records it: what it signed, the commitment it released with it and its coin
    openings.  Which keys are trusted is the verifier's, never the record's (`check`'s `trusted`)."""

    signed: mc.SignedSession
    commitment: mc.SessionCommitment
    openings: tuple[mc.CoinOpening, ...]

    def to_json(self) -> dict[str, Any]:
        return {"commitment": self.commitment.to_json(), "openings": [o.to_json() for o in self.openings],
                "signed": self.signed.to_json()}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> ChallengePhaseRecord:
        return cls(mc.SignedSession.from_json(doc["signed"]), mc.SessionCommitment.from_json(doc["commitment"]),
                   tuple(mc.CoinOpening.from_json(o) for o in doc["openings"]))


class Refused(ValueError):
    """The run is refused; the message is the first reason."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise Refused(reason)


def check_at(point: Point, schemes: Mapping[str, mc.Verify], trusted: Collection[bytes], setup: Setup, digests: bytes,
             session: ChallengePhaseRecord, session_id: bytes) -> None:
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
    _need(sc.session == session_id, "the session does not run under the id the auditor designated")
    v = mc.verdict(session.signed, sc, session.openings, lambda i: rows[i] if 0 <= i < q.blocks else None,
                   trusted=frozenset(trusted), schemes=schemes)
    _need(v.accepted, "; ".join(v.reasons))


@dataclass(frozen=True, slots=True)
class Opening:
    """One round's opening for `check_signed`: the row digest of the block its round asks for, and its Merkle path,
    siblings from the leaf up (`verifier.MerkleTree.path`)."""

    row: bytes
    path: tuple[bytes, ...]

    def to_json(self) -> dict[str, Any]:
        return {"path": [p.hex() for p in self.path], "row": self.row.hex()}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Opening:
        return cls(bytes.fromhex(doc["row"]), tuple(bytes.fromhex(p) for p in doc["path"]))


def check_signed_at(point: Point, schemes: Mapping[str, mc.Verify], trusted: Collection[bytes], setup: Setup,
                    session: ChallengePhaseRecord, signed_setup: mc.SignedSetupRecord, openings: Sequence[Opening],
                    session_id: bytes) -> None:
    """`check_signed` at `point` under `schemes`."""
    q, sc, rounds, r = point.params, session.commitment, session.signed.rounds, signed_setup.record
    _need(setup.blocks == q.blocks, f"the setup published {setup.blocks} blocks, not the point's {q.blocks}")
    _need(len(setup.salt) == SALT_BYTES, f"the setup's salt is {len(setup.salt)} bytes, not {SALT_BYTES}")
    _need(sc.blocks == q.blocks, f"the session audits {sc.blocks} blocks, not the setup's {q.blocks}")
    _need(sc.k == point.k, f"the session runs {sc.k} rounds, not the point's {point.k}")
    _need(sc.limit_ns <= point.limit_ns,
          f"the session's limit of {sc.limit_ns} ns is past the point's Δ + RTT = {point.limit_ns} ns")
    _need(sc.block_bytes == q.block_bytes, f"the session's answers are {sc.block_bytes} bytes, not the point's {q.block_bytes}")
    _need(sc.binding == setup.root, "the session is not bound to the root the setup published")
    _need(sc.session == session_id, "the session does not run under the id the auditor designated")
    _need(len(openings) == len(rounds),
          f"the case holds {len(openings)} openings, not one for each of the session's {len(rounds)} rounds")
    for j, (rd, o) in enumerate(zip(rounds, openings)):
        _need(0 <= rd.index < q.blocks and
              verify_path(setup.root, q.blocks, rd.index, row_leaf(q, setup.salt, rd.index, o.row), o.path),
              f"round {j}'s opening is not a row under the root the setup published")
    opened: dict[int, bytes] = {}
    for rd, o in zip(rounds, openings):
        opened.setdefault(rd.index, o.row)
    v = mc.verdict(session.signed, sc, session.openings, opened.get, trusted=frozenset(trusted), schemes=schemes)
    _need(v.accepted, "; ".join(v.reasons))
    _need(mc.setup_record_ok(signed_setup, scheme=session.signed.scheme, public_key=session.signed.public_key,
                             session=session_id, schemes=schemes),
          "the setup record is not signed by the session's key under the session's id")
    _need(r.blocks == setup.blocks, f"the setup record holds {r.blocks} blocks, not the published {setup.blocks}")
    _need(r.commitment == setup.commitment, "the setup record's commitment to W is not the published one")
    _need(r.salt == setup.salt, "the setup record's salt is not the published one")
    _need(r.root == setup.root, "the setup record's root is not the published one")


def check_signed(trusted: Collection[bytes], setup: Setup, session: ChallengePhaseRecord,
                 signed_setup: mc.SignedSetupRecord, openings: Sequence[Opening], session_id: bytes) -> None:
    """Accept the run on the memory challenger's signed setup (`PousCheck.SignedRun`), or raise `Refused` with the first
    reason, at `DEPLOYED` under S1's LMS: in place of the table, the setup record the session's key signed under the
    session's id, and one opening per round, its row under the published root.  Its guarantee is not proved."""
    check_signed_at(DEPLOYED, mc.SCHEMES, trusted, setup, session, signed_setup, openings, session_id)


def check(trusted: Collection[bytes], setup: Setup, digests: bytes, session: ChallengePhaseRecord,
          session_id: bytes) -> None:
    """Accept the run, or raise `Refused` with the first reason (module docstring).  `trusted` is the verifier's list of
    memory-challenger keys and `session_id` the id the audit fixes; neither comes from the run's record."""
    check_at(DEPLOYED, mc.SCHEMES, trusted, setup, digests, session, session_id)


__all__ = ["DEPLOYED", "SALT_BYTES", "ChallengePhaseRecord", "Opening", "Point", "Refused", "Setup", "check", "check_at",
           "check_signed", "check_signed_at"]
