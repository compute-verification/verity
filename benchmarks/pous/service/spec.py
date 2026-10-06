"""PoUS through the proof service: its `Spec`, the unit each answered round proves, C's hiding commitment, the
challenger's own copy of a session's ports, and PoUS's claims check.

**What PoUS commits** (`VALUES`). Before the session the prover commits `C` once: one ``hm96-sha512/row/v2`` leaf per
block (the block's ``sha512/row/v2`` digest under hm96's pinned key and a fresh secret salt `y_i`, `commit_c`), in a
frame-v3-sha512 tree whose root the memory challenger's signed header binds (`session_binding`). After the session
it commits the round's answers as the unit's two rows (`block`, `o0`, phase ``serve``): `block` row j is the block
answered in round j under C's salt `y_{i_j}`, so its commit string is C's leaf `i_j` exactly; `o0` is the same block
under the zero salt, whose commit string is `x ‖ H(0)` with `x` the round's signed `H(block)`.

**The check** (`unit`, `PousAnswerRow{N}`): the identity on an N-byte block, as a public Program. C-Flock proves each
instance's two rows are the same bits and publishes both commit strings; PoUS's claims check (`claims_check`) then holds
row j's two commit strings to C's leaf at the round's index (an opening under C's root) and to the signed `H(block)`.
That is `answer_circuit.PousAnswerOpens_v1`'s statement split at its public wires: the digest and the commit string
are C-Flock's row commitments, computed by the frame, not by gates (`PousAnswerOpens_v1` lowers to about two million
ANDs a round, which IL.lower runs out of memory on).

**The selection rule** is `All`: every round is proved. PoUS's certificate needs all `k` answers right, and the
memory challenger's coin tree is already the sample over C, so a second sample over the rounds would only weaken it.

**Public items** (`PUBLIC`), each with its reason. The outcome's one output is the verdict. The memory challenger's
signed session (header, round records, signature) and coin openings are visible to the claims check, and so are the
answer digests `x` (`o0`'s commit strings): the signature is verified in the clear until LMS is verified in the
proof (`verity.primitives.commitments.gates.lms`), which is the gap between this and a verdict-only audit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.primitives.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.primitives.circuits.program import Program
from verity.primitives.commitments import hm96
from verity.primitives.commitments.identity import identity_digest
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.limits import VerificationLimits
from verity.primitives.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening
from verity.primitives.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity.protocols.accounting.space.pous import memory_challenger as mc
from verity.protocols.accounting.space.pous.circuit import _traced_body, bits
from verity.protocols.verification.sampled_proofs import service as S
from verity.protocols.verification.sampled_proofs.one_stage import partition as P, registration as R

SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
HM = hm96.SHA512
KEY = HM.default_key
ZERO_SALT = bytes(HM.salt_bytes)
PORTS = ("block", "o0")
C_BINDING = b"verity-pous/service/c-commitment/v1\0"
C_SALT = b"verity-pous/service/c-salt/v1\0"
SESSION_BINDING = b"verity-pous/service/session-binding/v1\0"
FRAME_BINDING = "verity/flock-ir-frame/binding/v1"
CONTEXT = b"verity-pous/service/memory-challenger/v1"
CHALLENGER = "pous-memory-challenger"
LIMITS = VerificationLimits()


def _sig(S_):
    return (("block", bits(8 * S_.N)),), bits(8 * S_.N)


PousAnswerRow = CompositeDefinition(
    "PousAnswerRow", 1, ("N",), _sig, _traced_body(lambda S_: ("answer-row", S_.N), lambda S_: (lambda C, b: list(b)), _sig),
    register=False, conformance="protocol-defined",
    doc="The answered block, unchanged: its two rows (C's salt, the zero salt) are one block, by C-Flock's row commitments.")


def unit(block_bytes: int) -> SpecializedDefinition:
    return bind(PousAnswerRow, N=block_bytes)


VALUES = (S.Value("C", SCHEMA, "epoch"), S.Value("block", SCHEMA, "serve", fresh=True), S.Value("o0", SCHEMA, "serve", fresh=True))
PUBLIC = (
    S.Public("verdict", "output", "PoUS's output: the audit passed or it did not"),
    S.Public("population", "structure", "k, the session's rounds: every one is proved"),
    S.Public("window", "structure", "the session the rounds are of, by its id"),
    S.Public("leaves:block", "structure", "one row a round"),
    S.Public("domain:block", "structure", "the challenger's own domain for the session's answers"),
    S.Public("leaves:o0", "structure", "one row a round"),
    S.Public("domain:o0", "structure", "the challenger's own domain for the session's answer digests"),
    S.Public("C", "commitment", "C's salted root, fixed before the session and bound by the signed header"),
    S.Public("paths:C", "structure", "the opened leaves' paths under C's root: hiding leaves, so salted digests only"),
    S.Public("coins", "coins", "the memory challenger's coin openings: which blocks the rounds asked for"),
    S.Public("session", "structure", "the signed session (header, records, signature), verified in the clear until LMS "
                                     "is verified in the proof"),
    S.Public("digests", "structure", "each round's H(block), o0's commit string under the zero salt: in the signed "
                                     "records already, while those are in the clear"),
)


def spec(block_bytes: int) -> S.Spec:
    u = unit(block_bytes)
    return S.Spec(VALUES, S.All(), S.Check(P.program_sha512(Program(u)), P.template_instance_query(u)), PUBLIC, CONTEXT)


def c_salt(salt_key: bytes, i: int) -> bytes:
    """`y_i`: block i's hm96 salt, from the prover's secret salt key."""
    return hashlib.shake_256(C_SALT + salt_key + i.to_bytes(8, "big")).digest(HM.salt_bytes)


def c_domain(setup: bytes, blocks: int) -> CommitmentDomain:
    return CommitmentDomain(hashlib.sha512(C_BINDING + setup).digest()[:32], -1, RangeIndexedDomain(0, blocks), hash="sha512")


@dataclass
class CTree:
    """C's hiding commitment (module docstring): `root` over `blocks` leaves under `c_domain(setup, blocks)`."""

    setup: bytes
    blocks: int
    root: bytes
    levels: tuple[tuple[bytes, ...], ...]

    def path(self, i: int) -> list[bytes]:
        out, r = [], i
        for level in self.levels[:-1]:
            out.append(level[r ^ 1])
            r >>= 1
        return out


def commit_c(digests: Sequence[bytes], salt_key: bytes, setup: bytes) -> CTree:
    """C's leaves over its blocks' ``sha512/row/v2`` digests (`memory_challenger.block_digest`), salted by `c_salt`;
    `setup` binds the domain (the setup salt and the commitment to W)."""
    dom = c_domain(setup, len(digests))
    leaves = [dom.leaf(i, i, SCHEMA, HM.tree_leaf(KEY, HM.commit_string(KEY, d, c_salt(salt_key, i))))
              for i, d in enumerate(digests)]
    levels = tree_levels(dom, leaves)
    return CTree(setup, len(digests), levels[-1][0], levels)


def session_binding(c_root: bytes, blocks: int) -> bytes:
    """What the memory challenger's header binds (`SessionCommitment.binding`): C's root and block count."""
    return hashlib.sha512(SESSION_BINDING + blocks.to_bytes(8, "big") + c_root).digest()


def set_name(session: bytes) -> str:
    return f"pous-answers-{session.hex()[:32]}"


def ports(session: bytes, k: int) -> dict[str, dict[str, Any]]:
    """The challenger's own copy of a session's ports: `k` rows each, under the frame domain C-Flock binds to the set
    the session names (`verity_flock.circuit.statement`)."""
    out = {}
    for p in PORTS:
        b = bytes.fromhex(identity_digest(FRAME_BINDING, {"set": set_name(session), "lo": 0, "hi": k, "port": p, "schema": SCHEMA}))
        out[p] = {"leaves": k, "domain": CommitmentDomain(b, -1, RangeIndexedDomain(0, k), hash="sha512").domain_id.hex()}
    return out


def record(block_bytes: int, session: bytes, k: int, roots: Mapping[str, str]) -> dict[str, Any]:
    """The registration of one session's answers: one-stage's record over `k` instances of the unit."""
    u, prt = unit(block_bytes), ports(session, k)
    return R.record(P.population_program(u, k), P.template_instance_query(u), {"law": "subset", "k": k}, S.SCHEME,
                    [{"port": p, "root": roots[p], **prt[p]} for p in PORTS], {"kind": "served", "session": session.hex()})


def read_public(pub: bytes) -> tuple[dict[str, Any], dict[str, list[bytes]]]:
    """C-Flock's public file: its header, and each port's commit strings, instance-major in `PORTS` order."""
    head_raw, body = pub.split(b"\n", 1)
    head = json.loads(head_raw)
    n = int(head["instances"])
    if sorted(head["frame_v3"]["roots"]) != sorted(PORTS) or len(body) < 128 * n * len(PORTS):
        raise ValueError("not a public file of the unit's two rows")
    bc = {p: [body[128 * (len(PORTS) * i + j):128 * (len(PORTS) * i + j + 1)] for i in range(n)] for j, p in enumerate(PORTS)}
    return head, bc


def claims_check(signed: mc.SignedSession, commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening],
                 c_root: bytes, c_setup: bytes, c_paths: Sequence[Sequence[bytes]], pub: bytes, *,
                 trusted: frozenset[bytes] | set[bytes], schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> dict[str, Any]:
    """PoUS's own check of a proved session, on public items: the memory challenger's verdict on the signed session
    with the digests left to the proof, the header binding C's root, and each round's two commit strings in `pub` (the
    file the verifier of record verified): `block`'s opens C's leaf at the round's index (`c_paths[j]`), `o0`'s is the
    signed `H(block)` under the zero salt. Returns `{"accepted", "reasons"}`; the reasons are witness."""
    v = mc.verdict(signed, commitment, openings, None, trusted=trusted, schemes=schemes)
    reasons = list(v.reasons)
    if commitment.binding != session_binding(c_root, commitment.blocks):
        reasons.append("the signed header does not bind C's root")
    try:
        head, bc = read_public(pub)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        return {"accepted": False, "reasons": [*reasons, f"public file: {e}"]}
    rounds = signed.rounds
    if int(head["instances"]) != len(rounds) or len(c_paths) != len(rounds):
        reasons.append(f"{head['instances']} proved rows and {len(c_paths)} paths for {len(rounds)} rounds")
        return {"accepted": False, "reasons": reasons}
    dom = c_domain(c_setup, commitment.blocks)
    leaf = digest = 0
    for j, r in enumerate(rounds):
        try:
            ok = verify_opening(dom, Commitment(c_root, commitment.blocks),
                                Opening(r.index, HM.tree_leaf(KEY, bc["block"][j]), tuple(c_paths[j])), SCHEMA, LIMITS)
        except Exception:  # noqa: BLE001 - a malformed path opens nothing
            ok = False
        leaf += not ok
        digest += bc["o0"][j] != HM.commit_string(KEY, r.digest, ZERO_SALT)
    if leaf:
        reasons.append(f"{leaf} answered blocks are not C's leaves at their rounds' indices")
    if digest:
        reasons.append(f"{digest} answered blocks are not the signed H(block)")
    return {"accepted": not reasons, "reasons": reasons}


def answer_rows(blocks: Sequence[bytes]) -> Any:
    import numpy as np

    return {"block": np.stack([np.frombuffer(b, "<u2") for b in blocks])}
