"""PoUS's session staged for proofs: the ``k + 2`` statements whose proofs stand for the memory challenger's signed record
(`session_circuit`): unit 0 the signature (`PousSessionSig_v1`), unit 1 the signed digest (`PousSessionDigest_v1`), unit
j + 2 round j (`PousRound_v1`). A unit's rows are its parameters' values in order, each of a kind: ``own`` (the prover's,
committed under a hiding leaf), ``registered`` (a row of C as the setup registered it) or ``known`` (a value the verifier
computes: the key, the header, the signed context, j and i), and its outputs, each the prover's. Unit 1
outputs m and its records rows as read, which unit 0's ``m`` and each round's ``records`` read by wire
(`window.SessionIntegration`). The verifier reads the proofs' outcomes in `pous.verifier.session_proofs`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.circuits.defs import bind
from verity.commitments.crypto import lms as R

from .session_circuit import (RECORD_BYTES, RECORDS_PER_ROW, PousRound, PousSessionDigest, PousSessionSig, head_bytes,
                              records_rows, round_ok, session_digest, session_sig_ok, signed_context)

__all__ = ["KINDS", "SessionStatements", "SessionUnit", "prove_session", "session_statements"]

KINDS = ("own", "registered", "known")


@dataclass(frozen=True, slots=True)
class SessionUnit:
    """One statement: the bound Definition, each parameter's row and its kind (`KINDS`), and each output's bytes (a bit
    output one byte, 0 or 1)."""

    definition: object
    rows: tuple[bytes, ...]
    kinds: tuple[str, ...]
    outputs: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.kinds) or any(k not in KINDS for k in self.kinds):
            raise ValueError(f"a unit has one row per parameter, each of a kind in {KINDS}")


@dataclass(frozen=True, slots=True)
class SessionStatements:
    """Unit 0's, unit 1's and each round's statement."""

    signature: SessionUnit
    digest: SessionUnit
    rounds: tuple[SessionUnit, ...]


def session_statements(head: bytes, session: bytes, records: Sequence[bytes], m: bytes, key: bytes,
                       signature: bytes, indices: Sequence[int], rows: Sequence[bytes], *, limit_ns: int,
                       params: R.Params = R.DEPLOYED) -> SessionStatements:
    """The statements of a session of k = len(records) rounds: ``head`` the signed header, ``session`` the session id, ``records`` each round's 92-byte record, ``m`` the digest the signature signs,
    ``key`` the memory challenger's public key (``I ‖ root``), ``signature`` its LMS signature, ``indices`` each round's
    opened index and ``rows`` the registered row of C at each. Each unit's outputs are its reference's
    (`session_circuit`); a session whose signed digest is not ``m`` is refused, since unit 0 reads unit 1's."""
    k = len(records)
    if len(indices) != k or len(rows) != k or any(len(r) != RECORD_BYTES for r in records):
        raise ValueError(f"a session of {k} rounds has {k} records of {RECORD_BYTES} bytes, indices and registered rows")
    flat = b"".join(records) + bytes(RECORD_BYTES * (records_rows(k) * RECORDS_PER_ROW - k))
    width = RECORD_BYTES * RECORDS_PER_ROW
    packed = tuple(flat[width * t:width * (t + 1)] for t in range(records_rows(k)))
    head_row = head + bytes(head_bytes(len(head)) - len(head))
    ctx = signed_context(key[:16], session)
    ok, signed, echoed = session_digest(k, len(head), limit_ns, head_row, ctx, packed)
    if signed != m:
        raise ValueError("m is not the session's signed digest over its header and records")
    sig = SessionUnit(bind(PousSessionSig, N=params.N, W=params.W, H=params.H), (key, m, signature),
                      ("known", "own", "own"), (bytes([session_sig_ok(params.N, params.W, params.H, key, m, signature)]),))
    digest = SessionUnit(bind(PousSessionDigest, K=k, HEAD=len(head), LIMIT=limit_ns), (head_row, ctx, *packed),
                         ("known", "known", *("own",) * len(packed)), (bytes([ok]), m, *echoed))
    block = {len(r) for r in rows}
    if len(block) > 1:
        raise ValueError("the registered rows are one block size")
    rounds = tuple(SessionUnit(bind(PousRound, N=len(rows[j]), SLOT=j % RECORDS_PER_ROW),
                               (rows[j], packed[j // RECORDS_PER_ROW], j.to_bytes(4, "little"),
                                indices[j].to_bytes(8, "little")), ("registered", "own", "known", "known"),
                               (bytes([round_ok(len(rows[j]), j % RECORDS_PER_ROW, rows[j], packed[j // RECORDS_PER_ROW],
                                                j, indices[j])]),))
                   for j in range(k))
    return SessionStatements(sig, digest, rounds)


def prove_session(statements: SessionStatements):
    """Each statement's proof, one live ZK server session each, as `pous.verifier.session_proofs.SessionProofs`."""
    raise NotImplementedError("memory-accounting/pous-session-inside-proof")
