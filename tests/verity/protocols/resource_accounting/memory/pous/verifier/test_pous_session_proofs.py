"""PoUS's session checked inside proofs (`pous.verifier.session_proofs`): outcomes made from the prover's staged statements
(`pous.prover.session`) pass, and a broken link, a public input that is not the session's, a proof of another program, a
rejected proof, an untrusted key, an unknown scheme and a coin that doesn't open each reject. A committed row is a
stand-in here, SHA-512 of the row: the links compare committed rows and never read through them."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest

from verity.commitments.crypto import lms as R
from verity.protocols.resource_accounting.memory.pous.attester import memory_challenger as mc
from verity.protocols.resource_accounting.memory.pous.prover import session as P
from verity.protocols.resource_accounting.memory.pous.verifier import session_proofs as V

RNG = random.Random(3)
K, BLOCKS, BLOCK = 5, 64, 2056


def _rand(n: int) -> bytes:
    return bytes(RNG.getrandbits(8) for _ in range(n))


def _commit(row: bytes) -> bytes:
    return hashlib.sha512(b"stand-in" + row).digest()


def _outcome(unit: P.SessionUnit) -> V.UnitOutcome:
    return V.UnitOutcome(unit.definition.id, True, tuple(_commit(r) for r in (*unit.rows, *unit.outputs)),
                         tuple((p, r) for p, (r, kind) in enumerate(zip(unit.rows, unit.kinds)) if kind == "known"))


def _staged(k: int = K):
    session = _rand(32)
    tree = mc.CoinTree.draw(b"source", session, BLOCKS, k, key=mc.coin_key(_rand(mc.HM96.key_bytes)),
                            salts=[_rand(mc.HM96.salt_bytes) for _ in range(k)])
    c = mc.SessionCommitment(session, tree.root, tree.key, BLOCKS, k, 854_000, BLOCK, _rand(64))
    blocks = {i: _rand(BLOCK) for i in range(BLOCKS)}
    rounds = [mc.Round(j, tree.indices[j], 10 * j, 10 * j + 5, mc.block_digest(blocks[tree.indices[j]])) for j in range(k)]
    key = _rand(16 + 64)
    head = json.dumps({**c.header(), "ended": mc.COMPLETE, "rounds": k}, sort_keys=True, separators=(",", ":")).encode()
    m = R.memory_challenger_message(key[:16], session, mc.message(c, mc.COMPLETE, rounds))
    st = P.session_statements(head, session,
                              [r.value() for r in rounds], m, key, _rand(R.DEPLOYED.signature_bytes()), tree.indices,
                              [blocks[i] for i in tree.indices], limit_ns=c.limit_ns)
    return c, [tree.opening(j) for j in range(k)], key, st, lambda i: _commit(blocks[i])


def _setup():
    c, openings, key, st, registered = _staged()
    proofs = V.SessionProofs(_outcome(st.signature), _outcome(st.digest), tuple(_outcome(u) for u in st.rounds))
    return c, openings, key, proofs, registered


def _check(c, openings, key, proofs, registered, scheme=mc.LMS, trusted=None) -> mc.Verdict:
    return V.check_session(c, openings, scheme, key, proofs, registered, trusted={key} if trusted is None else trusted)


def test_the_staged_statements_pass() -> None:
    """The statements the prover stages, each proof accepted, pass: their programs, public inputs and links are the
    verifier's."""
    v = _check(*_setup())
    assert (v.accepted, v.reasons) == (True, ())


def _replace(o: V.UnitOutcome, **kw) -> V.UnitOutcome:
    return dataclasses.replace(o, **kw)


def _row(o: V.UnitOutcome, p: int, value: bytes) -> V.UnitOutcome:
    return _replace(o, rows=o.rows[:p] + (value,) + o.rows[p + 1:])


@pytest.mark.parametrize("break_, reason", [
    (lambda p: dataclasses.replace(p, signature=_row(p.signature, 1, bytes(64))), "the signed message is not"),
    (lambda p: dataclasses.replace(p, rounds=p.rounds[:3] + (_row(p.rounds[3], 1, p.digest.rows[-1]),) + p.rounds[4:]),
     "round 3's record is not"),
    (lambda p: dataclasses.replace(p, rounds=p.rounds[:-1]), "the session's proofs are not"),
    (lambda p: dataclasses.replace(p, digest=_replace(p.digest, accepted=False)), "the signed digest's proof is not"),
    (lambda p: dataclasses.replace(p, rounds=(_replace(p.rounds[0], program=p.rounds[1].program),) + p.rounds[1:]),
     "round 0's proof is not"),
    (lambda p: dataclasses.replace(p, digest=_replace(p.digest, known=p.digest.known[:2] + ((2, bytes(1)),))),
     "the signed digest's public inputs"),
    (lambda p: dataclasses.replace(p, rounds=(_row(p.rounds[0], 0, bytes(64)),) + p.rounds[1:]),
     "round 0's row is not the registered row"),
])
def test_a_broken_proof_rejects(break_, reason) -> None:
    c, openings, key, proofs, registered = _setup()
    v = _check(c, openings, key, break_(proofs), registered)
    assert not v.accepted and any(r.startswith(reason) for r in v.reasons), v.reasons


def test_the_clear_checks_reject() -> None:
    """An untrusted key, an unknown scheme and a round whose coin doesn't open each reject."""
    c, openings, key, proofs, registered = _setup()
    assert "not a trusted" in " ".join(_check(c, openings, key, proofs, registered, trusted={bytes(80)}).reasons)
    assert "is not accepted" in " ".join(_check(c, openings, key, proofs, registered, scheme="test/x").reasons)
    bad = openings[:2] + [dataclasses.replace(openings[2], index=(openings[2].index + 1) % BLOCKS)] + openings[3:]
    assert "round 2's index is not its committed coin" in _check(c, bad, key, proofs, registered).reasons


def test_proving_is_not_written_yet() -> None:
    c, openings, key, proofs, registered = _setup()
    with pytest.raises(NotImplementedError, match="memory-accounting/pous-session-inside-proof"):
        P.prove_session(proofs)


@pytest.mark.parametrize("k", [1, K])
def test_the_session_is_a_window_circuit(k) -> None:
    """The staged statements as a window's circuit (`window.SessionIntegration`): its templates and wires pass the
    window's checks, one unit each for the signature and the digest and one a round, at a K with no odd round too."""
    from verity.circuits.codec import descriptor_id
    from verity.protocols.resource_accounting.memory.pous.prover.window import SessionIntegration

    st = _staged(k)[3]
    circuit = SessionIntegration(st).circuit()
    assert circuit.units == {descriptor_id(st.signature.definition): 1, descriptor_id(st.digest.definition): 1,
                             **{descriptor_id(st.rounds[s].definition): len(st.rounds[s::2]) for s in range(min(k, 2))}}
    assert len(circuit.wires) == 1 + k
