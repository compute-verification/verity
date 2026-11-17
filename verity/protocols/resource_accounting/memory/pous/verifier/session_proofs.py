"""PoUS's session checked inside proofs: the verifier's side. The memory attester's signed record never reaches it in
the clear; it reads the outcomes of ``k + 2`` proofs, linked only by their committed rows.

    unit 0      PousSessionSig_v1{N, W, H}          (key, m, sig): the signature over m verifies under the key
    unit 1      PousSessionDigest_v1{k, HEAD, LIMIT} (head, ctx, r0 ... r⌈k/2⌉−1) -> (ok, m, r0 ... r⌈k/2⌉−1):
                m the digest the signature signs over the header and the records, ok that every record names its
                round and is on time, and the records rows as read
    unit j + 2  PousRound_v1{block_bytes, j mod 2}  (row, records, j, i): record j names round j and index i and holds the
                digest of the registered row at i

(`pous.prover.session_circuit`). A unit's outcome is its program, whether its proof was accepted with its ok output 1,
each parameter's committed row and then each output's as the proof system stages them (one ``hm96-sha512/row/v2`` row
each, `periods.interface.unit_rows`), and the values of the parameters that are public in its proof, which this module
computes again from the commitment, the openings and the key. The links, which the window's wires make: unit 0's ``m``
is unit 1's output m, and unit j + 2's records row is unit 1's records output ⌊j/2⌋. `check_session` is
`memory_attester.verdict` with the session's rows read only through those proofs; `run.check_at` calls
`memory_attester.verdict` until PoUS's verifier switches to it.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc

__all__ = ["SessionProofs", "UnitOutcome", "check_session", "clear_reasons", "known_reasons", "link_reasons",
           "programs"]

_ROLE = b"verity/memory-challenger-record/v1"
#: the LMS parameters (N, W, H) of each accepted scheme id, which unit 0's program is bound at
_LMS_PARAMS: Mapping[str, tuple[int, int, int]] = {mc.LMS: (64, 1, 20)}


@dataclass(frozen=True, slots=True)
class UnitOutcome:
    """One unit's proof as its verifier left it: ``program``, the bound Definition's id (``PousRound_v1{N=2056,SLOT=1}``),
    whether the proof was accepted, ``rows[p]`` parameter p's committed row, and ``known``, the parameters public in the
    proof with their values, in parameter order."""

    program: str
    accepted: bool
    rows: tuple[bytes, ...]
    known: tuple[tuple[int, bytes], ...] = ()


@dataclass(frozen=True, slots=True)
class SessionProofs:
    """Unit 0's, unit 1's and each round's outcome."""

    signature: UnitOutcome
    digest: UnitOutcome
    rounds: tuple[UnitOutcome, ...]


def _rows(k: int) -> int:
    return -(-k // 2)


def _head(commitment: mc.SessionCommitment) -> bytes:
    return json.dumps({**commitment.header(), "ended": mc.COMPLETE, "rounds": commitment.k}, sort_keys=True,
                      separators=(",", ":")).encode()


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def programs(commitment: mc.SessionCommitment, scheme: str) -> tuple[str, str, tuple[str, ...]]:
    """Unit 0's, unit 1's and each round's program: what each proof must be of."""
    n, w, h = _LMS_PARAMS[scheme]
    return (f"PousSessionSig_v1{{N={n},W={w},H={h}}}",
            f"PousSessionDigest_v1{{K={commitment.k},HEAD={len(_head(commitment))},LIMIT={commitment.limit_ns}}}",
            tuple(f"PousRound_v1{{N={commitment.block_bytes},SLOT={j % 2}}}" for j in range(commitment.k)))


def _by_j(openings: Sequence[mc.CoinOpening]) -> dict[int, mc.CoinOpening]:
    return {o.j: o for o in openings}


def clear_reasons(commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening], scheme: str, public_key: bytes,
                  *, trusted: frozenset[bytes] | set[bytes], schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> tuple[str, ...]:
    """What the verifier checks in the clear: the scheme is accepted (and its program known), the key is a trusted memory
    attester's, and every round's opening, the last one for its j, opens its committed coin."""
    reasons: list[str] = []
    if scheme not in schemes or scheme not in _LMS_PARAMS:
        reasons.append(f"signature scheme {scheme!r} is not accepted")
    elif public_key not in trusted:
        reasons.append("the session is signed by a key that is not a trusted memory challenger's")
    by_j = _by_j(openings)
    for j in range(commitment.k):
        o = by_j.get(j)
        if o is None or not mc.verify_coin(commitment, o):
            reasons.append(f"round {j}'s index is not its committed coin")
    return tuple(reasons)


def known_reasons(commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening], public_key: bytes,
                  proofs: SessionProofs, registered: Callable[[int], bytes | None]) -> tuple[str, ...]:
    """Each public parameter is the value the verifier computes: unit 0's key; unit 1's header (ended complete after k
    rounds) and the signed context; round j's j, its opened index and, as its committed
    row, the registered row at that index (`registered(i)`, none past C's blocks)."""
    reasons: list[str] = []
    ctx = _field(_ROLE) + public_key[:16] + _field(_field(commitment.session))
    by_j = _by_j(openings)
    head = _head(commitment)
    want = [(proofs.signature, "the signature's", {0: public_key}),
            (proofs.digest, "the signed digest's", {0: head + bytes(len(head) % 2), 1: ctx})]
    for j, out in enumerate(proofs.rounds):
        o = by_j.get(j)
        if o is not None:
            want.append((out, f"round {j}'s", {2: j.to_bytes(4, "little"), 3: o.index.to_bytes(8, "little")}))
            row = registered(o.index) if 0 <= o.index < commitment.blocks else None
            if row is None or out.rows[:1] != (row,):
                reasons.append(f"round {j}'s row is not the registered row at its index")
    for out, name, values in want:
        if out.known != tuple(sorted(values.items())):
            reasons.append(f"{name} public inputs are not the session's")
    return tuple(reasons)


def link_reasons(commitment: mc.SessionCommitment, proofs: SessionProofs) -> tuple[str, ...]:
    """The committed rows that link the proofs: unit 0's m (its row 1) is unit 1's output m (its row 3 + T, T = ⌈k/2⌉),
    and round j's records row (its row 1) is unit 1's records output ⌊j/2⌋ (its row 4 + T + ⌊j/2⌋)."""
    reasons: list[str] = []
    k, d, t = commitment.k, proofs.digest.rows, _rows(commitment.k)
    if len(proofs.rounds) != k or len(d) != 4 + 2 * t or len(proofs.signature.rows) != 4:
        return (f"the session's proofs are not one signature's, one signed digest's over {_rows(k)} records rows and "
                f"{k} rounds'",)
    if proofs.signature.rows[1] != d[3 + t]:
        reasons.append("the signed message is not the signed digest's")
    for j, out in enumerate(proofs.rounds):
        if len(out.rows) != 5 or out.rows[1] != d[4 + t + j // 2]:
            reasons.append(f"round {j}'s record is not the signed digest's")
    return tuple(reasons)


def check_session(commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening], scheme: str, public_key: bytes,
                  proofs: SessionProofs, registered: Callable[[int], bytes | None], *,
                  trusted: frozenset[bytes] | set[bytes], schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> mc.Verdict:
    """The session's verdict from its proofs: the clear checks, every proof accepted and of its program, every public
    input the session's, and the links. It fails closed; which rounds were late or wrong stays inside the proofs."""
    reasons = list(clear_reasons(commitment, openings, scheme, public_key, trusted=trusted, schemes=schemes))
    if scheme in _LMS_PARAMS:
        sig, dig, rounds = programs(commitment, scheme)
        units = [(proofs.signature, sig, "the signature's"), (proofs.digest, dig, "the signed digest's")]
        units += [(out, rounds[j], f"round {j}'s") for j, out in enumerate(proofs.rounds[:commitment.k])]
        for out, program, name in units:
            if out.program != program or not out.accepted:
                reasons.append(f"{name} proof is not an accepted proof of {program}")
    links = link_reasons(commitment, proofs)
    reasons += links
    if not links:
        reasons += known_reasons(commitment, openings, public_key, proofs, registered)
    return mc.Verdict(not reasons, tuple(reasons))
