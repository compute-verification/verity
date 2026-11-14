"""PoUS's session checked inside proofs: the unit that accepts the memory challenger's signature, and the unit that
accepts one round's answer, so that the auditor never reads the record the memory challenger signed.

    PousSessionSig_v1{N, W, H}   (key: Array<8(16 + N)>, m: Array<512>, sig: Array<8S>) -> Array<1>
                                 S = 4 + N(1 + P + H), the signature's bytes (`lms.Params.signature_bytes`)
    PousRound_v1{N, SLOT}        (row: Array<8N>, records: Array<1472>, j: Array<32>, i: Array<64>) -> Array<1>

Every array is of `Value<1>`.  A byte string is bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋ (a
``sha512/row/v2`` row's order, `verity.commitments.rowleaf.bit_row`), and an integer is bits least significant first.
Flock stages a Definition as one ``hm96-sha512/row/v2`` row per parameter and one per output
(`verity.commitments.windows.interface.unit_rows`), so each parameter here is one row, holding the bytes as the memory
challenger writes them.

**The signature's unit.** `PousSessionSig_v1{N, W, H}` is 1 iff ``sig`` is an LMS signature (`lms.Signature.encode`:
``u32be(q) ‖ C ‖ y ‖ path``) of the 64-byte message ``m`` under the public key ``key`` (``I ‖ root``, `lms.public_key`)
with q below 2^H, as `lms.verify` decides: ``LmsVerify_v1{N, W, H}`` over q's low H bits, and q's top 32 − H bits zero
(``PousEqual_v1{32 − H}`` against zeros, ANDed in).  A q at or past 2^H is no leaf of the key's tree, and `lms.verify`
refuses it; `LmsVerify_v1` reads q's low H bits only.  ``m`` is ``lms.memory_challenger_message(I, session, record)``,
the digest the memory challenger's signature signs (`memory_challenger.LmsSigner`).  The bytes are rewired into
`gates.lms`' words (each word's bytes big-endian, its bits least significant first) without a gate.  At the deployed
scheme, ``lms-sha512/n64-w1-h20/v0`` (`lms.DEPLOYED`), it is ``PousSessionSig_v1{64,1,20}``: rows of 640, 512 and
278,048 bits and one output bit, 12 ANDs beside the verifier's.

**The round's unit.** The records are registered two a row, ⌈k/2⌉ rows of 184 bytes (92 u16 words): record j is bytes
92(j mod 2) to 92(j mod 2) + 91 of row ⌊j/2⌋, the last row zero past the last record when k is odd.  A record is
`memory_challenger.Round.value`'s 92 bytes, ``u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)``, integers
big-endian, ``H(block)`` the answer's ``sha512/row/v2`` digest.  `PousRound_v1{N, SLOT}` reads the store's N-byte row
``row``, the records row ``records`` holding record j at slot SLOT = j mod 2, and the public j and i (the round and the
index the memory challenger drew for it, a u32 and a u64), and is 1 iff record j's j is ``j``, its index is ``i`` and its
``H(block)`` is ``row``'s ``sha512/row/v2`` digest (``PousRowDigest_v1{N}``): one ``PousEqual_v1{608}`` over the digest,
j and i against the record's fields.  These are the per-round checks of `memory_challenger.verdict` but the time: the
record names round j, its index is round j's coin, and its digest is that of the block at the index.  R_even and R_odd
are ``PousRound_v1{2056, 0}`` and ``PousRound_v1{2056, 1}``, two Programs.
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.types import Array
from verity.commitments.crypto import lms as R
from verity.commitments.gates.lms import DIGEST, IDENT, _check as _lms_params, _group, _value, _words, _Wires, verifier
from verity.commitments.rowleaf import sha512_row_v2_digest

from .answer_circuit import PousEqual, PousRowDigest
from .circuit import _check, bits

__all__ = ["RECORD_BYTES", "RECORDS_PER_ROW", "PousRound", "PousSessionSig", "round_ok", "session_sig_ok"]

#: a round record's bytes (`memory_challenger.Round.value`), and the records a registered row holds
RECORD_BYTES, RECORDS_PER_ROW = 92, 2
_J, _INDEX, _DIGEST = (0, 4), (4, 12), (28, 92)


def _sig_bytes(S) -> int:
    p = _lms_params(S)
    return 4 + S.N * (1 + p.P + S.H)


def _session_sig(S):
    _lms_params(S)
    return (("key", bits(8 * (16 + S.N))), ("m", bits(512)), ("sig", bits(8 * _sig_bytes(S)))), bits(1)


def _session_body(B, S, key, m, sig):
    p, n = _lms_params(S), S.N
    X = _Wires
    kb, mb, sb = (_group(X.refs(c), 8) for c in (key, m, sig))

    def words(t, data):
        return X.coll(t, [x for w in _words(data) for x in w])

    q = [sb[3 - t // 8][t % 8] for t in range(32)]
    C, y, path = sb[4:4 + n], sb[4 + n:4 + n * (1 + p.P)], sb[4 + n * (1 + p.P):]
    ok = B.call(verifier(n, S.W, S.H), words(IDENT, kb[:16]), words(_value(n), kb[16:]), words(DIGEST, mb),
                X.coll(bits(S.H), q[:S.H]), words(_value(n), C), words(Array(p.P, _value(n)), y),
                words(Array(S.H, _value(n)), path))
    top = B.call(bind(PousEqual, N=32 - S.H), X.coll(bits(32 - S.H), q[S.H:]), X.coll(bits(32 - S.H), [B.call(BL.constant(0)).refs[0]] * (32 - S.H)))
    return X.coll(bits(1), X.refs(B.call(BL.AND, BL.operands(ok, top[0]))))


PousSessionSig = CompositeDefinition(
    "PousSessionSig", 1, ("N", "W", "H"), _session_sig, _session_body, conformance="exact-model-tested",
    doc="1 iff sig, u32be(q) || C || y || path, is an LMS signature (LmsVerify_v1{N, W, H}) of the 64-byte message m under "
        "the public key I || root, with q below 2^H; every input a byte string in byte order; sound under the claim "
        "euf-cma/lms-sha512-n64 at N = 64")


def _round_sig(S):
    _check("PousRound", N=S.N)
    if S.SLOT not in range(RECORDS_PER_ROW):
        raise ValueError(f"PousRound's SLOT is a record's place in its row, below {RECORDS_PER_ROW}, not {S.SLOT!r}")
    return (("row", bits(8 * S.N)), ("records", bits(8 * RECORD_BYTES * RECORDS_PER_ROW)), ("j", bits(32)),
            ("i", bits(64))), bits(1)


def _round_body(B, S, row, records, j, i):
    X = _Wires
    rec = _group(X.refs(records), 8)[RECORD_BYTES * S.SLOT:RECORD_BYTES * (S.SLOT + 1)]

    def integer(lo, hi):
        return [rec[hi - 1 - t // 8][t % 8] for t in range(8 * (hi - lo))]

    digest = B.call(bind(PousRowDigest, N=S.N), row)
    held = [x for b in rec[_DIGEST[0]:_DIGEST[1]] for x in b] + integer(*_J) + integer(*_INDEX)
    return B.call(bind(PousEqual, N=608), X.coll(bits(608), X.refs(digest) + X.refs(j) + X.refs(i)),
                  X.coll(bits(608), held))


PousRound = CompositeDefinition(
    "PousRound", 1, ("N", "SLOT"), _round_sig, _round_body, conformance="exact-model-tested",
    doc="1 iff record SLOT of the records row (two 92-byte memory_challenger.Round.value records) names round j and index i "
        "and holds the N-byte row's sha512/row/v2 digest: R_even at SLOT 0, R_odd at SLOT 1")


def session_sig_ok(N: int, W: int, H: int, key: bytes, m: bytes, sig: bytes) -> int:
    """`PousSessionSig_v1{N, W, H}`'s output on its rows' bytes: `lms.verify` of the decoded signature."""
    p = R.Params(N, W, H, toy=N == R.TOY_N)
    if len(key) != 16 + N or len(m) != 64 or len(sig) != p.signature_bytes():
        raise ValueError(f"the rows are a {16 + N}-byte key, a 64-byte message and a {p.signature_bytes()}-byte signature")
    return int(R.verify(p, key, m, R.Signature.decode(p, sig)))


def round_ok(N: int, SLOT: int, row: bytes, records: bytes, j: int, i: int) -> int:
    """`PousRound_v1{N, SLOT}`'s output on its rows: record SLOT names round j and index i and holds row's digest."""
    if len(row) != N or len(records) != RECORD_BYTES * RECORDS_PER_ROW or SLOT not in range(RECORDS_PER_ROW):
        raise ValueError(f"the rows are an {N}-byte row and a {RECORD_BYTES * RECORDS_PER_ROW}-byte records row")
    rec = records[RECORD_BYTES * SLOT:RECORD_BYTES * (SLOT + 1)]
    return int(int.from_bytes(rec[slice(*_J)], "big") == j and int.from_bytes(rec[slice(*_INDEX)], "big") == i
               and rec[slice(*_DIGEST)] == sha512_row_v2_digest(row, 8 * N))
