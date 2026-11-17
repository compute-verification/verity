"""PoUS's session checked inside proofs: the unit that accepts the memory attester's signature, the unit that computes
the digest it signs, and the unit that accepts one round's answer, so that the auditor never reads the record the memory
attester signed.

    PousSessionSig_v1{N, W, H}          (key: B<16 + N>, m: B<64>, sig: B<S>) -> Array<1>
                                        S = 4 + N(1 + P + H), the signature's bytes (`lms.Params.signature_bytes`)
    PousSessionDigest_v1{K, HEAD, LIMIT} (head: B<HEAD + HEAD mod 2>, ctx: B<106>, r0 ... rT−1: B<184>)
                                        -> (ok: Array<1>, m: B<64>, r0 ... rT−1: B<184>), T = ⌈K/2⌉
    PousRound_v1{N, SLOT}               (row: B<N>, records: B<184>, j: B<4>, i: B<8>) -> Array<1>

B<n> is n bytes, ``Array<n, Array<8, Value<1>>>`` (`octets`), each byte's bits least significant first, so a row's bits
are in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋ (a ``sha512/row/v2`` row's order,
`verity.commitments.rowleaf.bit_row`); an integer (j, i) is its bytes little-endian, so its bits least significant first.
Flock stages a Definition as one ``hm96-sha512/row/v2`` row per parameter and one per output
(`verity.commitments.periods.interface.unit_rows`), whole u16 words of one element width, so each parameter here is one
row, holding the bytes as the memory attester writes them.

**The signature's unit.** `PousSessionSig_v1{N, W, H}` is 1 iff ``sig`` is an LMS signature (`lms.Signature.encode`:
``u32be(q) ‖ C ‖ y ‖ path``) of the 64-byte message ``m`` under the public key ``key`` (``I ‖ root``, `lms.public_key`)
with q below 2^H, as `lms.verify` decides: ``LmsVerify_v1{N, W, H}`` over q's low H bits, and q's top 32 − H bits zero
(``PousEqual_v1{32 − H}`` against zeros, ANDed in).  A q at or past 2^H is no leaf of the key's tree, and `lms.verify`
refuses it; `LmsVerify_v1` reads q's low H bits only.  ``m`` is ``lms.memory_attester_message(I, session, record)``,
the digest the memory attester's signature signs (`memory_attester.LmsSigner`); the registered row ``m`` is its 64
bytes in byte order, which is how the unit that computes ``m`` must output it.  The bytes are rewired into
`gates.lms`' words (each word's bytes big-endian, its bits least significant first) without a gate.  At the deployed
scheme, ``lms-sha512/n64-w1-h20/v0`` (`lms.DEPLOYED`), it is ``PousSessionSig_v1{64,1,20}``: rows of 640, 512 and
278,048 bits and one output bit, 12 ANDs beside the verifier's.

**The round's unit.** The records are registered two a row, ⌈k/2⌉ rows of 184 bytes (92 u16 words): record j is bytes
92(j mod 2) to 92(j mod 2) + 91 of row ⌊j/2⌋, the last row zero past the last record when k is odd.  A record is
`memory_attester.Round.value`'s 92 bytes, ``u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)``, integers
big-endian, ``H(block)`` the answer's ``sha512/row/v2`` digest.  `PousRound_v1{N, SLOT}` reads the store's N-byte row
``row``, the records row ``records`` holding record j at slot SLOT = j mod 2, and the public j and i (the round and the
index the memory attester drew for it, a u32 and a u64), and is 1 iff record j's j is ``j``, its index is ``i`` and its
``H(block)`` is ``row``'s ``sha512/row/v2`` digest (``PousRowDigest_v1{N}``): one ``PousEqual_v1{608}`` over the digest,
j and i against the record's fields.  These are the per-round checks of `memory_attester.verdict` but the time: the
record names round j, its index is round j's coin, and its digest is that of the block at the index.  R_even and R_odd
are ``PousRound_v1{2056, 0}`` and ``PousRound_v1{2056, 1}``, two Programs.

**The signed digest's unit.** `PousSessionDigest_v1{K, HEAD, LIMIT}` reads the signed header (``head``, HEAD bytes and a
zero byte when HEAD is odd), the signed context (``ctx``, `signed_context`: what `lms.memory_attester_message` hashes
before the record) and the K records in their rows. It outputs m, ``SHA-512(ctx ‖ u64be(64) ‖ record)`` of the record
``SHA-512(SESSION_TAG ‖ u64be(HEAD) ‖ head ‖ records)``, records the ``sha512/row/v2`` digest of the K records as one
stream (`memory_attester.records_digest`, ``PousRowDigest_v1{92K}``), so m is the digest the signature's unit verifies;
ok, 1 iff every
record names its round, its receipt is not before its send and receipt − send ≤ LIMIT (``PousVerdict_v1{K, 0, LIMIT}``
against the records' own digests); and the records rows as read. The signature's unit reads m and round j's unit its
records row ⌊j/2⌋ by wire (`window.SessionIntegration`), so the three kinds of proof are linked only by committed rows.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import part as _part
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple
from verity.commitments.crypto import lms as R
from verity.commitments.gates import frame_v3 as F
from verity.commitments.gates.lms import DIGEST, IDENT, _check as _lms_params, _group, _value, _words, _Wires, verifier
from verity.commitments.gates.sha512 import pad_tail
from verity.commitments.rowleaf import sha512_row_v2_digest

from .answer_circuit import PousEqual, PousRowDigest
from .circuit import _check, bits
from .verdict_circuit import PousVerdict, verdict_value

__all__ = ["CTX_BYTES", "RECORD_BYTES", "RECORDS_PER_ROW", "SESSION_BYTES", "SESSION_TAG", "PousRound", "PousSessionDigest",
           "PousSessionMessage", "PousSessionSig", "head_bytes", "octets", "records_rows", "round_ok", "session_digest",
           "session_sig_ok", "signed_context"]

#: a round record's bytes (`memory_attester.Round.value`), and the records a registered row holds
RECORD_BYTES, RECORDS_PER_ROW = 92, 2
_J, _INDEX, _DIGEST = (0, 4), (4, 12), (28, 92)
#: `memory_attester.SESSION_TAG`, which this place can't import
SESSION_TAG = b"verity-pous/memory-challenger/session/v1\0"
#: a session id's bytes, and the known row ``ctx``'s (`signed_context`)
_ROLE = R.ROLES["memory-challenger-record"].encode()
SESSION_BYTES = 32
CTX_BYTES = 8 + len(_ROLE) + 16 + 16 + SESSION_BYTES


def octets(n: int):
    """A row of n bytes: ``Array<n, Array<8, Value<1>>>``, its bits in byte order (a window's port is whole words)."""
    return Array(n, Array(8, BL.BIT))


def _sig_bytes(S) -> int:
    p = _lms_params(S)
    return 4 + S.N * (1 + p.P + S.H)


def _session_sig(S):
    _lms_params(S)
    return (("key", octets(16 + S.N)), ("m", octets(64)), ("sig", octets(_sig_bytes(S)))), bits(1)


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
    return (("row", octets(S.N)), ("records", octets(RECORD_BYTES * RECORDS_PER_ROW)), ("j", octets(4)),
            ("i", octets(8))), bits(1)


def _round_body(B, S, row, records, j, i):
    X = _Wires
    rec = _group(X.refs(records), 8)[RECORD_BYTES * S.SLOT:RECORD_BYTES * (S.SLOT + 1)]

    def integer(lo, hi):
        return [rec[hi - 1 - t // 8][t % 8] for t in range(8 * (hi - lo))]

    digest = B.call(bind(PousRowDigest, N=S.N), X.coll(bits(8 * S.N), X.refs(row)))
    held = [x for b in rec[_DIGEST[0]:_DIGEST[1]] for x in b] + integer(*_J) + integer(*_INDEX)
    return B.call(bind(PousEqual, N=608), X.coll(bits(608), X.refs(digest) + X.refs(j) + X.refs(i)),
                  X.coll(bits(608), held))


PousRound = CompositeDefinition(
    "PousRound", 1, ("N", "SLOT"), _round_sig, _round_body, conformance="exact-model-tested",
    doc="1 iff record SLOT of the records row (two 92-byte memory_attester.Round.value records) names round j and index i "
        "and holds the N-byte row's sha512/row/v2 digest: R_even at SLOT 0, R_odd at SLOT 1")


def signed_context(I: bytes, session: bytes) -> bytes:
    """The known row ``ctx``: what `lms.memory_attester_message` hashes before the record, ``u64be(len tag) ‖ tag ‖ I ‖
    u64be(40) ‖ u64be(32) ‖ session`` (its context is the session id's field, framed again)."""
    if len(I) != 16 or len(session) != SESSION_BYTES:
        raise ValueError(f"an LMS identifier is 16 bytes and a session id {SESSION_BYTES}")
    sid = SESSION_BYTES.to_bytes(8, "big") + session
    return len(_ROLE).to_bytes(8, "big") + _ROLE + I + len(sid).to_bytes(8, "big") + sid


def _message_sig(S):
    if isinstance(S.HEAD, bool) or not isinstance(S.HEAD, int) or S.HEAD < 1:
        raise ValueError(f"PousSessionMessage's HEAD is the header's bytes, an int >= 1, not {S.HEAD!r}")
    return (("head", bits(8 * S.HEAD)), ("records", bits(512)), ("ctx", bits(8 * CTX_BYTES))), bits(512)


def _hash_bytes(C, data: list[list]) -> list[list]:
    return F._bytes(F._sha512(C, F._words(data + F._const_bytes(pad_tail(len(data))))))


def _byte_order(row) -> list[list]:
    return [list(row[8 * b:8 * b + 8]) for b in range(len(row) // 8)]


def _message_build(C, S, *e):
    head, records, ctx = e
    record = _hash_bytes(C, F._const_bytes(SESSION_TAG + S.HEAD.to_bytes(8, "big")) + _byte_order(head)
                         + _byte_order(records))
    return [[x for b in _hash_bytes(C, _byte_order(ctx) + F._const_bytes((64).to_bytes(8, "big")) + record) for x in b]]


PousSessionMessage = _part(
    "PousSessionMessage", ("HEAD",), _message_sig, _message_build,
    "lms.memory_attester_message's digest SHA-512(ctx || u64be(64) || record), 64 bytes in byte order, of the record "
    "SHA-512(SESSION_TAG || u64be(HEAD) || head || records), the session's signed record over its HEAD-byte header and "
    "its records digest", calls=True)


def records_rows(K: int) -> int:
    """The records rows of a session of K rounds."""
    return -(-K // RECORDS_PER_ROW)


def head_bytes(HEAD: int) -> int:
    """The header row's bytes: the header, then a zero byte when HEAD is odd, since a row is whole u16 words."""
    return HEAD + HEAD % 2


def _digest_statics(S) -> None:
    _check("PousSessionDigest", K=S.K)


def _digest_sig(S):
    _digest_statics(S)
    _message_sig(S)
    row = octets(RECORD_BYTES * RECORDS_PER_ROW)
    rows = tuple((f"r{t}", row) for t in range(records_rows(S.K)))
    return ((("head", octets(head_bytes(S.HEAD))), ("ctx", octets(CTX_BYTES)), *rows),
            Tuple(bits(1), octets(64), *(row for _ in rows)))


def _digest_body(B, S, head, ctx, *rows):
    X = _Wires
    flat = [x for r in rows for x in X.refs(r)]
    recs = [flat[8 * RECORD_BYTES * t:8 * RECORD_BYTES * (t + 1)] for t in range(S.K)]
    stream = B.call(bind(PousRowDigest, N=RECORD_BYTES * S.K),
                    X.coll(bits(8 * RECORD_BYTES * S.K), [x for r in recs for x in r]))
    m = B.call(bind(PousSessionMessage, HEAD=S.HEAD), X.coll(bits(8 * S.HEAD), X.refs(head)[:8 * S.HEAD]), stream,
               X.coll(bits(8 * CTX_BYTES), X.refs(ctx)))
    timed = B.call(bind(PousVerdict, K=S.K, M=0, LIMIT=S.LIMIT),
                   X.coll(Array(S.K, bits(8 * RECORD_BYTES)), [x for r in recs for x in r]),
                   X.coll(Array(S.K, bits(512)), [x for r in recs for x in r[8 * _DIGEST[0]:8 * _DIGEST[1]]]))
    return tuple_of(X.coll(bits(1), X.refs(timed)), X.coll(octets(64), X.refs(m)), *rows)


PousSessionDigest = CompositeDefinition(
    "PousSessionDigest", 1, ("K", "HEAD", "LIMIT"), _digest_sig, _digest_body, conformance="exact-model-tested",
    doc="(ok, m, the records rows): m the digest the session's signature signs, SHA-512(ctx || u64be(64) || record) of the "
        "record SHA-512(SESSION_TAG || u64be(HEAD) || head || the sha512/row/v2 digest of the K records as one stream); ok 1 "
        "iff every record names its round, its receipt is not before its send and receipt - send <= LIMIT "
        "(PousVerdict_v1{K, 0, LIMIT} against its own digests); the records rows as read, for the rounds' units to read "
        "by wire")


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


def session_digest(K: int, HEAD: int, LIMIT: int, head: bytes, ctx: bytes, rows: Sequence[bytes]
                   ) -> tuple[int, bytes, tuple[bytes, ...]]:
    """`PousSessionDigest_v1{K, HEAD, LIMIT}`'s outputs on its rows, ``head`` the header row (`head_bytes`): whether every
    record is on time, `lms.memory_attester_message`'s digest under ``ctx`` of `memory_attester.message`'s record over
    the header and the K records the rows hold, and the rows."""
    row = RECORD_BYTES * RECORDS_PER_ROW
    if (K < 1 or HEAD < 1 or len(head) != head_bytes(HEAD) or len(ctx) != CTX_BYTES
            or [len(r) for r in rows] != [row] * records_rows(K)):
        raise ValueError(f"the rows are a {head_bytes(HEAD)}-byte header row, a {CTX_BYTES}-byte ctx and "
                         f"{records_rows(K)} records rows of {row} bytes")
    head = head[:HEAD]
    recs = [rows[t // RECORDS_PER_ROW][RECORD_BYTES * (t % 2):RECORD_BYTES * (t % 2 + 1)] for t in range(K)]
    stream = sha512_row_v2_digest(b"".join(recs), 8 * RECORD_BYTES * K)
    record = hashlib.sha512(SESSION_TAG + HEAD.to_bytes(8, "big") + head + stream).digest()
    m = hashlib.sha512(ctx + (64).to_bytes(8, "big") + record).digest()
    return verdict_value(K, 0, LIMIT, recs, [r[slice(*_DIGEST)] for r in recs]), m, tuple(rows)
