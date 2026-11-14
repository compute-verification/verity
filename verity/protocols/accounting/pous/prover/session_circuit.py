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
the digest the memory challenger's signature signs (`memory_challenger.LmsSigner`); the registered row ``m`` is its 64
bytes in byte order, which is how the unit that computes ``m`` must output it.  The bytes are rewired into
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
from verity.commitments.merkle import _LEAF, _PAD, _pad, _uint, frame_hash, levels, merkle_depth
from verity.commitments.rowleaf import sha512_row_v2_digest

from .answer_circuit import PousEqual, PousRowDigest
from .circuit import _check, bits
from .verdict_circuit import PousVerdict, verdict_value

__all__ = ["CTX_BYTES", "RECORD_BYTES", "RECORDS_PER_ROW", "RECORD_SCHEMA", "SESSION_BYTES", "SESSION_TAG", "MAX_ROUNDS", "PousRecordLeaf",
           "PousRecordPad", "PousRound", "PousSessionDigest", "PousSessionMessage", "PousSessionSig", "records_rows",
           "head_bytes", "round_ok", "session_digest", "session_sig_ok", "signed_context"]

#: a round record's bytes (`memory_challenger.Round.value`), and the records a registered row holds
RECORD_BYTES, RECORDS_PER_ROW = 92, 2
_J, _INDEX, _DIGEST = (0, 4), (4, 12), (28, 92)
#: `memory_challenger.SESSION_TAG` and `memory_challenger.RECORD_SCHEMA`, which this place can't import
SESSION_TAG = b"verity-pous/memory-challenger/session/v1\0"
RECORD_SCHEMA = "pous/round-record/v2"
#: the most rounds the signed digest's unit takes: every rank, pad rank and node index is then one byte of its uint
MAX_ROUNDS = 256
#: a session id's bytes, and the known row ``ctx``'s (`signed_context`)
_ROLE = R.ROLES["memory-challenger-record"].encode()
SESSION_BYTES = 32
CTX_BYTES = 8 + len(_ROLE) + 16 + 16 + SESSION_BYTES


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


def _record_bytes(value) -> list[list]:
    return [list(value[8 * b:8 * b + 8]) for b in range(RECORD_BYTES)]


PousRecordLeaf = _part(
    "PousRecordLeaf", (), lambda S: ((("domain", F.DIGEST), ("value", bits(8 * RECORD_BYTES)), ("x", bits(8))), F.DIGEST),
    lambda C, S, *e: F._sha512(C, F._message(_LEAF, [F._bytes(e[0:8]), [list(e[9])], [list(e[9])],
                                                    F._const_bytes(RECORD_SCHEMA.encode()), _record_bytes(e[8])])),
    "the frame-v3-sha512 leaf of a 92-byte round record at rank = position = x under the domain id domain and "
    "pous/round-record/v2; x one byte, LSB first", calls=True)

PousRecordPad = _part(
    "PousRecordPad", (), lambda S: ((("domain", F.DIGEST), ("x", bits(8))), F.DIGEST),
    lambda C, S, *e: F._sha512(C, F._message(_PAD, [F._bytes(e[0:8]), [list(e[8])]])),
    "the frame-v3-sha512 pad at rank x under the domain id domain; x one byte, LSB first", calls=True)


def signed_context(I: bytes, session: bytes) -> bytes:
    """The known row ``ctx``: what `lms.memory_challenger_message` hashes before the record, ``u64be(len tag) ‖ tag ‖ I ‖
    u64be(40) ‖ u64be(32) ‖ session`` (its context is the session id's field, framed again)."""
    if len(I) != 16 or len(session) != SESSION_BYTES:
        raise ValueError(f"an LMS identifier is 16 bytes and a session id {SESSION_BYTES}")
    sid = SESSION_BYTES.to_bytes(8, "big") + session
    return len(_ROLE).to_bytes(8, "big") + _ROLE + I + len(sid).to_bytes(8, "big") + sid


def _message_sig(S):
    if isinstance(S.HEAD, bool) or not isinstance(S.HEAD, int) or S.HEAD < 1:
        raise ValueError(f"PousSessionMessage's HEAD is the header's bytes, an int >= 1, not {S.HEAD!r}")
    return (("head", bits(8 * S.HEAD)), ("root", F.DIGEST), ("ctx", bits(8 * CTX_BYTES))), bits(512)


def _hash_bytes(C, data: list[list]) -> list[list]:
    return F._bytes(F._sha512(C, F._words(data + F._const_bytes(pad_tail(len(data))))))


def _byte_order(row) -> list[list]:
    return [list(row[8 * b:8 * b + 8]) for b in range(len(row) // 8)]


def _message_build(C, S, *e):
    head, root, ctx = e[0], e[1:9], e[9]
    record = _hash_bytes(C, F._const_bytes(SESSION_TAG + S.HEAD.to_bytes(8, "big")) + _byte_order(head) + F._bytes(root))
    return [[x for b in _hash_bytes(C, _byte_order(ctx) + F._const_bytes((64).to_bytes(8, "big")) + record) for x in b]]


PousSessionMessage = _part(
    "PousSessionMessage", ("HEAD",), _message_sig, _message_build,
    "lms.memory_challenger_message's digest SHA-512(ctx || u64be(64) || record), 64 bytes in byte order, of the record "
    "SHA-512(SESSION_TAG || u64be(HEAD) || head || root), the session's signed record over its HEAD-byte header and its "
    "records root", calls=True)


def records_rows(K: int) -> int:
    """The records rows of a session of K rounds."""
    return -(-K // RECORDS_PER_ROW)


def head_bytes(HEAD: int) -> int:
    """The header row's bytes: the header, then a zero byte when HEAD is odd, since a row is whole u16 words."""
    return HEAD + HEAD % 2


def _digest_statics(S) -> None:
    if isinstance(S.K, bool) or not isinstance(S.K, int) or not 1 <= S.K <= MAX_ROUNDS:
        raise ValueError(f"PousSessionDigest's K is an int in 1 ... {MAX_ROUNDS}, not {S.K!r}")


def _digest_sig(S):
    _digest_statics(S)
    _message_sig(S)
    row = bits(8 * RECORD_BYTES * RECORDS_PER_ROW)
    rows = tuple((f"r{t}", row) for t in range(records_rows(S.K)))
    return ((("head", bits(8 * head_bytes(S.HEAD))), ("domain", bits(512)), ("ctx", bits(8 * CTX_BYTES)), *rows),
            Tuple(bits(1), bits(512), *(row for _ in rows)))


def _digest_body(B, S, head, domain, ctx, *rows):
    X = _Wires
    consts: dict[int, object] = {}

    def const(b: int):
        if b not in consts:
            consts[b] = B.call(BL.constant(b)).refs[0]
        return consts[b]

    def byte(v: int):
        return X.coll(bits(8), [const((v >> t) & 1) for t in range(8)])

    recs = []
    for t in range(S.K):
        r = _group(X.refs(rows[t // RECORDS_PER_ROW]), 8)
        recs.append([x for b in r[RECORD_BYTES * (t % 2):RECORD_BYTES * (t % 2 + 1)] for x in b])
    dom = X.coll(F.DIGEST, [x for w in F._words(_group(X.refs(domain), 8)) for x in w])
    level = [B.call(PousRecordLeaf, dom, X.coll(bits(8 * RECORD_BYTES), r), byte(t)) for t, r in enumerate(recs)]
    level += [B.call(PousRecordPad, dom, byte(t)) for t in range(S.K, 1 << merkle_depth(S.K))]
    node, depth = bind(F.FrameV3Node, B=1), 0
    while len(level) > 1:
        d = X.coll(BL.BIT, [const(0)])
        level = [B.call(node, dom, byte(depth), byte(j), d, level[2 * j], level[2 * j + 1]) for j in range(len(level) // 2)]
        depth += 1
    m = B.call(bind(PousSessionMessage, HEAD=S.HEAD), X.coll(bits(8 * S.HEAD), X.refs(head)[:8 * S.HEAD]), level[0], ctx)
    timed = B.call(bind(PousVerdict, K=S.K, M=0, LIMIT=S.LIMIT),
                   X.coll(Array(S.K, bits(8 * RECORD_BYTES)), [x for r in recs for x in r]),
                   X.coll(Array(S.K, bits(512)), [x for r in recs for x in r[8 * _DIGEST[0]:8 * _DIGEST[1]]]))
    return tuple_of(X.coll(bits(1), X.refs(timed)), m, *rows)


PousSessionDigest = CompositeDefinition(
    "PousSessionDigest", 1, ("K", "HEAD", "LIMIT"), _digest_sig, _digest_body, conformance="exact-model-tested",
    doc="(ok, m, the records rows): m the digest the session's signature signs, SHA-512(ctx || u64be(64) || record) of the "
        "record SHA-512(SESSION_TAG || u64be(HEAD) || head || the frame-v3-sha512 root of the K records under the domain id "
        "domain); ok 1 iff every record names its round, its receipt is not before its send and receipt - send <= LIMIT "
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


def session_digest(K: int, HEAD: int, LIMIT: int, head: bytes, domain: bytes, ctx: bytes, rows: Sequence[bytes]
                   ) -> tuple[int, bytes, tuple[bytes, ...]]:
    """`PousSessionDigest_v1{K, HEAD, LIMIT}`'s outputs on its rows, ``head`` the header row (`head_bytes`): whether every
    record is on time, `lms.memory_challenger_message`'s digest under ``ctx`` of `memory_challenger.message`'s record over
    the header and the K records the rows hold under the domain id ``domain``, and the rows."""
    row = RECORD_BYTES * RECORDS_PER_ROW
    if (not 1 <= K <= MAX_ROUNDS or HEAD < 1 or len(head) != head_bytes(HEAD) or len(domain) != 64 or len(ctx) != CTX_BYTES
            or [len(r) for r in rows] != [row] * records_rows(K)):
        raise ValueError(f"the rows are a {head_bytes(HEAD)}-byte header row, a 64-byte domain id, a {CTX_BYTES}-byte ctx "
                         f"and {records_rows(K)} records rows of {row} bytes")
    head = head[:HEAD]
    recs = [rows[t // RECORDS_PER_ROW][RECORD_BYTES * (t % 2):RECORD_BYTES * (t % 2 + 1)] for t in range(K)]
    leaves = [frame_hash(_LEAF, domain, _uint(t), _uint(t), RECORD_SCHEMA.encode(), r) for t, r in enumerate(recs)]
    leaves += [_pad(domain, "sha512", t) for t in range(K, 1 << merkle_depth(K))]
    root = levels(domain, leaves)[-1][0]
    record = hashlib.sha512(SESSION_TAG + HEAD.to_bytes(8, "big") + head + root).digest()
    m = hashlib.sha512(ctx + (64).to_bytes(8, "big") + record).digest()
    return verdict_value(K, 0, LIMIT, recs, [r[slice(*_DIGEST)] for r in recs]), m, tuple(rows)
