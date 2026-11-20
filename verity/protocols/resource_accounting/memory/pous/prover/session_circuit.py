"""PoUS's session checked inside proofs: the unit that computes the digest the memory attester signed and compares it
with the one the auditor verified, and the unit that accepts one round's answer, so that the auditor never reads the
record the memory attester signed.

    PousSessionDigest_v2{K, HEAD, LIMIT} (head: W<(HEAD + HEAD mod 2)/2>, ctx: W<53>, salt: W<32>, m: W<32>,
                                          r0 ... rT−1: W<92>) -> (ok: VERDICT, r0 ... rT−1: W<92>), T = ⌈K/2⌉
    PousRound_v3{N, SLOT}                (row: W<N/2>, ok: W<1>, records: W<92>, j: U32, ihi: U32, ilo: U32) -> VERDICT
    PousSessionSig_v1{N, W, H}           (key: B<16 + N>, m: B<64>, sig: B<S>) -> VERDICT
                                         S = 4 + N(1 + P + H), the signature's bytes (`lms.Params.signature_bytes`)

W<n> is n 16-bit words, ``Array<n, Value<16>>``, each word two bytes little-endian, as a registered table's row is
(`interface.py`'s leaf widths): a byte string's port is its bytes as whole words, split to bits inside the unit
(`_bytes`), word w being byte 2w then byte 2w + 1, each byte's bits least significant first, so a row's bits are in byte
order (a ``sha512/row/v2`` row's order, `verity.commitments.rowleaf.bit_row`). B<n> is n bytes,
``Array<n, Array<8, Value<1>>>`` (`octets`), in the same bit order. U32 is ``Value<32>``, its row the integer's four
bytes little-endian. VERDICT is one u16 word, 1 or 0, since every field a block returns is whole words. Flock stages a
Definition as one ``hm96-sha512/row/v2`` row per parameter and one per output
(`verity.commitments.periods.interface.unit_rows`), whole u16 words of one element width, so each parameter here is one
row, holding the bytes as the memory attester writes them.

**What is public.** The message m, ``lms.memory_attester_message(I, session, record)`` of the record
``SHA-512(SESSION_TAG ‖ salt ‖ u64be(HEAD) ‖ head ‖ records digest)``: a hiding commitment to the records, salted with
the memory attester's 64-byte draw; and the signature over m, whose leaf number q shows how many signatures the memory
attester has made under its key. The salt and the records are witness, committed rows the prover holds. The record's
salt is the memory attester's own draw, `RECORD_SALT_BYTES` (64) bytes from its coins source as the session opens
(`challenger_party.Challenger._coins`, `wipe_challenger.MemoryAttesterParty.start`; `os.urandom` in
`MemoryAttester.__init__` when none is given; in Lean the device's own coins, `Session.Params.salt`), outside the
session's commitment and before its first index is revealed, and read once, when `close()` signs; it makes the signed
digest m a hiding commitment to the guessable header and records, and nothing in soundness rests on its being random or
fresh: a salt the prover could choose would still leave the signature binding the same records.

**The signed digest's unit.** `PousSessionDigest_v2{K, HEAD, LIMIT}` reads the signed header (``head``, HEAD bytes and a
zero byte when HEAD is odd), the signed context (``ctx``, `signed_context`: what `lms.memory_attester_message` hashes
before the record), the record's salt (``salt``, a committed row), the message m the auditor verified the signature over
(``m``, ``pous.session.message/v1``, which the verifier supplies) and the K records in their rows. It recomputes m from
the salt and the records rows the rounds read, ``SHA-512(ctx ‖ u64be(64) ‖ record)`` of the record
``SHA-512(SESSION_TAG ‖ salt ‖ u64be(HEAD) ‖ head ‖ records)``, records the ``sha512/row/v2`` digest of the K records as
one stream (`memory_attester.records_digest`, ``PousRowDigest_v1{92K}``). It outputs ok, 1 iff the m it recomputes is the
m supplied (``PousEqual_v1{512}``) and every record names its round, its receipt is not before its send and
receipt − send ≤ LIMIT (``PousVerdict_v1{K, 0, LIMIT}`` against the records' own digests); and the records rows as read,
which round j's unit reads (row ⌊j/2⌋) by wire, so the two kinds of proof are linked only by committed rows.

**The round's unit.** The records are registered two a row, ⌈k/2⌉ rows of 184 bytes (92 u16 words): record j is bytes
92(j mod 2) to 92(j mod 2) + 91 of row ⌊j/2⌋, the last row zero past the last record when k is odd.  A record is
`memory_attester.Round.value`'s 92 bytes, ``u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)``, integers
big-endian, ``H(block)`` the answer's ``sha512/row/v2`` digest.  `PousRound_v3{N, SLOT}` reads the store's N-byte row
``row`` as N/2 words (the row the store table's opening at the round's index returns, `gates.embedding`'s ``h``),
``ok`` the opening's verdict word (``EmbeddingRead_v3``'s ``ok``), the records row ``records`` holding record j at slot
SLOT = j mod 2, and ``j``, ``ihi`` and ``ilo``, the three big-endian 32-bit halves of round j's coin value
``u32be(j) ‖ u64be(i)`` (``pous.session.round_coin/v1`` under the layout ``halves``: the round and the index the memory
attester drew for it), whose ``ilo`` the store's opening reads as its key.
It is 1 iff record j's first 12 bytes are that value and its ``H(block)`` is ``row``'s ``sha512/row/v2`` digest
(``PousRowDigest_v1{N}``), the opening's verdict is 1 and ``ihi`` is 0: one ``PousEqual_v1{608}`` over the digest and
the halves rewired to the record's byte order against the record's, ANDed with ``PousEqual_v1{16}`` of ``ok`` against 1
and ``PousEqual_v1{32}`` of ``ihi`` against 0.  An opening's verdict is 0 at a key past the store's blocks or on a row
that doesn't open, so the round refuses both; and the opening reads ``ilo`` alone as its key, so ``ihi`` held at 0 makes
the key the record's whole index.  These are the per-round checks of
`memory_attester.verdict` but the time: the record names round j, its index is round j's coin, and its digest is that of
the block at the index.  R_even and R_odd are ``PousRound_v3{2056, 0}`` and ``PousRound_v3{2056, 1}``, two Programs.

**The signature's unit.** `PousSessionSig_v1{N, W, H}` is 1 iff ``sig`` is an LMS signature (`lms.Signature.encode`:
``u32be(q) ‖ C ‖ y ‖ path``) of the 64-byte message ``m`` under the public key ``key`` (``I ‖ root``, `lms.public_key`)
with q below 2^H, as `lms.verify` decides: ``LmsVerify_v1{N, W, H}`` over q's low H bits, and q's top 32 − H bits zero
(``PousEqual_v1{32 − H}`` against zeros, ANDed in).  A q at or past 2^H is no leaf of the key's tree, and `lms.verify`
refuses it; `LmsVerify_v1` reads q's low H bits only.  The bytes are rewired into `gates.lms`' words (each word's bytes
big-endian, its bits least significant first) without a gate.  At the deployed scheme, ``lms-sha512/n64-w1-h20/v0``
(`lms.DEPLOYED`), it is ``PousSessionSig_v1{64,1,20}``: rows of 640, 512 and 278,048 bits and its verdict word, 12 ANDs
beside the verifier's.  It is no unit of the session: the auditor verifies the signature over m in the clear under the
pinned key (`session_proofs.clear_reasons`), and the digest's unit holds m to the records.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import part as _part
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import tuple_of
from verity.circuits.population import BitsWord, WordBits
from verity.circuits.types import Array, Tuple, Value
from verity.commitments import session_coins as SC
from verity.commitments.crypto import lms as R
from verity.commitments.gates import frame_v3 as F
from verity.commitments.gates.lms import DIGEST, IDENT, _check as _lms_params, _group, _value, _words, _Wires, verifier
from verity.commitments.gates.sha512 import pad_tail
from verity.commitments.rowleaf import sha512_row_v2_digest

from .answer_circuit import PousEqual, PousRowDigest
from .circuit import _check, bits
from .verdict_circuit import PousVerdict, verdict_value

__all__ = ["CTX_BYTES", "RECORD_BYTES", "RECORDS_PER_ROW", "SALT_BYTES", "SESSION_BYTES", "SESSION_TAG", "PousRound",
           "PousSessionDigest", "PousSessionMessage", "PousSessionSig", "head_bytes", "octets", "records_rows", "round_ok",
           "session_digest", "session_message", "session_sig_ok", "signed_context", "words"]

#: a round record's bytes (`memory_attester.Round.value`), and the records a registered row holds
RECORD_BYTES, RECORDS_PER_ROW = 92, 2
_J, _INDEX, _DIGEST = (0, 4), (4, 12), (28, 92)
#: the round's coin value's big-endian 32-bit halves, its ports in order (``pous.session.round_coin/v1``'s ``halves``)
_HALVES = ("j", "ihi", "ilo")
#: `memory_attester.SESSION_TAG` and `memory_attester.RECORD_SALT_BYTES`, which this place can't import
SESSION_TAG = b"verity-pous/memory-challenger/session/v2\0"
SALT_BYTES = 64
#: a session id's bytes, and the known row ``ctx``'s (`signed_context`)
_ROLE = R.ROLES["memory-challenger-record"].encode()
SESSION_BYTES = 32
CTX_BYTES = 8 + len(_ROLE) + 16 + 16 + SESSION_BYTES


def octets(n: int):
    """A row of n bytes: ``Array<n, Array<8, Value<1>>>``, its bits in byte order (a window's port is whole words)."""
    return Array(n, Array(8, BL.BIT))


def words(n: int):
    """A row of n 16-bit words, ``Array<n, Value<16>>``: 2n bytes, each word's two bytes little-endian."""
    return Array(n, Value(16))


def _bytes(B, port) -> list:
    """A W<n> port's bits in byte order: each word split to its 16 bits, least significant first."""
    return _Wires.refs(B.batch(bind(WordBits, W=16), port, axes=(0,)))


#: a unit's verdict: one u16 word, 1 or 0, since every field a block returns is whole words
VERDICT = Value(16)


def _verdict(B, ok):
    """The verdict word of the bit ``ok``: ok in its low bit, its other bits 0."""
    zero = B.call(BL.constant(0)).refs[0]
    return B.call(bind(BitsWord, W=16), _Wires.coll(bits(16), [ok] + [zero] * 15))


def _sig_bytes(S) -> int:
    p = _lms_params(S)
    return 4 + S.N * (1 + p.P + S.H)


def _session_sig(S):
    _lms_params(S)
    return (("key", octets(16 + S.N)), ("m", octets(64)), ("sig", octets(_sig_bytes(S)))), VERDICT


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
    return _verdict(B, X.refs(B.call(BL.AND, BL.operands(ok, top[0])))[0])


PousSessionSig = CompositeDefinition(
    "PousSessionSig", 1, ("N", "W", "H"), _session_sig, _session_body, conformance="exact-model-tested",
    doc="the verdict word, 1 iff sig, u32be(q) || C || y || path, is an LMS signature (LmsVerify_v1{N, W, H}) of the 64-byte message m under "
        "the public key I || root, with q below 2^H; every input a byte string in byte order; sound under the claim "
        "euf-cma/lms-sha512-n64 at N = 64")


def _round_sig(S):
    _check("PousRound", N=S.N)
    if S.N % 2 or S.SLOT not in range(RECORDS_PER_ROW):
        raise ValueError(f"PousRound's N is a row's bytes, whole 16-bit words, and its SLOT a record's place in its row, "
                         f"below {RECORDS_PER_ROW}, not N={S.N!r}, SLOT={S.SLOT!r}")
    return (("row", Array(S.N // 2, Value(16))), ("ok", Value(16)), ("records", words(RECORD_BYTES * RECORDS_PER_ROW // 2)),
            *((h, Value(32)) for h in _HALVES)), VERDICT


def _round_body(B, S, row, ok, records, *halves):
    X = _Wires
    rec = _group(_bytes(B, records), 8)[RECORD_BYTES * S.SLOT:RECORD_BYTES * (S.SLOT + 1)]
    words = B.batch(bind(WordBits, W=16), row, axes=(0,))
    digest = B.call(bind(PousRowDigest, N=S.N), X.coll(bits(8 * S.N), X.refs(words)))
    held = [x for b in rec[_DIGEST[0]:_DIGEST[1]] + rec[_J[0]:_INDEX[1]] for x in b]
    hb = [X.refs(B.call(bind(WordBits, W=32), h)) for h in halves]
    coin = [x for h in hb for b in reversed(range(4)) for x in h[8 * b:8 * (b + 1)]]
    zero, one = (B.call(BL.constant(b)).refs[0] for b in (0, 1))
    same = B.call(bind(PousEqual, N=608), X.coll(bits(608), X.refs(digest) + coin), X.coll(bits(608), held))
    opened = B.call(bind(PousEqual, N=16), X.coll(bits(16), X.refs(B.call(bind(WordBits, W=16), ok))),
                    X.coll(bits(16), [one] + [zero] * 15))
    low = B.call(bind(PousEqual, N=32), X.coll(bits(32), hb[_HALVES.index("ihi")]), X.coll(bits(32), [zero] * 32))
    both = B.call(BL.AND, BL.operands(same[0], opened[0]))
    return _verdict(B, X.refs(B.call(BL.AND, BL.operands(both, low[0])))[0])


PousRound = CompositeDefinition(
    "PousRound", 3, ("N", "SLOT"), _round_sig, _round_body, conformance="exact-model-tested",
    doc="the verdict word, 1 iff record SLOT of the records row (92 words: two 92-byte memory_attester.Round.value records) begins with u32be(j) || "
        "u64be(i), read as its three big-endian 32-bit halves j, ihi, ilo, and holds the sha512/row/v2 digest of the N-byte row, its N/2 words little-endian, "
        "the row's opening verdict word ok is 1 and ihi is 0: R_even at SLOT 0, R_odd at SLOT 1")


def signed_context(I: bytes, session: bytes) -> bytes:
    """The known row ``ctx``: what `lms.memory_attester_message` hashes before the record, ``u64be(len tag) ‖ tag ‖ I ‖
    u64be(40) ‖ u64be(32) ‖ session`` (its context is the session id's field, framed again)."""
    if len(I) != 16 or len(session) != SESSION_BYTES:
        raise ValueError(f"an LMS identifier is 16 bytes and a session id {SESSION_BYTES}")
    return SC.signed_context(I, session)


def _message_sig(S):
    if isinstance(S.HEAD, bool) or not isinstance(S.HEAD, int) or S.HEAD < 1:
        raise ValueError(f"PousSessionMessage's HEAD is the header's bytes, an int >= 1, not {S.HEAD!r}")
    return (("head", bits(8 * S.HEAD)), ("salt", bits(8 * SALT_BYTES)), ("records", bits(512)),
            ("ctx", bits(8 * CTX_BYTES))), bits(512)


def _hash_bytes(C, data: list[list]) -> list[list]:
    return F._bytes(F._sha512(C, F._words(data + F._const_bytes(pad_tail(len(data))))))


def _byte_order(row) -> list[list]:
    return [list(row[8 * b:8 * b + 8]) for b in range(len(row) // 8)]


def _message_build(C, S, *e):
    head, salt, records, ctx = e
    record = _hash_bytes(C, F._const_bytes(SESSION_TAG) + _byte_order(salt) + F._const_bytes(S.HEAD.to_bytes(8, "big"))
                         + _byte_order(head) + _byte_order(records))
    return [[x for b in _hash_bytes(C, _byte_order(ctx) + F._const_bytes((64).to_bytes(8, "big")) + record) for x in b]]


PousSessionMessage = _part(
    "PousSessionMessage", ("HEAD",), _message_sig, _message_build,
    "lms.memory_attester_message's digest SHA-512(ctx || u64be(64) || record), 64 bytes in byte order, of the record "
    "SHA-512(SESSION_TAG || salt || u64be(HEAD) || head || records), the session's signed record over its 64-byte salt, "
    "its HEAD-byte header and its records digest", version=2, calls=True)


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
    row = words(RECORD_BYTES * RECORDS_PER_ROW // 2)
    rows = tuple((f"r{t}", row) for t in range(records_rows(S.K)))
    return ((("head", words(head_bytes(S.HEAD) // 2)), ("ctx", words(CTX_BYTES // 2)), ("salt", words(SALT_BYTES // 2)),
             ("m", words(32)), *rows), Tuple(VERDICT, *(row for _ in rows)))


def _digest_body(B, S, head, ctx, salt, m, *rows):
    X = _Wires
    flat = [x for r in rows for x in _bytes(B, r)]
    recs = [flat[8 * RECORD_BYTES * t:8 * RECORD_BYTES * (t + 1)] for t in range(S.K)]
    stream = B.call(bind(PousRowDigest, N=RECORD_BYTES * S.K),
                    X.coll(bits(8 * RECORD_BYTES * S.K), [x for r in recs for x in r]))
    mine = B.call(bind(PousSessionMessage, HEAD=S.HEAD), X.coll(bits(8 * S.HEAD), _bytes(B, head)[:8 * S.HEAD]),
                  X.coll(bits(8 * SALT_BYTES), _bytes(B, salt)), stream, X.coll(bits(8 * CTX_BYTES), _bytes(B, ctx)))
    same = B.call(bind(PousEqual, N=512), X.coll(bits(512), X.refs(mine)), X.coll(bits(512), _bytes(B, m)))
    timed = B.call(bind(PousVerdict, K=S.K, M=0, LIMIT=S.LIMIT),
                   X.coll(Array(S.K, bits(8 * RECORD_BYTES)), [x for r in recs for x in r]),
                   X.coll(Array(S.K, bits(512)), [x for r in recs for x in r[8 * _DIGEST[0]:8 * _DIGEST[1]]]))
    return tuple_of(_verdict(B, X.refs(B.call(BL.AND, BL.operands(timed, same[0])))[0]), *rows)


PousSessionDigest = CompositeDefinition(
    "PousSessionDigest", 2, ("K", "HEAD", "LIMIT"), _digest_sig, _digest_body, conformance="exact-model-tested",
    doc="(ok, the records rows): ok a verdict word, 1 iff m is the digest the session's signature signs, "
        "SHA-512(ctx || u64be(64) || record) of the record SHA-512(SESSION_TAG || salt || u64be(HEAD) || head || the "
        "sha512/row/v2 digest of the K records as one stream), recomputed from the committed salt and records rows "
        "(PousEqual_v1{512}), and every record names its round, its receipt is not before its send and receipt - send "
        "<= LIMIT (PousVerdict_v1{K, 0, LIMIT} against its own digests); the records rows as read, for the rounds' units "
        "to read by wire; every port whole 16-bit words")


def session_sig_ok(N: int, W: int, H: int, key: bytes, m: bytes, sig: bytes) -> int:
    """`PousSessionSig_v1{N, W, H}`'s output on its rows' bytes: `lms.verify` of the decoded signature."""
    p = R.Params(N, W, H, toy=N == R.TOY_N)
    if len(key) != 16 + N or len(m) != 64 or len(sig) != p.signature_bytes():
        raise ValueError(f"the rows are a {16 + N}-byte key, a 64-byte message and a {p.signature_bytes()}-byte signature")
    return int(R.verify(p, key, m, R.Signature.decode(p, sig)))


def round_ok(N: int, SLOT: int, row: bytes, records: bytes, coin: bytes, ok: int = 1) -> int:
    """`PousRound_v3{N, SLOT}`'s output on its rows (``row`` as its N bytes, ``ok`` its opening's verdict word, ``coin``
    the round's 12-byte coin value whose halves its three last ports read): record SLOT begins with ``coin`` and holds
    row's digest, ``ok`` is 1 and the coin's ``ihi`` half is 0."""
    if (len(row) != N or len(records) != RECORD_BYTES * RECORDS_PER_ROW or SLOT not in range(RECORDS_PER_ROW)
            or len(coin) != _INDEX[1]):
        raise ValueError(f"the rows are an {N}-byte row, a {RECORD_BYTES * RECORDS_PER_ROW}-byte records row and a "
                         f"{_INDEX[1]}-byte coin value")
    rec = records[RECORD_BYTES * SLOT:RECORD_BYTES * (SLOT + 1)]
    return int(rec[:_INDEX[1]] == coin and rec[slice(*_DIGEST)] == sha512_row_v2_digest(row, 8 * N) and ok == 1
               and coin[4:8] == bytes(4))


def session_message(K: int, HEAD: int, head: bytes, ctx: bytes, salt: bytes, rows: Sequence[bytes]) -> bytes:
    """`lms.memory_attester_message`'s digest under ``ctx`` of `memory_attester.message`'s record over ``salt``, the
    header row ``head`` (`head_bytes`) and the K records the rows hold: the m `PousSessionDigest_v2` recomputes."""
    recs = [rows[t // RECORDS_PER_ROW][RECORD_BYTES * (t % 2):RECORD_BYTES * (t % 2 + 1)] for t in range(K)]
    stream = sha512_row_v2_digest(b"".join(recs), 8 * RECORD_BYTES * K)
    record = hashlib.sha512(SESSION_TAG + salt + HEAD.to_bytes(8, "big") + head[:HEAD] + stream).digest()
    return hashlib.sha512(ctx + (64).to_bytes(8, "big") + record).digest()


def session_digest(K: int, HEAD: int, LIMIT: int, head: bytes, ctx: bytes, salt: bytes, m: bytes, rows: Sequence[bytes]
                   ) -> tuple[int, tuple[bytes, ...]]:
    """`PousSessionDigest_v2{K, HEAD, LIMIT}`'s outputs on its rows' bytes, ``head`` the header row (`head_bytes`):
    whether ``m`` is the `session_message` of the salt, the header and the records and every record is on time; and the
    rows."""
    row = RECORD_BYTES * RECORDS_PER_ROW
    if (K < 1 or HEAD < 1 or len(head) != head_bytes(HEAD) or len(ctx) != CTX_BYTES or len(salt) != SALT_BYTES
            or len(m) != 64 or [len(r) for r in rows] != [row] * records_rows(K)):
        raise ValueError(f"the rows are a {head_bytes(HEAD)}-byte header row, a {CTX_BYTES}-byte ctx, a {SALT_BYTES}-byte "
                         f"salt, a 64-byte m and {records_rows(K)} records rows of {row} bytes")
    recs = [rows[t // RECORDS_PER_ROW][RECORD_BYTES * (t % 2):RECORD_BYTES * (t % 2 + 1)] for t in range(K)]
    timed = verdict_value(K, 0, LIMIT, recs, [r[slice(*_DIGEST)] for r in recs])
    return int(timed == 1 and session_message(K, HEAD, head, ctx, salt, rows) == m), tuple(rows)
