"""A party's signed period root as a Boolean Definition: what a certification period's proof runs to accept the root the
party's lower key signed (`verity.core.attest`, Lean's `Attest.check`) without revealing the root, the record count or the
signature.

    AttestPeriod_v1{N, W, H, L}
        (I: Array<2, Word>, lower: Value_N, context: Array<L, Byte>, count: Array<64, Value<1>>, root: Digest,
         q: Array<H, Value<1>>, C: Value_N, y: Array<P, Value_N>, path: Array<H, Value_N>) -> Value<1>
        Byte = Array<8, Value<1>>, Digest = Array<8, Word>; N, W, H and the rest as `gates.lms`' LmsVerify_v1

1 iff ``u32(q) || C || y || path`` signs ``lms.period_root_message(I, ..., count, root)`` under the lower key
``I || lower`` (`lms.verify`). ``I || lower`` is the party's lower key, public, as every liveness record shows it, and the
verifier checks the identity's certificate of it in the clear (`Attest.checkKey`), so the proof runs one LMS verification.
``context`` is the period's context as `lms.period_root_message` hashes it before the count (`context`): the deployment's
hash, the party, the attachment id, the link and the period's index, each but the index length-prefixed, L bytes the
verifier lays out from the public statement. ``count`` (u64, LSB first) and ``root`` (the period tree's root) are witness,
as is the signature; they are the values the period's reads share, through the service's registration of them, to open
its records.

**The message** is SHA-512 of ``u64(21) || "verity/period-root/v1" || I || u64(L + 8) || context || u64be(count) ||
u64(64) || root`` (`lms._digest`), 133 + L bytes: three ``Sha512Compress_v1`` calls up to L = 234 (a memory challenger's
party on a device role of 8 bytes and a 32-byte attachment is L = 161), wired in the body over the arguments' bits and
the constant bytes, as `gates.lms` hashes a one-time public key. Its digest is ``LmsVerify_v1{N, W, H}``'s message. Every
digest and N-byte value is SHA-512 words as `gates.lms` lays them out (bytes big-endian, each word's bits LSB first); a
byte of ``context`` is 8 bits LSB first.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.crypto import lms as R

from .lms import DIGEST, IDENT, WORD, Sha512Compress, _bytes, _check, _group, _join, _value, _value_bits, _Wires, _words
from .lms import arguments as lms_arguments
from .lms import verifier
from .sha512 import IV, pad_tail

__all__ = ["AttestPeriod", "arguments", "context", "period"]

BIT = BL.BIT
BYTE = Array(8, BIT)
U64 = Array(64, BIT)
TAG = R.ROLES["period-root"].encode()


def context(deployment: bytes, party: str, attachment: bytes, link: str, period: int) -> bytes:
    """The period's context as ``lms.period_root_message`` hashes it, its count aside: ``context``'s bytes."""
    return R._period_context(deployment, party, attachment, link, period)


def _check_context(L: int) -> None:
    if isinstance(L, bool) or not isinstance(L, int) or L < 1:
        raise ValueError(f"a period's context is L bytes, a positive int, not {L!r}")


def _signature(S):
    p = _check(S)
    _check_context(S.L)
    V = _value(S.N)
    return ((("I", IDENT), ("lower", V), ("context", Array(S.L, BYTE)), ("count", U64), ("root", DIGEST),
             ("q", Array(S.H, BIT)), ("C", V), ("y", Array(p.P, V)), ("path", Array(S.H, V))), BIT)


def _body(B, S, I, lower, ctx, count, root, q, Cv, y, path):
    X = _Wires(B)
    const = lambda data: [X.const(b, 8) for b in data]  # noqa: E731
    data = (const(len(TAG).to_bytes(8, "big") + TAG) + _bytes(_group(X.refs(I), 64))
            + const((S.L + 8).to_bytes(8, "big")) + _group(X.refs(ctx), 8) + _bytes([X.refs(count)])
            + const((64).to_bytes(8, "big")) + _bytes(_group(X.refs(root), 64)))
    data += const(pad_tail(len(data)))
    words = _words(data)
    cv = X.coll(DIGEST, [b for v in IV for b in X.const(v, 64)])
    for blk in range(0, len(words), 16):
        cv = B.call(Sha512Compress, cv, X.coll(Array(16, WORD), _join(*words[blk:blk + 16])))
    return B.call(verifier(S.N, S.W, S.H), I, lower, cv, q, Cv, y, path)


AttestPeriod = CompositeDefinition(
    "AttestPeriod", 1, ("N", "W", "H", "L"), _signature, _body, conformance="exact-model-tested",
    doc="1 iff u32(q) || C || y || path is an LMS signature (LmsVerify_v1{N, W, H}) under the party's lower key I || lower of "
        "lms.period_root_message: SHA-512 of the tag verity/period-root/v1, I, the period's context (L bytes) with u64be(count), "
        "and root; sound under the claim euf-cma/lms-sha512-n64 at N = 64")


def period(N: int, W: int, H: int, L: int):
    return bind(AttestPeriod, N=N, W=W, H=H, L=L)


def arguments(params: R.Params, lower: bytes, context_: bytes, count: int, root: bytes,
              sig: R.Signature) -> tuple[list[int], ...]:
    """``AttestPeriod_v1{N, W, H, len(context_)}``'s nine arguments, each a flat list of bits in leaf order."""
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count < 2 ** 64 or len(root) != 64:
        raise ValueError("a period's count is a u64 and its root 64 bytes")
    _check_context(len(context_))
    I, top, _, q, C, y, path = lms_arguments(params, lower, bytes(64), sig)
    return (I, top, [(b >> j) & 1 for b in context_ for j in range(8)], [(count >> j) & 1 for j in range(64)],
            _value_bits(root), q, C, y, path)
