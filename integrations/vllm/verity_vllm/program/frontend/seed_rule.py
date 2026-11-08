"""The seed rule a request Program checks, which the verifier requires to be 1: the seed its sampler read is its certified body's
``seed`` where the body names one, else `served_seed.derived` of the window's verifier coin and the request's id, in gates.

    ServedSeed_v1{COIN, RID}  (seeded: bit, seed: Word, coin: Array<COIN, u8>, user: Array<RID, u8>) -> Word
    SeedCheck_v1{COIN, RID}   (served: Word, seeded, seed, coin, user) -> bit
    SeedSelect_v1             (seeded: bit, seed: Word, digest: Word) -> Word: seed if seeded, else digest's bits 1-63
    SeedEqual_v1              (a: Word, b: Word) -> bit: 1 iff a = b

(Word is ``Array<64, Value<1>>``, LSB first; a byte's bits are LSB first.) `ServedSeed_v1` is the rule: ``seed`` where
``seeded``, else the first 63 bits of SHA-512 over `served_seed.DERIVED`, u32be(COIN), the coin, u32be(RID) and the id
(`served_seed.derived`): Calls of `lms.Sha512Compress_v1` on the message's blocks (one where the message and its padding fit
128 bytes, else two), then `SeedSelect_v1` on the digest's first word. COIN and RID are the coin's and the id's lengths in
bytes, so the padding is constant. ``seeded``, ``seed`` and ``user`` are ingress schema v1's
(`boolean_io_shell.IngressSampled_v1`), certified, and the coin is the window's. `SeedCheck_v1` is 1 exactly when ``served``,
the seed the Program's sampler read, is the rule's (`SeedEqual_v1`): any other seed fails it.

No request Program returns the check yet, so a stochastic request's seed is tied to its certified body by the compiler
(`reads.seed_of`), not by the circuit. `reference` and `check_reference` are the two functions. Imports without torch.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import NOT, ZERO, and_reduce, mux
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, Explicit
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates import sha512 as H
from verity.ml.commitments.gates.lms import Sha512Compress
from verity_vllm.engine import served_seed

__all__ = ["SeedCheck", "SeedEqual", "SeedSelect", "ServedSeed", "blocks", "check_reference", "reference"]

BIT = BL.BIT
U8 = Array(8, BIT)
WORD = Array(64, BIT)
STATICS = ("COIN", "RID")
CONFORMANCE = "protocol-defined: the served seed rule, whose function is `served_seed.derived` or the body's seed (`reference`)"


def blocks(COIN: int, RID: int) -> int:
    """The SHA-512 blocks `ServedSeed_v1{COIN, RID}` compresses."""
    return -(-(len(served_seed.DERIVED) + 8 + COIN + RID + 17) // 128)


def _refuse(S: dict) -> str | None:
    bad = [k for k in STATICS if k in S and not (type(S[k]) is int and S[k] >= 1)]
    return f"the seed rule's {bad[0]} is a length in bytes, at least 1, not {S[bad[0]]!r}" if bad else None


SeedSelect = TR.part("SeedSelect", (), lambda S: ((("seeded", BIT), ("seed", WORD), ("digest", WORD)), WORD),
                     lambda C, S, seeded, seed, digest: [mux(C, seeded[0], list(seed), list(digest[1:]) + [ZERO])],
                     "seed where seeded, else bits 1-63 of a SHA-512 digest's first word: its first 8 bytes >> 1",
                     conformance=CONFORMANCE)
SeedEqual = TR.part("SeedEqual", (), lambda S: ((("a", WORD), ("b", WORD)), BIT),
                    lambda C, S, a, b: [[and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])]], "1 iff a = b",
                    conformance=CONFORMANCE)


def _rule_sig(S: Any) -> tuple:
    return (("seeded", BIT), ("seed", WORD), ("coin", Array(S.COIN, U8)), ("user", Array(S.RID, U8))), WORD


def _check_sig(S: Any) -> tuple:
    params, _ = _rule_sig(S)
    return (("served", WORD), *params), BIT


def _bits(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _rule(B: Any, S: Any, seeded: Coll, seed: Coll, coin: Coll, user: Coll) -> Coll:
    one = [B.call(BL.constant(b)).refs[0] for b in (0, 1)]

    def const(data: bytes) -> list[list]:
        return [[one[x >> i & 1] for i in range(8)] for x in data]

    cb, ub = _bits(coin), _bits(user)
    data = (const(served_seed.DERIVED + S.COIN.to_bytes(4, "big")) + [cb[8 * i:8 * i + 8] for i in range(S.COIN)]
            + const(S.RID.to_bytes(4, "big")) + [ub[8 * i:8 * i + 8] for i in range(S.RID)])
    data += const(H.pad_tail(len(data)))
    words = [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]
    cv = Coll(Array(8, WORD), Explicit(tuple(one[v >> i & 1] for v in H.IV for i in range(64))))
    for i in range(0, len(words), 16):
        cv = B.call(Sha512Compress, cv, Coll(Array(16, WORD), Explicit(tuple(x for w in words[i:i + 16] for x in w))))
    return B.call(SeedSelect, seeded, seed, cv[0])


def _check(B: Any, S: Any, served: Coll, seeded: Coll, seed: Coll, coin: Coll, user: Coll) -> Coll:
    return B.call(SeedEqual, served, B.call(bind(ServedSeed, COIN=S.COIN, RID=S.RID), seeded, seed, coin, user))


ServedSeed = CompositeDefinition("ServedSeed", 1, STATICS, _rule_sig, _rule, refuse=_refuse, conformance=CONFORMANCE,
                                 doc="A sampled request's seed: its body's where it names one, else served_seed.derived of the "
                                     "window's verifier coin (COIN bytes) and its id (RID bytes), on Sha512Compress_v1 Calls")
SeedCheck = CompositeDefinition("SeedCheck", 1, STATICS, _check_sig, _check, refuse=_refuse, conformance=CONFORMANCE,
                                doc="1 iff served, the seed a Program's sampler read, is ServedSeed_v1's of the certified inputs")


def reference(seeded: int, seed: int, coin: bytes, user: bytes) -> int:
    """`ServedSeed_v1`'s function."""
    return int(seed) if seeded else served_seed.derived(bytes(coin), bytes(user).decode("ascii"))


def check_reference(served: int, seeded: int, seed: int, coin: bytes, user: bytes) -> int:
    """`SeedCheck_v1`'s function."""
    return int(int(served) == reference(seeded, seed, coin, user))
