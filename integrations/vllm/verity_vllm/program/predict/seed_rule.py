"""The seed rule a request Program checks, which the verifier requires to be 1: the seed its sampler read is its certified body's
``seed`` where the body names one, else `served_seed.derived` of the window's verifier coin and the request's id, in gates.

    ServedSeed_v1{COIN, RID}  (seeded: bit, seed: Word, coin: Array<COIN, u8>, user: Array<RID, u8>) -> Word
    SeedCheck_v1{COIN, RID}   (served: Word, seeded, seed, coin, user) -> bit
    SeedBlocks_v1{COIN, RID}  (coin, user) -> Array<8 + 16 n, Word>: SHA-512's IV, then the message's n padded blocks
    SeedSelect_v1             (seeded: bit, seed: Word, digest: Word) -> Word: seed if seeded, else digest's bits 1-63
    SeedEqual_v1              (a: Word, b: Word) -> bit: 1 iff a = b

(Word is ``Array<64, Value<1>>``, LSB first; a byte's bits are LSB first, a block's words big-endian.) `ServedSeed_v1` is the
rule: ``seed`` where ``seeded``, else the first 63 bits of SHA-512 over `served_seed.DERIVED`, u32be(COIN), the coin, u32be(RID)
and the id (`served_seed.derived`). It and `SeedCheck_v1` are calls and wiring only: `SeedBlocks_v1`, then a Call of
`lms.Sha512Compress_v1` on each block (`blocks`: one where the message and its padding fit 128 bytes, else two), then
`SeedSelect_v1` on the digest's first word. COIN and RID are the coin's and the id's lengths in bytes, so the padding is
constant. ``seeded``, ``seed`` and ``user`` are ingress schema v1's (`boolean_io_shell.IngressSampled_v1`), certified, and the
coin is the window's. `SeedCheck_v1` is 1 exactly when ``served``, the seed the Program's sampler read, is the rule's
(`SeedEqual_v1`): any other seed fails it.

No request Program returns the check yet, so a stochastic request's seed is tied to its certified body by the compiler
(`reads.seed_of`), not by the circuit. `reference` and `check_reference` are the two functions. Imports without torch.
"""
from __future__ import annotations

from typing import Any

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import NOT, ZERO, and_reduce, const_bits, mux
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll, Explicit
from verity.circuits.types import Array
from verity.ml.commitments.gates import sha512 as H
from verity.ml.commitments.gates.lms import Sha512Compress
from verity_vllm.engine import served_seed

__all__ = ["SeedBlocks", "SeedCheck", "SeedEqual", "SeedSelect", "ServedSeed", "blocks", "check_reference", "reference"]

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


def _ins(S: Any) -> tuple:
    return ("coin", Array(S.COIN, U8)), ("user", Array(S.RID, U8))


def _blocks(C: Any, S: Any, *e: list) -> list[list]:
    def const(data: bytes) -> list[list]:
        return [const_bits(x, 8) for x in data]

    data = (const(served_seed.DERIVED + S.COIN.to_bytes(4, "big")) + [list(b) for b in e[:S.COIN]]
            + const(S.RID.to_bytes(4, "big")) + [list(b) for b in e[S.COIN:]])
    data += const(H.pad_tail(len(data)))
    return [const_bits(v, 64) for v in H.IV] + [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


SeedBlocks = TR.part("SeedBlocks", STATICS, lambda S: (_ins(S), Array(8 + 16 * blocks(S.COIN, S.RID), WORD)), _blocks,
                     "SHA-512's IV, then the padded blocks of served_seed.DERIVED, u32be(COIN), the coin, u32be(RID) and the id, "
                     "as 64-bit words: constants and wiring", conformance=CONFORMANCE)
SeedSelect = TR.part("SeedSelect", (), lambda S: ((("seeded", BIT), ("seed", WORD), ("digest", WORD)), WORD),
                     lambda C, S, seeded, seed, digest: [mux(C, seeded[0], list(seed), list(digest[1:]) + [ZERO])],
                     "seed where seeded, else bits 1-63 of a SHA-512 digest's first word: its first 8 bytes >> 1",
                     conformance=CONFORMANCE)
SeedEqual = TR.part("SeedEqual", (), lambda S: ((("a", WORD), ("b", WORD)), BIT),
                    lambda C, S, a, b: [[and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])]], "1 iff a = b",
                    conformance=CONFORMANCE)


def _rule_sig(S: Any) -> tuple:
    return (("seeded", BIT), ("seed", WORD), *_ins(S)), WORD


def _check_sig(S: Any) -> tuple:
    params, _ = _rule_sig(S)
    return (("served", WORD), *params), BIT


def _words(bits: list, k: int, n: int) -> Coll:
    return Coll(Array(n, WORD), Explicit(tuple(bits[64 * k:64 * (k + n)])))


def _rule(B: Any, S: Any, seeded: Coll, seed: Coll, coin: Coll, user: Coll) -> Coll:
    out = B.call(bind(SeedBlocks, COIN=S.COIN, RID=S.RID), coin, user)
    bits = [out.refs[k] for k in range(out.refs.count)]
    cv = _words(bits, 0, 8)
    for i in range(blocks(S.COIN, S.RID)):
        cv = B.call(Sha512Compress, cv, _words(bits, 8 + 16 * i, 16))
    return B.call(SeedSelect, seeded, seed, cv[0])


def _check(B: Any, S: Any, served: Coll, *ins: Coll) -> Coll:
    return B.call(SeedEqual, served, _rule(B, S, *ins))


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
