"""The seed rule a request Program checks, which the verifier requires to be 1: the seed its sampler read is its certified body's
``seed`` where the body names one, else `served_seed.derived` of the window's verifier coin and the request's id, in gates.

`ServedSeed_v1{COIN, RID}` (seeded: bit, seed: i64, coin: Array<COIN, u8>, user: Array<RID, u8>) -> i64 is the rule: ``seed``
where ``seeded``, else the first 63 bits of SHA-512 over `served_seed.DERIVED`, u32be(COIN), the coin, u32be(RID) and the id
(`served_seed.derived`), on `verity.ml.commitments.gates.sha512`'s compression. COIN and RID are the coin's and the id's lengths
in bytes, so the message's padding is constant and its constant bytes fold. ``seeded``, ``seed`` and ``user`` are ingress
schema v1's (`boolean_io_shell.IngressSampled_v1`), certified, and the coin is the window's. `SeedCheck_v1{COIN, RID}` (served:
i64, seeded, seed, coin, user) -> bit is 1 exactly when ``served``, the seed the Program's sampler read, is the rule's: any
other seed fails it.

No request Program returns the check yet, so a stochastic request's seed is tied to its certified body by the compiler
(`reads.seed_of`), not by the circuit. `reference` and `check_reference` are the two functions. Imports without torch.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ZERO, and_reduce, const_bits, mux
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates import sha512 as H
from verity_vllm.engine import served_seed

__all__ = ["SeedCheck", "ServedSeed", "check_reference", "reference", "rule"]

BIT = BL.BIT
U8 = Array(8, BIT)
I64 = Array(64, BIT)
STATICS = ("COIN", "RID")


def _refuse(S: dict) -> str | None:
    bad = [k for k in STATICS if k in S and not (type(S[k]) is int and S[k] >= 1)]
    return f"the seed rule's {bad[0]} is a length in bytes, at least 1, not {S[bad[0]]!r}" if bad else None


def _rule_sig(S: Any) -> tuple:
    return (("seeded", BIT), ("seed", I64), ("coin", Array(S.COIN, U8)), ("user", Array(S.RID, U8))), I64


def _check_sig(S: Any) -> tuple:
    params, _ = _rule_sig(S)
    return (("served", I64), *params), BIT


def _const(data: bytes) -> list[list]:
    return [const_bits(b, 8) for b in data]


def _sha512_word0(C: Any, data: list[list]) -> list:
    """The first word of SHA-512 over ``data``'s bytes (each 8 bits LSB first), 64 bits LSB first: its digest's first 8 bytes
    big-endian."""
    data = data + _const(H.pad_tail(len(data)))
    msg = [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]
    st = [const_bits(v, 64) for v in H.IV]
    for i in range(0, len(msg), 16):
        st = H.compress_gadget(C, st, msg[i:i + 16])
    return st[0]


def rule(C: Any, S: Any, seeded: Any, seed: list, coin: list[list], user: list[list]) -> list:
    """`ServedSeed_v1`'s gates: ``seed`` where ``seeded``, else `served_seed.derived`'s 63 bits."""
    head = _const(served_seed.DERIVED + S.COIN.to_bytes(4, "big"))
    w0 = _sha512_word0(C, head + list(coin) + _const(S.RID.to_bytes(4, "big")) + list(user))
    return mux(C, seeded, list(seed), w0[1:] + [ZERO])


def _rule(S: Any) -> Any:
    return lambda C, *e: [rule(C, S, e[0][0], e[1], e[2:2 + S.COIN], e[2 + S.COIN:])]


def _check(S: Any) -> Any:
    def run(C: Any, *e: Any) -> list:
        want = rule(C, S, e[1][0], e[2], e[3:3 + S.COIN], e[3 + S.COIN:])
        return [[and_reduce(C, [NOT(a ^ b) for a, b in zip(e[0], want)])]]
    return run


ServedSeed = EW.specialized("ServedSeed", 1, STATICS, _rule, _rule_sig, refuse=_refuse,
                            doc="A sampled request's seed: its body's where it names one, else served_seed.derived of the "
                                "window's verifier coin (COIN bytes) and its id (RID bytes), on SHA-512 gates")
SeedCheck = EW.specialized("SeedCheck", 1, STATICS, _check, _check_sig, refuse=_refuse,
                           doc="1 iff served, the seed a Program's sampler read, is ServedSeed_v1's of the certified inputs")


def reference(seeded: int, seed: int, coin: bytes, user: bytes) -> int:
    """`ServedSeed_v1`'s function."""
    return int(seed) if seeded else served_seed.derived(bytes(coin), bytes(user).decode("ascii"))


def check_reference(served: int, seeded: int, seed: int, coin: bytes, user: bytes) -> int:
    """`SeedCheck_v1`'s function."""
    return int(int(served) == reference(seeded, seed, coin, user))
