"""`verity.core.protocols.pouw.circuit`'s word primitives and in-circuit hashes against their native counterparts."""
from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import replace

import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32_WRAP
from verity.ml.circuits.types import Array

from verity.core.protocols.pouw.circuit import words as W
from verity.core.protocols.pouw.circuit.hashes import Emit, keccak_lanes, sha512, sha_words, shake256


def _definition(name, n_in, n_out, body):
    return bind(CompositeDefinition(name, 0, (), lambda S: ((("x", Array(n_in, W.U32)),), Array(n_out, W.U32)), body,
                                    register=False))


@pytest.mark.parametrize("nwords", [1, 9, 14, 15, 17, 30])
def test_sha512_matches_hashlib(nwords):
    """Multi-block messages, padding across a block edge, some words as inputs and some as constants."""
    rng = random.Random(nwords)
    data = bytes(rng.randrange(256) for _ in range(8 * nwords))
    words = sha_words(data)
    ins = list(range(0, nwords, 2))

    def body(B, S, x):
        e = Emit(B)
        return e.array(sha512(e, [(x[2 * ins.index(i)], x[2 * ins.index(i) + 1]) if i in ins else p
                                  for i, p in enumerate(words)]))

    out = evaluate(_definition(f"TSha{nwords}", 2 * len(ins), 16, body), [h for i in ins for h in words[i]])
    assert b"".join(((out[2 * i] << 32) | out[2 * i + 1]).to_bytes(8, "big") for i in range(8)) == hashlib.sha512(data).digest()


@pytest.mark.parametrize("nlanes,nbytes", [(1, 4), (7, 64), (16, 136), (17, 200), (20, 300)])
def test_shake256_matches_hashlib(nlanes, nbytes):
    """Padding in and after a block, a second absorbed block, and squeezing past one rate block."""
    rng = random.Random(nlanes * 1000 + nbytes)
    data = bytes(rng.randrange(256) for _ in range(8 * nlanes))
    lanes = keccak_lanes(data)
    ins = list(range(0, nlanes, 3))

    def body(B, S, x):
        e = Emit(B)
        return e.array(shake256(e, [(x[2 * ins.index(i)], x[2 * ins.index(i) + 1]) if i in ins else p
                                    for i, p in enumerate(lanes)], nbytes))

    out = evaluate(_definition(f"TShake{nlanes}_{nbytes}", 2 * len(ins), (nbytes + 3) // 4, body),
                   [h for i in ins for h in lanes[i]])
    assert b"".join(v.to_bytes(4, "little") for v in out)[:nbytes] == hashlib.shake_256(data).digest(nbytes)


@pytest.mark.parametrize("nlanes,nbytes,domain", [(1, 32, 0x21), (20, 32, 0x22), (21, 32, 0x1F), (22, 200, 0x01),
                                                  (64, 32, 0x7F)])
def test_turboshake128_matches_core(nlanes, nbytes, domain):
    """Core's TurboSHAKE128 (`verity.ml.crypto.turboshake`): the domain byte and the final bit in one lane (20 lanes), a
    whole padding block (21), a second absorbed block, squeezing past one rate block, and Pearl-C's message (32 words of C̃
    and 32 of U, 64 lanes)."""
    from verity.ml.crypto.turboshake import turboshake128 as native

    from verity.core.protocols.pouw.circuit.hashes import turboshake128
    rng = random.Random(nlanes * 1000 + domain)
    data = bytes(rng.randrange(256) for _ in range(8 * nlanes))
    lanes = keccak_lanes(data)
    ins = list(range(0, nlanes, 2))

    def body(B, S, x):
        e = Emit(B)
        return e.array(turboshake128(e, [(x[2 * ins.index(i)], x[2 * ins.index(i) + 1]) if i in ins else p
                                         for i, p in enumerate(lanes)], nbytes, domain))

    out = evaluate(_definition(f"TTs{nlanes}_{nbytes}_{domain}", 2 * len(ins), (nbytes + 3) // 4, body),
                   [h for i in ins for h in lanes[i]])
    assert b"".join(v.to_bytes(4, "little") for v in out)[:nbytes] == native(data, domain, nbytes)
    with pytest.raises(ValueError):
        turboshake128(Emit(None), lanes, nbytes, 0x80)


def test_i32_to_f32_matches_the_float_cast():
    rng = random.Random(3)
    cases = [0, 1, -1, 2 ** 24, 2 ** 24 + 1, 2 ** 24 + 3, 2 ** 31 - 1, -2 ** 31, 16777217, -16777219]
    cases += [rng.randrange(-2 ** 31, 2 ** 31) for _ in range(2000)]
    for v in cases:
        assert W.i32_to_f32_rn(v & W.M32) == struct.unpack("<I", struct.pack("<f", float(v)))[0], v


def test_route_u_step_and_c0():
    rng = random.Random(4)
    x = [rng.randrange(33, 224) for _ in range(16)]
    y = [rng.randint(-127, 126) for _ in range(16)]
    acc = rng.randrange(1 << 32)
    assert W.PouwDot16U8S8.evaluate(acc, *x, *(v & 0xFF for v in y)) == (acc + sum(a * b for a, b in zip(x, y))) & W.M32
    assert W.PouwNegBetaSum16.evaluate(acc, *(v & 0xFF for v in y)) == (acc - 128 * sum(y)) & W.M32


def test_route_u_step_is_half_an_imma_step():
    """The wrapping u8 x s8 IMMA step over k = 32 (`silicon.integer`, as the PRO 6000 computes it) is two route-U steps:
    `PouwDot16U8S8_v1` over the first 16 products, then over the last 16."""
    imma = replace(BLACKWELL_SM120_S8_M16N8K32_WRAP, dtype="u8")
    rng = random.Random(5)
    cases = [(acc, [xa] * 32, [yb] * 32) for acc in (0, 1, 2 ** 31 - 1, 2 ** 31, W.M32) for xa in (0, 255) for yb in (0x7F, 0x80)]
    cases += [(rng.randrange(1 << 32), [rng.randrange(256) for _ in range(32)], [rng.randrange(256) for _ in range(32)])
              for _ in range(20_000)]
    for acc, x, y in cases:
        half = W.PouwDot16U8S8.evaluate(acc, *x[:16], *y[:16])
        assert imma.step(acc, x, y) == W.PouwDot16U8S8.evaluate(half, *x[16:], *y[16:]), (acc, x, y)


def test_forming_primitives():
    for a in (-64, 0, 63):
        for e in (-31, 0, 32):
            assert W.PouwFormX.evaluate(a & 0x7F, e & 0xFF) == a + e + 128          # X + β is u8 on the domain
            assert W.PouwFormX3.evaluate(a & 0x7F, e & 0xFF, e & 0xFF) == 128 - (a + 2 * e)
    assert W.PouwNoiseByte.evaluate(0x3F00, 1) == 32 and W.PouwNoiseByte.evaluate(0, 2) == (-31) & 0xFF
