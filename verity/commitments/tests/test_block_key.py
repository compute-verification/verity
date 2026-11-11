"""`verity.definitions.commitments.gates.block_key`: the references are hashlib's SHA-512 over the packed tokens and the print's sum of
powers, BlockKey_v1 agrees with them at the deployed binding (64 tokens of 18 bits, two compressions) and at a toy one (one
compression), a request's chain through the circuit is `chain`'s, KeyPrint_v1 prints as BlockKey_v1 does, and each costs
the gates pinned here."""
from __future__ import annotations

import hashlib
import random
import struct

import pytest

from verity.circuits.boolean.gf128 import mul
from verity.circuits.defs import bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.definitions.commitments.gates import block_key as BK

#: (Definition, statics) -> its traced circuit's (And, Xor, Not) gates
PINS = {
    ("BlockKey", (64, 18)): (120576, 593934, 11195),
    ("BlockKey", (4, 16)): (62380, 305833, 6235),
    ("KeyPrint", ()): (6561, 34090, 0),
}


def _bits(x: int, n: int) -> list[int]:
    return [(x >> i) & 1 for i in range(n)]


def _int(bs) -> int:
    return sum(int(b) << i for i, b in enumerate(bs))


def _words(key: bytes) -> list[list[int]]:
    return [_bits(w, 64) for w in struct.unpack(">8Q", key)]


def _key(bits) -> bytes:
    return struct.pack(">8Q", *[_int(bits[64 * i:64 * (i + 1)]) for i in range(8)])


def _evaluate(B: int, TOK: int, cases: list[tuple[int, bytes, list[int]]]) -> list[tuple[bytes, int]]:
    """BlockKey_v1{B, TOK}'s (key, print) on each (delta, parent, tokens)."""
    cols = [[_bits(d, 128) for d, _, _ in cases], [_words(p) for _, p, _ in cases],
            [[_bits(t, TOK) for t in ts] for _, _, ts in cases]]
    out = evaluate_bits(bind(BK.BlockKey, B=B, TOK=TOK), *cols).reshape(len(cases), -1).tolist()
    return [(_key(o[:512]), _int(o[512:640])) for o in out]


def test_the_references() -> None:
    assert BK.pack_tokens([1, 2, 3, 4], 4) == b"\x12\x34"
    assert BK.pack_tokens([0x3ffff, 0, 1, 2], 18) == (0x3ffff << 54 | 1 << 18 | 2).to_bytes(9, "big")
    rng = random.Random(1)
    parent, toks = rng.randbytes(64), [rng.getrandbits(18) for _ in range(64)]
    assert BK.block_key(parent, toks, 18) == hashlib.sha512(parent + BK.pack_tokens(toks, 18)).digest()
    assert BK.ROOT == hashlib.sha512(b"verity block key v1").digest()
    for _ in range(20):
        key, d = rng.randbytes(64), rng.getrandbits(128)
        w = struct.unpack(">8Q", key)
        p, s = 1, 0
        for i in range(4):
            s ^= mul(p, w[2 * i] | w[2 * i + 1] << 64)
            p = mul(p, d)
        assert BK.key_print(key, d) == s
    toks = [rng.getrandbits(16) for _ in range(10)]
    keys = BK.chain(toks, 4, 16)
    assert keys == [BK.block_key(BK.ROOT, toks[:4], 16), BK.block_key(keys[0], toks[4:8], 16)]


@pytest.mark.parametrize("B, TOK, n", [(4, 16, 8), pytest.param(64, 18, 3, marks=pytest.mark.slow)])
def test_block_key_agrees_with_the_reference(B, TOK, n) -> None:
    """At (64, 18) about 10 s, the trace's first evaluation: the deployed binding's two-block message, which the toy's one
    block never reaches."""
    rng = random.Random(B)
    cases = [(rng.getrandbits(128), rng.randbytes(64), [rng.getrandbits(TOK) for _ in range(B)]) for _ in range(n)]
    cases.append((0, BK.ROOT, [(1 << TOK) - 1] * B))
    want = [(k, BK.key_print(k, d)) for d, p, ts in cases for k in [BK.block_key(p, ts, TOK)]]
    assert _evaluate(B, TOK, cases) == want


def test_a_chain_through_the_circuit_is_the_references() -> None:
    rng = random.Random(2)
    toks, d = [rng.getrandbits(16) for _ in range(16)], rng.getrandbits(128)
    parent, got = BK.ROOT, []
    for i in range(4):
        parent, pr = _evaluate(4, 16, [(d, parent, toks[4 * i:4 * (i + 1)])])[0]
        assert pr == BK.key_print(parent, d)
        got.append(parent)
    assert got == BK.chain(toks, 4, 16)
    assert len({BK.key_print(k, d) for k in got}) == 4


def test_key_print_agrees_with_the_reference() -> None:
    rng = random.Random(3)
    cases = [(rng.getrandbits(128), rng.randbytes(64)) for _ in range(16)]
    out = evaluate_bits(bind(BK.KeyPrint), [_bits(d, 128) for d, _ in cases], [_words(k) for _, k in cases])
    assert [_int(o) for o in out.reshape(len(cases), -1).tolist()] == [BK.key_print(k, d) for d, k in cases]


def test_the_parts_cost_the_pinned_gates() -> None:
    for (name, statics), (ands, xors, nots) in PINS.items():
        assert getattr(BK, name).traced(*statics).counts == {"and": ands, "xor": xors, "not": nots}, name


def test_bad_statics_and_tokens_are_refused() -> None:
    with pytest.raises(ValueError, match="3 tokens of 5 bits are not whole bytes"):
        BK.BlockKey.signature(type("S", (), {"B": 3, "TOK": 5}))
    with pytest.raises(ValueError, match="TOK is 0"):
        BK.BlockKey.signature(type("S", (), {"B": 8, "TOK": 0}))
    with pytest.raises(ValueError, match="token 16 is not 4 bits"):
        BK.pack_tokens([16, 0], 4)
    with pytest.raises(ValueError, match="a parent key is 64 bytes"):
        BK.block_key(bytes(32), [0, 0], 4)
