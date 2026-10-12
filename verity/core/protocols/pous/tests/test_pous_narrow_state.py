"""The narrow-state attack (`docs/p3-cryptanalysis.md` "Narrow-state H", ported from the cryptanalyst's
`narrow_state.py`): store each key's internal hash state instead of the key, then rebuild all of `C` from `W` and those
states alone.  Against SHAKE-256 as `H` it compresses `C`; against the overwrite chain every state that finishes a key
is wider than a label, so it cannot, and its narrow part (the last 512 bits) finishes no key.
"""

from __future__ import annotations

import hashlib
import random

from verity.core.protocols.pous.labels import salt_value, tag, transpose, w_labels
from verity.core.protocols.pous.oracle import PiQuery, Tweak, run
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import (CAPACITY_BITS, FeistelP, Primitives, RandomOracleKey, key_query, pad_block,
                                    reference)
from verity.core.protocols.pous.scheme import encode_prog

PARAMS = Params(16, 256, 12)
MASK = (1 << 64) - 1
ROT = [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14]
RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
      0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
      0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
      0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
      0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
RATE = 136


def keccak_f(a: list[int]) -> None:
    for rc in RC:
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        d = [c[(x - 1) % 5] ^ (((c[(x + 1) % 5] << 1) | (c[(x + 1) % 5] >> 63)) & MASK) for x in range(5)]
        for i in range(25):
            a[i] ^= d[i % 5]
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                v, r = a[x + 5 * y], ROT[x + 5 * y]
                b[y + 5 * ((2 * x + 3 * y) % 5)] = ((v << r) | (v >> (64 - r))) & MASK if r else v
        for x in range(5):
            for y in range(5):
                a[x + 5 * y] = b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y])
        a[0] ^= rc


def absorb(msg: bytes) -> list[int]:
    """SHAKE-256's 1600-bit state after absorbing `msg`: all a key needs."""
    pad = msg + b"\x1f" + b"\x00" * ((-len(msg) - 2) % RATE) + b"\x80"
    a = [0] * 25
    for off in range(0, len(pad), RATE):
        for i in range(RATE // 8):
            a[i] ^= int.from_bytes(pad[off + 8 * i:off + 8 * i + 8], "little")
        keccak_f(a)
    return a


def squeeze(state: list[int], n: int) -> bytes:
    a, out = list(state), b""
    while len(out) < n:
        out += b"".join(a[i].to_bytes(8, "little") for i in range(RATE // 8))
        if len(out) < n:
            keccak_f(a)
    return out[:n]


def shake_message(q) -> bytes:
    return (q.tag.to_bytes(32, "little") + bytes([q.layer]) + q.node.to_bytes(2, "little")
            + b"".join(label.to_bytes(PARAMS.m // 8, "little") for _, label in q.parents))


def rebuild(prims: Primitives, w: list[int], t: int, keys: tuple[list[int], list[int]]) -> list[int]:
    """The attacker's online phase: every lower label from `W`, the transpose, every top label."""
    n, wc = PARAMS.n, PARAMS.wc
    c1 = [prims.p.forward(Tweak(t, 0, v), w[v] ^ keys[0][v]) for v in range(n)]
    x2 = transpose(c1, n, wc)
    return [prims.p.forward(Tweak(t, 1, v), x2[v] ^ keys[1][v]) for v in range(n)]


def _world(prims: Primitives):
    w = random.Random(11).randbytes(PARAMS.w_bytes)
    salt = salt_value(b"\x0b" * 24)
    w_int = int.from_bytes(w, "little")
    [(c1, c2)], _ = run(encode_prog(PARAMS, prims.h, w_int, salt), prims)
    return w_labels(PARAMS, w_int)[0], tag(salt, 0), c1, c2


def test_keccak_is_shake256():
    for msg in (b"", b"abc", bytes(range(200)), b"x" * 500):
        for n in (1, 136, 137, 1024):
            assert squeeze(absorb(msg), n) == hashlib.shake_256(msg).digest(n)


def test_narrow_state_compresses_a_shake_h():
    mb = PARAMS.m // 8
    prims = Primitives(RandomOracleKey(lambda q: int.from_bytes(hashlib.shake_256(shake_message(q)).digest(mb), "little")),
                       FeistelP(PARAMS.m))
    w, t, c1, c2 = _world(prims)
    queries = [[key_query(PARAMS, t, layer, v, c1 if layer == 0 else c2) for v in range(PARAMS.n)] for layer in (0, 1)]
    states = [[absorb(shake_message(q)) for q in layer] for layer in queries]
    stored = 2 * PARAMS.n * 1600
    assert stored < PARAMS.n * PARAMS.m, "the states are smaller than C"
    keys = tuple([int.from_bytes(squeeze(s, mb), "little") for s in layer] for layer in states)
    assert rebuild(prims, w, t, keys) == c2


def _pre_pad_state(prims: Primitives, q) -> int:
    """The overwrite chain's value after every parent: only the pad call is left."""
    st = prims.h.stepper(q.tag, q.layer, q.node, [u for u, _ in q.parents])
    labels = dict(q.parents)
    while st.i < len(st.order):
        st.feed(prims.h.answer(st.query(labels)))
    return st.h


def test_narrow_state_fails_against_the_overwrite_chain():
    prims = reference(PARAMS)
    w, t, c1, c2 = _world(prims)
    queries = [[key_query(PARAMS, t, layer, v, c1 if layer == 0 else c2) for v in range(PARAMS.n)] for layer in (0, 1)]
    states = [[_pre_pad_state(prims, q) for q in layer] for layer in queries]
    pad = pad_block(PARAMS.m)
    finish = lambda h: prims.h.answer(PiQuery(pad | (h << PARAMS.m))) & ((1 << PARAMS.m) - 1)  # noqa: E731
    keys = tuple([finish(s) for s in layer] for layer in states)
    assert rebuild(prims, w, t, keys) == c2, "the chaining values do finish every key..."
    assert 2 * PARAMS.n * (PARAMS.m + CAPACITY_BITS) > PARAMS.n * PARAMS.m, "...but they are larger than C"
    narrow = [[s >> PARAMS.m << PARAMS.m for s in layer] for layer in states]
    keys = tuple([finish(s) for s in layer] for layer in narrow)
    assert not set(rebuild(prims, w, t, keys)) & set(c2), "the last 512 bits alone finish no key"
