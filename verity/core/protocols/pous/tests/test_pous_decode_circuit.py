"""P2's decode circuit (`pous.decode_circuit`) against `schemes.p2`'s reference decode and `hashlib`'s SHA-512."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as EB
from verity.core.protocols.pous import decode_circuit as D
from verity.core.protocols.pous.schemes import p2 as P2

RNG = random.Random(13)
FIXTURES = Path(__file__).parent / "fixtures"


def _bits(rows: list[bytes]) -> np.ndarray:
    return np.array([[(b >> k) & 1 for b in row for k in range(8)] for row in rows], dtype=np.uint8)


def _bytes(row) -> bytes:
    return bytes(int(sum(int(row[8 * i + k]) << k for k in range(8))) for i in range(len(row) // 8))


@pytest.mark.parametrize(("L", "NB"), [(128, 8), (D.P2_BASE_BYTES, 130), (256 + D.MAX_REST, 64)])
def test_sha512_ctr_is_hashlibs(L, NB):
    """The stream from the base's midstate and rest is hashlib's counter-mode SHA-512, with no rest, a 16-byte rest
    (P2's) and the longest rest one compression takes."""
    msgs = [bytes(RNG.getrandbits(8) for _ in range(L)) for _ in range(3)]
    M = L // 128
    mid = EB.evaluate_bits(bind(D.PousSha512Mid, M=M), _bits([m[:128 * M] for m in msgs]))
    rest = [_bits([m[128 * M:] for m in msgs])] if L % 128 else []
    for purpose in (b"K", b"t"):
        out = EB.evaluate_bits(bind(D.PousSha512Ctr, L=L, D=purpose[0], NB=NB), mid, *rest)
        want = [b"".join(hashlib.sha512(m + purpose + i.to_bytes(4, "big")).digest() for i in range(-(-NB // 64)))[:NB]
                for m in msgs]
        assert [_bytes(r) for r in out] == want
    with pytest.raises(ValueError, match="at most"):
        bind(D.PousSha512Ctr, L=128 + D.MAX_REST + 1, D=75, NB=8).signature()


def test_byte_swap_is_wiring():
    x = [bytes(RNG.getrandbits(8) for _ in range(5)) for _ in range(4)]
    fn = bind(D.PousByteSwap, N=5)
    assert [_bytes(r) for r in EB.evaluate_bits(fn, _bits(x))] == [b[::-1] for b in x]
    assert EB.compile(fn).counts == {"and": 0, "xor": 0, "not": 0}


def _decode_cases(params: P2.P2Params, n_random: int):
    """Codewords from the reference encode, edge blocks, and random blocks, with their expansion bases."""
    salt = bytes(RNG.getrandbits(8) for _ in range(24))
    w = bytes(RNG.getrandbits(8) for _ in range(params.segment_bytes))
    blocks = P2.encode_segment(params, None, salt, 0, w)
    assert P2.decode_segment(params, None, salt, 0, blocks) == w
    f, nb = P2._field(params), params.block_bytes
    cases = list(enumerate(blocks))
    cases += [(1, c.to_bytes(nb, "big")) for c in (0, 1, 2, f.p - 1, f.p, f.p + 1, (1 << params.w) - 1)]
    cases += [(RNG.randrange(params.blocks_per_segment), RNG.getrandbits(params.w).to_bytes(nb, "big"))
              for _ in range(n_random)]
    return salt, cases


def _reference(params: P2.P2Params, prefix: bytes, j: int, block: bytes) -> tuple[int, bytes | None]:
    f = P2._field(params)
    k, t = P2.sha512_keys(params, prefix, 0, j)
    c = int.from_bytes(block, "big")
    m = (f.sq(f.sigma(f.sq(c), k)) - t) % f.p
    if c >= f.p or m >> (8 * params.payload_bytes):
        return 0, None
    return 1, m.to_bytes(params.payload_bytes, "big")


@pytest.mark.parametrize(("w", "c0"), [(64, 189), (128, 173)])
def test_decode_is_the_reference(w, c0):
    """At 64 bits Karatsuba squarings, at 128 a Toom-3 level: codewords decode to their payload byte for byte, and every
    block the reference rejects (c ≥ p, or a result wider than the payload) has ok = 0."""
    params = P2.P2Params(w, c0, 4)
    salt, cases = _decode_cases(params, 8)
    prefix = P2.expand_prefix(params, salt)
    fn = bind(D.PousP2Decode, W=w, C0=c0, L=len(prefix) + 16)
    bases = [prefix + (0).to_bytes(8, "big") + j.to_bytes(8, "big") for j, _ in cases]
    out = EB.evaluate_bits(fn, _bits([b for _, b in cases]), _bits(bases))
    for (j, block), row in zip(cases, out):
        ok, payload = _reference(params, prefix, j, block)
        assert int(row[-1]) == ok
        if ok:
            assert _bytes(row[:-1]) == payload
    assert out[:4, -1].tolist() == [1, 1, 1, 1]


def test_finish_refuses_a_first_t_candidate_at_or_above_p():
    """`schemes.p2` reads t's next candidate when the first is ≥ p; the circuit reads only the first and says ok = 0
    (probability C0 / 2^W per block, so the full decode never meets it on a test vector).  Below p, the payload is
    (u − t) mod p."""
    w, c0 = 64, 189
    p, pay = (1 << w) - c0, (w - 16) // 8
    to = lambda v: v.to_bytes(8, "little")  # noqa: E731
    us = [RNG.randrange(p) for _ in range(12)]
    ts = [p, p + 1, (1 << w) - 1] + [(u - RNG.getrandbits(8 * pay)) % p for u in us[3:]]
    cs = [RNG.randrange(p) for _ in us]
    out = EB.evaluate_bits(bind(D.PousP2Finish, W=w, C0=c0), _bits([to(u) for u in us]), _bits([to(t) for t in ts]),
                           _bits([to(c) for c in cs]))
    assert out[:3, -1].tolist() == [0, 0, 0]
    for u, t, row in zip(us[3:], ts[3:], out[3:]):
        assert int(row[-1]) == 1
        assert _bytes(row[:-1]) == ((u - t) % p).to_bytes(pay, "big")


def test_sq_and_sigma_match_the_field():
    """On field elements (x < p), the only inputs an ok decode feeds them: c < p, and σ_K's output is below p."""
    params = P2.P2Params(64, 189, 4)
    f, p = P2._field(params), params.p
    xs = [0, 1, 2, p - 2, p - 1] + [RNG.randrange(p) for _ in range(20)]
    ks = [RNG.getrandbits(64) for _ in xs] + [RNG.getrandbits(64), RNG.getrandbits(64)]
    xs += [p - 1, p - 2]
    to = lambda v: v.to_bytes(8, "little")  # noqa: E731
    sq = EB.evaluate_bits(bind(D.PousP2Sq, W=64, C0=189), _bits([to(x) for x in xs]))
    assert [int.from_bytes(_bytes(r), "little") for r in sq] == [f.sq(x) for x in xs]
    sg = EB.evaluate_bits(bind(D.PousP2Sigma, W=64, C0=189), _bits([to(x) for x in xs]), _bits([to(k) for k in ks]))
    assert [int.from_bytes(_bytes(r), "little") for r in sg] == [f.sigma(x, k) for x, k in zip(xs, ks)]


SLOW = pytest.mark.skipif(not os.environ.get("VERITY_POUS_SLOW"), reason="minutes at P2's width: VERITY_POUS_SLOW=1")


@SLOW
def test_counts_at_p2():
    """The setup cost model's per-block decode (`decode_circuit`'s docstring)."""
    at = dict(W=P2.POINT[0], C0=P2.POINT[1])
    assert EB.compile(bind(D.PousP2Decode, L=D.P2_BASE_BYTES, **at)).counts == {"and": 31960530, "xor": 93920605, "not": 5381422}
    assert EB.compile(bind(D.PousP2Sq, **at)).counts == {"and": 14073739, "xor": 37860749, "not": 2504496}
    ctr = {"and": 1820247, "xor": 8901166, "not": 166687}
    assert EB.compile(bind(D.PousSha512Ctr, L=D.P2_BASE_BYTES, D=ord("K"), NB=2056)).counts == ctr


@SLOW
def test_decode_at_p2_is_the_registered_vector():
    """The registered scheme's pinned small instance (`fixtures/schemes/p2-16448-v4.json`): both blocks decode to W."""
    v = json.loads((FIXTURES / "schemes" / "p2-16448-v4.json").read_text())["small"]
    params = P2.P2Params(**{k: v["params"][k] for k in ("w", "c0", "blocks_per_segment", "segments")})
    salt, blocks = bytes.fromhex(v["salt"]), [bytes.fromhex(b) for b in v["blocks"][0]]
    prefix = P2.expand_prefix(params, salt)
    assert len(prefix) + 16 == D.P2_BASE_BYTES
    out = EB.evaluate_bits(bind(D.PousP2Decode, W=params.w, C0=params.c0, L=D.P2_BASE_BYTES), _bits(blocks),
                           _bits([prefix + bytes(8) + j.to_bytes(8, "big") for j in range(len(blocks))]))
    assert out[:, -1].tolist() == [1, 1]
    assert b"".join(_bytes(r[:-1]) for r in out) == P2.decode_segment(params, None, salt, 0, blocks)
