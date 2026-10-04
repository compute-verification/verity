"""`verity.ml.boolean.gf2k`: the references agree with schoolbook arithmetic and with C-Flock's constants, every circuit agrees
with its reference on random elements, Gf128Mul has Karatsuba's 2,187 ANDs, and every Definition is pinned by gate count and
program digest."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from verity.evaluation import bits as BS
from verity.evaluation import evaluate_batch
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import REGISTRY, bind
from verity.ml.boolean import gf2k as G

#: a residual schema with every term kind, operands of one and three terms, a squared operand, a residual of added terms only
#: and an empty one
S = {"nw": 130, "nv": 4,
     "ops": [[["w", 0]], [["w", 1], ["v", 2]], [["v", 0]], [["v", 1]], [["wt", 2, 5]], [["w", 129], ["w", 3], ["v", 3]]],
     "res": [{"mul": [[0, 2], [1, 3]], "add": [["v", 3]]}, {"mul": [[4, 0], [5, 5], [1, 2]], "add": []},
             {"mul": [], "add": [["w", 7], ["v", 1]]}, {"mul": [], "add": []}]}

TARGETS = [bind(G.Gf128Mul), bind(G.Gf256Mul), bind(G.GfScale, L=2), bind(G.GfResiduals, S=S)]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Gf128Mul_v1": (2187, 12608, 0, "aca44f49b4642424b8dac5f475e3cd7e44554210b7ced68bad3fd8000e859f52"),
    "Gf256Mul_v1": (6561, 38339, 0, "38c351a210051669bb69e02eb3a3d529d23a50189954b41d26ae90706e76d7b6"),
    "GfScale_v1{L=2}": (4374, 25216, 0, "fc34a33b163490b270390b229f7d9efa4d2a11aab186149991968f8c8ed0a078"),
    "GfResiduals_v1{S={5a0897f6f55a}}": (10935, 64064, 0, "88e6a6e2725ceab5afce68eb3516e19aa6cdacb8b9bd8e94615e8b654d222393"),
}


def _bits(x: int) -> list[int]:
    return [(x >> i) & 1 for i in range(128)]


def _val(bs) -> int:
    return sum(int(b) << i for i, b in enumerate(bs))


def _arr(rows) -> np.ndarray:
    return np.array([[_bits(x) for x in r] if isinstance(r, list) else _bits(r) for r in rows], dtype=np.uint8)


def _school(a: int, b: int) -> int:
    r = 0
    for i in range(128):
        if (b >> i) & 1:
            r ^= a << i
    return r


def test_the_references() -> None:
    rng = random.Random(1)
    for _ in range(300):
        a, b = rng.getrandbits(128), rng.getrandbits(128)
        assert G.clmul(a, b) == _school(a, b)
        assert G.mul(a, b) == G.mul(b, a)
        r = _school(a, b)
        for k in range(254, 127, -1):
            if (r >> k) & 1:
                r ^= (1 << k) | (0x87 << (k - 128))
        assert G.mul(a, b) == r
    a = rng.getrandbits(128)
    assert G.mul(a, G.inv(a)) == 1 and G.inv(0) == 0
    assert G.mul(G.XINV, 2) == 1
    assert G.from_bytes(G.to_bytes(a)) == a and G.of_hex(G.hex128(a)) == a
    # u^2 + u + x^-1 = 0 in K
    u = (0, 1)
    assert G.kmul(u, u) == (G.XINV, 1)
    # φ8's generator is a root of X^8 + X^4 + X^3 + X + 1 (C-Flock's PROTOCOL.md §2.2), and φ8 is additive
    g = [1]
    for _ in range(8):
        g.append(G.mul(g[-1], G.PHI8_GEN))
    assert g[8] ^ g[4] ^ g[3] ^ g[1] ^ g[0] == 0
    assert all(G.phi8(x ^ y) == G.phi8(x) ^ G.phi8(y) for x in range(0, 256, 7) for y in range(0, 256, 11))


def test_gf128mul_agrees_with_mul() -> None:
    rng = random.Random(2)
    edges = [0, 1, 2, G.XINV, G.MASK, 1 << 127]
    pairs = [(a, b) for a in edges for b in edges] + [(rng.getrandbits(128), rng.getrandbits(128)) for _ in range(200)]
    out = evaluate_batch(bind(G.Gf128Mul), _arr([a for a, _ in pairs]), _arr([b for _, b in pairs]))
    assert [_val(o) for o in out] == [G.mul(a, b) for a, b in pairs]


def test_gf256mul_agrees_with_kmul() -> None:
    rng = random.Random(3)
    xs = [(rng.getrandbits(128), rng.getrandbits(128)) for _ in range(100)] + [(0, 1), (1, 0)]
    ys = [(rng.getrandbits(128), rng.getrandbits(128)) for _ in range(100)] + [(0, 1), (0, 1)]
    out = evaluate_batch(bind(G.Gf256Mul), _arr([list(x) for x in xs]), _arr([list(y) for y in ys]))
    assert [(_val(o[0]), _val(o[1])) for o in out] == [G.kmul(x, y) for x, y in zip(xs, ys)]


def test_gfscale_agrees_with_mul() -> None:
    rng = random.Random(4)
    L = 5
    a = [rng.getrandbits(128) for _ in range(40)]
    b = [[rng.getrandbits(128) for _ in range(L)] for _ in a]
    out = evaluate_batch(bind(G.GfScale, L=L), _arr(a), _arr(b))
    assert [[_val(x) for x in o] for o in out] == [[G.mul(x, y) for y in r] for x, r in zip(a, b)]


def test_gfresiduals_agrees_with_residuals() -> None:
    rng = random.Random(5)
    w = [[rng.getrandbits(128) for _ in range(S["nw"])] for _ in range(24)]
    v = [[rng.getrandbits(128) for _ in range(S["nv"])] for _ in range(24)]
    out = evaluate_batch(bind(G.GfResiduals, S=S), _arr(w), _arr(v))
    assert [[_val(x) for x in o] for o in out] == [G.residuals(S, a, b) for a, b in zip(w, v)]
    assert G.residuals(S, w[0], v[0])[3] == 0
    t = sum(((w[0][2 + j] >> 5) & 1) << j for j in range(128))
    assert G.residuals(S, w[0], v[0])[1] == G.mul(t, w[0][0]) ^ G.mul(w[0][129] ^ w[0][3] ^ v[0][3], w[0][129] ^ w[0][3] ^ v[0][3]) \
        ^ G.mul(w[0][1] ^ v[0][2], v[0][0])


def test_gfresiduals_rejects_ports_of_the_wrong_length() -> None:
    with pytest.raises(ValueError):
        G.residuals(S, [0] * (S["nw"] - 1), [0] * S["nv"])


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


def test_karatsuba_ands_are_one_per_base_product() -> None:
    assert G.Gf128Mul.traced().counts["and"] == 3 ** 7
    assert G.Gf256Mul.traced().counts["and"] == 3 * 3 ** 7
