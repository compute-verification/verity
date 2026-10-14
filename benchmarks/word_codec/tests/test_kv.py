"""The KV cache's codec (`kv.py`) and the kernel's arithmetic twin (`kv_twin.py`) against it."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import kv as K  # noqa: E402
import kv_twin as T  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/kv/vectors").digest()
NAMES = sorted(K.FIELDS)
VECTORS = HERE / "kv_vectors.json"


def _probable_prime(n: int) -> bool:
    if n % 2 == 0:
        return n == 2
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _edges(f: K.Field) -> list[int]:
    p = f.p
    return [0, 1, 2, 3, p - 1, p - 2, (p - 1) // 2, (p + 1) // 2, 1 << (f.n - 1), (1 << 32) - 1, 1 << 32, f.c]


@pytest.mark.parametrize("name", NAMES)
def test_fields(name):
    K.check_fields()
    f = K.field(name)
    assert _probable_prime(f.p)
    assert not any(_probable_prime((1 << f.n) - c) for c in range(1, f.c, 4))
    assert f.root_squarings == f.n - 3
    assert f.root_multiplies == {"p512": 18, "p768": 18, "p1024": 21, "p1280": 18, "p1536": 19}[name]
    assert f.word_bytes == f.n // 8 and f.c < 1 << 16
    with pytest.raises(ValueError, match="unknown field"):
        K.field("p511")


def test_kernel_constants_match():
    src = (HERE.parent / "kv.cu").read_text()
    for f in K.FIELDS.values():
        assert re.search(rf"constexpr int NBITS = {f.n};\nconstexpr u32 CF = {f.c};", src)
    assert f"constexpr int LOW_MIN = {K.LOW_MIN};" in src and "constexpr int KMAX = 8;" in src


@pytest.mark.parametrize("n,c", [(3, 1), (4, 5), (5, 1), (5, 9), (6, 21), (7, 1)])
def test_rho_is_a_permutation_with_the_parity_rule(n, c):
    f = K.Field("small", n, c)
    assert f.p % 4 == 3 and _probable_prime(f.p)
    images = [K.rho(a, f) for a in range(f.p)]
    assert sorted(images) == list(range(f.p))
    assert all(K.rho_inv(y, f) == a for a, y in enumerate(images))


@pytest.mark.parametrize("name", NAMES)
def test_rho_and_its_inverse(name):
    f = K.field(name)
    rng = random.Random(f.n)
    for a in _edges(f) + [rng.randrange(f.p) for _ in range(4)]:
        y = K.rho(a, f)
        assert 0 <= y < f.p and K.rho_inv(y, f) == a
    with pytest.raises(ValueError, match="canonical"):
        K.rho(f.p, f)


def test_key_derivation():
    for name in NAMES:
        k1, k2 = K.key(name, SEED, 3), K.key(name, SEED, 5)
        assert k1.c == k2.c[:3] and k1.tau == k2.tau != 0 and len(set(k2.c)) == 5
        assert all(0 <= v < k1.field.p for v in k2.c)
    assert K.key("p512", SEED, 1).c[0] != K.key("p768", SEED, 1).c[0]
    with pytest.raises(ValueError, match="seed"):
        K.key("p512", SEED[:31], 1)
    with pytest.raises(ValueError, match="round"):
        K.key("p512", SEED, 0)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("rounds", [1, 2, 4])
def test_round_trip(name, rounds):
    k = K.key(name, SEED, rounds)
    f = k.field
    rng = random.Random(rounds)
    for i, m in enumerate(_edges(f) + [rng.randrange(f.p)]):
        y = K.encode_word(m, 1000 + i, k)
        assert 0 <= y < f.p and K.decode_word(y, 1000 + i, k) == m
        assert K.decode_word(y, 1001 + i, k) != m
    with pytest.raises(ValueError, match="below p"):
        K.encode_word(f.p, 0, k)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_masked_round_trip(name, rounds):
    plain, k = K.key(name, SEED, rounds), K.key(name, SEED, rounds, mask=True)
    f = k.field
    assert k.c == plain.c and k.tau == plain.tau and plain.mask is None
    assert k.mask.bit_length() <= f.n and abs(k.mask.bit_count() - f.n / 2) < f.n / 8
    rng = random.Random(rounds)
    for i, m in enumerate(_edges(f) + [rng.randrange(f.p)]):
        y = K.encode_word(m, 1000 + i, k)
        assert 0 <= y < f.p and K.decode_word(y, 1000 + i, k) == m
        assert (y == K.encode_word(m, 1000 + i, plain)) == (rounds == 1)


def test_sigma_is_an_involution_on_the_field():
    k = K.key("p512", SEED, 2, mask=True)
    p = k.field.p
    rejected = p ^ k.mask
    near = [rejected, rejected ^ 1, (p - 1) ^ k.mask, 0, p - 1, k.mask % p]
    for x in near + [random.Random(3).randrange(p) for _ in range(64)]:
        x %= p
        s = k.sigma(x)
        assert 0 <= s < p and k.sigma(s) == x
    assert rejected < p and k.sigma(rejected) == rejected


@pytest.mark.parametrize("name", NAMES)
def test_every_word_of_finite_bf16_or_e4m3_is_a_payload(name):
    f = K.field(name)
    s = f.word_bytes
    assert f.p >> (f.n - 16) == 0xFFFF
    assert int.from_bytes(b"\xff" * (s - 2) + b"\x7f\xff", "little") < f.p
    assert int.from_bytes(b"\xff" * (s - 1) + b"\xfe", "little") < f.p


@pytest.mark.parametrize("name", NAMES)
def test_token_layer_layout(name):
    k = K.key(name, SEED, 1)
    f = k.field
    assert K.TOKEN_LAYER_BYTES == 4096 and K.TOKEN_BYTES == 327_680
    assert K.token_layer_words(f) == {"p512": 64, "p768": 43, "p1024": 32, "p1280": 26, "p1536": 22}[name]
    assert K.token_words(f) == K.LAYERS * K.token_layer_words(f)
    rng = random.Random(7)
    data = bytes(rng.randrange(256) for _ in range(K.TOKEN_LAYER_BYTES))
    words = K.encode_token_layer(data, 3, 17, k)
    w0 = (3 * K.LAYERS + 17) * K.token_layer_words(f)
    assert words[1] == K.encode_word(K.payloads(data, f)[1], w0 + 1, k)
    assert K.decode_token_layer(words, 3, 17, k) == data
    with pytest.raises(ValueError, match="token-layer"):
        K.payloads(data[:-1], f)


@pytest.mark.parametrize("name", NAMES)
def test_twin_squaring_and_folds_over_the_lazy_range(name):
    f = K.field(name)
    rng = random.Random(f.n + 1)
    top = (1 << f.n) - 1
    for x in [top, top - 1, f.p, f.p + 1, f.p - 1, 0, 1, 1 << (f.n - 1)] + [rng.randrange(1 << f.n) for _ in range(6)]:
        y = K.from_limbs(T.sqr(K.to_limbs(x, f), f))
        assert y < 1 << f.n and y % f.p == x * x % f.p
        z = rng.randrange(1 << f.n)
        assert K.from_limbs(T.mul(K.to_limbs(x, f), K.to_limbs(z, f), f)) % f.p == x * z % f.p
        assert K.from_limbs(T.canon(K.to_limbs(x, f), f)) == x % f.p


@pytest.mark.parametrize("name", NAMES)
def test_twin_add_sub_neg_and_tweak(name):
    f = K.field(name)
    rng = random.Random(f.n + 2)
    edges = [0, 1, f.p - 1, f.p - 2, f.c, (f.p - 1) // 2]
    for a in edges + [rng.randrange(f.p)]:
        for b in edges + [rng.randrange(f.p)]:
            la, lb = K.to_limbs(a, f), K.to_limbs(b, f)
            assert K.from_limbs(T.add_mod(la, lb, f)) == (a + b) % f.p
            assert K.from_limbs(T.sub_mod(la, lb, f)) == (a - b) % f.p
        if a:
            assert K.from_limbs(T.neg(K.to_limbs(a, f), f)) == f.p - a
    tau = K.key(name, SEED, 1).tau
    for i in [0, 1, (1 << 32) - 1, 1 << 32, (1 << 64) - 1, rng.randrange(1 << 64)]:
        assert K.from_limbs(T.tweak(i, K.to_limbs(tau, f), f)) == tau * i % f.p
    big = K.to_limbs(f.p - 1, f)
    assert K.from_limbs(T.tweak((1 << 64) - 1, big, f)) == (f.p - 1) * ((1 << 64) - 1) % f.p


def _takes_rare(f: K.Field, op: int, a: int, b: int) -> bool:
    m32, n, c = (1 << 32) - 1, f.n, f.c
    if op == 0:
        return bool((a + b) >> n) and ((a + b) & m32) + c > m32
    if op == 1:
        return a < b and ((a - b) & m32) < c
    if op == 2:
        return bool(b & 1) and (a & m32) > (1 << 32) - c
    if op == 3:
        return a >> (n - 32) == m32
    if op == 5:
        return (a ^ b) >> (n - 32) == m32
    s1 = (a * a & ((1 << n) - 1)) + c * (a * a >> n)
    return (s1 & m32) + (s1 >> n) * c > m32


@pytest.mark.parametrize("name", ["p512", "p1536"])
def test_twin_primitives_at_their_rare_branches(name):
    """The kernel's fast paths leave a carry chain to a branch when it carries or borrows past limb 0 (or canon's
    top limb is all ones); the device gate (kv_exhaust's prim_gate) runs these same inputs through kv_prim."""
    f = K.field(name)
    cases = T.rare_cases(f, random.Random(f.n + 7))
    for op, rows in cases.items():
        assert sum(_takes_rare(f, op, a, b) for a, b in rows) >= 4, T.PRIMS[op]
        for a, b in rows:
            assert T.prim_twin(f, op, a, b) == T.prim_want(f, op, a, b), (T.PRIMS[op], a, b)
    sq = [a for a, _ in cases[4]]
    assert all(K.from_limbs(T.sqr(K.to_limbs(a, f), f)) % f.p == a * a % f.p for a in sq)


@pytest.mark.parametrize("name", NAMES)
def test_twin_root_is_the_chain(name):
    f = K.field(name)
    a = random.Random(f.n + 3).randrange(f.p)
    x, sq, mu = T.root(K.to_limbs(a, f), f)
    assert K.from_limbs(x) % f.p == pow(a, f.root_exp, f.p)
    assert (sq, mu) == (f.root_squarings, f.root_multiplies)


@pytest.mark.parametrize("name", NAMES)
def test_twin_encode_and_decode_equal_the_reference(name):
    k = K.key(name, SEED, 2)
    f = k.field
    for i, m in [(0, 0), (5, f.p - 1), ((1 << 40) + 3, random.Random(f.n).randrange(f.p))]:
        y = K.encode_word(m, i, k)
        assert T.encode_word(m, i, k.c, k.tau, f) == y
        assert T.decode_word(y, i, k.c, k.tau, f) == m


def _vectors() -> dict:
    out = {}
    for name in NAMES:
        for rounds in (1, 2):
            k = K.key(name, SEED, rounds)
            rng = random.Random(f"{name}/{rounds}")
            words = [(i, rng.randrange(k.field.p)) for i in (0, 1, 1 << 33)]
            out[f"{name}/k{rounds}"] = {
                "c": [format(v, "x") for v in k.c], "tau": format(k.tau, "x"),
                "words": [{"i": i, "payload": format(m, "x"), "word": format(K.encode_word(m, i, k), "x")}
                          for i, m in words]}
    return {"seed": SEED.hex(), "domain": K.DOMAIN.decode(), "vectors": out}


def test_pinned_vectors():
    assert json.loads(VECTORS.read_text()) == _vectors()


if __name__ == "__main__":
    VECTORS.write_text(json.dumps(_vectors(), indent=1) + "\n")
