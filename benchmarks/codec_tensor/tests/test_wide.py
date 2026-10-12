"""The wide-word codec's reference (`wide.py`), its tensor-core decomposition and lane carries, and `wt.cu`'s constants."""

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

import wide as W  # noqa: E402

SEED = hashlib.sha256(b"verity/codec-tensor/vectors").digest()
TC_FIELDS = [name for name, f in W.FIELDS.items() if W.tc_ok(f)]


def _probably_prime(n: int) -> bool:
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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


def test_fields():
    W.check_fields()
    assert sorted(W.FIELDS) == sorted(W.TASK_WORDS)
    assert TC_FIELDS == ["p255", "p511", "p767", "p1023", "m1279"]
    for f in W.FIELDS.values():
        assert _probably_prime(f.p)
        assert not any(_probably_prime((1 << f.n) - c) for c in range(1, f.c, 4)), f"{f.name}: c is the smallest"
        if W.tc_ok(f):
            assert f.n % 256 == 255 and (1 << (8 * f.word_bytes)) % f.p == 2 * f.c
    assert [n for n, f in W.FIELDS.items() if f.full_slot] == ["p512", "p768"]
    for name in ("p512", "p768"):
        f = W.field(name)
        top = int.from_bytes(b"\xff" * (f.word_bytes - 1) + b"\xfe", "little")
        assert f.payload_bytes == f.word_bytes and top < f.p, "S weights fit unless the last is 0xFF"
    assert ((1 << 255) - 19) % 4 == 1
    with pytest.raises(ValueError, match="unknown field"):
        W.field("p256")


def test_kernel_constants_match():
    src = (HERE.parent / "wt.cu").read_text()
    for f in W.FIELDS.values():
        m = re.search(rf"#(?:el)?if WT_FIELD == {f.n}\nconstexpr int NBITS = {f.n}, TW = (\d+);\nconstexpr u32 CF = (\d+);",
                      src)
        assert m, f.name
        assert (int(m[1]), int(m[2])) == (W.TASK_WORDS[f.name], f.c)
    rounds = re.search(r"constexpr int SHUF_ROUNDS = (.*);", src)[1]
    assert rounds == " : ".join(f"U == {u} ? {r}" for u, r in list(W.SHUF_ROUNDS.items())[:-1]) + \
        f" : {W.SHUF_ROUNDS[5]}"
    assert "constexpr int TASK_BYTES = 8192;" in src and W.TASK_BYTES == 8192


@pytest.mark.parametrize("n,c", [(3, 1), (4, 5), (5, 1), (5, 9), (6, 21), (7, 1), (13, 1)])
def test_rho_is_a_permutation_with_the_parity_rule(n, c):
    f = W.Field("small", n, c)
    p = f.p
    assert p % 4 == 3 and _probably_prime(p)
    images = [W.rho(a, f) for a in range(p)]
    assert sorted(images) == list(range(p))
    for a, y in enumerate(images):
        assert W.rho_inv(y, f) == a
        assert (y % 2 == 0) == (pow(a, (p - 1) // 2, p) != p - 1)


@pytest.mark.parametrize("name", list(W.FIELDS))
def test_rho_on_the_fields(name):
    f = W.field(name)
    rng = random.Random(name)
    for a in [0, 1, 2, f.p - 1, f.p - 2, (f.p - 1) // 2] + [rng.randrange(f.p) for _ in range(20)]:
        y = W.rho(a, f)
        assert W.rho_inv(y, f) == a
        assert (y % 2 == 0) == (pow(a, (f.p - 1) // 2, f.p) != f.p - 1)
    with pytest.raises(ValueError, match="canonical"):
        W.rho(f.p, f)
    with pytest.raises(ValueError, match="canonical"):
        W.rho_inv(-1, f)


def test_key_derivation():
    k = W.key("p767", SEED, 2)
    assert k.rounds == 2 and all(0 <= c < k.field.p for c in k.c) and 0 < k.tau < k.field.p
    assert W.key("p767", SEED, 3).c[:2] == k.c and W.key("p767", SEED, 3).tau == k.tau
    assert W.key("p1023", SEED, 2).c != k.c
    assert k.tweak(0) == 0 and k.tweak(5) == 5 * k.tau % k.field.p
    with pytest.raises(ValueError, match="seed"):
        W.key("p767", SEED[:31], 1)
    with pytest.raises(ValueError, match="round"):
        W.key("p767", SEED, 0)


@pytest.mark.parametrize("name", list(W.FIELDS))
@pytest.mark.parametrize("rounds", [1, 2])
def test_word_round_trip(name, rounds):
    k = W.key(name, SEED, rounds)
    f = k.field
    rng = random.Random(f"{name}/{rounds}")
    top = f.max_payload
    for m, i in [(0, 0), (top, 1), (1, (1 << 40) + 3)] + [(rng.randrange(top + 1), rng.randrange(1 << 32)) for _ in range(8)]:
        y = W.encode_word(m, i, k)
        assert 0 <= y < f.p
        assert W.bytes_word(W.word_bytes(y, f), f) == y and len(W.word_bytes(y, f)) == f.word_bytes
        assert W.decode_word(y, i, k) == m
    with pytest.raises(ValueError, match="payload"):
        W.encode_word(top + 1, 0, k)
    if f.full_slot:
        return
    for _ in range(4096):
        y = rng.randrange(f.p)
        try:
            W.decode_word(y, 7, k)
        except ValueError as e:
            assert "no payload" in str(e)
            break
    else:
        pytest.fail("every random word decoded")


@pytest.mark.parametrize("name", ["p255", "m521", "m1279", "p512", "p768"])
def test_tasks_round_trip(name):
    k = W.key(name, SEED, 1)
    f = k.field
    plain = random.Random(name).randbytes(2 * W.TASK_BYTES)
    store = W.encode_tasks(plain, k, start_task=3)
    assert len(store) == 2 * W.TASK_WORDS[name] * f.word_bytes
    assert W.decode_tasks(store, k, start_task=3) == plain
    assert W.decode_tasks(store[:len(store) // 2], k, start_task=3) == plain[:W.TASK_BYTES]
    first = W.task_payloads(plain[:W.TASK_BYTES], f)
    assert W.payloads_task(first, f) == plain[:W.TASK_BYTES]
    assert W.bytes_word(store[:f.word_bytes], f) == W.encode_word(first[0], 3 * W.TASK_WORDS[name], k)
    with pytest.raises(ValueError, match="whole tasks"):
        W.encode_tasks(plain[:100], k)


def test_vectors():
    """Pinned words: the format (XOF, key derivation, tweak, ρ) as the device gates check it."""
    vectors = json.loads((HERE / "vectors.json").read_text())
    assert vectors["seed"] == SEED.hex()
    assert {v["field"] for v in vectors["words"]} == set(W.FIELDS)
    for v in vectors["words"]:
        k = W.key(v["field"], SEED, v["rounds"])
        f = k.field
        m = int.from_bytes(bytes.fromhex(v["payload"]), "little")
        y = W.encode_word(m, v["index"], k)
        assert W.word_bytes(y, f).hex() == v["word"]
        assert W.decode_word(y, v["index"], k) == m


def test_tc_groups():
    assert [W.tc_groups(u) for u in range(1, 6)] == [[0], [0], [0], [0], [0, 4]]
    assert [W.tc_mmas(u) for u in range(1, 6)] == [4, 6, 8, 10, 12]
    for u in range(1, 9):
        for w, g in W.tc_plan(u):
            cb = W.tc_groups(u)[g]
            assert cb <= w and w + u <= cb + 8
    with pytest.raises(ValueError):
        W.tc_groups(9)


@pytest.mark.parametrize("u", [1, 2, 3, 4, 5])
def test_tc_square_coeffs(u):
    rng = random.Random(u)
    for limbs in ([255] * 32 * u, [0] * 32 * u, [0xAA, 0x55] * 16 * u, [rng.randrange(256) for _ in range(32 * u)],
                  [rng.randrange(256) for _ in range(32 * u - 5)]):
        coeffs = W.tc_square_coeffs(limbs, u)
        ref = [0] * (64 * u)
        for i, x in enumerate(limbs):
            for j, y in enumerate(limbs):
                ref[i + j] += x * y
        assert coeffs == ref
        a = sum(x << (8 * i) for i, x in enumerate(limbs))
        assert W.square_from_coeffs(coeffs) == a * a
    with pytest.raises(ValueError):
        W.tc_square_coeffs([256], u)


def _edges(f: W.Field) -> list[int]:
    s = 8 * f.word_bytes
    return [0, 1, 2, f.p - 1, f.p, f.p + 1, (1 << f.n) - 1, 1 << f.n, (1 << s) - 1, (1 << s) - 2,
            int.from_bytes(b"\xaa" * f.word_bytes, "little"), int.from_bytes(b"\x55" * f.word_bytes, "little"),
            int.from_bytes(b"\xff\x00" * (f.word_bytes // 2), "little"), (1 << s) - (1 << (s - 8 * (f.word_bytes // 32)))]


@pytest.mark.parametrize("name", TC_FIELDS)
def test_tc_square_lanes(name):
    f = W.field(name)
    rng = random.Random(name)
    for a in _edges(f) + [rng.randrange(1 << (8 * f.word_bytes)) for _ in range(12)]:
        v = W.tc_square_lanes(a, f)
        assert v < 1 << (8 * f.word_bytes) and v % f.p == a * a % f.p
    with pytest.raises(AssertionError):
        W.tc_square_lanes(1 << (8 * f.word_bytes), f)


@pytest.mark.parametrize("name", TC_FIELDS)
def test_tc_canon_lanes(name):
    f = W.field(name)
    rng = random.Random(name)
    for v in _edges(f) + [rng.randrange(1 << (8 * f.word_bytes)) for _ in range(200)] + \
            [f.p + rng.randrange(1 << 20) for _ in range(20)]:
        assert W.tc_canon_lanes(v, f) == v % f.p


def test_tc_squarings_chain():
    """Repeated lane squarings stay < 2^(8S) and track a^(2^j) mod p, the decode's and the root chain's use."""
    for name in TC_FIELDS:
        f = W.field(name)
        a = int.from_bytes(b"\xff" * f.word_bytes, "little")
        ref = a % f.p
        for _ in range(4):
            a = W.tc_square_lanes(a, f)
            ref = ref * ref % f.p
            assert W.tc_canon_lanes(a, f) == ref
