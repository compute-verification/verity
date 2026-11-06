import random

from verity_lin import code, field as F


def test_cubic_is_irreducible_and_the_two_adic_root_has_full_order():
    assert pow(F.CUBIC, (F.P - 1) // 3, F.P) != 1
    w = code.root_of_unity(1 << F.TWO_ADICITY)
    assert pow(w, 1 << F.TWO_ADICITY, F.P) == 1 and pow(w, 1 << (F.TWO_ADICITY - 1), F.P) != 1


def test_extension_arithmetic():
    rng = random.Random(1)
    for _ in range(20):
        a = tuple(rng.randrange(F.P) for _ in range(3))
        b = tuple(rng.randrange(F.P) for _ in range(3))
        assert F.mul(a, F.inv(a)) == F.ONE
        assert F.mul(a, b) == F.mul(b, F.mul(F.ONE, a))
        assert F.mul(F.add(a, b), b) == F.add(F.mul(a, b), F.mul(b, b))


def test_eq_table_is_the_multilinear_basis():
    rng = random.Random(2)
    point = [tuple(rng.randrange(F.P) for _ in range(3)) for _ in range(3)]
    table = F.eq_table(point)
    assert F.total(table) == F.ONE
    for x in range(8):
        bits = [F.embed((x >> i) & 1) for i in range(3)]
        assert F.eq_table(bits)[x] == F.ONE and F.total(F.eq_table(bits)) == F.ONE


def test_one_encoding_per_element():
    assert F.from_bytes(F.to_bytes(F.P - 1)) == [F.P - 1]
    assert F.from_bytes((F.P).to_bytes(8, "little")) is None
    assert F.from_bytes(b"\0" * 7) is None
