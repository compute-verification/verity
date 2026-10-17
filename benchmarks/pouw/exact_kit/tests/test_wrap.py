import numpy as np
import pytest
from exact_kit import wrap


def test_u_and_s_on_ints_and_arrays():
    xs = [0, 1, -1, 2**31, -2**31, 2**31 - 1, 2**32, 2**32 + 7, -2**33 - 3]
    for k in (1, 7, 8, 31, 32):
        for x in xs:
            assert wrap.u(x, k) == x % (1 << k)
            assert wrap.s(x, k) == (x + (1 << (k - 1))) % (1 << k) - (1 << (k - 1))
        arr = np.array(xs, dtype=np.int64)
        assert wrap.u(arr, k).tolist() == [x % (1 << k) for x in xs]
        assert wrap.s(arr, k).tolist() == [wrap.s(x, k) for x in xs]
    assert wrap.s32(2**31) == -2**31 and wrap.u32(-1) == 2**32 - 1
    assert wrap.u(np.array([-1], dtype=np.int64), 64).tolist() == [2**64 - 1]
    assert wrap.u(np.array([-1], dtype=np.int64), 63).tolist() == [2**63 - 1]
    with pytest.raises(ValueError, match="above 32 bits"):
        wrap.s(np.array([1]), 40)


def test_rotations():
    rng = np.random.default_rng(1)
    for k in (8, 13, 32):
        xs = rng.integers(0, 1 << k, 50, dtype=np.uint64)
        for r in (0, 1, 5, k - 1, k, k + 3):
            got = wrap.rotl(xs, r, k)
            want = [int(wrap.rotl(int(x), r, k)) for x in xs]
            assert got.tolist() == want
            assert wrap.rotr(got, r, k).tolist() == xs.tolist()
            for x in xs[:5]:
                bits = format(int(x), f"0{k}b")
                rr = r % k
                assert wrap.rotl(int(x), r, k) == int(bits[rr:] + bits[:rr], 2)


def test_v2():
    assert [wrap.v2(x) for x in (1, 2, 12, -8, 2**40)] == [0, 1, 2, 3, 40]
    with pytest.raises(ValueError):
        wrap.v2(0)
