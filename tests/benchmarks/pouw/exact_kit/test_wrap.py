import numpy as np
from exact_kit import wrap


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
