"""P2-FF r9: reference twins of the two receipt folds below duo, in pfill's lane-chain form, for FILL-SEC r4.

One pfill step absorbs a lane's four words (d0, d1, d2, d3) = rows (g, g + 8) x columns (2t, 2t + 1), in that
order, into a 32-bit state y (uint32 arithmetic, wrapping):

  XOR-in (rotations 13, 19; 6 SASS a step: SHF, LOP3, IMAD, SHF, LOP3, IMAD):
    y = (rotl(y, 13) ^ d0) * a0 + d1
    y = (rotl(y, 19) ^ d2) * a1 + d3
  LOP3+IMAD (4 SASS a step: LOP3, IMAD, LOP3, IMAD):
    y = (y ^ d0 ^ k0) * a0 + d1
    y = (y ^ d2 ^ k1) * a1 + d3

Keys: a0, a1 must pass key_ok_r9 (pfill's key_ok, plus no int8 ratio within 2^30 of ±a); k0, k1 are free words.
Run: uv run --no-project --with numpy python -B receipt_folds_r9.py   (checks the twins against fullfloor_r9.chain)
"""
import sys

import numpy as np

sys.dont_write_bytecode = True
M32 = 0xFFFFFFFF


def key_ok(a: int) -> bool:
    """pfill's key condition (pfill_ref_bu.key_ok)."""
    m, r = a & 0xFFFFFF, a & 31
    return bool(a & 1) and 256 < m < (1 << 24) - 256 and r not in (1, 31)


def int8_ratio(lam: int, bits: int = 30) -> bool:
    """lam b = b' mod 2^bits for some int8 b != 0 and b': a merge one combined int8 row or column computes."""
    return any(((lam * b + 128) % (1 << bits)) < 256 for b in range(-128, 128) if b)


def key_ok_r9(a: int) -> bool:
    """key_ok, and neither a nor -a is an int8 ratio mod 2^30, so no int8 merge sits at a bad lam of either fold.
    (XOR-in's other key-derived bad lam, ±a1 2^19, is never an int8 ratio: it needs b = 0 mod 2^11.)"""
    return key_ok(a) and not int8_ratio(a) and not int8_ratio((-a) & M32)


def rotl(x: np.ndarray, r: int) -> np.ndarray:
    x = x.astype(np.uint32)
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def xorin_step(y, d0, d1, d2, d3, a0, a1):
    y = (rotl(y, 13) ^ d0) * a0 + d1
    return (rotl(y, 19) ^ d2) * a1 + d3


def lop3imad_step(y, d0, d1, d2, d3, a0, a1, k0, k1):
    y = (y ^ d0 ^ k0) * a0 + d1
    return (y ^ d2 ^ k1) * a1 + d3


if __name__ == "__main__":
    sys.path.insert(0, "/tmp/p2ff")
    import fullfloor_r9 as ff

    rng = np.random.default_rng(13)
    while True:
        a0, a1, k0, k1 = (int(x) for x in rng.integers(0, 1 << 32, 4, dtype=np.uint64))
        if key_ok_r9(a0) and key_ok_r9(a1):
            break
    W = rng.integers(-(1 << 20), 1 << 20, (4096, 6, 4))
    with np.errstate(over="ignore"):
        for kind, step in (("XOR-in r13", lambda y, d: xorin_step(y, *d, np.uint32(a0), np.uint32(a1))),
                           ("LOP3+IMAD", lambda y, d: lop3imad_step(y, *d, *(np.uint32(v) for v in (a0, a1, k0, k1))))):
            y = np.full(4096, 0x1234567, np.uint32)
            for s in range(W.shape[1]):
                y = step(y, [(W[:, s, c] & M32).astype(np.uint32) for c in range(4)])
            ref = ff.chain(kind, W, tuple(np.uint64(v) for v in (a0, a1, k0, k1))) & ff.M32
            assert np.array_equal(y.astype(np.uint64), ref), kind
            print(f"[PASS] {kind} twin = fullfloor_r9.chain over 4096 x 6 steps")
    rej = sum(not key_ok_r9(int(a)) for a in rng.integers(0, 1 << 32, 4096, dtype=np.uint64) if key_ok(int(a)))
    print(f"[PASS] key_ok_r9 rejects {rej} of the key_ok keys among 4096 draws (expected ~2^-13 of them)")
    assert int8_ratio(1) and int8_ratio((-128) & M32) and not int8_ratio(1 << 13) and not int8_ratio(1 << 19)
    print("[PASS] int8_ratio: ±1 and ±128 are int8 ratios, 2^13 and 2^19 are not")
