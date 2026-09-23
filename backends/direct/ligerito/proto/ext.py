"""F_{p^6} = F_p[x] / (x^6 - BETA) over BabyBear, vectorised in numpy: an extension element is an int64 array whose LAST
axis has length 6 (coordinate d = the coefficient of x^d), every coordinate in [0, p).  BETA = 31 is the multiplicative
generator of F_p, hence neither a square nor a cube; 6 | p - 1, so x^6 - 31 is irreducible.

Products reduce every coordinate product mod p before summing (a coordinate product is < 2^62, six of them would
overflow int64).  Only the small host-side objects (sumcheck tables, claim coefficients, opened rows) use these; the
bulk device work is base-field (kernels / fp64 limb matmuls) and combines coordinates with ``mul_planes``.
"""
from __future__ import annotations

import numpy as np

P = 2013265921          # 2^31 - 2^27 + 1
BETA = 31
D = 6

__all__ = ["BETA", "D", "P", "add", "embed", "eq_pairs", "from_base", "inv", "mul", "mul_base", "neg", "one", "pow_int",
           "reduce_coords", "scalar_mul", "sub", "sum_ext", "zero"]


def zero(shape=()) -> np.ndarray:
    return np.zeros(tuple(shape) + (D,), dtype=np.int64)


def one(shape=()) -> np.ndarray:
    z = zero(shape)
    z[..., 0] = 1
    return z


def from_base(a) -> np.ndarray:
    """Base-field array (...,) -> extension array (..., 6) with zero high coordinates."""
    a = np.asarray(a, dtype=np.int64) % P
    z = np.zeros(a.shape + (D,), dtype=np.int64)
    z[..., 0] = a
    return z


embed = from_base


def add(a, b):
    return (a + b) % P


def sub(a, b):
    return (a - b) % P


def neg(a):
    return (-np.asarray(a, dtype=np.int64)) % P


def mul(a, b) -> np.ndarray:
    """(..., 6) x (..., 6) -> (..., 6), numpy broadcasting on the leading axes."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    out = np.zeros(shape + (D,), dtype=np.int64)
    for i in range(D):
        ai = a[..., i]
        for j in range(D):
            prod = (ai * b[..., j]) % P
            k = i + j
            if k >= D:
                out[..., k - D] += prod * BETA
            else:
                out[..., k] += prod
    return out % P


def mul_base(a, s) -> np.ndarray:
    """Extension (..., 6) times base scalar/array (...,) (broadcast on the leading axes)."""
    s = np.asarray(s, dtype=np.int64) % P
    return (np.asarray(a, dtype=np.int64) * s[..., None]) % P


scalar_mul = mul_base


def reduce_coords(a):
    return np.asarray(a, dtype=np.int64) % P


def sum_ext(a, axis=0):
    """Sum of extension elements along ``axis`` (coordinates < 2^31; up to 2^32 terms fit in int64)."""
    return np.sum(np.asarray(a, dtype=np.int64), axis=axis) % P


def pow_int(a, e: int) -> np.ndarray:
    """a^e for a single extension element (6,) by square-and-multiply."""
    result = one()
    base = np.asarray(a, dtype=np.int64) % P
    while e:
        if e & 1:
            result = mul(result, base)
        base = mul(base, base)
        e >>= 1
    return result


def inv(a) -> np.ndarray:
    """a^{p^6 - 2} (Fermat); scalar use only."""
    return pow_int(a, P**D - 2)


def eq_pairs(z: np.ndarray) -> np.ndarray:
    """(nvars, 6) point -> (nvars, 2, 6) pairs ((1 - z_j), z_j): the tensor factors of the multilinear eq vector."""
    z = np.asarray(z, dtype=np.int64) % P
    pairs = np.zeros((z.shape[0], 2, D), dtype=np.int64)
    pairs[:, 1] = z
    pairs[:, 0] = (-z) % P
    pairs[:, 0, 0] = (pairs[:, 0, 0] + 1) % P
    return pairs
