"""Small Programs, built with verity's constructive front end, for the demo and the tests.

Every stage is a registered Definition: `LINEAR` is `Gemm_v2` at K = N = 32 over the BF16 tensor-core step (`GEMM_AMPERE`,
`AmpereBF16TcDot16_v2`, the step sm_120 runs too), x[32] @ W[32, 32]^T, 1,024 multiply-adds and 32 roundings to BF16;
`DOT` is one of its coordinates, a dot product of two 32-word rows (PoUW's noised product, in miniature, when one row is a
trusted device's fresh data). `program` defines a Program's function from a body over named parameters, and `words` gives
a parameter seeded BF16 words.
"""
from __future__ import annotations

import random
import struct
from collections.abc import Callable

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.gemm import GEMM_AMPERE, Gemm, GemmCoordinate
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Type, Value

K = 32
BF16 = Value(16)
VEC = Array(K, BF16)
MAT = Array(K, VEC)
LINEAR = bind(Gemm, K=K, N=K, **GEMM_AMPERE)
DOT = bind(GemmCoordinate, K=K, **GEMM_AMPERE)


def program(name: str, params: dict[str, Type], ret: Type, body: Callable, registered: tuple[str, ...] = ()) -> Program:
    """The Program of `body(B, *params)`, a Builder and one collection per parameter, returning `ret`."""
    fn = CompositeDefinition(name, 1, (), lambda S: (tuple(params.items()), ret), lambda B, S, *a: body(B, *a),
                             register=False)
    return Program(fn, registered=registered)


def bf16(x: float) -> int:
    """`x` rounded toward zero to BF16's encoding."""
    return struct.unpack(">I", struct.pack(">f", x))[0] >> 16


def words(seed: str, n: int, scale: float = 1.0) -> list[int]:
    """`n` BF16 words drawn uniformly from [-scale, scale], seeded by `seed`."""
    rng = random.Random(f"datacenter/{seed}")
    return [bf16(rng.uniform(-scale, scale)) for _ in range(n)]


def two_stage(name: str = "TwoStage") -> Program:
    """x → h = x @ W1ᵀ → y = h @ W2ᵀ, the weights registered: the walkthrough's model and the tests'."""
    def body(B, x, w1, w2):
        h = B.call(LINEAR, x, w1, name="h")
        return B.call(LINEAR, h, w2, name="y")

    return program(name, {"x": VEC, "w1": MAT, "w2": MAT}, VEC, body, registered=("w1", "w2"))


WEIGHTS = {"w1": words("w1", K * K, 0.3), "w2": words("w2", K * K, 0.3)}
X = words("x", K)
