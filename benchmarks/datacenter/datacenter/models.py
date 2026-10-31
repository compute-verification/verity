"""Small Programs, built with verity's constructive front end, for the demo and the tests.

They are Boolean (`verity.ml.circuits.boolean`): a BF16 word is its 16 bits (`BF16`, least significant first), and every
stage is a registered Boolean Definition. `LINEAR` is `Gemm_v3` at K = N = 32 over the BF16 tensor-core step on bits
(`ops.boolean.gemm.GEMM_AMPERE`, `AmpereBF16TcDot16_v3`, the step sm_120 runs too), x[32] @ W[32, 32]^T, 1,024
multiply-adds and 32 roundings to BF16; `DOT` is one of its coordinates, a dot product of two 32-word rows (PoUW's noised
product, in miniature, when one row is a trusted device's fresh data). Each computes the word Definition it names
(`Gemm_v2`, `GemmCoordinate_v2`), whose kernel evaluates it. `s8_grid` is a served s8 matmul on bits that PoUW's
compiler takes. `program` defines a Program's function from a body over named parameters, and `words` gives a parameter
seeded BF16 words, one per element.
"""
from __future__ import annotations

import random
import struct
from collections.abc import Callable

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean.gemm import GEMM_AMPERE, Gemm, GemmCoordinate
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Type


def word(w: int) -> Type:
    """A w-bit word on bits, least significant first: one element."""
    return Array(w, BL.BIT)


K = 32
BF16 = word(16)
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
    """`n` BF16 words sampled uniformly from [-scale, scale], seeded by `seed`."""
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


def s8_grid(r: int = 4, c: int = 16, k: int = 64) -> Program:
    """A served s8 matmul on bits, the shape PoUW's sm_120 scheme (`eps1-sm120-h1fx1`) tiles: each of r rows of x against
    each of c rows of the registered w, a `GemmAccumulatorS8_v2` over the s8 IMMA step's Boolean form."""
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.refs import Coll, concat

    row = Array(k, word(8))
    acc = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)

    def body(B, x, w):
        xs = Coll(Array(r * c, row), concat([x[i].refs for i in range(r) for _ in range(c)]))
        ws = Coll(Array(r * c, row), concat([w[j].refs for _ in range(r) for j in range(c)]))
        return B.batch(acc, xs, ws, axes=(0, 0))

    return program("S8Grid", {"x": Array(r, row), "w": Array(c, row)}, Array(r * c, word(32)), body, registered=("w",))
