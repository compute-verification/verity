"""In-test stand-ins for the partition tests: two small Programs, `gemm_program()` (`Gemm_v1{K=32,N=6}`: Input batches,
a composite batch, nested calls) and `wide_program()`."""
from __future__ import annotations

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Tuple, Value

from verity_vllm.definitions.registry import b1
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.prims import F32


# ---------------------------------------------------------------------------------------------------------
# fixture programs
# ---------------------------------------------------------------------------------------------------------


def gemm_program() -> Program:
    """Root: Input16 x (32), Input16 w (192), batch GemmCoordinate_v1{K=32} x 6 -> 248 gates.
    GemmCoordinate body: call DotBf16 (3 gates: Const32 + 2 Ampere steps), F2fpBf16."""
    return Program(bind(b1.Gemm, K=32, N=6))


_Wide64 = PrimitiveDefinition("QpartWide64", 1, (), Value(64), lambda: 0, register=False)


def wide_program() -> Program:
    """A Program with a 64-bit Input and a 64-bit primitive gate (readiness violators of both kinds)."""
    W = CompositeDefinition("QpartWide", 1, (), lambda S: ((("seed", Value(64)),), Tuple(Value(64), F32)),
                            lambda B, S, seed: tuple_of(B.call(_Wide64), B.call(P.F32Add, B.call(b1.ZERO32), B.call(b1.ZERO32))),
                            register=False)
    return Program(bind(W))
