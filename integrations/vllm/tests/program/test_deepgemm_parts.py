"""`registry.deepgemm`'s ``Fp8GroupQuantUe8m0_v1{K,128}`` and `fp8`'s ``ScaledMmFp8BlockSharedScale_v1{K,N,128}``, the parts
`pipeline.serving_templates.PARTS` splits ``DeepGemmFp8Linear_v1{K,N}``'s rows into, and that GEMM's coordinates (``COORDS``): the
linear computes what they compose to (its q and s the quantisation's, its y the GEMM's on them), and the GEMM's word n is its
N = 1 form on ``w[n]`` and ``ws[n // 128]``, word for word (nonfinite words included), on activation-like rows and rows that
reach the quantiser's guards, at N one whole 128-wide weight block and a whole and a partial one.  The linear against its twin
is `test_deepgemm.test_linears_match_their_twins`."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.pipeline import serving_templates as ST
from verity.ml.definitions.numerics import deepgemm_twins as T
from verity.ml.definitions.registry import deepgemm as DG
from verity.ml.definitions.registry import fp8 as F8
from verity.ml.definitions.registry.fp8 import FP8_BLOCK
from tests.program.test_deepgemm import _bf16_rows, _guard_rows, _weights, _words

K = 256


@pytest.mark.parametrize("N", [128, 200])
def test_the_linear_is_its_quantisation_then_its_gemm_and_the_gemm_its_coordinates(N: int) -> None:
    assert ST.PART_STATICS["DeepGemmFp8Linear_v1"] == {"G": FP8_BLOCK}
    rng = np.random.default_rng(N)
    NB, KG = -(-N // FP8_BLOCK), K // FP8_BLOCK
    w, ws = _weights(rng, N, K), T.upcast_e8m0(rng.integers(112, 124, (NB, KG)))
    linear = bind(DG.DeepGemmFp8Linear, K=K, N=N)
    quant = bind(DG.Fp8GroupQuantUe8m0, K=K, G=FP8_BLOCK)
    gemm = bind(F8.ScaledMmFp8BlockSharedScale, K=K, N=N, G=FP8_BLOCK)
    coord = bind(F8.ScaledMmFp8BlockSharedScale, K=K, N=1, G=FP8_BLOCK)
    guards = _guard_rows(rng, K)
    for x in [*_bf16_rows(rng, 2, K, 3.0), guards[1], guards[4], guards[10]]:
        got = list(evaluate(linear, _words(x), _words(w), _words(ws)))
        qs = list(evaluate(quant, _words(x)))
        q, s = qs[:K], qs[K:]
        y = list(evaluate(gemm, q, s, _words(w), _words(ws)))
        assert got == qs + y
        assert [evaluate(coord, q, s, _words(w[n]), _words(ws[n // FP8_BLOCK]))[0] for n in range(N)] == y
