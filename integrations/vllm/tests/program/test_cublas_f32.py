"""`registry.cublas_f32`: the word Definitions agree with their Boolean versions word for word on random words and specials, and with
`cublas_f32_check`'s numpy models of the sm_120 kernels on the probe's kind of inputs (bf16-exact x, general f32 weights)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.evaluation import batch as E
from verity.evaluation import bits as BS
from verity.ir.defs import bind
from verity_vllm.program.frontend.rules.vllm_bindings import f32_linear as FL
from verity_vllm.program.registry import cublas_f32 as C
from verity_vllm.program.registry import cublas_f32_check as CK

U16, U32 = np.uint16, np.uint32
F32_SPECIALS = np.array([0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
                         0x7FC00000, 0xFFC00001, 0x3F800000, 0xBF800000], U32)
BF16_SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0x3F80, 0xBF80], U16)


def _u32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(U32)


def _operands(rng, B: int, K: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """x [B, K] bf16 words around 1 (rows 0 and 1 sprinkled with specials, row 2 zeros), w [B, N, K] f32 words around 2^-5 (rows
    0 and 1 sprinkled with specials); the other rows finite, so their outputs are finite sums that differ by tree."""
    x = (_u32(rng.standard_normal((B, K)).astype(np.float32)) >> 16).astype(U16)
    x[:2, ::7] = BF16_SPECIALS[rng.integers(0, len(BF16_SPECIALS), x[:2, ::7].shape)]
    x[2] = 0
    w = _u32((rng.standard_normal((B, N, K)) / 32).astype(np.float32))
    w[:2, :, 3::11] = F32_SPECIALS[rng.integers(0, len(F32_SPECIALS), w[:2, :, 3::11].shape)]
    return x, w


@pytest.mark.parametrize("K, tree", [(256, "gemv"), (32, "simt")])
def test_word_equals_the_boolean(K: int, tree: str) -> None:
    rng = np.random.default_rng(K)
    x, w = _operands(rng, 24, K, 2)
    got = np.asarray(E.evaluate_batch(bind(C.CublasF32LinearWord, K=K, N=2, TREE=tree), x, w)).astype(U32)
    want = np.asarray(BS.evaluate_words(bind(C.CublasF32Linear, K=K, N=2, TREE=tree), x, w)).astype(U32)
    assert np.array_equal(got, want)
    finite = (got & 0x7F800000) != 0x7F800000
    assert finite[2:].mean() > 0.5 and not finite[:2].all()


@pytest.mark.parametrize("tree", C.TREES)
def test_word_equals_the_numpy_model(tree: str) -> None:
    """At K = 512 (two gemv groups), on bf16-exact x (a zero row against all-negative weights, where the +0 start shows) and on
    23-bit f32 weights, where FMA and mul + add differ."""
    rng = np.random.default_rng(7)
    K, N = 512, 3
    x = (CK.probe_inputs(6)[:, :K]).copy()
    w = np.concatenate([CK.probe_weights("f32")[:2, :K], CK.probe_weights("neg")[:1, :K]])
    w[0] *= np.float32(2.0) ** rng.integers(-3, 4, K).astype(np.float32)
    want = CK.MODELS[tree](x, w)
    assert _u32(want)[0, 2] == 0
    got = np.asarray(E.evaluate_batch(bind(C.CublasF32LinearWord, K=K, N=N, TREE=tree), (_u32(x) >> 16).astype(U16),
                                      np.ascontiguousarray(np.broadcast_to(_u32(w), (len(x), N, K))))).astype(U32)
    assert np.array_equal(got, _u32(want))
    other = CK.MODELS[{"gemv": "simt", "simt": "gemv"}[tree]](x, w)
    assert not np.array_equal(_u32(other), _u32(want))


def test_bad_statics_are_refused() -> None:
    with pytest.raises(ValueError, match="K_multiple_of_256"):
        _ = bind(C.CublasGemvDotWord, K=320).body
    with pytest.raises(ValueError, match="TREE_known"):
        _ = bind(C.CublasF32LinearWord, K=32, N=2, TREE="tensorop").body


def test_kernel_table_is_one_tree_per_m() -> None:
    for (K, N), ranges in FL.F32_LINEAR_KERNELS.items():
        assert all(tree in C.TREES and lo <= hi for lo, hi, _, tree in ranges)
        spans = sorted((lo, hi) for lo, hi, _, _ in ranges)
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))
    assert FL.cublas_f32_kernel(2048, 64, 1)[1] == "gemv" and FL.cublas_f32_kernel(2048, 64, 256)[1] == "simt"
    assert FL.cublas_f32_kernel(2048, 64, 6) is None and FL.cublas_f32_kernel(2048, 32, 1) is None
