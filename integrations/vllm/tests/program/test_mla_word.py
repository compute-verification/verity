"""`registry.mla_word`: the word twins of `registry.mla`'s Boolean MLA decode and absorb bmm agree with the Boolean Definitions word
for word on random words and on the specials that reach every branch (the exp's small-argument path, the gate's subnormal and
non-finite sums, signed zeros, NaNs), and with `mla_check`'s numpy models of the sm_120 kernels on whole requests."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from verity.evaluation import batch as E
from verity.evaluation import bits as BS
from verity.ir.defs import bind
from verity.ml import mufu as MU
from verity.ml import prims as P
from verity.ml.boolean import gemm as BG
from verity_vllm.program.registry import mla as M
from verity_vllm.program.registry import mla_check as MK
from verity_vllm.program.registry import mla_word as MW

from tests.program.test_mla import F32_SPECIALS, _bf16_words, _f32_words

U16, U32 = np.uint16, np.uint32
SCALE = M.GLM47_FLASH["SCALE"]


def _words(target, *inputs) -> np.ndarray:
    return np.asarray(E.evaluate_batch(target, *inputs))


def _bits(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs))


def test_exp_sub_gate_and_out_equal_the_boolean() -> None:
    """`TritonExpSubWord_v1`, `TritonMlaGateWord_v1` and `TritonMlaOutWord_v1` against their Boolean twins on every pair of
    specials and 40k random words (sums around 1..4096 for the gate and the output, as the kernel makes them)."""
    rng = np.random.default_rng(11)
    gx, gm = (F32_SPECIALS[g] for g in np.array(list(itertools.product(range(len(F32_SPECIALS)), repeat=2))).T)
    x, m = np.concatenate([_f32_words(rng, 20_000), gx]), np.concatenate([_f32_words(rng, 20_000), gm])
    got = _words(bind(MW.TritonExpSubWord), x, m).astype(U32)
    want = _bits(bind(M.TritonExpSub), x, m).astype(U32)
    assert np.array_equal(got, want)
    assert np.count_nonzero(MK.f32(MK.mul(MK.sub(x, m), MK.LOG2E)) < -126) > 1000
    e_sum = np.concatenate([MK.u32(rng.uniform(1, 4096, 20_000).astype(np.float32)), rng.integers(0, 1 << 32, 20_000,
                            dtype=np.uint64).astype(U32), gm])
    e_max = np.concatenate([MK.u32(rng.uniform(-50, 50, 40_000).astype(np.float32)), gx])
    gate = _words(bind(MW.TritonMlaGateWord), e_max, e_sum).reshape(-1)
    assert np.array_equal(gate, _bits(bind(M.TritonMlaGate), e_max, e_sum).reshape(-1))
    assert 0.3 < gate.mean() < 0.9
    acc = MK.u32((rng.standard_normal(len(e_sum)) * 8).astype(np.float32))
    acc[::97] = F32_SPECIALS[np.arange(len(acc[::97])) % len(F32_SPECIALS)]
    rcp = MU.mufu_rcp_words(MU.div_full_divisor_words(e_sum), MU.table("rcp"))
    got = _words(bind(MW.TritonMlaOutWord), acc, e_sum, rcp, gate.astype(np.uint8)).astype(U16)
    assert np.array_equal(got, _bits(bind(M.TritonMlaOut), acc, e_sum, rcp, gate.astype(np.uint8)).astype(U16))


def _word_decode(T: int, H: int, DL: int, DR: int, scale: int = SCALE):
    return bind(MW.TritonMlaDecodeWord, H=H, T=T, DL=DL, DR=DR, BN=32, SPLITS=1, SCALE=scale, DOT=P.HopperBF16WgmmaDot16)


@pytest.mark.parametrize("T,kind", [(1, "spread"), (2, "special"), (33, "large"), (33, "special"), (64, "spread"),
                                    (65, "uniform")])
def test_decode_agrees_with_the_model(T: int, kind: str) -> None:
    """`TritonMlaDecodeWord_v1` against `mla_check.decode_rows` (one split): one block, masked last blocks, full blocks, the
    exp's small-argument path (``large``) and sprinkled specials, at a 16 + 16 latent and two heads."""
    rng = np.random.default_rng([T, len(kind), 7])
    n, H, DL, DR = 48, 2, 16, 16
    q, kv = _bf16_words(rng, (n, H, DL + DR), kind), _bf16_words(rng, (n, T, DL + DR), kind)
    got = _words(_word_decode(T, H, DL, DR), q, kv).astype(U16)
    want = np.stack([MK.decode_rows(q[i], kv[i], 1, SCALE, DL)["o"] for i in range(n)])
    bad = np.argwhere(got != want)
    assert not len(bad), f"T={T} {kind}: {len(bad)} words differ, first {bad[0]}"


def test_decode_equals_the_boolean_on_specials() -> None:
    """The word decode and the Boolean `TritonMlaDecode_v1` on the same requests, specials sprinkled, two blocks."""
    rng = np.random.default_rng(12)
    n, H, T, DL, DR = 32, 1, 33, 16, 16
    q, kv = _bf16_words(rng, (n, H, DL + DR), "special"), _bf16_words(rng, (n, T, DL + DR), "special")
    got = _words(_word_decode(T, H, DL, DR), q, kv).astype(U16)
    boolean = bind(M.TritonMlaDecode, H=H, T=T, DL=DL, DR=DR, BN=32, SPLITS=1, SCALE=SCALE, DOT=BG.HopperBF16WgmmaDot16)
    assert np.array_equal(got, _bits(boolean, q, kv).astype(U16))


def test_bmm_agrees_with_the_model() -> None:
    """`MlaBmmWord_v1` against `bmm_kernel`'s model (`mla_check.bmm_rows`) at two and three heads, specials in the weights."""
    rng = np.random.default_rng(13)
    for NH, K, N in ((2, 32, 3), (3, 192, 5)):
        n = 32
        x, w = _bf16_words(rng, (n, NH, K), "spread"), _bf16_words(rng, (n, NH, N, K), "special")
        got = _words(bind(MW.MlaBmmWord, NHEADS=NH, K=K, N=N, DOT=P.HopperBF16WgmmaDot16), x, w).astype(U16)
        want = np.stack([MK.bmm_rows(x[i][:, None, :], np.transpose(w[i], (0, 2, 1)))[:, 0] for i in range(n)])
        assert np.array_equal(got, want)


def test_the_word_decode_rejects_what_it_does_not_model() -> None:
    with pytest.raises(ValueError, match="SPLITS_is_1"):
        _ = bind(MW.TritonMlaDecodeHeadWord, T=2, DL=16, DR=16, BN=32, SPLITS=2, SCALE=SCALE, DOT=P.HopperBF16WgmmaDot16).body
    with pytest.raises(ValueError, match="BN_is_32"):
        _ = bind(MW.TritonMlaDecodeHeadWord, T=2, DL=16, DR=16, BN=16, SPLITS=1, SCALE=SCALE, DOT=P.HopperBF16WgmmaDot16).body
