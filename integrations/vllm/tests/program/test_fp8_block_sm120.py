"""`ScaledMmFp8Block_v2{K,N,G,DOT}`: the blockwise FP8 linear with the e4m3 k32 step as a static.  On the RTX PRO 6000 (cc 12.0,
`VLLM_USE_DEEP_GEMM=0`) vLLM's `CutlassFp8BlockScaledMMKernel` is `ScaledMmFp8Block_v1`'s schedule over `BlackwellE4m3QmmaDot32_v1`.

The fixture `data/fp8_block_sm120_operands.npz` holds served words of that kernel (run r20260930-164401-c04d, checked in full by
r20260930-164854-5f2c): 64 output columns of one qkv weight block (K = 2560) at M = 1, 65 and 257, one row of each of the three sm120
blockwise dispatch configs (swap_ab 128x32x128, pingpong 64x128x128, 128x128x128), with the served quantizer's words.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.defs import bind
from verity.ml.prims import BlackwellE4m3QmmaDot32, HopperE4m3QgmmaDot32
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
from verity_vllm.program.kernels import derived_rows as D
from verity_vllm.program.kernels import rows as RK
from verity_vllm.program.registry import fp8
from verity_vllm.program.registry import targets as TG

FIX = Path(__file__).parent / "data" / "fp8_block_sm120_operands.npz"
G = 128
MS = (1, 65, 257)
SM120 = DEFAULT_TARGET.with_overrides(compute_capability=(12, 0), num_sms=188)
H100 = DEFAULT_TARGET.with_overrides(compute_capability=(9, 0), num_sms=132)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIX)


def _block(fx):
    """The fixture's 64 weight rows as one 128-row weight block (zero rows after them) and its scale row."""
    w = np.zeros((G, fx["w"].shape[1]), dtype=np.uint8)
    w[:fx["w"].shape[0]] = fx["w"]
    return w, fx["ws"].reshape(1, -1)


@pytest.mark.parametrize("M", MS)
def test_the_quantizer_row_is_the_served_words(fx, M):
    for r in range(fx[f"x_{M}"].shape[0]):
        xq, sx = D.fp8_group_quant_row(fx[f"x_{M}"][r], G=G)
        assert xq.tolist() == fx[f"xq_{M}"][r].tolist() and sx.astype(np.uint32).tolist() == fx[f"xs_{M}"][r].tolist()


def test_the_sm120_step_row_is_the_served_gemm_words_and_hoppers_is_not(fx):
    w, ws = _block(fx)
    n = fx["w"].shape[0]
    hopper_off = 0
    for M in MS:
        for r in range(fx[f"xq_{M}"].shape[0]):
            ops = (fx[f"xq_{M}"][r], fx[f"xs_{M}"][r], w, ws)
            got = RK.scaled_mm_fp8_block_dot({"K": w.shape[1], "N": G, "G": G, "DOT": BlackwellE4m3QmmaDot32}, ops)["0"][:n]
            assert got.tolist() == fx[f"y_{M}"][r].tolist(), (M, r)
            hop = RK.scaled_mm_fp8_block({"K": w.shape[1], "N": G, "G": G}, ops)["0"][:n]
            hopper_off += int((hop != fx[f"y_{M}"][r]).sum())
    assert hopper_off > 0


def test_the_definition_is_the_served_word(fx):
    w, ws = _block(fx)
    coord = bind(fp8.ScaledMmFp8BlockCoordinateV2, K=w.shape[1], G=G, DOT=BlackwellE4m3QmmaDot32)
    for c in (0, 1, 63):
        got = evaluate(coord, fx["xq_65"][0].tolist(), fx["xs_65"][0].tolist(), w[c].tolist(), ws[0].tolist())
        got = got[0] if isinstance(got, (list, tuple)) else got
        assert int(got) & 0xFFFF == int(fx["y_65"][0][c]), c


def test_v2_on_hoppers_step_is_v1():
    rng = np.random.default_rng(5)
    K = 256
    v1, v2 = bind(fp8.ScaledMmFp8BlockCoordinate, K=K, G=G), bind(fp8.ScaledMmFp8BlockCoordinateV2, K=K, G=G, DOT=HopperE4m3QgmmaDot32)
    for _ in range(4):
        xq, wrow = (rng.integers(0, 0x7F, K).astype(int) | (rng.integers(0, 2, K) << 7)).tolist(), rng.integers(0, 0x7E, K).tolist()
        sx, sw = (rng.random(K // G).astype(np.float32) * 1e-2).view(np.uint32).tolist(), (rng.random(K // G).astype(np.float32)).view(np.uint32).tolist()
        assert evaluate(v1, xq, sx, wrow, sw) == evaluate(v2, xq, sx, wrow, sw)


def test_the_target_records_the_step():
    assert TG.fp8_block_dot(SM120) is BlackwellE4m3QmmaDot32 and TG.fp8_block_dot(H100) is None


class _Ctx:
    """The derive context `CutlassScaledMmFp8BlockRule.emit` reads (as in test_fp8_shared_scale.py)."""

    def __init__(self):
        from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3
        self.vocab, self.calls, self.records = PROFILE_B1_EAGER_V3, [], []

    def kind(self, role, node, **statics):
        return self.vocab.kind(role, **statics)

    def call(self, fn, *args, **kw):
        self.calls.append(fn)
        return fn

    def coll(self, v):
        return v

    def hint_name(self, node):
        return "model.layers.0.self_attn.qkv_proj"

    def bind_weight_role(self, *a, **kw):
        pass

    def rebind(self, *a, **kw):
        pass

    def record(self, rule, node, facts, emitted):
        self.records.append((facts, emitted))


@pytest.mark.parametrize("tp,scale", [(SM120, "shared-per-block"), (SM120, "per-coordinate"), (H100, "shared-per-block")])
def test_the_frontend_rule_binds_the_targets_step(monkeypatch, tp, scale):
    from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, observe_target_profile
    from verity_vllm.program.frontend.rules.vllm_bindings import fp8 as R

    K, N = 256, 384
    monkeypatch.setattr(R, "shape", lambda ctx, v: (K, N))
    monkeypatch.setattr(R, "_untransposed", lambda ctx, v, a, b, name: name)
    monkeypatch.setattr(R, "rows_of", lambda coll: ([coll], ()))
    monkeypatch.setattr(R, "from_rows", lambda outs, lead: outs)
    clear_observations()
    try:
        observe_target_profile(tp.with_overrides(fp8_scale_construction=scale))
        ctx = _Ctx()
        R.CutlassScaledMmFp8BlockRule().emit(types.SimpleNamespace(args=("out", "a", "b", "a_s", "b_s")), ctx)
    finally:
        clear_observations()
    if tp is SM120:
        defn = fp8.ScaledMmFp8BlockV2 if scale == "per-coordinate" else fp8.ScaledMmFp8BlockSharedScaleV2
        want = bind(defn, K=K, N=N, G=G, DOT=BlackwellE4m3QmmaDot32)
    else:
        want = bind(fp8.ScaledMmFp8BlockSharedScale, K=K, N=N, G=G)
    assert ctx.calls == [want]
    (facts, _), = ctx.records
    assert facts["fp8_dot"] == (BlackwellE4m3QmmaDot32.id if tp is SM120 else None)
