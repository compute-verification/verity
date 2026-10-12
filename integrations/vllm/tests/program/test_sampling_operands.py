"""[X-06] `verity_vllm.program.registry.sampling`: stochastic token select with seed / pos / temperature as OPERANDS.  CPU: agreement with the
RTX-4090-validated static-seed composites (`GumbelArgmaxBf16_v1`, `GumbelArgmaxTempBf16_v1`), the kernel's early returns, masked lanes,
and the plain-argmax limit.  The kernel correspondence itself is `registry/sampling_difftest.py` (G4, produce on the pod)."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.properties.admission import evaluate_spec
from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import sampling as S
from verity_vllm.program.registry.sampling import (
    GumbelArgmaxBf16,
    GumbelArgmaxTempBf16,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _tables_ok() -> bool:
    try:
        from verity_vllm.program.registry.prims import DivFullRcp
        DivFullRcp.evaluate(S.f32_bits(0.8))
        return True
    except Exception:  # noqa: BLE001  (RmsUnresolved: MUFU tables not on this machine)
        return False


needs_tables = pytest.mark.skipif(not _tables_ok(), reason="MUFU tables (rms_relation.tables) unavailable: div.full model needs them")


def _bf16_row(V: int, seed: int = 0, scale: float = 2.0) -> np.ndarray:
    f = (np.random.default_rng(seed).standard_normal(V) * scale).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def _f32_of_bf16(w: np.ndarray) -> np.ndarray:
    return (w.astype(np.uint32) << 16).view(np.float32)


def w64(x: int) -> np.ndarray:
    return np.array([x & S.M64], dtype=np.uint64)


def w32(x: int) -> np.ndarray:
    return np.array([x & S.M32], dtype=np.uint32)


def select(logits: np.ndarray, seed: int, pos: int, temp: float) -> int:
    V = int(logits.size)
    return evaluate_spec(bind(S.GumbelTokenSelect, V=V), [logits, w64(seed), w64(pos), w32(S.f32_bits(temp))])[0]


@needs_tables
@pytest.mark.parametrize("seed,pos", [(12345, 46), (0, 0), (2**62 - 1, 2**31), (7, 1151)])
def test_temperature_one_equals_the_validated_static_seed_definition(seed, pos):
    x = _bf16_row(1000, seed=1)
    ref = evaluate_spec(bind(GumbelArgmaxBf16, V=1000, SEED=seed, POS=pos), [x])[0]
    assert select(x, seed, pos, 1.0) == ref


@needs_tables
@pytest.mark.parametrize("temp", [0.8, 0.7, 0.05, 2.0])
def test_tempered_equals_the_validated_static_temperature_definition(temp):
    x = _bf16_row(1000, seed=2)
    tb = S.f32_bits(temp)
    ref = evaluate_spec(bind(GumbelArgmaxTempBf16, V=1000, SEED=12345, POS=46, T=tb), [x])[0]
    assert select(x, 12345, 46, temp) == ref


@needs_tables
def test_temperature_zero_is_the_greedy_first_maximum():
    x = _bf16_row(1000, seed=3)
    f = _f32_of_bf16(x)
    assert select(x, 99, 5, 0.0) == int(np.argmax(f))
    # ties: the FIRST maximum (tl.max tie_break_left + torch.argmax first)
    x2 = x.copy(); m = int(np.argmax(f)); x2[m + 7] = x2[m]; x2[m - 3] = x2[m]
    assert select(x2, 99, 5, 0.0) == m - 3
    # -0.0 == 0.0: the kernel's `temperature == 0.0` early return holds for the negative zero word too
    assert evaluate_spec(bind(S.GumbelTokenSelect, V=1000), [x, w64(99), w64(5), w32(0x80000000)])[0] == int(np.argmax(f))


@needs_tables
def test_seed_and_pos_are_inputs_not_structure():
    """The same specialization (V only) serves every request: seed and pos change the token through the operands."""
    x = np.zeros(1024, dtype=np.uint16)                       # all-equal logits: the token is the argmax of the noise alone
    toks = {select(x, s, p, 1.0) for s, p in [(1, 0), (2, 0), (1, 1), (3, 5), (2**40, 0)]}
    assert len(toks) >= 4
    assert bind(S.GumbelTokenSelect, V=1024).id == bind(S.GumbelTokenSelect, V=1024).id
    # deterministic: the same (seed, pos) draws the same noise
    assert select(x, 1, 0, 1.0) == select(x, 1, 0, 1.0)
    assert S.stream_key(1, 0) != S.stream_key(2, 0) != S.stream_key(1, 1)


@needs_tables
def test_masked_lanes_never_win_and_f32_entry_composes():
    x = _bf16_row(1000, seed=4)
    x32 = (x.astype(np.uint32) << 16)
    V = 1000
    best = evaluate_spec(bind(S.GumbelSelectF32, V=V), [x32, w64(1), w64(2), w32(S.f32_bits(1.0))])[0]
    masked = x32.copy(); masked[best] = 0xFF800000                                     # -inf as the top-p mask writes it
    b2 = evaluate_spec(bind(S.GumbelSelectF32, V=V), [masked, w64(1), w64(2), w32(S.f32_bits(1.0))])[0]
    assert b2 != best
    everything = np.full(V, 0xFF800000, dtype=np.uint32)
    assert evaluate_spec(bind(S.GumbelSelectF32, V=V), [everything, w64(1), w64(2), w32(S.f32_bits(1.0))])[0] == 0
    # composition = the chained Definition
    scaled = evaluate_spec(bind(S.TemperatureScale, V=V), [x, w32(S.f32_bits(0.8))])
    via = evaluate_spec(bind(S.GumbelSelectF32, V=V), [np.array(scaled, dtype=np.uint32), w64(5), w64(6), w32(S.f32_bits(0.8))])[0]
    assert via == select(x, 5, 6, 0.8)


@needs_tables
def test_temperature_scale_early_returns_and_div_full():
    from verity_vllm.program.registry.prims import DivFullRcp, DivFullScaleA, F32Mul
    x = _bf16_row(64, seed=5)
    xf = (x.astype(np.uint32) << 16)
    for t in (0.0, 1.0):
        out = evaluate_spec(bind(S.TemperatureScale, V=64), [x, w32(S.f32_bits(t))])
        assert np.array_equal(np.array(out, dtype=np.uint32), xf)                     # the widening only (kernel early return)
    tb = S.f32_bits(0.8)
    out = evaluate_spec(bind(S.TemperatureScale, V=64), [x, w32(tb)])
    rcp = DivFullRcp.evaluate(tb)
    want = [F32Mul.evaluate(DivFullScaleA.evaluate(int(v), tb), rcp) for v in xf]
    assert out == want


def test_libdevice_and_philox_helpers_are_the_validated_ones():
    """The libdevice logs and the Philox stream key at their fixed points."""
    assert S.nv_logf(S.f32_bits(1.0)) == 0 and S.nv_log1pf(0) == 0
    assert S.stream_key(0, 0) == S.philox4x32_10_word0(0, 0, 0, 0, 0, 0)
