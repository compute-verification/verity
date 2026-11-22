"""[X-06] `verity_vllm.definitions.registry.sampling`: stochastic token select with seed / pos / temperature as OPERANDS.  CPU: agreement with the
RTX-4090-validated static-seed composites (`GumbelArgmaxBf16_v1`, `GumbelArgmaxTempBf16_v1`), the kernel's early returns, masked lanes,
and the plain-argmax limit.  The kernel correspondence itself is `registry/sampling_difftest.py` (G4, produce on the pod)."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.definitions.registry import sampling as S

pytestmark = pytest.mark.filterwarnings("ignore")


def _tables_ok() -> bool:
    try:
        from verity_vllm.definitions.registry.prims import DivFullRcp
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


def test_libdevice_and_philox_helpers_are_the_validated_ones():
    """The libdevice logs and the Philox stream key at their fixed points."""
    assert S.nv_logf(S.f32_bits(1.0)) == 0 and S.nv_log1pf(0) == 0
    assert S.stream_key(0, 0) == S.philox4x32_10_word0(0, 0, 0, 0, 0, 0)
