"""The Definitions whose program values are inputs (eps, the attention scale) compute what their constant-holding twins compute,
and hold none of those values as constants."""

from __future__ import annotations

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.constants import program_data
from verity.ir.defs import bind
from verity.ir.program import Program
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P


def _bf16(rng, n: int, scale: float = 1.0) -> list[int]:
    return [int(P.bits_of(np.float32(v))) >> 16 for v in rng.standard_normal(n).astype(np.float32) * scale]


def _flat(x) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


def _held(spec) -> set[str]:
    return {cid for held in program_data(Program(spec).root).values() for _g, cid in held}


def _const_id(bits: int) -> str:
    return f"Const32[0x{bits:08x}]_v1"


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_rmsnorm_triton_eps_input(eps):
    rng = np.random.default_rng(1)
    N, eps_bits = 64, P.bits_of(np.float32(eps))
    old, new = bind(b1.RMSNormTriton, N=N, EPS=eps), bind(b1.RMSNormTritonV3, N=N)
    assert new.id == "RMSNormTriton_v3{N=64}"
    for _ in range(3):
        x, w = _bf16(rng, N), _bf16(rng, N)
        assert _flat(evaluate(new, x, w, eps_bits)) == _flat(evaluate(old, x, w))
    assert _const_id(eps_bits) in _held(old)
    assert _const_id(eps_bits) not in _held(new)


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_rmsnorm_fused_cuda_eps_input(eps):
    rng = np.random.default_rng(2)
    N, eps_bits = 64, P.bits_of(np.float32(eps))
    old, new = bind(b1.RMSNormFusedCuda, N=N, EPS=eps), bind(b1.RMSNormFusedCudaV4, N=N)
    assert new.id == "RMSNormFusedCuda_v4{N=64}"
    for _ in range(3):
        x, res, w = _bf16(rng, N), _bf16(rng, N), _bf16(rng, N)
        assert _flat(evaluate(new, x, res, w, eps_bits)) == _flat(evaluate(old, x, res, w))
    assert _const_id(eps_bits) in _held(old)
    assert _const_id(eps_bits) not in _held(new)


def test_attention_scale_input():
    """Two key blocks (T = BN + 4): the second block rescales, so every use of the scale is exercised."""
    rng = np.random.default_rng(3)
    T, NH, KVH, D, BN = 20, 2, 1, 16, 16
    scale = b1._scale_log2_bits(D)
    old = bind(b1.AttentionV3, T=T, NH=NH, KVH=KVH, D=D, BN=BN)
    new = bind(b1.AttentionV10, T=T, NH=NH, KVH=KVH, D=D, BN=BN)
    for _ in range(2):
        q, kc, vc = _bf16(rng, NH * D, 0.5), _bf16(rng, T * KVH * D, 0.5), _bf16(rng, T * KVH * D)
        assert _flat(evaluate(new, q, kc, vc, scale)) == _flat(evaluate(old, q, kc, vc))
    assert _const_id(scale) in _held(old)
    assert _const_id(scale) not in _held(new)
