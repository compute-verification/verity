"""The Build's lowering of the batch-invariant Triton GEMM, on a real `torch.export` of a small model, and the PoUW row
Definitions (`pouw_rows`).

The model has three linears. Each launches the batch-invariant Triton GEMM as vLLM's `linear_batch_invariant` does (the
export's carrier `verity_cba::triton_launch`, as in `test_gemm_bias.py`): qkv with its bias added after the GEMM, o without
a bias, and an lm_head.

- The model is `GemmBias_v1`, `Gemm_v1`, `Gemm_v1`.
- `NcpLinearRowBias_v1` is `BiasAdd_v1` after `NcpLinearRow_v1`, bit for bit.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate

from verity_vllm.definitions.registry import b1, pouw_rows
from verity_vllm.program.frontend.rules import triton_iface as _ti  # noqa: F401  (defines verity_cba::triton_launch)
from verity_vllm.program.frontend.rules.vllm_bindings import (
    clear_observations,
    gemm_pins_for,
    vllm_ruleset,
)
from verity_vllm.program.frontend.vllm_derive import derive_report

BF = torch.bfloat16
M, K, NQ, V = 3, 64, 64, 16
PROF = dict(mode="profile", profile="b1-eager-v3")


def _gemm_meta(m: int, n: int, k: int) -> str:
    pins, _src = gemm_pins_for(m, n, k)
    scalars = {"M": m, "N": n, "K": k, "stride_am": k, "stride_ak": 1, "stride_bk": 1, "stride_bn": k, "stride_cm": n, "stride_cn": 1,
               "NUM_SMS": pins["NUM_SMS"], **pins["constexprs"]}
    return json.dumps({"kernel": pins["kernel"], "src_sha256": pins["src_sha256"], "triton": "test", "grid": [1], "scalars": scalars,
                       "tensor_args": ["a_ptr", "b_ptr", "c_ptr"], "constexprs": list(pins["constexprs"]),
                       "geometry": {"num_warps": pins["num_warps"], "num_stages": pins["num_stages"]}, "tensor_meta": [None, None, None]})


def _gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`matmul_persistent(x, W.t())` into a fresh `c`, as the export records it."""
    c = torch.empty(x.shape[0], w.shape[0], dtype=BF)
    meta = _gemm_meta(x.shape[0], w.shape[0], w.shape[1])
    torch.ops.verity_cba.triton_launch(json.loads(meta)["kernel"], [x, w.t(), c], meta)
    return c


def _param(g: torch.Generator, *shape: int, scale: float = 0.3) -> nn.Parameter:
    return nn.Parameter((torch.randn(*shape, generator=g) * scale).to(BF), requires_grad=False)


class QKVParallelLinear(nn.Module):
    def __init__(self, k: int, n: int, g: torch.Generator) -> None:
        super().__init__()
        self.weight, self.bias = _param(g, n, k), _param(g, n, scale=1.0)

    def forward(self, x):
        return _gemm(x, self.weight) + self.bias


class RowParallelLinear(nn.Module):
    def __init__(self, k: int, n: int, g: torch.Generator) -> None:
        super().__init__()
        self.weight = _param(g, n, k)

    def forward(self, x):
        return _gemm(x, self.weight)


class ParallelLMHead(nn.Module):
    def __init__(self, k: int, n: int, g: torch.Generator) -> None:
        super().__init__()
        self.weight = _param(g, n, k)

    def forward(self, x):
        return _gemm(x, self.weight)


class Tiny(nn.Module):
    def __init__(self, seed: int = 3) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.qkv, self.o, self.lm_head = QKVParallelLinear(K, NQ, g), RowParallelLinear(NQ, K, g), ParallelLMHead(K, V, g)

    def forward(self, x):
        return self.lm_head(self.o(self.qkv(x)))


def _derive(m: nn.Module):
    clear_observations()
    try:
        return derive_report(m, (torch.empty(M, K, dtype=BF),), **PROF, name=type(m).__name__, rules=vllm_ruleset())
    finally:
        clear_observations()


def _hist(program) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in program.fn.body.nodes:
        fam = n.fn.id.split("{")[0]
        out[fam] = out.get(fam, 0) + 1
    return out


def test_the_model_keeps_its_gemms():
    rep = _derive(Tiny())
    assert rep.ok, rep.unsupported
    assert _hist(rep.program) == {"GemmBias_v1": M, "Gemm_v1": 2 * M}, _hist(rep.program)


@pytest.mark.slow
def test_the_fused_row_is_bias_add_after_the_row_bit_for_bit():
    rng = np.random.default_rng(4)
    fused, row, add = bind(pouw_rows.NcpLinearRowBias, K=K, N=NQ), bind(pouw_rows.NcpLinearRow, K=K, N=NQ), bind(b1.BiasAdd, N=NQ)

    def bf16(n, scale):
        return (np.asarray(rng.standard_normal(n) * scale, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    for scale in (1e-3, 1.0, 50.0, 1e-39, 0.0):
        x, w = bf16(K, scale), bf16(NQ * K, 0.5)
        biases: tuple[np.ndarray, ...] = (bf16(NQ, 1.0), np.full(NQ, 0x8000, np.uint16), np.zeros(NQ, np.uint16))
        for b in biases:
            got = list(evaluate(fused, [int(v) for v in x], [int(v) for v in w], [int(v) for v in b]))
            assert got == list(evaluate(add, list(evaluate(row, [int(v) for v in x], [int(v) for v in w])), [int(v) for v in b])), scale
