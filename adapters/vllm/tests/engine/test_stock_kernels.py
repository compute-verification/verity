"""`engine.stock_kernels`: each baseline calls its vLLM kernel as vLLM does. Over stand-ins of vLLM's modules, the calls are
the ones named; where vLLM is installed, each call binds to the signature of the kernel it names."""
from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest
import torch

from verity_vllm import config
from verity_vllm.engine import stock_kernels as K

#: each recorded call's name -> the module of vLLM's that defines it
KERNELS = {
    "scaled_int8_quant": "vllm._custom_ops",
    "triton_scaled_mm": "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm",
    "per_token_group_quant_fp8": "vllm.model_executor.layers.quantization.utils.fp8_utils",
    "fp8_gemm_nt": "vllm.utils.deep_gemm",
    "m_grouped_fp8_gemm_nt_contiguous": "vllm.utils.deep_gemm",
    "fp8_m_grouped_gemm_nt_masked": "vllm.utils.deep_gemm",
    "flash_attn_varlen_func": "vllm.vllm_flash_attn",
}
RETURNS = {"scaled_int8_quant": ("q", "s", None), "triton_scaled_mm": "y", "per_token_group_quant_fp8": ("xq", "xs")}


def stand_ins(monkeypatch: pytest.MonkeyPatch) -> list:
    """Stand-ins of the modules in `KERNELS` (and of ``torch.ops._C``), each call recorded as ``(name, args, kwargs)``."""
    log: list = []

    def recorder(name: str):
        def call(*args, **kwargs):
            log.append((name, args, kwargs))
            return kwargs["out"] if name == "flash_attn_varlen_func" else RETURNS.get(name)
        return call

    mods = {n: types.ModuleType(n) for n in {"vllm", "vllm.utils", *KERNELS.values()}}
    for name, mod in KERNELS.items():
        setattr(mods[mod], name, recorder(name))
    mods["vllm"]._custom_ops, mods["vllm.utils"].deep_gemm = mods["vllm._custom_ops"], mods["vllm.utils.deep_gemm"]
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setattr(torch.ops, "_C", types.SimpleNamespace(cutlass_scaled_mm=recorder("cutlass_scaled_mm")),
                        raising=False)
    return log


def drive() -> tuple:
    """Every baseline once, on stand-in operands; what each returned."""
    q = torch.zeros(2, 1, 16)
    return (K.scaled_int8_quant("x"), K.triton_scaled_mm("a", "b", "sa", "sb"),
            K.triton_scaled_mm("a", "b", "sa", "sb", torch.float16, "bias"), K.cutlass_scaled_mm("out", "a", "b", "sa", "sb"),
            K.per_token_group_quant_fp8("x", column_major_scales=True, use_ue8m0=True),
            K.fp8_gemm_nt(("aq", "as"), ("bq", "bs"), "out"),
            K.m_grouped_fp8_gemm_nt_contiguous(("aq", "as"), ("bq", "bs"), "out", "ids"),
            K.fp8_m_grouped_gemm_nt_masked(("aq", "as"), ("bq", "bs"), "out", "masked", 7),
            K.flash_attn_varlen(q, "k", "v", out="o", cu_seqlens_q="cu", max_seqlen_q=2, seqused_k="used", max_seqlen_k=9,
                                block_table="bt", fa_version=3))


def test_each_baseline_makes_the_call_it_names(monkeypatch):
    log = stand_ins(monkeypatch)
    assert drive() == (("q", "s"), "y", "y", "out", ("xq", "xs"), None, None, None, "o")
    flash = {"q": log[-1][2]["q"], "k": "k", "v": "v", "out": "o", "cu_seqlens_q": "cu", "max_seqlen_q": 2,
             "seqused_k": "used", "max_seqlen_k": 9, "block_table": "bt", "softmax_scale": 0.25, "causal": True,
             "fa_version": 3, "num_splits": 0}
    assert log == [
        ("scaled_int8_quant", ("x",), {}),
        ("triton_scaled_mm", ("a", "b", "sa", "sb", torch.bfloat16, None), {}),
        ("triton_scaled_mm", ("a", "b", "sa", "sb", torch.float16, "bias"), {}),
        ("cutlass_scaled_mm", ("out", "a", "b", "sa", "sb", None), {}),
        ("per_token_group_quant_fp8", ("x", 128),
         {"column_major_scales": True, "tma_aligned_scales": False, "use_ue8m0": True}),
        ("fp8_gemm_nt", (("aq", "as"), ("bq", "bs"), "out"), {}),
        ("m_grouped_fp8_gemm_nt_contiguous", (("aq", "as"), ("bq", "bs"), "out", "ids"), {}),
        ("fp8_m_grouped_gemm_nt_masked", (("aq", "as"), ("bq", "bs"), "out", "masked", 7), {}),
        ("flash_attn_varlen_func", (), flash),
    ]


def test_importing_it_imports_no_vllm(monkeypatch):
    for name in [m for m in sys.modules if m.split(".")[0] == "vllm"]:
        monkeypatch.delitem(sys.modules, name)
    importlib.reload(K)
    assert not [m for m in sys.modules if m.split(".")[0] == "vllm"]


def test_each_call_binds_to_its_kernel_at_the_installed_vllm(monkeypatch):
    vllm = pytest.importorskip("vllm")
    assert config.VLLM in vllm.__version__, f"vLLM {vllm.__version__} is installed, not the pin {config.VLLM}"
    real = {name: getattr(importlib.import_module(mod), name) for name, mod in KERNELS.items()}
    importlib.import_module("vllm._custom_ops")
    cutlass = torch.ops._C.cutlass_scaled_mm.default._schema
    log = stand_ins(monkeypatch)
    drive()
    for name, args, kwargs in log:
        if name == "cutlass_scaled_mm":
            assert len(args) == len(cutlass.arguments), (cutlass, args)
        else:
            inspect.signature(real[name]).bind(*args, **kwargs)
