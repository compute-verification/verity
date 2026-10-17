from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def profile(path: Path, stack: str | None, kernels: dict[str, float], prefill: dict[str, float]) -> Path:
    def ks(d):
        return [{"name": n, "ms": ms} for n, ms in d.items()]

    path.write_text(json.dumps({
        "args": {"stack": stack, "stack_k": 2},
        "shapes": [{"batch": 32, "ctx": 1536, "kv_bytes_per_step": 6.0e9, "span_ms": sum(kernels.values()),
                    "kernels": ks(kernels)}],
        "prefill": [{"chunk": 2048, "context": 2048, "wall_ms": sum(prefill.values()), "kernels": ks(prefill)}],
    }))
    return path


def test_each_term_added_over_stock(tmp_path: Path):
    stock = profile(tmp_path / "stock.json", None, {
        "_ZN7cutlass13device_kernelI19enable_sm120_familyINS_4gemm": 5.0,
        "void flash::flash_fwd_splitkv_kernel<x>": 4.0, "void flash::flash_fwd_splitkv_combine_kernel<x>": 1.0,
        "void vllm::reshape_and_cache_flash_kernel<x>": 0.1, "void vllm::act_and_mul_kernel<x>": 0.5,
    }, {"_ZN7cutlass13device_kernelI": 40.0, "void vllm::reshape_and_cache_flash_kernel<x>": 1.0})
    enc = profile(tmp_path / "enc.json", "stack", {
        "void wt::fused_kernel<2, 4, true>": 12.0, "kv::pg_reduce_kernel(float const*)": 1.0,
        "void kv::pg_attn_kernel<true, true>": 9.0, "kv::pg_combine": 1.0,
        "void kv::pg_append_multi_kernel<true>": 5.1, "rb::leaves_kernel": 1.5, "kv::pg_receipt_sparse_kernel": 0.5,
        "void vllm::act_and_mul_kernel<x>": 0.5,
    }, {"_ZN7cutlass13device_kernelI": 40.0, "void wt::decode_kernel<2, 3>": 20.0,
        "void kv::pg_append_multi_kernel<true>": 200.0})
    subprocess.run([sys.executable, str(HERE / "budget.py"), str(stock), str(enc), "--R", "1000", "--out",
                    str(tmp_path)], check=True)
    (run,) = json.loads((tmp_path / "budget.json").read_text())["runs"]
    assert run["label"] == "stack k2"
    (d,) = run["decode"]
    assert d["added_ms"] == pytest.approx({"weights": 8.0, "attention": 5.0, "kv_encode": 5.0, "receipts": 2.0,
                                           "other": 0.0})
    assert d["attention_over_stock"] == pytest.approx(2.0)
    assert d["attention_kv_of_R"] == pytest.approx(0.6)
    assert d["stock_attention_kv_of_R"] == pytest.approx(1.2)
    (p,) = run["prefill"]
    assert p["added_ms"]["weights"] == pytest.approx(20.0)
    assert p["added_ms"]["kv_encode"] == pytest.approx(199.0)
