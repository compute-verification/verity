"""attribute.py and attn_bytes.py's attribution over synthetic Nsight Compute CSVs: windows split by the marker, the step
counted by the lm_head kernel, the KV taken from the live tokens, and per-layer classes scaled to full depth."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"ephemeral_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = load("attribute")
AB = load("attn_bytes")
GEMM = "void cutlass::device_kernel<enable_sm120_family<gemm::GemmUniversal<...>>>"
LM = "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)"
MHA = "kernel_mha(unsigned int, float, const float *, Vec<__nv_bfloat16, 128> *)"
NORM = "vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8, 1>"
SAMPLE = "_gumbel_sample_kernel"
SPIN = "void at::cuda::<unnamed>::spin_kernel(long)"


def ncu_csv(rows: list[tuple[str, float, float]]) -> str:
    out = io.StringIO()
    w = csv.writer(out, quoting=csv.QUOTE_ALL)
    w.writerow(["ID", "Kernel Name", A.READ, A.WRITE])
    w.writerow(["", "", "byte", "byte"])
    for i, (name, rd, wr) in enumerate(rows):
        w.writerow([i, name, f"{rd:,.0f}", f"{wr:,.0f}"])
    return "==PROF== Connected\n" + out.getvalue()


def arm(layers: int, hidden: int = 4096, lin: int = 1000, lm: int = 500, kv_tl: int = 2) -> dict:
    return {"budget": {"linear_bytes": lin, "linear_scale_bytes": 0, "lm_head_bytes": lm, "layers": layers,
                       "kv_bytes_per_token_layer": kv_tl},
            "model_shape": {"hidden": hidden}, "profile_steps": 2, "compiled": False, "points": []}


def node_step(layers: int, kv_layer: float, act: float = 3.0) -> list[tuple[str, float, float]]:
    rows = []
    for _ in range(layers):
        rows += [(GEMM, 1000 / layers + act, 1.0), (MHA, kv_layer + 1.0, 2.0), (NORM, act, 0.0)]
    return rows + [(LM, 500.0, 2.0), (SAMPLE, 4.0, 0.0)]


def test_node_split_scales_layers_and_keeps_per_step_kernels_once():
    a = arm(layers=4)
    pt = {"batch": 2, "context": 100, "tokens_open": 200, "tokens_close": 204}
    kv = 202 * 4 * 2  # live tokens x layers x bytes a token-layer
    w = A.windows(A.rows_from_text(ncu_csv([(SPIN, 0, 0)] + node_step(4, kv / 4) * 2)))
    assert len(w) == 1
    s = A.split(a, pt, w[0])
    assert s["steps"] == 2
    assert s["kv_read"] == pytest.approx(kv)
    assert s["kv_check"] == pytest.approx((kv + 4) / kv)
    assert s["unreg_encoded_lm"] == pytest.approx(4 * (3 + 1 + 3) + 4)
    assert s["classes"]["lm_head"]["kernels"] == 1
    f = s["full_depth"]  # 4096 hidden -> 32 layers: the per-layer bytes x 8, the lm_head and sampler once
    assert f["layers"] == 32
    assert f["ephemeral_ex_lm"] == pytest.approx(32 * (3 + 1 + 3) + 4)
    assert f["linear"] == pytest.approx(8000)
    assert f["write"] == pytest.approx(32 * 3 + 2)


def test_graph_split_counts_steps_by_lm_head_and_refuses_a_mismatch():
    a = arm(layers=80, hidden=8192)
    pt = {"batch": 1, "context": 10}
    rows = [(SPIN, 0, 0), ("graph", 1050.0, 9.0), (LM, 500.0, 0.0), (SAMPLE, 2.0, 0.0)] * 1
    rows += [("graph", 1050.0, 9.0), (LM, 500.0, 0.0), (SAMPLE, 2.0, 0.0)]
    w = A.windows(A.rows_from_text(ncu_csv(rows)))[0]
    s = A.split(a, pt, w)
    assert s["steps"] == 2 and "full_depth" not in s
    kv = (10 + 4 + 1.5) * 80 * 2
    assert s["graph_excess_read"] == pytest.approx(1050 - 1000 - kv)
    assert A.split(a, pt, w[:-2])["fail"].startswith("2 decode graphs for 1")


def test_attention_kernel_is_not_the_lm_head_however_large():
    assert A.classify(MHA, 10**12, 500) == "attention"
    assert A.classify(GEMM, 440.0, 500) == "gemm"
    assert A.classify(GEMM, 460.0, 500) == "lm_head"
    fused = "triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_div_fused_add_rms_norm_max_mul_reciprocal_unsqueeze_1"
    assert A.classify(fused, 1.0, 500) == "triton"
    once = "triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_div_embedding_max_mul_reciprocal_rms_norm_unsqueeze_0"
    assert A.classify(once, 1.0, 500) == "embed"


def test_attn_bytes_attribute(tmp_path: Path):
    res = {"cells": [{"batch": 2, "context": 64, "plain_bytes": 1000, "encoded_bytes": 1000}],
           "windows": [{"batch": 2, "context": 64, "form": 3, "splits": 4, "layers": 2, "partial_bytes_max": 8},
                       {"batch": 2, "context": 64, "form": 7, "splits": 4, "layers": 2, "partial_bytes_max": 4}]}
    out = tmp_path / "attn.json"
    out.write_text(json.dumps(res))
    rows = [(SPIN, 0, 0), ("attn_mma_kernel", 1010, 4), ("attn_combine", 2, 0)] + [("attn_mma_kernel", 1010, 4),
                                                                                    ("attn_combine", 2, 0)]
    rows += [(SPIN, 0, 0), ("attn_occ_kernel", 1000, 0), ("attn_combine", 0, 0)] * 1 + [("attn_occ_kernel", 1000, 0)]
    (tmp_path / "ncu.csv").write_text(ncu_csv(rows))

    assert AB.attribute(SimpleNamespace(csv=str(tmp_path / "ncu.csv"), out=str(out))) == 0
    w = json.loads(out.read_text())["windows"]
    assert w[0]["read"] == pytest.approx(1012) and w[0]["excess_read"] == pytest.approx(12)
    assert w[0]["combine_read"] == pytest.approx(2) and w[0]["write"] == pytest.approx(4)
    assert w[1]["excess_read"] == pytest.approx(0)


def test_main_passes_a_timing_only_run_and_fails_a_profiled_arm_without_its_csv(tmp_path: Path, monkeypatch):
    budget = {"linear_bytes": 1, "lm_head_bytes": 1, "kv_bytes_per_token_layer": 1, "layers": 1}
    (tmp_path / "8b-L0-0-time.json").write_text(json.dumps({"budget": budget, "profile_steps": 0, "points": []}))
    monkeypatch.setattr("sys.argv", ["attribute.py", str(tmp_path)])
    assert A.main() == 0
    (tmp_path / "8b-L8-0-node.json").write_text(json.dumps({"budget": budget, "profile_steps": 2, "points": [],
                                                           "profiled": [{"batch": 1, "context": 64}]}))
    assert A.main() == 1
