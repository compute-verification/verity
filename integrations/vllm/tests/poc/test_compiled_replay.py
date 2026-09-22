"""CPU-testable pieces of the compiled-preserving replay (CONTRACT §16.1): generated-kernel identity
collection/comparison and the launch-argument spec (integer-argument model, node naming)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm_adapter import kernel_identity as kid
from verity_vllm_adapter import launchspec

KERNEL_A = "@triton.jit\ndef triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    x = tl.load(in_ptr0)\n    tl.store(out_ptr0, x * 2)\n"
KERNEL_B = "@triton.jit\ndef triton_per_fused_rms_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):\n    pass\n"


def _wrapper(cache_root: str, aot: str, kernels: list[tuple[str, str]], call_extra: str = "") -> str:
    parts = [f"# AOT ID: ['{aot}']\nfrom torch._inductor.async_compile import AsyncCompile\nasync_compile = AsyncCompile()\n"]
    for name, src in kernels:
        parts.append(f"# kernel path: {cache_root}/xx/c{hashlib.sha256(src.encode()).hexdigest()[:20]}.py\n")
        parts.append(f"{name} = async_compile.triton('{name}', '''{src}''', device_str='cuda')\n\n")
    parts.append("\ndef call(args):\n")
    for name, _ in kernels:
        parts.append(f"    {name}.run(buf0, buf1, 576, stream=stream0)\n")
    parts.append(f"    extern_kernels.mm(buf1, w, out=buf2)\n{call_extra}    return (buf2,)\n\n\ndef benchmark_compiled_module(times=10, repeat=10):\n    pass\n")
    return "".join(parts)


def _make_cache(root: Path, aot: str, kernels: list[tuple[str, str]]) -> Path:
    d = root / "inductor" / "ab"
    d.mkdir(parents=True)
    (d / "cwrapper.py").write_text(_wrapper(str(root / "inductor"), aot, kernels))
    return root


def test_collect_parses_wrapper_kernels_and_call(tmp_path: Path) -> None:
    root = _make_cache(tmp_path / "p", "aot-1", [("triton_poi_fused_1", KERNEL_A), ("triton_per_fused_rms_0", KERNEL_B)])
    doc = kid.collect(str(root / "inductor"), None)
    assert doc["summary"]["wrappers"] == 1
    assert doc["summary"]["distinct_kernel_sources"] == 2
    w = doc["wrappers"][0]
    assert [k["name"] for k in w["kernels"]] == ["triton_poi_fused_1", "triton_per_fused_rms_0"]
    assert w["kernels"][0]["source_sha256"] == hashlib.sha256(KERNEL_A.encode()).hexdigest()
    assert w["aot_id"] == "'aot-1'"
    assert w["launches"] == 2
    assert w["extern_calls"] == ["extern_kernels.mm"]
    assert doc["kernels"]["triton_poi_fused_1"] == hashlib.sha256(KERNEL_A.encode()).hexdigest()
    assert kid.kernel_sources(str(root / "inductor"))[doc["kernels"]["triton_per_fused_rms_0"]] == KERNEL_B


def test_compare_same_sources_different_cache_root(tmp_path: Path) -> None:
    """Production and replay compiled under different cache roots: raw wrapper hashes differ (path
    comments), sources / call programs / normalised wrappers are identical."""
    a = kid.collect(str(_make_cache(tmp_path / "prod", "aot-1", [("triton_poi_fused_1", KERNEL_A)]) / "inductor"), None)
    b = kid.collect(str(_make_cache(tmp_path / "replay", "aot-2", [("triton_poi_fused_1", KERNEL_A)]) / "inductor"), None)
    cmp = kid.compare(a, b)
    assert cmp["kernel_sources_identical"] is True
    assert cmp["call_programs_identical"] is True
    assert cmp["wrappers_identical"] is False
    assert cmp["wrappers_identical_modulo_cache_path"] is True
    assert cmp["b_kernel_sources_subset_of_a"] is True
    assert cmp["cubins_identical"] is None and cmp["ptx_code_identical"] is None


def _add_best_config(root: Path, sub: str, kernel_src: str, cfg: dict) -> None:
    d = root / "inductor" / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / "ckernel.py").write_text(kernel_src)
    (d / "deadbeef.best_config").write_text(json.dumps({**cfg, "configs_hash": "c0ffee", "found_by_coordesc": False, "time_taken_ms": 5}))


def test_autotune_best_config_collected_and_compared(tmp_path: Path) -> None:
    """The runtime autotuner's chosen launch config is a per-process, timing-based choice: recorded per kernel
    and compared by name (the SiLU kernel case of the B0 run: production XBLOCK 1024/4 warps, replay 512/8)."""
    pa = _make_cache(tmp_path / "prod", "x", [("triton_poi_fused_1", KERNEL_A)])
    pb = _make_cache(tmp_path / "replay", "x", [("triton_poi_fused_1", KERNEL_A)])
    _add_best_config(pa, "na", "import triton\n@triton.jit\ndef triton_poi_fused_mul_silu_slice_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    pass\n", {"XBLOCK": 1024, "num_warps": 4, "num_stages": 1})
    _add_best_config(pb, "na", "import triton\n@triton.jit\ndef triton_poi_fused_mul_silu_slice_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    pass\n", {"XBLOCK": 512, "num_warps": 8, "num_stages": 1})
    _add_best_config(pa, "pg", "@triton.jit\ndef triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    pass\n", {"XBLOCK": 1024, "num_warps": 4, "num_stages": 1})
    _add_best_config(pb, "pg", "@triton.jit\ndef triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    pass\n", {"XBLOCK": 1024, "num_warps": 4, "num_stages": 1})
    a = kid.collect(str(pa / "inductor"), None)
    b = kid.collect(str(pb / "inductor"), None)
    assert a["summary"]["autotuned_kernels"] == 2
    assert {e["kernel"] for e in a["autotune_configs"]} == {"triton_poi_fused_mul_silu_slice_t_1", "triton_poi_fused_3"}
    assert a["autotune_configs"][0]["config"]["XBLOCK"] in (512, 1024)
    cmp = kid.compare(a, b)
    assert cmp["kernel_sources_identical"] is True
    at = cmp["autotune"]
    assert at["a"] == 2 and at["b"] == 2 and at["candidates_identical"] is True
    assert at["chosen_configs_identical"] is False
    assert len(at["differing"]) == 1
    d = at["differing"][0]
    assert d["kernel"] == "triton_poi_fused_mul_silu_slice_t_1" and len(d["kernel_module_sha256"]) == 64
    assert (d["a"], d["b"]) == ({"XBLOCK": 1024, "num_warps": 4, "num_stages": 1}, {"XBLOCK": 512, "num_warps": 8, "num_stages": 1})
    # two standalone modules both named ``triton_`` (Inductor's generic name) stay distinct: keyed by module sha
    _add_best_config(pa, "q1", "@triton.jit\ndef triton_(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    a = 1\n", {"XBLOCK": 1024, "num_warps": 4, "num_stages": 1})
    _add_best_config(pa, "q2", "@triton.jit\ndef triton_(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    a = 2\n", {"XBLOCK": 256, "num_warps": 2, "num_stages": 1})
    a2 = kid.collect(str(pa / "inductor"), None)
    assert a2["summary"]["autotuned_kernels"] == 4
    assert kid.compare(a2, a2)["autotune"] == {"a": 4, "b": 4, "chosen_configs_identical": True, "candidates_identical": True, "differing": []}
    # no best_config files at all: nothing to say
    assert kid.compare(kid.collect(str(_make_cache(tmp_path / "p2", "x", [("k", KERNEL_A)]) / "inductor"), None),
                       kid.collect(str(_make_cache(tmp_path / "r2", "x", [("k", KERNEL_A)]) / "inductor"), None))["autotune"]["chosen_configs_identical"] is None


def test_compare_reports_changed_kernel_by_name(tmp_path: Path) -> None:
    a = kid.collect(str(_make_cache(tmp_path / "prod", "x", [("triton_poi_fused_1", KERNEL_A), ("triton_per_fused_rms_0", KERNEL_B)]) / "inductor"), None)
    b = kid.collect(str(_make_cache(tmp_path / "replay", "x", [("triton_poi_fused_1", KERNEL_A.replace("* 2", "* 3")), ("triton_per_fused_rms_0", KERNEL_B)]) / "inductor"), None)
    cmp = kid.compare(a, b)
    assert cmp["kernel_sources_identical"] is False
    assert cmp["kernel_sources"] == {"a": 2, "b": 2, "common": 1, "only_a": ["triton_poi_fused_1"], "only_b": ["triton_poi_fused_1"]}
    assert cmp["b_kernel_sources_subset_of_a"] is False


def test_ptx_code_strips_only_debug_and_assert_paths() -> None:
    ptx = (
        "//\n.version 8.7\n.target sm_89\n\t.file\t1 \"/workspace/cache/prod/inductor/ny/c.py\"\n"
        ".visible .entry triton_poi_fused_1(\n)\n{\n\t.loc\t1 18 0\n\tadd.s32 \t%r1, %r2, %r3;\n}\n"
        ".global .align 1 .b8 assertFile_0[105] = {47, 119};\n"
        "\t.section\t.debug_abbrev\n\t{\n.b8 1\n\t}\n\t.section\t.debug_info\n\t{\n.b8 2\n\t}\n"
    )
    other = ptx.replace("/workspace/cache/prod", "/workspace/cache/replay").replace("assertFile_0[105] = {47, 119}", "assertFile_0[101] = {47, 120}").replace(".b8 2", ".b8 3")
    assert ptx != other
    assert kid.ptx_code(ptx) == kid.ptx_code(other)
    assert "add.s32" in kid.ptx_code(ptx)
    assert kid.ptx_code(ptx) != kid.ptx_code(ptx.replace("add.s32", "sub.s32"))


def test_triton_cache_ptx_code_identity(tmp_path: Path) -> None:
    for root, path in (("prod", "/c/prod"), ("replay", "/c/replay")):
        d = tmp_path / root / "K1"
        d.mkdir(parents=True)
        (d / "triton_poi_fused_1.cubin").write_bytes(b"\x7fELF" + path.encode())
        (d / "triton_poi_fused_1.ptx").write_text(f'\t.file\t1 "{path}/x.py"\nadd.s32 r1;\n')
    a = kid.collect(None, str(tmp_path / "prod"))
    b = kid.collect(None, str(tmp_path / "replay"))
    cmp = kid.compare(a, b)
    assert cmp["cubins_identical"] is False
    assert cmp["ptx_code_identical"] is True
    assert cmp["ptx_code"]["only_a"] == [] and cmp["ptx_code"]["only_b"] == []


def test_compare_cli_roundtrip(tmp_path: Path) -> None:
    a = kid.collect(str(_make_cache(tmp_path / "prod", "x", [("k", KERNEL_A)]) / "inductor"), None)
    (tmp_path / "a.json").write_text(json.dumps(a))
    b = json.loads((tmp_path / "a.json").read_text())
    assert kid.compare(a, b)["kernel_sources_identical"] is True


# ---------------- launch spec ----------------


def test_intarg_linear_model_and_violation() -> None:
    ia = launchspec.IntArg("xnumel")
    ia.add(64, 64 * 576)
    assert ia.model() == {"model": "underdetermined", "obs": [(64, 36864)]}
    ia.add(1, 576)
    assert ia.model() == {"model": "linear", "a": 576, "b": 0}
    ia.add(17, 17 * 576)
    assert ia.violations == 0
    ia.add(5, 5 * 576 + 1)
    assert ia.violations == 1
    assert ia.model()["model"] == "inconsistent"


def test_intarg_const_and_int_value() -> None:
    ia = launchspec.IntArg("r0_numel")
    ia.add(64, 576)
    ia.add(1, 576)
    assert ia.model() == {"model": "const", "value": 576}
    assert launchspec.int_value({"name": "r0_numel", "model": ia.model()}, 1) == 576
    assert launchspec.int_value({"name": "xnumel", "model": {"model": "linear", "a": 576, "b": 0}}, 1) == 576
    assert launchspec.int_value({"name": "stride", "model": {"model": "linear", "a": 0, "b": 4}}, 7) == 4
    with pytest.raises(ValueError):
        launchspec.int_value({"name": "n", "model": {"model": "underdetermined", "obs": [(64, 3)]}}, 1)


def test_intarg_same_rows_different_values_is_inconsistent() -> None:
    ia = launchspec.IntArg("n")
    ia.add(8, 100)
    ia.add(8, 200)
    assert ia.model()["model"] == "inconsistent"


def test_classify_name_rules() -> None:
    gemm = launchspec.classify_name("matmul_kernel_persistent", "matmul_kernel_persistent", ["layers.3.mlp.gate_up_proj.weight"], ["3"])
    assert gemm == ("gemm", "gemm", None, "layers.3.mlp.gate_up_proj")
    rms = launchspec.classify_name("triton_per_fused_fused_add_rms_norm_t_0", "triton_per_fused_fused_add_rms_norm_t_0", ["layers.3.post_attention_layernorm.weight"], ["3"])
    assert rms == ("rmsnorm", "rmsnorm", None, "layers.3.post_attention_layernorm")
    emb = launchspec.classify_name("triton_per_fused__to_copy_embedding_rms_norm_t_0", "triton_per_fused__to_copy_embedding_rms_norm_t_0", ["embed_tokens.weight", "layers.0.input_layernorm.weight"], [])
    assert emb == ("rmsnorm", "rmsnorm", None, "layers.0.input_layernorm")
    rope = launchspec.classify_name("triton_poi_fused_1", "triton_poi_fused_1", ["layers.3.self_attn.rotary_emb.cos_sin_cache"], ["3"])
    assert rope == ("rope", "rope", None, "layers.3.self_attn.rotary_emb")
    silu = launchspec.classify_name("triton_poi_fused_mul_silu_slice_t_1", "triton_poi_fused_mul_silu_slice_t_1", [], ["3"])
    assert silu == ("act_mul", "silu_mul", None, "layers.3.mlp.act_fn")
    other = launchspec.classify_name("triton_poi_fused_3", "triton_poi_fused_3", [], ["7"])
    assert other == ("fused", "other", "layers.7.fused.triton_poi_fused_3", "layers.7")
    top = launchspec.classify_name("triton_poi_fused_3", "triton_poi_fused_3", [], [])
    assert top == ("fused", "other", "top.fused.triton_poi_fused_3", "top")


def test_module_of_param() -> None:
    assert launchspec.module_of_param("layers.0.self_attn.qkv_proj.weight") == "layers.0.self_attn.qkv_proj"
    assert launchspec.module_of_param("layers.0.self_attn.rotary_emb.cos_sin_cache") == "layers.0.self_attn.rotary_emb"
    assert launchspec.module_of_param("norm.weight") == "norm"


# ---------------- weights by checkpoint reference (§16.8) ----------------


def test_hf_tensor_mapping() -> None:
    from verity_vllm_adapter import weight_sources as ws

    assert ws.hf_tensors("embed_tokens.weight") == (["model.embed_tokens.weight"], 0)
    assert ws.hf_tensors("norm.weight") == (["model.norm.weight"], 0)
    assert ws.hf_tensors("layers.12.self_attn.qkv_proj.weight") == (
        ["model.layers.12.self_attn.q_proj.weight", "model.layers.12.self_attn.k_proj.weight", "model.layers.12.self_attn.v_proj.weight"], 0)
    assert ws.hf_tensors("layers.3.mlp.gate_up_proj.weight") == (["model.layers.3.mlp.gate_proj.weight", "model.layers.3.mlp.up_proj.weight"], 0)
    assert ws.hf_tensors("layers.3.mlp.down_proj.weight") == (["model.layers.3.mlp.down_proj.weight"], 0)
    assert ws.hf_tensors("layers.3.input_layernorm.weight") == (["model.layers.3.input_layernorm.weight"], 0)
    assert ws.hf_tensors("layers.0.self_attn.rotary_emb.cos_sin_cache") is None
    assert ws.hf_tensors("layers.0.self_attn.attn._k_scale") is None


def test_source_for_records_checkpoint_identity() -> None:
    from verity_vllm_adapter import weight_sources as ws

    cp = {"repo": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f", "safetensors_sha256": {"model.safetensors": "80" * 32}}
    src = ws.source_for("layers.1.self_attn.qkv_proj.weight", cp)
    assert src == {"kind": "checkpoint", "repo": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f", "file": "model.safetensors",
                   "file_sha256": "80" * 32, "tensors": ["model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight",
                                                          "model.layers.1.self_attn.v_proj.weight"], "cat_dim": 0}
    assert ws.source_for("layers.0.self_attn.rotary_emb.cos_sin_cache", cp) is None
    assert ws.source_for("norm.weight", {"safetensors_sha256": {"a": "1", "b": "2"}}) is None  # sharded: stored
