"""A block-FP8 fused-MoE checkpoint (Qwen3-30B-A3B-FP8, Qwen3-235B-A22B-FP8): `quant_refusal` admits its routers in
`modules_to_not_convert` only with the MoE's own config.json and only when every layer's router is listed, the MoE weights type
carries e4m3 experts with their f32 block scales on the rank's grid (vLLM d9105ea80 `refine_fp8_moe_block_shape`: 64 at TP8 for the
235B), the routers and norms stay bf16, and the profile binds the served expert scale parameters to those fields."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.observe.fold.profiles import generic as G
from verity_vllm.program import model as M

#: Qwen/Qwen3-235B-A22B-FP8 at revision 39eb2b06, the bytes the 235B pin records as its config_sha256
CFG_235B_PATH = dg.HF_CONFIG_DIR / "QWEN3_235B_A22B_FP8.config.json"
CFG_235B_SHA256 = "702c46d431bb984db9035a1225186bbfdb52c0d19c82104df4a37cd005e0369e"
LAYER = "model.layers.0."
EXPERTS = "model.layers.0.mlp.experts.routed_experts."


def _cfg(name: str) -> dict:
    return json.loads(CFG_235B_PATH.read_text() if name == "235B" else (dg.HF_CONFIG_DIR / f"{name}.config.json").read_text())


def _kept(cfg: dict, kept: list[str]) -> dict:
    return dict(cfg, quantization_config=dict(cfg["quantization_config"], modules_to_not_convert=kept))


def _shapes(t) -> list[int]:
    out = []
    while hasattr(t, "elem"):
        out.append(t.n)
        t = t.elem
    return out


def _layer(C: dict) -> dict:
    return dict(dict(M.weights_type(C).fields)["layers"].elem.fields)


def test_235b_config_is_the_pinned_bytes():
    assert hashlib.sha256(CFG_235B_PATH.read_bytes()).hexdigest() == CFG_235B_SHA256


@pytest.mark.parametrize("name", ["235B", "QWEN3_30B_A3B_FP8"])
def test_fp8_moe_checkpoints_listing_every_router_are_admitted(name):
    cfg = _cfg(name)
    routers = {f"model.layers.{n}.mlp.gate" for n in range(cfg["num_hidden_layers"])}
    assert routers <= set(cfg["quantization_config"]["modules_to_not_convert"])
    assert M.quant_refusal(cfg["quantization_config"], cfg) is None and G.structural_refusal(cfg) is None


@pytest.mark.parametrize("extra", ["model.layers.3.mlp.experts", "model.layers.3.mlp.experts.7.down_proj",
                                   "model.layers.3.mlp.experts.7.gate_proj", "model.layers.3.self_attn.o_proj"])
def test_a_linear_kept_bf16_beside_fp8_experts_is_mixed_precision(extra):
    """An FP8 dense part whose experts (or any other linear) stay bf16 is still refused."""
    cfg = _cfg("235B")
    bad = _kept(cfg, [*cfg["quantization_config"]["modules_to_not_convert"], extra])
    assert "mixed-precision" in M.quant_refusal(bad["quantization_config"], bad)
    assert "mixed-precision" in G.structural_refusal(bad)
    with pytest.raises(ValueError, match="mixed-precision"):
        G.derive_config(bad, repo=None, revision=None)


def test_routers_are_admitted_only_with_a_moe_config():
    moe = _cfg("235B")
    qc = moe["quantization_config"]
    assert "mixed-precision" in M.quant_refusal(qc)                              # no config: a router is just a bf16 linear
    dense = _cfg("QWEN3_4B_FP8")
    with_router = _kept(dense, [*dense["quantization_config"]["modules_to_not_convert"], "model.layers.0.mlp.gate"])
    assert M.quant_refusal(dense["quantization_config"], dense) is None
    assert "mixed-precision" in M.quant_refusal(with_router["quantization_config"], with_router)
    beyond = _kept(moe, [*qc["modules_to_not_convert"], f"model.layers.{moe['num_hidden_layers']}.mlp.gate"])
    assert "mixed-precision" in M.quant_refusal(beyond["quantization_config"], beyond)    # no such layer


@pytest.mark.parametrize("drop", [{"model.layers.93.mlp.gate"}, {f"model.layers.{n}.mlp.gate" for n in range(94)}])
def test_an_fp8_router_is_refused(drop):
    cfg = _cfg("235B")
    bad = _kept(cfg, [m for m in cfg["quantization_config"]["modules_to_not_convert"] if m not in drop])
    why = M.quant_refusal(bad["quantization_config"], bad)
    assert "quantizes the MoE routers" in why and f"({len(drop)} of 94 layers" in why
    assert "quantizes the MoE routers" in G.structural_refusal(bad)


@pytest.mark.parametrize("world,I_r,g,qkv_rows", [(1, 1536, 128, 72), (2, 768, 128, 36), (4, 384, 128, 18)])
def test_235b_rank_record_carries_e4m3_experts_and_their_block_scales(world, I_r, g, qkv_rows):
    C, _ = G.derive_config(_cfg("235B"), repo=None, revision=None, world=world)
    assert (C["I"], C["H"], C["NUM_EXPERTS"], C["FP8_BLOCK"]) == (I_r, 4096, 128, 128)
    L = _layer(C)
    assert {n: _shapes(L[n]) for n in ("gate_w", "w13", "w13_ws", "w2", "w2_ws", "qkv_ws")} == {
        "gate_w": [128, 4096], "w13": [128, 2 * I_r, 4096], "w13_ws": [128, 2 * I_r // g, 4096 // g],
        "w2": [128, 4096, I_r], "w2_ws": [128, 4096 // g, I_r // g], "qkv_ws": [qkv_rows, 32]}
    assert {n: L[n].homogeneous_width() for n in ("ln1", "gate_w", "w13", "w2", "w13_ws", "w2_ws")} == {
        "ln1": 16, "gate_w": 16, "w13": 8, "w2": 8, "w13_ws": 32, "w2_ws": 32}
    assert "gate_up_w" not in L and "gate_w_ws" not in L


def test_235b_tp8_rank_record_is_on_the_refined_64_grid():
    """TP8's rank holds I/8 = 192 expert columns, not a multiple of 128: vLLM refines the scales to 64 x 64 blocks."""
    C4, _ = G.derive_config(_cfg("235B"), repo=None, revision=None, world=4)
    C8 = dict(C4, I=192, NH=8, KVH=1, QKV=10 * 128, GU=384, WORLD=8)            # the TP8 rank: 64/8 heads, 4 KV heads replicated
    L = _layer(C8)
    assert {n: _shapes(L[n]) for n in ("w13", "w13_ws", "w2", "w2_ws", "qkv_ws", "o_ws")} == {
        "w13": [128, 384, 4096], "w13_ws": [128, 6, 64], "w2": [128, 4096, 192], "w2_ws": [128, 64, 3],
        "qkv_ws": [10, 32], "o_ws": [32, 8]}


@pytest.mark.parametrize("G_,I_r,H,g", [(128, 1536, 4096, 128), (128, 768, 2048, 128), (128, 192, 4096, 64), (128, 96, 4096, 32),
                                        (128, 320, 4096, 64), (128, 192, 2080, 32), (128, 1408, 2048, 128)])
def test_fp8_moe_block_is_vllms_refinement(G_, I_r, H, g):
    assert M.fp8_moe_block(G_, I_r, H) == g


@pytest.mark.parametrize("G_,I_r,H", [(128, 48, 4096), (128, 192, 2064), (128, 100, 4096)])
def test_fp8_moe_block_refuses_what_vllm_refuses(G_, I_r, H):
    with pytest.raises(ValueError, match="refuses the checkpoint"):
        M.fp8_moe_block(G_, I_r, H)


def test_fp8_moe_block_agrees_with_vllms_function():
    oracle = pytest.importorskip("vllm.model_executor.layers.fused_moe.oracle.fp8")
    for I_r in range(16, 4097, 16):
        for H, tp in ((4096, 2), (2048, 4), (2080, 8), (4096, 1)):
            got = oracle.refine_fp8_moe_block_shape(SimpleNamespace(intermediate_size_per_partition=I_r, tp_size=tp, hidden_dim=H), [128, 128])
            want = 128 if I_r % 128 == 0 else (got[0] if got else None)   # None: `create_weights` raises "not divisible by ... block_n"
            try:
                ours = M.fp8_moe_block(128, I_r, H)
            except ValueError:
                ours = None
            assert ours == want, (I_r, H, tp)


def test_the_profile_binds_the_expert_scales_only_under_fp8_experts():
    mod = G.make_module(f"verity_vllm.observe.fold.profiles.{G.module_name('QWEN3_30B_A3B_FP8', 1)}")
    wf, rc = mod.weight_field, mod.root_class
    assert wf(EXPERTS + "w13_weight_scale_inv") == ("layers", 0, "w13_ws") and rc(EXPERTS + "w13_weight_scale_inv") == "weight"
    assert wf(EXPERTS + "w2_weight_scale_inv") == ("layers", 0, "w2_ws") and rc(EXPERTS + "w2_weight_scale_inv") == "weight"
    assert wf(EXPERTS + "w13_weight") == ("layers", 0, "w13") and wf(LAYER + "mlp.gate.weight") == ("layers", 0, "gate_w")
    for unbound in ("mlp.gate.weight_scale_inv", "mlp.gate_up_proj.weight_scale_inv", "mlp.down_proj.weight_scale_inv",
                    "mlp.experts.routed_experts.w13_input_scale", "mlp.experts.w13_weight_scale_inv"):
        assert wf(LAYER + unbound) is None, unbound
    bf16 = G.make_module(f"verity_vllm.observe.fold.profiles.{G.module_name('QWEN3_30B_A3B', 1)}").weight_field
    assert bf16(EXPERTS + "w13_weight") == ("layers", 0, "w13") and bf16(EXPERTS + "w13_weight_scale_inv") is None
