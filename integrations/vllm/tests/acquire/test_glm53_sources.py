"""GLM-5.3's acquisition sources on CPU, driven the way the pinned vLLM's call sites drive the ops they wrap, against the capture map
(`manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json`): every name a source commits in one step is a name the map lists for
that source, and none is missing.  The vLLM modules are stand-ins installed in `sys.modules` with the pinned signatures; the tensors
are shape-and-dtype stand-ins on a "cuda" device, so nothing here needs vLLM or a GPU."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from verity_vllm.acquire.sources import moe_source as MS
from verity_vllm.check import capture_map as CM

ROOT = Path(__file__).parents[2]
CMAP = CM.load(ROOT / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json")
CONFIG = json.loads((ROOT / "tests" / "check" / "fixtures" / "glm_moe_dsa" / "config.json").read_text())
TP = CMAP["target"]["tensor_parallel_size"]
ENV = CM.symbols(CMAP, CONFIG, TP)
E, TOPK, MIL, H = ENV["E"], ENV["K"], ENV["MIL"], ENV["H"]


class T:
    """A device tensor as the sources read it: device, shape, dtype, numel, element_size; `view` keeps the storage's identity."""

    device = types.SimpleNamespace(type="cuda")

    def __init__(self, *shape: int, dtype: torch.dtype = torch.bfloat16) -> None:
        self.shape, self.dtype = tuple(shape), dtype

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    def view(self, *shape: int) -> "T":
        known = 1
        for d in shape:
            known *= d if d != -1 else 1
        v = T(*(self.numel() // known if d == -1 else d for d in shape), dtype=self.dtype)
        v.base = self
        return v


class NC:
    def __init__(self) -> None:
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1


class Committer:
    """The committer's side of a source: the current step job, `_acquire`, the gather collector's flush, `errors`."""

    gather = True

    def __init__(self) -> None:
        self._cur = types.SimpleNamespace(step=0)
        self._nc = NC()
        self.log: list[tuple[str, str]] = []
        self.acquired: dict[str, Any] = {}
        self.errors: list[str] = []

    def _acquire(self, name: str, t: Any, job: Any) -> None:
        self.log.append(("acquire", name))
        self.acquired[name] = t

    def flush_after_plane(self, what: str) -> bool:
        self.log.append(("flush", what))
        self._nc.flush()
        return True


# ---------------------------------------------------------------- the pinned vLLM's MoE call sites, as stand-ins

def fake_vllm(monkeypatch, *, fused_grouped_topk: bool = True) -> types.SimpleNamespace:
    """`sys.modules` entries for the modules the MoE source patches, with the pinned signatures and results' shapes."""
    calls: list[str] = []

    def mod(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        parent, _, leaf = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], leaf, m, raising=False)
        return m

    def topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize=False):
        calls.append("topk_softmax")

    def moe_sum(inp, out):
        calls.append("moe_sum")

    def silu_and_mul_per_block_quant(x, group_size=128, quant_dtype=None, scale_ub=None):
        calls.append("silu_and_mul_per_block_quant")
        return T(x.shape[0], x.shape[1] // 2, dtype=torch.float8_e4m3fn), T(x.shape[0], x.shape[1] // 2 // group_size, dtype=torch.float32)

    def invoke_fused_moe_triton_kernel(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                                       mul_routed_weight, top_k, config, compute_type=None, use_fp8_w8a8=False, use_int8_w8a8=False,
                                       use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, block_shape=None, B_bias=None):
        calls.append(f"launch:{mul_routed_weight}")

    def grouped_topk(hidden_states, gating_output, topk, renormalize, num_expert_group=0, topk_group=0, scoring_func="softmax",
                     routed_scaling_factor=1.0, e_score_correction_bias=None):
        calls.append("grouped_topk")
        return T(gating_output.shape[0], topk, dtype=torch.float32), T(gating_output.shape[0], topk, dtype=torch.int32)

    mod("vllm")
    mod("vllm.envs", VLLM_USE_FUSED_MOE_GROUPED_TOPK=fused_grouped_topk)
    mod("vllm.platforms", current_platform=types.SimpleNamespace(is_cuda=lambda: True))
    ops = mod("vllm._custom_ops", topk_softmax=topk_softmax, moe_sum=moe_sum, silu_and_mul_per_block_quant=silu_and_mul_per_block_quant)
    for name in ("vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
                 "vllm.model_executor.layers.fused_moe.experts", "vllm.model_executor.layers.fused_moe.router"):
        mod(name)
    fm = mod("vllm.model_executor.layers.fused_moe.fused_moe", invoke_fused_moe_triton_kernel=invoke_fused_moe_triton_kernel)
    tm = mod("vllm.model_executor.layers.fused_moe.experts.triton_moe", invoke_fused_moe_triton_kernel=invoke_fused_moe_triton_kernel)
    gr = mod("vllm.model_executor.layers.fused_moe.router.grouped_topk_router", grouped_topk=grouped_topk)
    return types.SimpleNamespace(ops=ops, fm=fm, tm=tm, gr=gr, calls=calls)


class MoERunner(torch.nn.Module):
    """One MoE layer's forward as the pinned vLLM runs it for GLM-5.3 at TP8: `GroupedTopKRouter._compute_routing` (the module-global
    `grouped_topk`, by keyword), then `TritonExperts.apply` with 128x128 block FP8 and SiLU (the w13 launch, the fused SiLU block quant,
    the w2 launch, `moe_sum`).  `block_fp8=False` runs the bf16 path instead (`self.activation`, then an unquantized w2 input)."""

    def __init__(self, v: types.SimpleNamespace, *, block_fp8: bool = True) -> None:
        super().__init__()
        self.v, self.block_fp8 = v, block_fp8
        self.bias = T(E, dtype=torch.float32)

    def forward(self, hidden_states: T, router_logits: T) -> T:
        v, M = self.v, hidden_states.shape[0]
        topk_weights, topk_ids = v.gr.grouped_topk(
            hidden_states=hidden_states, gating_output=router_logits, topk=TOPK, renormalize=True, num_expert_group=1, topk_group=1,
            scoring_func="sigmoid", routed_scaling_factor=2.5, e_score_correction_bias=self.bias)
        N = 2 * MIL
        cache1, cache3, out = T(M, TOPK, N), T(M, TOPK, H), T(M, H)
        sorted_ids, expert_ids, npp, config = T(M * TOPK, dtype=torch.int32), T(M, dtype=torch.int32), T(1, dtype=torch.int32), {}
        fp8 = dict(compute_type=None, use_fp8_w8a8=self.block_fp8, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                   per_channel_quant=False, block_shape=[128, 128] if self.block_fp8 else None, B_bias=None)
        a1q_scale = T(M, H // 128, dtype=torch.float32) if self.block_fp8 else None
        v.tm.invoke_fused_moe_triton_kernel(hidden_states, T(E, N, H), cache1, a1q_scale, None, None, sorted_ids, expert_ids, npp, False,
                                            TOPK, config, **fp8)
        if self.block_fp8:
            q2, a2q_scale = v.ops.silu_and_mul_per_block_quant(cache1.view(-1, N), group_size=128, quant_dtype=torch.float8_e4m3fn)
        else:
            q2, a2q_scale = T(M * TOPK, MIL), None
        v.tm.invoke_fused_moe_triton_kernel(q2, T(E, H, MIL), cache3, a2q_scale, None, topk_weights, sorted_ids, expert_ids, npp, True, 1,
                                            config, **fp8)
        v.ops.moe_sum(cache3, out)
        return out


def moe_model(v: types.SimpleNamespace, layers: list[int], **kw) -> torch.nn.Module:
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for i in range(max(layers) + 1):
        layer = torch.nn.Module()
        if i in layers:
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = MoERunner(v, **kw)
        model.model.layers.append(layer)
    return model


def run_step(model: torch.nn.Module, layers: list[int], M: int) -> None:
    for i in layers:
        model.model.layers[i].mlp.experts(T(M, H), T(M, E, dtype=torch.float32))


MOE_LAYERS = CM.layer_set(CMAP, "moe", ENV)


def test_the_moe_source_commits_exactly_the_maps_moe_taps_on_a_block_fp8_step(monkeypatch):
    v = fake_vllm(monkeypatch)
    c = Committer()
    model = moe_model(v, MOE_LAYERS)
    src = MS.MoeBlockSource(c, model)
    orig = v.gr.grouped_topk
    src.install()
    assert v.gr.grouped_topk is not orig
    try:
        run_step(model, MOE_LAYERS, M=256)
    finally:
        src.uninstall()
    assert v.gr.grouped_topk is orig and v.calls.count("grouped_topk") == len(MOE_LAYERS)
    names = [n for kind, n in c.log if kind == "acquire"]
    got = CM.bind_names(CMAP, ENV, names, profile="0123456789ab", request="r0", t=0, phase="prefill", routes=("dense-mha",), rank=0,
                        binders=("moe",))
    assert got["unknown"] == [] and got["missing"] == [] and len(names) == len(set(names)) == 7 * len(MOE_LAYERS)
    assert got["ids"] == sorted(CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase="prefill",
                                              routes=("dense-mha",), rank=0, binders=("moe",)))
    first = MOE_LAYERS[0]
    assert c.log[:8] == [("acquire", f"moe/L{first}/router_w"), ("acquire", f"moe/L{first}/router_ids"), ("acquire", f"moe/L{first}/up"),
                         ("flush", "up"), ("acquire", f"moe/L{first}/act"), ("acquire", f"moe/L{first}/act_scale"),
                         ("acquire", f"moe/L{first}/down"), ("acquire", f"moe/L{first}/sum")]
    # each name is the tensor the map says, at its dtype and shape on one rank
    want = CM.committer_names(CMAP, ENV, routes=("dense-mha",), binders=("moe",))
    for name, t in c.acquired.items():
        tap, _layer = want[name]
        assert str(t.dtype).replace("torch.", "") == tap["dtype"], name
        assert list(t.shape) == [CM.dim(d, {**ENV, "M": 256}, {}) for d in tap["shape"]], name
    assert src.stats["tensors_per_layer"] == 7 and src.act_scaled is True and c.errors == []
    assert src.stats["router"] == {"sigmoid:block": len(MOE_LAYERS)} and src.router_steps == {"block": [0]}
    decl = src.declaration_classes()[0]
    assert decl["tensors_per_layer"] == 7 and decl["router_kernel_steps"] == {"block": [0]}


def test_the_router_kernel_follows_the_custom_ops_launch_rule(monkeypatch):
    assert [MS.router_kernel(256, m, 8) for m in (1, 1024, 1025, 8192)] == ["block", "block", "warp", "warp"]
    assert MS.router_kernel(128, 4, 8) == "warp" and MS.router_kernel(2048, 8192, 8) == "block" and MS.router_kernel(896, 8, 16) is None
    v = fake_vllm(monkeypatch)
    c = Committer()
    model = moe_model(v, [3])
    src = MS.MoeBlockSource(c, model)
    src.install()
    run_step(model, [3], M=1025)
    c._cur = types.SimpleNamespace(step=1)
    run_step(model, [3], M=4)
    src.uninstall()
    assert src.router_steps == {"warp": [0], "block": [1]}


def test_grouped_topk_on_its_torch_path_is_not_bound(monkeypatch):
    v = fake_vllm(monkeypatch, fused_grouped_topk=False)
    c = Committer()
    model = moe_model(v, [3])
    src = MS.MoeBlockSource(c, model)
    src.install()
    run_step(model, [3], M=4)
    src.uninstall()
    names = [n for kind, n in c.log if kind == "acquire"]
    assert "moe/L3/router_w" not in names and len(names) == 5 and c.x03_expected == 1
    assert any("torch path" in e for e in c.errors)


def test_the_bf16_path_keeps_six_tensors_and_a_switch_mid_run_is_an_error(monkeypatch):
    v = fake_vllm(monkeypatch)
    c = Committer()
    model = moe_model(v, [3], block_fp8=False)
    src = MS.MoeBlockSource(c, model)
    src.install()
    run_step(model, [3], M=4)
    assert [n for kind, n in c.log if kind == "acquire"] == [f"moe/L3/{p}" for p in ("router_w", "router_ids", "up", "act", "down", "sum")]
    assert src.stats["tensors_per_layer"] == MS.TENSORS_PER_LAYER and c.errors == []
    model.model.layers[3].mlp.experts.block_fp8 = True
    run_step(model, [3], M=4)
    src.uninstall()
    assert any("A_scale" in e for e in c.errors) and src.stats["tensors_per_layer"] == MS.TENSORS_PER_LAYER


def test_grouped_topk_args_reads_positional_and_keyword_calls():
    a = MS.grouped_topk_args(("h", "g", 8, True), {"scoring_func": "sigmoid", "e_score_correction_bias": "b"})
    assert (a["gating_output"], a["topk"], a["num_expert_group"], a["scoring_func"], a["e_score_correction_bias"]) == ("g", 8, 0, "sigmoid", "b")
