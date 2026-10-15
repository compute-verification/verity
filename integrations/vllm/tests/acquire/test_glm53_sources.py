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
from verity_vllm.commit import capture_map as CM

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

    def per_token_group_fp8_quant(input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0=False):
        calls.append("group_quant")

    monkeypatch.setattr(torch.ops._C, MS.GROUP_QUANT_OP, per_token_group_fp8_quant, raising=False)
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


def group_quant(x: T) -> tuple[T, T]:
    """`per_token_group_quant_fp8(x, 128, use_ue8m0=True)` as fp8_utils runs it on CUDA: allocate (x_q, x_s), then the `_C` op."""
    x_q, x_s = T(*x.shape, dtype=torch.float8_e4m3fn), T(x.shape[0], x.shape[1] // 128, dtype=torch.float32)
    torch.ops._C.per_token_group_fp8_quant(x, x_q, x_s, 128, 1e-10, -448.0, 448.0, True)
    return x_q, x_s


class MoERunner(torch.nn.Module):
    """One MoE layer's forward as the pinned vLLM runs it for GLM-5.3 at TP8: `GroupedTopKRouter._compute_routing` (the module-global
    `grouped_topk`, by keyword), then `TritonExperts.apply` with 128x128 block FP8 and SiLU.  `path` "served" is the default kernels' path
    (DeepGEMM's ue8m0 on: the input's group quant, the w13 launch, `_C.silu_and_mul`, the activation's group quant, the w2 launch,
    `moe_sum`), "fused" the SiLU fused into the quant (`silu_and_mul_per_block_quant`), "bf16" no quant.  The forward returns the
    shared expert's output plus the routed sum, a fresh tensor; `reduce_results` is False, as GLM-5.3 builds it."""

    reduce_results = False

    def __init__(self, v: types.SimpleNamespace, *, path: str = "served") -> None:
        super().__init__()
        self.v, self.path = v, path
        self.bias = T(E, dtype=torch.float32)
        self.shared_quant = False           # a shared-expert group quant between the routed path's calls

    def forward(self, hidden_states: T, router_logits: T) -> T:
        v, M, fp8_on = self.v, hidden_states.shape[0], self.path != "bf16"
        topk_weights, topk_ids = v.gr.grouped_topk(
            hidden_states=hidden_states, gating_output=router_logits, topk=TOPK, renormalize=True, num_expert_group=1, topk_group=1,
            scoring_func="sigmoid", routed_scaling_factor=2.5, e_score_correction_bias=self.bias)
        N = 2 * MIL
        cache1, cache3, out = T(M, TOPK, N), T(M, TOPK, H), T(M, H)
        sorted_ids, expert_ids, npp, config = T(M * TOPK, dtype=torch.int32), T(M, dtype=torch.int32), T(1, dtype=torch.int32), {}
        fp8 = dict(compute_type=None, use_fp8_w8a8=fp8_on, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                   per_channel_quant=False, block_shape=[128, 128] if fp8_on else None, B_bias=None)
        a1, a1q_scale = group_quant(hidden_states) if fp8_on else (hidden_states, None)
        v.tm.invoke_fused_moe_triton_kernel(a1, T(E, N, H), cache1, a1q_scale, None, None, sorted_ids, expert_ids, npp, False,
                                            TOPK, config, **fp8)
        if self.path == "fused":
            q2, a2q_scale = v.ops.silu_and_mul_per_block_quant(cache1.view(-1, N), group_size=128, quant_dtype=torch.float8_e4m3fn)
        else:
            cache2 = T(M * TOPK, MIL)                                       # torch.ops._C.silu_and_mul(cache2, cache1.view(-1, N))
            q2, a2q_scale = group_quant(cache2) if fp8_on else (cache2, None)
        if self.shared_quant:
            group_quant(T(M, H))
        v.tm.invoke_fused_moe_triton_kernel(q2, T(E, H, MIL), cache3, a2q_scale, None, topk_weights, sorted_ids, expert_ids, npp, True, 1,
                                            config, **fp8)
        v.ops.moe_sum(cache3, out)
        return T(M, H)


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


def run_source(monkeypatch, layers: list[int], M: int = 4, *, world: int = TP, **kw) -> tuple[Committer, MS.MoeBlockSource, list[str]]:
    """One step of the MoE source on `layers` (installed, run, uninstalled): the committer, the source and the names it acquired."""
    v = fake_vllm(monkeypatch)
    c = Committer()
    model = moe_model(v, layers, **kw)
    src = MS.MoeBlockSource(c, model, world=world)
    src.install()
    try:
        run_step(model, layers, M=M)
    finally:
        src.uninstall()
    return c, src, [n for kind, n in c.log if kind == "acquire"]


def test_the_moe_source_commits_exactly_the_maps_moe_taps_on_a_served_step(monkeypatch):
    from verity_vllm.query.manifest.format import MOE_FP8_PLANES

    v = fake_vllm(monkeypatch)
    c = Committer()
    model = moe_model(v, MOE_LAYERS)
    src = MS.MoeBlockSource(c, model, world=TP)
    orig, orig_q = v.gr.grouped_topk, getattr(torch.ops._C, MS.GROUP_QUANT_OP)
    src.install()
    assert v.gr.grouped_topk is not orig and getattr(torch.ops._C, MS.GROUP_QUANT_OP) is not orig_q
    try:
        run_step(model, MOE_LAYERS, M=256)
    finally:
        src.uninstall()
    assert v.gr.grouped_topk is orig and getattr(torch.ops._C, MS.GROUP_QUANT_OP) is orig_q
    assert v.calls.count("grouped_topk") == len(MOE_LAYERS) and v.calls.count("group_quant") == 2 * len(MOE_LAYERS)
    names = [n for kind, n in c.log if kind == "acquire"]
    got = CM.bind_names(CMAP, ENV, names, profile="0123456789ab", request="r0", t=0, phase="prefill", routes=("dense-mha",), rank=0,
                        binders=("moe",))
    assert got["unknown"] == [] and got["missing"] == [] and len(names) == len(set(names)) == 11 * len(MOE_LAYERS)
    assert got["ids"] == sorted(CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase="prefill",
                                              routes=("dense-mha",), rank=0, binders=("moe",)))
    first = MOE_LAYERS[0]
    planes = [("acquire", f"moe/L{first}/{p}") for p in MOE_FP8_PLANES]
    assert c.log[:12] == planes[:5] + [("flush", "up")] + planes[5:] + [("acquire", f"model.layers.{first}.mlp.experts/out")]
    # each name is the tensor the map says, at its dtype and shape on one rank
    want = CM.committer_names(CMAP, ENV, routes=("dense-mha",), binders=("moe",))
    for name, t in c.acquired.items():
        tap, _layer = want[name]
        assert str(t.dtype).replace("torch.", "") == tap["dtype"], name
        assert list(t.shape) == [CM.dim(d, {**ENV, "M": 256}, {}) for d in tap["shape"]], name
    assert src.stats["tensors_per_layer"] == len(MOE_FP8_PLANES) == 10 and src.planes == MOE_FP8_PLANES and c.errors == []
    assert src.act_scaled is True and src.stats["outputs"] == len(MOE_LAYERS)
    assert src.out_site_names() == sorted(f"model.layers.{k}.mlp.experts" for k in MOE_LAYERS)
    assert src.stats["router"] == {"sigmoid:block": len(MOE_LAYERS)} and src.router_steps == {"block": [0]}
    decl = src.declaration_classes()[0]
    assert decl["tensors_per_layer"] == 10 and decl["planes"] == list(MOE_FP8_PLANES) and decl["router_kernel_steps"] == {"block": [0]}


def test_the_act_plane_is_the_input_of_the_quant_that_wrote_the_w2_input(monkeypatch):
    """A shared expert's group quant between the routed activation's quant and the w2 launch does not move `act`: it is matched to the
    w2's A by identity, not taken as the last quant."""
    seen: list[tuple[Any, Any]] = []
    v = fake_vllm(monkeypatch)
    inner = getattr(torch.ops._C, MS.GROUP_QUANT_OP)

    def spy(input, output_q, *a, **k):
        seen.append((input, output_q))
        return inner(input, output_q, *a, **k)

    monkeypatch.setattr(torch.ops._C, MS.GROUP_QUANT_OP, spy)
    c = Committer()
    model = moe_model(v, [3])
    model.model.layers[3].mlp.experts.shared_quant = True
    src = MS.MoeBlockSource(c, model, world=TP)
    src.install()
    run_step(model, [3], M=4)
    src.uninstall()
    (x_in, q_in), (x_act, q_act), (x_shared, _q) = seen
    assert c.acquired["moe/L3/in_q"] is q_in and c.acquired["moe/L3/act"] is x_act and c.acquired["moe/L3/act_q"] is q_act
    assert x_shared is not x_act and c.errors == [] and src.stats["tensors_per_layer"] == 10


def test_the_fused_silu_quant_binds_nine_planes_without_act(monkeypatch):
    from verity_vllm.query.manifest.format import MOE_FP8_PLANES

    c, src, names = run_source(monkeypatch, [3], path="fused")
    planes = tuple(p for p in MOE_FP8_PLANES if p != "act")
    assert names == [f"moe/L3/{p}" for p in planes] + ["model.layers.3.mlp.experts/out"]
    assert src.planes == planes and src.stats["tensors_per_layer"] == 9 and c.errors == []


def test_a_w2_input_that_no_group_quant_of_the_layer_wrote_is_an_error(monkeypatch):
    v = fake_vllm(monkeypatch)
    raw = getattr(torch.ops._C, MS.GROUP_QUANT_OP)
    c = Committer()
    model = moe_model(v, [3])
    src = MS.MoeBlockSource(c, model, world=TP)
    src.install()
    setattr(torch.ops._C, MS.GROUP_QUANT_OP, raw)          # a call site that holds the op from before the install: no quant is seen
    run_step(model, [3], M=4)
    src.uninstall()
    names = [n for kind, n in c.log if kind == "acquire"]
    assert "moe/L3/act" not in names and "moe/L3/in_q" in names and c.x03_expected == 1
    assert any("no per_token_group_fp8_quant output" in e for e in c.errors)


def test_the_moe_output_is_bound_only_on_a_tp_rank_of_a_module_that_does_not_reduce_it(monkeypatch):
    _c, src, names = run_source(monkeypatch, [3], world=1)
    assert src.outputs == set() and not any(n.endswith("/out") for n in names) and len(names) == 10
    v = fake_vllm(monkeypatch)
    model = moe_model(v, [3])
    model.model.layers[3].mlp.experts.reduce_results = True
    assert MS.MoeBlockSource(Committer(), model, world=TP).outputs == set()


def test_identity_check_holds_a_served_step_to_ten_planes_per_layer_and_its_out(monkeypatch):
    from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta

    _c, src, names = run_source(monkeypatch, [3, 4])
    com = NativeHostCommitter(None, run_id="glm53-moe", gpu_tree=True)
    com._modules, com.sources = [], [src]
    man = com.expected_manifest()
    assert man["modules"] == ["model.layers.3.mlp.experts", "model.layers.4.mlp.experts"] and man["moe_tensors_per_layer"] == 10

    def check(steps: list[list[str]]) -> dict[str, Any]:
        com._layouts = {}
        for k, ns in enumerate(steps):
            com._layouts[k] = [TensorMeta(i, n, "bfloat16", (4, 8), 64, GPU_CHUNK_BYTES, i, 1) for i, n in enumerate(ns)]
            for i, m in enumerate(com._layouts[k]):
                m.dev_off = m.stream_off = 64 * i
        return com.identity_check()

    assert check([names, names])["ok"] is True
    r = check([names, [n for n in names if n != "moe/L4/act"]])
    assert r["ok"] is False and r["violations"][0]["moe_layers_bad"] == {4: 9}
    r = check([names, [n for n in names if n != "model.layers.4.mlp.experts/out"]])
    assert r["ok"] is False and r["violations"][0]["missing_modules"] == ["model.layers.4.mlp.experts"]


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
    assert "moe/L3/router_w" not in names and len(names) == 8 and c.x03_expected == 1
    assert any("torch path" in e for e in c.errors)


def test_the_bf16_path_keeps_six_tensors_and_a_switch_mid_run_is_an_error(monkeypatch):
    v = fake_vllm(monkeypatch)
    c = Committer()
    model = moe_model(v, [3], path="bf16")
    src = MS.MoeBlockSource(c, model)
    src.install()
    run_step(model, [3], M=4)
    assert [n for kind, n in c.log if kind == "acquire"] == [f"moe/L3/{p}" for p in ("router_w", "router_ids", "up", "act", "down", "sum")]
    assert src.stats["tensors_per_layer"] == MS.TENSORS_PER_LAYER and c.errors == []
    model.model.layers[3].mlp.experts.path = "served"
    run_step(model, [3], M=4)
    src.uninstall()
    assert any("A_scale" in e for e in c.errors) and src.stats["tensors_per_layer"] == MS.TENSORS_PER_LAYER


def test_grouped_topk_args_reads_positional_and_keyword_calls():
    a = MS.grouped_topk_args(("h", "g", 8, True), {"scoring_func": "sigmoid", "e_score_correction_bias": "b"})
    assert (a["gating_output"], a["topk"], a["num_expert_group"], a["scoring_func"], a["e_score_correction_bias"]) == ("g", 8, 0, "sigmoid", "b")


# ---------------------------------------------------------------- fused_allreduce_rms_norm, as DeepseekV32DecoderLayer calls it

MODEL_MODULE = "vllm.models.deepseek_v32.nvidia.model"
FLASHINFER_MAX_TOKENS = 42


def fake_far(monkeypatch, world: int = TP) -> types.SimpleNamespace:
    """The pinned `fused_allreduce_rms_norm` and the model module that imports it: FlashInfer's fused path (a fresh norm_out, the new
    residual written into hidden_states, `norm` never called) when M <= 42, else the all-reduce then `norm(reduced, residual)`."""
    calls: list[str] = []

    def fused_allreduce_rms_norm(hidden_states, residual, norm):
        if world == 1:
            return norm(hidden_states, residual)
        if hidden_states.shape[0] <= FLASHINFER_MAX_TOKENS:
            calls.append("flashinfer")
            return T(*hidden_states.shape), hidden_states
        calls.append("all_reduce")
        return norm(T(*hidden_states.shape), residual)

    for name in ("vllm", "vllm.models", "vllm.models.common", "vllm.models.common.ops", "vllm.models.deepseek_v32",
                 "vllm.models.deepseek_v32.nvidia"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    home = types.ModuleType("vllm.models.common.ops.fused_allreduce_rms_norm")
    home.fused_allreduce_rms_norm = fused_allreduce_rms_norm
    user = types.ModuleType(MODEL_MODULE)
    user.fused_allreduce_rms_norm = fused_allreduce_rms_norm
    monkeypatch.setitem(sys.modules, home.__name__, home)
    monkeypatch.setitem(sys.modules, MODEL_MODULE, user)
    return types.SimpleNamespace(home=home, user=user, calls=calls)


class Norm(torch.nn.Module):
    """RMSNorm as the pinned vLLM builds it: a weight and variance_epsilon; with a residual it returns (out, residual)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(H, dtype=torch.bfloat16), requires_grad=False)
        self.variance_epsilon = 1e-5

    def forward(self, x, residual=None):
        return T(*x.shape) if residual is None else (T(*x.shape), residual)


class DecoderLayer(torch.nn.Module):
    """DeepseekV32DecoderLayer.forward without sequence parallelism: layer 0 normalizes the embedding directly; every later
    input_layernorm and every post_attention_layernorm goes through the model module's `fused_allreduce_rms_norm` global."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = Norm(), Norm()

    def forward(self, hidden_states, residual):
        far = sys.modules[MODEL_MODULE].fused_allreduce_rms_norm
        if residual is None:
            residual, hidden_states = hidden_states, self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = far(hidden_states, residual, self.input_layernorm)
        hidden_states = T(*hidden_states.shape)                       # o_proj's partial (reduce_results=False)
        hidden_states, residual = far(hidden_states, residual, self.post_attention_layernorm)
        return T(*hidden_states.shape), residual                      # the MLP's or MoE's partial


class Model(torch.nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(DecoderLayer() for _ in range(n_layers))
        self.norm = Norm()

    def forward(self, M: int):
        hidden_states, residual = T(M, H), None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        hidden_states, _ = sys.modules[MODEL_MODULE].fused_allreduce_rms_norm(hidden_states, residual, self.norm)
        return hidden_states


for _cls in (DecoderLayer, Model):
    _cls.__module__ = MODEL_MODULE


class Root(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Model(ENV["L"])


class LoggingNC(NC):
    def __init__(self, log: list) -> None:
        super().__init__()
        self.log = log

    def flush(self) -> None:
        super().flush()
        self.log.append(("flush", "nc"))


def far_committer() -> Committer:
    c = Committer()
    c._nc = LoggingNC(c.log)
    return c


def module_hooks(model: torch.nn.Module, c: Committer) -> list:
    """The committer's module hook on every norm (native_host `_make_hook`): `<module>` for a tensor, `<module>/<k>` for a tuple."""
    def hook(name):
        def h(mod, args, out):
            for k, o in (enumerate(out) if isinstance(out, tuple) else [(None, out)]):
                c._acquire(name if k is None else f"{name}/{k}", o, c._cur)
        return h
    return [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if isinstance(m, Norm)]


class FarRun:
    """One rank's TP partial source on the model above, with the committer's module hooks on every norm; `step` runs one forward."""

    def __init__(self, monkeypatch, *, rank: int = 3) -> None:
        from verity_vllm.acquire.sources.partial_source import TPPartialSource

        self.v = fake_far(monkeypatch)
        self.orig = self.v.home.fused_allreduce_rms_norm
        self.c = far_committer()
        self.root = Root()
        self.src = TPPartialSource(self.c, self.root, rank=rank, world=TP)
        self.steps: list[list[str]] = []

    def step(self, M: int) -> list[str]:
        c, src = self.c, self.src
        c._cur = types.SimpleNamespace(step=len(self.steps))
        handles = module_hooks(self.root, c)
        src.install()
        assert self.v.user.fused_allreduce_rms_norm is not self.orig and self.v.home.fused_allreduce_rms_norm is not self.orig
        n0 = len(c.log)
        try:
            src.on_step_begin()
            self.root.model(M)
            src.on_step_end(c._cur)
        finally:
            src.uninstall()
            for h in handles:
                h.remove()
        assert self.v.user.fused_allreduce_rms_norm is self.orig and self.v.home.fused_allreduce_rms_norm is self.orig
        assert not src.far._hooked
        self.steps.append([n for kind, n in c.log[n0:] if kind == "acquire"])
        return self.steps[-1]

    def identity_check(self) -> dict[str, Any]:
        """The committer's X-03 identity check over the steps run, with this source attached and the norms as its modules."""
        from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta

        com = NativeHostCommitter(None, run_id="glm53-far", gpu_tree=True)
        com._modules = [(n, m) for n, m in self.root.named_modules() if isinstance(m, Norm)]
        com.sources = [self.src]
        for k, names in enumerate(self.steps):
            com._layouts[k] = [TensorMeta(i, n, "bfloat16", (4, 8), 64, GPU_CHUNK_BYTES, i, 1) for i, n in enumerate(names)]
            for i, m in enumerate(com._layouts[k]):
                m.dev_off = m.stream_off = 64 * i
        return com.identity_check()


@pytest.mark.parametrize("M,path", [(4, "F"), (256, "N")])
def test_the_allreduce_norm_wrap_binds_the_maps_names_on_either_path(monkeypatch, M, path):
    run = FarRun(monkeypatch)
    c, src = run.c, run.src
    names = run.step(M)
    route = ("dense-mha",)
    got = CM.bind_names(CMAP, ENV, names, profile="0123456789ab", request="r0", t=0, phase="prefill", routes=route, rank=3,
                        binders=("allreduce_norm",))
    # layer 0's input_layernorm is the module hook's own tap (no residual: a tensor), every other name is the map's
    assert got["unknown"] == ["model.layers.0.input_layernorm"] and got["missing"] == [] and len(names) == len(set(names))
    assert got["ids"] == sorted(CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase="prefill", routes=route,
                                              rank=3, binders=("allreduce_norm",)))
    assert run.v.calls == [{"F": "flashinfer", "N": "all_reduce"}[path]] * (2 * ENV["L"])
    sites = src.far.site_names()
    assert len(sites) == 2 * ENV["L"] and "model.layers.0.input_layernorm" not in sites and "model.norm" in sites
    assert src.site_counts() == dict.fromkeys(sites, 1) and set(src.site_names()) == set(sites)
    assert src.far.paths == [(0, path * (2 * ENV["L"]))] and src.far.stats[path] == 2 * ENV["L"] and c.errors == []
    # the partial is copied before the call runs (the fused path then writes the new residual into it); the fused path's returns at once
    i = c.log.index(("acquire", "model.layers.1.input_layernorm/part_rank3"))
    assert c.log[i + 1] == ("flush", "nc")
    if path == "F":
        assert c.log[i + 2:i + 5] == [("acquire", "model.layers.1.input_layernorm/0"), ("acquire", "model.layers.1.input_layernorm/1"),
                                      ("flush", "nc")]
    r = run.identity_check()
    assert r["ok"] is True and r["manifest"]["tp_sites"] == 2 * ENV["L"], r["violations"]


def test_the_allreduce_norm_partial_is_the_calls_hidden_states_and_the_returns_its_outputs(monkeypatch):
    seen: dict[str, Any] = {}
    run = FarRun(monkeypatch, rank=0)
    orig = run.orig

    def spy(hidden_states, residual, norm):
        out = orig(hidden_states, residual, norm)
        seen.setdefault("call", (hidden_states, out))
        return out

    run.v.home.fused_allreduce_rms_norm = run.v.user.fused_allreduce_rms_norm = run.orig = spy
    run.step(4)
    hs, (normed, residual) = seen["call"]                                  # the first call: layer 0's post_attention_layernorm
    site = "model.layers.0.post_attention_layernorm"
    assert run.c.acquired[f"{site}/part_rank0"] is hs and residual is hs
    assert run.c.acquired[f"{site}/0"] is normed and run.c.acquired[f"{site}/1"] is hs


@pytest.mark.parametrize("M,fused", [(42, True), (43, False)])
def test_the_meta_export_shim_emits_the_fused_leaf_only_on_flashinfers_served_range(monkeypatch, M, fused):
    from verity_vllm.program.frontend.export_compat import fused_allreduce_rms_norm_leaf

    v = fake_far(monkeypatch)
    model = Root()
    seen: list[tuple] = []

    class Bus:
        settings = {"flashinfer": True}

        def fused_allreduce_rms_norm(self, hidden_states, residual, weight, eps):
            seen.append((tuple(hidden_states.shape), tuple(residual.shape), tuple(weight.shape), eps))
            return T(*hidden_states.shape), T(*residual.shape)

    parallel = types.ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: types.SimpleNamespace(world_size=TP, bus=Bus())
    distributed = types.ModuleType("vllm.distributed")
    distributed.parallel_state = parallel
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)

    hidden, residual = T(M, H), T(M, H)
    with fused_allreduce_rms_norm_leaf(model):
        out = v.user.fused_allreduce_rms_norm(hidden, residual, model.model.layers[0].input_layernorm)
    assert bool(seen) is fused
    assert v.calls == ([] if fused else ["all_reduce"])
    assert tuple(out[0].shape) == (M, H) and tuple(out[1].shape) == (M, H)


def test_a_model_that_calls_it_and_makes_no_call_declares_a_site_no_step_binds(monkeypatch):
    from verity_vllm.acquire.sources.allreduce_norm import NO_CALL_SITE
    from verity_vllm.acquire.sources.partial_source import TPPartialSource

    fake_far(monkeypatch)
    src = TPPartialSource(far_committer(), Root(), rank=0, world=TP)
    assert src.far.used and src.far.site_names() == [NO_CALL_SITE] and src.site_counts()[NO_CALL_SITE] == 1
    assert len(src.far.candidates) == 2 * ENV["L"] + 1 and src.extra_step_bytes({}, None)["tensors_per_step"] == 2 * ENV["L"] + 1
    assert TPPartialSource(far_committer(), Root(), rank=0, world=1).far.site_names() == []


def test_steps_on_different_paths_commit_the_same_names_and_a_step_missing_a_call_fails_identity_check(monkeypatch):
    run = FarRun(monkeypatch)
    run.step(256)
    run.step(4)
    assert run.steps[0] == run.steps[1]                                  # the same names in the same order: the module hook fires inside
    assert [p for _s, p in run.src.far.paths] == ["N" * (2 * ENV["L"]), "F" * (2 * ENV["L"])]
    assert run.identity_check()["ok"] is True
    run.root.model.layers[5].forward = lambda hs, res: (T(*hs.shape), res)
    run.step(4)
    r = run.identity_check()
    bad = r["violations"][0]
    assert r["ok"] is False and bad["step"] == 2
    assert bad["tp_sites_bad"] == {"model.layers.5.input_layernorm": 0, "model.layers.5.post_attention_layernorm": 0}
    assert bad["missing_modules"] == ["model.layers.5.input_layernorm", "model.layers.5.post_attention_layernorm"]
