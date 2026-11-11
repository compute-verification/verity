"""The MoE source on DeepSeek-V4-Pro's path (`verity_vllm/acquire/sources/moe_source.py`), on CPU against the capture map
`deepseek-v4-eager-hopper-tp8-v0`: each MoE layer runs `FusedTopKBiasRouter._compute_routing` (the module-global `fused_topk_bias`,
by keyword, with a hash table on the first HASH layers) and `MarlinExperts.apply` (the w13 and w2 `ops.moe_wna16_marlin_gemm`
launches, then `moe_sum`) as the pinned vLLM does, on the shape-and-dtype stand-ins of `test_glm53_sources`.

Its manifest of record names no `moe_block_stream` plane, so the source binds at the block's Calls (`at_calls`): each value under the
MoE module, `model.layers.<i>.ffn.experts/<label>`, a hash layer's router as router_w.hash, ids and in.

It catches a name the map lists for the moe binder that a step does not bind, or binds at another shape or dtype; a value bound from
the wrong tensor (the w2 launch's output as act, the w13 launch's input as up); a missing flush after either launch; a scored router
call that does not take `dsv4_topk` bound anyway; a router bound twice; a launch whose flags would bind values under swapped names; a
label that binds no row of the manifest's shape; and a hash layer's extra value that `identity_check` would count as a module."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import sys
import types

import pytest
import torch

from tests.acquire.dsv4_model import CMAP, ENV
from tests.acquire.test_glm53_sources import Committer, T, fake_vllm
from verity_vllm.acquire.sources import moe_source as MS
from verity_vllm.commit import capture_map as CM
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta, hazard_of_record
from verity_vllm.query.manifest.bind import Binder

L, HASH, E, K, MIL, H = ENV["L"], ENV["HASH"], ENV["E"], ENV["K"], ENV["MIL"], ENV["H"]
DECODE = ("hc-fused", "indexer-short", "decode-tokens")
PLANES = ("router_w", "router_ids", "up", "act", "down", "sum")
#: a manifest that names no `moe_block_stream` plane, as DeepSeek-V4-Pro's of record
AT_CALLS = {"identities": [{"family": "call_boundaries"}]}


def name(i: int, label: str) -> str:
    return f"model.layers.{i}.ffn.experts/{label}"


def labels(i: int) -> tuple[str, ...]:
    """What layer i commits, in order: a hash layer's router as its weights, ids and in."""
    return ("router_w.hash", "ids", "in", *PLANES[2:]) if i < HASH else PLANES


def fake_dsv4(monkeypatch) -> types.SimpleNamespace:
    """GLM's stand-ins plus the two sites DeepSeek-V4-Pro reaches: `ops.moe_wna16_marlin_gemm` and the router module's
    `fused_topk_bias` and `can_use_dsv4_topk`, with the pinned signatures and dispatch."""
    v = fake_vllm(monkeypatch)

    def moe_wna16_marlin_gemm(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros, g_idx, perm, workspace,
                              sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k, mul_topk_weights,
                              b_q_type, size_m, size_n, size_k, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float, thread_k=-1,
                              thread_n=-1, blocks_per_sm=-1):
        v.calls.append(f"marlin:{mul_topk_weights}")
        return output

    def can_use_dsv4_topk(gating_output, correction_bias, topk, renormalize, indices_dtype):
        return (gating_output.dtype == torch.float32 and len(gating_output.shape) == 2 and gating_output.shape[1] in (256, 384)
                and correction_bias is not None and topk == 6 and renormalize and indices_dtype in (torch.int32, torch.uint32, torch.int64))

    def fused_topk_bias(hidden_states, gating_output, scoring_func, e_score_correction_bias, topk, renormalize, indices_type=None,
                        input_tokens=None, hash_indices_table=None, routed_scaling_factor=1.0, bias_vl=None, image_sentinel_lo=0):
        m, idx = hidden_states.shape[0], torch.int32 if indices_type is None else indices_type
        w, ids = T(m, topk, dtype=torch.float32), T(m, topk, dtype=idx)
        if scoring_func == "sqrtsoftplus" and hash_indices_table is None and rt.can_use_dsv4_topk(
                gating_output, e_score_correction_bias, topk, renormalize, idx):
            v.calls.append("dsv4_topk")
        elif scoring_func == "softmax":
            sys.modules["vllm._custom_ops"].topk_softmax(w, ids, T(m, topk, dtype=torch.int32), gating_output, renormalize)
        else:
            v.calls.append("topk_hash_softplus_sqrt" if scoring_func == "sqrtsoftplus" else f"topk_{scoring_func}")
        return w, ids

    monkeypatch.setattr(v.ops, "moe_wna16_marlin_gemm", moe_wna16_marlin_gemm, raising=False)
    rt = types.ModuleType(MS.TOPK_BIAS_MODULE)
    rt.fused_topk_bias, rt.can_use_dsv4_topk = fused_topk_bias, can_use_dsv4_topk
    monkeypatch.setitem(sys.modules, MS.TOPK_BIAS_MODULE, rt)
    v.rt = rt
    return v


class MoERunner(torch.nn.Module):
    """One MoE layer as the pinned vLLM runs DeepSeek-V4-Pro at TP8 on Hopper: `fused_topk_bias` by its module global, then
    `_fused_marlin_moe` (the w13 launch into intermediate_cache1, the activation into intermediate_cache2, the w2 launch from it into
    intermediate_cache3) and `MarlinExperts.moe_sum`.  `seen` keeps what each kernel produced.  `positional` passes every launch
    argument by position; `weight_on_input` swaps the launches' flags as `apply_router_weight_on_input` does."""

    def __init__(self, layer: int, *, scoring: str = "sqrtsoftplus", positional: bool = False, weight_on_input: bool = False) -> None:
        super().__init__()
        self.scoring, self.positional, self.weight_on_input = scoring, positional, weight_on_input
        self.bias = T(E, dtype=torch.float32)
        self.table = T(ENV["V"], K, dtype=torch.int32) if layer < HASH else None
        self.seen: dict[str, object] = {}

    def _launch(self, a: T, out: T, top_k: int, mul: bool, size_m: int, size_n: int, size_k: int, topk_weights: T) -> T:
        ops = sys.modules["vllm._custom_ops"]
        meta = T(1, dtype=torch.int32)
        head = (a, out, T(1, dtype=torch.uint8), None, T(1, dtype=torch.uint8), None, None, None, None, None, meta, meta, meta, meta,
                topk_weights)
        tail = dict(moe_block_size=16, top_k=top_k, mul_topk_weights=mul, b_q_type="float4_e2m1f", size_m=size_m, size_n=size_n,
                    size_k=size_k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        return ops.moe_wna16_marlin_gemm(*head, *tail.values()) if self.positional else ops.moe_wna16_marlin_gemm(*head, **tail)

    def forward(self, hidden_states: T, router_logits: T, input_ids: T) -> T:
        m = hidden_states.shape[0]
        w, ids = sys.modules[MS.TOPK_BIAS_MODULE].fused_topk_bias(
            hidden_states=hidden_states, gating_output=router_logits, scoring_func=self.scoring, e_score_correction_bias=self.bias,
            topk=K, renormalize=True, indices_type=None, input_tokens=input_ids, hash_indices_table=self.table, routed_scaling_factor=2.5,
            bias_vl=None, image_sentinel_lo=0)
        cache1 = self._launch(hidden_states, T(m * K, 2 * MIL), K, self.weight_on_input, m, 2 * MIL, H, w)
        cache2 = T(m * K, MIL)
        cache3 = self._launch(cache2, T(m * K, H), 1, not self.weight_on_input, m * K, H, MIL, w)
        out = T(m, H)
        sys.modules["vllm._custom_ops"].moe_sum(cache3.view(-1, K, H), out)
        self.seen = {"router_w": w, "router_ids": ids, "router_w.hash": w, "ids": ids, "up": cache1, "act": cache2, "down": cache3,
                     "sum": out}
        return out


def dsv4_moe(layers: range | list[int], **kw) -> torch.nn.Module:
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    for i in range(max(layers) + 1):
        layer = torch.nn.Module()
        layer.ffn = torch.nn.Module()
        layer.ffn.experts = MoERunner(i, **kw)
        root.model.layers.append(layer)
    return root


def run_step(model: torch.nn.Module, layers, m: int, gating_dtype: torch.dtype = torch.float32) -> None:
    for i in layers:
        model.model.layers[i].ffn.experts(T(m, H), T(m, E, dtype=gating_dtype), T(m, dtype=torch.int64))


def attached(monkeypatch, layers, **kw) -> tuple[types.SimpleNamespace, Committer, torch.nn.Module, MS.MoeBlockSource]:
    v = fake_dsv4(monkeypatch)
    v.unwrapped = (v.rt.fused_topk_bias, v.ops.moe_wna16_marlin_gemm)
    c = Committer()
    model = dsv4_moe(layers, **kw)
    src = MS.MoeBlockSource(c, model, manifest=AT_CALLS)
    src.install()
    assert (v.rt.fused_topk_bias, v.ops.moe_wna16_marlin_gemm) != v.unwrapped
    return v, c, model, src


def acquired(c: Committer) -> list[str]:
    return [n for kind, n in c.log if kind == "acquire"]


@pytest.mark.parametrize("positional", [False, True])
def test_every_layer_binds_the_maps_moe_names_from_marlins_launches_and_the_sqrtsoftplus_router(monkeypatch, positional):
    v, c, model, src = attached(monkeypatch, range(L), positional=positional)
    for step, m in enumerate((4, 300)):
        c._cur = types.SimpleNamespace(step=step)
        src.on_step_begin()
        start = len(c.log)
        run_step(model, range(L), m)
        log = c.log[start:]
        names = [n for kind, n in log if kind == "acquire"]
        got = CM.bind_names(CMAP, ENV, names, profile="0123456789ab", request="r0", t=0, phase="decode", routes=DECODE, rank=0,
                            binders=("moe",))
        assert got["unknown"] == [] and got["missing"] == [] and len(names) == len(set(names)) == len(PLANES) * L + HASH
        assert log == [e for i in range(L) for e in (
            *(("acquire", name(i, x)) for x in labels(i)[:-4]), ("acquire", name(i, "up")), ("flush", "up"),
            ("acquire", name(i, "act")), ("acquire", name(i, "down")), ("flush", "act"), ("acquire", name(i, "sum")))]
        want = CM.committer_names(CMAP, ENV, phase="decode", routes=DECODE, binders=("moe",))
        for i in range(L):
            seen = model.model.layers[i].ffn.experts.seen
            for label in labels(i):
                t, (tap, _layer) = c.acquired[name(i, label)], want[name(i, label)]
                if label != "in":
                    assert (t.base if label in ("router_ids", "ids") else t) is seen[label], name(i, label)
                assert str(t.dtype).replace("torch.", "") == tap["dtype"], name(i, label)
                assert list(t.shape) == [CM.dim(d, {**ENV, "M": m}, {}) for d in tap["shape"]], name(i, label)
            assert want[name(i, labels(i)[0])][0]["program"] == ("hash-router" if i < HASH else "sqrtsoftplus-router")
    src.uninstall()
    assert (v.rt.fused_topk_bias, v.ops.moe_wna16_marlin_gemm) == v.unwrapped
    assert v.calls.count("dsv4_topk") == 2 * (L - HASH) and v.calls.count("topk_hash_softplus_sqrt") == 2 * HASH
    assert src.stats["router"] == {"sqrtsoftplus:topk_hash_softplus_sqrt": 2 * HASH, "sqrtsoftplus:dsv4_topk": 2 * (L - HASH)}
    assert src.router_steps == {"topk_hash_softplus_sqrt": [0, 1], "dsv4_topk": [0, 1]}
    assert src.stats["plane_flushes"] == 2 * 2 * L and src.stats["tensors_per_layer"] == MS.TENSORS_PER_LAYER
    assert src.act_scaled is False and c.errors == [] and not hasattr(c, "x03_expected")
    decl = src.declaration_classes()[0]
    assert decl["tensors_per_layer"] == MS.TENSORS_PER_LAYER and decl["router_kernel_steps"] == src.router_steps
    assert decl["at_calls"] is True and decl["hash_layers"] == list(range(HASH))


def test_a_scored_router_call_that_dsv4_topk_refuses_is_not_bound(monkeypatch):
    """bf16 router logits fail `can_use_dsv4_topk`, so vLLM takes `topk_hash_softplus_sqrt` without a table: not the scored
    router's Program, so the source binds neither output and fails the step; a hash layer's call is still bound."""
    _v, c, model, src = attached(monkeypatch, [HASH - 1, HASH])
    run_step(model, [HASH - 1, HASH], 4, gating_dtype=torch.bfloat16)
    src.uninstall()
    names = acquired(c)
    assert {name(HASH - 1, x) for x in ("router_w.hash", "ids", "in")} <= set(names)
    assert not {name(HASH, "router_w"), name(HASH, "router_ids")} & set(names) and len(names) == 2 * len(PLANES) - 1
    assert c.x03_expected == 1 and any("dsv4_topk" in e for e in c.errors)
    assert src.stats["router"] == {"sqrtsoftplus:topk_hash_softplus_sqrt": 1}


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
def test_another_scoring_function_is_left_to_the_launch_it_reaches(monkeypatch, scoring):
    """`fused_topk_bias` binds only sqrtsoftplus: under softmax `topk_softmax`'s own wrap binds the router once, and under sigmoid
    (whose launch no wrap reaches) nothing binds it, which `identity_check` fails as two missing tensors."""
    _v, c, model, src = attached(monkeypatch, [HASH], scoring=scoring)
    run_step(model, [HASH], 4)
    src.uninstall()
    router = [n for n in acquired(c) if "/router_" in n]
    assert router == ([name(HASH, "router_w"), name(HASH, "router_ids")] if scoring == "softmax" else [])
    assert src.stats["router"] == {} and c.errors == []


def test_a_launch_with_the_router_weight_on_its_input_is_refused(monkeypatch):
    """With `apply_router_weight_on_input` the w13 launch multiplies by the router weights and the w2 launch does not: the w13
    launch (top_k K, not 1) is refused rather than bound as w2's planes."""
    _v, c, model, src = attached(monkeypatch, [HASH], weight_on_input=True)
    run_step(model, [HASH], 4)
    src.uninstall()
    assert c.x03_expected == 1 and any(f"mul_topk_weights True, top_k {K}" in e for e in c.errors)
    assert name(HASH, "act") not in acquired(c)


def test_a_manifest_that_names_moe_planes_has_them_bound_and_any_other_the_calls():
    """No manifest, or one with `moe_block_stream` rows, binds `moe/L<k>/<plane>`; DeepSeek-V4-Pro's binds under the MoE module.
    `layer_of` is the layer of the source's own names only: not the module's `out` (the TP partial source's) nor another module's."""
    assert MS.binds_planes(None) and MS.binds_planes({"identities": [{"family": "moe_block_stream"}]})
    assert not MS.binds_planes(AT_CALLS) and not MS.binds_planes({"identities": []})
    model = dsv4_moe([0, 3])
    planes, calls = MS.MoeBlockSource(Committer(), model), MS.MoeBlockSource(Committer(), model, manifest=AT_CALLS)
    assert planes.ports() == {} and calls.ports() == MS.PORTS
    assert [planes.layer_of(n) for n in ("moe/L3/up", "moe/L0/router_w", "moe/L3", name(3, "up"))] == [3, 0, None, None]
    assert [calls.layer_of(n) for n in (name(3, "up"), name(0, "in"), name(0, "ids"), name(3, "out"), name(3, "0"), "moe/L3/up",
                                        "model.layers.3.ffn.experts", "model.layers.3.ffn.gate/up")] == [3, 0, 0, None, None, None, None, None]


def test_a_hash_layers_in_is_each_tokens_id_once_per_gather_in_the_tables_dtype():
    """`fused_topk_bias` casts the token ids to the table's dtype; the K gathers of a token read its id, token by token."""
    ids = torch.tensor([7, 0, ENV["V"] - 1], dtype=torch.int64)
    got = MS.hash_router_input(ids, torch.zeros(ENV["V"], K, dtype=torch.int32), K)
    assert got.dtype == torch.int32 and got.tolist() == [t for t in ids.tolist() for _ in range(K)]


def manifest_rows(step: int, m: int) -> list[dict]:
    """One step's rows of the MoE layers as the manifest of record spells them on rank 0: the two Marlin GEMMs' Calls on every layer,
    and the router's (`dsv4_topk`'s ports 0 and 1 on a scored layer; `MoeRouterHash_v1`'s out, the gathers' ids and the token ids
    they read on a hash layer)."""
    def call(e: str, fx: str, spec: str, port: str = "out") -> dict:
        return {"family": "call_boundaries", "op_path": f"{e}/{fx}", "output_member": port, "spec": spec, "port": port}

    rows = []
    for i in range(L):
        e, n = f"model.layers.{i}.ffn.experts", 3 * i
        rows += [call(e, f"marlin_moe_mxfp4_gemm_{n}", f"MarlinMoeExpertGemm_v1{{M={m}}}"),
                 call(e, f"marlin_moe_mxfp4_gemm_{n + 1}", f"MarlinMoeExpertGemmW_v1{{M={m}}}")]
        if i < HASH:
            rows += [call(e, "topk_softplus_sqrt" + (f"_{i}" if i else ""), "MoeRouterHash_v1{E=384}"),
                     {"family": "instance_outputs", "op_path": e, "output_member": "ids", "spec": "GatherI32x129280_v1", "port": "out"},
                     {"family": "input_token_ids", "op_path": e, "output_member": "in", "spec": None}]
        else:
            rows += [call(e, f"triton_launch_{n + 2}", "MoeRouterSqrtSoftplus_v1", p) for p in ("0", "1")]
    return [{**r, "engine_step": step, "rank": 0} for r in rows]


def test_every_label_binds_its_row_of_the_manifest_of_record(monkeypatch):
    """Through `bind.Binder` with the source's `PORTS`: up and down at the two Marlin Calls (told apart by Definition), router_w and
    router_ids at `dsv4_topk`'s ports, router_w.hash at the hash router's Call, ids and in at the module's own rows; act and sum
    bind none (the GEMMs' Calls hold them)."""
    _v, c, model, src = attached(monkeypatch, range(L))
    run_step(model, range(L), 4)
    names = acquired(c)
    rows = manifest_rows(0, 4)
    got = Binder(rows, rank=0, ports=src.ports()).bind(0, names)
    bound = {n: f"{op}/{member}" for n, (op, member, _inv, how) in zip(names, got) if how != "unbound"}
    assert sorted(bound.values()) == sorted(f"{r['op_path']}/{r['output_member']}" for r in rows)
    n = 3 * HASH
    assert {k: bound[name(HASH, k)] for k in ("router_w", "router_ids", "up", "down")} == {
        "router_w": f"model.layers.{HASH}.ffn.experts/triton_launch_{n + 2}/0", "router_ids": f"model.layers.{HASH}.ffn.experts/triton_launch_{n + 2}/1",
        "up": f"model.layers.{HASH}.ffn.experts/marlin_moe_mxfp4_gemm_{n}/out", "down": f"model.layers.{HASH}.ffn.experts/marlin_moe_mxfp4_gemm_{n + 1}/out"}
    assert {k: bound[name(0, k)] for k in ("router_w.hash", "ids", "in")} == {
        "router_w.hash": "model.layers.0.ffn.experts/topk_softplus_sqrt/out", "ids": "model.layers.0.ffn.experts/ids",
        "in": "model.layers.0.ffn.experts/in"}
    assert set(names) - set(bound) == {name(i, x) for i in range(L) for x in ("act", "sum")}


def test_identity_check_holds_each_layer_to_its_values_and_a_hash_layer_to_one_more(monkeypatch):
    """A scored layer commits six values a step and a hash layer seven (`expected_manifest`'s `moe_hash_layers`); the source's names
    are none of the committer's modules (here the router gate's), so the hash layers' `<experts>/in` is no unexpected module."""
    _v, c, model, src = attached(monkeypatch, range(L))
    run_step(model, range(L), 4)
    gate = "model.layers.3.ffn.gate"
    names = [gate, *acquired(c)]
    com = NativeHostCommitter(None, run_id="dsv4-moe", gpu_tree=True)
    com._modules, com.sources = [(gate, torch.nn.Linear(1, 1))], [src]
    man = com.expected_manifest()
    assert man["modules"] == [gate] and man["moe_layers"] == list(range(L)) and man["moe_tensors_per_layer"] == 6
    assert man["moe_hash_layers"] == list(range(HASH))

    def check(steps: list[list[str]]) -> dict:
        com._layouts = {}
        for k, ns in enumerate(steps):
            com._layouts[k] = [TensorMeta(i, n, "bfloat16", (4, 8), 64, GPU_CHUNK_BYTES, i, 1) for i, n in enumerate(ns)]
            for i, meta in enumerate(com._layouts[k]):
                meta.dev_off = meta.stream_off = 64 * i
        return com.identity_check()

    assert check([names, names])["ok"] is True
    r = check([names, [n for n in names if n != name(0, "in")]])
    assert r["ok"] is False and r["violations"][0]["moe_layers_bad"] == {0: 6}
    r = check([names, [n for n in names if n != name(HASH, "act")]])
    assert r["ok"] is False and r["violations"][0]["moe_layers_bad"] == {HASH: 5}


def test_the_up_value_is_snapshotted_at_its_launch_whatever_its_prefix():
    """The w2 launch writes into `up`'s workspace: the committer copies `up` at the launch, not batched at the step's end."""
    assert hazard_of_record(name(3, "up"), frozenset()) and hazard_of_record("moe/L3/up", frozenset())
    assert not hazard_of_record(name(3, "down"), frozenset()) and not hazard_of_record(name(0, "in"), frozenset())
