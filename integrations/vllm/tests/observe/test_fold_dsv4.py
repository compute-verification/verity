"""DeepSeek-V4-Pro's H200 folds (`observe.fold.patterns.dsv4`), each the Build's rule over the observer log:

* the mHC's TileLang kernels leave no event, so each wrapper's allocations resolve to its Calls: layer 0's pre_broadcast to
  `MhcPreBroadcast_v1` and `MhcResidualBroadcast_v1`, each fused post_pre to the path and split count the step's tokens select, the
  final call to `MhcPost_v1` and `HcHead_v1`; inputs are the previous call's outputs and the last all-reduce's, weights the
  sublayer's by name;
* a call whose split buffer disagrees with the dispatch's NS, or whose weights the capture has no root for, is refused;
* the KV compressors' `aten.mm.dtype` is the Build's f32-output cuBLAS GEMM on its captured route;
* the fused q / kv RMSNorm launch is a `TritonBlockRMSNorm_v1` of each token's q row and of its kv row, refused off the wrapper's
  launch or the PTX's 16-divisible specialization;
* the q norm / RoPE / window insert is the Build's `Dsv4QNormRopeKvInsert_v1` per token, its kv row's NoPE, scale bytes and RoPE
  written to the token's slot of the layer's KV group in fp8_ds_mla's block layout, one Compute per cache block.
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import dataclasses
import importlib

import pytest

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc
from verity_vllm.observe.fold.patterns.dsv4 import Mhc, QKvRmsNorm, QNormRopeKvInsert
from verity_vllm.observe.fold.patterns.target import target_of_header
from verity_vllm.observe.fold.resolver import Group, PerRow, Shared, StepContext, Unsupported, spec_label
from verity_vllm.observe.fold.tree import build_trees
from verity_vllm.program.frontend.rules.vllm_bindings.pins import QKV_RMS_PINS

CUDA, HC, H, OUT = "cuda:0", 4, 7168, 24
H200 = dataclasses.replace(target_of_header(None)[0], compute_capability=(9, 0), num_sms=132, batch_invariant=False)
PROFILE = importlib.import_module("verity_vllm.engine.profiles.derived_DSV4_PRO_tp8").PROFILE


def view(alloc, shape, dtype="bfloat16"):
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return ViewDesc(alloc, 0, tuple(shape), tuple(reversed(strides)), dtype, CUDA)


def _weights(L, side, broadcast=False):
    pre = f"model.layers.{L}"
    fn = (f"{pre}.hc_{side}_fn_broadcast", (OUT, H)) if broadcast else (f"{pre}.hc_{side}_fn", (OUT, HC * H))
    return [(fn[0], fn[1], "float32"), (f"{pre}.hc_{side}_scale", (3,), "float32"), (f"{pre}.hc_{side}_base", (OUT,), "float32"),
            (f"{pre}.{side}_norm.weight", (H,), "bfloat16")]


ROOT_NAMES = (_weights(0, "attn", True) + _weights(0, "ffn") + _weights(1, "attn") +
              [("model.hc_head_fn", (HC, HC * H), "float32"), ("model.hc_head_scale", (1,), "float32"), ("model.hc_head_base", (HC,), "float32")])
ROOTS = [Root(view(10 + i, s, d), "weight", n) for i, (n, s, d) in enumerate(ROOT_NAMES)]
ALLOC = {r.name: r.view.alloc for r in ROOTS}


def _ctx(T, roots=ROOTS):
    hdr = RunHeader("verity-capture/log/v1", "2026-10-18T00:00:00Z", "pid",
                    {"gpu": {"name": "NVIDIA H200", "compute_capability": [9, 0], "multi_processor_count": 132}, "backends": {},
                     "target": H200.to_json()}, {"repo": "deepseek-ai/DeepSeek-V4-Pro", "config": {}}, {"requests": []})
    start = StepStart(0, [("r0", i) for i in range(T)], list(range(T)), list(range(T)), {"r0": T}, {"r0": 0}, {"r0": [0]}, 64,
                      list(range(T)), [("r0", T - 1)], [T - 1], 1, T, T)
    rmap: dict[int, list[Root]] = {}
    for r in roots:
        rmap.setdefault(r.view.alloc, []).append(r)
    return StepContext(hdr, start, None, [], rmap, PROFILE.config, PROFILE)


def _node(op, outs, module, args=()):
    ev = [DispatchEnter(1, None, 0, "forward", op, "", list(args), {}, 7, module), DispatchExit(1, outs)]
    return build_trees(ev)[0][0]


def _all_reduce(alloc, T, module):
    return _node("vllm.all_reduce.default", view(alloc, (T, H)), module, [view(alloc - 1, (T, H))])


def _empties(module, views):
    return [_node("aten.empty.memory_format", v, module) for v in views]


def _pre_broadcast(T, NS=28):
    return [_all_reduce(101, T, "model.embed_tokens"),
            *_empties("model.layers.0", [view(200, (T, HC, H)), view(201, (T, HC), "float32"), view(202, (T, HC * HC), "float32"),
                                         view(203, (T, H)), view(204, (NS, T, OUT), "float32"), view(205, (NS, T), "float32")])]


def _post_pre(T, NS, base, module="model.layers.0", x=301):
    return [_all_reduce(x, T, f"{module}.attn.wo_b"),
            *_empties(module, [view(base, (NS, T, OUT), "float32"), view(base + 1, (NS, T), "float32"), view(base + 2, (T, HC, H)),
                               view(base + 3, (T, HC), "float32"), view(base + 4, (T, HC * HC), "float32"), view(base + 5, (T, H))])]


def _run(nodes, ctx):
    p = Mhc()
    out = [p.match(n, ctx) for n in nodes]
    assert all(r is None for r in out[:-1]), out
    return out[-1]


def test_layer_zeros_pre_broadcast_is_the_pre_and_the_streams_copy():
    ctx = _ctx(256)
    g = _run(_pre_broadcast(256), ctx)
    assert isinstance(g, Group) and [spec_label(c.spec) for c in g.parts] == [
        "MhcPreBroadcast_v1{HC=4,H=7168,NS=28,TN=64,REPEAT=20,EPS=1e-06,HC_EPS=1e-06}", "MhcResidualBroadcast_v1{HC=4,H=7168}"]
    pre, copy = g.parts
    assert pre.rows == copy.rows == 256 and isinstance(pre.operands[0], PerRow) and pre.operands[0].view.alloc == 101
    assert [o.view.alloc for o in pre.operands[1:]] == [ALLOC[n] for n, _, _ in _weights(0, "attn", True)]
    assert all(isinstance(o, Shared) for o in pre.operands[1:]) and pre.roots[1] == "model.layers.0.hc_attn_fn_broadcast"
    assert [o.view.alloc for o in pre.outputs] == [203, 201, 202] and [o.view.alloc for o in copy.outputs] == [200]


@pytest.mark.parametrize("T, NS, label", [(1, 4, "MhcFusedPostPre_v1{HC=4,H=7168,NS=4,TH=256,TN=64,REPEAT=20,EPS=1e-06,HC_EPS=1e-06}"),
                                          (256, 33, "MhcPostPreTf32_v1{HC=4,H=7168,NS=33,TN=64,REPEAT=20,EPS=1e-06,HC_EPS=1e-06}")])
def test_a_fused_post_pre_runs_the_path_and_splits_of_the_steps_tokens(T, NS, label):
    ctx = _ctx(T)
    assert isinstance(_run(_pre_broadcast(T), ctx), Group)
    g = _run(_post_pre(T, NS, 400), ctx)
    (c,) = g.parts
    assert spec_label(c.spec) == label and c.rows == T
    assert [o.view.alloc for o in c.operands] == [301, 200, 201, 202] + [ALLOC[n] for n, _, _ in _weights(0, "ffn")]
    assert [o.view.alloc for o in c.outputs] == [402, 405, 403, 404]
    g = _run(_post_pre(T, NS, 500, "model.layers.1", x=311), ctx)
    assert [o.view.alloc for o in g.parts[0].operands] == [311, 402, 403, 404] + [ALLOC[n] for n, _, _ in _weights(1, "attn")]


def test_the_final_call_is_the_post_then_the_head():
    ctx = _ctx(1)
    _run(_pre_broadcast(1), ctx)
    _run(_post_pre(1, 4, 400), ctx)
    g = _run([_all_reduce(601, 1, "model.layers.0.ffn.experts"), *_empties("model", [view(700, (1, HC, H)), view(701, (1, H))])], ctx)
    post, head = g.parts
    assert spec_label(post.spec) == "MhcPost_v1{HC=4,N=7168}" and [o.view.alloc for o in post.operands] == [601, 402, 403, 404]
    assert spec_label(head.spec) == "HcHead_v1{HC=4,H=7168,TH=128,EPS=1e-06,HC_EPS=1e-06}"
    assert [o.view.alloc for o in head.operands] == [700, ALLOC["model.hc_head_fn"], ALLOC["model.hc_head_scale"], ALLOC["model.hc_head_base"]]
    assert [o.view.alloc for o in head.outputs] == [701]


def _mm(T, N, w_alloc):
    w = view(w_alloc, (N, H))
    wt = ViewDesc(w.alloc, 0, (H, N), (1, H), "bfloat16", CUDA)
    return _node("aten.mm.dtype", view(900, (T, N), "float32"), "model.layers.0.attn", [view(901, (T, H)), wt])


@pytest.mark.parametrize("T, N, label", [
    (256, 1024, "CublasSplitKGemmF32Out_v1{K=7168,N=1024,SPLIT_K=2,DOT=HopperBF16WgmmaDot16_v1}"),
    (256, 2048, "GemmF32Out_v1{K=7168,N=2048,DOT=HopperBF16WgmmaDot16_v1}")])
def test_a_compressor_mm_is_the_builds_f32_output_cublas_gemm_over_its_weight(T, N, label):
    from verity_vllm.observe.fold.patterns.cublas import CublasMmF32Out
    w = Root(view(950, (N, H)), "weight", "model.layers.0.attn.compressor.wkv_gate.weight")
    c = CublasMmF32Out().match(_mm(T, N, 950), _ctx(T, ROOTS + [w]))
    assert spec_label(c.spec) == label and c.rows == T and c.roots[1] == w.name
    assert [o.view.alloc for o in c.operands] == [901, 950] and c.operands[1].view.shape == (N, H) and c.outputs[0].view.alloc == 900
    assert isinstance(CublasMmF32Out().match(_mm(T, N, 950), _ctx(T)), Unsupported), "a weight with no root is refused"


def _qkv_norm(T, q_offset=0, block=2048):
    args = {"q_ptr": ViewDesc(960, q_offset, (T, 1536), (2048, 1), "bfloat16", CUDA), "q_out_ptr": view(961, (T, 1536)),
            "q_weight_ptr": view(962, (1536,)), "q_in_stride": 2048, "q_out_stride": 1536,
            "kv_ptr": ViewDesc(960, q_offset + 1536, (T, 512), (2048, 1), "bfloat16", CUDA), "kv_out_ptr": view(963, (T, 512)),
            "kv_weight_ptr": view(964, (512,)), "kv_in_stride": 2048, "kv_out_stride": 512, "eps": 1e-6}
    consts = {"Q_SIZE": 1536, "KV_SIZE": 512, "BLOCK_SIZE": block, "launch_pdl": True}
    ev = TritonLaunch(1, None, 0, "forward", QKV_RMS_PINS["kernel"], QKV_RMS_PINS["src_sha256"], {**args, **consts}, consts, [T, 2], 8, 3, 7,
                      "model.layers.0.attn")
    return build_trees([ev])[0][0]


QKV_ROOTS = ROOTS + [Root(view(962, (1536,)), "weight", "model.layers.0.attn.q_norm.weight"),
                     Root(view(964, (512,)), "weight", "model.layers.0.attn.kv_norm.weight")]


@pytest.mark.parametrize("T", [256, 1])
def test_the_fused_q_kv_rmsnorm_is_a_block_rmsnorm_of_the_q_row_and_of_the_kv_row_per_token(T):
    g = QKvRmsNorm().match(_qkv_norm(T), _ctx(T, QKV_ROOTS))
    assert [spec_label(c.spec) for c in g.parts] == ["TritonBlockRMSNorm_v1{N=1536,BLOCK=2048,WARPS=8,EPS=1e-06}",
                                                     "TritonBlockRMSNorm_v1{N=512,BLOCK=2048,WARPS=8,EPS=1e-06}"]
    assert all(c.rows == T for c in g.parts) and [c.roots[1] for c in g.parts] == [r.name for r in QKV_ROOTS[-2:]]
    assert [[o.view.alloc for o in c.operands + c.outputs] for c in g.parts] == [[960, 962, 961], [960, 964, 963]]
    assert g.parts[1].operands[0].view.offset == 1536


def test_a_fused_q_kv_rmsnorm_off_the_wrappers_launch_or_ptx_specialization_or_without_weight_roots_is_refused():
    assert isinstance(QKvRmsNorm().match(_qkv_norm(1, block=4096), _ctx(1, QKV_ROOTS)), Unsupported)
    assert isinstance(QKvRmsNorm().match(_qkv_norm(1, q_offset=4), _ctx(1, QKV_ROOTS)), Unsupported), "a pointer 8 bytes off 16"
    assert isinstance(QKvRmsNorm().match(_qkv_norm(1), _ctx(1)), Unsupported)


BLOCK_BYTES, POS, CACHE_STRIDE = 64 * (576 + 8), 970, 1435968
CS_ROOT = Root(view(971, (4096, 64), "float32"), "weight", "model.layers.0.attn.rotary_emb.cos_sin_cache")


def _insert_ctx(T, slots, layer="model.layers.0.attn.swa_cache", roots=(CS_ROOT,)):
    ctx = _ctx(T, ROOTS + list(roots))
    group = {"layers": [layer], "block_size": 64, "block_tables": {}, "slot_mapping": list(slots)}
    start = dataclasses.replace(ctx.start, positions_view=view(POS, (T,), "int64"), kv_groups=[group, {"layers": ["other"]}])
    return dataclasses.replace(ctx, start=start)


def _insert(T, cache_bytes=BLOCK_BYTES):
    kw = {"q_in": ViewDesc(972, 0, (T, 16, 512), (8192, 512, 1), "bfloat16", CUDA), "kv": view(973, (T, 512)),
          "k_cache": ViewDesc(974, 0, (40, cache_bytes), (CACHE_STRIDE, 1), "uint8", CUDA), "slot_mapping": view(975, (T,), "int64"),
          "position_ids": view(POS, (T,), "int64"), "cos_sin_cache": CS_ROOT.view, "q_head_padded": 64, "eps": 1e-6, "cache_block_size": 64}
    ev = [DispatchEnter(1, None, 0, "forward", "_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.default", "", [], kw, 7, "model.layers.0.attn"),
          DispatchExit(1, view(976, (T, 64, 512)))]
    return build_trees(ev)[0][0]


def test_the_insert_is_the_builds_call_per_token_writing_each_blocks_window_rows():
    slots = [27 * 64 + 62, 27 * 64 + 63, 5 * 64]
    g = QNormRopeKvInsert().match(_insert(3), _insert_ctx(3, slots))
    assert [spec_label(c.spec) for c in g.parts] == ["Dsv4QNormRopeKvInsert_v1{NH=16,NHP=64,DN=448,DR=64,QB=64,EPS=1e-06}"] * 2
    first, second = g.parts
    assert (first.rows, second.rows) == (2, 1) and first.roots[2] == CS_ROOT.name
    assert first.operands[2].indices == (0, 1) and second.operands[0].view.offset == 2 * 8192
    nope, scale, rope = first.outputs[1:]
    base = 27 * CACHE_STRIDE
    assert (nope.view.offset, nope.view.strides, nope.view.dtype, nope.indices) == (base, (576, 1), "float8_e4m3fn", (62, 63))
    assert (scale.view.offset, scale.view.shape) == (base + 64 * 576, (64, 8))
    assert (rope.view.offset, rope.view.strides, rope.view.dtype) == ((base + 448) // 2, (288, 1), "bfloat16")
    assert second.outputs[1].view.offset == 5 * CACHE_STRIDE and second.outputs[1].indices == (0,)
    assert second.outputs[0].view.offset == 2 * 64 * 512


def test_an_insert_off_the_fp8_ds_mla_block_or_its_layers_kv_group_or_without_the_rotary_root_is_refused():
    assert isinstance(QNormRopeKvInsert().match(_insert(1, cache_bytes=64 * 576), _insert_ctx(1, [64])), Unsupported)
    assert isinstance(QNormRopeKvInsert().match(_insert(1), _insert_ctx(1, [64], layer="model.layers.1.attn.swa_cache")), Unsupported)
    assert isinstance(QNormRopeKvInsert().match(_insert(1), _insert_ctx(1, [64], roots=())), Unsupported)


def test_a_split_buffer_other_than_the_dispatchs_or_a_weight_without_a_root_is_refused():
    ctx = _ctx(1)
    _run(_pre_broadcast(1), ctx)
    assert isinstance(_run(_post_pre(1, 8, 400), ctx), Unsupported)
    assert isinstance(_run(_pre_broadcast(1, NS=27), _ctx(1)), Unsupported)
    r = _run(_pre_broadcast(1), _ctx(1, [r for r in ROOTS if not r.name.endswith("_broadcast")]))
    assert isinstance(r, Unsupported) and r.detail["missing"] == ["model.layers.0.hc_attn_fn_broadcast"]
