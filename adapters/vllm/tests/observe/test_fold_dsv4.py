"""DeepSeek-V4-Pro's H200 folds (`observe.fold.patterns.dsv4`, `dsv4_flashmla`), each the Build's rule over the observer log:

* the mHC's TileLang kernels leave no event, so each wrapper's allocations resolve to its Calls: layer 0's pre_broadcast to
  `MhcPreBroadcast_v1` and `MhcResidualBroadcast_v1`, each fused post_pre to the path and split count the step's tokens select, the
  final call's streams to `MhcPost_v1` and its head's allocation to `HcHead_v1`; inputs are the previous call's outputs and the last
  all-reduce's, weights the sublayer's by name;
* a call whose split buffer disagrees with the dispatch's NS, or whose weights the capture has no root for, is refused;
* the KV compressors' `aten.mm.dtype` is the Build's f32-output cuBLAS GEMM on its captured route;
* the fused q / kv RMSNorm launch is a `TritonBlockRMSNorm_v1` of each token's q row and of its kv row, refused off the wrapper's
  launch or the PTX's 16-divisible specialization;
* the q norm / RoPE / window insert is the Build's `Dsv4QNormRopeKvInsert_v1` per token, its kv row's NoPE, scale bytes and RoPE
  written to the token's slot of the layer's KV group in fp8_ds_mla's block layout, one Compute per cache block;
* save_partial_states moves each token's kv row to its compressor state slot and writes the Build's `Dsv4CompressorScore_v1{W}`
  of its score row and ape[pos % R] beside it, per cache block;
* the o-projection input's inverse RoPE and FP8 quantisation is the Build's `Dsv4InvRopeQuantHead_v1` per token and head, into its
  group's row and the wrapper's TMA-aligned scale layout, refused off the Definition's floor or that layout;
* DeepGEMM's wo_a einsum leaves only z's allocation after it, which is the Build's `ScaledMmFp8BlockGrouped_v1` per token over the
  quant's rows and wo_a's roots;
* the indexer's compressor (its state write and its compress) folds to nothing, as the Build's indexer leaf on the short path,
  and its compress off its pin or outside the indexer's compressor is refused;
* a head-512 compressor's compress, which leaves no event, is the attention's: at each window end its state rows' `Dsv4MlaCompressC128_v1`
  or `Dsv4MlaCompressC4_v1{ROWS}` into the layer's compressed cache, then the decode's `FlashMlaDecode_v1` over the window rows and the
  compressed rows at FlashMLA's planned splits (its -1 index constant emitted when no row is padding, as the Build's), or the prefill's `FlashMlaGatherRow_v1` of each workspace row and `FlashMlaPrefill_v1` per
  run of tokens with one NP.
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import dataclasses
import importlib

import pytest

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc
from verity_vllm.observe.fold.patterns.dsv4 import (INDEXER_COMPRESS_KERNEL, INDEXER_COMPRESS_SRC_SHA, INV_ROPE_KERNEL, INV_ROPE_SRC_SHA,
                                                    SAVE_STATES_KERNEL, SAVE_STATES_SRC_SHA, Fp8Einsum, IndexerCompress, InvRopeQuant, Mhc,
                                                    QKvRmsNorm, QNormRopeKvInsert, SavePartialStates)
from verity_vllm.observe.fold.patterns.dsv4_flashmla import SPARSE_DECODE_OP, SPARSE_PREFILL_OP, SparseDecode, SparsePrefill
from verity_vllm.observe.fold.patterns.target import target_of_header
from verity_vllm.observe.fold.resolver import Group, PerRow, Shared, StepContext, Unsupported, spec_label
from verity_vllm.observe.fold.tree import build_trees
from verity_vllm.program.frontend.rules.vllm_bindings.pins import QKV_RMS_PINS

CUDA, HC, H, OUT = "cuda:0", 4, 7168, 24
H200 = dataclasses.replace(target_of_header(None)[0], compute_capability=(9, 0), num_sms=132, batch_invariant=False)
PROFILE = importlib.import_module("verity_vllm.observe.fold.profiles.derived_DSV4_PRO_tp8").PROFILE


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


def test_the_final_call_is_the_post_at_streams_then_the_head_at_its_allocation():
    ctx = _ctx(1)
    _run(_pre_broadcast(1), ctx)
    _run(_post_pre(1, 4, 400), ctx)
    post = _run([_all_reduce(601, 1, "model.layers.0.ffn.experts"), *_empties("model", [view(700, (1, HC, H))])], ctx)
    assert spec_label(post.spec) == "MhcPost_v1{HC=4,N=7168}" and [o.view.alloc for o in post.operands] == [601, 402, 403, 404]
    assert [o.view.alloc for o in post.outputs] == [700]
    head = _run(_empties("model", [view(701, (1, H))]), ctx)
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


def _insert_ctx(T, slots, layer="model.layers.0.attn.swa_cache", roots=(CS_ROOT,), bs=64):
    ctx = _ctx(T, ROOTS + list(roots))
    group = {"layers": [layer], "block_size": bs, "block_tables": {}, "slot_mapping": list(slots)}
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
    g = QNormRopeKvInsert().match(_insert(3), ctx := _insert_ctx(3, slots))
    assert ctx.scratch["dsv4_cos_sin"] == {"model.layers.0.attn": (CS_ROOT.view, CS_ROOT.name)}, "the layer's compress reads its rotary table"
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


STATE_STRIDE = CACHE_STRIDE // 4
APE_ROOT = Root(view(980, (128, 512), "float32"), "weight", "model.layers.0.attn.compressor.ape")


def _save_states(T, W=512, R=128, bs=8, module="model.layers.0.attn.compressor", ape=APE_ROOT.view, width=None):
    args = {"kv_ptr": ViewDesc(981, 0, (T, W), (2 * W, 1), "float32", CUDA), "kv_stride": 2 * W,
            "score_ptr": ViewDesc(981, W, (T, W), (2 * W, 1), "float32", CUDA), "score_stride": 2 * W, "ape_ptr": ape, "ape_stride": W,
            "positions_ptr": view(POS, (T,), "int64"),
            "state_cache_ptr": ViewDesc(974, 0, (40, bs, width or 2 * W), (STATE_STRIDE, width or 2 * W, 1), "float32", CUDA),
            "state_cache_stride0": STATE_STRIDE, "state_cache_stride1": width or 2 * W, "slot_mapping_ptr": view(975, (T,), "int64"),
            "block_size": bs}
    consts = {"HEAD_SIZE": W, "TRITON_BLOCK_SIZE": W, "STATE_WIDTH": W, "COMPRESS_RATIO": R}
    ev = TritonLaunch(1, None, 0, "forward", SAVE_STATES_KERNEL, SAVE_STATES_SRC_SHA, {**args, **consts}, consts, [T], 4, 3, 7, module)
    return build_trees([ev])[0][0]


def test_save_partial_states_moves_each_kv_row_and_is_the_builds_score_call_per_token_into_its_state_slot():
    layer = "model.layers.0.attn.compressor.state_cache"
    g = SavePartialStates().match(_save_states(3), _insert_ctx(3, [27 * 8 + 6, 27 * 8 + 7, 5 * 8], layer, (APE_ROOT,), bs=8))
    kv1, score1, kv2, score2 = g.parts
    assert [p.effect for p in (kv1, kv2)] == ["move"] * 2 and [spec_label(c.spec) for c in (score1, score2)] == ["Dsv4CompressorScore_v1{W=512}"] * 2
    assert (kv1.rows, score1.rows, kv2.rows) == (2, 2, 1) and score1.roots[1] == APE_ROOT.name and score1.operands[1].indices == (0, 1)
    assert (kv1.src.view.offset, kv1.dst.view.offset, kv1.dst.view.strides, kv1.dst.indices) == (0, 27 * STATE_STRIDE, (1024, 1), (6, 7))
    assert (score1.operands[0].view.offset, score1.outputs[0].view.offset, score1.outputs[0].view.shape) == (512, 27 * STATE_STRIDE + 512, (8, 512))
    assert kv2.src.view.offset == 2 * 1024 and score2.outputs[0].view.offset == 5 * STATE_STRIDE + 512 and score2.outputs[0].indices == (0,)


def test_save_partial_states_records_a_head_512_compressors_window_ends_for_the_layers_attention():
    ape = Root(view(983, (4, 1024), "float32"), "weight", "model.layers.2.attn.compressor.ape")
    ctx = _insert_ctx(8, [16 + p for p in range(8)], "model.layers.2.attn.compressor.state_cache", (ape,), bs=4)
    SavePartialStates().match(_save_states(8, W=1024, R=4, bs=4, module="model.layers.2.attn.compressor", ape=ape.view), ctx)
    cache, W, R, ends = ctx.scratch["dsv4_compress"]["model.layers.2.attn"]
    assert (cache.shape, W, R, ends) == ((40, 4, 2048), 1024, 4, ((("r0", 3), 3), (("r0", 7), 7)))


def test_save_partial_states_off_its_state_layout_or_kv_group_or_without_the_ape_root_is_refused():
    layer = "model.layers.0.attn.compressor.state_cache"
    assert isinstance(SavePartialStates().match(_save_states(1, width=2048), _insert_ctx(1, [8], layer, (APE_ROOT,), bs=8)), Unsupported)
    assert isinstance(SavePartialStates().match(_save_states(1), _insert_ctx(1, [8], layer, (APE_ROOT,), bs=4)), Unsupported)
    assert isinstance(SavePartialStates().match(_save_states(1), _insert_ctx(1, [8], layer, (), bs=8)), Unsupported)


INV_ROPE_STATICS = {"heads_per_group": 8, "quant_group_size": 128, "chunks_per_head": 4, "rope_start": 64, "half_rope": 32,
                    "tma_aligned_scales": False, "fp8_max": 448.0, "n_groups": 2, "d": 4096, "scale_inner": 32}


def _inv_rope(T, tma, scale_strides=None, eps=1e-10):
    o, pos, cs = ViewDesc(990, 0, (T, 16, 512), (32768, 512, 1), "bfloat16", CUDA), view(POS, (T,), "int64"), CS_ROOT.view
    fp8 = ViewDesc(991, 0, (2, T, 4096), (T * 4096, 4096, 1), "float8_e4m3fn", CUDA)
    sc = ViewDesc(992, 0, (2, T, 32), scale_strides or (32 * tma, 1, tma), "float32", CUDA)
    kw = {"o": o, "positions": pos, "cos_sin_cache": cs, **INV_ROPE_STATICS, "tma_aligned_T": tma, "num_tokens": T}
    consts = {"heads_per_group": 8, "fp8_max": 448.0, "eps": eps, "QUANT_GROUP_SIZE": 128, "CHUNKS_PER_HEAD": 4, "ROPE_START": 64,
              "HALF_ROPE": 32, "TMA_ALIGNED_SCALES": False, "USE_GDC": True, "launch_pdl": True}
    args = {"o_ptr": o, "positions_ptr": pos, "cos_sin_cache_ptr": cs, "fp8_ptr": fp8, "scale_ptr": sc, "num_tokens": T, "o_stride_token": 32768,
            "o_stride_head": 512, "cache_stride_pos": 64, "fp8_stride_group": T * 4096, "fp8_stride_token": 4096,
            "scale_stride_group": sc.strides[0], "scale_stride_k": sc.strides[2], **consts}
    ev = [DispatchEnter(1, None, 0, "forward", "vllm.fused_inv_rope_fp8_quant_kernel.default", "", [], kw, 7, "model.layers.0.attn"),
          TritonLaunch(2, 1, 0, "forward", INV_ROPE_KERNEL, INV_ROPE_SRC_SHA, args, consts, [tma, 16], 1, 1, 7, "model.layers.0.attn"),
          DispatchExit(1, [fp8, sc])]
    return build_trees(ev)[0][0]


@pytest.mark.parametrize("T, tma", [(3, 4), (256, 256)])
def test_the_inverse_rope_quant_is_the_builds_head_call_per_token_into_its_group_row_and_tma_aligned_scales(T, tma):
    g = InvRopeQuant().match(_inv_rope(T, tma), _insert_ctx(T, []))
    assert [spec_label(c.spec) for c in g.parts] == ["Dsv4InvRopeQuantHead_v1{CHUNKS=4,QG=128,ROPE=64}"] * 16 and g.consumed == (1, 2)
    assert all(c.rows == T and c.roots[1] == CS_ROOT.name for c in g.parts) and [c.consumed for c in g.parts[:2]] == [(1,), ()]
    c = g.parts[9]
    head, cs = c.operands
    q, s = c.outputs
    assert (head.view.offset, head.view.shape, head.view.strides) == (9 * 512, (T, 512), (32768, 1)) and cs.indices == tuple(range(T))
    assert (q.view.offset, q.view.strides) == (T * 4096 + 512, (4096, 1))
    assert (s.view.offset, s.view.shape, s.view.strides) == (32 * tma + 4 * tma, (T, 4), (1, tma))


def test_an_inverse_rope_quant_off_the_definitions_floor_or_the_wrappers_scale_layout_or_without_the_rotary_root_is_refused():
    assert isinstance(InvRopeQuant().match(_inv_rope(1, 4, eps=1e-6), _insert_ctx(1, [])), Unsupported)
    assert isinstance(InvRopeQuant().match(_inv_rope(3, 4, scale_strides=(96, 1, 3)), _insert_ctx(3, [])), Unsupported), \
        "a chunk stride under the aligned rows lets a padding program's zero land on a live scale"
    assert isinstance(InvRopeQuant().match(_inv_rope(1, 4), _insert_ctx(1, [], roots=())), Unsupported)


WO_A_ROOTS = (CS_ROOT, Root(view(993, (2, 1024, 4096), "float8_e4m3fn"), "weight", "model.layers.0.attn.wo_a.weight"),
              Root(view(994, (2, 8, 32), "float32"), "weight", "model.layers.0.attn.wo_a.weight_scale_inv"))


def _einsum(T, z_shape=None, roots=WO_A_ROOTS):
    ctx = _insert_ctx(T, [], roots=roots)
    assert isinstance(InvRopeQuant().match(_inv_rope(T, 4), ctx), Group)
    return Fp8Einsum().match(_node("aten.empty.memory_format", view(995, z_shape or (T, 2, 1024)), "model.layers.0.attn"), ctx)


def test_the_o_projection_einsum_is_the_builds_grouped_call_per_token_into_z_after_the_inverse_rope():
    c = _einsum(3)
    assert spec_label(c.spec) == "ScaledMmFp8BlockGrouped_v1{H=2,K=4096,N=1024,G=128}" and c.rows == 3 and c.consumed == (1,)
    xq, sx, w, ws = c.operands
    assert (xq.view.alloc, xq.view.shape, xq.view.strides) == (991, (3, 2, 4096), (4096, 3 * 4096, 1))
    assert (sx.view.alloc, sx.view.shape, sx.view.strides) == (992, (3, 2, 32), (1, 128, 4))
    assert [o.view.alloc for o in (w, ws)] == [993, 994] and c.roots[2:] == tuple(r.name for r in WO_A_ROOTS[1:])
    assert c.outputs[0].view.alloc == 995
    ctx = _insert_ctx(3, [], roots=WO_A_ROOTS)
    assert Fp8Einsum().match(_node("aten.empty.memory_format", view(995, (3, 2, 1024)), "model.layers.0.attn"), ctx) is None, \
        "an allocation with no inverse RoPE before it is not the einsum's"


def test_an_einsum_allocation_off_z_or_without_the_wo_a_roots_is_refused():
    assert isinstance(_einsum(3, z_shape=(3, 2048)), Unsupported)
    assert isinstance(_einsum(3, roots=(CS_ROOT,)), Unsupported)


IX = "model.layers.2.attn.indexer.compressor"


def _indexer(T, src=INDEXER_COMPRESS_SRC_SHA, module=IX):
    consts = {"HEAD_SIZE": 128, "TRITON_BLOCK_SIZE": 128, "STATE_WIDTH": 256, "COMPRESS_RATIO": 4, "ROPE_HEAD_DIM": 64}
    ev = TritonLaunch(1, None, 0, "forward", INDEXER_COMPRESS_KERNEL, src, {"positions_ptr": view(POS, (T,), "int64"), **consts}, consts, [T],
                      1, 3, 7, module)
    return build_trees([ev])[0][0]


def test_the_indexers_compressor_folds_to_nothing_as_the_builds_indexer_leaf_on_its_short_path():
    r = IndexerCompress().match(_indexer(8), _ctx(8))
    assert (r.effect, r.consumed, r.pattern) == ("noop", (1,), "dsv4.indexer_compress")
    r = SavePartialStates().match(_save_states(8, W=256, R=4, bs=4, module=IX), ctx := _ctx(8))
    assert (r.effect, r.consumed) == ("noop", (1,)) and "dsv4_compress" not in ctx.scratch, "its window ends are no attention's"


def test_an_indexer_compress_off_its_pin_or_outside_the_indexers_compressor_is_refused():
    assert isinstance(IndexerCompress().match(_indexer(8, src="0" * 64), _ctx(8)), Unsupported)
    assert isinstance(IndexerCompress().match(_indexer(8, module="model.layers.2.attn.compressor"), _ctx(8)), Unsupported)


L0, L2, SCALE = "model.layers.0.attn", "model.layers.2.attn", 0.04419417382415922
SWA_BT, KV_BT, STATE_BT, KV_OFFSET, NEG1 = [21, 22, 23, 24, 25], [11, 12], list(range(39, -1, -1)), 37440, 0xFFFFFFFF
ATTN_ROOTS = {L: (Root(view(1010 + i, (64,), "float32"), "weight", f"{L}.attn_sink"),
                  Root(view(1012 + i, (512,)), "weight", f"{L}.compressor.norm.weight")) for i, L in enumerate((L0, L2))}
DOT = "DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922"


def _cache(offset, rows):
    return ViewDesc(1020, offset, (40, rows, 1, 584), (CACHE_STRIDE, 584, 584, 1), "uint8", CUDA)


def _attn_ctx(layer, positions, ends=(), R=128, W=512, bs=8, roots=None, buffers=True):
    """A step of request r0's tokens at `positions` in `layer`: its compressed, window and state KV groups, the capture's compressed and
    window caches, and what the layer's insert and state write leave its attention (the rotary table, the window ends)."""
    ctx = _ctx(len(positions), ROOTS + [CS_ROOT, *(ATTN_ROOTS[layer] if roots is None else roots)])
    groups = [{"layers": [layer], "block_size": 256, "block_tables": {"r0": KV_BT}},
              {"layers": [f"{layer}.swa_cache"], "block_size": 64, "block_tables": {"r0": SWA_BT}},
              {"layers": [f"{layer}.compressor.state_cache"], "block_size": bs, "block_tables": {"r0": STATE_BT}}]
    start = dataclasses.replace(ctx.start, rows=[("r0", p) for p in positions], positions=list(positions), kv_groups=groups)
    caches = {f"kv_cache[{layer}.swa_cache_layer]": _cache(0, 64), f"kv_cache[{layer}]": _cache(KV_OFFSET, 256 // R)}
    ctx = dataclasses.replace(ctx, start=start, buffers=caches if buffers else {})
    state = ViewDesc(1021, 0, (40, bs, 2 * W), (STATE_STRIDE, 2 * W, 1), "float32", CUDA)
    ctx.scratch.update(dsv4_cos_sin={layer: (CS_ROOT.view, CS_ROOT.name)}, dsv4_compress={layer: (state, W, R, tuple((("r0", p), p) for p in ends))})
    return ctx


def _dispatch(op, kw, outs, module):
    return build_trees([DispatchEnter(1, None, 0, "forward", op, "", [], kw, 7, module), DispatchExit(1, outs)])[0][0]


def _decode(layer, T=1, R=128):
    q, out = (ViewDesc(a, 0, (T, 1, 64, 512), (32768, 32768, 512, 1), "bfloat16", CUDA) for a in (1030, 1031))
    kw = {"q": q, "kv": _cache(0, 64), "attn_sink": ATTN_ROOTS[layer][0].view, "extra_kv": _cache(KV_OFFSET, 256 // R), "d_v": 512,
          "sm_scale": SCALE, "out_": out}
    return _dispatch(f"{SPARSE_DECODE_OP}.default", kw, [out, view(1032, (T, 64, 1), "float32")], layer)


def _prefill(layer, T, ws_rows):
    q, out = (ViewDesc(a, 0, (T, 64, 512), (32768, 512, 1), "bfloat16", CUDA) for a in (1040, 1041))
    kw = {"q": q, "kv": view(1042, (ws_rows, 1, 512)), "sm_scale": SCALE, "d_v": 512, "attn_sink": ATTN_ROOTS[layer][0].view, "out_": out}
    return _dispatch(f"{SPARSE_PREFILL_OP}.default", kw, [out, view(1043, (T, 64), "float32")], layer)


def _window_row(s, block_table=SWA_BT):
    return block_table[s // 64] * CACHE_STRIDE + s % 64 * 576


def test_the_decode_at_a_window_end_is_the_layers_compress_then_the_builds_flashmla_decode_over_its_window_and_compressed_rows():
    g = SparseDecode().match(node := _decode(L0), _attn_ctx(L0, [255], ends=(255,)))
    compress, attn = g.parts
    assert spec_label(compress.spec) == "Dsv4MlaCompressC128_v1{W=8,R=16,D=512,QB=64,EPS=1e-06}" and compress.rows == 1
    kv, score, w, cs = compress.operands
    state = [STATE_BT[s // 8] * STATE_STRIDE + s % 8 * 1024 for s in range(128, 256)]
    assert [p.offset for p in kv.rows[0]] == state and [p.offset for p in score.rows[0]] == [o + 512 for o in state]
    assert w.view == ATTN_ROOTS[L0][1].view and cs.indices == (128,) and compress.roots[2:] == (ATTN_ROOTS[L0][1].name, CS_ROOT.name)
    nope, rope, sc = compress.outputs
    base = KV_OFFSET + KV_BT[0] * CACHE_STRIDE
    assert (nope.view.offset, nope.view.shape, nope.indices) == (base, (2, 448), (1,)) and (sc.view.offset, sc.indices) == (base + 2 * 576, (1,))
    assert spec_label(attn.spec) == f"FlashMlaDecode_v1{{NH=16,SPLITS=(0,1,2,3),NW=128,DN=448,DR=64,QT=64,BN=64,{DOT}}}"
    q, nopes, scales, ropes, idx, swa, nc, sink = attn.operands
    assert (q.view.shape, q.view.strides) == ((1, 16, 512), (32768, 512, 1)) and attn.consumed == (node.id,)
    comp = [base, base + 576]
    assert [p.offset for p in nopes.rows[0]] == [_window_row(s) for s in range(128, 256)] + comp + comp[:1] * 62
    assert scales.rows[0][0].offset == SWA_BT[2] * CACHE_STRIDE + 64 * 576 and scales.rows[0][0].shape == (7,)
    assert ropes.rows[0][-1].offset == (base + 448) // 2 and ropes.rows[0][-1].shape == (64,)
    assert [(c.bits, c.count) for c in idx.rows[0]] == [(0, 128), (NEG1, 0), (0, 2), (NEG1, 62)] and (swa.bits, nc.bits) == (128, 2), \
        "a run of no copies keeps its constant, as the Build's [i32(-1)] * 0"
    assert sink.view.shape == (16,) and attn.roots[-1] == ATTN_ROOTS[L0][0].name


def test_a_ratio_4_compress_reads_the_state_rows_at_and_past_position_0_from_each_halfs_second_head():
    g = SparseDecode().match(_decode(L2, R=4), _attn_ctx(L2, [3], ends=(3,), R=4, W=1024, bs=4))
    compress, attn = g.parts
    assert spec_label(compress.spec) == "Dsv4MlaCompressC4_v1{ROWS=4,D=512,QB=64,EPS=1e-06}" and compress.operands[3].indices == (0,)
    state = [STATE_BT[0] * STATE_STRIDE + s * 2048 + 512 for s in range(4)]
    assert [p.offset for p in compress.operands[0].rows[0]] == state and [p.offset for p in compress.operands[1].rows[0]] == [o + 1024 for o in state]
    assert compress.outputs[0].view.offset == KV_OFFSET + KV_BT[0] * CACHE_STRIDE and compress.outputs[0].indices == (0,)
    assert spec_label(attn.spec).startswith("FlashMlaDecode_v1{NH=16,") and ",NW=64," in spec_label(attn.spec)
    assert [(c.bits, c.count) for c in attn.operands[4].rows[0]] == [(0, 4), (NEG1, 60), (0, 1), (NEG1, 63)]


def test_the_prefill_is_the_layers_compress_the_builds_gather_of_each_workspace_row_and_its_prefill_per_run_of_one_np():
    g = SparsePrefill().match(node := _prefill(L0, 130, 131), _attn_ctx(L0, range(130), ends=(127,)))
    compress, gather, short, long = g.parts
    assert spec_label(compress.spec) == "Dsv4MlaCompressC128_v1{W=8,R=16,D=512,QB=64,EPS=1e-06}" and compress.operands[3].indices == (0,)
    assert len(compress.operands[0].rows[0]) == 128 and compress.outputs[0].indices == (0,)
    assert spec_label(gather.spec) == "FlashMlaGatherRow_v1{DN=448,DR=64,QT=64}" and gather.rows == 131
    assert [r[0].offset for r in gather.operands[0].rows] == [KV_OFFSET + KV_BT[0] * CACHE_STRIDE] + [_window_row(s) for s in range(130)]
    assert (gather.outputs[0].view.shape, gather.outputs[0].view.strides) == ((131, 512), (512, 1))
    assert gather.attribution[:2] == (("r0", 127), ("r0", 0)) and gather.consumed == ()
    assert spec_label(short.spec) == f"FlashMlaPrefill_v1{{NH=16,NP=1,D=512,BN=64,{DOT}}}" and short.rows == 127
    assert spec_label(long.spec) == f"FlashMlaPrefill_v1{{NH=16,NP=2,D=512,BN=64,{DOT}}}" and long.rows == 3
    q, kv, idx, n, skv, sink = long.operands
    assert (q.view.offset, q.view.shape) == (127 * 32768, (3, 16, 512)) and (short.consumed, long.consumed) == ((node.id,), ())
    assert [p.offset for p in kv.rows[0]] == [0] + [512 * (1 + s) for s in range(128)] + [0] * 127, "compressed row 0, then the window"
    assert [p.offset for p in kv.rows[2]] == [0] + [512 * (1 + s) for s in range(2, 130)] + [0] * 127
    assert [(c.bits, c.count) for c in idx.rows[0]] == [(0, 129), (NEG1, 127)] and n.rows[0][0].bits == 129 and skv.bits == 1
    assert [p.offset for p in short.operands[1].rows[0]] == [512] * 128 and short.operands[3].rows[0][0].bits == 1


def test_a_sparse_attention_over_a_batch_or_without_its_compress_caches_or_sink_root_is_refused():
    assert isinstance(SparseDecode().match(_decode(L0, T=2), _attn_ctx(L0, [5, 9])), Unsupported), "the batch's schedule is not folded"
    ctx = _attn_ctx(L0, [255], ends=(255,))
    ctx.scratch["dsv4_compress"].clear()
    assert isinstance(SparseDecode().match(_decode(L0), ctx), Unsupported), "a compress whose state write did not resolve"
    assert isinstance(SparseDecode().match(_decode(L0), _attn_ctx(L0, [5], roots=ATTN_ROOTS[L0][1:])), Unsupported)
    assert isinstance(SparseDecode().match(_decode(L2, R=4), _attn_ctx(L2, [4099], R=4, W=1024, bs=4)), Unsupported), \
        "more compressed rows than the indexer's top-k"
    assert isinstance(SparsePrefill().match(_prefill(L0, 130, 131), _attn_ctx(L0, range(130), ends=(127,), buffers=False)), Unsupported)
    ctx = _attn_ctx(L0, range(2))
    ctx = dataclasses.replace(ctx, start=dataclasses.replace(ctx.start, rows=[("r0", 0), ("r1", 0)]))
    assert isinstance(SparsePrefill().match(_prefill(L0, 2, 2), ctx), Unsupported), "the chunk plan's workspace is folded at one request"


def test_a_split_buffer_other_than_the_dispatchs_or_a_weight_without_a_root_is_refused():
    ctx = _ctx(1)
    _run(_pre_broadcast(1), ctx)
    assert isinstance(_run(_post_pre(1, 8, 400), ctx), Unsupported)
    assert isinstance(_run(_pre_broadcast(1, NS=27), _ctx(1)), Unsupported)
    r = _run(_pre_broadcast(1), _ctx(1, [r for r in ROOTS if not r.name.endswith("_broadcast")]))
    assert isinstance(r, Unsupported) and r.detail["missing"] == ["model.layers.0.hc_attn_fn_broadcast"]
