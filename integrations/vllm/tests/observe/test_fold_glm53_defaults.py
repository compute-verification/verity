"""GLM-5.3's served-default kernels on cc 9.0, folded as the Build binds them (VLLM-14):

* FlashInfer's fused all-reduce + residual + RMSNorm is one collective Compute (`AllReduceRmsNormMnnvl_v1`), its new residual written
  back into the partial's buffer; another pattern code, quantisation or world is refused;
* vLLM's CUDA rms_norm / fused_add_rms_norm off batch invariance fold to `RMSNormCudaVector8_v1` / `RMSNormFusedCudaVector8_v1`;
* an unbiased bf16 `aten.linear` folds to cuBLAS's captured route, and a shape without one is refused;
* MLA's absorb `bmm` folds to `BatchedGemmWord_v1` per token over permuted views, the `out=` variant included;
* none of them claims another op.
"""
from __future__ import annotations

import dataclasses

import pytest

from verity_vllm.engine.profiles import profile_for_case
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, ViewDesc
from verity_vllm.observe.fold.patterns.collectives import TpFusedAllReduceRmsNorm
from verity_vllm.observe.fold.patterns.cublas import CublasLinear, MlaAbsorbBmm
from verity_vllm.observe.fold.patterns.norms import CudaFusedAddRmsNorm, CudaRmsNorm
from verity_vllm.observe.fold.patterns.target import target_of_header
from verity_vllm.observe.fold.resolver import Collective, Compute, PerRow, Shared, StepContext, Unsupported, spec_label
from verity_vllm.observe.fold.tree import build_trees

CUDA, H = "cuda:0", 6144
#: the served GLM-5.3 engine's declared target: cc 9.0, off batch invariance (the default-kernels ruling)
H200 = dataclasses.replace(target_of_header(None)[0], compute_capability=(9, 0), num_sms=132, batch_invariant=False)


def view(alloc, shape, dtype="bfloat16", strides=None, offset=0):
    if strides is None:
        out, acc = [], 1
        for s in reversed(shape):
            out.append(acc)
            acc *= s
        strides = tuple(reversed(out))
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides), dtype, CUDA)


def _ctx(M, roots=()):
    p = profile_for_case("QWEN3_4B_FP8")
    hdr = RunHeader("verity-capture/log/v1", "2026-10-15T00:00:00Z", "pid",
                    {"gpu": {"name": "NVIDIA H200", "compute_capability": [9, 0], "multi_processor_count": 132,
                             "tuned_gemm_arch_family": "hopper"}, "backends": {}, "target": H200.to_json()},
                    {"repo": "zai-org/GLM-5.3", "config": {}}, {"requests": []})
    rows = [("r0", i) for i in range(M)]
    start = StepStart(0, rows, list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0]}, 16, list(range(M)), [("r0", M - 1)],
                      [M - 1], 1, M, M)
    rmap: dict[int, list[Root]] = {}
    for r in roots:
        rmap.setdefault(r.view.alloc, []).append(r)
    return StepContext(hdr, start, None, [], rmap, p.config, p)


def _node(op, schema, args, kwargs=None, outs=None, module="model.layers.0"):
    ev = [DispatchEnter(1, None, 0, "forward", op, schema, list(args), kwargs or {}, 7, module), DispatchExit(1, outs)]
    return build_trees(ev)[0][0]


FAR_SCHEMA = ("vllm::flashinfer_trtllm_fused_allreduce_norm(Tensor(a0!) allreduce_in, Tensor(a1!) residual, Tensor rms_gamma, float rms_eps, "
              "SymInt world_size, bool launch_with_pdl, bool fp32_acc, SymInt max_token_num, SymInt pattern_code, Tensor(a9!)? norm_out=None, "
              "Tensor(a10!)? quant_out=None, Tensor(a11!)? scale_out=None, Tensor? scale_factor=None, float weight_bias=0.) -> ()")


def _far(M=1, world=8, code=1, fp32=True):
    return _node("vllm.flashinfer_trtllm_fused_allreduce_norm.default", FAR_SCHEMA,
                 [view(5623, (M, H)), view(5594, (M, H)), view(23, (H,)), 1e-05, world, True, fp32, 42, code, view(5624, (M, H))])


def test_the_fused_allreduce_norm_is_one_collective_compute_writing_the_residual_in_place():
    r = TpFusedAllReduceRmsNorm(world=8).match(_far(), _ctx(1))
    assert isinstance(r, Compute), r
    assert spec_label(r.spec) == "AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144,SHOT=1,CLUSTER=8,EPS=1e-05,FMA=True}" and r.rows == 1
    coll, res, w = r.operands
    assert isinstance(coll, Collective) and coll.kind == "fused_allreduce_rms_norm" and coll.site == "model.layers.0" and coll.own.alloc == 5623
    assert isinstance(res, PerRow) and res.view.alloc == 5594 and isinstance(w, Shared) and w.view.alloc == 23
    assert [o.view.alloc for o in r.outputs] == [5624, 5623]


def test_the_norms_name_their_weight_root_so_the_match_maps_each_layer_apart():
    gamma = Root(view(23, (H,)), "weight", "model.layers.3.post_attention_layernorm.weight")
    r = TpFusedAllReduceRmsNorm(world=8).match(_far(), _ctx(1, roots=(gamma,)))
    assert isinstance(r, Compute) and r.roots == (None, None, gamma.name)
    fa = _node("_C.fused_add_rms_norm.default",
               "_C::fused_add_rms_norm(Tensor($0! -> ) input, Tensor($1! -> ) residual, Tensor? weight, float epsilon) -> ()",
               [view(2315, (256, H)), view(2284, (256, H)), view(23, (H,)), 1e-05], module="")
    r = CudaFusedAddRmsNorm().match(fa, _ctx(256, roots=(gamma,)))
    assert isinstance(r, Compute) and r.hint == gamma.name


@pytest.mark.parametrize("kw", [{"world": 4}, {"code": 2}, {"fp32": False}])
def test_another_fused_allreduce_norm_configuration_is_refused(kw):
    assert isinstance(TpFusedAllReduceRmsNorm(world=8).match(_far(**kw), _ctx(1)), Unsupported)


def test_the_cuda_rms_norms_fold_to_the_vector8_definitions():
    rn = _node("_C.rms_norm.default", "_C::rms_norm(Tensor($0! -> ) result, Tensor input, Tensor? weight, float epsilon) -> ()",
               [view(2285, (256, H)), view(2284, (256, H)), view(22, (H,)), 1e-05], module="model.layers.0.input_layernorm")
    r = CudaRmsNorm().match(rn, _ctx(256))
    assert isinstance(r, Compute) and spec_label(r.spec).startswith("RMSNormCudaVector8_v1{N=6144,EPS=1e-05,BLOCK=")
    assert [o.view.alloc for o in r.operands] == [2284, 22] and [o.view.alloc for o in r.outputs] == [2285]
    fa = _node("_C.fused_add_rms_norm.default",
               "_C::fused_add_rms_norm(Tensor($0! -> ) input, Tensor($1! -> ) residual, Tensor? weight, float epsilon) -> ()",
               [view(2315, (256, H)), view(2284, (256, H)), view(23, (H,)), 1e-05], module="model.layers.0.post_attention_layernorm")
    r = CudaFusedAddRmsNorm().match(fa, _ctx(256))
    assert isinstance(r, Compute) and spec_label(r.spec).startswith("RMSNormFusedCudaVector8_v1{N=6144,EPS=1e-05,BLOCK=")
    assert [o.view.alloc for o in r.outputs] == [2315, 2284]
    assert CudaRmsNorm().match(fa, _ctx(256)) is None and CudaFusedAddRmsNorm().match(rn, _ctx(256)) is None


LIN_SCHEMA = "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"
WK = Root(view(7, (160, H)), "weight", "model.layers.0.self_attn.indexer.wk_weights_proj.weight")


def test_an_unbiased_linear_folds_to_its_captured_cublas_route():
    node = _node("aten.linear.default", LIN_SCHEMA, [view(2285, (1, H)), view(7, (160, H))], outs=view(2288, (1, 160)),
                 module="model.layers.0.self_attn.indexer.wk_weights_proj")
    r = CublasLinear().match(node, _ctx(1, [WK]))
    assert isinstance(r, Compute) and spec_label(r.spec).startswith("CublasSplitKGemm_v1{K=6144,N=160,SPLIT_K=4,")
    assert [type(o).__name__ for o in r.operands] == ["PerRow", "Shared"] and r.outputs[0].view.alloc == 2288


def test_a_linear_without_a_captured_route_or_weight_root_is_refused():
    odd = _node("aten.linear.default", LIN_SCHEMA, [view(2285, (1, 128)), view(9, (96, 128))], outs=view(2288, (1, 96)))
    assert isinstance(CublasLinear().match(odd, _ctx(1, [Root(view(9, (96, 128)), "weight", "w")])), Unsupported)
    node = _node("aten.linear.default", LIN_SCHEMA, [view(2285, (1, H)), view(7, (160, H))], outs=view(2288, (1, 160)))
    assert isinstance(CublasLinear().match(node, _ctx(1)), Unsupported)


BMM_SCHEMA = "aten::bmm(Tensor self, Tensor mat2) -> Tensor"
BMM_OUT_SCHEMA = "aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"
W_UK_T = Root(view(2, (8, 192, 512), strides=(448, 1, 3584)), "weight", "model.layers.0.self_attn.W_UK_T")


@pytest.mark.parametrize("M", [1, 256])
def test_the_mla_absorb_bmm_folds_per_token_over_permuted_views(M):
    x = view(5604, (8, M, 192), strides=(256, 2048, 1))
    node = _node("aten.bmm.default", BMM_SCHEMA, [x, W_UK_T.view], outs=view(5605, (8, M, 512)), module="model.layers.0.self_attn")
    r = MlaAbsorbBmm().match(node, _ctx(M, [W_UK_T]))
    assert isinstance(r, Compute) and r.rows == M
    assert spec_label(r.spec).startswith("BatchedGemm_v1{B=8,K=192,N=512,")
    xv, wv, ov = r.operands[0].view, r.operands[1].view, r.outputs[0].view
    assert (xv.shape, xv.strides) == ((M, 8, 192), (2048, 256, 1))
    assert (wv.shape, wv.strides) == ((8, 512, 192), (448, 3584, 1))
    assert ov.shape == (M, 8, 512)


def test_the_bmm_out_variant_writes_its_out_view():
    w_uv = Root(view(2, (8, 512, 256), strides=(448, 3584, 1), offset=192), "weight", "model.layers.0.self_attn.W_UV")
    node = _node("aten.bmm.out", BMM_OUT_SCHEMA, [view(5618, (8, 1, 512), strides=(512, 4096, 1)), w_uv.view],
                 kwargs={"out": view(5599, (8, 1, 256), strides=(256, 2048, 1))}, module="model.layers.0.self_attn")
    r = MlaAbsorbBmm().match(node, _ctx(1, [w_uv]))
    assert isinstance(r, Compute) and r.outputs[0].view.alloc == 5599 and r.outputs[0].view.shape == (1, 8, 256)


def test_a_bmm_whose_weight_is_no_parameter_view_is_refused():
    node = _node("aten.bmm.default", BMM_SCHEMA, [view(5604, (8, 1, 192), strides=(256, 2048, 1)), W_UK_T.view],
                 outs=view(5605, (8, 1, 512)))
    assert isinstance(MlaAbsorbBmm().match(node, _ctx(1)), Unsupported)


def test_a_transposed_weight_root_writes_one_run_per_inner_row_with_its_leaves_unchanged():
    """`memory.address_ordered`: W_UK_T's (448, 1, 3584)-strided view as 4,096 address runs, not one per element (VLLM-14)."""
    from verity.ml.circuits.refs import Affine
    from verity_vllm.observe.fold.memory import address_ordered
    from verity_vllm.observe.fold.views import element_runs

    v = W_UK_T.view
    r = Affine("p", 0, 1000, 8 * 192 * 512)
    pv, pr = address_ordered(v, r)
    assert len(element_runs(pv)) == 8 * 512 and len(element_runs(v)) == 8 * 192 * 512
    for b, k, n in ((0, 0, 0), (7, 191, 511), (3, 17, 300)):
        assert pv.offset + n * pv.strides[0] + b * pv.strides[1] + k * pv.strides[2] == b * 448 + k + n * 3584
        assert pr[n * 8 * 192 + b * 192 + k] == r[b * 192 * 512 + k * 512 + n]
    contiguous, refs = view(7, (160, H)), Affine("p", 0, 0, 160 * H)
    assert address_ordered(contiguous, refs) == (contiguous, refs)


def test_a_permuted_weight_root_answers_its_transposed_reads_from_its_own_map():
    """`MemoryModel.write_root`: W_UK_T and W_UV interleave in one allocation; a read of `W_UV^T` (one element run per element) or of
    W_UK_T comes from the root's address -> leaf map, equal leaf for leaf to the interval scan, until a write over it (VLLM-14)."""
    from verity.ml.circuits.refs import Affine, runs
    from verity_vllm.observe.fold.memory import MemoryModel

    def slow(m, v):
        a = m._allocs[v.alloc]
        keep, a.root_maps = a.root_maps, []
        try:
            return runs(m.read(v))
        finally:
            a.root_maps = keep

    w_uv = view(2, (8, 512, 256), strides=(448, 3584, 1), offset=192)
    m = MemoryModel()
    m.write_root(W_UK_T.view, Affine("p", 3, 0, 8 * 192 * 512))
    m.write_root(w_uv, Affine("p", 4, 0, 8 * 512 * 256))
    assert len(m._allocs[2].root_maps) == 2
    w_uv_t = view(2, (8, 256, 512), strides=(448, 1, 3584), offset=192)
    uk, uv_t = m.read(W_UK_T.view), m.read(w_uv_t)
    for h, k, n in ((0, 0, 0), (7, 191, 511), (3, 17, 300)):
        assert uk[(h * 192 + k) * 512 + n] == ("p", 3, (h * 192 + k) * 512 + n)
    for h, r, c in ((0, 0, 0), (7, 255, 511), (5, 100, 33)):
        assert uv_t[(h * 256 + r) * 512 + c] == ("p", 4, (h * 512 + c) * 256 + r)
    boxes = [view(2, (2, 16, 24), strides=(448, 1, 3584), offset=192 + 5 * 448 + 7 * 3584),
             view(2, (8, 24, 4), strides=(448, 3584, 1)), view(2, (4, 100, 7), strides=(448, 1, 3584), offset=448 + 3 + 3584 * 9)]
    for v in boxes:
        before = m.stats().extra["fast_reads"]
        assert runs(m.read(v)) == slow(m, v)
        assert m.stats().extra["fast_reads"] == before + 1
    straddling = view(2, (8, 300, 512), strides=(448, 1, 3584))       # W_UK_T's 192 columns and 108 of W_UV's
    assert m._root_map_read(m._allocs[2], straddling) is None
    m.write(view(2, (4,), offset=192 + 5 * 448 + 7 * 3584), Affine("n", 9, 0, 4))
    assert len(m._allocs[2].root_maps) == 0
    assert runs(m.read(boxes[0])) == slow(m, boxes[0])
    assert m.read(boxes[0])[0] == ("n", 9, 0)
