"""`fp8.DynamicDeepGemmFp8Linear`: vLLM's default Hopper block-FP8 linear op, folded to the Definitions the Build's
`DynamicDeepGemmFp8LinearRule` emits (VLLM-14).

* below 32 rows: `FlashInferFp8Linear_v1{K,N}`, its one output the op's;
* from 32 rows: `DeepGemmFp8Linear_v1{K,N}`, (q, s, y) with only y in memory, so q and s are never written;
* the weight and its scale are the op's own parameters, named per operand, the op's whole subtree consumed;
* other knobs, shapes or a weight that is no checkpoint parameter are refused, never approximated;
* the fold leaves a `None` output unwritten.
"""
from __future__ import annotations

import pytest

from verity_vllm.engine.profiles import profile_for_case
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, ViewDesc
from verity_vllm.observe.fold.patterns.fp8 import DynamicDeepGemmFp8Linear
from verity_vllm.observe.fold.resolver import Compute, PerRow, Shared, StepContext, Unsupported, spec_label
from verity_vllm.observe.fold.tree import build_trees

CUDA = "cuda:0"
K, N = 6144, 2624
A_X, A_W, A_WS, A_OUT = 101, 10, 11, 102
OP = "vllm.dynamic_flashinfer_deepgemm_blockscale_gemm.default"
SCHEMA = ("vllm::dynamic_flashinfer_deepgemm_blockscale_gemm(Tensor input, Tensor weight, Tensor weight_scale, int group_size, "
          "bool use_deep_gemm_e8m0) -> Tensor")
W_NAME = "model.layers.0.self_attn.fused_qkv_a_proj.weight"
WS_NAME = "model.layers.0.self_attn.fused_qkv_a_proj.weight_scale_inv"


def view(alloc, shape, dtype):
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return ViewDesc(alloc, 0, tuple(shape), tuple(reversed(strides)), dtype, CUDA)


def _ctx(M, roots=True):
    p = profile_for_case("QWEN3_4B_FP8")
    hdr = RunHeader("verity-capture/log/v1", "2026-10-15T00:00:00Z", "pid",
                    {"gpu": {"name": "NVIDIA H200", "compute_capability": [9, 0], "multi_processor_count": 132,
                             "tuned_gemm_arch_family": "hopper"}, "backends": {}}, {"repo": "zai-org/GLM-5.3", "config": {}}, {"requests": []})
    rows = [("r0", i) for i in range(M)]
    start = StepStart(0, rows, list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0]}, 16, list(range(M)), [("r0", M - 1)],
                      [M - 1], 1, M, M)
    rmap: dict[int, list[Root]] = {}
    if roots:
        rmap = {A_W: [Root(view(A_W, (N, K), "float8_e4m3fn"), "weight", W_NAME)],
                A_WS: [Root(view(A_WS, (-(-N // 128), K // 128), "float32"), "weight", WS_NAME)]}
    return StepContext(hdr, start, None, [], rmap, p.config, p)


def _node(M, group=128, ue8m0=True, out_n=N):
    args = [view(A_X, (M, K), "bfloat16"), view(A_W, (N, K), "float8_e4m3fn"), view(A_WS, (-(-N // 128), K // 128), "float32"), group, ue8m0]
    ev = [DispatchEnter(1, None, 0, "forward", OP, SCHEMA, args, {}, 7, "model.layers.0.self_attn.fused_qkv_a_proj"),
          DispatchExit(1, view(A_OUT, (M, out_n), "bfloat16"))]
    return build_trees(ev)[0][0]


@pytest.mark.parametrize("M, label, outputs", [(1, "FlashInferFp8Linear_v1", ("PerRow",)),
                                                (31, "FlashInferFp8Linear_v1", ("PerRow",)),
                                                (32, "DeepGemmFp8Linear_v1", (None, None, "PerRow")),
                                                (256, "DeepGemmFp8Linear_v1", (None, None, "PerRow"))])
def test_the_op_folds_to_the_builds_definition_for_its_rows(M, label, outputs):
    r = DynamicDeepGemmFp8Linear().match(_node(M), _ctx(M))
    assert isinstance(r, Compute), r
    assert spec_label(r.spec) == f"{label}{{K={K},N={N}}}" and r.rows == M
    assert [type(o).__name__ for o in r.operands] == ["PerRow", "Shared", "Shared"]
    assert tuple(None if o is None else type(o).__name__ for o in r.outputs) == outputs
    out = r.outputs[-1]
    assert isinstance(out, PerRow) and out.view.alloc == A_OUT and out.view.shape == (M, N)
    assert isinstance(r.operands[1], Shared) and r.operands[1].view.shape == (N, K)
    assert r.roots == (None, W_NAME, WS_NAME) and r.consumed == (1,)


@pytest.mark.parametrize("node_kw, ctx_kw, why", [
    ({"group": 64}, {}, "group_size 128"),
    ({"ue8m0": False}, {}, "group_size 128"),
    ({"out_n": N + 1}, {}, "shapes are not"),
    ({}, {"roots": False}, "not checkpoint parameters"),
])
def test_other_knobs_shapes_or_weights_are_refused(node_kw, ctx_kw, why):
    r = DynamicDeepGemmFp8Linear().match(_node(4, **node_kw), _ctx(4, **ctx_kw))
    assert isinstance(r, Unsupported) and why in r.reason, r


def test_other_ops_are_not_this_pattern():
    ev = [DispatchEnter(1, None, 0, "forward", "aten.mm.default", "aten::mm(Tensor self, Tensor mat2) -> Tensor",
                        [view(A_X, (4, K), "bfloat16"), view(A_W, (K, N), "bfloat16")], {}, 7, "m"), DispatchExit(1, view(A_OUT, (4, N), "bfloat16"))]
    assert DynamicDeepGemmFp8Linear().match(build_trees(ev)[0][0], _ctx(4)) is None
