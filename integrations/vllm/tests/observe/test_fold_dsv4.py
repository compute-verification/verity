"""DeepSeek-V4-Pro's H200 folds (`observe.fold.patterns.dsv4`), each the Build's rule over the observer log:

* the mHC's TileLang kernels leave no event, so each wrapper's allocations resolve to its Calls: layer 0's pre_broadcast to
  `MhcPreBroadcast_v1` and `MhcResidualBroadcast_v1`, each fused post_pre to the path and split count the step's tokens select, the
  final call to `MhcPost_v1` and `HcHead_v1`; inputs are the previous call's outputs and the last all-reduce's, weights the
  sublayer's by name;
* a call whose split buffer disagrees with the dispatch's NS, or whose weights the capture has no root for, is refused.
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import dataclasses
import importlib

import pytest

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, ViewDesc
from verity_vllm.observe.fold.patterns.dsv4 import Mhc
from verity_vllm.observe.fold.patterns.target import target_of_header
from verity_vllm.observe.fold.resolver import Group, PerRow, Shared, StepContext, Unsupported, spec_label
from verity_vllm.observe.fold.tree import build_trees

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


def test_a_split_buffer_other_than_the_dispatchs_or_a_weight_without_a_root_is_refused():
    ctx = _ctx(1)
    _run(_pre_broadcast(1), ctx)
    assert isinstance(_run(_post_pre(1, 8, 400), ctx), Unsupported)
    assert isinstance(_run(_pre_broadcast(1, NS=27), _ctx(1)), Unsupported)
    r = _run(_pre_broadcast(1), _ctx(1, [r for r in ROOTS if not r.name.endswith("_broadcast")]))
    assert isinstance(r, Unsupported) and r.detail["missing"] == ["model.layers.0.hc_attn_fn_broadcast"]
