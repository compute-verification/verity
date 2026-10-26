"""lane ov-moe — where a data-dependent gather can and cannot sit in a Verity Program (the representability statement of
REPORT.md, as three hand-built one-call Programs under `vu_query`).

`vu_query` (D82) decides a child unit's admissibility from the reference structure alone: a child is a VU iff it fits the
leaf budget, its returns are among the instance's returns and EVERY argument gate resolves outside the instance (a `p`-space
reference chasing to the instance's own operands).  An argument produced by another node of the body (`n`-space) is an
interior value: the child yields nothing and is not descended into.  Hence:

  A. `RowGatherThenGemm(idx, X, W)` — the kernel's schedule made literal: gather the routed token row `X[idx]` at the top
     of the body, then a dense GEMM on the gathered row.  The gather node's returns are interior, the GEMM's `x` operand
     is interior -> as soon as the instance is oversize there is NO unit at all (`vus == 0`, granularity `none`): the
     whole grouped GEMM is zero-mass under sampling.  Real sizes (X = M x 2048, W13 = 64 x 4096 x 2048 leaves) are always
     oversize, so this shape is unverifiable by G5.
  B. `MoeExpertGemm(x, e, w)` (quarantine/ov_moe) — pair-major: the token row is a STATIC slice (pairs are enumerated by
     (token, slot), the shape of `topk_ids`, never by expert block), the routing index `e` is an ordinary I32 VALUE
     operand and the data-dependent selection is the per-element gather over the expert axis INSIDE the coordinate.
     Every coordinate reads only operands -> N `MoeExpertCoordinate` units tile the returns.
  C. the "one giant composite over the whole launch" `FusedMoEPairs(X, ids, W)` = batch of B over the M x S pairs.  It is
     representable and decomposes to the same coordinate units (depth 2): the VALUES of `topk_ids` are never needed to
     enumerate units — only a per-EXPERT enumeration (which token rows did expert e read?) would need them, and that is
     the kernel's schedule (`sorted_token_ids` / `expert_ids`), not the Program's dataflow.
"""
from __future__ import annotations

import pytest

from verity_vllm.check.replay import vu_query as vq
from verity_vllm.observe.fold.fold import FoldResult
from verity_vllm.program.replay import replayer
from verity.ml.circuits.defs import Builder, CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, array_of, strided_view
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16, I32
from verity_vllm.program.registry.quarantine.ov_moe.moe import MoeExpertGemm
from verity.ml.circuits.types import Array

M, S, E, K, N = 4, 2, 4, 16, 8


def one_call_result(spec) -> FoldResult:
    """A FoldResult whose flat body is the single instance `spec(*params)` (what `difftest.evaluate_spec` builds)."""
    params, ret = spec.signature()
    B = Builder(params, ret)
    colls = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    body = B.finish(B.call(spec, *colls))
    defn = CompositeDefinition("Toy", 1, (), lambda S_, _p=params, _r=ret: (_p, _r), replayer(body), doc="", conformance="derived: test",
                               register=False)
    top = bind(defn)
    top.set_body(body)
    return FoldResult(program=Program(top), spec=top, report=None, digest="", outputs={},
                      instances=[{"node": 0, "spec": spec.id, "step": 0, "event": 0}], body=body)


def _row_gather_then_gemm(B, S_, idx, X, W):
    cols = Coll(Array(S_.K, Array(S_.M, BF16)), strided_view(X, 0, S_.M, 1, S_.K, S_.K * S_.M))   # column k over the M rows
    xrow = B.batch(P.gather_bf16(S_.M), idx, cols, axes=(None, 0))                                # X[idx]: an interior value
    return B.call(bind(b1.Gemm, K=S_.K, N=S_.N), xrow, W)


RowGatherThenGemm = CompositeDefinition(
    "RowGatherThenGemm", 1, ("M", "K", "N"),
    lambda S_: ((("idx", I32), ("X", Array(S_.M, Array(S_.K, BF16))), ("W", Array(S_.N, Array(S_.K, BF16)))), Array(S_.N, BF16)),
    _row_gather_then_gemm, doc="shape A: the routed row gathered at the top of the body", conformance="not-yet-established", register=False)


def _fused_moe_pairs(B, S_, X, ids, W):
    pairs = []
    for t in range(S_.M):
        for s in range(S_.S):
            pairs.append(B.call(bind(MoeExpertGemm, E=S_.E, K=S_.K, N=S_.N), X[t], ids[t][s], W))
    return array_of(pairs)


FusedMoEPairs = CompositeDefinition(
    "FusedMoEPairs", 1, ("M", "S", "E", "K", "N"),
    lambda S_: ((("X", Array(S_.M, Array(S_.K, BF16))), ("ids", Array(S_.M, Array(S_.S, I32))), ("W", Array(S_.E, Array(S_.N, Array(S_.K, BF16))))),
                Array(S_.M * S_.S, Array(S_.N, BF16))),
    _fused_moe_pairs, doc="shape C: one composite over the whole launch, pair-major inside", conformance="not-yet-established", register=False)

A_LEAVES = 1 + M * K + N * K                 # 193
B_LEAVES = K + 1 + E * N * K                 # 529
COORD_LEAVES = K + 1 + E * K                 # 81: x, e, the E candidate rows of one weight element row


@pytest.mark.parametrize("budget", [100, 40, 8])
def test_shape_a_row_gather_at_the_top_is_zero_mass_once_oversize(budget):
    s = vq.index(one_call_result(bind(RowGatherThenGemm, M=M, K=K, N=N)), max_arg_leaves=budget, depth=3).summary()
    assert budget < A_LEAVES and s["oversize_instances"] == 1
    assert s["count"] == 0 and s["instances_without_units"] == 1
    assert s["specs"][f"RowGatherThenGemm_v1{{M={M},K={K},N={N}}}"] == {"instances": 1, "vus": 0, "granularity": ["none"]}


def test_shape_a_fits_only_as_a_whole_instance():
    s = vq.index(one_call_result(bind(RowGatherThenGemm, M=M, K=K, N=N)), max_arg_leaves=A_LEAVES, depth=3).summary()
    assert s["count"] == 1 and s["oversize_instances"] == 0


@pytest.mark.parametrize("budget", [200, 100, COORD_LEAVES])
def test_shape_b_pair_gemm_decomposes_into_coordinate_units(budget):
    s = vq.index(one_call_result(bind(MoeExpertGemm, E=E, K=K, N=N)), max_arg_leaves=budget, depth=3).summary()
    assert budget < B_LEAVES and s["oversize_instances"] == 1 and s["instances_without_units"] == 0
    assert s["count"] == N
    assert s["specs"][f"MoeExpertGemm_v1{{E={E},K={K},N={N}}}"] == {"instances": 1, "vus": N, "granularity": [f"child:MoeExpertCoordinate_v1{{E={E},K={K}}}"]}


def test_shape_b_below_the_coordinate_size_has_no_units_like_any_gemm():
    s = vq.index(one_call_result(bind(MoeExpertGemm, E=E, K=K, N=N)), max_arg_leaves=COORD_LEAVES - 1, depth=3).summary()
    assert s["count"] == 0 and s["instances_without_units"] == 1


def test_shape_c_whole_launch_composite_decomposes_to_the_same_units_without_reading_ids():
    r = one_call_result(bind(FusedMoEPairs, M=M, S=S, E=E, K=K, N=N))
    ix = vq.index(r, max_arg_leaves=100, depth=3)
    s = ix.summary()
    assert s["oversize_instances"] == 1 and s["instances_without_units"] == 0
    assert s["count"] == M * S * N
    assert s["specs"][f"FusedMoEPairs_v1{{M={M},S={S},E={E},K={K},N={N}}}"]["granularity"] == [f"child:MoeExpertCoordinate_v1{{E={E},K={K}}}"]
    covered: set[int] = set()
    for vu in ix.of_instance(0):
        covered.update(vq.returns_of(r, vu))
    assert len(covered) == M * S * N            # the units tile every return leaf of the launch
    # depth 1 is not enough: the pair GEMM (529 leaves) is oversize and its coordinates are one level further down
    assert vq.index(r, max_arg_leaves=100, depth=1).summary()["count"] == M * S * N
    assert vq.index(r, max_arg_leaves=100, depth=0).summary()["count"] == 0
