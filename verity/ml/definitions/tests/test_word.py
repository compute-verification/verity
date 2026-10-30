"""`verity.ml.definitions.word`'s cut, the copy circuit-check's partition verdict reads, held to core's reference
(`verity.ml.circuits.cut`) on a spread of small Definitions: `Graph` is `CallGraph` field by field, `partition` gives
`cut_word`'s owner per gate, kind per unit and head per gate, and `cut_verdict` with `units`' width rule gives `check_cut`'s
verdict, detail for detail, on the cut and on cuts broken every way the verdict names."""
from __future__ import annotations

import json
from functools import cache

import numpy as np
import pytest

from verity.ml.circuits import cut
from verity.ml.circuits.defs import bind
from verity.ml.definitions import word as W
from verity.ml.definitions.call_scope import specialization


def _spec(fam: str, **statics):
    return lambda: specialization(fam, statics)


def _bound(module: str, name: str, **statics):
    def build():
        import importlib
        return bind(getattr(importlib.import_module(f"verity.ml.definitions.registry.{module}"), name), **statics)
    return build


def _moe(routed: bool, E: int, K: int, N: int, swap: bool):
    """GLM-5.3's MoE expert FP8 GEMM (w13; w2 `routed`) on Hopper's tiles (G = KB = 128), scaled down in E, K and N."""
    from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32
    from verity.ml.definitions.registry import fp8_moe_experts as XM
    return bind(XM.MoeExpertFp8GemmW if routed else XM.MoeExpertFp8Gemm, E=E, K=K, N=N, G=128, KB=128, SWAP=swap,
                DOT=HopperE4m3QgmmaDot32 if swap else HopperE4m3MmaV2Dot32)


def _twice():
    """The same F32Add computed twice, each copy read by its own output: a value computed in two units."""
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Array, Tuple
    from verity.ml.definitions.registry import prims as P

    def body(B, S, x):
        a = B.call(P.F32Add, x[0], x[1])
        b = B.call(P.F32Add, x[0], x[1])
        return tuple_of(B.call(P.F32Mul, a, x[2]), B.call(P.F32Mul, b, x[3]))
    return bind(CompositeDefinition("Twice", 1, (), lambda S: ((("x", Array(4, P.F32)),), Tuple(P.F32, P.F32)), body, register=False))


def _dead_input():
    """A parameter read only by wiring nothing reads (committed, unread), beside one returned through wiring."""
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Array, Tuple
    from verity.ml.definitions.registry import prims as P

    def body(B, S, x, y):
        B.call(P.Bf16ToF32, y[0])
        return tuple_of(B.call(P.F32Add, B.call(P.Bf16ToF32, x[0]), B.call(P.Bf16ToF32, x[1])), B.call(P.Bf16ToF32, y[1]))
    return bind(CompositeDefinition("DeadInput", 1, (), lambda S: ((("x", Array(2, P.BF16)), ("y", Array(2, P.BF16))), Tuple(P.F32, P.F32)),
                                    body, register=False))


#: popread's small set (expert gathers, FP8 block scales, a router's top-k, a fused norm, a separable body), registry Definitions
#: over every branch of the graph and its check (wiring and constants, a wide gate, a value computed twice in one unit, attention,
#: the ordered router, the MoE expert FP8 GEMMs over two K tiles and a partial weight block, two Boolean Definitions), and two that
#: fail the invariant (a value computed in two units; a parameter committed and read by no unit)
CORPUS = {
    "moe-fp8-gemm": _spec("MoeExpertFp8Gemm_v1", E=4, K=128, N=16, G=128, KB=128, SWAP=False, DOT="HopperE4m3MmaV2Dot32_v1"),
    "fp8-linear": _spec("FlashInferFp8Linear_v1", K=128, N=16),
    "router-sigmoid": _spec("MoeRouterSigmoidBias_v1", E=8, TOPK=2, SCALE=2.5, KERNEL="block"),
    "fused-norm": _spec("RMSNormFusedCuda_v2", N=16, EPS=1e-5),
    "moe-gemm": _spec("MoeExpertGemm_v2", E=4, K=16, N=4, DOT="HopperBF16WgmmaDot16_v1"),
    "norm": _spec("RMSNormTriton_v1", N=64, EPS=1e-05),
    "gemm": _spec("Gemm_v1", K=64, N=8),
    "attention": _spec("Attention_v3", T=3, NH=2, KVH=1, D=16, BN=128),
    "select": _spec("TokenSelect_v1", V=8),
    "sampler": _bound("sampling", "GumbelTopPTokenSelect", V=16),
    "wide": _spec("EmbeddingShard_v1", VS=cut.WIDE_FAN_IN + 3, H=4, START=0),
    "router-ordered": _bound("moe", "MoeRouterTopKOrdered", E=64, TOPK=8, VPT=8),
    "moe-w13": lambda: _moe(False, E=4, K=256, N=130, swap=True),
    "moe-w2": lambda: _moe(True, E=4, K=256, N=48, swap=False),
    "boolean-tanh": _bound("boolean_activation", "Bf16Tanh", N=8),
    "boolean-embedding": _bound("boolean_gather", "Embedding", V=64, H=16),
    "twice": _twice,
    "dead-input": _dead_input,
}


@cache
def _graphs(name: str) -> tuple[cut.CallGraph, W.Graph]:
    fn = CORPUS[name]()
    return cut.CallGraph(fn), W.Graph(fn)


def _same(got, want) -> bool:
    return json.dumps(got.to_json()) == json.dumps(want.to_json())


@pytest.mark.parametrize("name", CORPUS)
def test_the_graph_is_the_references_field_by_field(name):
    ref, g = _graphs(name)
    for a in ("n", "total", "free", "outputs", "returned_inputs", "recomputed", "input_reads", "free_by_prim", "prims", "scopes",
              "prim_names", "scope_names"):
        assert getattr(g, a) == getattr(ref, a), a
    for a in ("keep", "width", "prim", "scope", "src", "dst", "input_ids"):
        assert getattr(g, a).tolist() == list(getattr(ref, a)), a
    assert g.external.tolist() == [bool(x) for x in ref.external]
    assert g.input_read.tolist() == [i for i, _q in ref.input_reads] and g.input_reader.tolist() == [q for _i, q in ref.input_reads]


@pytest.mark.parametrize("X", [8, 16])
@pytest.mark.parametrize("name", CORPUS)
def test_the_partition_and_its_verdict_are_cut_words_and_check_cuts(name, X):
    """Owner per gate, kind per unit and head per gate are `cut_word`'s on the reference graph; the heads are the committed set
    core derives; and the verdict, `cut_verdict`'s codes and detail with `units`' width rule, is `check_cut`'s."""
    ref, g = _graphs(name)
    owner, kinds, head = W.partition(g, X)
    r_owner, r_kinds, r_head = cut.cut_word(ref, X)
    assert owner.tolist() == r_owner and kinds == r_kinds and head.tolist() == r_head
    heads = np.flatnonzero(head >= 0).tolist()
    assert cut.derived_committed(ref, r_owner) == heads
    got = W.cut_verdict(g, owner, head)
    assert _same(got, cut.check_cut(ref, r_owner, heads, limits=None))
    R = W.units(g, X, 32)
    assert (R["out_bits"].tolist(), R["out_gates"].tolist()) == cut.boundary_widths(ref, r_owner, len(r_kinds))
    wide = [[u, b, c] for u, (b, c, ok) in enumerate(zip(R["out_bits"].tolist(), R["out_gates"].tolist(), R["ok"].tolist())) if not ok]
    full = cut.check_cut(ref, r_owner, limits=(X, 32, 0))
    assert full.codes == got.codes + (("unit-too-wide",) if wide else ())
    assert full.detail.get("unit-too-wide", {}).get("gates", []) == wide[:64]
    assert (name in ("twice", "dead-input")) == bool(got.codes)


@pytest.mark.parametrize("name", CORPUS)
def test_the_verdict_is_check_cuts_on_broken_cuts(name):
    """`cut_verdict` gives `check_cut`'s verdict, its codes, counts, listed gates and detail order, on cuts broken every way it
    names: gates in no unit, a committed value dropped (an uncommitted read or output), a value committed that no other unit reads,
    units merged and split (a recompute across units), and random owners."""
    ref, g = _graphs(name)
    owner, kinds, head = W.partition(g, 16)
    rng = np.random.default_rng(0)
    cuts = []
    heads = np.flatnonzero(head >= 0)
    for _ in range(2):
        o, h = owner.copy(), head.copy()
        if g.n:
            o[rng.integers(0, g.n, max(1, g.n // 40))] = -1
            h[rng.integers(0, g.n, max(1, g.n // 40))] = len(kinds)
        if len(heads):
            h[rng.choice(heads, max(1, len(heads) // 3), replace=False)] = -1
        cuts += [(o, head), (owner, h), (o, h), (rng.integers(0, max(1, len(kinds)), g.n), head), (np.zeros(g.n, np.int64), head),
                 (np.arange(g.n, dtype=np.int64), np.arange(g.n, dtype=np.int64))]
    codes = set()
    for o, h in cuts:
        got = W.cut_verdict(g, o, h)
        assert _same(got, cut.check_cut(ref, o.tolist(), np.flatnonzero(h >= 0).tolist(), limits=None))
        codes |= set(got.codes)
    if g.n > 8:
        assert {"gate-not-certified-once", "gate-count", "read-uncommitted", "committed-unread"} <= codes, codes
