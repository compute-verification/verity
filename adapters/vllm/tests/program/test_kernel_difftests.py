"""The fused-MoE difftest adapter without a GPU (`program.moe_difftest`): its cases carry every edge kind of a difftest
and bind a spec whose parameters take exactly the case's input words, its spec reads each (token, slot) pair's own routed
weight, and its cases route distinct experts and take both block schedules."""
from __future__ import annotations

import random

import numpy as np
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.definitions.registry import b1
from verity.definitions.registry.moe import MoeExpertGemm, MoeExpertGemmW, MoeSum

from verity_vllm.check.replay.stoch_recompute import evaluate_spec
from verity_vllm.program import moe_difftest as MD

#: the edge kinds a difftest's cases must carry, beside random words
EDGE_KINDS = ("zeros", "max", "subnormal", "alternating")


def _bf16_words(n: int, g: np.random.Generator) -> np.ndarray:
    f = (g.standard_normal(n) * float(g.choice([0.05, 1.0, 8.0]))).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def test_cases_carry_every_edge_kind_and_bind_their_inputs():
    rng = random.Random(3)
    kinds = set()
    for i in range(20):
        c = MD.cases(rng, i)
        kinds |= {s["kind"] for s in c["inputs"]}
        params, _ = MD.spec(c["statics"]).signature()
        assert [pt.leaves for _, pt in params] == [int(np.prod(s["shape"])) for s in c["inputs"]]
    assert set(EDGE_KINDS) <= kinds


def test_moe_spec_reads_each_pairs_expert_and_weight():
    st = {"E": 4, "TOPK": 2, "H": 32, "I": 16, "M": 2}
    E, TOPK, H, inter, M = (st[k] for k in ("E", "TOPK", "H", "I", "M"))
    g = np.random.default_rng(5)
    x, w13, w2 = _bf16_words(M * H, g), _bf16_words(E * 2 * inter * H, g), _bf16_words(E * H * inter, g)
    ids = np.array([3, 0, 1, 2], np.uint32)
    wts = np.array([0x3F400000, 0x3E800000, 0x3F000000, 0x3F000000], np.uint32)   # 0.75, 0.25, 0.5, 0.5
    got = evaluate_spec(MD.spec(st), [x, w13, w2, ids, wts])
    up, act, down, out = [], [], [], []
    for t in range(M):
        dt = []
        for s in range(TOPK):
            p = t * TOPK + s
            u = list(evaluate(bind(MoeExpertGemm, E=E, K=H, N=2 * inter), x[t * H:(t + 1) * H].tolist(), [int(ids[p])], w13.tolist()))
            a = list(evaluate(bind(b1.SiluMul, I=inter), u))
            d = list(evaluate(bind(MoeExpertGemmW, E=E, K=inter, N=H), a, [int(ids[p])], [int(wts[p])], w2.tolist()))
            up += u
            act += a
            dt += d
        down += dt
        out += list(evaluate(bind(MoeSum, TOPK=TOPK, H=H), dt))
    assert got == up + act + down + out
    swapped = evaluate_spec(MD.spec(st), [x, w13, w2, ids[[1, 0, 2, 3]], wts])
    assert swapped[: 2 * inter] != got[: 2 * inter]


def test_moe_cases_route_distinct_experts_and_take_both_block_schedules():
    naive = set()
    for i in range(20):
        c = MD.cases(random.Random(i), i)
        E, TOPK, M = (c["statics"][k] for k in ("E", "TOPK", "M"))
        ids = c["inputs"][3]["words"]
        assert all(len(set(ids[t * TOPK:(t + 1) * TOPK])) == TOPK for t in range(M)) and max(ids) < E
        naive.add(M * TOPK * 4 <= E)
    assert naive == {True, False}
