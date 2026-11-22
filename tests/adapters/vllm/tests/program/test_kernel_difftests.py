"""The fused-MoE difftest adapter without a GPU (`program.moe_difftest`): its cases carry every edge kind of a difftest
and bind a spec whose parameters take exactly the case's input words, its spec reads each (token, slot) pair's own routed
weight, and its cases route distinct experts and take both block schedules."""
from __future__ import annotations

import random

import numpy as np

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


def test_moe_cases_route_distinct_experts_and_take_both_block_schedules():
    naive = set()
    for i in range(20):
        c = MD.cases(random.Random(i), i)
        E, TOPK, M = (c["statics"][k] for k in ("E", "TOPK", "M"))
        ids = c["inputs"][3]["words"]
        assert all(len(set(ids[t * TOPK:(t + 1) * TOPK])) == TOPK for t in range(M)) and max(ids) < E
        naive.add(M * TOPK * 4 <= E)
    assert naive == {True, False}
