"""The kernel difftest adapters without a GPU (`difftests.rope_difftest`, `difftests.rope_gptj_difftest`, `registry.moe_difftest`,
`difftests.collectives_difftest`): their cases carry every edge kind admission requires, each case binds a spec whose parameters take exactly
the case's input words, the rotary specs are the NeoX and the interleaved rotation of the query and the key by one cache row, the fused-MoE spec reads each (token, slot) pair's own expert id and
routed weight, and the two-rank collectives are a bf16 add and a concatenation in rank order."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import moe_difftest as MD
from verity_vllm.properties.difftests import rope_difftest as RD
from verity_vllm.properties.difftests import rope_gptj_difftest as RG
from verity_vllm.properties.difftests import collectives_difftest as TD
from verity_vllm.program.registry.moe import MoeExpertGemm, MoeExpertGemmW, MoeSum
from verity_vllm.properties import admission as A


def _words(c: dict, seed: int) -> list[np.ndarray]:
    g = np.random.default_rng(seed)
    return [np.asarray(s["words"], dtype=A._np_dtype(s["dtype"])) if "words" in s else A.words_for(s["kind"], s["shape"], s["dtype"], g)
            for s in c["inputs"]]


@pytest.mark.parametrize("mod,n", [(RD, 48), (RG, 60), (MD, 20), (TD, 30)])
def test_cases_carry_every_edge_kind_and_bind_their_inputs(mod, n):
    rng = random.Random(3)
    kinds = set()
    for i in range(n):
        c = mod.cases(rng, i)
        kinds |= {s["kind"] for s in c["inputs"]}
        params, _ = mod.spec(c["statics"]).signature()
        assert [pt.leaves for _, pt in params] == [int(np.prod(s["shape"])) for s in c["inputs"]]
    assert set(A.REQUIRED_EDGE_KINDS) <= kinds


def _bf16_rne(f: np.ndarray) -> np.ndarray:
    u = f.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def test_rope_spec_is_the_neox_rotation_of_query_and_key():
    for i in range(len(RD.SHAPES)):
        c = RD.cases(random.Random(i), i)
        c["inputs"] = [dict(s, kind="random") for s in c["inputs"]]
        q, k, cs = _words(c, i)
        D = c["statics"]["D"]
        got = np.asarray(A.evaluate_spec(RD.spec(c["statics"]), [q, k, cs]), dtype=np.uint16)
        f = lambda w: (w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        cos, sin = f(cs[: D // 2]), f(cs[D // 2:])
        want = []
        for x in (q, k):
            h = f(x).reshape(-1, D)
            lo, hi = h[:, : D // 2], h[:, D // 2:]
            want.append(np.concatenate([_bf16_rne(lo * cos - hi * sin), _bf16_rne(hi * cos + lo * sin)], axis=1).reshape(-1))
        assert np.array_equal(got, np.concatenate(want)), c["statics"]


def test_rope_gptj_spec_is_the_interleaved_rotation_of_query_and_key():
    for i in range(len(RG.SHAPES)):
        c = RG.cases(random.Random(i), i)
        c["inputs"] = [dict(s, kind="random") for s in c["inputs"]]
        q, k, cs = _words(c, i)
        D = c["statics"]["D"]
        got = np.asarray(A.evaluate_spec(RG.spec(c["statics"]), [q, k, cs]), dtype=np.uint16)
        f = lambda w: (w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        cos, sin = f(cs[: D // 2]), f(cs[D // 2:])
        want = []
        for x in (q, k):
            h = f(x).reshape(-1, D)
            ev, od = h[:, 0::2], h[:, 1::2]
            out = np.empty(h.shape, np.uint16)
            out[:, 0::2], out[:, 1::2] = _bf16_rne(ev * cos - od * sin), _bf16_rne(od * cos + ev * sin)
            want.append(out.reshape(-1))
        assert np.array_equal(got, np.concatenate(want)), c["statics"]


def test_moe_spec_reads_each_pairs_expert_and_weight():
    st = {"E": 4, "TOPK": 2, "H": 32, "I": 16, "M": 2}
    E, TOPK, H, I, M = (st[k] for k in ("E", "TOPK", "H", "I", "M"))
    g = np.random.default_rng(5)
    x, w13, w2 = A.words_for("random", [M * H], "bfloat16", g), A.words_for("random", [E * 2 * I * H], "bfloat16", g), \
        A.words_for("random", [E * H * I], "bfloat16", g)
    ids = np.array([3, 0, 1, 2], np.uint32)
    wts = np.array([0x3F400000, 0x3E800000, 0x3F000000, 0x3F000000], np.uint32)   # 0.75, 0.25, 0.5, 0.5
    got = A.evaluate_spec(MD.spec(st), [x, w13, w2, ids, wts])
    up, act, down, out = [], [], [], []
    for t in range(M):
        dt = []
        for s in range(TOPK):
            p = t * TOPK + s
            u = list(evaluate(bind(MoeExpertGemm, E=E, K=H, N=2 * I), x[t * H:(t + 1) * H].tolist(), [int(ids[p])], w13.tolist()))
            a = list(evaluate(bind(b1.SiluMul, I=I), u))
            d = list(evaluate(bind(MoeExpertGemmW, E=E, K=I, N=H), a, [int(ids[p])], [int(wts[p])], w2.tolist()))
            up += u; act += a; dt += d
        down += dt
        out += list(evaluate(bind(MoeSum, TOPK=TOPK, H=H), dt))
    assert got == up + act + down + out
    swapped = A.evaluate_spec(MD.spec(st), [x, w13, w2, ids[[1, 0, 2, 3]], wts])
    assert swapped[: 2 * I] != got[: 2 * I]


def test_moe_cases_route_distinct_experts_and_take_both_block_schedules():
    naive = set()
    for i in range(20):
        c = MD.cases(random.Random(i), i)
        E, TOPK, M = (c["statics"][k] for k in ("E", "TOPK", "M"))
        ids = c["inputs"][3]["words"]
        assert all(len(set(ids[t * TOPK:(t + 1) * TOPK])) == TOPK for t in range(M)) and max(ids) < E
        naive.add(M * TOPK * 4 <= E)
    assert naive == {True, False}


def test_two_rank_collectives_are_a_bf16_add_and_a_concatenation_in_rank_order():
    f = lambda w: (w.astype(np.uint32) << 16).view(np.float32)
    for i in range(10):
        c = TD.cases(random.Random(i), i)
        c["inputs"] = [dict(s, kind="random") for s in c["inputs"]]
        a, b = _words(c, i)
        got = np.asarray(A.evaluate_spec(TD.spec(c["statics"]), [a, b]), dtype=np.uint16)
        if c["statics"]["op"] == "all_gather":
            assert np.array_equal(got, np.concatenate([a, b]))
        else:
            assert np.array_equal(got, _bf16_rne(f(a) + f(b)))
