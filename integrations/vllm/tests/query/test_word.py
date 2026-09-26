"""`Q_word_v1{X,W}` (`verity_vllm.query.word`): the partition of a Call's gates, its widths as the IR's own |out(S)| (checked against
`verity.ir.layout.GateSelection.out` on the explicit circuit), the committed interior values it derives for a norm, a Gemm and FA2
attention, and the check failing by name."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ir.layout import GateSelection
from verity_vllm.query import word as W


def _fn(fam: str, **statics):
    return W.specialization(fam, statics)


@pytest.fixture(scope="module")
def norm():
    return _fn("RMSNormTriton_v1", N=2048, EPS=1e-05)


@pytest.fixture(scope="module")
def attn():
    return _fn("Attention_v3", T=21, NH=4, KVH=2, D=64, BN=128)


def test_query_id_and_parse():
    assert W.query_id() == "Q_word_v1{X=16,W=32}"
    assert W.parse("16") == (16, 32, 0) and W.parse("16/32+2") == (16, 32, 2) and W.parse("8/8") == (8, 8, 0)


@pytest.mark.parametrize("fam,statics", [("RMSNormTriton_v1", {"N": 2048, "EPS": 1e-05}), ("Gemm_v1", {"K": 64, "N": 8}),
                                         ("Attention_v3", {"T": 21, "NH": 4, "KVH": 2, "D": 64, "BN": 128})])
def test_every_gate_in_one_unit_and_widths_are_the_irs_out(fam, statics):
    fn = _fn(fam, **statics)
    G = W.Graph(fn)
    R = W.units(G, 16, 32)
    owner = R["owner"]
    assert len(owner) == fn.gates and owner.min() >= 0 and len(R["kinds"]) == int(owner.max()) + 1
    c, call = W._call_scope(fn)
    rng = np.random.default_rng(0)
    for u in rng.choice(len(R["kinds"]), size=min(12, len(R["kinds"])), replace=False):
        gates = (np.flatnonzero(owner == u) + call.offset).tolist()
        out = GateSelection(c, gates).out
        assert sum(c.gate(g).prim.ret.w for g in out) == R["out_bits"][u]
        assert len(out) == R["out_gates"][u]


def test_a_norm_commits_its_scale_and_recomputes_the_upcasts(norm):
    r = W.unit_rule(norm)
    assert r["committed_interior_words"] == 1 and not r["violations"]
    assert r["by_kind"] == {"output": 2048, "recompute": 2048, "committed": 1}
    out = next(c for c in r["classes"] if c["kind"] == "output")
    assert out["out_bits"] == 16 and out["out_gates"] == 1


def test_a_gemm_is_one_unit_per_output():
    r = W.unit_rule(_fn("Gemm_v1", K=64, N=8))
    assert r["by_kind"] == {"output": 8} and r["committed_interior_words"] == 0 and r["classes"][0]["out_bits"] == 16


def test_attention_commits_scores_p_row_values_and_inverse_sum(attn):
    r = W.unit_rule(attn)
    assert not r["violations"]
    heads = {(i["scope"].split("{")[0], i["head"]): i["words"] for i in r["committed_interior"]}
    assert heads[("DotBf16_v1", "AmpereBF16TcDot16_v1")] == 4 * 21                  # S: one score per (head, key)
    assert heads[("AttnBlock_v3", "F2fpBf16_v1")] == 4 * 21                          # P (bf16): what the PV products read
    assert heads[("AttentionHead_v3", "Fa2InvSum_v1")] == 4                          # FIN inv_sum, one per head
    assert r["by_kind"]["recompute"] >= 4 * 21                                       # exp2(s * scale - max): 2 gates of committed words


def test_a_width_over_x_and_w_is_named():
    with pytest.raises(W.QueryRuleViolation, match=r"Gemm_v1\{K=64,N=8\}: output:F2fpBf16_v1"):
        W.check_calls([{"family": "Gemm_v1", "statics": {"K": 64, "N": 8}, "definition": "Gemm_v1{K=64,N=8}", "calls": 3, "where": "m.proj"}], 8, 8)


def test_uncommitted_boundaries_are_named():
    g = {"family": "Gemm_v1", "statics": {"K": 64, "N": 8}, "definition": "Gemm_v1{K=64,N=8}", "calls": 2, "where": "m.proj",
         "outputs_committed": False}
    r = W.check_calls([g], strict=False)
    assert [v["class"] for v in r["violations"]] == ["output-not-committed"] and r["violations"][0]["where"] == ["m.proj"]
    with pytest.raises(W.QueryRuleViolation, match="output-not-committed, 2 Call"):
        W.check_calls([g])
    ok = W.check_calls([dict(g, outputs_committed=True)])
    assert ok["ok"] and ok["units"] == 16 and ok["gates"] == 2 * _fn("Gemm_v1", K=64, N=8).gates


def test_the_manifest_build_runs_the_check(tmp_path):
    from tests.check.test_sampled_replay import _program
    from verity_vllm.pipeline import manifest as M
    from verity_vllm.query.program_view import from_instances
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.required import population
    inst, _w = _program(attn=True)
    P = from_instances(inst)
    corr = Correspondence(P)
    pop = population(P, corr)
    r = M.word_check(P, corr, pop.required, "16/32", strict=False)
    assert r["query"] == "Q_word_v1{X=16,W=32}" and r["calls"] == len(inst["rows"]) - len(pop.literals)
    assert r["units"] > r["calls"]
    # the test Program's o_proj output feeds no other module: interior to its body, so no unit can end at it
    assert {(v["definition"].split("{")[0], v["class"]) for v in r["violations"]} == {("Gemm_v1", "output-not-committed")}
    with pytest.raises(W.QueryRuleViolation, match="o_proj"):
        M.word_check(P, corr, pop.required, "16/32")


def test_a_dead_gate_joins_the_unit_it_reads():
    r = W.unit_rule(_fn("TokenSelect_v1", V=8))
    assert r["by_kind"] == {"output": 1} and r["classes"][0]["out_bits"] == 32 and not r["violations"]
