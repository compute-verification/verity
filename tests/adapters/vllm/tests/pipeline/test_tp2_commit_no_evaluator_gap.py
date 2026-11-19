"""A TP Commit must not pass with families no gate evaluates.

The witness-conditioned match oracle rests the faithfulness of every exempt tensor on the in-run checks: the sampled replay against the
committed schedule, linkage, collectives and tokens.  The sampled replay evaluates only families with a registered evaluator, and the
dense verdict fails a family without one unless a complete form (B) covers it (`c2_rules._complete_replay_population_gap`).  Without
that rule, a TP Commit whose oracle compares 2 of 5920 tensors per rank passes with the router, the routed experts' SiluMulClamp_v1,
FlashMLA, the norms and the mHC `unreplayable`, so a Commit that routes a token to other experts, and runs them, passes every replayed
row (the align and the Marlin GEMMs read the committed ids) and every other gate.  The TP verdict applies the dense rule per rank
(`no_evaluator_gate`), and each waiver it allows is declared and tested here: a collective family the cross-rank pass evaluates, while
that pass runs, and a selector whose every committed sampled id the Match record confirms.

No engine, no torch: the gates are pure functions of the record."""
from __future__ import annotations

from verity_vllm.check.c2_rules import _complete_replay_population_gap
from verity_vllm.program.frontend.collective_kinds import cross_rank_families

XF8 = cross_rank_families(8)
#: a DeepSeek-V4-Pro TP8 Commit's rank-0 unreplayable families with no evaluator and no `evaluated_by` (rows over 8 ranks), a selection
NO_EVALUATOR = {"MoeRouterSqrtSoftplus_v1": 133168, "MoeRouterHash_v1": 6888, "SiluMulClamp_v1": 980392, "FlashMlaPrefill_v1": 124928,
                "FlashMlaDecode_v1": 15128, "TritonBlockRMSNorm_v1": 280112, "MhcPostPreTf32_v1": 247808}
#: its replay population (rows over 8 ranks)
POPULATION = {"DeepGemmFp8Linear_v1": 624640, "FlashInferFp8Linear_v1": 75640, "Gemm_v2": 256, "MarlinMoeExpertGemmW_v1": 840336,
              "MarlinMoeExpertGemm_v1": 840336, "MoeAlignBlocks_v1": 15616, "MoeAlignRanks_v1": 15616, "MoeSum_v1": 140056,
              "ScaledMmFp8BlockGrouped_v1": 140056, "TokenSelect_v1": 256}
#: its match oracle on rank 0 (summary.json match_oracle["0"], as that Commit's oracle wrote it)
ORACLE = {"verdict": "PASS", "steps_compared": 32, "tensors": 5920, "compared": 2, "equal": 2, "exempt": 5918, "mismatch_n": 0, "missing_n": 0,
          "not_in_oracle_n": 0, "witness_counts": {"equal": 978, "differs": 974, "unpaired": 0},
          "exempt_by": [{"rank": 0, "step": 0, "k": 2, "module": "model.layers.0.ffn.experts", "members": ["align.rank", "align.start", "align.blk"]}]}
#: the world-8 collectives the cross-rank pass evaluates, as the replay names them
COLLECTIVES = {"AllReduce_v2": 61440, "AllGather_v1": 256, "EmbeddingShard_v1": 256}


def _sr(unreplayable: dict[str, int], why: str = "no registered evaluator") -> dict:
    checks = {**{k: {"n": 0, "first": []} for k in ("vus_outside_query", "identities_without_rows", "rows_disagree")},
              "identities_unreachable_by_alias": {"n": 0, "first": [], "alias_entries": 15},
              "ranks": {"replay": [0], "manifest": [0], "equal": True}, "accounted": True, "unstated_families": []}
    pop = {"vus": sum(POPULATION.values()) // 8, "by_family": dict(POPULATION), "scope": "Q_module_v1(P)",
           "query_total": {"identities": 336601, "query_id": "Q_module_v1", "cross_check": {"agree": True}}, "query_checks": checks | {"ok": True}}
    return {"result": True, "grade": "complete", "population": pop, "sample": {"picked": 11600, "evaluated": 11600, "equal": 11600, "mismatch_n": 0},
            "unreplayable": [{"family": f, "n": n, "why": why, "ranks": list(range(8))} for f, n in unreplayable.items()]}




def test_the_dense_verdict_fails_a_no_evaluator_family_without_a_complete_form_b():
    """The rule the TP verdict lacked: the router with no evaluator and an incomplete oracle compare is a named FAIL on the dense path."""
    sr = {"result": True, "population": {"not_evaluable": {f"no registered evaluator for {f}": n for f, n in NO_EVALUATOR.items()}}}
    oc = {"result": True, "complete": False, "mismatch_n": 0, "compared": 2}
    note, fail = _complete_replay_population_gap(sr, oc)
    assert note is None and fail is not None
    assert fail.startswith("C2 of record not established: ") and "MoeRouterSqrtSoftplus_v1" in fail












