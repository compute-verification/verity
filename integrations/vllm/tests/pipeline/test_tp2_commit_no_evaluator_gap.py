"""Red-team repro (circuits W6, claim 2): a TP Commit passes with families no gate evaluates.

The witness-conditioned match oracle rests the faithfulness of every exempt tensor on the in-run checks: the sampled replay against the
committed schedule, linkage, collectives and tokens.  The sampled replay evaluates only families with a registered evaluator, and the
dense verdict fails a family without one unless a complete form (B) covers it (`c2_rules._complete_replay_population_gap`, COORD
M-0721 (1)).  The TP verdict (`pipeline.tp.c2.replay_gates`) has no such rule.  DeepSeek-V4-Pro's TP8 Commit r20261018-023527-3dc1
passed with its oracle comparing 2 of 5920 tensors per rank (5918 exempt) while its replay population held 10 families: the router
(MoeRouterSqrtSoftplus_v1 x133168, MoeRouterHash_v1 x6888), the routed experts' SiluMulClamp_v1 (x980392), FlashMLA, the norms and the
mHC were `unreplayable` with "no registered evaluator" and no other pass evaluating them.  A Commit that routed a token to other
experts, and ran them, passes every replayed row (the align and the Marlin GEMMs read the committed ids) and every other gate.

No engine, no torch: the gates are pure functions of the record."""
from __future__ import annotations

import pytest

from verity_vllm.check.c2_rules import _complete_replay_population_gap
from verity_vllm.pipeline.tp.c2 import c2_and_pin_gates

#: r20261018-023527-3dc1 rank 0's unreplayable families with no evaluator and no `evaluated_by` (rows over 8 ranks), a selection
NO_EVALUATOR = {"MoeRouterSqrtSoftplus_v1": 133168, "MoeRouterHash_v1": 6888, "SiluMulClamp_v1": 980392, "FlashMlaPrefill_v1": 124928,
                "FlashMlaDecode_v1": 15128, "TritonBlockRMSNorm_v1": 280112, "MhcPostPreTf32_v1": 247808}
#: its replay population (rows over 8 ranks)
POPULATION = {"DeepGemmFp8Linear_v1": 624640, "FlashInferFp8Linear_v1": 75640, "Gemm_v2": 256, "MarlinMoeExpertGemmW_v1": 840336,
              "MarlinMoeExpertGemm_v1": 840336, "MoeAlignBlocks_v1": 15616, "MoeAlignRanks_v1": 15616, "MoeSum_v1": 140056,
              "ScaledMmFp8BlockGrouped_v1": 140056, "TokenSelect_v1": 256}
#: its match oracle on rank 0 (summary.json match_oracle["0"])
ORACLE = {"verdict": "PASS", "steps_compared": 32, "tensors": 5920, "compared": 2, "equal": 2, "exempt": 5918, "mismatch_n": 0, "missing_n": 0,
          "not_in_oracle_n": 0, "witness_counts": {"equal": 978, "differs": 974, "unpaired": 0},
          "exempt_by": [{"rank": 0, "step": 0, "k": 2, "module": "model.layers.0.ffn.experts", "members": ["align.rank", "align.start", "align.blk"]}]}


def _sr() -> dict:
    checks = {k: {"n": 0, "first": []} for k in ("vus_outside_query", "identities_without_rows", "identities_unreachable_by_alias", "rows_disagree")}
    pop = {"vus": sum(POPULATION.values()) // 8, "by_family": dict(POPULATION), "scope": "Q_module_v1(P)",
           "query_total": {"identities": 336601, "query_id": "Q_module_v1", "cross_check": {"agree": True}}, "query_checks": checks | {"ok": True}}
    return {"result": True, "grade": "complete", "population": pop, "sample": {"picked": 11600, "evaluated": 11600, "equal": 11600, "mismatch_n": 0},
            "unreplayable": [{"family": f, "n": n, "why": "no registered evaluator", "ranks": list(range(8))} for f, n in NO_EVALUATOR.items()]}


def _gates(oracle: dict) -> tuple[bool | None, str | None]:
    cblock = {"match_oracle": dict(oracle)}
    f = {"sampled_replay": {"sampled_replay": _sr(), "boundary_linkage": {"result": True, "linked": 32, "pairs": 32}},
         "weights_pin": {"result": True, "compared": 2139}}
    return c2_and_pin_gates(cblock, f, True, None, rank=0, replay_on=True, required=True, bundle="bundle_pair0_rank0")


def test_the_dense_verdict_fails_a_no_evaluator_family_without_a_complete_form_b():
    """The rule the TP verdict lacks: the router with no evaluator and an incomplete oracle compare is a named FAIL on the dense path."""
    sr = {"result": True, "population": {"not_evaluable": {f"no registered evaluator for {f}": n for f, n in NO_EVALUATOR.items()}}}
    oc = {"result": True, "complete": False, "mismatch_n": 0, "compared": 2}
    note, fail = _complete_replay_population_gap(sr, oc)
    assert note is None and fail is not None
    assert fail.startswith("C2 of record not established (COORD M-0721 (1))") and "MoeRouterSqrtSoftplus_v1" in fail


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="claim 2 BROKEN: the TP verdict passes a rank whose router, routed activation and attention have no evaluator while "
                          "the witness rule exempts every collective they reach; fix: the dense rule (_complete_replay_population_gap) in "
                          "replay_gates, form (B) being the oracle's compared tensors, or evaluators for those families")
def test_the_tp_verdict_fails_no_evaluator_families_the_oracle_exempts():
    ok, why = _gates(ORACLE)
    assert ok is False, why
    assert "MoeRouterSqrtSoftplus_v1" in (why or "")
