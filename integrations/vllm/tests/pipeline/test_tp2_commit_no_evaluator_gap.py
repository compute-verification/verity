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
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.pipeline.tp.c2 import TP_FORM_B_SCOPE, c2_and_pin_gates
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


def _gates(sr: dict, xrank=XF8, **cblock) -> tuple[bool | None, str | None, dict]:
    cb = {"match_oracle": dict(ORACLE)} | cblock
    f = {"sampled_replay": {"sampled_replay": sr, "boundary_linkage": {"result": True, "linked": 32, "pairs": 32}},
         "weights_pin": {"result": True, "compared": 2139}}
    ok, why = c2_and_pin_gates(cb, f, True, None, rank=0, replay_on=True, required=True, bundle="bundle_pair0_rank0", xrank_families=xrank)
    return ok, why, cb


def test_the_dense_verdict_fails_a_no_evaluator_family_without_a_complete_form_b():
    """The rule the TP verdict lacked: the router with no evaluator and an incomplete oracle compare is a named FAIL on the dense path."""
    sr = {"result": True, "population": {"not_evaluable": {f"no registered evaluator for {f}": n for f, n in NO_EVALUATOR.items()}}}
    oc = {"result": True, "complete": False, "mismatch_n": 0, "compared": 2}
    note, fail = _complete_replay_population_gap(sr, oc)
    assert note is None and fail is not None
    assert fail.startswith("C2 of record not established: ") and "MoeRouterSqrtSoftplus_v1" in fail


def test_the_tp_verdict_fails_no_evaluator_families_the_oracle_exempts():
    ok, why, cb = _gates(_sr(NO_EVALUATOR))
    assert ok is False, why
    assert "MoeRouterSqrtSoftplus_v1" in (why or "")
    assert why.startswith("no-evaluator families (rank 0): C2 of record not established: ") and TP_FORM_B_SCOPE in why
    assert cb["no_evaluator_gap"]["fail"] and cb["no_evaluator_gap"]["waived_cross_rank"] == []


def test_the_cross_rank_collectives_are_waived_while_that_pass_runs_and_only_then():
    ok, why, cb = _gates(_sr(COLLECTIVES))
    assert (ok, why) == (True, None) and cb["no_evaluator_gap"]["waived_cross_rank"] == sorted(COLLECTIVES)
    ok, why, cb = _gates(_sr(COLLECTIVES), xrank=None)
    assert ok is False and all(f in why for f in COLLECTIVES) and cb["no_evaluator_gap"]["cross_rank_pass"] is False
    ok, why, _ = _gates(_sr(COLLECTIVES | {"MoeRouterHash_v1": 6888}))
    assert ok is False and "MoeRouterHash_v1" in why and "AllReduce_v2" not in why


def test_the_rule_reads_the_population_codes_the_replay_stamps():
    """A rank's own record carries `population.not_evaluable` and its codes: the rule decides from those, as the dense verdict does."""
    sr = _sr({})
    reasons = {f"no registered evaluator for {f}": n for f, n in (COLLECTIVES | {"SiluMulClamp_v1": 980392}).items()}
    sr["population"] |= {"not_evaluable": reasons, "not_evaluable_codes": {k: RC.reason_code("no-evaluator", k.rsplit(" ", 1)[1]) for k in reasons}}
    ok, why, cb = _gates(sr)
    assert ok is False and "980392 x no registered evaluator for SiluMulClamp_v1" in why and "AllGather_v1" not in why
    assert cb["no_evaluator_gap"]["waived_cross_rank"] == sorted(COLLECTIVES)


def test_a_plane_fed_producer_fails_by_name_and_a_family_without_identity_is_no_gap():
    plane = {"MoeRouterTopK_v1 (moe_block_stream plane producer: no evaluator)": 512}
    ok, why, _ = _gates(_sr(plane))
    assert ok is False and "plane-fed VUs" in why and "MoeRouterTopK_v1" in why
    inert = {"Const32_v1 (structure / interior: no identity)": 9, "Gemm_v1 (interior: composed into its epilogue's VU / a hinted row of the module is its returned tensor)": 4}
    assert _gates(_sr(inert))[:2] == (True, None)


def test_the_replay_versions_stated_domain_is_not_gated():
    ok, why, _ = _gates(_sr({"AttentionHead_v1": 64}, why=RC.NOT_YET["AttentionHead_v1"]))
    assert (ok, why) == (True, None)


def test_a_selector_passes_only_when_the_match_record_confirms_every_sampled_id():
    sel = {"GumbelTopPTokenSelect_v1": 32}
    ok, why, _ = _gates(_sr(sel))
    assert ok is False and "selector output VUs" in why and "request attribution not complete" in why
    att = {"ok": True, "steps_checked": 32, "rows_checked": 32, "mismatches_n": 0}
    ok, why, cb = _gates(_sr(sel), attribution=att)
    assert (ok, why) == (True, None) and "value-checked by form (B)" in cb["no_evaluator_gap"]["note"]
    ok, why, _ = _gates(_sr(sel), attribution=att | {"ok": False, "mismatches_n": 1})
    assert ok is False and "GumbelTopPTokenSelect_v1" in why
