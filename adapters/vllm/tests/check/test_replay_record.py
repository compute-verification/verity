"""The fold of the sampled replay's picks and the grade it gives (`verity_vllm.check.replay.replay_record`), outcome by outcome: the texts are what
a stored record carries and the Commit verdict prints, so each is held to the letter."""
from __future__ import annotations

from collections import Counter

from verity_vllm.check.replay import replay_record as RR


def _vu(stratum: str, family: str = "Gemm_v1") -> dict:
    return {"stratum": stratum, "family": family, "spec": f"{family}{{K=4}}", "request_id": "r0", "phase": "mid", "op_path": "model.x"}


def _equal() -> dict:
    return {"result": True}


def _mismatch(why: str) -> dict:
    return {"result": False, "why": why, "spec": "Gemm_v1{K=4}", "request_id": "r0", "phase": "mid", "op_path": "model.x"}


def _unevaluated(why: str) -> dict:
    return {"result": None, "why": why, "spec": "Gemm_v1{K=4}", "request_id": "r0", "step": 2, "op_path": "model.x", "row": 0}


def _graded(results: list, strata: dict, weights_check: dict | None = None) -> tuple:
    fold = RR.fold(results, strata)
    return RR.grade(fold, weights_check, sorted(k for k, v in strata.items() if v["population"] > 0 and v["evaluated"] == 0)), fold


def test_every_pick_evaluated_and_equal_is_complete_and_the_strata_carry_their_counts():
    strata = {"a": {"population": 3}, "b": {"population": 1}}
    got, fold = _graded([(_vu("a"), _equal(), 0.5), (_vu("a"), _equal(), 0.25), (_vu("b", "RoPE_v1"), _equal(), 1.0)], strata)
    assert got == (True, "complete", None)
    assert (fold.evaluated, fold.equal, fold.mismatch, fold.not_evaluated) == (3, 3, [], [])
    assert strata == {"a": {"population": 3, "evaluated": 2}, "b": {"population": 1, "evaluated": 1}}       # no `mismatch` key without one
    assert fold.family_seconds == {"Gemm_v1": 0.75, "RoPE_v1": 1.0} and dict(fold.family_n) == {"Gemm_v1": 2, "RoPE_v1": 1}


def test_a_mismatch_fails_with_the_first_ones_reason_and_the_count_of_the_rest():
    strata = {"a": {"population": 3}}
    got, fold = _graded([(_vu("a"), _mismatch("out differs at word 7"), 0.1), (_vu("a"), _equal(), 0.1), (_vu("a"), _mismatch("second"), 0.1)], strata)
    assert got == (False, "mismatch", "out differs at word 7 (+1 more)")
    assert strata["a"] == {"population": 3, "evaluated": 3, "mismatch": 2}
    by = RR.mismatch_by(fold)
    assert by["by_family"] == {"Gemm_v1": 2} and by["by_module"] == {"Gemm_v1|model.x": 2} and by["strata"] == {"a": 2}
    one, _ = _graded([(_vu("a"), _mismatch("out differs at word 7"), 0.1)], {"a": {"population": 1}})
    assert one == (False, "mismatch", "out differs at word 7")


def test_weights_that_are_not_the_committed_ones_fail_before_anything_else():
    results = [(_vu("a"), _equal(), 0.1)]
    got, _ = _graded(results, {"a": {"population": 1}}, {"equal": False, "why": "root differs"})
    assert got == (False, "mismatch", "weights at replay != committed weights root (root differs)")
    got, _ = _graded(results, {"a": {"population": 1}}, {"equal": False})
    assert got == (False, "mismatch", "weights at replay != committed weights root (the registered inputs the relations consume are not the committed ones)")
    assert _graded(results, {"a": {"population": 1}}, {"equal": True})[0] == (True, "complete", None)


def test_nothing_evaluated_is_not_evaluated_and_never_a_pass():
    got, fold = _graded([(_vu("a"), _unevaluated("committed bytes not retained for (r0, step 2, model.x/out) [0, 16)"), 0.1)], {"a": {"population": 1}})
    assert got == (None, "not-evaluated", "no VU could be evaluated: committed bytes not retained for (r0, step 2, model.x/out) [0, 16)")
    assert fold.not_evaluated[0]["why_class"] == "missing-input" and fold.unevaluated_by_stratum["a"]["classes"] == ["missing-input"]
    assert _graded([], {})[0] == (None, "not-evaluated", "no VU could be evaluated: empty population")


def test_a_stratum_with_no_evaluated_pick_is_partial_and_named():
    strata = {"a": {"population": 2}, "b": {"population": 2}, "empty": {"population": 0}}
    got, fold = _graded([(_vu("a"), _equal(), 0.1), (_vu("b"), _unevaluated("no registered evaluator for Gemm_v9"), 0.1),
                         (_vu("b"), _unevaluated("evaluator error: KeyError('x')"), 0.1)], strata)
    assert got == (None, "partial", "1 strata not evaluated: ['b']")                                     # a stratum nothing populates is not one
    assert fold.unevaluated_by_stratum["b"] == {
        "stratum": "b", "spec": "Gemm_v1{K=4}", "request_id": "r0", "phase": "mid", "op_path": "model.x", "n": 2, "why": "no registered evaluator for Gemm_v9",
        "why_class": "no-evaluator", "classes": ["no-evaluator", "evaluator-error"]}


def test_a_pick_not_evaluated_inside_an_evaluated_stratum_is_partial_with_the_first_fault_named():
    strata = {"a": {"population": 3}}
    got, _ = _graded([(_vu("a"), _equal(), 0.1), (_vu("a"), _unevaluated("no registered evaluator for Gemm_v9"), 0.1),
                      (_vu("a"), _unevaluated("committed bytes not retained"), 0.1)], strata)
    fault = ("2 pick(s) not evaluated inside evaluated strata (missing-input x 1, no-evaluator x 1) -- run-time fault, not a gap on "
             "the record: Gemm_v1{K=4} model.x (req r0, step 2): committed bytes not retained")
    assert got == (None, "partial", fault)
    got, _ = _graded([(_vu("a"), _equal(), 0.1), (_vu("a"), _unevaluated("no registered evaluator for Gemm_v9"), 0.1)], {"a": {"population": 2}})
    by_name = ("1 pick(s) not evaluated inside evaluated strata (no-evaluator x 1): Gemm_v1{K=4} model.x (req r0, step 2): "
               "no registered evaluator for Gemm_v9")
    assert got == (None, "partial", by_name)


def test_unreplayable_names_each_family_with_its_count_and_its_reason():
    got = RR.unreplayable(Counter({"RoPE_v9": 4, "ConstPad_v1 (structure)": 2, "Gemm_v1 (interior)": 3, "MoeSum_v1 (plane mapping failed)": 1,
                                   "Attention_v3 (recomputed prefix)": 5, "AttentionHead_v1": 7}))
    assert got == [
        {"family": "AttentionHead_v1", "n": 7, "why": "consumes the request's K/V history across steps (state-linked replay, v2)"},
        {"family": "Attention_v3 (recomputed prefix)", "n": 5, "why": "recomputed prefix: computed by the warm request's replay, never served by it"},
        {"family": "ConstPad_v1 (structure)", "n": 2, "why": "structure / interior: no identity"},
        {"family": "Gemm_v1 (interior)", "n": 3, "why": "interior producer: composed into its consumer's VU"},
        {"family": "MoeSum_v1 (plane mapping failed)", "n": 1, "why": "MoE plane row -> plane slice mapping failed: see population.not_evaluable"},
        {"family": "RoPE_v9", "n": 4, "why": "no registered evaluator"}]
