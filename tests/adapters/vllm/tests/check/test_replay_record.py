"""The fold of the sampled replay's picks and the grade it gives (`verity_vllm.check.replay.replay_record`), outcome by outcome: the texts are what
a stored record carries and the Commit verdict prints, so each is held to the letter.  Two read the record's population
block and the record a whole replay writes (`verity_vllm.check.replay.driver`): what its strata's keys range over, and each
stratum it left unevaluated."""
from __future__ import annotations

from collections import Counter

import numpy as np

from tests.check import test_sampled_replay as T
from verity_vllm.check.replay import driver as DR
from verity_vllm.check.replay import replay_record as RR
from verity_vllm.records import replay_codes as RC


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
    assert (by["by_request"], by["by_phase"], by["by_family_phase"], by["by_family_request_phase"]) == (
        {"r0": 2}, {"mid": 2}, {"Gemm_v1|mid": 2}, {"Gemm_v1|r0|mid": 2})
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
    assert fold.not_evaluated == [{"spec": "Gemm_v1{K=4}", "request_id": "r0", "step": 2, "op_path": "model.x", "row": 0, "why_class": "missing-input",
                                   "why": "committed bytes not retained for (r0, step 2, model.x/out) [0, 16)"}]
    assert fold.unevaluated_by_stratum["a"]["classes"] == ["missing-input"]
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
    long = "committed bytes not retained for " + "x" * 200                                # the fault's text is cut at 160
    got, _ = _graded([(_vu("a"), _equal(), 0.1), (_vu("a"), _unevaluated(long), 0.1)], {"a": {"population": 2}})
    assert got[2] == ("1 pick(s) not evaluated inside evaluated strata (missing-input x 1) -- run-time fault, not a gap on the record: "
                      f"Gemm_v1{{K=4}} model.x (req r0, step 2): {long[:160]}")


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


def test_the_population_record_codes_each_reason_as_the_replay_stamped_it_and_an_unstamped_one_as_other():
    qpop = {k: k for k in ("query_total", "sampled_strata", "uncovered_by_name", "not_executed_by_family", "not_executed_rule",
                           "uncovered_identities", "query_checks", "scope")} | {"not_executed_by_rule": {"n": 0}}
    stamped = RC.reason_code("mapping", "Gemm_v1")
    got = RR.population_record([{"family": "Gemm_v1"}], ["r0"], Counter({"a reason": 2, "another": 1}), Counter({"X_v1": 3}),
                               {"a reason": stamped}, 1, qpop, [{"request_id": "r0", "stratum": "s"}])
    assert got["not_evaluable"] == {"a reason": 2, "another": 1}
    assert got["not_evaluable_codes"] == {"a reason": stamped, "another": RC.reason_code("other")}
    assert got["not_executed_by_rule"] == {"n": 0, "vus": 1, "vus_by_request": {"r0": 1},
                                           "strata_excluded": {"s": {"population": 1, "label": "not evaluable: unexecuted"}}}
    assert (got["vus"], got["by_family"], got["not_evaluable_by_family"], got["strata_n"], got["controller"]) == (
        1, {"Gemm_v1": 1}, {"X_v1": 3}, 1, None)
    assert sorted(got) == sorted(["vus", "source", "requests", "by_family", "not_evaluable_by_family", "not_evaluable", "not_evaluable_codes",
                                  "strata_n", "authority", "controller", *qpop])


def test_a_family_the_replay_does_not_evaluate_is_footnoted_by_the_reason_of_its_name_before_the_bracket():
    left = Counter({"AttentionHead_v1 (no registered evaluator)": 2, "X_v1 (no registered evaluator)": 1, "X_v1 (interior producer)": 4})
    assert RR.unreplayable(left) == [
        {"family": "AttentionHead_v1 (no registered evaluator)", "n": 2, "why": RC.NOT_YET["AttentionHead_v1"]},
        {"family": "X_v1 (interior producer)", "n": 4, "why": "interior producer: composed into its consumer's VU"},
        {"family": "X_v1 (no registered evaluator)", "n": 1, "why": "no registered evaluator"}]


def test_a_whole_replay_s_record_says_what_its_strata_range_over_and_names_each_it_left_unevaluated():
    """The sampled replay's CPU-served world (`tests.check.test_sampled_replay`), two requests of three steps, with one committed
    input of the first decode left out of the store: the two rows of each request that read it are not evaluated."""
    inst, woff = T._program()
    world = T.Honest(inst, woff, T._bf16(np.random.default_rng(7), max(hi for _lo, hi in woff.values()), 0.5), {"r0": [1, 2], "r1": [5, 3]})
    bmap, committed = world.bmap_and_reader(mutate=lambda store, entries: store.pop((1, "model.layers.0.mlp.act_fn/out")))
    rep = DR.sampled_replay(T._programs(world), bmap, committed, T._weights(world), per_stratum=8, seed=1, workers=0)
    keys, ranged = [k.split("|") for k in rep["strata"]], rep["strata_keys"]
    assert ranged["mechanisms"] == sorted({k[0] for k in keys}) and len(ranged["mechanisms"]) == 8
    assert ranged["requests"] == ["r0", "r1"] and ranged["phases"] == ["first-decode", "last", "prefill"]
    assert ranged["steps"] == {"early": [0, 1], "late": [2]} and ranged["modules"] == len({k[4] for k in keys}) == 8
    assert sorted(ranged) == ["key", "lifetime", "mechanisms", "modules", "phase", "phase_anchor", "phases", "ranks", "requests", "steps"]
    assert ranged["ranks"] == ["rank 0 (tp1, or the rank-0 shard of a per-rank map)"]
    assert sorted(ranged["lifetime"]) == ["ring_slots", "rule", "sampled", "values"]
    assert sorted(rep["timing"]) == ["cpu_s_sum", "per_family_n", "per_family_s", "wall_s", "workers"] and rep["timing"]["workers"] == 1
    assert sorted(rep["timing"]["per_family_s"]) == sorted(rep["timing"]["per_family_n"]) and sum(rep["timing"]["per_family_n"].values()) > 0
    # what the record says of its own keys and of its population: fixed texts, each held where it ends
    population = rep["population"]
    assert ranged["key"].endswith("attention rows: spec with T -> NB = ceil(T/BN) (the block-count mechanism; the row's own spec keeps T)")
    assert ranged["lifetime"]["rule"].endswith("else fresh-window; unwindowed collectors: one value, stated")
    assert ranged["phase"].endswith("steps past the prefix are 'not-executed', split out of the pool by name)")
    assert population["source"] == "Program instances.json.gz rows (every row; evaluable = families with a registered evaluator)"
    assert population["authority"] == "Q(P) -- the required manifest; `vus` above is the replay's enumeration of the Program rows, reconciled below"
    assert population["query_total"]["source"].endswith("(the v2 query's population; NOT a Q(P) statement -- weights / ids / hidden streams absent)")
    left = [f"{spec}|{rid}|rank0|first-decode|model.layers.0.mlp.{module}|n/a (unwindowed storage)"
            for spec, module in (("Gemm_v1{K=16,N=16}", "down_proj"), ("SiluMul_v1{I=16}", "act_fn")) for rid in ("r0", "r1")]
    assert (rep["result"], rep["grade"], rep["why"]) == (None, "partial", f"4 strata not evaluated: {left[:3]}")
    assert rep["strata_unevaluated"] == left
    assert [u["stratum"] for u in rep["unevaluated_strata"]] == left
    assert rep["unevaluated_strata"][0] == {
        "stratum": left[0], "spec": "Gemm_v1{K=16,N=16}", "request_id": "r0", "phase": "first-decode", "op_path": "model.layers.0.mlp.down_proj",
        "n": 1, "why": "committed bytes not retained for (r0, step 1, model.layers.0.mlp.act_fn/out) [0, 16)", "why_class": "missing-input",
        "classes": ["missing-input"]}
