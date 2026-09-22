"""F-r17-17 (rev M-0765): the compiled workload_actual record names the arrivals rule per request and folds it into `consistent`.

The declaration's arrive_step is a loop ITERATION of verity_capture.arrivals.run_requests_arrivals; the binding map observes the FORWARD
index of admission.  `replay_arrivals` replays that loop (declared idle rule, output lag) and `arrivals_rule` names per request which rule
admitted it and whether the observed forward index is the predicted one.  Fixture = the llama32-1b H100 B8 int-3 row of record
(r17_2226_b8_int3_502f1fac4, rev's reconstruction in M-0765): r1/r2 by the idle rule, r3..r7 declared minus the two no-forward
iterations, 152 iterations = 149 forwards + 3 drains.  Torch-free.
"""
from __future__ import annotations

from verity_capture.sweep.compiled_workload_actual import arrivals_rule, replay_arrivals

RIDS = [f"r{i}" for i in range(8)]
GEN = [2, 19, 128, 119, 13, 89, 57, 83]          # observed forwards per request (== caps on this row)
DECL = [0, 7, 24, 26, 28, 33, 38, 56]            # declared arrive_step (loop iterations)
OBS = [0, 2, 21, 24, 26, 31, 36, 54]             # observed forward index of the first bound identity


def test_replay_reproduces_the_llama_h100_b8_row():
    R = replay_arrivals(DECL, GEN, lag=1)
    assert (R["iterations"], R["forwards"]) == (152, 149)
    per = R["per_request"]
    assert [per[i]["forward_at_admission"] for i in range(8)] == OBS
    assert [per[i]["rule"] for i in range(8)] == ["declared", "idle", "idle", "declared", "declared", "declared", "declared", "declared"]
    assert per[1]["admit_iter"] == 3 and per[2]["admit_iter"] == 23          # admitted the iteration after the previous finish drained


def test_arrivals_rule_names_the_rule_per_request_and_ties_the_totals():
    A = arrivals_rule(RIDS, OBS, GEN, DECL, engine_steps=149, iterations_runs_jsonl=152)
    assert A["applicable"] and A["holds"] and A["arrive_equal_declared_under_rule"]
    assert A["by_rule"] == {"declared": 6, "idle": 2, "DEVIATION": 0}
    assert A["lag"] == 1 and A["no_forward_iterations"] == 3 and A["forwards_tie_engine_steps"] is True
    idle = [r for r in A["requests"] if r["rule"] == "idle"]
    assert [r["request_id"] for r in idle] == ["r1", "r2"] and all("every earlier request had finished" in r["why"] for r in idle)


def test_a_real_deviation_is_named_and_breaks_consistency():
    obs = list(OBS)
    obs[3] = 25                                   # r3 one forward late
    A = arrivals_rule(RIDS, obs, GEN, DECL, engine_steps=149, iterations_runs_jsonl=152)
    assert not A["holds"] and A["by_rule"]["DEVIATION"] == 1
    dev = next(r for r in A["requests"] if r["rule"] == "DEVIATION")
    assert dev["request_id"] == "r3" and "observed forward 25 != predicted 24" in dev["why"]


def test_an_untied_iteration_count_does_not_hold():
    A = arrivals_rule(RIDS, OBS, GEN, DECL, engine_steps=149, iterations_runs_jsonl=151)
    assert not A["holds"] and A["lag_status"].startswith("UNTIED")


def test_lag_zero_engine_has_no_drain_iterations():
    R = replay_arrivals(DECL, GEN, lag=0)
    assert R["iterations"] == R["forwards"] == 149
