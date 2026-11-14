"""G6 of the global match (`verity_vllm.check.match.chronology`) where the Program or the record says of one request's
chronology what the other does not, on the toy of `test_global_match.py` under both implementations.  Each case changes one field
of a request's declaration, or one event of the log as it is persisted (the fold stays that of the events as served), and holds
what G6 names: the programmed steps, the idle rule for an early admission, rows that are not contiguous, a logits row without its
sample or twice in a step, a sampled token that is not the next input row's or its link's, and a row after an EOS stop's last one
that is fed onward."""
from __future__ import annotations

import dataclasses

import pytest

from tests.check import test_global_match as tgm
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import global_match_fast as gmf

memory = tgm.memory                                      # the toy's fixture, under its own name

IMPLS = pytest.mark.parametrize("impl", [gm, gmf], ids=["record", "fast"])
R0_FOUR = {"id": "r0", "prompt": [5, 7, 2], "sampled": [9, 3, 1, 4]}        # r0 served one step longer: engine steps 0..3


@pytest.fixture
def toy(monkeypatch):
    """`tgm.assemble`, with `edit` applied to the batched record's events as its log is persisted, and with the manifest that of
    `declared` where the Program is to declare other requests than the record served."""
    def assemble(reqs, d, edit=None, declared=None, **stated):
        persist, workload = tgm.persist, tgm.workload

        def persisted(result, events, where, served):
            return persist(result, edit(list(events)) if edit is not None and where.name == "batched" else events, where, served)
        with monkeypatch.context() as m:
            m.setattr(tgm, "persist", persisted)
            m.setattr(tgm, "workload", lambda served, path: workload(served if declared is None else declared, path))
            return tgm.assemble(reqs, d, **stated)
    return assemble


def _at(kind: str, step: int, **changed):
    """An edit of the log: in its first event of `kind` at engine step `step`, each named field becomes what its function gives."""
    def edit(events):
        i = next(k for k, e in enumerate(events) if type(e).__name__ == kind and e.step == step)
        events[i] = dataclasses.replace(events[i], **{name: to(getattr(events[i], name)) for name, to in changed.items()})
        return events
    return edit


def _without_the_last(kind: str):
    """An edit of the log: its last event of `kind` is gone."""
    def edit(events):
        del events[max(k for k, e in enumerate(events) if type(e).__name__ == kind)]
        return events
    return edit


def _g6(impl, gp, oracle) -> tuple[dict, dict]:
    """The document and its G6, which alone fails when it names a problem: the fold is the served events' in every case."""
    res = impl.check(gp, oracle, record_alternate=False)
    g6 = tgm._check(res, "G6")
    assert res["failed_checks"] == (["G6"] if g6["problems"] else []), res["reasons"]
    return res, g6


@IMPLS
@pytest.mark.parametrize("programmed, problems", [
    ({"prefill": 0, "decode": [1, 2]}, []),
    ({"prefill": 1}, ["r0: programmed prefill step 1 != observed 0"]),
    ({"prefill": 0, "decode": [2]}, ["r0: programmed first decode step 2 != observed 1"]),
], ids=["as-observed", "prefill", "first-decode"])
def test_a_requests_programmed_steps_are_compared_with_the_engine_steps_its_rows_are_in(impl, memory, tmp_path, toy, programmed, problems):
    gp, oracle = toy([tgm.R0, tgm.R1], tmp_path)
    gp["requests"][0]["programmed_steps"] = programmed
    assert _g6(impl, gp, oracle)[1]["problems"] == problems


@IMPLS
def test_a_request_admitted_before_its_declared_step_after_every_earlier_one_finished_is_the_declared_idle_rule(impl, memory, tmp_path, toy):
    """The Program declares r1's arrival at engine step 5; the record served it at step 3, r0 having finished at step 2."""
    gp, oracle = toy([tgm.R0, tgm.R1 | {"arrive": 3}], tmp_path, declared=[tgm.R0, tgm.R1 | {"arrive": 5}])
    res, g6 = _g6(impl, gp, oracle)
    assert res["verdict"] == "PASS" and g6["arrivals_by_idle_rule"] == 1
    r1 = tgm._r(res, "r1")
    assert (r1["arrive_step_declared"], r1["first_step"], r1["finish_step"]) == (5, 3, 4)
    assert r1["arrival_note"] == "admitted at 3 < declared 5: every earlier request had finished (declared idle rule)"


@IMPLS
@pytest.mark.parametrize("edit, named, n_problems", [
    # r0's sample of step 0 is another token than the one its row of step 1 holds, its link carries and the driver served
    (_at("SampleEnd", 0, sampled_token_ids=lambda served: [99, *served[1:]]),
     ["r0: sampled tokens per engine step [99, 3, 1] != driver's generated_token_ids [9, 3, 1] (output association)",
      "r0: token sampled at step 0 (99) != input row token at step 1 (9)",
      "r0: StepLink value 9 at step 1 != sampled 99 at src_step 0"], 3),
    (_at("StepLink", 1, value=lambda _: 99), ["r0: StepLink value 99 at step 1 != sampled 9 at src_step 0"], 1),
    # r0's row of step 1 is at another position than the one after its sampled row
    (_at("StepStart", 1, rows=lambda rows: [("r0", 7), *rows[1:]]),
     ["r0: no input row at position 3 in engine step 1 for the token sampled at step 0"], 1),
    # step 1 states r0's logits row twice and r1's not at all
    (_at("StepStart", 1, logits_rows=lambda rows: [rows[0], rows[0]]),
     ["r0: 2 logits rows in engine step 1", "r0: 3 TokenSelect instances in the fold but 4 sampled logits rows in the record",
      "r0: fold T_observed 2 != output events - 1 (3)", "r1: 2 TokenSelect instances in the fold but 1 sampled logits rows in the record",
      "r1: fold T_observed 1 != output events - 1 (0)"], 12),
    # step 1 states r0's row as r1's: r0 has rows in steps 0 and 2 alone
    (_at("StepStart", 1, rows=lambda rows: [("r1", rows[0][1]), *rows[1:]]),
     ["r0: rows not contiguous over engine steps 0..2 (preempted / resumed: not in the Program)",
      "r0: 3 TokenSelect instances in the fold but 2 sampled logits rows in the record",
      "r0: no input row at position 3 in engine step 2 for the token sampled at step 0",
      "r0: fold T_observed 2 != output events - 1 (1)"], 9),
    # the log ends without the samples of step 2, where r0 runs alone
    (_without_the_last("SampleEnd"),
     ["r0: no SampleEnd token for engine step 2 logits row 0", "r0: 3 TokenSelect instances in the fold but 2 sampled logits rows in the record",
      "r0: fold T_observed 2 != output events - 1 (1)"], 7),
], ids=["sample", "link", "row-position", "logits-row-twice", "rows-not-contiguous", "no-sample"])
def test_a_log_whose_step_or_sample_is_not_what_the_fold_was_served_is_named_by_request_and_engine_step(
        impl, memory, tmp_path, toy, edit, named, n_problems):
    """`named` is what G6 says of the changed event itself, in its order among G6's problems; the others follow from it (the
    driver's lengths, the stop rule, the scheduled live requests) and are counted."""
    gp, oracle = toy([tgm.R0, tgm.R1], tmp_path, edit)
    problems = _g6(impl, gp, oracle)[1]["problems"]
    assert [p for p in problems if p in named] == named and len(problems) == n_problems, problems


@IMPLS
def test_a_row_after_an_eos_stop_whose_sample_a_later_row_of_the_request_is_fed_is_no_placeholder_row(impl, memory, tmp_path, toy):
    """Under a declared LAG of 1 the row after r1's EOS is computed and its sample is fed nowhere: the record passes and the row is
    accepted.  With a row of r1 in step 3 at the position after it, the same record's row is named and no row is accepted."""
    for d, edit in ((tmp_path / "served", None),
                    (tmp_path / "fed", _at("StepStart", 3, rows=lambda rows: [*rows, ("r1", 4)], input_token_ids=lambda toks: [*toks, 77],
                                           positions=lambda positions: [*positions, 4]))):
        gp, oracle = toy([R0_FOUR, tgm.R1_LAG], d, edit, lag=1)
        tgm._set_async_scheduling(oracle, True)
        tgm._driver_stop_at(oracle, "r1", n_answer=2, eos_step=1)
        res, g6 = _g6(impl, gp, oracle)
        r1 = tgm._r(res, "r1")
        if edit is None:
            assert res["verdict"] == "PASS" and (r1["eos_verified"], r1["eos_lag_rows_accepted"], r1["lag_rows"]) == (True, 1, 1)
        else:
            assert g6["problems"][0] == ("r1: the lag row at engine step 2 (after the EOS at 1) sampled 77 which is fed to the request's row at "
                                         "position 4 in engine step 3: not a placeholder row")
            assert r1["eos_verified"] is False and "eos_lag_rows_accepted" not in r1
