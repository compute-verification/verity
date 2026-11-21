"""The top-level protocol `run`'s one monitoring period (`verity.protocols.run.period`)."""

from __future__ import annotations

from verity.protocols.run import period as P

PUBLIC = P.PublicParameters(hardware=None, limits=None, window="4,27,2", sampling_design="rate:1/4")


def test_the_wipe_then_one_registration_then_each_window_accounting_with_compilation_first_then_policy_then_the_result():
    calls = []
    stages = P.Stages(
        registration=lambda public, w, state: calls.append(f"registration after {state.wipe['note']}") or "registered",
        windows=lambda public, w, state: calls.append("windows") or ("w0", "w1"),
        accounting=lambda public, w, state, window: calls.append(f"accounting {window}") or f"outcomes of {window}",
        policy=lambda public, w, state, window, got: calls.append(f"policy {window} on {got}") or window,
        result=lambda public, w, state: calls.append(f"result {len(state.windows)}") or "result")
    state = P.run_period(PUBLIC, "workload", stages)
    assert calls == [f"registration after {P.WIPE_BY_LAUNCHER}", "windows", "accounting w0", "policy w0 on outcomes of w0", "accounting w1",
                     "policy w1 on outcomes of w1", "result 2"]
    assert (state.registration, state.result) == ("registered", "result")
    assert [x["accounting"]["compilation"] for x in state.windows] == [
        {"accepted": True, "note": P.COMPILATION_NOT_VERIFIED}] * 2
