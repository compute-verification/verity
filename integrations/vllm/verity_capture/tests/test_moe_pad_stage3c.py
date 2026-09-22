"""Stage 3c (R17-42 x R17-1) on the fixture engine: two requests with different routes and stops -- one early EOS with the declared
post-EOS row, one cap exhaustion -- as assignments to ONE padded Program; the three R17-1 negatives fail by name; variable lag surfaces.
Torch-free (`verity_capture.moe_pad_fixture`, reference evaluator)."""

from __future__ import annotations

import json

import pytest

from verity_capture import moe_pad_fixture as F
from verity.ir import Program
from verity.ir.codec import encode_program, program_digest

E, TOPK, VPT, H, I, S = 4, 2, 2, 16, 16, 5


@pytest.fixture(scope="module")
def w():
    return F.one_hot_weights(E, H, I)


def _check(trace, decl, LAG, w):
    return F.check(trace, decl, S, LAG, w, E, TOPK, VPT, H, I)


def test_two_requests_early_eos_plus_declared_lag_pass_with_the_two_length_fields_distinct(w):
    decl = [F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    r = _check(F.run_engine(decl, S, E, TOPK, H, engine_lag=1), decl, 1, w)
    assert r.verdict == "PASS" and r.failures == [] and r.alerts == []
    assert r.served_output_len == [2, 5] and r.executed_prefix_len == [3, 5]          # distinct fields; EOS request: served + LAG; cap: served
    assert r.rows["0/2"] == "lag-row" and r.rows["0/3"] == "continuation-disabled-row" and r.rows["0/4"] == "continuation-disabled-row"
    assert all(r.rows[f"1/{s}"] == "served-row" for s in range(S))                     # the cap request has no lag row
    p = r.population
    assert p["expanded_slot_bodies"] == 2 * S * E == 40 and p["compact_pairs"] == (3 + 5) * TOPK == 16
    assert p["continuation_disabled_rows"] == 2 == p["disabled_rows"] and p["unselected_disabled_slots"] == 8 * (E - TOPK) == 16   # the two inactivity kinds, separately
    assert p["enabled_rows"] == 8 and p["lag_rows"] == 1 and p["lag_row_enabled_slots"] == TOPK and p["cap_exhausted_requests"] == 1


def test_the_lag_row_is_active_internal_computation_not_a_served_output(w):
    decl = [F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    r = _check(F.run_engine(decl, S, E, TOPK, H, engine_lag=1), decl, 1, w)
    assert r.verdict == "PASS"
    assert r.rows["0/2"] == "lag-row"                        # row 2 = the post-EOS row: an ENABLED block (TOPK enabled slots, E-TOPK disabled) ...
    assert r.served_output_len[0] == 2                       # ... and served_output_len stays 2: its sampled token is not served


def test_negative_lag0_header_on_a_lag1_trace_fails_by_name(w):
    decl = [F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    r = _check(F.run_engine(decl, S, E, TOPK, H, engine_lag=1), decl, 0, w)
    assert r.verdict == "FAIL" and len(r.failures) == 1
    assert r.failures[0].startswith("R17-1/LAG-ROW-BEYOND-DECLARED: request 0 row 2 executed; declared executed_prefix_len 2")
    assert r.executed_prefix_len == [2, 5]                   # the declaration is NOT amended from the trace


def test_negative_lag1_header_on_a_lag0_trace_fails_by_name_no_inference_either_way(w):
    decl = [F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    r = _check(F.run_engine(decl, S, E, TOPK, H, engine_lag=0), decl, 1, w)
    assert r.verdict == "FAIL" and len(r.failures) == 1
    assert r.failures[0].startswith("R17-1/MISSING-EXECUTED-ROW: request 0 row 2 declared executed (executed_prefix_len 3 = served 2 + LAG 1)")


def test_negative_variable_lag_surfaces_as_an_alert_and_the_row_fails_by_name_never_absorbed(w):
    decl = [F.Declared(1, F.EOS), F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    r = _check(F.run_engine(decl, S, E, TOPK, H, engine_lag=[1, 2, 0]), decl, 1, w)
    assert r.verdict == "FAIL"
    assert any(a.startswith("VARIABLE-LAG: observed post-EOS rows differ across EOS-stopped requests {0: 1, 1: 2}") for a in r.alerts)
    assert r.failures == [f for f in r.failures if f.startswith("R17-1/LAG-ROW-BEYOND-DECLARED: request 1 row 3")]  # only request 1's extra row
    assert r.executed_prefix_len == [2, 3, 5]                # LAG stays the declared 1 for every request


def test_negative_widening_eos_to_absorb_the_lag_row_is_caught(w):
    decl_true = [F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    trace = F.run_engine(decl_true, S, E, TOPK, H, engine_lag=1)
    widened = [F.Declared(3, F.EOS), F.Declared(S, F.CAP)]  # "the EOS row's successor was served": served 3 -> executed must be 4
    r = _check(trace, widened, 1, w)
    assert r.verdict == "FAIL" and r.failures[0].startswith("R17-1/MISSING-EXECUTED-ROW: request 0 row 3")


def test_limit_the_pair_mutation_served_widened_and_lag_redeclared_0_passes_at_fixture_level_by_construction(w):
    """revb S-r17b-45: NOT a fixture-caught negative.  served 2 -> 3 AND LAG 1 -> 0 give the same executed_prefix_len [3, 5]; the checker
    has only the two declared lengths, so this PASSES here (different digest).  At record level the two fields have independent sources
    (tokens.json tied to the fold; the attested async_scheduling header) and the workload digest moves with LAG -- that is the defence."""
    decl_true = [F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    trace = F.run_engine(decl_true, S, E, TOPK, H, engine_lag=1)
    truth = _check(trace, decl_true, 1, w)
    pair = _check(trace, [F.Declared(3, F.EOS), F.Declared(S, F.CAP)], 0, w)
    assert truth.verdict == pair.verdict == "PASS" and truth.executed_prefix_len == pair.executed_prefix_len == [3, 5]
    assert pair.served_output_len == [3, 5] != truth.served_output_len and pair.digest != truth.digest
    assert pair.rows["0/2"] == "served-row" and truth.rows["0/2"] == "lag-row"          # the lag row is mislabelled served: undetectable here


def test_declared_lag_beyond_the_allocated_rows_is_refused_internal_capacity_allocates_for_the_lag():
    with pytest.raises(ValueError, match="internal capacity must allocate for the declared lag"):
        F.executed_prefix_len(F.Declared(S, F.EOS), S, 1)
    assert F.executed_prefix_len(F.Declared(S, F.CAP), S, 1) == S                     # cap exhaustion: no row follows the cap
    with pytest.raises(ValueError, match="cap exhaustion means served_output_len == cap rows"):
        F.executed_prefix_len(F.Declared(S - 1, F.CAP), S, 1)


def test_the_digest_carries_the_declared_lag_and_not_the_routes_or_stops(w):
    d0 = program_digest(encode_program(Program(F.fixture(2, S, 0, E, TOPK, VPT, H, I))))
    d1 = program_digest(encode_program(Program(F.fixture(2, S, 1, E, TOPK, VPT, H, I))))
    assert d0 != d1                                                                   # R17-1: LAG is a declared specialization parameter
    decl_a = [F.Declared(2, F.EOS), F.Declared(S, F.CAP)]
    decl_b = [F.Declared(1, F.EOS), F.Declared(S, F.CAP)]
    ra = _check(F.run_engine(decl_a, S, E, TOPK, H, engine_lag=1, seed=1), decl_a, 1, w)
    rb = _check(F.run_engine(decl_b, S, E, TOPK, H, engine_lag=1, seed=2), decl_b, 1, w)
    assert ra.verdict == rb.verdict == "PASS" and ra.digest == rb.digest == str(d1)   # different routes and stops: the same Program


def test_cli_writes_the_five_cases(tmp_path):
    out = tmp_path / "s3c.json"
    assert F.main(["--out", str(out)]) == 0
    d = json.loads(out.read_text())
    v = {k: c["verdict"] for k, c in d["cases"].items()}
    assert v == {"two_requests_early_eos_plus_declared_lag1": "PASS", "neg_lag0_header_on_lag1_trace": "FAIL",
                 "neg_lag1_header_on_lag0_trace": "FAIL", "neg_variable_lag_surfaces": "FAIL", "neg_eos_widened_to_absorb_the_lag_row": "FAIL"}
